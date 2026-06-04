# OSNet 파인튜닝 RUNBOOK — Ubuntu / Linux (CUDA) 판

> Windows(PowerShell)판 `RUNBOOK.md` 의 우분투 대응본입니다.
> **코드는 동일**합니다(경로를 pathlib/argparse로 처리해 OS 독립적). 바뀌는 건 셸 명령뿐입니다.
> 데이터(`data/`, `log/`, `exports/`)는 git에 포함되지 않으므로 **별도 전달** 후 같은 폴더 구조로 둡니다.

## 0. 전제

```bash
# 작업 폴더(예시). 팀원 환경에 맞게 한 번만 잡습니다.
export FT=~/osnet_finetune
cd "$FT"
```

이후 모든 단계는 `cd "$FT"` 상태 기준입니다.

폴더 구조(코드는 git에서 받고, data/log/exports는 직접 만들거나 전달받음):

```
osnet_finetune/
├── 01_extract_tracks.py ... 06_export_onnx.py
├── _recluster_*.py, _scan_thresholds.py
├── configs/train_config.yaml
├── requirements.txt
├── data/raw_videos/        ⬅ 입력 AVI 3개 (별도 전달)
├── data/market1501_root/   ⬅ 라벨링된 데이터셋 (별도 전달 or 1~4단계로 생성)
├── log/                    ⬅ 학습 체크포인트·TensorBoard 로그(자동 생성)
└── exports/                ⬅ 최종 ONNX(자동 생성)
```

---

## Phase A — 1회성 환경 셋업

### A.1 시스템 패키지

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg git build-essential
nvidia-smi          # GPU/드라이버 확인 (CUDA 버전 메모)
```

### A.2 가상환경

```bash
cd "$FT"
python3 -m venv .venv-finetune
source .venv-finetune/bin/activate
python -m pip install --upgrade pip
# 프롬프트 앞에 (.venv-finetune) 가 붙으면 정상
```

### A.3 PyTorch (드라이버 CUDA에 맞춰)

```bash
# 예: CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
# 확인
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
# False 면 드라이버/CUDA 호환 문제 → 여기서 해결 후 진행
```

### A.4 나머지 의존성

```bash
pip install -r requirements.txt
# GPU가 없는 환경이면 requirements.txt 의 onnxruntime-gpu 를 onnxruntime 으로 바꿔 설치
```

### A.5 torchreid 소스 설치

```bash
cd "$FT/.."
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
python setup.py develop
cd "$FT"
python -c "import torchreid; print(torchreid.__version__)"
```

### A.6 YOLO weights 확인

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"   # 현재 폴더에 yolov8s.pt 다운로드되면 성공
```

---

## Phase B — 입력 영상 배치

```bash
mkdir -p "$FT/data/raw_videos"
# cam01.avi, cam02.avi, cam03.avi 를 여기에 둠 (파일명 정확히 cam<2자리>.avi)
# 카메라별 시작 시각 보정이 필요하면:
echo '{"1": 0.0, "2": 1.2, "3": -0.5}' > "$FT/data/offsets.json"
```

---

## Phase C — Dry-run (권장)

```bash
mkdir -p "$FT/data/raw_videos_60s"
for c in 01 02 03; do
  ffmpeg -ss 0 -t 60 -i "$FT/data/raw_videos/cam$c.avi" -c copy "$FT/data/raw_videos_60s/cam$c.avi"
done
# 아래 본 실행 명령에서 경로만 _60s 로 바꿔 1~4단계를 한 번 통과시켜 본 뒤 본 실행
```

---

## Phase D — 본 실행 6단계

> Windows판과 인자는 동일합니다. 줄바꿈만 백슬래시(`\`)로 표기.

### D.1 단계 1 — YOLO + ByteTrack 트랙 크롭

```bash
python 01_extract_tracks.py \
  --videos data/raw_videos/cam01.avi data/raw_videos/cam02.avi data/raw_videos/cam03.avi \
  --out data/crops \
  --meta data/tracks_meta.csv \
  --yolo-weights yolov8s.pt \
  --conf 0.4 --iou 0.5 \
  --sample-fps 2 \
  --min-size 64 128 \
  --device 0
# 확인
ls data/crops/cam01 | head
wc -l data/tracks_meta.csv
find data/crops -name '*.jpg' | wc -l
```

### D.2 단계 2 — Cross-camera 매칭 제안

```bash
python 02_propose_matches.py \
  --crops data/crops \
  --meta data/tracks_meta.csv \
  --out data/proposals.json \
  --sim-threshold 0.55 \
  --lambda-t 0.20 --tau 2.0 \
  --device cuda
# offsets 있으면 --offsets data/offsets.json 추가
# 콘솔 끝 'multi-track=NNN, singletons=MMM' 확인. 다중-track 30 미만이면 --sim-threshold 0.45 재실행
```

### D.3 단계 3 — Streamlit 동일인 검증

```bash
streamlit run 03_verify_ui.py -- \
  --crops data/crops \
  --proposals data/proposals.json \
  --out data/persons.json \
  --max-thumbs 12
# 브라우저 http://localhost:8501
# 원격 서버면: streamlit run ... --server.address 0.0.0.0 --server.port 8501  후 SSH 포트포워딩
#   (로컬 PC에서)  ssh -L 8501:localhost:8501 user@서버
# 작업 중 수시로 좌상단 💾 Save persons.json 클릭
```

### D.4 단계 4 — Market-1501 디렉터리 생성

```bash
python 04_build_market1501.py \
  --crops data/crops \
  --persons data/persons.json \
  --out data/market1501_root/market1501 \
  --test-ratio 0.2 \
  --min-images-per-id 8 \
  --max-images-per-id 80
# 확인
cat data/market1501_root/market1501/_summary.json
ls data/market1501_root/market1501/bounding_box_train | wc -l
```

### D.5 단계 5 — Torchreid 파인튜닝

```bash
python 05_train_osnet.py --config configs/train_config.yaml
# best 체크포인트
ls -t log/osnet_x0_25_eyed/model | head -3   # model.pth.tar-60 등
# rank-1 가이드: >75% 양호 / 60~75% 라벨 점검 / <60% 데이터 보강
```

### D.6 단계 6 — ONNX 추출

```bash
mkdir -p exports
python 06_export_onnx.py \
  --weights log/osnet_x0_25_eyed/model/model.pth.tar-60 \
  --out exports/osnet_x0_25.onnx \
  --image-size 256 128 \
  --l2-norm
# 콘솔 끝 'PyTorch vs ONNX max abs diff = X.XXe-07' 보이면 성공 (< 1e-3)
```

---

## Phase F — 측정값 그래프 (TensorBoard)

학습 중 `log/osnet_x0_25_eyed/` 에 TensorBoard 로그가 자동 기록됩니다(loss/acc/lr/rank1/mAP).

```bash
pip install tensorboard   # torch 설치 시 보통 함께 깔림
tensorboard --logdir log/osnet_x0_25_eyed --host 0.0.0.0 --port 6006
# 브라우저 http://localhost:6006  (원격 서버면 ssh -L 6006:localhost:6006 user@서버)
```

---

## Windows ↔ Ubuntu 차이 요약

| 항목 | Windows(PowerShell) | Ubuntu(bash) |
| --- | --- | --- |
| 환경변수 | `$FT = "..."` | `export FT=~/...` |
| venv 활성화 | `.\.venv-finetune\Scripts\activate` | `source .venv-finetune/bin/activate` |
| 폴더 생성 | `New-Item -ItemType Directory -Force` | `mkdir -p` |
| ffmpeg 설치 | `winget install Gyan.FFmpeg` | `sudo apt install ffmpeg` |
| 경로 구분자 | `\` | `/` |
| 줄바꿈 | 백틱 `` ` `` | 백슬래시 `\` |
| 파일 확인 | `Get-Content`, `Measure-Object` | `cat`, `wc -l`, `jq` |
| 원격 UI | 로컬 실행 | SSH 포트포워딩(`ssh -L`) |

> 코드(.py)·config(.yaml)는 그대로 사용. 가상환경 `.venv` 와 다운로드 가중치(`*.pt`)는
> OS별로 다시 만들어야 하므로 **전달물에 포함하지 말 것**(git에서도 제외됨).
