# OSNet 파인튜닝 RUNBOOK — 적용 순서 (Windows / 로컬 CUDA 환경)

> 이 문서는 **이미 생성된 코드/파일을 처음부터 끝까지 어떻게 돌리는지**를
> 명령어 단위로 정리한 작업 매뉴얼입니다. README.md 와 함께 보세요.

## 0. 지금 어디에 무엇이 있는가

전부 다음 한 폴더 안에 있습니다 (EYE-D 레포 **바깥**):

```
C:\Users\murim\OneDrive\문서\Claude\Projects\엣지 게이트웨이 기반 지능형 침입 탐지 및 인물 재식별(Re-ID) 시스템\osnet_finetune\
├── README.md                       (개념·옵션 설명)
├── RUNBOOK.md                      (이 파일 — 적용 순서)
├── requirements.txt
├── .gitignore
├── configs\train_config.yaml       (학습 하이퍼파라미터)
├── 01_extract_tracks.py            ── 단계 1
├── 02_propose_matches.py           ── 단계 2
├── 03_verify_ui.py                 ── 단계 3 (Streamlit)
├── 04_build_market1501.py          ── 단계 4
├── 05_train_osnet.py               ── 단계 5
├── 06_export_onnx.py               ── 단계 6
├── data\raw_videos\                ⬅ 입력 AVI 3개 넣을 곳
├── exports\                        ⬅ 최종 ONNX 산출 위치
└── log\                            ⬅ 학습 체크포인트
```

EYE-D 레포는 단 한 줄도 손대지 않습니다. 산출물 ONNX 만 마지막 단계에서
교체합니다.

이하 명령어는 **PowerShell 기준**. 폴더 경로가 길어 한 번 환경변수로 잡습니다.

```powershell
$FT = "C:\Users\murim\OneDrive\문서\Claude\Projects\엣지 게이트웨이 기반 지능형 침입 탐지 및 인물 재식별(Re-ID) 시스템\osnet_finetune"
cd $FT
```

이후 모든 단계는 `cd $FT` 한 상태에서 실행한다고 가정합니다.

---

## Phase A — 1회성 환경 셋업 (15~30분)

### A.1 Python 가상환경 생성

```powershell
python -m venv .venv-finetune
.\.venv-finetune\Scripts\activate
python -m pip install --upgrade pip
```

프롬프트 앞에 `(.venv-finetune)` 가 붙어야 정상.

### A.2 PyTorch (CUDA 버전 맞춰서)

GPU 드라이버에 맞는 CUDA toolkit 버전 확인 후 한 줄만 실행.

```powershell
# 예: CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

확인:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
# 예상: 2.2.2 True NVIDIA GeForce RTX ....
```

`False` 가 나오면 CUDA toolkit / 드라이버 호환 문제이므로 여기서 멈추고 해결.

### A.3 나머지 의존성

```powershell
pip install -r requirements.txt
```

### A.4 torchreid 소스 설치 (공식 권장)

```powershell
cd ..
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
python setup.py develop
cd $FT
```

확인:

```powershell
python -c "import torchreid; print(torchreid.__version__)"
# 예상: 1.4.0 정도
```

만약 빌드 에러가 나면 Visual C++ Build Tools 가 필요합니다 (Microsoft C++ Build Tools 검색).

### A.5 YOLO weights 자동 다운로드 확인

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
# yolov8s.pt 가 현재 폴더에 받아져 있으면 성공
```

---

## Phase B — 입력 영상 배치 (1분)

3개 AVI 파일을 다음 위치에 두세요.

```
$FT\data\raw_videos\cam01.avi
$FT\data\raw_videos\cam02.avi
$FT\data\raw_videos\cam03.avi
```

**파일명 규칙은 정확해야 합니다.** `cam<숫자2자리>.avi` 형식.
다른 이름으로 받았다면 복사 후 이름만 바꿔 두세요.

영상 시작 시각이 카메라별로 다르다면, JSON 으로 offset(초)을 정의합니다.

```powershell
# (필요할 때만) 카메라별 시작 시각 보정
'{"1": 0.0, "2": 1.2, "3": -0.5}' | Out-File -Encoding utf8 data\offsets.json
```

---

## Phase C — Dry-run (강력 권장, 약 20~40분)

5시간 영상을 바로 돌리면 한 실수가 5시간을 날립니다.
**먼저 각 카메라에서 1분만 잘라 동일 파이프라인을 통과시켜 보세요.**

ffmpeg 가 없다면: `winget install Gyan.FFmpeg` 또는 https://www.gyan.dev/ffmpeg/builds/

```powershell
# 1분짜리 짧은 영상 3개 생성
mkdir data\raw_videos_60s
ffmpeg -ss 0 -t 60 -i data\raw_videos\cam01.avi -c copy data\raw_videos_60s\cam01.avi
ffmpeg -ss 0 -t 60 -i data\raw_videos\cam02.avi -c copy data\raw_videos_60s\cam02.avi
ffmpeg -ss 0 -t 60 -i data\raw_videos\cam03.avi -c copy data\raw_videos_60s\cam03.avi
```

이 짧은 영상으로 단계 1~4 만 빠르게 통과 (단계 5,6 은 데이터가 적어서 본 실행 때).
명령어 형태는 본 실행과 동일하니, **본 실행 명령어의 경로만 `_60s` 로 바꿔 한 번 돌려본 뒤,
잘 되면 본 실행으로 진행**하세요.

---

## Phase D — 본 실행 6단계

### D.1 단계 1 — YOLO + ByteTrack 으로 트랙별 크롭 추출

```powershell
python 01_extract_tracks.py `
  --videos data\raw_videos\cam01.avi data\raw_videos\cam02.avi data\raw_videos\cam03.avi `
  --out data\crops `
  --meta data\tracks_meta.csv `
  --yolo-weights yolov8s.pt `
  --conf 0.4 --iou 0.5 `
  --sample-fps 2 `
  --min-size 64 128 `
  --device 0
```

**예상 소요**: 5시간 영상 × 3대 = 15시간 분량 → RTX 3060 기준 1.5~3시간.

**산출물 확인**:

```powershell
Get-ChildItem data\crops\cam01 | Select-Object -First 5      # track 폴더들
(Get-Content data\tracks_meta.csv | Measure-Object -Line).Lines   # 행 수
Get-ChildItem -Recurse data\crops -Include *.jpg | Measure-Object | Select-Object Count
```

크롭 이미지가 수만~수십만 장 나옵니다. 디스크 5~20 GB 정도 확보 권장.

### D.2 단계 2 — Cross-camera 매칭 제안

```powershell
python 02_propose_matches.py `
  --crops data\crops `
  --meta data\tracks_meta.csv `
  --out data\proposals.json `
  --sim-threshold 0.55 `
  --lambda-t 0.20 --tau 2.0 `
  --device cuda
# 카메라 offset 이 있으면 --offsets data\offsets.json 추가
```

**예상 소요**: 30분~1시간 (track 수 × 32장 임베딩).

**산출물 확인**:

```powershell
Get-Content data\proposals.json | ConvertFrom-Json | Select-Object -ExpandProperty clusters | Get-Member -Type NoteProperty | Measure-Object | Select-Object Count
# 예: 320 (=총 클러스터 수)
```

콘솔 로그 끝에 `multi-track=NNN, singletons=MMM` 출력. 다중-track 클러스터가 30개 미만이면
임계값 낮춰서 (`--sim-threshold 0.45`) 재실행.

### D.3 단계 3 — Streamlit 으로 동일인 검증

```powershell
streamlit run 03_verify_ui.py -- `
  --crops data\crops `
  --proposals data\proposals.json `
  --out data\persons.json `
  --max-thumbs 12
```

브라우저 `http://localhost:8501` 에서:

1. 우상단 체크박스 **"다중-track cluster만 보기"** ON (먼저 카메라 간 매칭 후보부터 처리)
2. 각 cluster 마다:
   - 같은 사람이면 **Approve as ONE person**
   - 잘못 묶였으면 keep 체크 해제 → 한 명만 남기고 Approve, 나머진 다음에 재처리
   - 다른 cluster 와 합쳐야 하면 **Merge into PID** (이미 만든 PID 4자리 입력)
   - 노이즈(반사·잘림·배경 사람)면 **Discard whole cluster**
3. **수시로 좌상단 💾 Save persons.json** 클릭 — 누르지 않으면 새로고침 시 작업 손실
4. 일단 다중-track 다 처리한 뒤, 체크박스 해제하고 singleton(한 카메라 only) 도 처리

**예상 소요**: 200~500 ID 목표 시 **3~6시간**. 점심 전후로 나눠 진행 권장.

**산출물 확인**:

```powershell
Get-Content data\persons.json | ConvertFrom-Json | Select-Object -ExpandProperty persons | Get-Member -Type NoteProperty | Measure-Object | Select-Object Count
# 예: 287
```

### D.4 단계 4 — Market-1501 디렉터리 생성

학습 스크립트가 기대하는 위치는 `data\market1501_root\market1501\` (configs 참조)이라
`--out` 을 그렇게 맞춰 줍니다.

```powershell
python 04_build_market1501.py `
  --crops data\crops `
  --persons data\persons.json `
  --out data\market1501_root\market1501 `
  --test-ratio 0.2 `
  --min-images-per-id 8 `
  --max-images-per-id 80
```

**예상 소요**: 5~15분 (이미지 복사 위주).

**산출물 확인**:

```powershell
Get-Content data\market1501_root\market1501\_summary.json
# train_pids, test_pids, train_images, gallery_images, query_images 출력
Get-ChildItem data\market1501_root\market1501\bounding_box_train | Measure-Object | Select-Object Count
```

학습 PID 가 30 미만이면 데이터 부족이므로 라벨링 더 모으거나 `--min-images-per-id` 를
낮춰 (예: 5) 재실행.

### D.5 단계 5 — Torchreid 파인튜닝

```powershell
python 05_train_osnet.py --config configs\train_config.yaml
```

**예상 소요**: RTX 3060 기준 60 epoch 약 1.5~3시간 (학습 PID/이미지 수에 비례).

학습 도중 콘솔에 `Epoch: 10  Loss x.xxx  Acc xx.x%` 같은 로그가 흐릅니다.
10 epoch 마다 평가가 들어가 `rank-1: 85.3%  mAP: 71.2%` 같은 결과가 찍힙니다.

**Best 체크포인트 확인**:

```powershell
Get-ChildItem log\osnet_x0_25_eyed\model | Sort-Object LastWriteTime | Select-Object -Last 3
# model.pth.tar-60 같은 파일이 최종
```

**rank-1 가이드**:
- > 75% : 양호. 다음 단계로
- 60~75%: 라벨 노이즈 점검(검증 UI 재방문), epoch 추가
- < 60% : 학습 ID 수 부족 or 이미지 품질 문제. 데이터 보강 후 재학습

### D.6 단계 6 — ONNX 추출

```powershell
mkdir exports -Force
python 06_export_onnx.py `
  --weights log\osnet_x0_25_eyed\model\model.pth.tar-60 `
  --out exports\osnet_x0_25.onnx `
  --image-size 256 128 `
  --l2-norm
```

**예상 소요**: 1~2분.

스크립트가 자동으로:
1. ONNX export
2. onnx-simplifier 로 그래프 단순화
3. PyTorch vs ONNX 출력 max diff 확인 (< 1e-3 이어야 통과)
4. L2 norm 1.0 근접 확인

콘솔 끝에 `PyTorch vs ONNX  max abs diff = X.XXe-07` 가 보이면 성공.

---

## Phase E — EYE-D 서버에 배포 (10~30분, 팀 합의 후)

### E.1 백업

```powershell
$EYED = "C:\Users\murim\OneDrive\문서\Claude\Projects\엣지 게이트웨이 기반 지능형 침입 탐지 및 인물 재식별(Re-ID) 시스템\EYE-D"
$TS = Get-Date -Format "yyyyMMdd_HHmm"
Copy-Item "$EYED\edge\osnet_x0_25.onnx"        "$EYED\edge\osnet_x0_25.onnx.bak_$TS"
# 데이터 파일이 별도로 있는 경우(외부 weights)
if (Test-Path "$EYED\edge\osnet_x0_25.onnx.data") {
  Copy-Item "$EYED\edge\osnet_x0_25.onnx.data" "$EYED\edge\osnet_x0_25.onnx.data.bak_$TS"
}
# server 쪽도 (있다면)
if (Test-Path "$EYED\server\models\osnet_x0_25.onnx") {
  Copy-Item "$EYED\server\models\osnet_x0_25.onnx" "$EYED\server\models\osnet_x0_25.onnx.bak_$TS"
}
```

### E.2 교체

```powershell
Copy-Item "$FT\exports\osnet_x0_25.onnx" "$EYED\edge\osnet_x0_25.onnx" -Force
if (Test-Path "$EYED\server\models") {
  Copy-Item "$FT\exports\osnet_x0_25.onnx" "$EYED\server\models\osnet_x0_25.onnx" -Force
}
```

### E.3 dry-run 검증 (팀과 사전 합의된 시나리오)

1. EYE-D 서버 재기동 → lifespan 단일 인스턴스로 새 모델이 로드되는지 콘솔 확인
2. 에지 측 `reid_extractor.py` 가 새 모델로 같은 입력에 대해 정상 임베딩(512-d, L2≈1)을 내는지 확인
3. 동일 영상 짧은 구간에 대해 신/구 모델 비교 (cosine 분포, top-1 일치율)
4. 문제 시 백업 파일로 즉시 롤백:
   ```powershell
   Copy-Item "$EYED\edge\osnet_x0_25.onnx.bak_$TS" "$EYED\edge\osnet_x0_25.onnx" -Force
   ```

**팀장(joonwhan.bae) 합의 사항 (메모리 기반)**:
- 본인이 안 한 EYE-D 변경이 git status 에 잡혀 있는지 먼저 확인
- dry-run 직전 4항목 합의 후 진행
- DB 포트는 5433 (conn.py fallback 포함)

---

## 자주 마주치는 문제 & 대처

| 증상 | 위치 | 대처 |
| --- | --- | --- |
| `CUDA not available` | A.2 | 드라이버/CUDA toolkit 재확인 |
| `torchreid` import error | A.4 | git clone 한 폴더에서 setup.py develop 했는지 |
| 단계 1 메모리 부족 | D.1 | `--imgsz 640` 으로 축소 |
| Track id 가 너무 자주 바뀜 | D.1 | bytetrack.yaml 의 `track_buffer` 30→60 으로 (ultralytics 폴더 안) |
| 검증 UI 가 너무 느림 | D.3 | `--max-thumbs 8` 로 |
| `data\market1501_root\market1501\bounding_box_train` 가 비어 있음 | D.4 | 04 단계에서 `--min-images-per-id` 낮추기 |
| 학습 OOM | D.5 | configs 의 `batch_size` 32, `height/width` 192/96 으로 |
| ONNX max diff > 1e-3 | D.6 | weights 로드 경로 또는 OSNetWrapper 의 l2 옵션 확인 |
| 서버에서 새 ONNX 로딩 실패 | E.3 | 즉시 백업 롤백 → 입출력 shape 재확인 (N,3,256,128 / N,512) |

---

## 한 줄 요약

```
A. venv + PyTorch + requirements + torchreid 설치
B. cam01~03.avi 를 data\raw_videos\ 에 배치
C. 1분 영상으로 dry-run (실수 비용 방지)
D. 01 → 02 → 03(Streamlit) → 04 → 05 → 06 순서대로 본 실행
E. exports\osnet_x0_25.onnx 를 EYE-D 의 edge\, server\models\ 로 백업 후 교체
```
