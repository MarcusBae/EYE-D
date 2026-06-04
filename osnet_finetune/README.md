# OSNet 파인튜닝 파이프라인 (EYE-D / PS Center)

3대 카메라 × 5시간 AVI 영상에서 시작해 **server/models/osnet_x0_25.onnx 와 호환되는 파인튜닝 모델**까지 만드는 6단계 파이프라인입니다.

---

## 0. 전체 흐름

```
AVI (cam01, cam02, cam03)
        │  ① 01_extract_tracks.py
        ▼
crops/<cam>/<track_id>/*.jpg  +  tracks_meta.csv
        │  ② 02_propose_matches.py
        ▼
proposals.json    (cross-camera 동일인 후보)
        │  ③ 03_verify_ui.py  (Streamlit 수동 검증)
        ▼
persons.json      (확정된 Person ID ↔ (cam,track) 매핑)
        │  ④ 04_build_market1501.py
        ▼
market1501/
  ├─ bounding_box_train/
  ├─ bounding_box_test/
  └─ query/
        │  ⑤ 05_train_osnet.py   (Torchreid 파인튜닝)
        ▼
log/osnet_x0_25_eyed/model/model.pth.tar
        │  ⑥ 06_export_onnx.py
        ▼
server/models/osnet_x0_25.onnx  ← 기존 서버에 그대로 교체 가능
```

각 단계는 **독립 실행 가능**하며, 산출물 파일을 통해 다음 단계로 연결됩니다.

---

## 1. 사전 준비

### 1.1 의존성 설치

로컬 NVIDIA GPU(CUDA) 환경 가정. PyTorch는 CUDA 버전에 맞춰 별도 설치 권장.

```bash
# (1) PyTorch (예: CUDA 12.1)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# (2) 나머지 패키지
pip install -r requirements.txt

# (3) torchreid (소스 설치가 가장 안정적)
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid && python setup.py develop && cd ..
```

> Windows 사용자 메모: `setup.py develop` 시 Visual C++ Build Tools 필요할 수 있음.
> `pip install torchreid` 도 동작하지만, 데이터셋 커스텀 등록은 소스 설치가 편함.

### 1.2 입력 영상 배치

```
osnet_finetune/
└── data/
    └── raw_videos/
        ├── cam01.avi
        ├── cam02.avi
        └── cam03.avi
```

파일명 규칙: `cam<숫자2자리>.avi`. (예: `cam01.avi` → camera_id=1)
3대 영상의 **녹화 시작 시각이 동일하다고 가정**합니다. 다를 경우 `01_extract_tracks.py --offset-csv` 로 카메라별 offset(초)을 지정하세요.

### 1.3 GPU 메모리 가이드

| 단계 | 권장 VRAM | 비고 |
| --- | --- | --- |
| ① YOLO + ByteTrack | 4 GB+ | yolov8s.pt 기준 |
| ② OSNet 임베딩 | 2 GB+ | osnet_x0_25 |
| ⑤ Torchreid 학습 | 8 GB+ | batch=64, image 256×128 기준 |

---

## 2. 단계별 실행

### Step 1 — 트랙별 크롭 추출

```bash
python 01_extract_tracks.py \
  --videos data/raw_videos/cam01.avi data/raw_videos/cam02.avi data/raw_videos/cam03.avi \
  --out data/crops \
  --yolo-weights yolov8s.pt \
  --conf 0.4 --iou 0.5 \
  --sample-fps 2 \
  --min-size 64 128
```

- YOLOv8 + ByteTrack 내장 트래커로 사람만 검출/추적
- `--sample-fps 2` : 트랙당 초당 2장씩만 저장(저장공간/노이즈 감소)
- 너무 작거나 검출 신뢰도가 낮은 박스는 자동 제외
- **부산물**:
  - `data/crops/cam01/<track_id>/<frame>.jpg`
  - `data/tracks_meta.csv` (camera, track_id, frame, t_sec, bbox, conf)

### Step 2 — Cross-camera 매칭 제안

```bash
python 02_propose_matches.py \
  --crops data/crops \
  --meta data/tracks_meta.csv \
  --out data/proposals.json \
  --temporal-overlap-sec 3.0 \
  --sim-threshold 0.55
```

- 사전학습 OSNet(`osnet_x0_25`, ImageNet/Market-1501)으로 **track별 평균 임베딩** 산출
- 부분 중첩 환경이므로 **시간 동시 발생(co-occurrence)** 을 보조 신호로 사용
  - 동일 시각(±tolerance)에 서로 다른 카메라에서 잡힌 track 쌍에 가중치
- **Hungarian + threshold** 로 카메라간 1:1 후보를 묶음
- 결과: `data/proposals.json` (cluster_id → [(cam, track_id), ...])

### Step 3 — Streamlit 검증 UI

```bash
streamlit run 03_verify_ui.py
```

브라우저에서:

1. **Pending Clusters** 탭에서 후보 클러스터를 순회
2. 각 cluster의 thumbnail grid를 보고
   - Approve : 동일인 확정
   - Split   : 잘못 묶인 track 을 분리
   - Merge   : 다른 cluster 와 합치기
   - Discard : 노이즈 트랙(반사, 그림자, 잘림) 제거
3. 좌측 상단 **Save persons.json** 으로 진행 상황 저장 (수시로!)

권장 작업 시간: 200~500 ID 기준 **3~6 시간**. 점심/오후로 나눠 진행.

### Step 4 — Market-1501 포맷 변환

```bash
python 04_build_market1501.py \
  --crops data/crops \
  --persons data/persons.json \
  --out data/market1501 \
  --test-ratio 0.2 \
  --min-images-per-id 8
```

- Person ID 4자리, 카메라 ID 6 cameras 까지 (Market-1501 호환)
- 파일명: `0001_c1s1_000023_01.jpg`
- **ID 단위 split**: 학습 ID와 테스트 ID는 겹치지 않게 (Re-ID 표준)
- `query/` 는 각 테스트 ID에서 카메라별 1장씩 샘플링

### Step 5 — Torchreid 파인튜닝

```bash
python 05_train_osnet.py --config configs/train_config.yaml
```

기본 설정:

- backbone: `osnet_x0_25`
- 초기화: Market-1501 pretrained weight
- loss: softmax + triplet (combined)
- optimizer: Adam + warmup + cosine
- epochs: 60 (max-epoch), lr=3e-4
- 입력: 256×128, RandomErasing, ColorJitter
- 평가: rank-1, rank-5, mAP

체크포인트: `log/osnet_x0_25_eyed/model/model.pth.tar-XX`

### Step 6 — ONNX 추출 (서버 사양 호환)

```bash
python 06_export_onnx.py \
  --weights log/osnet_x0_25_eyed/model/model.pth.tar-60 \
  --out exports/osnet_x0_25.onnx \
  --image-size 256 128 \
  --l2-norm
```

- 출력 텐서: **(N, 512)**, **L2 normalized** → 기존 `server/models/osnet_x0_25.onnx` 와 1:1 교체 가능
- onnxruntime fp32. (필요시 `--fp16` 로 변환)
- 검증: 같은 입력에 대해 PyTorch vs ONNX 출력의 max 차이 < 1e-4 확인

---

## 3. 산출물 / 다른 프로젝트 연계

이 파이프라인의 **단계별 산출물은 모두 표준 포맷**이라, 다음 작업에 재활용 가능합니다.

- `tracks_meta.csv` + `crops/` → **arttrace_master** 의 관객 행동 분석(체류/이동 패턴) 입력
- `persons.json` → **Smart Retail** 의 VIP/단골 카탈로그 부트스트랩
- `market1501/` → 새로운 backbone(osnet_x1_0, fastreid 등) 학습에도 그대로 사용
- `exports/osnet_x0_25.onnx` → EYE-D server `lifespan` 모델 핫스왑

---

## 4. 문제 해결 체크리스트

| 증상 | 원인/대처 |
| --- | --- |
| YOLO가 사람을 거의 못 잡음 | `--conf 0.25` 로 낮추거나 yolov8m/yolov8l 로 교체 |
| Track ID가 너무 자주 바뀜 (id-switch) | `--track-buffer 60` 또는 ByteTrack 의 match-thresh 조정 |
| 사전학습 OSNet 유사도가 전반적으로 낮음 | 조명·각도 차가 큼 → `--sim-threshold` 를 0.45 까지 낮춰 후보를 넓힘 (검증 UI 부담 ↑) |
| Streamlit 메모리 폭증 | 한 cluster 당 표시 이미지 수를 `--max-thumbs 16` 로 제한 |
| Torchreid CUDA OOM | `batch_size` ↓, `image_size` 192×96 로 축소 |
| 파인튜닝 후 rank-1 < pretrained | (1) ID 라벨 노이즈 점검, (2) 학습 ID 수 < 30 이면 데이터 부족 |

---

## 5. 작업 순서 요약 (체크리스트)

- [ ] `data/raw_videos/cam0X.avi` 3개 배치
- [ ] `pip install -r requirements.txt` + torchreid 설치
- [ ] `01_extract_tracks.py` 실행 → `tracks_meta.csv` 생성 확인
- [ ] `02_propose_matches.py` 실행 → `proposals.json` 생성 확인
- [ ] `streamlit run 03_verify_ui.py` → 검증 → `persons.json` 저장
- [ ] `04_build_market1501.py` → `data/market1501/` 생성 확인 (ID 수, 이미지 수 로그 점검)
- [ ] `05_train_osnet.py` 실행 → rank-1 / mAP 확인
- [ ] `06_export_onnx.py` 실행 → onnx 검증 출력 < 1e-4
- [ ] `server/models/osnet_x0_25.onnx` 교체 → EYE-D 서버 재기동 dry-run
