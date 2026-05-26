# PRD 성능 목표 달성 Plan 

## 목차

- [성능 목표](#성능-목표)
- [목표치 설정 근거](#목표치-설정-근거)
- [Rank-1 80% 달성을 위한 조합 전체 비교표](#rank-1-80-달성을-위한-조합-전체-비교표)
  - [4가지 축](#4가지-축)
  - [조합 전체 비교](#조합-전체-비교)
  - [권장 조합](#권장-조합)
- [향후 프로젝트 진행 계획](#향후-프로젝트-진행-계획)
  - [현재 완료된 항목](#현재-완료된-항목-planmd-기준-done-12개)
  - [Phase 1 — 즉시 개선](#phase-1--즉시-개선-local-pc--저장-영상-기반)
  - [Phase 2 — 고도화 및 RTSP 스트림 검증](#phase-2--고도화-및-rtsp-스트림-검증-local-pc--rtsp-에뮬레이터)
  - [Phase 3 — Jetson 실기기 배포 및 실환경 검증](#phase-3--jetson-실기기-배포-및-실환경-검증)
  - [단계별 요약](#단계별-요약)
  - [Action Item 보충 설명](#action-item-보충-설명)
- [9. 파이프라인 단계별 대안 옵션](#9-파이프라인-단계별-대안-옵션)
  - [9.1 사람 인식 대안](#91-1단계-사람-인식-person-detection-대안)
  - [9.2 Motion Tracking 대안](#92-2단계-motion-tracking-대안)
  - [9.3 트래커 선택에 따른 구조 변화](#93-트래커-선택에-따른-파이프라인-구조-변화)
  - [9.4 실용적 추천 변경 조합](#94-실용적-추천-변경-조합)
- [10. OSNet 도메인 파인튜닝 실행 가이드](#10-osnet-도메인-파인튜닝-실행-가이드)
  - [10.1 전체 흐름](#101-전체-흐름)
  - [10.2 크롭 이미지 자동 추출](#102-1단계-크롭-이미지-자동-추출)
  - [10.3 Person ID 매핑](#103-2단계-person-id-매핑-영상-간-동일인-연결)
  - [10.4 Market-1501 형식으로 정리](#104-3단계-market-1501-형식으로-정리)
  - [10.5 Torchreid 파인튜닝 실행](#105-4단계-torchreid-파인튜닝-실행)
  - [10.6 데이터 요건 및 기대 효과](#106-데이터-요건-및-기대-효과)
  - [10.7 주의사항](#107-주의사항)

---

## 성능 목표

| 우선순위 | 항목 | Phase 1 (Local PC · 파일 기반) | Phase 2 (RTSP 에뮬레이터) | Phase 3 (Jetson · 실 카메라) |
|:---:|---|:---:|:---:|:---:|
| 1 | Re-ID Rank-1 정확도 | **≥ 70%** | **≥ 80%** | **≥ 80%** |
| 2 | Re-ID mAP ² | **≥ 45%** | **≥ 60%** | **≥ 55%** |
| 3 | False Negative (재인식 실패) ¹ | **≤ 20%** | **≤ 12%** | **≤ 15%** |
| 4 | False Positive (오인식) | **≤ 5%** | **≤ 5%** | **≤ 5%** |
| 5 | ID Switching | **≤ 5%** | **≤ 5%** | **≤ 5%** |
| 6 | FPS | — | — | **45~60 FPS** (3채널 합산) |
| 7 | 벡터 검색 응답 | — | — | **50ms 이내** |
| 8 | 대시보드 갱신 | — | — | **500ms 이내** |
| 9 | 가용성 | — | — | **99% 이상** |

---

## Rank-1 80% 달성을 위한 조합 전체 비교표

### 4가지 축

| 축 | 선택지 |
|---|---|
| **모델** | osnet_x0_25 (현재·경량) / osnet_x1_0 (풀사이즈) / TransReID (ViT 기반·고성능) |
| **학습 방법** | 범용 사전학습 (현재) / 도메인 파인튜닝 (실환경 데이터 재학습) |
| **Matching 전략** | Single Detection (현재) / Mean Pooling / Confidence-weighted Pooling |
| **임계값 전략** | 0.85 고정 (현재) / 0.73 고정 (권장) / 영상별 적응형 |

### 조합 전체 비교

| # | 모델 | 학습 | Matching | 임계값 | 예상 Rank-1 | Jetson FPS | 구현 난이도 | 80% 달성 |
|:---:|---|---|---|---|:---:|:---:|:---:|:---:|
| 1 | osnet_x0_25 | 범용 | Single | 0.85 (현재) | **61.5%** (실측) | 빠름 | — | ✗ |
| 2 | osnet_x0_25 | 범용 | Single | 0.73 (권장) | **71.0%** (실측) | 빠름 | 매우 낮음 | ✗ |
| 3 | osnet_x0_25 | 범용 | Single | 영상별 적응형 | ~73% | 빠름 | 낮음 | ✗ |
| 4 | osnet_x0_25 | 범용 | Mean Pooling | 0.73 | ~74~76% | 빠름 | 낮음 | ✗ |
| 5 | osnet_x0_25 | 범용 | Conf-weighted Pooling | 0.73 | ~75~77% | 빠름 | 낮음 | ✗ |
| 6 | osnet_x0_25 | 범용 | Conf-weighted Pooling | 영상별 적응형 | ~76~78% | 빠름 | 중간 | ✗ |
| 7 | osnet_x1_0 | 범용 | Single | 0.73 | ~70~72% | 보통 | 낮음 | ✗ |
| 8 | osnet_x1_0 | 범용 | Mean Pooling | 영상별 적응형 | ~74~77% | 보통 | 중간 | ✗ |
| 9 | TransReID | 범용 | Single | 영상별 적응형 | ~76~82% | 느림 ⚠️ | 중간 | △ |
| 10 | TransReID | 범용 | Mean Pooling | 영상별 적응형 | ~79~84% | 느림 ⚠️ | 중간 | △ |
| 11 | osnet_x0_25 | 파인튜닝 | Single | 0.73 | ~78~82% | 빠름 | 높음 | △ |
| 12 | osnet_x0_25 | 파인튜닝 | Single | 영상별 적응형 | ~79~83% | 빠름 | 높음 | △ |
| 13 | **osnet_x0_25** | **파인튜닝** | **Mean Pooling** | **0.73** | **~80~85%** | **빠름** | **높음** | **✓** |
| 14 | osnet_x0_25 | 파인튜닝 | Conf-weighted Pooling | 영상별 적응형 | ~81~86% | 빠름 | 높음 | ✓ |
| 15 | osnet_x1_0 | 파인튜닝 | Single | 영상별 적응형 | ~80~85% | 보통 | 높음 | ✓ |
| 16 | **osnet_x1_0** | **파인튜닝** | **Mean Pooling** | **영상별 적응형** | **~82~87%** | **보통** | **높음** | **✓** |
| 17 | osnet_x1_0 | 파인튜닝 | Conf-weighted Pooling | 영상별 적응형 | ~83~88% | 보통 | 매우 높음 | ✓ |
| 18 | TransReID | 파인튜닝 | Mean Pooling | 영상별 적응형 | ~85~90% | 느림 ⚠️ | 매우 높음 | ✓ |

#### 판정 기호
- ✓ : 80% 달성 가능 (예상 범위 하한이 80% 이상)
- △ : 조건부 달성 가능 (예상 범위가 80% 경계에 걸침)
- ✗ : 80% 미달
- ⚠️ : Jetson Orin Nano에서 FPS 목표(채널당 15 FPS) 달성 불확실

### 권장 조합

| 우선순위 | 조합 | 이유 |
|:---:|---|---|
| **1순위** | **#13** osnet_x0_25 + 파인튜닝 + Mean Pooling + 0.73 | Jetson FPS 유지 + 80% 달성 + 구현 현실적 |
| **2순위** | **#16** osnet_x1_0 + 파인튜닝 + Mean Pooling + 적응형 | 더 높은 정확도, FPS 검증 필요 |
| **빠른 개선** | **#4** osnet_x0_25 + 범용 + Mean Pooling + 0.73 | 파인튜닝 없이 즉시 적용 가능, 74~76% 예상 |

---

## 향후 프로젝트 진행 계획

> 환경: Phase 1~2는 Local PC, Phase 3부터 Jetson Orin Nano 실기기 투입

### 현재 완료된 항목 (plan.md 기준 Done, 12개)

| 분류 | 완료 항목 |
|---|---|
| Edge 코어 | YOLO 탐지 엔진, 파이프라인 러너 통합, 로컬 Qdrant DB 연동, 실행 진입점 구조화, Best-shot 선별 |
| Edge 인프라 | Mediamtx 기반 RTSP 에뮬레이터 |
| Server | Analytics 통계 API, pgvector 유사도 검색 기초 |
| Frontend | 대시보드 차트 시각화 |
| Testing | 단위 테스트(55개), 모의 트래픽 주입 E2E 검증 |
| Common | OSNet 모델 가중치 에지/서버 일치 |

> Phase 1~3의 Action Item은 모두 위 Done 항목 위에 추가로 수행해야 하는 작업들임.

---

### Phase 1 — 즉시 개선 (Local PC + 저장 영상 기반)
**목표**: 코드·설정 변경만으로 Re-ID 성능 향상 및 서버·프론트엔드 기본 기능 완성

| # | 분류 | Action Item | 근거 | 예상 효과 | 노트북 검증 | Status |
|:---:|---|---|---|---|:---:|:---:|
| 1-1 | **Re-ID 튜닝** | `.env` `REID_SIMILARITY_THRESHOLD=0.73` 적용 | 실측: FN율 39%→18%, Rank-1 +9.5%p | 즉시 | ✓ 직접 | `Done` |
| 1-3 | **Re-ID 튜닝** | Temporal Mean Pooling 구현 (plan.md 3순위 개선안) | Rank-1 +3~7%p 예상 | 중간 | ✓ 직접 | `Todo` |
| 1-11 | **평가 지표** | 노트북에 mAP 평가 구현 (gallery/query 분리 기반) | mAP 기준값 측정 및 Phase 1 목표(≥45%) 검증 | 낮음 | ✓ 직접 | `Todo` |
| 1-2 | **Re-ID 튜닝** | 화질 게이트 구현 (라플라시안 분산 기반 저품질 프레임 Re-ID 스킵) | 저품질 프레임 오염 제거 | 낮음~중간 | △ 간접 | `Todo` |
| 1-4 | **파이프라인** | Re-ID 전용 큐 분리 (표시용 크기 1~2 / Re-ID용 크기 5~10) | Best-shot 품질 보장 | 중간 | △ 간접 | `Todo` |
| 1-5 | **파이프라인** | `max_missing_frames` 값 튜닝 (재연결 대응) | 짧은 끊김 시 Track ID 유지 | 설정 변경 | ✗ 불가 | `Todo` |
| 1-6 | **서버** | OSNet 모델 Lifespan 캐싱 구현 | API 응답 지연 제거 | 중간 | ✗ 불가 | `Todo` |
| 1-7 | **서버** | 서버 이미지 검색 API 구현 (`POST /v1/search/identity`) | 이미지 기반 인물 검색 기능 | 중간 | ✗ 불가 | `Todo` |
| 1-8 | **서버** | pgvector HNSW 인덱스 생성 | 벡터 검색 50ms 목표 달성 | 낮음 | ✗ 불가 | `Todo` |
| 1-9 | **프론트엔드** | 실시간 데이터 바인딩 (WebSocket / HTTP Polling) | 대시보드 실시간 갱신 | 중간 | ✗ 불가 | `Todo` |
| 1-10 | **프론트엔드** | 이미지 검색 업로드 위젯 구현 | Re-ID 검색 UI 완성 | 중간 | ✗ 불가 | `Todo` |

> **노트북 검증 기호**
> - ✓ 직접: pkl 벡터 데이터 기반으로 노트북에서 즉시 파라미터·코드 변경 → 효과 확인 가능
> - △ 간접: 파이프라인 코드 수정 후 영상 재처리 → 새 pkl 생성 → 노트북으로 전후 비교
> - ✗ 불가: 서버·프론트엔드·RTSP 런타임 동작, 노트북 검증 범위 밖

#### 노트북 검증 불가 항목의 대체 검증 방법

| # | Action Item | 검증 방법 | 검증 도구 | 합격 기준 |
|:---:|---|---|---|---|
| 1-5 | `max_missing_frames` 튜닝 | Mediamtx 에뮬레이터로 RTSP 송출 중 `tc netem`으로 스트림 강제 차단(3~5초) 후 복구 → 재연결 전후 Track ID 유지 여부 로그 확인 | `tc netem`, 파이프라인 로그 | 끊김 전후 동일 Track ID 유지 |
| 1-6 | OSNet Lifespan 캐싱 | 서버 기동 후 API 첫 요청(모델 로딩 포함)과 이후 요청의 응답 시간 비교 | `curl` + `time`, k6 / locust | 2번째 요청부터 100ms 이내 |
| 1-7 | 이미지 검색 API | 인물 크롭 이미지를 `POST /v1/search/identity`에 업로드 → 유사 인물 이력(카메라 ID·타임스탬프·스냅샷 경로) 반환 확인. pytest 자동화 추가 권장 | `curl` / Postman, pytest | 동일 인물 이력 Top-K 반환 |
| 1-8 | pgvector HNSW 인덱스 | 인덱스 생성 전후 `EXPLAIN ANALYZE`로 벡터 유사도 쿼리 실행 시간 측정. 10,000건 삽입 후 비교 | PostgreSQL `EXPLAIN ANALYZE` | 쿼리 응답 50ms 이내 |
| 1-9 | 실시간 데이터 바인딩 | 에지 파이프라인이 이벤트를 서버로 전송한 시각과 대시보드 차트에 반영된 시각 차이 측정 | 브라우저 개발자 도구 Network 탭, 서버 로그 타임스탬프 | 전송→갱신 500ms 이내 |
| 1-10 | 이미지 검색 위젯 | 브라우저에서 인물 이미지 Drag & Drop → 검색 결과 카드(카메라·시간·스냅샷) 정상 렌더링 확인. 잘못된 파일 형식·빈 결과 등 엣지 케이스도 포함 | 브라우저 직접 테스트, 개발자 도구 콘솔 | 이미지 업로드 후 결과 카드 표시 |

**Phase 1 완료 기준**
- Rank-1 정확도 ≥ 70% (로컬 파일 기준)
- mAP ≥ 45% (노트북 측정)
- 서버 벡터 검색 응답 50ms 이내
- 대시보드에서 실시간 통계 확인 가능

---

### Phase 2 — 고도화 및 RTSP 스트림 검증 (Local PC + RTSP 에뮬레이터)
**목표**: 실 스트림 환경 대응, 도메인 파인튜닝으로 Rank-1 80% 달성 시도, 시스템 안정성 검증

> **완료된 기반 항목 (plan.md Done)**: Mediamtx RTSP 에뮬레이터 인프라 구축 완료. 아래 2-1은 이를 활용한 테스트 활동으로 별도 수행 필요.

| # | 분류 | Action Item | 근거 | 예상 효과 | Status |
|:---:|---|---|---|---|:---:|
| 2-1 | **RTSP 테스트** | Mediamtx 에뮬레이터로 3채널 동시 RTSP 스트림 테스트 수행 | 실기기 없이 네트워크 스트림 재현 | — | `Done` (인프라) / `Todo` (테스트) |
| 2-2 | **RTSP 테스트** | I-프레임 우선 Re-ID 추출 구현 및 스트림 환경 Rank-1 측정 | 압축 아티팩트 최소화 | 중간 | `Todo` |
| 2-3 | **Re-ID 고도화** | 실환경 데이터 수집 (설치 환경 영상 촬영) | 도메인 파인튜닝 전제 조건 | — | `Todo` |
| 2-4 | **Re-ID 고도화** | osnet_x0_25 도메인 파인튜닝 | Rank-1 +10~15%p 예상, 80% 목표 | 높음 | `Todo` |
| 2-5 | **Re-ID 고도화** | 영상별 적응형 임계값 적용 (카메라 환경별 최적값 사용) | 환경 변화 대응 | 중간 | `Todo` |
| 2-6 | **추적기 검증** | 짧은 클립 수동 라벨링 + `py-motmetrics`로 MOTA/IDF1 측정 | 추적기 성능 Re-ID와 분리 측정 | — | `Todo` |
| 2-7 | **파이프라인** | 이벤트 트리거 전송 구현 (Enter/Exit/Line Crossing) | 서버 부하 절감 | 중간 | `Todo` |
| 2-8 | **파이프라인** | 다채널 병렬 처리 완성 (MultiStreamPipelineRunner) | 3채널 동시 입력 대응 | 높음 | `Todo` |
| 2-9 | **안정성** | SQLite 오프라인 버퍼 + Batch Flush 데몬 구현 | 네트워크 단절 시 데이터 유실 방지 | 높음 | `Todo` |
| 2-10 | **안정성** | 네트워크 장애 모의 E2E 테스트 | 데이터 정합성 검증 | 중간 | `Todo` |
| 2-11 | **프론트엔드** | 보행자 동선 타임라인 위젯 구현 | 인물 이동 경로 시각화 | 중간 | `Todo` |
| 2-12 | **서버** | 카메라 교차 매칭 임계값 튜닝 | 크로스 카메라 Re-ID 정확도 개선 | 중간 | `Todo` |

**Phase 2 완료 기준**
- Rank-1 정확도 ≥ 80% (파인튜닝 후, 로컬 파일 기준)
- RTSP 에뮬레이터 3채널 환경에서 안정적 동작 확인
- 네트워크 단절 복구 후 데이터 정합성 100%
- MOTA/IDF1 1회 이상 측정 완료

---

### Phase 3 — Jetson 실기기 배포 및 실환경 검증
**목표**: Jetson Orin Nano 하드웨어 최적화, 실 IP 카메라 연동, 전체 시스템 통합 검증

| # | 분류 | Action Item | 근거 | 예상 효과 | Status |
|:---:|---|---|---|---|:---:|
| 3-1 | **하드웨어 가속** | Jetson에서 YOLOv8n TensorRT FP16 엔진 변환 및 FPS 측정 | 채널당 15 FPS 이상 목표 | 높음 | `Todo` |
| 3-2 | **하드웨어 가속** | OSNet ONNX Runtime CUDA 가속 검증 (`onnxruntime-gpu`) | Re-ID 추론 속도 확보 | 높음 | `Todo` |
| 3-3 | **실기기 검증** | 3채널 동시 실행 FPS 벤치마크 (목표: 45~60 FPS 합산) | PRD 목표 달성 확인 | — | `Todo` |
| 3-4 | **실기기 검증** | Jetson GPU/VRAM/온도 모니터링 (`jtop`) 하에 장시간 안정성 테스트 | OOM·과열 리스크 점검 | — | `Todo` |
| 3-5 | **실환경 연동** | 실 IP 카메라 RTSP 스트림 연결 및 화질 확인 | 카메라 비트레이트·해상도 설정 최적화 | — | `Todo` |
| 3-6 | **실환경 연동** | 실환경 RTSP 기반 Rank-1 재측정 (Phase 2 파인튜닝 모델 적용) | 스트림 환경 실제 성능 확인 | — | `Todo` |
| 3-7 | **배포** | Docker 컨테이너 빌드 (Jetson Native Build) | 재현 가능한 배포 환경 구성 | — | `Todo` |
| 3-8 | **배포** | `manage_stream.sh` 연동 및 프로덕션 실행 검증 | 통합 실행 스크립트 안정화 | — | `Todo` |
| 3-9 | **모니터링** | 에지 장비 모니터링 카드 (CPU/GPU/온도 실시간 전송) | 대시보드 헬스 상태 가시화 | 중간 | `Todo` |
| 3-10 | **통합 검증** | E2E 전체 시스템 통합 테스트 (카메라 → 에지 → 서버 → 대시보드) | PRD Definition of Done 전 항목 검증 | — | `Todo` |

**Phase 3 완료 기준 (PRD Definition of Done)**
- 3채널 RTSP 입력 안정적 연결, 45 FPS 이상 유지
- Re-ID Rank-1 정확도 ≥ 80% (실 스트림 환경 기준)
- 서버 DB 정상 적재, 대시보드 500ms 이내 갱신
- Docker 기반 전체 시스템 배포 가능, 모니터링 동작

---

### 단계별 요약

| Phase | 환경 | 핵심 목표 | 주요 리스크 |
|:---:|---|---|---|
| **Phase 1** | Local PC | 설정·코드 최소 변경으로 즉시 개선, 서버·UI 기본 완성 | 낮음 |
| **Phase 2** | Local PC + RTSP 에뮬레이터 | 도메인 파인튜닝으로 Rank-1 80% 달성, 스트림 안정성 확보 | 파인튜닝 데이터 수집 부담 |
| **Phase 3** | Jetson Orin Nano + 실 카메라 | TensorRT 가속으로 FPS 목표 달성, 실환경 최종 검증 | FPS 미달 가능성, 하드웨어 OOM |

---

### Action Item 보충 설명

#### Phase 1

**1-1. `.env` REID_SIMILARITY_THRESHOLD=0.73 적용**
* 현재 임계값 0.85는 너무 엄격해 같은 사람을 새 사람으로 판단하는 FN율이 39.4%에 달한다.
* 노트북 실측 결과 0.73이 FN+FP 합산 오류를 최소화하는 최적값으로 도출됐다.
* `.env` 파일 한 줄 변경으로 FN율을 17.9%로 낮추고 Rank-1을 61.5%→71.0%로 즉시 개선할 수 있다.
* 영상마다 최적값이 0.70~0.77로 다르므로 적용 후 결과를 노트북으로 재확인하는 것이 좋다.

**1-2. 화질 게이트 (라플라시안 분산 기반)**
* 라플라시안 필터로 프레임의 선명도 점수를 계산해, 일정 임계값 미만의 흐릿한 프레임은 Re-ID 추출을 건너뛴다.
* 네트워크 압축 아티팩트나 모션 블러로 오염된 프레임이 임베딩 품질을 저하시키는 것을 방지한다.
* `cv2.Laplacian(frame, cv2.CV_64F).var()`로 20줄 이내 구현 가능하며, `ImagePreprocessor` 또는 `ReIDExtractor` 호출 직전에 삽입한다.

**1-3. Temporal Mean Pooling 구현**
* 동일 Track ID의 체류 기간 동안 추출된 복수의 Re-ID 벡터를 단순 평균(Mean Pooling)하거나
* YOLO 신뢰도 가중 평균(Confidence-weighted Pooling)으로 합산해 단일 고품질 벡터를 생성한다.
* 단일 프레임 벡터는 포즈·조명 노이즈에 취약하지만, 평균 벡터는 이를 통계적으로 상쇄한다.
* `BestShotSelector`에 누적 벡터 목록을 유지하다가 트랙 소멸 시 평균을 전송하도록 수정한다.

**1-4. Re-ID 전용 큐 분리**
* 현재 `ThreadedPipelineRunner`는 실시간성을 위해 프레임 큐 크기를 1~2로 유지하며 최신 프레임만 처리한다.
* 이 전략은 화면 표시에는 적합하지만 Re-ID Best-shot 후보 프레임이 드롭되는 문제를 일으킨다.
* 표시용 큐(크기 1~2)와 Re-ID 추출용 큐(크기 5~10)를 분리해, 표시는 최신 프레임만 유지하되 Re-ID는 더 많은 후보를 평가할 수 있게 한다.

**1-5. `max_missing_frames` 값 튜닝**
* `BestShotSelector`의 `max_missing_frames`는 객체가 연속 몇 프레임 동안 미검출될 때 소멸로 판정할지 결정한다.
* 기본값 30프레임(약 1~2초)은 RTSP 재연결 시간보다 짧을 수 있다.
* 재연결 예상 시간(예: 3~5초 → 75~125프레임)에 맞게 값을 늘리면 짧은 끊김 후 재연결 시 동일 Track ID를 유지할 수 있다.

**1-6. OSNet 모델 Lifespan 캐싱 구현**
* 현재 서버의 이미지 검색 API는 호출마다 OSNet 모델을 로드하는 구조로, 응답 지연이 수 초에 달할 수 있다.
* FastAPI의 `lifespan` 이벤트(startup)에서 OSNet 모델을 RAM/VRAM에 1회 로드하고 전역 상태로 유지하면,
* 이후 API 호출은 모델 로딩 없이 즉시 임베딩 추출이 가능하다. `server/models/osnet_x0_25.onnx` 경로는 에지와 동일한 가중치를 참조해야 한다.

**1-7. 서버 이미지 검색 API 구현 (`POST /v1/search/identity`)**
* 관리자가 웹 대시보드에서 인물 이미지를 업로드하면 서버가 OSNet으로 임베딩을 추출하고 pgvector DB와 코사인 유사도를 비교해 해당 인물의 과거 이력을 반환하는 API다.
* 입력은 `.jpg`/`.png` 멀티파트 폼, 출력은 유사도 상위 K건의 카메라 ID·타임스탬프·스냅샷 경로 목록이다.
* 1-6의 Lifespan 캐싱이 완료된 후 구현해야 응답 속도가 보장된다.

**1-8. pgvector HNSW 인덱스 생성**
* pgvector는 인덱스 없이도 정확한 코사인 유사도 검색이 가능하지만, 데이터가 수만 건을 넘으면 전수 탐색(Brute-force) 방식으로 인해 응답 시간이 급격히 증가한다.
* HNSW(Hierarchical Navigable Small World) 인덱스를 사전에 빌드해두면 10,000건 기준 50ms 이내 검색이 가능해진다.
* `CREATE INDEX ON events USING hnsw (vector vector_cosine_ops);` 한 줄로 생성하며, 인덱스 빌드는 데이터 적재 전 수행하는 것이 권장된다.

**1-9. 실시간 데이터 바인딩 (WebSocket / HTTP Polling)**
* 현재 대시보드 차트는 정적 데이터를 표시한다.
* WebSocket 또는 주기적 HTTP Polling을 추가해 에지에서 전송되는 실시간 입/퇴장 카운트와 체류 시간 통계를 차트에 즉시 반영한다.
* 구현 난이도가 낮은 HTTP Polling(3~5초 주기)으로 먼저 구성하고, 이후 WebSocket으로 전환하는 것이 안전하다.

**1-10. 이미지 검색 업로드 위젯 구현**
* 1-7의 서버 API와 연동하는 프론트엔드 컴포넌트다.
* Drag & Drop으로 인물 이미지를 업로드하면 `POST /v1/search/identity`를 호출하고, 반환된 이력(카메라·시간·스냅샷)을 카드 형태로 표시한다.
* React의 `react-dropzone` 라이브러리를 활용하면 빠르게 구현 가능하다.

**1-11. 노트북 mAP 평가 구현**
* mAP(mean Average Precision)는 Rank-1과 함께 Re-ID 성능의 완전한 그림을 제공하는 표준 지표다.
* Rank-1이 "1위 매칭의 정확성"을 보는 반면, mAP는 "한 인물의 모든 출현을 얼마나 빠짐없이 찾는가"를 측정하므로 SS-01(실종자 전수 조사) use case에서 특히 중요하다.
* 구현 방법: `results/*.pkl`의 트랙 임베딩을 gallery/query로 절반씩 분리하고 `average_precision_score`로 per-query AP를 계산한 뒤 평균을 낸다.
* Phase 1 목표(≥45%)는 추정치이므로, 실측 기준값에 따라 Phase 2/3 목표를 재조정한다.

---

#### Phase 2

**2-1. Mediamtx 에뮬레이터 3채널 동시 RTSP 스트림 테스트**
* Mediamtx 인프라는 이미 구축(Done)됐다.
* 이제 실제 영상 파일 3개를 동시에 RTSP로 송출하고, `MultiStreamPipelineRunner`가 3채널을 안정적으로 수신·처리하는지 확인하는 테스트를 수행한다.
* 네트워크 지연과 패킷 손실을 `tc netem`으로 인위적으로 주입해 RTSP 환경에서의 성능 열화를 측정한다.

**2-2. I-프레임 우선 Re-ID 추출**
* H.264/H.265 스트림에서 P/B 프레임은 이전 프레임 차분으로 구성돼 블로킹 아티팩트가 심하다.
* GStreamer 파이프라인 또는 OpenCV 프레임 메타데이터에서 I-프레임 여부를 감지하고, Re-ID 추출을 I-프레임에 우선 할당한다. 2-1 테스트와 병행해 Rank-1 변화를 측정한다.

**2-3. 실환경 데이터 수집**
* 도메인 파인튜닝(2-4)의 전제 조건이다.
* 실제 카메라가 설치될 공간(또는 유사 환경)에서 다양한 조명·의상·각도로 10~30명을 촬영해 Re-ID 학습 데이터를 구성한다.
* 인물당 최소 6개 카메라 뷰(2개 카메라 × 3회 이상 재방문) 이상 확보해야 파인튜닝 효과가 나타난다.
* 데이터 수집량이 적을수록 파인튜닝 효과가 제한적이므로 최대한 다양한 조건에서 수집한다.

**2-4. osnet_x0_25 도메인 파인튜닝**
* 2-3에서 수집한 실환경 데이터로 osnet_x0_25를 추가 학습한다.
* Torchreid의 `ImageDataManager`와 `Engine`을 사용하면 기존 코드 구조를 크게 바꾸지 않고 파인튜닝이 가능하다.
* Triplet Loss 또는 ID Loss + Triplet Loss 조합이 Re-ID에 효과적이다.
* 파인튜닝 전후 노트북으로 Rank-1을 재측정해 효과를 정량화하고, 개선이 미미하면 osnet_x1_0 파인튜닝(조합 #16)으로 전환을 고려한다.

**2-5. 영상별 적응형 임계값**
* 단일 고정 임계값(0.73)은 영상마다 최적값이 0.70~0.77로 다른 상황에서 차선책이다.
* 카메라별로 `.env`에 별도 임계값 변수(`REID_THRESHOLD_CAM01`, `REID_THRESHOLD_CAM02` 등)를 두거나,
* 초기 구동 시 짧은 캘리브레이션 구간에서 intra/inter 유사도 분포를 자동 측정해 최적값을 계산하는 자동 보정 로직을 구현한다.

**2-6. 짧은 클립 수동 라벨링 + MOTA/IDF1 측정**
* 50~100명 규모의 영상 클립(5~10분)에 BBox + 인물 ID를 수동 라벨링하고
* `py-motmetrics` 라이브러리로 MOTA, IDF1, IDsw를 측정한다.
* 현재 노트북의 임베딩 기반 proxy 지표와 달리, 이 지표는 Re-ID 모델의 영향을 받지 않는 순수한 추적기 성능값이다. Phase 2 파인튜닝 전후 추적기 성능 변화도 함께 확인한다.

**2-7. 이벤트 트리거 전송 구현 (Enter/Exit/Line Crossing)**
* 현재는 Best-shot 선별 후 트랙 소멸 시 전송하는 방식이다.
* 진입(Enter), 가상 카운팅 라인 통과(Line Crossing), 퇴장(Exit) 등 비즈니스 이벤트 시점에만 선택적으로 전송하면 서버 수신 부하를 크게 줄일 수 있다.
* `AnalyticsEngine`의 `counting_line_y` 로직을 확장해 이벤트 상태를 관리하고, 해당 시점에만 `http_sender`를 호출하도록 수정한다.

**2-8. 다채널 병렬 처리 완성 (MultiStreamPipelineRunner)**
* 현재 `MultiStreamPipelineRunner` 클래스는 설계는 돼 있으나 실제 RTSP 환경 테스트가 미완이다.
* 채널당 디코딩 스레드가 독립적으로 동작하고 단일 GPU 추론 인스턴스를 공유하는 구조를 검증한다.
* 채널 간 프레임 타이밍 충돌로 인한 GPU 락 경합 여부와 OOM 발생 여부를 2-1 테스트 환경에서 집중 검증한다.

**2-9. SQLite 오프라인 버퍼 + Batch Flush 데몬**
* 네트워크 단절 시 추출된 벡터·스냅샷 데이터를 로컬 SQLite 큐에 임시 저장하고, 재연결 후 서버 벌크 API(`POST /v1/events/bulk`)로 병렬 전송하는 백그라운드 데몬을 구현한다. 전송 성공한 레코드만 SQLite에서 삭제(트랜잭션 보장)하고, 재전송 시 타임스탬프 순서를 보존해 서버 통계 정합성을 유지한다.

**2-10. 네트워크 장애 모의 E2E 테스트**
* `tc netem`으로 에지-서버 간 네트워크를 인위적으로 차단(drop 100%)한 뒤 일정 시간 후 복구하고, SQLite 적재 → 복구 감지 → Batch Flush 순서가 데이터 유실 없이 동작하는지 검증한다. 장애 전후 서버 DB 레코드 수와 에지 SQLite 잔여 레코드 수를 비교해 정합성 100%를 확인한다.

**2-11. 보행자 동선 타임라인 위젯**
* 특정 Global ID(또는 이미지 검색 결과)에 대해 "CAM_01 → 09:12 → CAM_03 → 09:15 → CAM_02 → 09:18" 형태의 시간순 동선을 시각화하는 타임라인 컴포넌트를 구현한다. `GET /v1/events?global_id=X` API 응답을 파싱해 카메라별 스냅샷과 타임스탬프를 가로 타임라인으로 렌더링하며, 동선 타임라인은 1-10의 이미지 검색 결과와 연동된다.

**2-12. 카메라 교차 매칭 임계값 튜닝**
* 단일 카메라 내 Re-ID와 달리 카메라 간 교차 매칭은 조명·각도 차이로 인해 유사도가 낮게 나온다. 서버의 `matcher.py`에서 사용하는 크로스 카메라 코사인 유사도 임계값을 단일 카메라 임계값(0.73)보다 낮게 조정하거나, 카메라 쌍별로 별도 임계값을 `.env`에서 관리한다. 2-6의 MOTA/IDF1 측정과 함께 교차 매칭 성능을 정량적으로 평가한다.

---

#### Phase 3

**3-1. YOLOv8n TensorRT FP16 엔진 변환 및 FPS 측정**
* Jetson에서 `yolov8n.pt`를 TensorRT FP16 `.engine` 파일로 변환한다. 변환은 최초 실행 시 자동으로 이루어지지만, 변환 시간(수 분)을 배포 전 미리 완료해두는 것이 권장된다. `.engine` 파일은 빌드한 Jetson 보드에 종속되므로 다른 장비로 복사 불가하다. 변환 후 단일 채널 FPS를 측정해 15 FPS 이상 달성 여부를 확인한다.

**3-2. OSNet ONNX Runtime CUDA 가속 검증**
* Jetson에 `onnxruntime-gpu` 패키지를 설치하고 `CUDAExecutionProvider`로 OSNet 추론 속도를 측정한다. CPU 대비 4~6배 속도 향상이 예상되며, 이를 달성하지 못하면 Re-ID 추출이 전체 파이프라인의 병목이 된다. ONNX 파일은 에지와 서버가 동일한 `osnet_x0_25.onnx`를 참조해야 임베딩 벡터 분포가 일치한다.

**3-3. 3채널 동시 실행 FPS 벤치마크**
* 3-1, 3-2 완료 후 `MultiStreamPipelineRunner`로 3채널을 동시 실행하며 합산 FPS를 측정한다. 목표는 45 FPS 이상(채널당 15 FPS)이며, GPU 공유 구조로 인해 단일 채널 FPS의 3배에 미치지 못할 수 있다. 병목이 GPU 추론인지 디코딩인지를 `jtop`으로 프로파일링해 원인을 파악한다.

**3-4. 장시간 안정성 테스트 (jtop 모니터링)**
* 3채널 동시 실행 상태에서 최소 1~2시간 연속 가동하며 GPU/CPU 사용률, VRAM 점유량, 코어 온도를 `jtop`으로 기록한다. OOM(Out of Memory) 발생, 온도 임계값 초과로 인한 스로틀링, 메모리 누수에 의한 점진적 성능 저하를 점검한다. Jetson Orin Nano의 TDP는 7~15W이며 팬리스 케이스 사용 시 과열 리스크가 있다.

**3-5. 실 IP 카메라 RTSP 스트림 연결 및 화질 확인**
* 실제 IP 카메라를 Jetson과 동일 LAN에 연결하고 RTSP URL로 스트림을 수신한다. 카메라 비트레이트(최소 2Mbps 권장), 해상도(1080p), 프레임레이트(15~20 FPS), 코덱(H.264 권장) 설정을 확인하고 최적화한다.

**3-6. 실환경 RTSP 기반 Rank-1 재측정**
* Phase 2 파인튜닝 모델을 탑재한 상태에서 실 IP 카메라 스트림으로 Rank-1 정확도를 측정한다. 로컬 파일 기반 수치(Phase 2 목표 80%)보다 5~13%p 낮을 것으로 예상되므로, 실환경 기준 목표치를 별도로 설정하는 것이 필요하다. 결과에 따라 임계값 재조정 또는 추가 파인튜닝 여부를 결정한다.

**3-7. Docker 컨테이너 빌드 (Jetson Native Build)**
* 베이스 이미지는 `nvcr.io/nvidia/l4t-pytorch` (ARM64 전용)이며, x86 호스트 PC에서 빌드한 이미지는 사용 불가하다. `docker compose up -d`로 Qdrant와 파이프라인을 함께 기동하며, 재부팅 후 자동 실행(`restart: always`) 설정을 포함한다.

**3-8. `manage_stream.sh` 연동 및 프로덕션 실행 검증**
* 통합 스트림 관리 스크립트 `manage_stream.sh`가 3채널 RTSP 수신, Qdrant 기동, 파이프라인 실행을 단일 명령으로 제어하는지 검증한다. 비정상 종료 후 자동 재시작 로직과 로그 로테이션 설정이 프로덕션 환경에 적합한지 점검한다.

**3-9. 에지 장비 모니터링 카드**
* Jetson의 CPU/GPU 사용률, VRAM 점유량, 코어 온도를 `jtop` Python API를 통해 주기적으로 수집하고 서버로 전송한다. 온도 임계값(예: 80°C 이상) 초과 시 경고 알림을 발생시키는 로직도 포함한다.

**3-10. E2E 전체 시스템 통합 테스트**
* 실 IP 카메라 → Jetson 에지 파이프라인 → FastAPI 서버 → PostgreSQL/Qdrant → 웹 대시보드로 이어지는 전체 경로를 PRD Definition of Done 체크리스트 기준으로 검증한다. 3채널 45 FPS 이상 유지, Re-ID Rank-1 ≥ 80%, 대시보드 500ms 이내 갱신, Docker 배포 가능, RTSP 자동 복구 동작을 순서대로 확인하고 결과를 기록한다.

---

## 9. 파이프라인 단계별 대안 옵션

> 현재 파이프라인 순서: **사람 인식 (YOLO) → Motion Tracking (ByteTrack) → Re-ID (OSNet)**
> 각 단계에서 선택 가능한 대안 옵션을 정리한다.

### 9.1 1단계: 사람 인식 (Person Detection) 대안

| 모델 | 속도 | 정확도 | Jetson TensorRT | 특징 |
|---|:---:|:---:|:---:|---|
| **YOLOv8n** (현재) | 빠름 | 중 | ✓ | 경량, TensorRT 잘 지원 |
| YOLOv8s/m | 중간 | 중상 | ✓ | n보다 정확, VRAM 추가 필요 |
| **YOLOv10n** | 빠름 | 중 | ✓ | NMS 제거로 추론 단순화 (PRD에도 언급) |
| **YOLO11n** | 빠름 | 중상 | ✓ | Ultralytics 최신, v8n 대비 정확도 개선 |
| RT-DETR | 느림 | 높음 | △ | 고정밀 Transformer 기반, Jetson 속도 부담 |
| NanoDet | 매우 빠름 | 중하 | ✓ | 초경량 엣지 전용 |
| **CrowdHuman 파인튜닝 YOLOv8n** | 빠름 | 높음 | ✓ | 밀집 군중 특화 데이터셋 가중치, 실내 보행자 탐지율 개선 |

**핵심 고려사항**
- 현재 YOLOv8n은 80개 클래스 범용 모델 → 사람(`class=0`)만 필터링해 사용 중
- **CrowdHuman** 기반 가중치 교체 시 구조 변경 없이 실내 탐지율 향상 가능
- Jetson에서 s/m 이상 모델은 TensorRT 없이 FPS 목표 달성 어려움

### 9.2 2단계: Motion Tracking 대안

BoxMOT 라이브러리가 이미 도입돼 있어 **`create_tracker()` 인자 변경만으로** 대부분 교체 가능.

| 트래커 | Re-ID 내장 | 가림 복원 | 속도 | 특징 |
|---|:---:|:---:|:---:|---|
| **ByteTrack** (현재) | ✗ | 중 | 매우 빠름 | 저신뢰도 탐지도 활용, 단순·안정적 |
| **OC-SORT** | ✗ | 상 | 빠름 | 가림 구간 관측값 기반 보정, ByteTrack보다 재매칭 강함 |
| **BotSORT** | ✓ (선택) | 상 | 빠름 | ByteTrack + 카메라 모션 보정 + Re-ID 재매칭 |
| **StrongSORT** | ✓ | 상 | 중간 | DeepSORT 개선판, Re-ID 통합 강화 |
| **Deep OC-SORT** | ✓ | 매우 상 | 중간 | OC-SORT + Re-ID 결합, 현재 SOTA급 |
| DeepSORT | ✓ | 중 | 중간 | 고전적 방법, StrongSORT에 밀림 |
| SORT | ✗ | 하 | 매우 빠름 | 가장 단순, 가림에 취약 |

### 9.3 트래커 선택에 따른 파이프라인 구조 변화

트래커 종류에 따라 Re-ID와의 통합 방식이 달라진다.

| 구조 | 해당 트래커 | 특징 |
|---|---|---|
| **탐지 → 트래킹 → Re-ID 분리** (현재) | ByteTrack, OC-SORT, SORT | Re-ID를 완전 분리 제어, VRAM 효율적 |
| **탐지 → Re-ID 내장 트래킹** | BotSORT, StrongSORT, Deep OC-SORT | 트래커 내부 Re-ID + OSNet Re-ID 이중 실행 → Jetson VRAM 추가 소비 주의 |

> Re-ID 내장 트래커 사용 시 트래커 내부 Re-ID 모델을 OSNet 동일 가중치로 통일해야 임베딩 분포 일관성 유지 가능.

### 9.4 실용적 추천 변경 조합

| 목적 | 추천 변경 | 변경 비용 |
|---|---|:---:|
| 가림 대응 즉시 개선 | ByteTrack → **OC-SORT** | 설정값 변경만 |
| 재연결 후 ID 복원 강화 | ByteTrack → **BotSORT** (Re-ID 선택 사용) | 설정값 변경만 |
| 탐지율 개선 (밀집·실내) | YOLOv8n → **CrowdHuman 파인튜닝 YOLOv8n** | 가중치 파일 교체만 |
| 최신 경량 모델 테스트 | YOLOv8n → **YOLO11n** | 가중치 파일 교체만 |
| 최고 정확도 추적 (속도 감수) | ByteTrack → **Deep OC-SORT** | 설정값 변경 + VRAM 검증 |

---

## 10. OSNet 도메인 파인튜닝 실행 가이드

> 수동 프레임 캡처 불필요. 기존 파이프라인 도구로 80~90% 자동화 가능.

### 10.1 전체 흐름

```
동영상 파일 (보유 중)
    ↓ 자동
YOLO + ByteTrack → Track ID별 크롭 이미지 추출
    ↓ 자동 + 소량 수동 검증
영상 간 동일인 연결 (Person ID 확정)
    ↓ 자동
Market-1501 형식 디렉터리 구성
    ↓ 자동
Torchreid 파인튜닝 실행
```

### 10.2 1단계: 크롭 이미지 자동 추출

`results/*.pkl` 파일에 이미 Track ID별 크롭 썸네일(`thumbnail` 필드)이 저장돼 있음. 아래 스크립트로 즉시 파일로 추출 가능.

```python
import pickle, cv2, pathlib

for pkl_file in pathlib.Path("results").glob("*.pkl"):
    data = pickle.load(open(pkl_file, "rb"))
    video_name = pkl_file.stem

    for track_id, records in data["data_x025"].items():
        out_dir = pathlib.Path(f"reid_dataset/raw/{video_name}/track_{track_id:04d}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for rec in records:
            cv2.imwrite(
                str(out_dir / f"f{rec['frame']:06d}.jpg"),
                rec["thumbnail"]   # 이미 크롭된 인물 이미지
            )
```

### 10.3 2단계: Person ID 매핑 (영상 간 동일인 연결)

Track ID는 영상 내에서만 유효하므로 영상 간 동일인을 연결해야 함.

**자동 처리**: 노트북 섹션 10의 `track_pair_sim()` 함수로 영상 간 트랙 유사도를 계산해 자동 클러스터링
```
video_A / track_2  ↔  video_B / track_5  (유사도 0.898)  → 동일 Person ID 부여
```

**수동 검증 (최소화)**: 자동 매칭 결과를 썸네일 그리드로 확인 → 오류만 수정.
인물 50명 기준 **1~2시간** 내 완료 가능.

### 10.4 3단계: Market-1501 형식으로 정리

Torchreid가 요구하는 디렉터리 구조.

```
reid_dataset/
  bounding_box_train/
    0001_c1_f000001.jpg    # 인물ID_카메라ID_프레임번호
    0001_c1_f000050.jpg
    0001_c2_f000120.jpg    ← 같은 사람, 다른 카메라/시점
    0002_c1_f000001.jpg
    ...
  query/
  bounding_box_test/
```

2단계에서 확정된 Person ID를 파일명 앞에 붙이는 rename 스크립트로 자동화 가능.

### 10.5 4단계: Torchreid 파인튜닝 실행

```python
import torchreid

# 데이터 로더 (Re-ID 전용 증강 포함)
datamanager = torchreid.data.ImageDataManager(
    root="reid_dataset",
    sources="market1501",
    targets="market1501",
    height=256, width=128,
    batch_size=32,
    transforms=["random_flip", "random_crop", "random_erasing", "color_jitter"]
)

# 사전학습 가중치에서 시작 (MSMT17 기반)
model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=datamanager.num_train_pids,
    pretrained=True
)

# ID Loss + Triplet Loss 조합 권장
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model,
    lr=0.0003,          # 파인튜닝은 낮은 lr 사용
    weight_decay=5e-4,
)

engine.run(
    max_epoch=60,
    save_dir="finetuned_osnet",
    eval_freq=10,
)
```

### 10.6 데이터 요건 및 기대 효과

| 수집 인물 수 | 인물당 이미지 수 | 조건 | 예상 Rank-1 향상 |
|:---:|:---:|---|---|
| 10~20명 | 6장 이상 | 단일 카메라 | +5~10%p |
| 30~50명 | 10장 이상 | 다각도 포함 | +10~15%p |
| **50명 이상** | **카메라 2개 이상 뷰** | 다양한 조명·의상 | **80% 목표 달성 가능** |

> 현재 보유한 동영상 4개(pkl 파일 기준)의 추출 가능 인물·크롭 수를 먼저 확인하는 것이 시작점.
> 인물 수가 부족하면 실제 카메라 설치 환경에서 추가 촬영 필요 (Action Item 2-3).

### 10.7 주의사항

- **모델 가중치 동기화**: 파인튜닝 완료 후 에지(edge)와 서버(server) 양측의 OSNet 가중치를 동일 파일로 교체해야 임베딩 분포가 일치함 (Action Item Common - OSNet 가중치 일치, Done 참조)
- **ONNX 재변환 필요**: 파인튜닝 후 `.pt` → `.onnx` 재변환 필요. `ReIDExtractor`의 Auto-Export 로직이 자동 처리하지만, 기존 캐시된 `.onnx` 파일을 삭제한 후 재실행해야 함
- **임계값 재조정**: 파인튜닝 후 임베딩 분포가 달라지므로 노트북으로 최적 임계값을 재계산해야 함 (Action Item 1-1 재수행)
