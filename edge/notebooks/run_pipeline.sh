#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh  —  Stage 1 (트랙 수집) → Stage 2 (ReID 추출) 2단계 실행
#
# 사용법 (로컬):
#   bash run_pipeline.sh /path/to/videos/*.avi
#
# 사용법 (Colab):
#   !COLAB=1 DRIVE_ROOT=/content/drive/MyDrive/projects/EYE-D/EYE-D \
#       IMAGE_DIR=dataset bash run_pipeline.sh .../data/*.avi
#
# 옵션 환경변수:
#   TRACKS_DIR   Stage 1 결과 폴더  (기본값: tracks  /  Colab: <DRIVE_ROOT>/tracks)
#   OUTPUT_DIR   Stage 2 결과 폴더  (기본값: results /  Colab: <DRIVE_ROOT>/results)
#   IMAGE_DIR    학습용 이미지 폴더  (기본값: 비어있음 = 저장 안 함)
#   FRAME_STEP   Stage 2 샘플링 간격 (기본값: 5)
#   MAX_FRAMES   최대 처리 프레임    (기본값: inf)
#   THRESHOLD    Re-ID 임계값        (기본값: 0.85)
#   CLEAN        1 = 기존 결과 삭제 후 재실행
#   COLAB        1 = Colab 모드
#   DRIVE_ROOT   Colab Drive 루트
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

COLAB="${COLAB:-0}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/projects/EYE-D/EYE-D}"

if [ "$COLAB" = "1" ]; then
    TRACKS_DIR="${TRACKS_DIR:-${DRIVE_ROOT}/tracks}"
    OUTPUT_DIR="${OUTPUT_DIR:-${DRIVE_ROOT}/results}"
else
    TRACKS_DIR="${TRACKS_DIR:-tracks}"
    OUTPUT_DIR="${OUTPUT_DIR:-results}"
fi

IMAGE_DIR="${IMAGE_DIR:-}"
if [ "$COLAB" = "1" ] && [ -n "${IMAGE_DIR:-}" ] && [[ "$IMAGE_DIR" != /* ]]; then
    IMAGE_DIR="${DRIVE_ROOT}/${IMAGE_DIR}"
fi

FRAME_STEP="${FRAME_STEP:-5}"
MAX_FRAMES="${MAX_FRAMES:-inf}"
CLEAN="${CLEAN:-0}"

_ENV_FILE="$(dirname "$0")/../../server/.env"
if [ -z "${THRESHOLD:-}" ] && [ -f "$_ENV_FILE" ]; then
    _ENV_VAL=$(grep -E '^REID_SIMILARITY_THRESHOLD=' "$_ENV_FILE" | tail -1 | cut -d'=' -f2 | tr -d '[:space:]')
    THRESHOLD="${_ENV_VAL:-0.85}"
else
    THRESHOLD="${THRESHOLD:-0.85}"
fi

_REPO_DIR="$(dirname "$0")/../.."
GIT_COMMIT=$(git -C "$_REPO_DIR" rev-parse HEAD             2>/dev/null || echo "unknown")
GIT_BRANCH=$(git -C "$_REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

NB_S1="$(dirname "$0")/collect_tracks.ipynb"
NB_S2="$(dirname "$0")/reid_performance.ipynb"

# Colab: papermill은 로컬 임시 폴더에 쓰고, 성공 시 Drive로 복사
if [ "$COLAB" = "1" ]; then
    NB_OUT_DIR="${DRIVE_ROOT}/edge/notebooks/outputs"
    NB_OUT_TMP="/content/nb_outputs"
else
    NB_OUT_DIR="$(dirname "$0")/outputs"
    NB_OUT_TMP="$NB_OUT_DIR"
fi

mkdir -p "$NB_OUT_TMP" "$NB_OUT_DIR" "$TRACKS_DIR" "$OUTPUT_DIR"

if [ $# -eq 0 ]; then
    echo "사용법: bash run_pipeline.sh <video1> [video2] ..."
    exit 1
fi

if ! command -v papermill &>/dev/null; then
    echo "[오류] papermill 이 설치되어 있지 않습니다. pip install papermill"
    exit 1
fi

BATCH_START=$(date +%s)
echo "═══════════════════════════════════════════════════"
echo "  Re-ID 파이프라인 시작  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  영상 수    : $#"
echo "  tracks 폴더: $TRACKS_DIR"
echo "  결과 폴더  : $OUTPUT_DIR"
echo "  이미지 폴더: $([ -n "$IMAGE_DIR" ] && echo "$IMAGE_DIR" || echo "저장 안 함")"
echo "  frame_step : $FRAME_STEP"
echo "  최대 프레임: $MAX_FRAMES"
echo "  임계값     : $THRESHOLD"
echo "  클린 모드  : $([ "$CLEAN" != "0" ] && echo "ON" || echo "OFF")"
echo "  git commit : ${GIT_COMMIT:0:12}..."
echo "═══════════════════════════════════════════════════"

_fmt_elapsed() { printf '%02d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

# tracks.pkl이 완전히 처리됐는지 확인
_tracks_complete() {
    local pkl="$1" video="$2"
    python3 -c "
import pickle, cv2, sys
try:
    d = pickle.load(open('$pkl','rb'))
    done = d.get('frames_processed', 0)
    total = d.get('total_frames') or int(cv2.VideoCapture('$video').get(cv2.CAP_PROP_FRAME_COUNT))
    sys.exit(0 if done >= total else 1)
except Exception:
    sys.exit(1)
"
}

_pkl_complete() {
    local pkl="$1" video="$2"
    python3 -c "
import pickle, cv2, sys
try:
    d = pickle.load(open('$pkl','rb'))
    done = d.get('frames_processed')
    if done is None:
        sys.exit(0)
    total = int(cv2.VideoCapture('$video').get(cv2.CAP_PROP_FRAME_COUNT))
    sys.exit(0 if done >= total else 1)
except Exception:
    sys.exit(1)
"
}

run_one() {
    local video="$1"
    local name; name=$(basename "${video%.*}")
    local tracks_pkl="$TRACKS_DIR/${name}.pkl"
    local result_pkl="$OUTPUT_DIR/${name}.pkl"
    local t0; t0=$(date +%s)

    echo ""
    echo "▶ $video  [$(date '+%H:%M:%S')]"

    if [ ! -e "$video" ]; then
        echo "  ✗ 파일 없음 — 건너뜁니다."
        return 0
    fi

    if [ "$CLEAN" != "0" ]; then
        [ -f "$tracks_pkl" ] && rm "$tracks_pkl" && echo "  🗑 삭제: $tracks_pkl"
        [ -f "$result_pkl" ] && rm "$result_pkl" && echo "  🗑 삭제: $result_pkl"
    fi

    mkdir -p "$NB_OUT_TMP" "$NB_OUT_DIR" "$TRACKS_DIR" "$OUTPUT_DIR"

    # ── Stage 1: 트랙 수집 ────────────────────────────────────────────────────
    if [ -f "$tracks_pkl" ] && _tracks_complete "$tracks_pkl" "$video"; then
        echo "  ⏭ Stage 1 완료 (건너뜀)"
    else
        local t_s1; t_s1=$(date +%s)
        echo "  [Stage 1] 트랙 수집 중...  [$(date '+%H:%M:%S')]"
        local colab_env=""
        [ "$COLAB" = "1" ] && colab_env="EYE_D_COLAB=1 EYE_D_DRIVE_ROOT=${DRIVE_ROOT}"
        if env $colab_env papermill "$NB_S1" "$NB_OUT_TMP/${name}_s1.ipynb" \
            -p VIDEO_PATH  "$video"      \
            -p TRACKS_DIR  "$TRACKS_DIR" \
            -p MAX_FRAMES  "$MAX_FRAMES" \
            -p FRAME_STEP  "$FRAME_STEP" \
            -p GIT_COMMIT  "$GIT_COMMIT" \
            -p GIT_BRANCH  "$GIT_BRANCH" \
            --log-output 2>&1 | grep -E "(완료|오류|Error|체크포인트|재개|트랙|프레임|Executing)"; then
            [ "$COLAB" = "1" ] && cp "$NB_OUT_TMP/${name}_s1.ipynb" "$NB_OUT_DIR/"
            echo "  ✓ Stage 1 완료 → $tracks_pkl  ($(_fmt_elapsed $(( $(date +%s) - t_s1 ))))"
        else
            echo "  ✗ Stage 1 실패  ($(_fmt_elapsed $(( $(date +%s) - t_s1 ))))"
            return 1
        fi
    fi

    # ── Stage 2: ReID 추출 ────────────────────────────────────────────────────
    if [ -f "$result_pkl" ] && _pkl_complete "$result_pkl" "$video"; then
        echo "  ⏭ Stage 2 완료 (건너뜀)"
    else
        local t_s2; t_s2=$(date +%s)
        echo "  [Stage 2] ReID 추출 중... (frame_step=${FRAME_STEP})  [$(date '+%H:%M:%S')]"
        local colab_env=""
        [ "$COLAB" = "1" ] && colab_env="EYE_D_COLAB=1 EYE_D_DRIVE_ROOT=${DRIVE_ROOT}"
        if env $colab_env papermill "$NB_S2" "$NB_OUT_TMP/${name}_s2.ipynb" \
            -p VIDEO_PATH        "$video"       \
            -p TRACKS_DIR        "$TRACKS_DIR"  \
            -p OUTPUT_DIR        "$OUTPUT_DIR"  \
            -p IMAGE_DIR         "$IMAGE_DIR"   \
            -p FRAME_STEP        "$FRAME_STEP"  \
            -p MAX_FRAMES        "$MAX_FRAMES"  \
            -p CURRENT_THRESHOLD "$THRESHOLD"   \
            -p GIT_COMMIT        "$GIT_COMMIT"  \
            -p GIT_BRANCH        "$GIT_BRANCH"  \
            --execution-timeout -1              \
            --log-output 2>&1 | grep -E "(완료|오류|Error|체크포인트|재개|트랙|임베딩|프레임|Executing)"; then
            [ "$COLAB" = "1" ] && cp "$NB_OUT_TMP/${name}_s2.ipynb" "$NB_OUT_DIR/"
            echo "  ✓ Stage 2 완료 → $result_pkl  ($(_fmt_elapsed $(( $(date +%s) - t_s2 ))))"
        else
            echo "  ✗ Stage 2 실패  ($(_fmt_elapsed $(( $(date +%s) - t_s2 ))))"
            return 1
        fi
    fi
}

export -f run_one _fmt_elapsed
export NB_S1 NB_S2 NB_OUT_DIR NB_OUT_TMP TRACKS_DIR OUTPUT_DIR IMAGE_DIR FRAME_STEP MAX_FRAMES THRESHOLD CLEAN COLAB DRIVE_ROOT GIT_COMMIT GIT_BRANCH

for video in "$@"; do run_one "$video"; done

TOTAL=$(( $(date +%s) - BATCH_START ))
echo ""
echo "═══════════════════════════════════════════════════"
echo "  파이프라인 완료  [$(date '+%Y-%m-%d %H:%M:%S')]  총 소요: $(_fmt_elapsed $TOTAL)"
echo "  tracks    : $TRACKS_DIR/"
echo "  결과 pkl  : $OUTPUT_DIR/"
[ -n "$IMAGE_DIR" ] && echo "  이미지    : $IMAGE_DIR/"
echo "  출력 노트북: $NB_OUT_DIR/"
echo "  다음 단계 : reid_aggregate.ipynb 실행"
echo "═══════════════════════════════════════════════════"
