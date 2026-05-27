#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_batch.sh  —  여러 영상에 대해 reid_performance.ipynb 를 자동 실행합니다.
#
# 사용법 (로컬):
#   bash run_batch.sh /path/to/videos/*.avi
#   bash run_batch.sh /data/cam1.avi /data/cam2.avi /data/cam3.avi
#
# 사용법 (Google Colab 셀):
#   !COLAB=1 bash /content/drive/MyDrive/EYE-D/edge/notebooks/run_batch.sh \
#       /content/drive/MyDrive/EYE-D/data/16000003.avi
#   # DRIVE_ROOT 기본값: /content/drive/MyDrive/EYE-D  (다를 경우 명시)
#   !COLAB=1 DRIVE_ROOT=/content/drive/MyDrive/MyProject bash run_batch.sh ...

#   #COLAB=1 CLEAN=1 MAX_FRAMES=3000 DRIVE_ROOT=/content/drive/MyDrive/projects/EYE-D/EYE-D bash ./run_batch.sh /content/drive/MyDrive/projects/EYE-D/EYE-D/data/16000000.avi
#   #COLAB=0 CLEAN=1 MAX_FRAMES=100 bash run_batch.sh ../../data/16000000.avi
# 
# 옵션 환경변수:
#   OUTPUT_DIR   결과 pkl 저장 폴더  (기본값: results  /  Colab: <DRIVE_ROOT>/results)
#   MAX_FRAMES   분석할 최대 프레임   (기본값: inf, 전체 프레임)
#   THRESHOLD    Re-ID 판별 임계값    (기본값: 0.85)
#   CLEAN        1 로 설정 시 기존 pkl·출력 노트북 삭제 후 재실행 (기본값: 0)
#   PARALLEL     동시 실행 수         (기본값: 1, 순차 실행)
#   COLAB        1 로 설정 시 Colab 모드 활성화 (기본값: 0)
#   DRIVE_ROOT   Colab 모드에서 Drive 내 프로젝트 루트 (기본값: /content/drive/MyDrive/EYE-D)
#
# 예시:
#   OUTPUT_DIR=results MAX_FRAMES=2000 PARALLEL=2 bash run_batch.sh /data/*.avi
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

COLAB="${COLAB:-0}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/projects/EYE-D/EYE-D}"

if [ "$COLAB" = "1" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-${DRIVE_ROOT}/results}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-results}"
fi
MAX_FRAMES="${MAX_FRAMES:-inf}"

# .env 에서 REID_SIMILARITY_THRESHOLD 읽기 (THRESHOLD 환경변수가 없을 때만)
_ENV_FILE="$(dirname "$0")/../../server/.env"
if [ -z "${THRESHOLD:-}" ] && [ -f "$_ENV_FILE" ]; then
    _ENV_VAL=$(grep -E '^REID_SIMILARITY_THRESHOLD=' "$_ENV_FILE" | tail -1 | cut -d'=' -f2 | tr -d '[:space:]')
    THRESHOLD="${_ENV_VAL:-0.85}"
else
    THRESHOLD="${THRESHOLD:-0.85}"
fi

CLEAN="${CLEAN:-0}"   # 0 이외의 값이면 모두 클린 모드로 동작 (예: CLEAN=1, CLEAN=yes)
PARALLEL="${PARALLEL:-1}"

# Git 버전 정보 수집 (노트북 출력·pkl 에 기록돼 나중에 소스 버전을 추적할 수 있음)
_REPO_DIR="$(dirname "$0")/../.."
GIT_COMMIT=$(git -C "$_REPO_DIR" rev-parse HEAD             2>/dev/null || echo "unknown")
GIT_BRANCH=$(git -C "$_REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

NB_IN="$(dirname "$0")/reid_performance.ipynb"
NB_OUT_DIR="$(dirname "$0")/outputs"

mkdir -p "$NB_OUT_DIR" "$OUTPUT_DIR"

if [ $# -eq 0 ]; then
    echo "사용법: bash run_batch.sh <video1> [video2] ..."
    exit 1
fi

if ! command -v papermill &>/dev/null; then
    echo "[오류] papermill 이 설치되어 있지 않습니다."
    echo "  pip install papermill"
    exit 1
fi

BATCH_START=$(date +%s)

echo "═══════════════════════════════════════════════════"
echo "  Re-ID 배치 분석 시작  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  영상 수    : $#"
echo "  결과 폴더  : $OUTPUT_DIR"
echo "  최대 프레임: $MAX_FRAMES"
echo "  임계값     : $THRESHOLD"
echo "  클린 모드  : $([ "$CLEAN" != "0" ] && echo "ON (기존 pkl·노트북 삭제)" || echo "OFF")"
echo "  동시 실행  : $PARALLEL"
echo "  git branch : $GIT_BRANCH"
echo "  git commit : ${GIT_COMMIT:0:12}..."
if [ "$COLAB" = "1" ]; then
echo "  실행 환경  : Google Colab (Drive: $DRIVE_ROOT)"
fi
echo "═══════════════════════════════════════════════════"

_fmt_elapsed() {
    local secs=$1
    printf '%02d:%02d:%02d' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

run_one() {
    local video="$1"
    local name
    name=$(basename "$video")
    name="${name%.*}"
    local nb_out="$NB_OUT_DIR/${name}.ipynb"
    local t0
    t0=$(date +%s)

    echo ""
    echo "▶ 처리 중: $video  [$(date '+%H:%M:%S')]"

    if [ ! -e "$video" ]; then
        echo "  ✗ 파일 없음: $video — 건너뜁니다."
        return 0
    fi

    if [ "$CLEAN" != "0" ]; then
        [ -f "$nb_out" ]                      && rm "$nb_out"                      && echo "  🗑 삭제: $nb_out"
        [ -f "$OUTPUT_DIR/${name}.pkl" ]       && rm "$OUTPUT_DIR/${name}.pkl"       && echo "  🗑 삭제: $OUTPUT_DIR/${name}.pkl"
    fi

    local colab_env=""
    if [ "$COLAB" = "1" ]; then
        colab_env="EYE_D_COLAB=1 EYE_D_DRIVE_ROOT=${DRIVE_ROOT}"
    fi

    if env $colab_env papermill "$NB_IN" "$nb_out" \
        -p VIDEO_PATH         "$video"      \
        -p OUTPUT_DIR         "$OUTPUT_DIR" \
        -p MAX_FRAMES         "$MAX_FRAMES" \
        -p CURRENT_THRESHOLD  "$THRESHOLD"  \
        -p GIT_COMMIT         "$GIT_COMMIT" \
        -p GIT_BRANCH         "$GIT_BRANCH" \
        --log-output 2>&1 | grep -E "(완료|오류|Error|WARNING|완전|Executing|비디오|전체|처리 예정|캐시)"; then
        local elapsed=$(( $(date +%s) - t0 ))
        echo "  ✓ 완료 → $nb_out  ($(_fmt_elapsed $elapsed))"
        echo "  ✓ pkl  → $OUTPUT_DIR/${name}.pkl"
    else
        local elapsed=$(( $(date +%s) - t0 ))
        echo "  ✗ 실패: $video  ($(_fmt_elapsed $elapsed))"
    fi
}

export -f run_one _fmt_elapsed
export NB_IN NB_OUT_DIR OUTPUT_DIR MAX_FRAMES THRESHOLD CLEAN COLAB DRIVE_ROOT BATCH_START GIT_COMMIT GIT_BRANCH

if [ "$PARALLEL" -gt 1 ]; then
    # GNU parallel 사용 (설치된 경우)
    if command -v parallel &>/dev/null; then
        printf '%s\n' "$@" | parallel -j "$PARALLEL" run_one {}
    else
        echo "[경고] GNU parallel 없음. 순차 실행합니다."
        for video in "$@"; do run_one "$video"; done
    fi
else
    for video in "$@"; do run_one "$video"; done
fi

TOTAL_ELAPSED=$(( $(date +%s) - BATCH_START ))
echo ""
echo "═══════════════════════════════════════════════════"
echo "  배치 실행 완료  [$(date '+%Y-%m-%d %H:%M:%S')]  총 소요: $(_fmt_elapsed $TOTAL_ELAPSED)"
echo "  pkl 결과 : $OUTPUT_DIR/"
echo "  출력 노트북: $NB_OUT_DIR/"
echo ""
echo "  다음 단계: reid_aggregate.ipynb 실행"
echo "  jupyter notebook notebooks/reid_aggregate.ipynb"
echo "═══════════════════════════════════════════════════"
