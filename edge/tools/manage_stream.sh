#!/bin/bash

# ==============================================================================
# EYE-D RTSP 스트림 및 미디어 서버 통합 관리 스크립트 (manage_stream.sh)
#
# 역할: RTSP 미디어 서버(MediaMTX) 컨테이너 및 다채널 비디오 스트림(FFmpeg)의
#       구동, 중단, 상태 모니터링, 실시간 재생(ffplay)을 통합하여 관리합니다.
# ==============================================================================

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EDGE_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

# 기본 설정값
CONTAINER_NAME="mediamtx"
RTSP_SERVER="rtsp://localhost:8554"
IP_ADDR="localhost"

FILE1="$DATA_DIR/14300002.avi"
FILE2="$DATA_DIR/16300002.avi"
FILE3="$DATA_DIR/16300003.avi"

# 기본 실행 옵션 플래그
SERVER_ONLY=false
CLIENT_ONLY=false
PREVIEW_ONLY=false
FOREGROUND=false
FOLLOW_LOGS=false

# 도움말 출력 함수
show_help() {
    echo -e "${BLUE}EYE-D RTSP 스트림 및 미디어 서버 통합 관리 도구${NC}"
    echo "사용법: $0 [Command] [Options]"
    echo ""
    echo "명령어(Command):"
    echo "  start       RTSP 서버(MediaMTX) 및 FFmpeg 스트림 송출을 시작합니다."
    echo "  stop        RTSP 서버, FFmpeg 스트림 및 ffplay 모니터링을 중단합니다."
    echo "  restart     RTSP 서버 및 스트림을 재시작합니다 (stop 후 start)."
    echo "  status      RTSP 서버, FFmpeg 및 ffplay 프로세스의 구동 상태를 확인합니다."
    echo "  view        ffplay를 사용하여 cam01, cam02, cam03 스트림을 실시간 모니터링(재생)합니다."
    echo "  logs        MediaMTX 컨테이너의 로그를 출력합니다."
    echo ""
    echo "옵션(Options) [start / restart 시 사용 가능]:"
    echo "  -s, --server-only          MediaMTX 서버만 시작합니다."
    echo "  -c, --client-only          FFmpeg 스트림 송출만 시작합니다. (MediaMTX 서버 제외)"
    echo "  -f, --foreground           MediaMTX 서버를 포어그라운드로 실행합니다. (-s 와 함께 권장)"
    echo "  --file1 <path>             CAM 1 비디오 파일 경로 지정 (기본값: $FILE1)"
    echo "  --file2 <path>             CAM 2 비디오 파일 경로 지정 (기본값: $FILE2)"
    echo "  --file3 <path>             CAM 3 비디오 파일 경로 지정 (기본값: $FILE3)"
    echo "  --rtsp-server <url>        RTSP 대상 서버 주소 지정 (기본값: $RTSP_SERVER)"
    echo ""
    echo "옵션(Options) [stop 시 사용 가능]:"
    echo "  -s, --server-only          MediaMTX 서버만 중지합니다."
    echo "  -c, --client-only          FFmpeg 스트림 송출만 중지합니다."
    echo "  -p, --preview-only         ffplay 모니터링 화면만 닫습니다."
    echo ""
    echo "옵션(Options) [view 시 사용 가능]:"
    echo "  -i, --ip <ip>              대상 RTSP 서버 IP 주소 지정 (기본값: localhost)"
    echo ""
    echo "옵션(Options) [logs 시 사용 가능]:"
    echo "  -f, --follow               실시간 로그 흐름(tail -f)을 모니터링합니다."
    echo ""
    echo "예시(Examples):"
    echo "  * 전체 통합 실행:       $0 start"
    echo "  * 미디어 서버만 실행:   $0 start -s"
    echo "  * 스트림 송출만 실행:   $0 start -c"
    echo "  * 실시간 스트림 보기:   $0 view"
    echo "  * 타겟 IP 스트림 보기:   $0 view -i 192.168.45.7"
    echo "  * 전체 서비스 중단:     $0 stop"
    echo "  * 로그 실시간 모니터링: $0 logs -f"
}

# Docker 실행 환경 및 권한 체크
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[ERROR] Docker가 설치되어 있지 않거나 PATH에 존재하지 않습니다.${NC}"
        exit 1
    fi

    if ! docker ps &> /dev/null; then
        echo -e "${RED}[ERROR] Docker 데몬이 실행 중이 아니거나 권한이 없습니다.${NC}"
        echo -e "${YELLOW}[TIP] 'sudo systemctl start docker' 또는 현재 사용자를 docker 그룹에 추가하십시오: 'sudo usermod -aG docker \$USER'${NC}"
        exit 1
    fi
}

# MediaMTX 서버 시작
start_server() {
    check_docker
    echo "========================================================"
    echo -e "   ${BLUE}RTSP 미디어 서버 (MediaMTX) 상태 점검 및 구동${NC}"
    echo "========================================================"

    # 이미 구동 중인지 확인
    if docker ps --filter "name=^/${CONTAINER_NAME}$" --filter "status=running" | grep -q "${CONTAINER_NAME}"; then
        echo -e "${GREEN}[INFO] MediaMTX 컨테이너가 이미 백그라운드에서 구동 중입니다.${NC}"
        echo -e "RTSP 서버 주소: $RTSP_SERVER"
        echo "========================================================"
        return 0
    fi

    # 중지 상태인 동일 이름의 컨테이너 정리
    if docker ps -a --filter "name=^/${CONTAINER_NAME}$" | grep -q "${CONTAINER_NAME}"; then
        echo -e "기존에 존재하던 중지 상태의 ${CONTAINER_NAME} 컨테이너를 정리합니다..."
        docker rm -f ${CONTAINER_NAME} > /dev/null 2>&1
    fi

    # 구동 방식에 따른 분기 실행
    if [ "$FOREGROUND" = true ]; then
        echo -e "${YELLOW}MediaMTX 컨테이너를 포어그라운드에서 시작합니다...${NC}"
        echo "콘솔 로그 출력을 시작합니다. 종료하려면 Ctrl+C를 누르십시오."
        echo "--------------------------------------------------------"
        docker run -it --rm --name ${CONTAINER_NAME} --network=host bluenviron/mediamtx:latest
    else
        echo -e "${YELLOW}MediaMTX 컨테이너를 백그라운드에서 시작합니다...${NC}"
        docker run -d --rm --name ${CONTAINER_NAME} --network=host bluenviron/mediamtx:latest > /dev/null

        # 기동 대기
        echo "네트워크 바인딩 및 서버 기동 대기 중 (2초)..."
        sleep 2

        # 최종 구동 상태 확인
        if docker ps --filter "name=^/${CONTAINER_NAME}$" --filter "status=running" | grep -q "${CONTAINER_NAME}"; then
            echo -e "${GREEN}[SUCCESS] MediaMTX 컨테이너가 백그라운드에서 성공적으로 구동되었습니다.${NC}"
            echo -e "RTSP 서버 주소: $RTSP_SERVER"
        else
            echo -e "${RED}[ERROR] MediaMTX 컨테이너 구동에 실패했습니다. 'docker logs ${CONTAINER_NAME}' 명령으로 원인을 확인하십시오.${NC}"
            exit 1
        fi
    fi
    echo "========================================================"
}

# FFmpeg 스트림 시작
start_streams() {
    # ffmpeg 설치 여부 확인
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "${RED}[ERROR] ffmpeg이 설치되어 있지 않거나 PATH에 존재하지 않습니다.${NC}"
        echo "Ubuntu 기준 설치법: sudo apt update && sudo apt install -y ffmpeg"
        exit 1
    fi

    echo "========================================================"
    echo -e "   ${BLUE}다채널 RTSP 스트림 무한 루프 송출을 시작합니다${NC}"
    echo "========================================================"

    # 중복 실행 방지: 이미 ffmpeg이 돌고 있으면 안내
    if pgrep ffmpeg > /dev/null; then
        echo -e "${YELLOW}[WARNING] 이미 실행 중인 FFmpeg 프로세스가 감지되었습니다.${NC}"
        echo -e "기존 스트림을 중단하려면 '$0 stop -c' 를 실행하고 다시 시작하십시오."
        echo "--------------------------------------------------------"
    fi

    # cam01 송출
    if [ -f "$FILE1" ]; then
        echo -e "[CAM 1] 송출 시작: $FILE1 -> $RTSP_SERVER/cam01"
        ffmpeg -re -stream_loop -1 -i "$FILE1" -c:v libx264 -preset ultrafast -an -f rtsp "$RTSP_SERVER/cam01" > /dev/null 2>&1 &
    else
        echo -e "${RED}[ERROR] CAM 1 비디오 파일을 찾을 수 없습니다: $FILE1${NC}"
    fi

    # cam02 송출
    if [ -f "$FILE2" ]; then
        echo -e "[CAM 2] 송출 시작: $FILE2 -> $RTSP_SERVER/cam02"
        ffmpeg -re -stream_loop -1 -i "$FILE2" -c:v libx264 -preset ultrafast -an -f rtsp "$RTSP_SERVER/cam02" > /dev/null 2>&1 &
    else
        echo -e "${RED}[ERROR] CAM 2 비디오 파일을 찾을 수 없습니다: $FILE2${NC}"
    fi

    # cam03 송출
    if [ -f "$FILE3" ]; then
        echo -e "[CAM 3] 송출 시작: $FILE3 -> $RTSP_SERVER/cam03"
        ffmpeg -re -stream_loop -1 -i "$FILE3" -c:v libx264 -preset ultrafast -an -f rtsp "$RTSP_SERVER/cam03" > /dev/null 2>&1 &
    else
        echo -e "${RED}[ERROR] CAM 3 비디오 파일을 찾을 수 없습니다: $FILE3${NC}"
    fi

    echo "--------------------------------------------------------"
    echo -e "${GREEN}[SUCCESS] 모든 스트림 송출이 백그라운드에서 실행되었습니다.${NC}"
    echo "스트림 송출을 중단하려면 다음 명령어를 실행하십시오:"
    echo "  $0 stop -c"
    echo "========================================================"
}

# MediaMTX 서버 중지
stop_server() {
    check_docker
    echo "========================================================"
    echo -e "   ${BLUE}RTSP 미디어 서버 (MediaMTX) 컨테이너 중단${NC}"
    echo "========================================================"
    if docker ps --filter "name=^/${CONTAINER_NAME}$" --filter "status=running" | grep -q "${CONTAINER_NAME}"; then
        echo "구동 중인 ${CONTAINER_NAME} Docker 컨테이너를 중지합니다..."
        docker stop ${CONTAINER_NAME} > /dev/null 2>&1
        echo -e "${GREEN}[SUCCESS] MediaMTX 서버 컨테이너가 안전하게 종료되었습니다.${NC}"
    else
        echo -e "[INFO] 실행 중인 MediaMTX 컨테이너가 없습니다."
    fi
    echo "========================================================"
}

# FFmpeg 스트림 중지
stop_streams() {
    echo "========================================================"
    echo -e "   ${BLUE}FFmpeg RTSP 스트림 송출 중단${NC}"
    echo "========================================================"
    if pgrep ffmpeg > /dev/null; then
        pkill ffmpeg
        echo -e "${GREEN}[SUCCESS] 모든 FFmpeg 스트리밍 프로세스가 중단되었습니다.${NC}"
    else
        echo -e "[INFO] 실행 중인 FFmpeg 프로세스가 없습니다."
    fi
    echo "========================================================"
}

# ffplay 모니터링 중지
stop_previews() {
    echo "========================================================"
    echo -e "   ${BLUE}ffplay 모니터링 화면 종료${NC}"
    echo "========================================================"
    if pgrep ffplay > /dev/null; then
        pkill ffplay
        echo -e "${GREEN}[SUCCESS] 모든 ffplay 모니터링 프로세스가 중단되었습니다.${NC}"
    else
        echo -e "[INFO] 실행 중인 ffplay 프로세스가 없습니다."
    fi
    echo "========================================================"
}

# ffplay 스트림 재생
start_previews() {
    # ffplay 설치 여부 확인
    if ! command -v ffplay &> /dev/null; then
        echo -e "${RED}[ERROR] ffplay가 설치되어 있지 않거나 PATH에 존재하지 않습니다.${NC}"
        echo "Ubuntu 기준 설치법: sudo apt install ffmpeg"
        exit 1
    fi

    # RTSP 서버 주소 재구성 (입력된 IP_ADDR 사용)
    local target_server="rtsp://$IP_ADDR:8554"

    echo "========================================================"
    echo -e "   ${BLUE}다채널 RTSP 스트림 모니터링 시작 (ffplay)${NC}"
    echo -e "   대상 서버: $target_server"
    echo "========================================================"

    echo "[CAM 01] 재생 시도..."
    ffplay -loglevel error -window_title "CAM 01 - $target_server/cam01" "$target_server/cam01" > /dev/null 2>&1 &

    echo "[CAM 02] 재생 시도..."
    ffplay -loglevel error -window_title "CAM 02 - $target_server/cam02" "$target_server/cam02" > /dev/null 2>&1 &

    echo "[CAM 03] 재생 시도..."
    ffplay -loglevel error -window_title "CAM 03 - $target_server/cam03" "$target_server/cam03" > /dev/null 2>&1 &

    echo "--------------------------------------------------------"
    echo -e "${GREEN}[SUCCESS] 모든 모니터링 창이 개별 백그라운드 스레드로 실행되었습니다.${NC}"
    echo "창을 모두 일괄 종료하려면 다음 명령을 실행하십시오:"
    echo "  $0 stop -p"
    echo "========================================================"
}

# 구동 상태 확인
show_status() {
    echo "========================================================"
    echo -e "   ${BLUE}EYE-D RTSP 스트림 및 미디어 서버 구동 상태${NC}"
    echo "========================================================"

    # 1. MediaMTX Docker 상태
    local running_server=""
    if command -v docker &> /dev/null; then
        running_server=$(docker ps -f "name=^/${CONTAINER_NAME}$" --format "{{.Status}}")
    fi

    if [ -n "$running_server" ]; then
        echo -e "1. MediaMTX 서버 ($CONTAINER_NAME): ${GREEN}RUNNING (${running_server})${NC}"
    else
        echo -e "1. MediaMTX 서버 ($CONTAINER_NAME): ${RED}STOPPED${NC}"
    fi

    # 2. FFmpeg 스트림 송출 상태
    echo ""
    echo "2. FFmpeg 스트리밍 프로세스 정보:"
    if pgrep ffmpeg > /dev/null; then
        echo -e "상태: ${GREEN}RUNNING${NC}"
        ps -eo pid,args | grep '[f]fmpeg' | while read -r line; do
            local pid=$(echo "$line" | awk '{print $1}')
            local args=$(echo "$line" | cut -d' ' -f2-)
            local stream_addr=$(echo "$args" | grep -oE "rtsp://[^ ]+")
            if [ -n "$stream_addr" ]; then
                echo -e "  [PID: $pid] -> $stream_addr"
            else
                echo -e "  [PID: $pid] $args"
            fi
        done
    else
        echo -e "상태: ${RED}STOPPED${NC}"
    fi

    # 3. ffplay 모니터링 상태
    echo ""
    echo "3. ffplay 모니터링 프로세스 정보:"
    if pgrep ffplay > /dev/null; then
        echo -e "상태: ${GREEN}RUNNING${NC}"
        ps -eo pid,args | grep '[f]fplay' | while read -r line; do
            local pid=$(echo "$line" | awk '{print $1}')
            local args=$(echo "$line" | cut -d' ' -f2-)
            local stream_addr=$(echo "$args" | grep -oE "rtsp://[^ ]+")
            if [ -n "$stream_addr" ]; then
                echo -e "  [PID: $pid] 모니터링 중: $stream_addr"
            else
                echo -e "  [PID: $pid] $args"
            fi
        done
    else
        echo -e "상태: ${RED}STOPPED (실행 중인 프리뷰 화면이 없습니다.)${NC}"
    fi
    echo "========================================================"
}

# MediaMTX 로그 확인
show_logs() {
    check_docker
    if ! docker ps -a --filter "name=^/${CONTAINER_NAME}$" | grep -q "${CONTAINER_NAME}"; then
        echo -e "${RED}[ERROR] $CONTAINER_NAME 컨테이너가 존재하지 않습니다.${NC}"
        exit 1
    fi

    echo "========================================================"
    echo -e "   ${BLUE}MediaMTX 서버 로그 출력 (${CONTAINER_NAME})${NC}"
    echo "========================================================"
    if [ "$FOLLOW_LOGS" = true ]; then
        docker logs -f "$CONTAINER_NAME"
    else
        docker logs --tail 100 "$CONTAINER_NAME"
    fi
}

# 파라미터 처리
ACTION="$1"
shift

case "$ACTION" in
    start)
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -s|--server-only)
                    SERVER_ONLY=true
                    shift
                    ;;
                -c|--client-only)
                    CLIENT_ONLY=true
                    shift
                    ;;
                -f|--foreground)
                    FOREGROUND=true
                    shift
                    ;;
                --file1)
                    FILE1="$2"
                    shift 2
                    ;;
                --file2)
                    FILE2="$2"
                    shift 2
                    ;;
                --file3)
                    FILE3="$2"
                    shift 2
                    ;;
                --rtsp-server)
                    RTSP_SERVER="$2"
                    shift 2
                    ;;
                -h|--help)
                    show_help
                    exit 0
                    ;;
                *)
                    echo -e "${RED}알 수 없는 옵션입니다: $1${NC}"
                    show_help
                    exit 1
                    ;;
            esac
        done

        # 상호 배타적인 옵션 체크
        if [ "$SERVER_ONLY" = true ] && [ "$CLIENT_ONLY" = true ]; then
            echo -e "${RED}[ERROR] -s (--server-only)와 -c (--client-only) 옵션은 동시에 사용할 수 없습니다.${NC}"
            exit 1
        fi

        # 실행 분기
        if [ "$SERVER_ONLY" = true ]; then
            start_server
        elif [ "$CLIENT_ONLY" = true ]; then
            start_streams
        else
            start_server
            if [ "$FOREGROUND" = false ]; then
                start_streams
            fi
        fi
        ;;

    stop)
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -s|--server-only)
                    SERVER_ONLY=true
                    shift
                    ;;
                -c|--client-only)
                    CLIENT_ONLY=true
                    shift
                    ;;
                -p|--preview-only)
                    PREVIEW_ONLY=true
                    shift
                    ;;
                -h|--help)
                    show_help
                    exit 0
                    ;;
                *)
                    echo -e "${RED}알 수 없는 옵션입니다: $1${NC}"
                    show_help
                    exit 1
                    ;;
            esac
        done

        # 기본 동작: 옵션이 없으면 전부 중단
        if [ "$SERVER_ONLY" = false ] && [ "$CLIENT_ONLY" = false ] && [ "$PREVIEW_ONLY" = false ]; then
            stop_previews
            stop_streams
            stop_server
        else
            if [ "$PREVIEW_ONLY" = true ]; then
                stop_previews
            fi
            if [ "$CLIENT_ONLY" = true ]; then
                stop_streams
            fi
            if [ "$SERVER_ONLY" = true ]; then
                stop_server
            fi
        fi
        ;;

    restart)
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -s|--server-only)
                    SERVER_ONLY=true
                    shift
                    ;;
                -c|--client-only)
                    CLIENT_ONLY=true
                    shift
                    ;;
                -f|--foreground)
                    FOREGROUND=true
                    shift
                    ;;
                --file1)
                    FILE1="$2"
                    shift 2
                    ;;
                --file2)
                    FILE2="$2"
                    shift 2
                    ;;
                --file3)
                    FILE3="$2"
                    shift 2
                    ;;
                --rtsp-server)
                    RTSP_SERVER="$2"
                    shift 2
                    ;;
                *)
                    echo -e "${RED}알 수 없는 옵션입니다: $1${NC}"
                    show_help
                    exit 1
                    ;;
            esac
        done

        if [ "$SERVER_ONLY" = true ] && [ "$CLIENT_ONLY" = true ]; then
            echo -e "${RED}[ERROR] -s (--server-only)와 -c (--client-only) 옵션은 동시에 사용할 수 없습니다.${NC}"
            exit 1
        fi

        if [ "$SERVER_ONLY" = true ]; then
            stop_server
            start_server
        elif [ "$CLIENT_ONLY" = true ]; then
            stop_streams
            start_streams
        else
            stop_previews
            stop_streams
            stop_server
            start_server
            if [ "$FOREGROUND" = false ]; then
                start_streams
            fi
        fi
        ;;

    status)
        show_status
        ;;

    view)
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -i|--ip)
                    IP_ADDR="$2"
                    shift 2
                    ;;
                -h|--help)
                    show_help
                    exit 0
                    ;;
                *)
                    echo -e "${RED}알 수 없는 옵션입니다: $1${NC}"
                    show_help
                    exit 1
                    ;;
            esac
        done
        start_previews
        ;;

    logs)
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -f|--follow)
                    FOLLOW_LOGS=true
                    shift
                    ;;
                -h|--help)
                    show_help
                    exit 0
                    ;;
                *)
                    echo -e "${RED}알 수 없는 옵션입니다: $1${NC}"
                    show_help
                    exit 1
                    ;;
            esac
        done
        show_logs
        ;;

    -h|--help|"")
        show_help
        ;;

    *)
        echo -e "${RED}알 수 없는 명령어: $ACTION${NC}"
        show_help
        exit 1
        ;;
esac
