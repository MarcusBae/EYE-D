#!/bin/bash
echo "=== 인텔 GPU 드라이버 및 권한 자동 설정 스크립트 (v2) ==="

# 1. 에러 유발 인텔 APT 저장소 임시 제거
if ls /etc/apt/sources.list.d/*intel*.list 1>/dev/null 2>&1; then
    echo "[1] 에러를 유발하는 기존 인텔 APT 저장소 파일을 /tmp/ 로 백업합니다."
    sudo mv /etc/apt/sources.list.d/*intel*.list /tmp/
fi

# 2. Ubuntu 24.04 (noble)용 인텔 그래픽스 PPA 추가 및 설치
echo "[2] 인텔 그래픽스 공식 PPA를 추가하고 드라이버를 설치합니다."
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt update
sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd libze-dev

# 3. 사용자 계정을 render, video 그룹에 등록 (GPU 접근 권한 부여)
echo "[3] 현재 사용자($USER)를 render 및 video 그룹에 등록합니다."
sudo usermod -aG render,video $USER

echo "=== 드라이버 설치 및 권한 등록이 성공적으로 완료되었습니다! ==="
echo "동작을 적용하기 위해 현재 터미널 창을 완전히 닫고, '새 터미널 창'을 열어주세요."
echo "새 터미널 창을 연 후 아래 테스트 명령어를 실행해 보세요:"
echo "conda activate cv_poc"
echo "python -c \"import torch; print('XPU Available:', torch.xpu.is_available())\""
