#!/bin/bash
# ==============================================
# Capstone 환경 설정 스크립트 (Conda 깨짐 없이 안전)
# ==============================================

echo "🚀 === Capstone 환경 설정 시작 ==="

# PYTHONPATH 초기화 (환경 충돌 방지)
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# ROS Noetic 환경
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    source /opt/ros/noetic/setup.bash
    echo "✓ ROS Noetic 환경 설정 완료"
else
    echo "⚠ ROS Noetic setup.bash를 찾을 수 없습니다"
fi

# Capstone 워크스페이스 환경
if [ -f "$HOME/capstone/devel/setup.bash" ]; then
    source "$HOME/capstone/devel/setup.bash"
    echo "✓ Capstone 워크스페이스 환경 설정 완료"
else
    echo "⚠ Capstone 워크스페이스 devel/setup.bash를 찾을 수 없습니다"
fi

# CARLA Python egg 추가 (선택)
CARLA_EGG="$HOME/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux_x86_64.egg"
if [ -f "$CARLA_EGG" ]; then
    export PYTHONPATH="$CARLA_EGG:$PYTHONPATH"
    echo "✓ CARLA egg 파일 추가됨"
fi

# 시스템 Python 패키지 경로 추가
export PYTHONPATH="$PYTHONPATH:/usr/lib/python3/dist-packages"

# ROS Python 설정
export ROS_PYTHON_VERSION=3
export ROS_PYTHON_EXECUTABLE="/usr/bin/python3"

# 필수 Python 패키지 확인
echo ""
echo "=== Python 패키지 확인 ==="
python3 -c "import networkx" 2>/dev/null || echo "⚠ networkx 미설치"
python3 -c "import numpy" 2>/dev/null || echo "⚠ numpy 미설치"
python3 -c "import cv2" 2>/dev/null || echo "⚠ cv2 미설치"

# 환경 상태 출력
echo ""
echo "=== 환경 상태 ==="
echo "Python 경로: $(which python3)"
echo "Python 버전: $(python3 --version)"
echo "ROS 버전: $ROS_DISTRO"
echo "워크스페이스: $HOME/capstone"
echo ""
echo "🎉 Capstone 환경 설정 완료!"
echo ""
echo "필요 패키지 설치: pip3 install -r requirements.txt"
echo "빌드: catkin build"
