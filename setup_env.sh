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

# 워크스페이스 설정 (devel이 있는 경우)
if [ -f "/home/jamie/capstone/devel/setup.bash" ]; then
    source /home/jamie/capstone/devel/setup.bash
    echo "✓ capstone 워크스페이스 설정 완료"
fi

# ROS 패키지를 위한 시스템 Python 패키지 경로 추가
export PYTHONPATH="$PYTHONPATH:/usr/lib/python3/dist-packages"

# Python 3.8을 기본으로 사용하도록 설정
if [ -f "/home/ctrl/anaconda3/envs/ros/bin/python3.8" ]; then
    export PATH="/home/ctrl/anaconda3/envs/ros/bin:$PATH"
    export PYTHON="/home/ctrl/anaconda3/envs/ros/bin/python3.8"
    export PYTHON3="/home/ctrl/anaconda3/envs/ros/bin/python3.8"
    export PYTHON_EXECUTABLE="/home/ctrl/anaconda3/envs/ros/bin/python3.8"
fi

# ROS가 올바른 Python을 사용하도록 환경변수 설정
export ROS_PYTHON_VERSION=3
export ROS_PYTHON_EXECUTABLE="/home/ctrl/anaconda3/envs/ros/bin/python3.8"

# CARLA Python API 경로 추가
CARLA_ROOT="/home/jamie/carla"
CARLA_BUILD_PATH="$CARLA_ROOT/PythonAPI/carla/build/lib.linux-x86_64-cpython-38"
CARLA_AGENTS_PATH="$CARLA_ROOT/PythonAPI/carla"

if [ -d "$CARLA_BUILD_PATH" ]; then
    export PYTHONPATH="$CARLA_BUILD_PATH:$PYTHONPATH"
    echo "✓ CARLA build 경로 추가됨"
fi

if [ -d "$CARLA_AGENTS_PATH" ]; then
    export PYTHONPATH="$CARLA_AGENTS_PATH:$PYTHONPATH"
    echo "✓ CARLA agents 경로 추가됨"
fi

export CARLA_ROOT="$CARLA_ROOT"

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
echo "워크스페이스: /home/jamie/capstone"
echo ""
echo "🎉 Capstone 환경 설정 완료!"
echo ""
echo "필요 패키지 설치: pip3 install -r requirements.txt"
echo "빌드: catkin build"
