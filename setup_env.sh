#!/bin/bash

# capstone 워크스페이스 환경 설정 스크립트
# conda 환경과 ROS 환경의 충돌을 해결합니다.

echo "=== Capstone 워크스페이스 환경 설정 ==="

# PYTHONPATH 초기화 (환경 충돌 방지)
echo "PYTHONPATH 초기화..."
unset PYTHONPATH
unset CONDA_BACKUP_PYTHONPATH
export PYTHONPATH=""

# conda ros 환경 활성화 (경고 무시)
if [ -d "/home/ctrl/anaconda3/envs/ros" ]; then
    echo "conda ros 환경 활성화..."
    conda activate ros 2>/dev/null || {
        echo "수동으로 conda 환경 설정..."
        export CONDA_DEFAULT_ENV=ros
        export PATH="/home/ctrl/anaconda3/envs/ros/bin:$PATH"
        export CONDA_PREFIX="/home/ctrl/anaconda3/envs/ros"
        export CONDA_PYTHON_EXE="/home/ctrl/anaconda3/envs/ros/bin/python3.8"
    }
fi

# ROS 환경 설정
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    source /opt/ros/noetic/setup.bash
    echo "✓ ROS Noetic 환경 설정 완료"
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

# 필요한 Python 패키지 확인
echo ""
echo "=== Python 패키지 확인 ==="
python -c "import networkx; print('✓ networkx:', networkx.__version__)" 2>/dev/null || echo "⚠ networkx가 설치되지 않음 (pip install networkx)"
python -c "import numpy; print('✓ numpy:', numpy.__version__)" 2>/dev/null || echo "⚠ numpy가 설치되지 않음"
python -c "import cv2; print('✓ opencv:', cv2.__version__)" 2>/dev/null || echo "⚠ opencv가 설치되지 않음"

# 환경 확인
echo ""
echo "=== 환경 상태 ==="
echo "Python 버전: $(python --version 2>/dev/null || echo 'Python not found')"
echo "ROS 버전: $ROS_DISTRO"
echo "워크스페이스: /home/jamie/capstone"
echo ""

echo "🎉 환경 설정이 완료되었습니다!"
echo ""
echo "필요한 패키지 설치:"
echo "  pip install -r requirements.txt"
echo ""
echo "빌드:"
echo "  catkin build"

