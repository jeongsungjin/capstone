#!/bin/bash
# ==============================================
# Capstone 환경 설정 스크립트 (Conda 깨짐 없이 안전)
# ==============================================

echo "🚀 === Capstone 환경 설정 시작 ==="

# 사용자 홈과 워크스페이스 루트
USER_HOME="${HOME:-/home/jamie}"
WS_ROOT="$USER_HOME/capstone"
CARLA_ROOT_DEFAULT="$USER_HOME/carla"

# 가능한 한 사용자 환경을 존중하되, 필요 경로만 추가
export PYTHONNOUSERSITE=1

# ROS Noetic 환경
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    source /opt/ros/noetic/setup.bash
    echo "✓ ROS Noetic 환경 설정 완료"
else
    echo "⚠ ROS Noetic setup.bash를 찾을 수 없습니다"
fi

# 워크스페이스 설정 (devel이 있는 경우만)
if [ -f "$WS_ROOT/devel/setup.bash" ]; then
    source "$WS_ROOT/devel/setup.bash"
    echo "✓ capstone 워크스페이스 설정 완료"
fi

# ROS 패키지를 위한 시스템 Python 패키지 경로 추가
export PYTHONPATH="$PYTHONPATH:/usr/lib/python3/dist-packages"

# CARLA Python API 경로 설정
CARLA_ROOT="${CARLA_ROOT:-$CARLA_ROOT_DEFAULT}"
export CARLA_ROOT

# 1) 빌드 경로 탐색 (lib.linux-x86_64-cpython-XX)
CARLA_BUILD_PATH=""
for cand in "$CARLA_ROOT/PythonAPI/carla"/build/lib.linux-x86_64-cpython-*; do
    if [ -d "$cand" ]; then
        CARLA_BUILD_PATH="$cand"
        break
    fi
done

# 2) dist egg 탐색 (carla-*-py3*.egg)
CARLA_EGG_PATH=""
for egg in "$CARLA_ROOT/PythonAPI/carla"/dist/carla-*-py3*.egg; do
    if [ -f "$egg" ]; then
        CARLA_EGG_PATH="$egg"
        break
    fi
done

# 선택 적용: 빌드 우선, 없으면 egg
if [ -n "$CARLA_BUILD_PATH" ]; then
    export CARLA_PYTHON_PATH="$CARLA_BUILD_PATH"
    export PYTHONPATH="$CARLA_PYTHON_PATH:$PYTHONPATH"
    echo "✓ CARLA build 경로 추가됨: $CARLA_BUILD_PATH"
elif [ -n "$CARLA_EGG_PATH" ]; then
    export CARLA_PYTHON_PATH="$CARLA_EGG_PATH"
    export PYTHONPATH="$CARLA_PYTHON_PATH:$PYTHONPATH"
    echo "✓ CARLA egg 추가됨: $CARLA_EGG_PATH"
else
    echo "⚠ CARLA Python 모듈을 찾지 못했습니다."
    echo "  - 빌드 경로 시도: $CARLA_ROOT/PythonAPI/carla/build/lib.linux-x86_64-cpython-<ver>"
    echo "  - 또는 egg: $CARLA_ROOT/PythonAPI/carla/dist/carla-*-py3*.egg"
fi

# Agents/예제 스크립트 경로 추가
CARLA_AGENTS_PATH="$CARLA_ROOT/PythonAPI/carla"
if [ -d "$CARLA_AGENTS_PATH" ]; then
    export PYTHONPATH="$CARLA_AGENTS_PATH:$PYTHONPATH"
    echo "✓ CARLA agents 경로 추가됨"
fi

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
echo "워크스페이스: $WS_ROOT"
echo "CARLA_ROOT: $CARLA_ROOT"
if [ -n "$CARLA_PYTHON_PATH" ]; then
    echo "CARLA_PYTHON_PATH: $CARLA_PYTHON_PATH"
fi
echo ""
echo "🎉 Capstone 환경 설정 완료!"
echo ""
echo "필요 패키지 설치: pip3 install -r requirements.txt"
echo "빌드: catkin build"
