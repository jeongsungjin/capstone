# carla_camera_stream

CARLA 시뮬레이터에서 고정 카메라와 교통 차량을 스폰하는 ROS 패키지입니다.

## 개요

이 패키지는 두 가지 주요 기능을 제공합니다:

### 1. 고정 카메라 스폰 (`spawn_cameras.py`)
다음 두 위치에 높이 7m로 카메라를 스폰합니다:
- **카메라 1 (좌하단)**: Position(-97.330, 7.528, 7.0) → 중심점을 향함
- **카메라 2 (우상단)**: Position(-116.797, 37.642, 7.0) → 중심점을 향함

각 카메라는 아래를 향하도록 설정되어 있으며(pitch=-20°), RGB 이미지를 ROS 토픽으로 퍼블리시합니다.

### 2. 교통 차량 스폰 (`spawn_traffic_vehicles.py`)
현재 맵에 4륜 차량(오토바이, 자전거 제외) 10대를 스폰하고 CARLA Autopilot을 활성화하여 자율주행하게 합니다.

## 의존성

- ROS Noetic (또는 ROS Melodic)
- CARLA Simulator (0.9.x)
- Python 3
- cv_bridge
- sensor_msgs
- OpenCV

## 빌드

```bash
cd /home/ctrl/capstone_2025
catkin_make --only-pkg-with-deps carla_camera_stream
# 또는
catkin build carla_camera_stream
```

빌드 후 환경 설정:
```bash
source devel/setup.bash
```

## 실행

### 1. CARLA 시뮬레이터 실행
먼저 CARLA 시뮬레이터를 실행해야 합니다:
```bash
cd /path/to/carla
./CarlaUE4.sh
```

### 2. 카메라 스폰 노드 실행

#### 방법 1: rosrun으로 실행
```bash
rosrun carla_camera_stream spawn_cameras.py
```

#### 방법 2: launch 파일로 실행 (권장)
```bash
roslaunch carla_camera_stream spawn_cameras.launch
```

#### 방법 3: 파라미터를 사용한 실행
launch 파일 사용:
```bash
roslaunch carla_camera_stream spawn_cameras.launch \
    host:=localhost \
    port:=2000 \
    image_width:=1920 \
    image_height:=1080 \
    fov:=110.0
```

rosrun 사용:
```bash
rosrun carla_camera_stream spawn_cameras.py \
    _host:=localhost \
    _port:=2000 \
    _image_width:=1280 \
    _image_height:=720 \
    _fov:=90.0
```

### 3. 교통 차량 스폰 노드 실행

#### 방법 1: launch 파일로 실행 (권장)
```bash
roslaunch carla_camera_stream spawn_traffic_vehicles.launch
```

#### 방법 2: 차량 수를 변경하여 실행
```bash
roslaunch carla_camera_stream spawn_traffic_vehicles.launch num_vehicles:=20
```

#### 방법 3: rosrun으로 실행
```bash
rosrun carla_camera_stream spawn_traffic_vehicles.py
```

#### 방법 4: 파라미터를 사용한 실행
```bash
rosrun carla_camera_stream spawn_traffic_vehicles.py \
    _num_vehicles:=15 \
    _traffic_manager_port:=8000 \
    _safe_mode:=true
```

## ROS 파라미터

### 카메라 노드 파라미터 (`spawn_cameras.py`)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `~host` | `localhost` | CARLA 서버 호스트 주소 |
| `~port` | `2000` | CARLA 서버 포트 |
| `~image_width` | `1280` | 이미지 가로 해상도 |
| `~image_height` | `720` | 이미지 세로 해상도 |
| `~fov` | `90.0` | 카메라 시야각(Field of View) |

### 교통 차량 노드 파라미터 (`spawn_traffic_vehicles.py`)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `~host` | `localhost` | CARLA 서버 호스트 주소 |
| `~port` | `2000` | CARLA 서버 포트 |
| `~num_vehicles` | `10` | 스폰할 차량 수 |
| `~traffic_manager_port` | `8000` | Traffic Manager 포트 |
| `~safe_mode` | `true` | 안전 모드 (동기화 비활성화) |

## 퍼블리시되는 토픽

- `/carla/camera_1/image_raw` (sensor_msgs/Image)
  - 첫 번째 카메라의 RGB 이미지
- `/carla/camera_1/camera_info` (sensor_msgs/CameraInfo)
  - 첫 번째 카메라의 캘리브레이션 정보

- `/carla/camera_2/image_raw` (sensor_msgs/Image)
  - 두 번째 카메라의 RGB 이미지
- `/carla/camera_2/camera_info` (sensor_msgs/CameraInfo)
  - 두 번째 카메라의 캘리브레이션 정보

## 이미지 확인

RViz에서 이미지를 확인하려면:
```bash
rviz
```

또는 rqt_image_view를 사용:
```bash
rqt_image_view /carla/camera_1/image_raw
rqt_image_view /carla/camera_2/image_raw
```

## 종료

노드를 종료하려면 `Ctrl+C`를 누르세요. 스폰된 카메라와 차량은 자동으로 정리됩니다.

## 교통 차량 기능 상세

### 차량 필터링
- **포함**: Audi, BMW, Chevrolet, Citroen, Dodge, Ford, Jeep, Lincoln, Mercedes, Mini, Nissan, Seat, Tesla, Toyota 등
- **제외**: 오토바이, 자전거 등 2륜 차량
- **확인**: `number_of_wheels == 4`인 차량만 스폰

### Autopilot 설정
각 차량은 개별적으로 다음과 같이 설정됩니다:
- **속도 변화**: -30% ~ +20% 랜덤
- **차선 변경**: 활성화
- **앞차와의 거리**: 2.0 ~ 4.0m 랜덤
- **신호등 준수**: 100% (무시 0%)
- **표지판 준수**: 100% (무시 0%)

### Traffic Manager
- Traffic Manager를 통해 모든 차량의 자율주행이 관리됩니다
- 포트: 8000 (기본값, 변경 가능)
- Safe Mode: 비동기 모드로 실행 (권장)

## 카메라 위치 및 방향 변경

카메라 위치와 방향을 변경하려면 `scripts/spawn_cameras.py` 파일의 다음 부분을 수정하세요:

```python
# 카메라 위치 정의
cam1_pos = {'x': -97.330, 'y': 7.528, 'z': 7.0}
cam2_pos = {'x': -116.797, 'y': 37.642, 'z': 7.0}

# 중심점 계산
center_x = (cam1_pos['x'] + cam2_pos['x']) / 2.0
center_y = (cam1_pos['y'] + cam2_pos['y']) / 2.0

# 각 카메라에서 중심점을 향하는 yaw 각도 계산 (degrees)
cam1_yaw = math.degrees(math.atan2(center_y - cam1_pos['y'], center_x - cam1_pos['x']))
cam2_yaw = math.degrees(math.atan2(center_y - cam2_pos['y'], center_x - cam2_pos['x']))

self.camera_positions = [
    {'x': cam1_pos['x'], 'y': cam1_pos['y'], 'z': cam1_pos['z'], 'yaw': cam1_yaw, 'name': 'camera_1'},
    {'x': cam2_pos['x'], 'y': cam2_pos['y'], 'z': cam2_pos['z'], 'yaw': cam2_yaw, 'name': 'camera_2'}
]
```

### 수동으로 yaw 각도 지정

자동 계산 대신 수동으로 yaw를 지정하려면:

```python
self.camera_positions = [
    {'x': -97.330, 'y': 7.528, 'z': 7.0, 'yaw': 45.0, 'name': 'camera_1'},
    {'x': -116.797, 'y': 37.642, 'z': 7.0, 'yaw': -135.0, 'name': 'camera_2'}
]
```

**참고**: 
- `yaw`는 degrees 단위입니다 (0° = +X 방향, 90° = +Y 방향)
- `pitch=-45.0`으로 설정되어 카메라가 대각선 아래를 향합니다
- 현재 설정은 두 카메라가 공통 중심점을 향하도록 자동 계산됩니다

## 문제 해결

### CARLA Python API import 실패
PYTHONPATH에 CARLA Python API가 포함되어 있는지 확인하세요:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.X-py3.X-linux-x86_64.egg
```

### 카메라가 스폰되지 않음
- CARLA 시뮬레이터가 실행 중인지 확인
- 포트 번호가 올바른지 확인 (기본값: 2000)
- 지정된 위치가 맵 내부에 있는지 확인

## 라이센스

MIT

