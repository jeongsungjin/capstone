# carla_multi_vehicle_planner

ROS1 package that spawns multiple CARLA vehicles, plans per-vehicle NetworkX A* routes across the HD map, executes Pure Pursuit control, and renders a BEV visualization for RViz.

## Nodes

- `multi_vehicle_spawner.py`: spawns `ego_vehicle_N` actors in CARLA and publishes their odometry.
- `networkx_path_planner.py`: builds a dense waypoint graph, runs A* to random destinations, publishes ROS paths, and draws debug lines.
- `multi_vehicle_controller.py`: applies Pure Pursuit steering/throttle and publishes Ackermann commands.
- `bev_visualizer.py`: paints a BEV image and RViz markers for vehicles/paths with unique colors.

## Usage

### 통합 시스템 실행 (권장)
모든 시스템과 RViz를 한 번에 실행하려면:

```
roslaunch carla_multi_vehicle_planner integrated_multi_vehicle_system.launch
```

이 명령어는 다음을 순서대로 실행합니다:
1. `multi_vehicle_autonomy.launch` - 멀티 비히클 자율 주행 시스템
2. `manual_goal_tools.launch` - 수동 목표 설정 도구  
3. RViz - 자동으로 설정된 디스플레이로 실행

### 개별 시스템 실행
개별 런치 파일만 실행하려면:

```
# 멀티 비히클 자율 주행 시스템만 실행
roslaunch carla_multi_vehicle_planner multi_vehicle_autonomy.launch

# 수동 목표 설정 도구만 실행
roslaunch carla_multi_vehicle_planner manual_goal_tools.launch
```

### 설정 옵션
통합 시스템 실행 시 다음 옵션을 사용할 수 있습니다:

```bash
# 비히클 개수 변경 (기본값: 3)
roslaunch carla_multi_vehicle_planner integrated_multi_vehicle_system.launch num_vehicles:=5

# RViz 비활성화
roslaunch carla_multi_vehicle_planner integrated_multi_vehicle_system.launch enable_rviz:=false

# 커스텀 RViz 설정 파일 사용
roslaunch carla_multi_vehicle_planner integrated_multi_vehicle_system.launch rviz_config:=/path/to/your/config.rviz
```

## 사전 요구사항

Start the CARLA simulator (0.9.16) before launching the ROS stack. The nodes expect CARLA to run on `localhost:2000` and the CARLA Python build path to be located at `/home/jamie/carla/PythonAPI/carla/build/lib.linux-x86_64-cpython-38`.

## RViz 설정

통합 런치 파일은 자동으로 다음 RViz 디스플레이를 설정합니다:
- Grid (맵 그리드)
- Path (3개 비히클의 계획된 경로)
- MarkerArray (비히클 마커와 경로 마커)
- Map (오커판시 그리드)

RViz 설정은 `config/multi_vehicle_planner.rviz`에서 수정할 수 있습니다.
