------------------- 운 용 법 ------------------------------

1. roscore
2. rosrun carla_multi_vehicle_planner inference_recv.py 
# 기본 포트 60200으로 설정 (인지 서버에서 통합된 추론 정보를 받아, /bev_info topic 으로 발행)
3. roslaunch carla_multi_vehicle full_system.launch
# enable_platooning:=true/false(군집주행 사용 여부, 군집주행 시 최소 3대)
# num_vehicles:=1/2/3/4/5/6(맵에 올려둔, 굴릴 차량 대수) enable_bev_pipeline:=true/false (실제 카메라 인지 사용 여부 즉, 실차 주행/시뮬레이션)

full_system.launch 까지 켜면 다 킨거긴합니다.

신경써야 할 것 - 인지 정보와 관련하여
1. http://192.168.0.165:18000/fusion/admin (인지 서버 주소 - 바뀔수있음. 안들어가지면 서버 연 주체에게 확인)
2. 서버의 인지를 확인한 후 차량별 색상이나, yaw각이 안맞는 것을 상시 확인 후 실제 환경에 맞게 정렬
# yaw를 뒤집어도 다시 뒤집힌다면, 인지 서버에서 칼만을 쓰기에 조종기로 진행 방향에 맞게 직진 조금 하고 다시 플립하면 웬만하면 맞는 인지로 고정됨
3. 서버의 인지를 확인한 후 맞는 인지가 들어왔을때 full_system.launch를 켜는것이 안정적 
# full_System.launch 에서 carla 차량별 초기 헝가리안 색상 매칭, 위치 자세 정렬 진행하기에, launch시 정렬된 인지를 받는게 중요

신경써야 할 것 - 차량 ip, 색상 매칭, carla_vehicle_number
1. idp_ackermann_senders.launch 의 6~11번째 라인에서 carla vehicle number와 아이피를 매칭시킵니다. 차량 번호와 ip에 맞게 제어 명령을 송신함
# 색상과 아이피는 주기되어있음. (모르면 자이카 운용병에게 물어보기)
  <arg name="vehicle_1_ip" default="10.15.129.7" /> <!-- 노랑  -->
  <arg name="vehicle_2_ip" default="10.15.129.5" /> <!-- 초록 -->
  <arg name="vehicle_3_ip" default="10.15.129.4" /> <!-- 발강 -->
  <arg name="vehicle_4_ip" default="10.15.129.51" /> <!-- 보라 -->
  <arg name="vehicle_5_ip" default="127.0.0.1" /> 
  <arg name="vehicle_6_ip" default="127.0.0.1" />

2. bev_id_teleporter.launch 의 12~17번째 라인에서 매칭된 색상 문자열에 맞게 차량을 위치에 맞게 스폰 및 투영시킵니다.
# vehicle_n 숫자를 항상 색상 순서에 맞게 신경쓰기
  <arg name="vehicle_1_color" default="yellow" />
  <arg name="vehicle_2_color" default="green" />
  <arg name="vehicle_3_color" default="red" />
  <arg name="vehicle_4_color" default="purple" />
  <arg name="vehicle_5_color" default="" />
  <arg name="vehicle_6_color" default="" />

3. full_system.launch 의 22~27번째 라인에도 vehicle_n이 있습니다. bev_id_teleporter.launch에게 주는 인자입니다.
# vehicle_n 숫자를 항상 색상 순서에 맞게 신경쓰기
  <arg name="vehicle_1_color" default="yellow" />
  <arg name="vehicle_2_color" default="green" />
  <arg name="vehicle_3_color" default="red" />
  <arg name="vehicle_4_color" default="purple" />
  <arg name="vehicle_5_color" default="" />
  <arg name="vehicle_6_color" default="" />

4. multi_vehicle_spawner.py의 32~39번째 라인, 차량 번호에 따른 스폰 차량 색상을 정의해두는 리스트
# 앞서 세팅한 차량 번호와 색상 순서대로 아래의 리스트에 색상에 맞게 자이카 모델을 잘 넣어야합니다
DEFAULT_MODEL_MAP: List[str] = [
    "vehicle.vehicle.yellowxycar",   # ego_vehicle_1
    "vehicle.vehicle.greenxycar",     # ego_vehicle_2
    "vehicle.vehicle.redxycar",  # ego_vehicle_3
    "vehicle.vehicle.purplexycar",  # ego_vehicle_4
    "vehicle.vehicle.pinkxycar",    # ego_vehicle_5
    "vehicle.vehicle.blackxycar",   # ego_vehicle_6
]

