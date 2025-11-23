# 통힙 런치를 켜지 않을 경우
0. roslaunch
1. 인지 결과 수신 및 차량 아이디, 좌표, 자세 토픽 퍼블리쉬 코드 inference_recv.py
2. 인지 결과 토픽 바탕으로 carla상 차량을 teleport하는 런치 bev_id_teleport.launch
3. carla상 차량의 좌표, 자세 기반 planning 및 제어 명령 발행 런치 conflict_free_system.launch
4. 계산된 차량별 제어 명령을 실제 자이카로 전송하는 런치 udp_ackermann_control.launch

# 통합 런치 켤 경우
1. full_system.launch
+ 일반적으로 사용 가능한 인자
  - num_vehicle:=1~6 - 사용 차량 수
  - enable_platooning:=True/False - 플래투닝 사용 여부
  - _udp/-ports:=60200 - 카메라 인지 결과 수신 포트


  # 추가 정보
  1. udp_ackermann_sender.launch에서 치량별 ip 신경쓰기  <arg name="vehicle_1_ip" default="10.15.129.54" /> name의 번호대로 대수가 올라갑니다.
  2. 차량별 세부 제어가 필요한 경우 다음 차량별 파라미터 튜닝이 가능합니다.
    <param name="vehicles/ego_vehicle_1/angle_scale" value="1.3" />
    <param name="vehicles/ego_vehicle_1/angle_clip" value="50" />
    <param name="vehicles/ego_vehicle_1/angle_min_abs" value="0" />
    <param name="vehicles/ego_vehicle_1/angle_invert" value="false" />
    <param name="vehicles/ego_vehicle_1/angle_center_rad" value="-0.15" />
    <param name="vehicles/ego_vehicle_1/speed_scale" value="0.97" />
    <param name="vehicles/ego_vehicle_1/speed_min_abs" value="1.0" />
    <param name="vehicles/ego_vehicle_1/force_min_speed_on_zero" value="true" />
    <param name="vehicles/ego_vehicle_1/zero_speed_value" value="0.0" />


