#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA 교통 차량 스폰 스크립트

현재 맵에 차량(오직 4륜 차량만) 10대를 스폰하고 
CARLA의 autopilot 기능을 활성화하여 자율주행하게 합니다.
"""

import rospy
import random
import sys

try:
    import carla
except ImportError:
    rospy.logerr("CARLA Python API를 import할 수 없습니다. PYTHONPATH를 확인하세요.")
    sys.exit(1)


class TrafficVehicleSpawner:
    """
    CARLA에서 교통 차량들을 스폰하고 자율주행을 활성화하는 클래스
    """
    
    def __init__(self):
        rospy.init_node('carla_traffic_vehicle_spawner', anonymous=True)
        
        # ROS 파라미터
        self.host = rospy.get_param('~host', 'localhost')
        self.port = rospy.get_param('~port', 2000)
        self.num_vehicles = rospy.get_param('~num_vehicles', 10)
        self.tm_port = rospy.get_param('~traffic_manager_port', 8000)
        self.safe_mode = rospy.get_param('~safe_mode', True)
        
        # CARLA 연결
        self.client = None
        self.world = None
        self.traffic_manager = None
        self.vehicles = []
        
        # 차량 블루프린트 필터 (오토바이, 자전거 제외)
        self.vehicle_filters = [
            'vehicle.audi.*',
            'vehicle.bmw.*',
            'vehicle.chevrolet.*',
            'vehicle.citroen.*',
            'vehicle.dodge.*',
            'vehicle.ford.*',
            'vehicle.jeep.*',
            'vehicle.lincoln.*',
            'vehicle.mercedes.*',
            'vehicle.mini.*',
            'vehicle.nissan.*',
            'vehicle.seat.*',
            'vehicle.tesla.*',
            'vehicle.toyota.*',
        ]
        
        # 제외할 차량 타입 (2륜 차량)
        self.excluded_keywords = ['bike', 'motorcycle', 'motorbike', 'bicycle', 'omafiets', 'century', 'diamondback']
        
        self._init_carla()
        
    def _init_carla(self):
        """CARLA에 연결하고 Traffic Manager를 초기화합니다."""
        try:
            rospy.loginfo(f"CARLA 서버에 연결 중... ({self.host}:{self.port})")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            
            # Traffic Manager 설정
            rospy.loginfo(f"Traffic Manager 초기화 중... (포트: {self.tm_port})")
            self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            
            if self.safe_mode:
                self.traffic_manager.set_synchronous_mode(False)
                rospy.loginfo("Traffic Manager: Safe Mode 활성화")
            
            rospy.loginfo("CARLA 초기화 완료")
            
        except Exception as e:
            rospy.logerr(f"CARLA 초기화 실패: {e}")
            sys.exit(1)
    
    def _get_vehicle_blueprints(self):
        """4륜 차량 블루프린트만 필터링하여 반환합니다."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = []
        
        # 차량 필터로 블루프린트 수집
        for filter_pattern in self.vehicle_filters:
            blueprints = blueprint_library.filter(filter_pattern)
            vehicle_blueprints.extend(blueprints)
        
        # 2륜 차량 제외
        filtered_blueprints = []
        for bp in vehicle_blueprints:
            bp_id = bp.id.lower()
            if not any(keyword in bp_id for keyword in self.excluded_keywords):
                # 4륜 차량인지 확인 (wheel 개수가 4개인지)
                if bp.has_attribute('number_of_wheels'):
                    num_wheels = bp.get_attribute('number_of_wheels').as_int()
                    if num_wheels == 4:
                        filtered_blueprints.append(bp)
                else:
                    # number_of_wheels 속성이 없으면 ID로 판단
                    filtered_blueprints.append(bp)
        
        rospy.loginfo(f"사용 가능한 4륜 차량 블루프린트: {len(filtered_blueprints)}개")
        return filtered_blueprints
    
    def spawn_vehicles(self):
        """차량들을 스폰합니다."""
        try:
            # 4륜 차량 블루프린트 가져오기
            vehicle_blueprints = self._get_vehicle_blueprints()
            
            if not vehicle_blueprints:
                rospy.logerr("사용 가능한 차량 블루프린트가 없습니다.")
                return
            
            # 스폰 포인트 가져오기
            spawn_points = self.world.get_map().get_spawn_points()
            
            if len(spawn_points) < self.num_vehicles:
                rospy.logwarn(f"요청한 차량 수({self.num_vehicles})가 사용 가능한 스폰 포인트({len(spawn_points)})보다 많습니다.")
                self.num_vehicles = len(spawn_points)
            
            # 스폰 포인트 섞기
            random.shuffle(spawn_points)
            
            rospy.loginfo(f"{self.num_vehicles}대의 차량 스폰 시작...")
            
            spawned_count = 0
            for i in range(self.num_vehicles):
                # 랜덤 차량 블루프린트 선택
                blueprint = random.choice(vehicle_blueprints)
                
                # 블루프린트 복사 (색상 변경을 위해)
                blueprint_copy = blueprint
                
                # 랜덤 색상 설정 (가능한 경우)
                if blueprint_copy.has_attribute('color'):
                    color = random.choice(blueprint_copy.get_attribute('color').recommended_values)
                    blueprint_copy.set_attribute('color', color)
                
                # role_name 설정
                if blueprint_copy.has_attribute('role_name'):
                    blueprint_copy.set_attribute('role_name', f'traffic_{i+1}')
                
                # 차량 스폰
                spawn_point = spawn_points[i]
                vehicle = None
                
                # 최대 3번 시도
                for attempt in range(3):
                    vehicle = self.world.try_spawn_actor(blueprint_copy, spawn_point)
                    if vehicle is not None:
                        break
                    
                    rospy.logwarn(f"차량 {i+1} 스폰 실패 (시도 {attempt + 1}/3)")
                    
                    # 다음 스폰 포인트 시도
                    if i + attempt + 1 < len(spawn_points):
                        spawn_point = spawn_points[i + attempt + 1]
                
                if vehicle is not None:
                    self.vehicles.append(vehicle)
                    
                    # Autopilot 활성화
                    vehicle.set_autopilot(True, self.tm_port)
                    
                    # Traffic Manager 개별 차량 설정
                    # 속도 변화 (-30% ~ +20%)
                    speed_diff = random.uniform(-30.0, 20.0)
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)
                    
                    # 차선 변경 허용
                    self.traffic_manager.auto_lane_change(vehicle, True)
                    
                    # 앞차와의 거리 (2.0 ~ 4.0m)
                    distance = random.uniform(2.0, 4.0)
                    self.traffic_manager.distance_to_leading_vehicle(vehicle, distance)
                    
                    # 신호등 준수 (0% 무시 = 100% 준수)
                    self.traffic_manager.ignore_lights_percentage(vehicle, 0.0)
                    
                    # 표지판 준수
                    self.traffic_manager.ignore_signs_percentage(vehicle, 0.0)
                    
                    spawned_count += 1
                    rospy.loginfo(f"차량 {spawned_count}/{self.num_vehicles} 스폰 완료: {blueprint.id} [ID: {vehicle.id}]")
                else:
                    rospy.logerr(f"차량 {i+1} 스폰 실패")
            
            rospy.loginfo(f"총 {spawned_count}대의 차량이 스폰되고 자율주행을 시작했습니다.")
            
            if spawned_count < self.num_vehicles:
                rospy.logwarn(f"{self.num_vehicles - spawned_count}대의 차량 스폰 실패")
                
        except Exception as e:
            rospy.logerr(f"차량 스폰 중 오류 발생: {e}")
    
    def destroy_all_vehicles(self):
        """스폰된 모든 차량을 삭제합니다."""
        rospy.loginfo("차량 정리 중...")
        destroyed_count = 0
        for vehicle in self.vehicles:
            try:
                vehicle.destroy()
                destroyed_count += 1
            except Exception as e:
                rospy.logerr(f"차량 삭제 실패 [ID: {vehicle.id}]: {e}")
        
        rospy.loginfo(f"{destroyed_count}대의 차량이 삭제되었습니다.")
        self.vehicles.clear()
    
    def run(self):
        """메인 루프 실행"""
        rospy.loginfo("교통 차량 스폰 노드 실행 중... (Ctrl+C로 종료)")
        
        # 차량 스폰
        self.spawn_vehicles()
        
        try:
            # 노드가 종료될 때까지 대기
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("종료 신호 수신")
        finally:
            self.destroy_all_vehicles()


def main():
    try:
        node = TrafficVehicleSpawner()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

