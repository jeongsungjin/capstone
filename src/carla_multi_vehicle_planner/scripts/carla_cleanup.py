#!/usr/bin/env python3

import os
import sys
import signal
import rospy

# ==== CARLA egg 경로 자동 추가 ====
# def append_carla_egg():
#     carla_python_path = os.getenv("CARLA_PYTHON_PATH")
#     if carla_python_path is None:
#         print("Warning: CARLA_PYTHON_PATH 환경변수가 설정되지 않았습니다.")
#         return False

#     # 예: carla-0.9.13-py3.7-linux-x86_64.egg
#     for fname in os.listdir(carla_python_path):
#         if fname.startswith("carla-") and fname.endswith(".egg") and "py3.7" in fname:
#             full_path = os.path.join(carla_python_path, fname)
#             if full_path not in sys.path:
#                 sys.path.append(full_path)
#             return True
#     return False

# if not append_carla_egg():
#     print("CARLA egg 파일을 찾을 수 없습니다. CARLA_PYTHON_PATH를 확인하세요.")
#     sys.exit(1)

# ==== carla 모듈 임포트 ====
try:
    import carla
except ImportError as e:
    print(f"CARLA 모듈을 import할 수 없습니다: {e}")
    sys.exit(1)


class CarlaCleanup:
    def __init__(self):
        rospy.init_node("carla_cleanup", anonymous=True)
        
        self.client = None
        self.world = None
        self.cleanup_done = False
        
        # 종료 시그널 처리
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # CARLA 연결
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            rospy.loginfo("CARLA Cleanup 노드가 연결되었습니다.")
            
        except Exception as e:
            rospy.logerr(f"CARLA 연결 실패: {e}")
            sys.exit(1)
    
    def cleanup_all_actors(self):
        """CARLA 월드의 모든 액터 정리"""
        if self.cleanup_done or self.world is None:
            return
            
        try:
            rospy.loginfo("CARLA 월드 정리를 시작합니다...")
            
            # 모든 액터 가져오기
            all_actors = self.world.get_actors()
            
            # 액터 타입별 분류
            vehicles = []
            sensors = []
            walkers = []
            others = []
            
            for actor in all_actors:
                if 'vehicle' in actor.type_id.lower():
                    vehicles.append(actor)
                elif 'sensor' in actor.type_id.lower() or 'camera' in actor.type_id.lower():
                    sensors.append(actor)
                elif 'walker' in actor.type_id.lower():
                    walkers.append(actor)
                else:
                    others.append(actor)
            
            rospy.loginfo(f"발견된 액터: 차량 {len(vehicles)}대, 센서 {len(sensors)}개, 보행자 {len(walkers)}명, 기타 {len(others)}개")
            
            # 센서 먼저 정리 (카메라 등)
            destroyed_sensors = 0
            for sensor in sensors:
                try:
                    if hasattr(sensor, 'stop'):
                        sensor.stop()  # 센서 데이터 수집 중단
                    sensor.destroy()
                    destroyed_sensors += 1
                except Exception as e:
                    rospy.logdebug(f"센서 {sensor.id} 정리 실패: {e}")
            
            # 차량 정리
            destroyed_vehicles = 0  
            for vehicle in vehicles:
                try:
                    vehicle.destroy()
                    destroyed_vehicles += 1
                except Exception as e:
                    rospy.logdebug(f"차량 {vehicle.id} 정리 실패: {e}")
            
            # 보행자 정리
            destroyed_walkers = 0
            for walker in walkers:
                try:
                    walker.destroy()
                    destroyed_walkers += 1
                except Exception as e:
                    rospy.logdebug(f"보행자 {walker.id} 정리 실패: {e}")
            
            # 기타 액터 정리 (traffic lights, signs 등은 제외)
            destroyed_others = 0
            for other in others:
                try:
                    # 중요한 월드 액터들은 건드리지 않음
                    if ('traffic' not in other.type_id.lower() and 
                        'static' not in other.type_id.lower() and
                        'spectator' not in other.type_id.lower()):
                        other.destroy()
                        destroyed_others += 1
                except Exception as e:
                    rospy.logdebug(f"기타 액터 {other.id} 정리 실패: {e}")
            
            rospy.loginfo(f"정리 완료: 차량 {destroyed_vehicles}대, 센서 {destroyed_sensors}개, 보행자 {destroyed_walkers}명, 기타 {destroyed_others}개")
            
            # 동기 모드 해제 (만약 활성화되어 있다면)
            try:
                settings = self.world.get_settings()
                if settings.synchronous_mode:
                    settings.synchronous_mode = False
                    self.world.apply_settings(settings)
                    rospy.loginfo("동기 모드가 해제되었습니다.")
            except Exception as e:
                rospy.logwarn(f"동기 모드 해제 실패: {e}")
            
            # Traffic Manager 정리
            try:
                traffic_manager = self.client.get_trafficmanager(8000)
                traffic_manager.set_synchronous_mode(False)
                rospy.loginfo("Traffic Manager 동기 모드가 해제되었습니다.")
            except Exception as e:
                rospy.logwarn(f"Traffic Manager 정리 실패: {e}")
                
            self.cleanup_done = True
            rospy.loginfo("CARLA 월드 정리가 완료되었습니다.")
            
        except Exception as e:
            rospy.logerr(f"정리 과정에서 오류 발생: {e}")
    
    def signal_handler(self, signum, frame):
        """시그널 핸들러 - Ctrl+C 또는 종료 시그널 처리"""
        rospy.loginfo(f"종료 시그널 {signum} 수신. 정리 중...")
        self.cleanup_all_actors()
        rospy.loginfo("Cleanup 완료. 노드를 종료합니다.")
        sys.exit(0)
    
    def run(self):
        """메인 실행 루프 - ROS 노드가 살아있는 동안 대기"""
        try:
            rospy.loginfo("CARLA Cleanup 노드가 실행 중입니다. Ctrl+C로 종료하세요.")
            rospy.spin()  # ROS 노드가 종료될 때까지 대기
        except rospy.ROSInterruptException:
            pass
        finally:
            if not self.cleanup_done:
                self.cleanup_all_actors()


class CarlaInstantCleanup:
    """즉시 정리를 수행하는 클래스 (스크립트 직접 실행용)"""
    def __init__(self):
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print("CARLA에 연결되었습니다.")
        except Exception as e:
            print(f"CARLA 연결 실패: {e}")
            sys.exit(1)
    
    def cleanup_now(self):
        """즉시 모든 액터 정리"""
        try:
            print("CARLA 월드 정리를 시작합니다...")
            
            all_actors = self.world.get_actors()
            vehicles = [a for a in all_actors if 'vehicle' in a.type_id.lower()]
            sensors = [a for a in all_actors if 'sensor' in a.type_id.lower() or 'camera' in a.type_id.lower()]
            walkers = [a for a in all_actors if 'walker' in a.type_id.lower()]
            
            print(f"발견된 액터: 차량 {len(vehicles)}대, 센서 {len(sensors)}개, 보행자 {len(walkers)}명")
            
            # 센서부터 정리
            for sensor in sensors:
                try:
                    if hasattr(sensor, 'stop'):
                        sensor.stop()
                    sensor.destroy()
                except:
                    pass
            
            # 차량 정리
            for vehicle in vehicles:
                try:
                    vehicle.destroy()
                except:
                    pass
            
            # 보행자 정리
            for walker in walkers:
                try:
                    walker.destroy()
                except:
                    pass
            
            # 설정 초기화
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                
                traffic_manager = self.client.get_trafficmanager(8000)
                traffic_manager.set_synchronous_mode(False)
            except:
                pass
            
            print("CARLA 월드 정리가 완료되었습니다.")
            
        except Exception as e:
            print(f"정리 중 오류 발생: {e}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--instant':
        # 즉시 정리 모드
        cleanup = CarlaInstantCleanup()
        cleanup.cleanup_now()
    else:
        # ROS 노드 모드 
        try:
            cleanup_node = CarlaCleanup()
            cleanup_node.run()
        except Exception as e:
            print(f"오류: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main() 