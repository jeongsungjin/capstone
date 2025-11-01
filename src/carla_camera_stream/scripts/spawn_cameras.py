#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA 고정 카메라 2대 스폰 및 이미지 퍼블리시 노드

두 개의 고정된 위치에 높이 7m로 카메라를 스폰하고 
각각의 이미지를 ROS 토픽으로 퍼블리시합니다.
"""

import rospy
import cv2
import numpy as np
import math
import sys

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

try:
    import carla
except ImportError:
    rospy.logerr("CARLA Python API를 import할 수 없습니다. PYTHONPATH를 확인하세요.")
    sys.exit(1)


class CarlaCameraStream:
    """
    CARLA에서 두 개의 고정 카메라를 스폰하고 이미지를 퍼블리시하는 클래스
    """
    
    def __init__(self):
        rospy.init_node('carla_camera_stream', anonymous=True)
        
        # ROS 파라미터
        self.host = rospy.get_param('~host', 'localhost')
        self.port = rospy.get_param('~port', 2000)
        self.image_width = rospy.get_param('~image_width', 1280)
        self.image_height = rospy.get_param('~image_height', 720)
        self.fov = rospy.get_param('~fov', 90.0)
        
        # 카메라 위치 정의 (높이 7m)
        # 두 카메라 사이의 중심점을 계산하여 서로가 중심을 향하도록 yaw 설정
        cam1_pos = {'x': -5.79, 'y': -22.81, 'z': 7.0}
        cam2_pos = {'x': 5.79, 'y': -22.81, 'z': 7.0}
        
        # 중심점 계산
        center_x = (cam1_pos['x'] + cam2_pos['x']) / 2.0
        center_y = (cam1_pos['y'] + cam2_pos['y']) / 2.0
        
        print(f"중심점: ({center_x:.2f}, {center_y:.2f})")
        
        # 각 카메라에서 중심점을 향하는 yaw 각도 계산 (degrees)
        cam1_yaw = math.degrees(math.atan2(center_y - cam1_pos['y'], center_x - cam1_pos['x']))
        cam2_yaw = math.degrees(math.atan2(center_y - cam2_pos['y'], center_x - cam2_pos['x']))
        
        self.camera_positions = [
            {'x': cam1_pos['x'], 'y': cam1_pos['y'], 'z': cam1_pos['z'], 'yaw': cam1_yaw, 'name': 'camera_1'},
            {'x': cam2_pos['x'], 'y': cam2_pos['y'], 'z': cam2_pos['z'], 'yaw': cam2_yaw, 'name': 'camera_2'}
        ]
        
        rospy.loginfo(f"카메라 1 yaw: {cam1_yaw:.2f}°, 카메라 2 yaw: {cam2_yaw:.2f}°")
        
        # ROS Publisher 생성
        self.bridge = CvBridge()
        self.publishers = {}
        self.camera_info_pubs = {}
        
        for cam_pos in self.camera_positions:
            cam_name = cam_pos['name']
            self.publishers[cam_name] = rospy.Publisher(
                f'/carla/{cam_name}/image_raw', 
                Image, 
                queue_size=10
            )
            self.camera_info_pubs[cam_name] = rospy.Publisher(
                f'/carla/{cam_name}/camera_info', 
                CameraInfo, 
                queue_size=10, 
                latch=True
            )
        
        # CARLA 연결 및 카메라 스폰
        self.client = None
        self.world = None
        self.cameras = []
        self.camera_info_msgs = {}
        
        self._init_carla()
        
    def _init_carla(self):
        """CARLA에 연결하고 카메라를 스폰합니다."""
        try:
            # CARLA 클라이언트 연결
            rospy.loginfo(f"CARLA 서버에 연결 중... ({self.host}:{self.port})")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            blueprint_library = self.world.get_blueprint_library()
            
            # 카메라 블루프린트 설정
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.image_width))
            camera_bp.set_attribute('image_size_y', str(self.image_height))
            camera_bp.set_attribute('fov', str(self.fov))
            
            # CameraInfo 메시지 생성
            for cam_pos in self.camera_positions:
                cam_name = cam_pos['name']
                self.camera_info_msgs[cam_name] = self._build_camera_info(cam_name)
                # CameraInfo 퍼블리시 (latch=True이므로 한 번만)
                self.camera_info_pubs[cam_name].publish(self.camera_info_msgs[cam_name])
            
            # 각 위치에 카메라 스폰
            for cam_pos in self.camera_positions:
                cam_name = cam_pos['name']
                x, y, z = cam_pos['x'], cam_pos['y'], cam_pos['z']
                yaw = cam_pos['yaw']
                
                # 카메라 위치 및 회전 설정 (pitch=-45도로 아래를 향하고, yaw로 방향 설정)
                camera_transform = carla.Transform(
                    carla.Location(x=x, y=y, z=z),
                    carla.Rotation(pitch=-20.0, yaw=yaw, roll=0.0)
                )
                
                # 카메라 스폰
                camera = self.world.spawn_actor(camera_bp, camera_transform)
                self.cameras.append({'actor': camera, 'name': cam_name})
                
                # 콜백 함수 등록
                camera.listen(lambda image, name=cam_name: self._camera_callback(image, name))
                
                rospy.loginfo(f"카메라 스폰 완료: {cam_name} at ({x:.2f}, {y:.2f}, {z:.2f}), yaw: {yaw:.2f}° [ID: {camera.id}]")
            
            rospy.loginfo(f"총 {len(self.cameras)}대의 카메라가 스폰되었습니다.")
            
        except Exception as e:
            rospy.logerr(f"CARLA 초기화 실패: {e}")
            sys.exit(1)
    
    def _build_camera_info(self, cam_name):
        """CameraInfo 메시지를 생성합니다."""
        info = CameraInfo()
        info.width = self.image_width
        info.height = self.image_height
        info.distortion_model = 'plumb_bob'
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 카메라 내부 파라미터 계산
        fx = self.image_width / (2.0 * math.tan(math.radians(self.fov) / 2.0))
        fy = fx
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        
        info.K = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.P = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.header.frame_id = f"{cam_name}_optical"
        
        return info
    
    def _camera_callback(self, image, cam_name):
        """CARLA 이미지를 ROS Image 메시지로 변환하여 퍼블리시합니다."""
        try:
            # CARLA 이미지를 numpy 배열로 변환
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # BGRA
            frame_bgr = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
            
            # ROS Image 메시지로 변환
            ros_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
            ros_msg.header.stamp = rospy.Time.now()
            ros_msg.header.frame_id = f"{cam_name}_optical"
            
            # 퍼블리시
            self.publishers[cam_name].publish(ros_msg)
            
        except Exception as e:
            rospy.logerr(f"이미지 콜백 에러 ({cam_name}): {e}")
    
    def destroy(self):
        """카메라 액터를 정리합니다."""
        rospy.loginfo("카메라 정리 중...")
        for cam in self.cameras:
            try:
                cam['actor'].destroy()
                rospy.loginfo(f"카메라 삭제됨: {cam['name']}")
            except Exception as e:
                rospy.logerr(f"카메라 삭제 실패 ({cam['name']}): {e}")
    
    def run(self):
        """메인 루프 실행"""
        rospy.loginfo("카메라 스트림 노드 실행 중... (Ctrl+C로 종료)")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("종료 신호 수신")
        finally:
            self.destroy()


def main():
    try:
        node = CarlaCameraStream()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

