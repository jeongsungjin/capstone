#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA: N대 카메라 스폰 및 이미지 퍼블리시 노드

기본값
- 카메라 수: 1대
- 위치: (0, 0, 7)
- 회전: pitch=-30, yaw=45, roll=0

ROS 파라미터로 확장 가능:
- ~host, ~port, ~image_width, ~image_height, ~fov
- ~num_cameras (int)
- ~positions (list of dict: {x,y,z,pitch,yaw,roll,name})
- ~default_x, ~default_y, ~default_z, ~default_pitch, ~default_yaw, ~default_roll
"""

import math
import sys
from typing import List, Dict

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

try:
    import carla
except ImportError:
    rospy.logerr("CARLA Python API를 import할 수 없습니다. PYTHONPATH를 확인하세요.")
    sys.exit(1)


class MultiCameraSpawner:
    def __init__(self) -> None:
        rospy.init_node('carla_camera_spawner', anonymous=True)

        # ROS 파라미터
        self.host = rospy.get_param('~host', 'localhost')
        self.port = rospy.get_param('~port', 2000)
        self.image_width = rospy.get_param('~image_width', 1280)
        self.image_height = rospy.get_param('~image_height', 720)
        self.fov = rospy.get_param('~fov', 90.0)

        self.num_cameras = int(rospy.get_param('~num_cameras', 1))

        self.default_x = float(rospy.get_param('~default_x', 0.0))
        self.default_y = float(rospy.get_param('~default_y', -4.0))
        self.default_z = float(rospy.get_param('~default_z', 9.0))
        self.default_pitch = float(rospy.get_param('~default_pitch', -30.0))
        self.default_yaw = float(rospy.get_param('~default_yaw', -65.0))
        self.default_roll = float(rospy.get_param('~default_roll', 0.0))

        # 코너 기반 자동 배치 파라미터 (좌상, 우상, 우하, 좌하 순)
        # boundary_corners/corners/cover_quad 중 첫 번째로 발견되는 값을 사용
        self.boundary_corners = rospy.get_param(
            '~boundary_corners', rospy.get_param('~corners', rospy.get_param('~cover_quad', []))
        )
        self.corner_z = float(rospy.get_param('~corner_z', 6.0))
        self.corner_pitch = float(rospy.get_param('~corner_pitch', -40.0))
        self.corner_roll = float(rospy.get_param('~corner_roll', 0.0))

        # 중앙점(yaw 수렴) 설정: center_x/center_y 우선, 없으면 boundary_corners로 평균
        self.center_x_param = rospy.get_param('~center_x', None)
        self.center_y_param = rospy.get_param('~center_y', None)
        self.force_center_yaw = rospy.get_param('~force_center_yaw', True)

        # positions는 list[dict] 형태 기대 ({x,y,z,pitch,yaw,roll,name})
        self.positions = rospy.get_param('~positions', [])
        if not isinstance(self.positions, list):
            rospy.logwarn("~positions 파라미터가 리스트가 아닙니다. 기본 위치를 사용합니다.")
            self.positions = []

        # 퍼블리셔/브릿지
        self.bridge = CvBridge()
        self.image_pubs: Dict[str, rospy.Publisher] = {}
        self.camerainfo_pubs: Dict[str, rospy.Publisher] = {}
        self.camera_info_msgs: Dict[str, CameraInfo] = {}

        # CARLA 관련
        self.client = None
        self.world = None
        self.actors: List[Dict[str, object]] = []

        self._connect_and_spawn()

    def _resolve_camera_specs(self) -> List[Dict[str, object]]:
        """카메라 스펙 목록 생성.
        우선순위: boundary_corners(또는 동의어) → positions → 기본값 num_cameras
        """
        specs: List[Dict[str, object]] = []
        # 중앙점 해석 (가능하다면 미리 계산)
        center_xy = self._get_center_xy()
        # 1) 코너 기반 자동 배치: 좌상→우상→우하→좌하 순으로 제공된 4개 좌표
        if isinstance(self.boundary_corners, list) and len(self.boundary_corners) >= 4:
            # 4개까지만 사용 (초과분 무시)
            corners = self.boundary_corners[:4]
            try:
                cx = sum(float(c.get('x')) for c in corners) / 4.0
                cy = sum(float(c.get('y')) for c in corners) / 4.0
            except Exception:
                rospy.logwarn("boundary_corners 항목에 x,y 값이 올바르지 않습니다. positions 로직으로 대체합니다.")
                corners = []

            if corners:
                for idx, c in enumerate(corners, start=1):
                    x = float(c.get('x', 0.0))
                    y = float(c.get('y', 0.0))
                    z = float(c.get('z', self.corner_z))
                    pitch = float(c.get('pitch', self.corner_pitch))
                    roll = float(c.get('roll', self.corner_roll))
                    yaw = math.degrees(math.atan2(cy - y, cx - x))
                    specs.append({
                        'name': f'camera_{idx}',
                        'x': x,
                        'y': y,
                        'z': z,
                        'pitch': pitch,
                        'yaw': yaw,
                        'roll': roll,
                    })
                rospy.loginfo(
                    f"코너 기반 자동 배치 활성화 (center=({cx:.2f}, {cy:.2f}), z={self.corner_z:.1f}, pitch={self.corner_pitch:.1f})"
                )
                return specs

        # 2) 명시적 positions 사용
        if len(self.positions) > 0:
            for idx, p in enumerate(self.positions, start=1):
                name = p.get('name', f'camera_{idx}')
                x = float(p.get('x', self.default_x))
                y = float(p.get('y', self.default_y))
                z = float(p.get('z', self.default_z))
                pitch = float(p.get('pitch', self.default_pitch))
                roll = float(p.get('roll', self.default_roll))
                if self.force_center_yaw and center_xy is not None:
                    cx, cy = center_xy
                    yaw = math.degrees(math.atan2(cy - y, cx - x))
                else:
                    yaw = float(p.get('yaw', self.default_yaw))
                specs.append({
                    'name': name,
                    'x': x,
                    'y': y,
                    'z': z,
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                })
            if self.force_center_yaw and center_xy is not None:
                rospy.loginfo(f"positions 기반 배치에 중앙 수렴 yaw 적용 (center=({center_xy[0]:.2f}, {center_xy[1]:.2f}))")
        else:
            for idx in range(1, self.num_cameras + 1):
                specs.append({
                    'name': f'camera_{idx}',
                    'x': self.default_x,
                    'y': self.default_y,
                    'z': self.default_z,
                    'pitch': self.default_pitch,
                    'yaw': self.default_yaw,
                    'roll': self.default_roll,
                })
        return specs

    def _get_center_xy(self):
        """중앙점(cx, cy) 결정: center_x/center_y가 있으면 우선 사용, 없으면 코너 평균.
        유효한 중앙점을 계산하지 못하면 None 반환.
        """
        # 1) 명시적 center 파라미터
        try:
            if self.center_x_param is not None and self.center_y_param is not None:
                cx = float(self.center_x_param)
                cy = float(self.center_y_param)
                return (cx, cy)
        except Exception:
            rospy.logwarn("center_x/center_y 파라미터 파싱 실패. 코너 기반 계산으로 시도합니다.")

        # 2) 코너 평균
        try:
            if isinstance(self.boundary_corners, list) and len(self.boundary_corners) >= 4:
                corners = self.boundary_corners[:4]
                cx = sum(float(c.get('x')) for c in corners) / 4.0
                cy = sum(float(c.get('y')) for c in corners) / 4.0
                return (cx, cy)
        except Exception:
            rospy.logwarn("boundary_corners에서 중앙점 계산 실패")

        return None

    def _build_camera_info(self, cam_name: str) -> CameraInfo:
        info = CameraInfo()
        info.width = self.image_width
        info.height = self.image_height
        info.distortion_model = 'plumb_bob'
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]

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

    def _connect_and_spawn(self) -> None:
        try:
            rospy.loginfo(f"CARLA 서버에 연결 중... ({self.host}:{self.port})")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()

            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.image_width))
            camera_bp.set_attribute('image_size_y', str(self.image_height))
            camera_bp.set_attribute('fov', str(self.fov))
            camera_bp.set_attribute('sensor_tick', str(1.0/10.0))  # FPS 30으로 고정

            specs = self._resolve_camera_specs()

            # 퍼블리셔 준비 및 CameraInfo latch 퍼블리시
            for spec in specs:
                name = spec['name']
                self.image_pubs[name] = rospy.Publisher(
                    f'/carla/{name}/image_raw', Image, queue_size=10
                )
                self.camerainfo_pubs[name] = rospy.Publisher(
                    f'/carla/{name}/camera_info', CameraInfo, queue_size=10, latch=True
                )
                info = self._build_camera_info(name)
                self.camera_info_msgs[name] = info
                self.camerainfo_pubs[name].publish(info)

            # 스폰 및 리스너 등록
            for spec in specs:
                name = spec['name']
                transform = carla.Transform(
                    carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
                    carla.Rotation(pitch=spec['pitch'], yaw=spec['yaw'], roll=spec['roll'])
                )
                actor = self.world.spawn_actor(camera_bp, transform)
                self.actors.append({'actor': actor, 'name': name})
                actor.listen(lambda image, n=name: self._camera_callback(image, n))
                rospy.loginfo(
                    f"카메라 스폰: {name} @ ({spec['x']:.2f}, {spec['y']:.2f}, {spec['z']:.2f}), "
                    f"pitch={spec['pitch']:.1f}, yaw={spec['yaw']:.1f}, roll={spec['roll']:.1f} [ID: {actor.id}]"
                )

            rospy.loginfo(f"총 {len(self.actors)}대 스폰 완료")

        except Exception as e:
            rospy.logerr(f"CARLA 초기화/스폰 실패: {e}")
            self.destroy()
            sys.exit(1)

    def _camera_callback(self, image, cam_name: str) -> None:
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame_bgr = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

            msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = f"{cam_name}_optical"
            self.image_pubs[cam_name].publish(msg)
        except Exception as e:
            rospy.logerr(f"이미지 콜백 에러 ({cam_name}): {e}")

    def run(self) -> None:
        rospy.loginfo("카메라 스폰 노드 실행 중... (Ctrl+C로 종료)")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("종료 신호 수신")
        finally:
            self.destroy()

    def destroy(self) -> None:
        rospy.loginfo("카메라 액터 정리 중...")
        for item in self.actors:
            try:
                item['actor'].destroy()
                rospy.loginfo(f"삭제됨: {item['name']}")
            except Exception as e:
                rospy.logerr(f"삭제 실패 ({item['name']}): {e}")


def main() -> None:
    try:
        node = MultiCameraSpawner()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()


