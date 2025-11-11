#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA: 단일 카메라 스폰 스크립트

특정 좌표(x, y, z)와 회전(pitch, yaw, roll)을 받아 CARLA에 카메라(sensor.camera.rgb)를 스폰하고
ROS 토픽으로 이미지를 퍼블리시합니다.

ROS private parameters (예시: rosrun carla_camera_stream spawn_single_camera.py _x:=10 _y:=20 _z:=7 _yaw:=90):
- ~host (str, default 'localhost')
- ~port (int, default 2000)
- ~name (str, default 'camera_1')
- ~x, ~y, ~z (float, default 0.0, 0.0, 7.0)
- ~pitch, ~yaw, ~roll (float, default -20.0, 0.0, 0.0)
- ~image_width, ~image_height (int, default 1280, 720)
- ~fov (float, default 90.0)
- ~sensor_tick (float, default 0.0333)
- ~attach_to_actor_id (int, optional)  # 지정 시 해당 액터에 상대 좌표로 부착

퍼블리시 토픽:
- /carla/<name>/image_raw (sensor_msgs/Image)
- /carla/<name>/camera_info (sensor_msgs/CameraInfo, latch)
"""

import math
import sys
from typing import Optional

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

try:
    import carla
except ImportError:
    print("[ERROR] CARLA Python API를 import할 수 없습니다. PYTHONPATH를 확인하세요.")
    sys.exit(1)


class SingleCameraSpawner:
    def __init__(self) -> None:
        rospy.init_node('carla_spawn_single_camera', anonymous=True)

        # Params
        self.host: str = rospy.get_param('/host', 'localhost')
        self.port: int = int(rospy.get_param('/port', 2000))
        self.width: int = int(rospy.get_param('/image_width', 1280))
        self.height: int = int(rospy.get_param('/image_height', 720))
        self.fov: float = float(rospy.get_param('/fov', 90.0))
        
        self.name: str = rospy.get_param('/topic_name_prefix', 'camera_1')

        self.x: float = float(rospy.get_param('~x', 0.0))
        self.y: float = float(rospy.get_param('~y', 0.0))
        self.z: float = float(rospy.get_param('~z', 15))

        self.pitch: float = float(rospy.get_param('~pitch', -45.0))
        self.yaw: float = float(rospy.get_param('~yaw', 0.0))
        self.roll: float = float(rospy.get_param('~roll', 0.0))

        self.camera_id = rospy.get_param('~camera_id', 0)
        
        self.sensor_tick: float = float(rospy.get_param('sensor_tick', 1.0/30.0))
        self.attach_to_actor_id: Optional[int] = None
        if rospy.has_param('attach_to_actor_id'):
            try:
                self.attach_to_actor_id = int(rospy.get_param('attach_to_actor_id'))
            except Exception:
                self.attach_to_actor_id = None

        # ROS publishers
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher(f'{self.name}{self.camera_id}/image_raw', Image, queue_size=10)
        self.info_pub = rospy.Publisher(f'{self.name}{self.camera_id}/camera_info', CameraInfo, queue_size=10, latch=True)
        self.info_msg = self._build_camera_info()
        self.info_pub.publish(self.info_msg)

        # CARLA connection
        self.client = None
        self.world = None
        self.actor = None

        self._connect_and_spawn()

    def _build_camera_info(self) -> CameraInfo:
        info = CameraInfo()
        info.width = self.width
        info.height = self.height
        info.distortion_model = 'plumb_bob'
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]

        fx = self.width / (2.0 * math.tan(math.radians(self.fov) / 2.0))
        fy = fx
        cx = self.width / 2.0
        cy = self.height / 2.0

        info.K = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.P = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.header.frame_id = f"{self.name}_optical"
        return info

    def _connect_and_spawn(self) -> None:
        try:
            rospy.loginfo(f"CARLA 서버 접속: {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            bp_lib = self.world.get_blueprint_library()

            bp = bp_lib.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(self.width))
            bp.set_attribute('image_size_y', str(self.height))
            bp.set_attribute('fov', str(self.fov))
            bp.set_attribute('sensor_tick', str(self.sensor_tick))

            transform = carla.Transform(
                carla.Location(x=self.x, y=self.y, z=self.z),
                carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll)
            )

            attach_parent = None
            if self.attach_to_actor_id is not None:
                try:
                    attach_parent = self.world.get_actor(self.attach_to_actor_id)
                    rospy.loginfo(f"부착 대상 액터 ID={self.attach_to_actor_id} 발견. 상대 좌표로 부착합니다.")
                except Exception:
                    rospy.logwarn(f"attach_to_actor_id={self.attach_to_actor_id} 액터를 찾지 못했습니다. 월드 좌표에 스폰합니다.")

            if attach_parent is not None:
                self.actor = self.world.spawn_actor(bp, transform, attach_to=attach_parent)
            else:
                self.actor = self.world.spawn_actor(bp, transform)

            self.actor.listen(lambda image: self._image_callback(image))
            rospy.loginfo(
                f"카메라 스폰 완료: {self.name} @ (x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}), "
                f"pitch={self.pitch:.1f}, yaw={self.yaw:.1f}, roll={self.roll:.1f} [ID: {self.actor.id}]"
            )
        except Exception as e:
            rospy.logerr(f"CARLA 연결/스폰 실패: {e}")
            self.destroy()
            sys.exit(1)

    def _image_callback(self, image) -> None:
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = f"{self.name}_optical"
            self.image_pub.publish(msg)
        except Exception as e:
            rospy.logerr(f"이미지 콜백 에러: {e}")

    def run(self) -> None:
        rospy.loginfo("단일 카메라 노드 실행 중... (Ctrl+C로 종료)")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
        finally:
            self.destroy()

    def destroy(self) -> None:
        if self.actor is not None:
            try:
                self.actor.destroy()
                rospy.loginfo("카메라 삭제 완료")
            except Exception as e:
                rospy.logwarn(f"카메라 삭제 실패: {e}")


def main() -> None:
    node = SingleCameraSpawner()
    node.run()


if __name__ == '__main__':
    main()
