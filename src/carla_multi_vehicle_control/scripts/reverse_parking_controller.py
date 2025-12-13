#!/usr/bin/env python3
"""
간단 후진 주차 실행 노드:
- /parking_target_{role} (PoseStamped) 구독: 플래너가 저전압 시 지정한 주차 목표
- /carla/{role}/odometry 구독: 현재 상태
- 후진 직선 경로를 따라 CARLA VehicleControl을 직접 적용(reverse=True)
"""
import math
from typing import Optional

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

try:
    import carla  # type: ignore
except Exception as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


def quaternion_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ReverseParkingController:
    def __init__(self) -> None:
        rospy.init_node("reverse_parking_controller", anonymous=True)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.role = rospy.get_param("~role", "ego_vehicle_1")
        self.target_speed = float(rospy.get_param("~target_speed", 3.0))  # m/s, magnitude for reverse
        self.stop_distance = float(rospy.get_param("~stop_distance", 0.5))
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        timeout = float(rospy.get_param("~carla_timeout", 10.0))

        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.vehicle: Optional[carla.Actor] = None

        self.target: Optional[PoseStamped] = None
        self.last_odom: Optional[Odometry] = None

        rospy.Subscriber(f"/parking_target_{self.role}", PoseStamped, self._target_cb, queue_size=1)
        rospy.Subscriber(f"/carla/{self.role}/odometry", Odometry, self._odom_cb, queue_size=5)

    def _target_cb(self, msg: PoseStamped) -> None:
        self.target = msg
        rospy.loginfo(f"{self.role}: received parking target ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg
        if self.vehicle is None:
            self.vehicle = self._find_vehicle(self.role)

    def _find_vehicle(self, role: str) -> Optional[carla.Actor]:
        actors = self.world.get_actors().filter("vehicle.*")
        for actor in actors:
            if actor.attributes.get("role_name", "") == role:
                return actor
        return None

    def _compute_reverse_control(self) -> Optional[carla.VehicleControl]:
        if self.vehicle is None or self.last_odom is None or self.target is None:
            return None
        pos = self.last_odom.pose.pose.position
        yaw = quaternion_to_yaw(self.last_odom.pose.pose.orientation)
        tx = self.target.pose.position.x
        ty = self.target.pose.position.y
        dx = tx - pos.x
        dy = ty - pos.y
        dist = math.hypot(dx, dy)
        if dist <= self.stop_distance:
            ctrl = carla.VehicleControl()
            ctrl.brake = 1.0
            ctrl.throttle = 0.0
            ctrl.reverse = True
            return ctrl

        # For reverse, we want to steer so that backing trajectory points to target
        target_angle = math.atan2(dy, dx)
        # Heading when driving backward is yaw + pi
        heading_back = yaw + math.pi
        angle_err = target_angle - heading_back
        while angle_err > math.pi:
            angle_err -= 2.0 * math.pi
        while angle_err < -math.pi:
            angle_err += 2.0 * math.pi

        steer = math.atan2(2.0 * 1.74 * math.sin(angle_err), max(0.5, dist))  # wheelbase 1.74 default
        steer = max(-0.5, min(0.5, steer))
        # Reverse driving: steering response is inverted
        steer_cmd = -steer

        ctrl = carla.VehicleControl()
        ctrl.reverse = True
        ctrl.throttle = 0.4
        ctrl.brake = 0.0
        ctrl.steer = max(-1.0, min(1.0, steer_cmd / 0.5))
        return ctrl

    def spin(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            ctrl = self._compute_reverse_control()
            if ctrl is not None and self.vehicle is not None:
                self.vehicle.apply_control(ctrl)
            rate.sleep()


if __name__ == "__main__":
    try:
        node = ReverseParkingController()
        node.spin()
    except Exception as e:
        rospy.logfatal(f"reverse_parking_controller crashed: {e}")
        raise

