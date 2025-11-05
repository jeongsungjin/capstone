#!/usr/bin/env python3

import math
import socket
import struct
from typing import Dict, Tuple

import rospy
from ackermann_msgs.msg import AckermannDrive


FMT = "!iiI"


class UdpAckermannSender:
    def __init__(self) -> None:
        rospy.init_node("udp_ackermann_sender", anonymous=True)

        # Networking
        self.dest_ip = rospy.get_param("~dest_ip", "10.30.78.103")
        self.dest_port = int(rospy.get_param("~dest_port", 5555))
        self.bind_ip = rospy.get_param("~bind_ip", "0.0.0.0")
        self.send_buf_kb = int(rospy.get_param("~send_buf_kb", 256))

        # Scaling/tuning
        self.xy_scale = float(rospy.get_param("~xy_scale", 500.0))
        self.xy_clip = int(rospy.get_param("~xy_clip", 50))
        self.xy_min_abs = int(rospy.get_param("~xy_min_abs", 8))
        self.xy_invert = bool(rospy.get_param("~xy_invert", False))
        self.xy_hold_frames = int(rospy.get_param("~xy_hold_frames", 2))

        # Vehicle roles to subscribe
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 1))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buf_kb * 1024)
        except OSError:
            pass
        self.sock.bind((self.bind_ip, 0))
        self.dest = (self.dest_ip, self.dest_port)

        self.seq = 0
        self.last_angle_by_role: Dict[str, int] = {}
        self.hold_count_by_role: Dict[str, int] = {}

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/carla/{role}/vehicle_control_cmd"
            rospy.Subscriber(topic, AckermannDrive, self._make_cb(role), queue_size=1)

        rospy.on_shutdown(self._shutdown)

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _make_cb(self, role: str):
        def _callback(msg: AckermannDrive) -> None:
            steer = float(msg.steering_angle)
            speed = float(msg.speed)

            xy_angle = int(round(steer * self.xy_scale))
            if 0 < abs(xy_angle) < self.xy_min_abs:
                xy_angle = self.xy_min_abs if xy_angle > 0 else -self.xy_min_abs
            if self.xy_invert:
                xy_angle = -xy_angle
            xy_angle = max(-self.xy_clip, min(self.xy_clip, xy_angle))

            last_angle = self.last_angle_by_role.get(role, 0)
            hold_count = self.hold_count_by_role.get(role, 0)
            if xy_angle == 0 and last_angle != 0 and hold_count < self.xy_hold_frames:
                xy_angle = last_angle
                hold_count += 1
            else:
                last_angle = xy_angle
                hold_count = 0
            self.last_angle_by_role[role] = last_angle
            self.hold_count_by_role[role] = hold_count

            xy_speed = int(round(speed))
            xy_speed = max(-50, min(50, xy_speed))

            rospy.loginfo_throttle(0.2, f"[UDP:{role}] angle={xy_angle}, speed={xy_speed}, steer={steer:.3f}rad ({math.degrees(steer):.1f}°)")
            pkt = struct.pack(FMT, xy_angle, xy_speed, self.seq & 0xFFFFFFFF)
            try:
                self.sock.sendto(pkt, self.dest)
                self.seq += 1
            except OSError as exc:
                rospy.logwarn_throttle(1.0, f"UDP send failed: {exc}")

        return _callback

    def _shutdown(self) -> None:
        try:
            for _ in range(5):
                pkt = struct.pack(FMT, 0, 0, self.seq & 0xFFFFFFFF)
                self.sock.sendto(pkt, self.dest)
                self.seq += 1
        except Exception:
            pass


if __name__ == "__main__":
    try:
        UdpAckermannSender()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


