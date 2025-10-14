#!/usr/bin/env python3

import math
import socket
import struct

import rospy
from ackermann_msgs.msg import AckermannDrive

FMT = "!iiI"  # int angle, int speed, uint32 seq


class RCCarUdpSender:
    def __init__(self):
        rospy.init_node("rccar_udp_sender", anonymous=True)

        dest_ip = rospy.get_param("~dest_ip", "127.0.0.1")
        dest_port = int(rospy.get_param("~dest_port", 5555))
        bind_ip = rospy.get_param("~bind_ip", "0.0.0.0")
        topic = rospy.get_param("~topic", "/carla/ego_vehicle_1/vehicle_control_cmd")

        self.scale = float(rospy.get_param("~angle_scale", 500.0))
        self.clip = int(rospy.get_param("~angle_clip", 50))
        self.min_abs = int(rospy.get_param("~angle_min_abs", 8))
        self.invert = bool(rospy.get_param("~angle_invert", False))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)
        except OSError:
            pass
        self.sock.bind((bind_ip, 0))
        self.dest = (dest_ip, dest_port)
        self.seq = 0

        rospy.Subscriber(topic, AckermannDrive, self._cb, queue_size=10)
        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(f"RC UDP sender ready: bind={bind_ip}:0 -> dest={dest_ip}:{dest_port}, topic={topic}")

    def _cb(self, msg: AckermannDrive):
        steer = float(msg.steering_angle)
        speed = float(msg.speed)

        xy_angle = int(round(steer * self.scale))
        if 0 < abs(xy_angle) < self.min_abs:
            xy_angle = self.min_abs if xy_angle > 0 else -self.min_abs
        if self.invert:
            xy_angle = -xy_angle
        xy_angle = max(-self.clip, min(self.clip, xy_angle))

        xy_speed = int(round(speed))
        xy_speed = max(-50, min(50, xy_speed))

        rospy.loginfo_throttle(0.2, f"[RC-UDP] angle={xy_angle}, speed={xy_speed}, steer={steer:.3f}rad ({math.degrees(steer):.1f}°)")

        pkt = struct.pack(FMT, xy_angle, xy_speed, self.seq & 0xFFFFFFFF)
        self.sock.sendto(pkt, self.dest)
        self.seq += 1

    def _shutdown(self):
        try:
            for _ in range(5):
                pkt = struct.pack(FMT, 0, 0, self.seq & 0xFFFFFFFF)
                self.sock.sendto(pkt, self.dest)
                self.seq += 1
        except Exception:
            pass


def main():
    try:
        RCCarUdpSender()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


