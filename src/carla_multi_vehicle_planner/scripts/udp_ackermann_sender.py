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

        self.bind_ip = rospy.get_param("~bind_ip", "0.0.0.0")
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.num_vehicles = max(1, min(6, self.num_vehicles))

        # Per-vehicle state
        self.vehicles = {}

        for idx in range(1, self.num_vehicles + 1):
            role = f"ego_vehicle_{idx}"
            base = f"~vehicles/{role}"
            topic = rospy.get_param(f"{base}/topic", f"/carla/{role}/vehicle_control_cmd")
            dest_ip = rospy.get_param(f"{base}/dest_ip", "127.0.0.1")
            dest_port = int(rospy.get_param(f"{base}/dest_port", 5555))
            angle_scale = float(rospy.get_param(f"{base}/angle_scale", 500.0))
            angle_clip = int(rospy.get_param(f"{base}/angle_clip", 50))
            angle_min_abs = int(rospy.get_param(f"{base}/angle_min_abs", 8))
            angle_invert = bool(rospy.get_param(f"{base}/angle_invert", False))

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)
            except OSError:
                pass
            sock.bind((self.bind_ip, 0))

            self.vehicles[role] = {
                "sock": sock,
                "dest": (dest_ip, dest_port),
                "scale": angle_scale,
                "clip": angle_clip,
                "min_abs": angle_min_abs,
                "invert": angle_invert,
                "seq": 0,
            }
            rospy.Subscriber(topic, AckermannDrive, self._cb, callback_args=role, queue_size=10)
            rospy.loginfo(
                f"[RC-UDP] ready: {role} topic={topic} -> {dest_ip}:{dest_port} "
                f"scale={angle_scale} clip={angle_clip} min_abs={angle_min_abs} invert={angle_invert}"
            )

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(f"RC UDP sender bound on {self.bind_ip} for {len(self.vehicles)} vehicles")

    def _cb(self, msg: AckermannDrive, role: str):
        v = self.vehicles.get(role)
        if v is None:
            return
        steer = float(msg.steering_angle)
        speed = float(msg.speed)

        xy_angle = int(round(steer * float(v["scale"])))
        if 0 < abs(xy_angle) < int(v["min_abs"]):
            xy_angle = int(v["min_abs"]) if xy_angle > 0 else -int(v["min_abs"])
        if bool(v["invert"]):
            xy_angle = -xy_angle
        xy_angle = max(-int(v["clip"]), min(int(v["clip"]), xy_angle))

        xy_speed = int(round(speed))
        xy_speed = max(-50, min(50, xy_speed))

        rospy.loginfo_throttle(
            0.2,
            f"[RC-UDP][{role}] angle={xy_angle}, speed={xy_speed}, steer={steer:.3f}rad ({math.degrees(steer):.1f}°)",
        )

        pkt = struct.pack(FMT, xy_angle, xy_speed, int(v["seq"]) & 0xFFFFFFFF)
        try:
            v["sock"].sendto(pkt, v["dest"])
            v["seq"] = int(v["seq"]) + 1
        except Exception as exc:
            rospy.logwarn_throttle(1.0, f"[RC-UDP][{role}] send failed: {exc}")

    def _shutdown(self):
        try:
            for role, v in self.vehicles.items():
                for _ in range(5):
                    pkt = struct.pack(FMT, 0, 0, int(v["seq"]) & 0xFFFFFFFF)
                    try:
                        v["sock"].sendto(pkt, v["dest"])
                    except Exception:
                        pass
                    v["seq"] = int(v["seq"]) + 1
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