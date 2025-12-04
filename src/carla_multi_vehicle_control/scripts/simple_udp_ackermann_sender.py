#!/usr/bin/env python3

import socket
import struct
from typing import Dict, List

import rospy
from ackermann_msgs.msg import AckermannDrive

FMT_FLOAT = "!fiI"  # float angle(rad), int speed, uint32 seq


class SimpleUdpAckermannSender:
    """
    최소 기능 UDP 송신기:
    - 각 차량의 AckermannDrive 구독(/carla/{role}/vehicle_control_cmd)
    - angle(rad) float, speed int 형식으로 per-vehicle IP:PORT로 송신
    - 보정용 파라미터: angle_center_rad, angle_scale, angle_invert, speed_scale, speed_min_abs
    - 기타 복잡한 옵션 제거
    """

    def __init__(self) -> None:
        rospy.init_node("rccar_udp_sender", anonymous=True)

        self.bind_ip = str(rospy.get_param("~bind_ip", "0.0.0.0"))
        self.num_vehicles = max(1, min(6, int(rospy.get_param("~num_vehicles", 3))))
        self.send_hz = float(rospy.get_param("~send_hz", 30.0))
        self.drop_stale_sec = float(rospy.get_param("~drop_stale_sec", 0.3))
        self.send_zero_on_missing = bool(rospy.get_param("~send_zero_on_missing", False))
        self.log_throttle_sec = float(rospy.get_param("~log_throttle_sec", 0.5))

        self.vehicles: Dict[str, Dict] = {}
        self.cache: Dict[str, Dict] = {}
        self.roles: List[str] = []

        for idx in range(1, self.num_vehicles + 1):
            role = f"ego_vehicle_{idx}"
            base = f"~vehicles/{role}"
            topic = rospy.get_param(f"{base}/topic", f"/carla/{role}/vehicle_control_cmd")
            dest_ip = str(rospy.get_param(f"{base}/dest_ip", "127.0.0.1"))
            dest_port = int(rospy.get_param(f"{base}/dest_port", 5555))
            angle_scale = float(rospy.get_param(f"{base}/angle_scale", 1.0))
            angle_invert = bool(rospy.get_param(f"{base}/angle_invert", False))
            angle_center_rad = float(rospy.get_param(f"{base}/angle_center_rad", 0.0))
            speed_scale = float(rospy.get_param(f"{base}/speed_scale", 1.0))
            speed_min_abs = int(rospy.get_param(f"{base}/speed_min_abs", 1))
            force_min_speed_on_zero = bool(rospy.get_param(f"{base}/force_min_speed_on_zero", False))
            zero_speed_value = int(rospy.get_param(f"{base}/zero_speed_value", 1))

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)
            except OSError:
                pass
            sock.bind((self.bind_ip, 0))

            self.vehicles[role] = {
                "sock": sock,
                "dest": (dest_ip, dest_port),
                "angle_scale": angle_scale,
                "angle_invert": angle_invert,
                "angle_center_rad": angle_center_rad,
                "speed_scale": speed_scale,
                "speed_min_abs": speed_min_abs,
                "force_min_speed_on_zero": force_min_speed_on_zero,
                "zero_speed_value": zero_speed_value,
                "seq": 0,
                "last_log": rospy.Time(0),
            }
            rospy.Subscriber(topic, AckermannDrive, self._cb, callback_args=role, queue_size=10)
            self.cache[role] = {"steer": 0.0, "speed": 0.0, "stamp": rospy.Time(0)}
            self.roles.append(role)

        self.roles.sort(key=lambda r: int(r.split("_")[-1]) if r.split("_")[-1].isdigit() else 999)
        self._rr_index = 0
        rospy.on_shutdown(self._shutdown)

        if self.send_hz > 0.0 and len(self.roles) > 0:
            tick_hz = self.send_hz * len(self.roles)
            rospy.Timer(rospy.Duration(1.0 / tick_hz), self._tick)

        rospy.loginfo(f"[RC-UDP] bound on {self.bind_ip} for {len(self.roles)} vehicles")

    def _cb(self, msg: AckermannDrive, role: str) -> None:
        self.cache[role] = {"steer": float(msg.steering_angle), "speed": float(msg.speed), "stamp": rospy.Time.now()}

    def _tick(self, _evt) -> None:
        if not self.roles:
            return
        role = self.roles[self._rr_index % len(self.roles)]
        self._rr_index += 1
        v = self.vehicles.get(role)
        if v is None:
            return
        now = rospy.Time.now()
        cached = self.cache.get(role) or {}
        steer = float(cached.get("steer", 0.0))
        speed = float(cached.get("speed", 0.0))
        stamp = cached.get("stamp", rospy.Time(0))
        if (now - stamp).to_sec() > self.drop_stale_sec and self.send_zero_on_missing:
            steer = 0.0
            speed = 0.0

        send_angle = (steer + float(v["angle_center_rad"])) * float(v["angle_scale"])
        if bool(v["angle_invert"]):
            send_angle = -send_angle
        sp = speed * float(v["speed_scale"])
        xy_speed = int(round(sp))
        if xy_speed != 0 and abs(xy_speed) < int(v["speed_min_abs"]):
            xy_speed = (1 if xy_speed > 0 else -1) * int(v["speed_min_abs"])
        if xy_speed == 0 and bool(v.get("force_min_speed_on_zero", False)):
            z = int(v.get("zero_speed_value", 1)) or 1
            xy_speed = (1 if sp >= 0.0 else -1) * abs(z)
        xy_speed = max(-50, min(50, xy_speed))

        last_log = v.get("last_log", rospy.Time(0))
        if self.log_throttle_sec <= 0.0 or (now - last_log).to_sec() >= self.log_throttle_sec:
            di, dp = v.get("dest", ("", 0))
            rospy.loginfo(f"[RC-UDP][{role}] angle(rad)={send_angle:.4f}, speed={xy_speed} -> {di}:{dp}")
            v["last_log"] = now

        pkt = struct.pack(FMT_FLOAT, float(send_angle), int(xy_speed), int(v["seq"]) & 0xFFFFFFFF)
        try:
            v["sock"].sendto(pkt, v["dest"])
            v["seq"] = int(v["seq"]) + 1
        except Exception as exc:
            rospy.logwarn_throttle(1.0, f"[RC-UDP][{role}] send failed: {exc}")

    def _shutdown(self) -> None:
        for role, v in self.vehicles.items():
            try:
                for _ in range(3):
                    pkt = struct.pack(FMT_FLOAT, float(0.0), int(0), int(v["seq"]) & 0xFFFFFFFF)
                    v["sock"].sendto(pkt, v["dest"])
                    v["seq"] = int(v["seq"]) + 1
            except Exception:
                pass


if __name__ == "__main__":
    try:
        SimpleUdpAckermannSender()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleUdpAckermannSender crashed: {e}")
        raise



