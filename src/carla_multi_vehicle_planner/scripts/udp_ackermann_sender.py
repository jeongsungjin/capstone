#!/usr/bin/env python3

import math
import socket
import struct

import rospy
from ackermann_msgs.msg import AckermannDrive

FMT_INT = "!iiI"  # int angle, int speed, uint32 seq
FMT_FLOAT = "!fiI"  # float angle(rad), int speed, uint32 seq


class RCCarUdpSender:
    def __init__(self):
        rospy.init_node("rccar_udp_sender", anonymous=True)

        self.bind_ip = rospy.get_param("~bind_ip", "0.0.0.0")
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.num_vehicles = max(1, min(6, self.num_vehicles))

        # Packet mode
        self.send_angle_as_float = bool(rospy.get_param("~send_angle_as_float", False))
        self.pkt_fmt = FMT_FLOAT if self.send_angle_as_float else FMT_INT

        # Deterministic round-robin sender
        self.send_hz = float(rospy.get_param("~send_hz", 30.0))
        self.drop_stale_sec = float(rospy.get_param("~drop_stale_sec", 0.3))
        self.send_zero_on_missing = bool(rospy.get_param("~send_zero_on_missing", False))
        self.log_throttle_sec = float(rospy.get_param("~log_throttle_sec", 0.5))

        # Per-vehicle state
        self.vehicles = {}
        self.cache = {}  # role -> {"steer": float, "speed": float, "stamp": rospy.Time}
        self.roles = []

        for idx in range(1, self.num_vehicles + 1):
            role = f"ego_vehicle_{idx}"
            base = f"~vehicles/{role}"
            topic = rospy.get_param(f"{base}/topic", f"/carla/{role}/vehicle_control_cmd")
            dest_ip = rospy.get_param(f"{base}/dest_ip", "127.0.0.1")
            dest_port = int(rospy.get_param(f"{base}/dest_port", 5555))
            angle_scale = float(rospy.get_param(f"{base}/angle_scale", 1.0))
            angle_clip = int(rospy.get_param(f"{base}/angle_clip", 50))
            angle_min_abs = int(rospy.get_param(f"{base}/angle_min_abs", 0))
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
                "scale": angle_scale,
                "clip": angle_clip,
                "min_abs": angle_min_abs,
                "invert": angle_invert,
                "center_rad": angle_center_rad,
                "speed_scale": speed_scale,
                "speed_min_abs": speed_min_abs,
                "force_min_speed_on_zero": force_min_speed_on_zero,
                "zero_speed_value": zero_speed_value,
                "seq": 0,
            }
            rospy.Subscriber(topic, AckermannDrive, self._cb, callback_args=role, queue_size=10)
            # rospy.loginfo(
            #     f"[RC-UDP] ready: {role} topic={topic} -> {dest_ip}:{dest_port} "
            #     f"mode={'float(rad)' if self.send_angle_as_float else 'int(scaled)'} scale={angle_scale} clip={angle_clip} min_abs={angle_min_abs} invert={angle_invert}"
            # )

            # init cache and order
            self.cache[role] = {"steer": 0.0, "speed": 0.0, "stamp": rospy.Time(0)}
            self.roles.append(role)

        # stable order: ego_vehicle_1..N
        self.roles.sort(key=lambda r: int(r.split("_")[-1]) if r.split("_")[-1].isdigit() else 999)
        self._rr_index = 0
        self._last_log = {r: rospy.Time(0) for r in self.roles}

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(f"RC UDP sender bound on {self.bind_ip} for {len(self.vehicles)} vehicles")
        # Start deterministic sender: send one role per tick to evenly space packets
        if self.send_hz > 0.0 and len(self.roles) > 0:
            tick_hz = self.send_hz * len(self.roles)
            rospy.Timer(rospy.Duration(1.0 / tick_hz), self._tick)

    def _cb(self, msg: AckermannDrive, role: str):
        # Cache only; sending happens in _tick in fixed order/time
        self.cache[role] = {
            "steer": float(msg.steering_angle),
            "speed": float(msg.speed),
            "stamp": rospy.Time.now(),
        }

    def _tick(self, _evt):
        now = rospy.Time.now()
        if not self.roles:
            return
        role = self.roles[self._rr_index % len(self.roles)]
        self._rr_index += 1
        v = self.vehicles.get(role)
        if v is None:
            return
        cached = self.cache.get(role) or {}
        steer = float(cached.get("steer", 0.0))
        speed = float(cached.get("speed", 0.0))
        stamp = cached.get("stamp", rospy.Time(0))
        # Stale handling
        if (now - stamp).to_sec() > self.drop_stale_sec:
            if self.send_zero_on_missing:
                steer = 0.0
                speed = 0.0
            # otherwise keep last known

        # Build and send packet
        if self.send_angle_as_float:
            # Apply center offset, scaling, and optional invert for float mode too
            send_angle = (steer + float(v["center_rad"])) * float(v["scale"])
            if bool(v["invert"]):
                send_angle = -send_angle
            sp = speed * float(v["speed_scale"])
            xy_speed = int(round(sp))
            if xy_speed != 0 and abs(xy_speed) < int(v["speed_min_abs"]):
                xy_speed = (1 if xy_speed > 0 else -1) * int(v["speed_min_abs"])
            if xy_speed == 0 and bool(v.get("force_min_speed_on_zero", False)):
                z = int(v.get("zero_speed_value", 1)) or 1
                xy_speed = (1 if sp >= 0.0 else -1) * abs(z)
            xy_speed = max(-50, min(50, xy_speed))
            # Per-role throttle
            if self.log_throttle_sec <= 0.0 or (now - self._last_log.get(role, rospy.Time(0))).to_sec() >= self.log_throttle_sec:
                rospy.loginfo(f"[RC-UDP][{role}] angle(rad)={send_angle:.4f}, speed={xy_speed}")
                self._last_log[role] = now
            pkt = struct.pack(self.pkt_fmt, float(send_angle), xy_speed, int(v["seq"]) & 0xFFFFFFFF)
        else:
            raw = (steer + float(v["center_rad"])) * float(v["scale"])
            xy_angle = int(round(raw) * 2.0)
            if 0 < abs(xy_angle) < int(v["min_abs"]):
                xy_angle = int(v["min_abs"]) if xy_angle > 0 else -int(v["min_abs"])
            if bool(v["invert"]):
                xy_angle = -xy_angle
            xy_angle = max(-int(v["clip"]), min(int(v["clip"]), xy_angle))
            sp = speed * float(v["speed_scale"])
            xy_speed = int(round(sp))
            if xy_speed != 0 and abs(xy_speed) < int(v["speed_min_abs"]):
                xy_speed = (1 if xy_speed > 0 else -1) * int(v["speed_min_abs"])
            if xy_speed == 0 and bool(v.get("force_min_speed_on_zero", False)):
                z = int(v.get("zero_speed_value", 1)) or 1
                xy_speed = (1 if sp >= 0.0 else -1) * abs(z)
            xy_speed = max(-50, min(50, xy_speed))
            if self.log_throttle_sec <= 0.0 or (now - self._last_log.get(role, rospy.Time(0))).to_sec() >= self.log_throttle_sec:
                rospy.loginfo(f"[RC-UDP][{role}] angle={xy_angle}, speed={xy_speed}")
                self._last_log[role] = now
            pkt = struct.pack(self.pkt_fmt, xy_angle, xy_speed, int(v["seq"]) & 0xFFFFFFFF)

        try:
            v["sock"].sendto(pkt, v["dest"])
            v["seq"] = int(v["seq"]) + 1
        except Exception as exc:
            rospy.logwarn_throttle(1.0, f"[RC-UDP][{role}] send failed: {exc}")

    def _shutdown(self):
        try:
            for role, v in self.vehicles.items():
                for _ in range(5):
                    # Send zeros using current packet format
                    if self.send_angle_as_float:
                        pkt = struct.pack(self.pkt_fmt, float(0.0), 0, int(v["seq"]) & 0xFFFFFFFF)
                    else:
                        pkt = struct.pack(self.pkt_fmt, int(0), 0, int(v["seq"]) & 0xFFFFFFFF)
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