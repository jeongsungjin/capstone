#!/usr/bin/env python3
import socket
from typing import Dict

import rospy
from carla_multi_vehicle_control.msg import TrafficLightPhase

UDP_PORT = 4210


class TrafficLightPublisher:
    def __init__(self) -> None:
        rospy.init_node("traffic_light_publisher", anonymous=True)

        # 교차로별 보드 IP (ROS 파라미터로 오버라이드 가능)
        self.ip_map: Dict[str, str] = {
            "A": rospy.get_param("~ip_A", "192.168.0.41"),
            "B": rospy.get_param("~ip_B", "192.168.0.44"),
            "C": rospy.get_param("~ip_C", "192.168.0.47"),
        }

        self.verbose: bool = bool(rospy.get_param("~verbose", False))

        # UDP 소켓
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        rospy.loginfo("=== traffic_light_publisher started ===")
        for iid, ip in self.ip_map.items():
            rospy.loginfo("  - intersection %s -> %s", iid, ip)

        # /traffic_phase 구독해서 들어오는 페이즈를 하드웨어로 중계
        rospy.Subscriber("/traffic_phase", TrafficLightPhase, self._phase_cb, queue_size=20)

        rospy.on_shutdown(self._on_shutdown)

    def _phase_cb(self, msg: TrafficLightPhase) -> None:
        iid = (msg.intersection_id or "").strip()  # "A", "B", "C"
        phase = (msg.phase or "").strip()          # "P1_MAIN_GREEN" 등

        if not iid or not phase:
            return

        ip = self.ip_map.get(iid)
        if not ip:
            rospy.logwarn("traffic_light_publisher: no IP configured for intersection_id=%s", iid)
            return

        # 하드웨어가 기대하는 문자열 형식: "A_P1_MAIN_GREEN" / "B_P2_YELLOW" / "C_P3_SIDE_GREEN" ...
        payload_str = f"{iid}_{phase}"
        payload = payload_str.encode("utf-8")

        try:
            self.sock.sendto(payload, (ip, UDP_PORT))
            if self.verbose:
                rospy.loginfo("UDP -> [%s] %s", ip, payload_str)
        except OSError as exc:
            rospy.logwarn("traffic_light_publisher: failed to send to %s (%s): %s", ip, payload_str, exc)

    def _on_shutdown(self) -> None:
        rospy.loginfo("traffic_light_publisher: sending 'finish' to all boards and closing socket...")
        for iid, ip in self.ip_map.items():
            try:
                self.sock.sendto(b"finish", (ip, UDP_PORT))
                rospy.loginfo("  finish sent to %s (iid=%s)", ip, iid)
            except OSError as exc:
                rospy.logwarn("  failed to send finish to %s: %s", ip, exc)
        self.sock.close()

    def spin(self) -> None:
        rospy.loginfo("traffic_light_publisher: relaying /traffic_phase to UDP (no internal timing).")
        rospy.spin()


if __name__ == "__main__":
    node = TrafficLightPublisher()
    node.spin()