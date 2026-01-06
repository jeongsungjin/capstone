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
        self.send_rate_hz: float = float(rospy.get_param("~send_rate_hz", 20.0))
        # 교차로별 최신 페이즈 캐시
        self.latest_phases: Dict[str, str] = {}

        # UDP 소켓
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # /traffic_phase 구독해서 들어오는 페이즈를 하드웨어로 중계
        rospy.Subscriber("/traffic_phase", TrafficLightPhase, self._phase_cb, queue_size=20)

        # 최신 상태를 설정 주기로 반복 전송
        rospy.Timer(rospy.Duration(1.0 / max(self.send_rate_hz, 0.1)), self._tick_send)

        rospy.on_shutdown(self._on_shutdown)

    def _phase_cb(self, msg: TrafficLightPhase) -> None:
        iid = (msg.intersection_id or "").strip()  # "A", "B", "C"
        phase = (msg.phase or "").strip()          # "P1_MAIN_GREEN" 등

        if not iid or not phase:
            return

        # 교차로별 최신 상태 저장
        self.latest_phases[iid] = phase

    def _tick_send(self, _event=None) -> None:
        if not self.latest_phases:
            return

        for iid, phase in list(self.latest_phases.items()):
            ip = self.ip_map.get(iid)
            if not ip:
                rospy.logwarn_throttle(5.0, "traffic_light_publisher: no IP configured for intersection_id=%s", iid)
                continue

            # 하드웨어가 기대하는 문자열 형식: "A_P1_MAIN_GREEN" / "B_P2_YELLOW" / "C_P3_SIDE_GREEN" ...
            payload_str = f"{iid}_{phase}"
            payload = payload_str.encode("utf-8")

            try:
                self.sock.sendto(payload, (ip, UDP_PORT))
                if self.verbose:
                    rospy.loginfo_throttle(1.0, "sent %s to %s", payload_str, ip)
            except OSError as exc:
                rospy.logwarn("traffic_light_publisher: failed to send to %s (%s): %s", ip, payload_str, exc)

    def _on_shutdown(self) -> None:
        for iid, ip in self.ip_map.items():
            try:
                self.sock.sendto(b"finish", (ip, UDP_PORT))
            except OSError as exc:
                rospy.logwarn("  failed to send finish to %s: %s", ip, exc)
        self.sock.close()

    def spin(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    node = TrafficLightPublisher()
    node.spin()