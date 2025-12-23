#!/usr/bin/env python3
"""
ROS 노드: 플래너/업링크 정보를 JSON으로 UDP 송신.

- port A: 차량별 배터리 전압 + 경로 시작/끝 좌표
    {
      "cars": [
        { "car_id": 1, "battery": 8.31, "start": {"x": ..., "y": ...}, "end": {"x": ..., "y": ...} },
        ...
      ]
    }
- port B: 장애물로 인한 경로 변경 여부(현재 기본값) + 전체 경로 좌표 집합
    {
      "payload": {
        "carsStatus": [ { "car_id": 1, "routeChanged": false }, ... ],
        "planning":   [ { "car_id": 1, "path": [[x, y], ...] }, ... ]
      }
    }

파라미터 (모두 선택적):
  ~dest_ip                : UDP 수신지 IP (기본 "127.0.0.1")
  ~port_a                 : 포트 A 번호 (기본 60070)
  ~port_b                 : 포트 B 번호 (기본 60071)
  ~rate_a_hz              : 포트 A 송신 주기 (Hz, 기본 1.0)
  ~route_changed_check_hz : 포트 B 전송 이벤트 감시 주기 (Hz, 기본 5.0)
  ~num_vehicles           : 차량 수 (vehicle_ids 미설정 시 사용, 기본 5)
  ~vehicle_ids            : 차량 ID 목록 (예: [1,2,3] 또는 "1,2,3")
  ~uplink_topic           : 업링크 구독 토픽 (기본 "/imu_uplink")
  ~path_topic_prefix      : 글로벌 경로 토픽 prefix (기본 "/global_path_")
  ~path_max_points        : 포트 B로 보내는 경로 최대 점 개수(0=무제한, 기본 500)
"""

import json
import socket
from typing import Dict, List, Tuple

import rospy
from capstone_msgs.msg import Uplink
from nav_msgs.msg import Path


class UdpStatusBroadcaster:
    def __init__(self) -> None:
        rospy.init_node("udp_status_broadcaster", anonymous=True)

        self.dest_ip = str(rospy.get_param("~dest_ip", "127.0.0.1"))
        self.port_a = int(rospy.get_param("~port_a", 60070))
        self.port_b = int(rospy.get_param("~port_b", 60071))
        self.rate_a_hz = float(max(0.1, rospy.get_param("~rate_a_hz", 1.0)))
        self.route_changed_check_hz = float(max(0.1, rospy.get_param("~route_changed_check_hz", 5.0)))
        self.path_max_points = int(rospy.get_param("~path_max_points", 500))

        # 차량 ID 결정
        vehicle_ids = self._parse_vehicle_ids()
        self.vehicle_ids = tuple(sorted(set(vehicle_ids)))

        # 토픽 이름 구성
        self.uplink_topic = str(rospy.get_param("~uplink_topic", "/imu_uplink"))
        self.path_topic_prefix = str(rospy.get_param("~path_topic_prefix", "/global_path_"))

        # 최신 상태 캐시
        self._voltages: Dict[int, float] = {}
        self._paths: Dict[int, List[Tuple[float, float]]] = {}
        self._path_sig: Dict[int, Tuple[int, Tuple[float, float], Tuple[float, float]]] = {}
        self._path_dirty: Dict[int, bool] = {vid: False for vid in self.vehicle_ids}
        self._path_seen: Dict[int, bool] = {vid: False for vid in self.vehicle_ids}
        self._initial_b_sent = False

        # 소켓은 1개를 재사용
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        rospy.Subscriber(self.uplink_topic, Uplink, self._uplink_cb, queue_size=20)
        for vid in self.vehicle_ids:
            topic = f"{self.path_topic_prefix}{self._role_name(vid)}"
            rospy.Subscriber(topic, Path, self._path_cb, callback_args=vid, queue_size=1)
            rospy.logdebug("Subscribed path topic: %s (car_id=%d)", topic, vid)

        # 타이머: A는 고정 주기 송신, B는 이벤트 감시
        rospy.Timer(rospy.Duration(1.0 / self.rate_a_hz), self._timer_port_a)
        rospy.Timer(rospy.Duration(1.0 / self.route_changed_check_hz), self._timer_port_b)

    def _parse_vehicle_ids(self) -> List[int]:
        raw = rospy.get_param("~vehicle_ids", None)
        ids: List[int] = []
        if isinstance(raw, list):
            for v in raw:
                try:
                    ids.append(int(v))
                except Exception:
                    pass
        elif isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            for p in parts:
                try:
                    ids.append(int(p))
                except Exception:
                    pass
        if ids:
            return ids
        num = int(rospy.get_param("~num_vehicles", 5))
        return list(range(1, max(1, num) + 1))

    def _role_name(self, vehicle_id: int) -> str:
        return f"ego_vehicle_{vehicle_id}"

    def _uplink_cb(self, msg: Uplink) -> None:
        try:
            vid = int(msg.vehicle_id)
            self._voltages[vid] = float(msg.voltage)
        except Exception:
            pass

    def _path_cb(self, msg: Path, vehicle_id: int) -> None:
        pts: List[Tuple[float, float]] = []
        for pose in msg.poses:
            try:
                pts.append((float(pose.pose.position.x), float(pose.pose.position.y)))
            except Exception:
                continue
        vid = int(vehicle_id)
        self._paths[vid] = pts
        rospy.logdebug(
            "[car %d] path received (%d pts)", vid, len(pts)
        )
        # 경로 변경 감지: 길이, 시작/끝 좌표로 단순 서명
        if pts:
            sig = (len(pts), pts[0], pts[-1])
        else:
            sig = (0, (0.0, 0.0), (0.0, 0.0))
        prev = self._path_sig.get(vid)
        if prev is None or prev != sig:
            self._path_dirty[vid] = True
            self._path_sig[vid] = sig
        self._path_seen[vid] = True

    def _build_port_a_payload(self):
        cars = []
        for vid in self.vehicle_ids:
            entry = {"car_id": vid, "battery": self._voltages.get(vid)}
            path = self._paths.get(vid)
            if path:
                entry["start"] = {"x": path[0][0], "y": path[0][1]}
                entry["end"] = {"x": path[-1][0], "y": path[-1][1]}
            cars.append(entry)
        return {"cars": cars}

    def _build_port_b_payload(self):
        cars_status = []
        planning = []
        for vid in self.vehicle_ids:
            # 요청: routeChanged는 현재 항상 False 고정
            cars_status.append({"car_id": vid, "routeChanged": False})
            path = self._paths.get(vid, [])
            limited = path
            if self.path_max_points > 0 and len(path) > self.path_max_points:
                limited = path[: self.path_max_points]
            planning.append(
                {"car_id": vid, "path": [[float(x), float(y)] for x, y in limited]}
            )
        return {"payload": {"carsStatus": cars_status, "planning": planning}}

    def _send_payload(self, payload, port: int) -> None:
        try:
            data_str = json.dumps(payload, separators=(",", ":"))
            self.sock.sendto(data_str.encode("utf-8"), (self.dest_ip, port))
            rospy.loginfo("UDP send port=%d data=%s", port, data_str)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "UDP send failed (port=%d): %s", port, exc)

    def _timer_port_a(self, _evt) -> None:
        # 1초 주기(기본)로 항상 송신
        self._send_payload(self._build_port_a_payload(), self.port_a)

    def _timer_port_b(self, _evt) -> None:
        # 초기 1회: 모든 차량 경로가 최소 한 번 수신된 경우 송신
        if not self._initial_b_sent and all(self._path_seen.values()):
            self._send_payload(self._build_port_b_payload(), self.port_b)
            self._initial_b_sent = True
            self._clear_dirty()
            return

        # 경로 재계획(변경) 발생 시에만 송신
        if any(self._path_dirty.values()):
            self._send_payload(self._build_port_b_payload(), self.port_b)
            self._clear_dirty()

    def _clear_dirty(self) -> None:
        for vid in self.vehicle_ids:
            self._path_dirty[vid] = False


def main() -> None:
    UdpStatusBroadcaster()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

