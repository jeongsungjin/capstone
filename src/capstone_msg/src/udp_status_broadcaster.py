#!/usr/bin/env python3
"""
ROS 노드: 플래너/업링크 정보를 JSON으로 UDP 송신 (단일 포트, type 구분).

type:
- "carStatus": 1초 주기
    { "type":"carStatus", "car_id":1, "battery":8.31 }
- "route": 경로 변경 이벤트 시
    {
      "type":"route",
      "payload":{
        "carsStatus":[{"car_id":1,"routeChanged":false}, ...],
        "planning":[{"car_id":1,"path":[[x,y],...]}, ...]
      }
    }
- "end": 수동 목적지 지정 이벤트
    { "type":"end", "car_id":1, "end":"hospital" }

파라미터 (모두 선택적):
  ~dest_ip           : UDP 수신지 IP (기본 "127.0.0.1")
  ~port              : 전송 포트 (기본 60070)
  ~rate_status_hz    : carStatus 주기 (기본 1.0)
  ~route_check_hz    : route 이벤트 감시 주기 (기본 5.0)
  ~num_vehicles      : 차량 수 (vehicle_ids 미설정 시 사용, 기본 5)
  ~vehicle_ids       : 차량 ID 목록 (예: [1,2,3] 또는 "1,2,3")
  ~uplink_topic      : 업링크 구독 토픽 (기본 "/imu_uplink")
  ~path_topic_prefix : 글로벌 경로 토픽 prefix (기본 "/global_path_")
  ~override_prefix   : 오버라이드 goal 토픽 prefix (기본 "/override_goal_")
  ~path_max_points   : route 전송 시 경로 최대 점 개수(0=무제한, 기본 500)
  ~map_yaml          : 목적지 이름 매핑용 YAML (regions[].center)
  ~dest_names        : 목적지 이름 리스트 (예: ["home","hospital",...])
"""

import json
import socket
from typing import Dict, List, Tuple

import rospy
import yaml
from capstone_msgs.msg import Uplink
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class UdpStatusBroadcaster:
    def __init__(self) -> None:
        rospy.init_node("udp_status_broadcaster", anonymous=True)

        self.dest_ip = str(rospy.get_param("~dest_ip", "127.0.0.1"))
        self.port = int(rospy.get_param("~port", 60070))
        self.rate_status_hz = float(max(0.1, rospy.get_param("~rate_status_hz", 1.0)))
        self.route_check_hz = float(max(0.1, rospy.get_param("~route_check_hz", 5.0)))
        self.path_max_points = int(rospy.get_param("~path_max_points", 0))

        # 차량 ID 결정
        vehicle_ids = self._parse_vehicle_ids()
        self.vehicle_ids = tuple(sorted(set(vehicle_ids)))

        # 토픽 이름 구성
        self.uplink_topic = str(rospy.get_param("~uplink_topic", "/imu_uplink"))
        self.path_topic_prefix = str(rospy.get_param("~path_topic_prefix", "/global_path_"))
        # 기본값은 슬래시 형태로 맞추고, 호환을 위해 언더스코어 형태도 함께 구독
        self.override_prefix = str(rospy.get_param("~override_prefix", "/override_goal/"))
        self.override_name_prefix = str(rospy.get_param("~override_name_prefix", "/override_goal_name/"))
        self.map_yaml = str(rospy.get_param("~map_yaml", ""))
        self.dest_names = list(rospy.get_param("~dest_names", ["home", "hospital", "school", "store", "park", "garage", "lake"]))

        # 최신 상태 캐시
        self._voltages: Dict[int, float] = {}
        self._paths: Dict[int, List[Tuple[float, float]]] = {}
        self._path_sig: Dict[int, Tuple[int, Tuple[float, float], Tuple[float, float]]] = {}
        self._path_dirty: Dict[int, bool] = {vid: False for vid in self.vehicle_ids}
        self._path_seen: Dict[int, bool] = {vid: False for vid in self.vehicle_ids}
        self._initial_route_sent = False
        # 수동 목적지 이름 캐시
        self._override_names: Dict[int, str] = {}
        # 목적지 이름 매핑
        self._dest_centers: List[Tuple[str, float, float]] = self._load_destinations(self.map_yaml, self.dest_names)

        # 소켓은 1개를 재사용
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        rospy.Subscriber(self.uplink_topic, Uplink, self._uplink_cb, queue_size=20)
        for vid in self.vehicle_ids:
            topic = f"{self.path_topic_prefix}{self._role_name(vid)}"
            rospy.Subscriber(topic, Path, self._path_cb, callback_args=vid, queue_size=1)
            rospy.logdebug("Subscribed path topic: %s (car_id=%d)", topic, vid)
        for vid in self.vehicle_ids:
            role = self._role_name(vid)
            # pose override: 슬래시형/언더스코어형 모두 구독
            topics_pose = {
                f"{self.override_prefix}{role}",
                f"/override_goal_{role}",
            }
            for t in topics_pose:
                rospy.Subscriber(t, PoseStamped, self._override_cb, callback_args=vid, queue_size=1)
            # name override: 슬래시형/언더스코어형 모두 구독
            topics_name = {
                f"{self.override_name_prefix}{role}",
                f"/override_goal_name_{role}",
            }
            for t in topics_name:
                rospy.Subscriber(t, String, self._override_name_cb, callback_args=vid, queue_size=1)

        # 타이머: status 주기 송신, route 이벤트 감시
        rospy.Timer(rospy.Duration(1.0 / self.rate_status_hz), self._timer_status)
        rospy.Timer(rospy.Duration(1.0 / self.route_check_hz), self._timer_route)

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

    def _load_destinations(self, yaml_file: str, names: List[str]) -> List[Tuple[str, float, float]]:
        results: List[Tuple[str, float, float]] = []
        if not yaml_file:
            return results
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
            regions = data.get("regions", []) if isinstance(data, dict) else []
        except Exception as exc:
            rospy.logwarn(f"udp_status_broadcaster: failed to load map_yaml {yaml_file}: {exc}")
            return results
        for idx, reg in enumerate(regions, start=1):
            if idx > len(names):
                break
            center = reg.get("center")
            if not center or len(center) < 2:
                continue
            name = names[idx - 1] if idx - 1 < len(names) else f"dest{idx}"
            try:
                cx, cy = float(center[0]), float(center[1])
            except Exception:
                continue
            results.append((name, cx, cy))
        return results

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
        # carStatus payload: 개별 메시지로 송신
        cars = []
        for vid in self.vehicle_ids:
            cars.append({"car_id": vid, "battery": self._voltages.get(vid)})
        return cars

    def _build_port_b_payload(self):
        cars_status = []
        planning = []
        for vid in self.vehicle_ids:
            cars_status.append({"car_id": vid, "routeChanged": False})
            path = self._paths.get(vid, [])
            limited = path
            if self.path_max_points > 0 and len(path) > self.path_max_points:
                limited = path[: self.path_max_points]
            planning.append(
                {"car_id": vid, "path": [[float(x), float(y)] for x, y in limited]}
            )
        return {"payload": {"carsStatus": cars_status, "planning": planning}}

    def _send_payload(self, payload) -> None:
        try:
            data_str = json.dumps(payload, separators=(",", ":"))
            self.sock.sendto(data_str.encode("utf-8"), (self.dest_ip, self.port))
            msg_type = payload.get("type", "unknown") if isinstance(payload, dict) else "unknown"
            if msg_type == "carStatus":
                rospy.loginfo("UDP carStatus -> port=%d data=%s", self.port, data_str)
            elif msg_type == "route":
                # route는 길어질 수 있어 타입과 차량 수만 요약
                try:
                    cars = payload.get("payload", {}).get("carsStatus", [])
                    rospy.loginfo("UDP route -> port=%d cars=%d", self.port, len(cars))
                except Exception:
                    rospy.loginfo("UDP route -> port=%d", self.port)
            elif msg_type == "end":
                rospy.loginfo("UDP end -> port=%d data=%s", self.port, data_str)
            else:
                rospy.loginfo("UDP send -> port=%d data=%s", self.port, data_str)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "UDP send failed (port=%d): %s", self.port, exc)

    def _timer_status(self, _evt) -> None:
        # 1초 주기(기본)로 carStatus 전송 (차량별 개별 메시지)
        for car in self._build_port_a_payload():
            car["type"] = "carStatus"
            self._send_payload(car)

    def _timer_route(self, _evt) -> None:
        # 경로 재계획(변경) 발생 시에만 전송
        if any(self._path_dirty.values()) or (not self._initial_route_sent and all(self._path_seen.values())):
            payload = self._build_port_b_payload()
            payload["type"] = "route"
            self._send_payload(payload)
            self._initial_route_sent = True
            self._clear_dirty()

    def _clear_dirty(self) -> None:
        for vid in self.vehicle_ids:
            self._path_dirty[vid] = False

    def _override_cb(self, msg: PoseStamped, vehicle_id: int) -> None:
        try:
            vid = int(vehicle_id)
        except Exception:
            return
        name = self._override_names.get(vid)
        # 이름이 아직 안 왔으면 전송 보류 (name cb에서 전송)
        if not name:
            rospy.logdebug("override_cb: name not available yet for car_id=%d; waiting", vid)
            return
        payload = {"type": "end", "car_id": vid, "end": name}
        self._send_payload(payload)

    def _override_name_cb(self, msg: String, vehicle_id: int) -> None:
        try:
            vid = int(vehicle_id)
        except Exception:
            return
        try:
            txt = str(msg.data).strip()
            if txt:
                self._override_names[vid] = txt
                # 이름이 도착했을 때도 end 이벤트 한 번 더 전송
                payload = {"type": "end", "car_id": vid, "end": txt}
                self._send_payload(payload)
        except Exception:
            pass

    def _guess_dest_name(self, x: float, y: float) -> str:
        if not self._dest_centers:
            return "unknown"
        best = None
        best_d2 = float("inf")
        for name, cx, cy in self._dest_centers:
            d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y)
            if d2 < best_d2:
                best_d2 = d2
                best = name
        return best if best is not None else "unknown"


def main() -> None:
    UdpStatusBroadcaster()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

