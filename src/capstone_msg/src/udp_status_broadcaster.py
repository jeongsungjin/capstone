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
import message_filters

import rospy
import yaml
from capstone_msgs.msg import Uplink, PathMeta
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from carla_multi_vehicle_control.msg import TrafficLightPhase


class UdpStatusBroadcaster:
    def __init__(self) -> None:
        rospy.init_node("udp_status_broadcaster", anonymous=True)

        self.dest_ip = str(rospy.get_param("~dest_ip", "192.168.0.31"))
        self.port = int(rospy.get_param("~port", 60070))
        self.rate_status_hz = float(max(0.1, rospy.get_param("~rate_status_hz", 1.0)))
        self.route_check_hz = float(max(0.1, rospy.get_param("~route_check_hz", 5.0)))
        self.traffic_hz = float(max(0.1, rospy.get_param("~traffic_hz", 1.0)))
        self.path_max_points = int(rospy.get_param("~path_max_points", 0))
        self.path_resolution = float(rospy.get_param("~path_resolution", 0.1))

        # 차량 ID 결정
        vehicle_ids = self._parse_vehicle_ids()
        self.vehicle_ids = tuple(sorted(set(vehicle_ids)))

        # 토픽 이름 구성
        self.uplink_topic = str(rospy.get_param("~uplink_topic", "/imu_uplink"))
        self.path_topic_prefix = str(rospy.get_param("~path_topic_prefix", "/global_path_"))
        self.path_meta_topic_prefix = str(rospy.get_param("~path_meta_topic_prefix", "/global_path_meta_"))
        
        # 기본값은 슬래시 형태로 맞추고, 호환을 위해 언더스코어 형태도 함께 구독
        self.override_prefix = str(rospy.get_param("~override_prefix", "/override_goal/"))
        self.override_name_prefix = str(rospy.get_param("~override_name_prefix", "/override_goal_name/"))
        self.map_yaml = str(rospy.get_param("~map_yaml", ""))
        self.dest_names = list(rospy.get_param("~dest_names", ["home", "hospital", "school", "store", "park", "garage", "lake"]))
        self.traffic_topic = str(rospy.get_param("~traffic_topic", "/traffic_phase"))

        # 최신 상태 캐시
        self._voltages: Dict[int, float] = {}
        # TODO : !!!!!!!!!!!!!!
        self._paths: Dict[int, dict] = {vid: {"path": [], "category": "", "resolution": 0.1, "s_start": [], "s_end": []} for vid in self.vehicle_ids}

        self._path_sig: Dict[int, Tuple[int, Tuple[float, float], Tuple[float, float]]] = {}
        self._path_dirty: Dict[int, bool] = {vid: False for vid in self.vehicle_ids}
        self._path_seen: Dict[int, bool] = {vid: False for vid in self.vehicle_ids}
        self._initial_route_sent = False
        self._override_names: Dict[int, str] = {}
        self._dest_centers: List[Tuple[str, float, float]] = self._load_destinations(self.map_yaml, self.dest_names)
        self._traffic_phase: TrafficLightPhase = None  # type: ignore
        self._traffic_phases: Dict[str, TrafficLightPhase] = {}
        self._tl_name_to_id: Dict[str, int] = {}
        self._tl_fourway_ids: set = set()
        self._next_tl_id = 1

        # 소켓은 1개를 재사용
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        rospy.Subscriber(self.uplink_topic, Uplink, self._uplink_cb, queue_size=20)
        for vid in self.vehicle_ids:
            path_topic = f"{self.path_topic_prefix}{self._role_name(vid)}"
            path_meta_topic = f"{self.path_meta_topic_prefix}{self._role_name(vid)}"

            path_sub = message_filters.Subscriber(path_topic, Path)
            meta_sub = message_filters.Subscriber(path_meta_topic, PathMeta)
            ts = message_filters.TimeSynchronizer([path_sub, meta_sub], 10)
            ts.registerCallback(lambda path, meta, vid=vid: self._path_cb(path, meta, vid))

            rospy.logdebug("Subscribed path topic: %s (car_id=%d)", path_topic, vid)
        
        for vid in self.vehicle_ids:
            role = self._role_name(vid)

            topics_pose = {
                f"{self.override_prefix}{role}",
                f"/override_goal_{role}",
            }

            for t in topics_pose:
                rospy.Subscriber(t, PoseStamped, self._override_cb, callback_args=vid, queue_size=1)

            topics_name = {
                f"{self.override_name_prefix}{role}",
                f"/override_goal_name_{role}",
            }

            for t in topics_name:
                rospy.Subscriber(t, String, self._override_name_cb, callback_args=vid, queue_size=1)

        rospy.Subscriber(self.traffic_topic, TrafficLightPhase, self._traffic_cb, queue_size=5)

        # 타이머: status 주기 송신, route 이벤트 감시
        rospy.Timer(rospy.Duration(1.0 / self.rate_status_hz), self._timer_status)
        rospy.Timer(rospy.Duration(1.0 / self.route_check_hz), self._timer_route)
        rospy.Timer(rospy.Duration(1.0 / self.traffic_hz), self._timer_traffic)

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

    def _path_cb(self, path: Path, meta: PathMeta, vehicle_id: int) -> None:
        pts: Tuple[Tuple[float, float]] = tuple(
            (float(pose.pose.position.x), float(pose.pose.position.y)) for pose in path.poses
        )

        category = meta.category.data
        resolution = meta.resolution.data
        s_starts = meta.s_starts.data
        s_ends = meta.s_ends.data

        self._paths[vehicle_id]["path"] = pts
        self._paths[vehicle_id]["category"] = category
        self._paths[vehicle_id]["resolution"] = resolution
        self._paths[vehicle_id]["s_start"] = s_starts if category == "obstacle" else []
        self._paths[vehicle_id]["s_end"] = s_ends if category == "obstacle" else []

        rospy.logdebug("[car %d] path received (%d pts)", vehicle_id, len(pts))

        # 경로 변경 감지: 길이, 시작/끝 좌표로 단순 서명
        sig = (len(pts), pts) if pts else (0, ((0.0, 0.0),))
        prev = self._path_sig.get(vehicle_id)
        if prev is None or prev != sig:
            self._path_dirty[vehicle_id] = True
            self._path_sig[vehicle_id] = sig

        self._path_seen[vehicle_id] = True

    def _build_port_a_payload(self):
        # carStatus payload: 개별 메시지로 송신
        cars = []
        for vid in self.vehicle_ids:
            cars.append({"car_id": vid, "battery": self._voltages.get(vid)})
        return cars

    def _build_port_b_payload(self):
        ret = {
            "type": "route",
            "resolution": self.path_resolution,
            "payload": []
        }

        for vid in self.vehicle_ids:
            if not self._path_dirty.get(vid):
                continue

            info = self._paths.get(vid, {"category": "", "path": []})
            
            ret["payload"].append({
                "vid": vid,
                "category": info["category"],
                "s_start": info["s_start"],
                "s_end": info["s_end"],
                "planning": info["path"]
            })

        return ret

    def _send_payload(self, payload) -> None:
        return

        try:
            data_str = json.dumps(payload, separators=(",", ":"))
            self.sock.sendto(data_str.encode("utf-8"), (self.dest_ip, self.port))
            
            msg_type = payload.get("type", "unknown") if isinstance(payload, dict) else "unknown"
            if msg_type == "carStatus":
                pass
                # rospy.loginfo("UDP carStatus -> port=%d data=%s", self.port, data_str)
            
            elif msg_type == "route":
                # route는 길어질 수 있어 타입과 차량 수만 요약
                try:
                    cars = payload.get("payload")
                    rospy.logfatal(f"UDP route -> port={self.port} cars={[cars[i]['vid'] for i in range(len(cars))]} count={len(cars)}")
                except Exception:
                    rospy.logfatal("UDP route -> port=%d", self.port)
            
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

    def _traffic_cb(self, msg: TrafficLightPhase) -> None:
        # 매 수신 시 intersection_id + 이름/인덱스 조합으로 신규 ID를 고정 배정 (교차로 추가도 반영)
        if msg is not None:
            keys = []
            for idx, ap in enumerate(msg.approaches):
                nm = ap.name or f"ap{idx}"
                key = f"{msg.intersection_id}:{nm}"
                keys.append(key)
            for key in sorted(keys):
                if key not in self._tl_name_to_id:
                    self._tl_name_to_id[key] = self._next_tl_id
                    # 4구 신호(좌회전 화살표 포함) 여부 기록
                    if idx == 0 or "LEFT" in nm.upper() or "M_LR" in nm.upper():
                        self._tl_fourway_ids.add(self._next_tl_id)
                    self._next_tl_id += 1
            iid = str(msg.intersection_id or "default")
            self._traffic_phases[iid] = msg
        self._traffic_phase = msg

    def _timer_traffic(self, _evt) -> None:
        if not self._traffic_phases:
            return
        # 모든 교차로 메시지를 순회하며 송신 (ID는 교차로+어프로치 이름 기준으로 고정)
        for iid in sorted(self._traffic_phases.keys()):
            phase_msg = self._traffic_phases[iid]
            for idx, ap in enumerate(phase_msg.approaches):
                name = ap.name or f"ap{idx}"
                key = f"{iid}:{name}"
                tl_id = self._tl_name_to_id.get(key)
                if tl_id is None:
                    tl_id = self._next_tl_id
                    self._tl_name_to_id[key] = tl_id
                    if idx == 0 or "LEFT" in name_upper or "M_LR" in name_upper:
                        self._tl_fourway_ids.add(tl_id)
                    self._next_tl_id += 1
                name_upper = name.upper()
                color_int = int(ap.color)
                if color_int == 0:
                    light = "red"
                    left_green = False
                elif color_int == 1:
                    light = "yellow"
                    left_green = False
                elif color_int == 2:
                    # 4구 신호(좌회전 포함): light는 green, 좌회전은 별도 필드로 표기
                    if tl_id in self._tl_fourway_ids:
                        light = "green"
                        left_green = True
                    else:
                        light = "green"
                        left_green = False
                else:
                    light = "red"
                    left_green = False

                # rospy.logfatal("TrafficLight ID=%d Name=%s Light=%s LeftGreen=%s", tl_id, name, light, left_green)
                payload = {"type": "trafficLight", "trafficLight_id": tl_id, "light": light}
                if tl_id in self._tl_fourway_ids:
                    payload["left_green"] = bool(left_green)
                self._send_payload(payload)

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