#!/usr/bin/env python3

import os
import time
from typing import Dict, List, Tuple, Optional

import rospy
import yaml
from std_msgs.msg import Header, String
from carla_multi_vehicle_control.msg import TrafficLightPhase, TrafficApproach


def get_light_states(phase_name: str) -> Dict[str, bool]:
    # 기본 False로 초기화
    keys = {
        "M_LR_R": True, "M_LR_Y": False, "M_LR_G": False, "M_LR_LEFT": False,
        "M_RL_R": True, "M_RL_Y": False, "M_RL_G": False,
        "S_R": True, "S_Y": False, "S_G": False,
    }
    if phase_name == "P1_MAIN_GREEN":
        keys.update({
            "M_LR_R": True, "M_LR_G": False,
            "M_RL_R": False, "M_RL_G": True,
        })
    elif phase_name == "P1_YELLOW":
        keys.update({
            "M_LR_R": True, "M_LR_Y": False,
            "M_RL_R": False, "M_RL_Y": True,
        })
    elif phase_name == "P2_MAIN_GREEN":
        keys.update({
            "M_LR_R": False, "M_LR_LEFT": True,
            "M_RL_R": True, "M_LR_G": True,
        })
    elif phase_name == "P2_YELLOW":
        keys.update({
            "M_LR_R": False, "M_LR_Y": True,
            "M_RL_R": True,
        })
    elif phase_name == "P3_SIDE_GREEN":
        keys.update({"S_R": False, "S_G": True})
    elif phase_name == "P3_YELLOW":
        keys.update({"S_R": False, "S_Y": True})
    return keys


class SignalsPhasePublisher:
    def __init__(self) -> None:
        rospy.init_node("signals_phase_publisher", anonymous=True)
        self.yaml_path = str(rospy.get_param("~signals_yaml", rospy.get_param("~config", ""))).strip()
        self.publish_hz = float(rospy.get_param("~publish_hz", 10.0))
        # 외부 페이즈 입력(선택): /tm/phase_name (std_msgs/String) 구독
        self.external_phase_topic = str(rospy.get_param("~external_phase_topic", "/tm/phase_name")).strip()
        self.use_external_phase = bool(rospy.get_param("~use_external_phase", False))
        default_offsets = {"A": 20.0, "B": 0.0, "C": 38.0}
        cfg_offsets = rospy.get_param("~intersection_offsets", default_offsets)
        self.intersection_offsets = {}
        for k, v in (cfg_offsets or {}).items():
            try:
                self.intersection_offsets[str(k)] = float(v)
            except Exception:
                pass

        self._phase_name = "P1_MAIN_GREEN"
        if self.use_external_phase and self.external_phase_topic:
            rospy.Subscriber(self.external_phase_topic, String, self._phase_cb, queue_size=10)

        self.signals = self._load_signals(self.yaml_path)
        self.by_intersection: Dict[str, List[Dict]] = {}
        for sid, s in self.signals.items():
            iid = str(s.get("intersection", "default"))
            self.by_intersection.setdefault(iid, []).append(s)

        self.pub = rospy.Publisher("/traffic_phase", TrafficLightPhase, queue_size=10)

        # 내부 사이클(옵션): 외부 입력 없으면 내부 순환
        self.sequence = [
            ("P1_MAIN_GREEN", float(rospy.get_param("~p1_green_s", 20.0))),
            ("P1_YELLOW", float(rospy.get_param("~p1_yellow_s", 3.0))),
            ("P2_MAIN_GREEN", float(rospy.get_param("~p2_green_s", 20.0))),
            ("P2_YELLOW", float(rospy.get_param("~p2_yellow_s", 3.0))),
            ("P3_SIDE_GREEN", float(rospy.get_param("~p3_green_s", 15.0))),
            ("P3_YELLOW", float(rospy.get_param("~p3_yellow_s", 3.0))),
        ]

        '''
        # shortest version
        
        self.sequence = [
            ("P1_MAIN_GREEN", float(rospy.get_param("~p1_green_s", 8.0))),
            ("P1_YELLOW", float(rospy.get_param("~p1_yellow_s", 1.0))),
            ("P2_MAIN_GREEN", float(rospy.get_param("~p2_green_s", 8.0))),
            ("P2_YELLOW", float(rospy.get_param("~p2_yellow_s", 1.0))),
            ("P3_SIDE_GREEN", float(rospy.get_param("~p3_green_s", 6.0))),
            ("P3_YELLOW", float(rospy.get_param("~p3_yellow_s", 1.0))),
        ]
        '''

        self._cycle_time = sum([d for _, d in self.sequence]) or 1.0
        self._t0 = time.time()

    def _phase_cb(self, msg: String) -> None:
        name = msg.data.strip()
        if name:
            self._phase_name = name

    def _load_signals(self, path: str) -> Dict[str, Dict]:
        if not path or not os.path.exists(path):
            rospy.logwarn("signals_phase_publisher: signals yaml not found: %s", path)
            return {}
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            rospy.logwarn("signals_phase_publisher: yaml load failed: %s", exc)
            return {}
        signals = {}
        for key, val in (data.get("signals", {}) or {}).items():
            # stop_region 다각형 → bbox, stopline은 폴리곤 평균점
            poly = val.get("stop_region", [])
            if isinstance(poly, list) and poly:
                xs = [float(p[0]) for p in poly if isinstance(p, list) and len(p) >= 2]
                ys = [float(p[1]) for p in poly if isinstance(p, list) and len(p) >= 2]
                if xs and ys:
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    sx = sum(xs) / len(xs)
                    sy = sum(ys) / len(ys)
                    val["bbox"] = (xmin, xmax, ymin, ymax)
                    val["stopline"] = (sx, sy)
            signals[str(key)] = val
        return signals

    def _phase_from_elapsed(self, elapsed: float) -> str:
        t = elapsed % max(1e-3, self._cycle_time)
        acc = 0.0
        for name, dur in self.sequence:
            if acc <= t < acc + dur:
                return name
            acc += dur
        return self.sequence[-1][0]

    def _phase_for_intersection(self, iid: str) -> str:
        if self.use_external_phase:
            return self._phase_name
        offset = float(self.intersection_offsets.get(iid, 0.0))
        elapsed = time.time() - self._t0 + offset
        return self._phase_from_elapsed(elapsed)

    def _compute_color(self, state_keys: Dict, light_states: Dict[str, bool]) -> int:
        # 0=Red, 1=Yellow, 2=Green
        try:
            red_k = state_keys.get("red")
            yellow_k = state_keys.get("yellow")
            sgreen_k = state_keys.get("straight_green")
            lgreen_k = state_keys.get("left_green")
            if red_k and light_states.get(red_k, False):
                return 0
            if yellow_k and light_states.get(yellow_k, False):
                return 1
            if (sgreen_k and light_states.get(sgreen_k, False)) or (lgreen_k and light_states.get(lgreen_k, False)):
                return 2
            # 기본 red
            return 0
        except Exception:
            return 0

    def _build_msg_for_intersection(self, iid: str, light_states: Dict[str, bool], phase: str) -> TrafficLightPhase:
        msg = TrafficLightPhase()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="map")
        msg.intersection_id = iid
        msg.phase = phase
        for s in self.by_intersection.get(iid, []):
            tap = TrafficApproach()
            tap.name = str(s.get("name", ""))
            tap.color = self._compute_color(s.get("state_keys", {}) or {}, light_states)
            bbox = s.get("bbox", (-1e9, 1e9, -1e9, 1e9))
            tap.xmin, tap.xmax, tap.ymin, tap.ymax = map(float, bbox)
            stop = s.get("stopline", (0.0, 0.0))
            tap.stopline_x, tap.stopline_y = map(float, stop)
            msg.approaches.append(tap)
        return msg

    def spin(self) -> None:
        rate = rospy.Rate(max(1.0, self.publish_hz))
        while not rospy.is_shutdown():
            for iid in self.by_intersection.keys():
                phase = self._phase_for_intersection(iid)
                ls = get_light_states(phase)
                msg = self._build_msg_for_intersection(iid, ls, phase)
                self.pub.publish(msg)
            rate.sleep()


if __name__ == "__main__":
    try:
        node = SignalsPhasePublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass
