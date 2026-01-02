#!/usr/bin/env python3

import math
import sys
import os
import json
from typing import Dict, List, Tuple, Optional
import threading
import yaml

import rospy

# Add planner scripts path for GlobalPlanner import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_planner_scripts = os.path.join(_script_dir, '..', '..', 'carla_multi_vehicle_planner', 'scripts')
if _planner_scripts not in sys.path:
    sys.path.insert(0, _planner_scripts)

try:
    from global_planner import GlobalPlanner
except Exception as e:
    GlobalPlanner = None
    rospy.logwarn(f"Failed to import GlobalPlanner: {e}")
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header, String
from std_msgs.msg import Bool
from capstone_msgs.msg import Uplink  # type: ignore

try:
    from carla_multi_vehicle_control.msg import TrafficLightPhase, TrafficApproach  # type: ignore
except Exception:
    TrafficLightPhase = None  # type: ignore
    TrafficApproach = None  # type: ignore

try:
    import carla  # type: ignore
except Exception as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


def quaternion_to_yaw(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SimpleMultiVehicleController:
    """
    최소 기능 컨트롤러:
    - /planned_path_{role} 구독, /carla/{role}/odometry 구독
    - Pure Pursuit으로 조향/속도 산출
    - CARLA VehicleControl 적용 + /carla/{role}/vehicle_control_cmd 퍼블리시
    - 안전/우선순위/플래툰/영역 게인/추가 오버라이드 기능 제거
    """

    def __init__(self) -> None:
        rospy.init_node("multi_vehicle_controller", anonymous=True)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 5))
        self.emergency_stop_active = False
        self.lookahead_distance = float(rospy.get_param("~lookahead_distance", 3.0))
        # 조향/헤딩 오차 기반 lookahead 조정 (LPF만 적용)
        self.min_lookahead_m = float(rospy.get_param("~min_lookahead_m", 3.0))
        self.max_lookahead_m = float(rospy.get_param("~max_lookahead_m", 7.0))
        self.heading_ld_gain = float(rospy.get_param("~heading_ld_gain", 2.0))  # ld / (1 + gain * |heading_err|)
        self.heading_deadzone_rad = abs(float(rospy.get_param("~heading_deadzone_deg", 2.0))) * math.pi / 180.0
        self.heading_lpf_alpha = float(rospy.get_param("~heading_lpf_alpha", 0.3))  # 0~1, 0=hold, 1=no filter
        self.heading_lpf_alpha = max(0.0, min(1.0, self.heading_lpf_alpha))
        # 곡률 기반 LD 조절
        self.curv_ld_enable = bool(rospy.get_param("~curv_ld_enable", True))
        self.curv_ld_min = float(rospy.get_param("~curv_ld_min", 2.0))   # 최소 2m
        self.curv_ld_max = float(rospy.get_param("~curv_ld_max", 7.0))   # 직선에서 7m
        self.curv_ld_gain = float(rospy.get_param("~curv_ld_gain", 10.0))  # ld = max / (1 + gain*|kappa|)
        self.wheelbase = float(rospy.get_param("~wheelbase", 1.74))
        # max_steer: 차량의 물리적 최대 조향각(rad) – CARLA 정규화에 사용 (fallback)
        self.max_steer = float(rospy.get_param("~max_steer", 0.5))
        # cmd_max_steer: 명령으로 허용할 최대 조향(rad) – Ackermann/제어 내부 클램프
        self.cmd_max_steer = float(rospy.get_param("~cmd_max_steer", 0.5))
        # 기본 주행 속도: 모든 차량 동일 속도 사용 (플래툰 속도 제어 없음)
        self.target_speed = float(rospy.get_param("~target_speed", 5.0))
        self.control_frequency = float(rospy.get_param("~control_frequency", 30.0))
        # Low-voltage speed gating + parking slowdown
        self.low_voltage_threshold = float(rospy.get_param("~low_voltage_threshold", 5.0))
        self.parking_dest_x = float(rospy.get_param("~parking_dest_x", -23.0))
        self.parking_dest_y = float(rospy.get_param("~parking_dest_y", -16.5))
        self.parking_speed = float(rospy.get_param("~parking_speed", 5.0))
        self.parking_speed_radius = float(rospy.get_param("~parking_speed_radius", 5.0))
        self.parking_stop_radius = float(rospy.get_param("~parking_stop_radius", 0.5))
        self.progress_backtrack_window_m = float(rospy.get_param("~progress_backtrack_window_m", 5.0))
        self.progress_sequential_window_m = float(rospy.get_param("~progress_sequential_window_m", 35.0))
        self.progress_heading_limit_deg = float(rospy.get_param("~progress_heading_limit_deg", 150.0))
        self._progress_heading_limit_rad = max(
            0.0,
            min(math.pi, abs(self.progress_heading_limit_deg) * math.pi / 180.0),
        )
        # Optional: 외부 속도 오버라이드(플래툰 등)
        self.use_speed_override = bool(rospy.get_param("~use_speed_override", False))
        self.speed_override_timeout = float(rospy.get_param("~speed_override_timeout", 0.5))
        # Traffic light gating (region-based hard stop)
        self.tl_yellow_policy = str(rospy.get_param("~tl_yellow_policy", "cautious")).strip()  # cautious|permissive
        # Collision stop gating (forward cone)
        self.collision_stop_enable = bool(rospy.get_param("~collision_stop_enable", True))
        self.collision_stop_angle_deg = float(rospy.get_param("~collision_stop_angle_deg", 40.0))  # +/-deg ahead
        self.collision_stop_distance_m = float(rospy.get_param("~collision_stop_distance_m", 7.0))

        # CARLA world
        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        timeout = float(rospy.get_param("~carla_timeout", 5.0))
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        
        # GlobalPlanner for edge localization
        self.route_planner = None
        if GlobalPlanner is not None:
            try:
                self.route_planner = GlobalPlanner(self.carla_map, sampling_resolution=1.0)
                rospy.loginfo("[Controller] GlobalPlanner initialized for edge localization")
            except Exception as e:
                rospy.logwarn(f"[Controller] Failed to create GlobalPlanner: {e}")

        # State
        self.vehicles: Dict[str, carla.Actor] = {}
        self.states: Dict[str, Dict] = {}
        self.state_locks: Dict[str, threading.Lock] = {}  # Locks for thread-safe state access
        self.control_publishers: Dict[str, rospy.Publisher] = {}
        self.pose_publishers: Dict[str, rospy.Publisher] = {}
        self._tl_phase: Dict[str, Dict] = {}
        self._voltage: Dict[int, float] = {}
        self._speed_override: Dict[str, Dict[str, float]] = {}
        
        self.idx_backtrack_allow = int(rospy.get_param("~idx_backtrack_allow", 0))
        self.seq_min_jump = int(rospy.get_param("~seq_min_jump", 10))
        self.platoon_mode = bool(rospy.get_param("~platoon_mode", False))
        # idx 윈도우 제한을 기본 비활성화(0 or 음수면 제한 없음)
        self.platoon_idx_window = int(rospy.get_param("~platoon_idx_window", 0))
        # 로깅: 모두 INFO로 출력 (주기 제한 없음)
        self.progress_log_enable = True
        self.progress_log_period = 0.0
        self.path_log_enable = True
        self.path_log_period = 0.0

        # Hardware ID -> IP -> 논리 ID 매핑 설정
        self.hw_ip_prefix = str(rospy.get_param("~hw_ip_prefix", "192.168.0."))
        self.hw_ip_offset = int(rospy.get_param("~hw_ip_offset", 10))
        self.allow_hwid_fallback = bool(rospy.get_param("~allow_hwid_fallback", True))
        self.vehicle_ips = self._load_vehicle_ips(self.num_vehicles)
        self.ip_to_logical = {ip: idx + 1 for idx, ip in enumerate(self.vehicle_ips)}
        if not self.vehicle_ips:
            rospy.logwarn("[Controller] vehicle_ips not set; hwid fallback=%s", self.allow_hwid_fallback)
        elif len(self.vehicle_ips) < self.num_vehicles:
            rospy.logwarn("[Controller] vehicle_ips shorter than num_vehicles (%d < %d)", len(self.vehicle_ips), self.num_vehicles)

        self.traffic_interference = {
            # 하단 진출
            ((10, 11), (6, 11), (6, 7), (11, 9), (9, 3), (9, 1)):
                ((2, 4), (4, 8), (8, 7), (8, 18)),

            # 좌단 진출
            ((26, 15), (21, 15), (26, 22)):
                ((9, 3), (3, 16), (16, 22), (16, 24), (5, 3)),

            # 우단 진출
            ((13, 25), (13, 19), (23, 19), (19, 5)):
                ((2, 17), (17, 20), (20, 25), (20, 12)),

            # 좌상단 진출
            ((16, 24), (24, 23), (21, 24)):
                ((25, 26), (26, 15), (26, 22)),

            # 우상단 진출
            ((20, 25), (13, 25), (25, 26)):
                ((23, 12), (23, 19), (24, 23)),

            # 좌하단 진출
            ((26, 22), (22, 6), (16, 22)):
                ((18, 21), (21, 24), (21, 15)),
            
            # 우하단 진출
            ((23, 12), (20, 12), (12, 10)):
                ((7, 13), (13, 19), (13, 25)),

            # 좌중단 진출
            ((10, 11), (10, 18), (18, 21), (8, 18)):
                ((22, 6), (6, 11), (6, 7)),

            # 우중단 진출
            ((6, 7), (7, 13), (8, 18), (8, 7)):
                ((12, 10), (10, 11), (10, 18))
        }

        # 신호등별 정지 차량 추적: Dict[signal_name, List[role]]
        # 신호등 이름은 traffic_signals.yaml 기준 하드코딩
        self._vehicles_at_tl: Dict[str, List[str]] = {
            "A_MAIN_3": [], "A_MAIN_4": [], "A_SIDE": [],
            "B_MAIN_3": [], "B_MAIN_4": [], "B_SIDE": [],
            "C_MAIN_3": [], "C_MAIN_4": [], "C_SIDE": []
        }
        self._vehicles_at_tl_pub = rospy.Publisher("/vehicles_at_tl", String, queue_size=1, latch=True)

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            self.states[role] = {
                "path": [],  # List[(x, y)]
                "path_idx": [],
                "position": None,
                "orientation": None,
                "twist": None,
                "current_index": 0,
                "current_seq": None,
                "s_profile": [],
                "progress_s": 0.0,
                "progress_idx_ratio": 0.0,
                "idx_min": 0,
                "idx_max": 0,
                "path_length": 0.0,
            }
            self.state_locks[role] = threading.Lock()  # Lock for thread-safe state access

            path_topic = f"/planned_path_{role}"
            odom_topic = f"/carla/{role}/odometry"
            cmd_topic = f"/carla/{role}/vehicle_control_cmd_raw"
            pose_topic = f"/{role}/pose"
            rospy.Subscriber(path_topic, Path, self._path_cb, callback_args=role, queue_size=1)
            rospy.Subscriber(odom_topic, Odometry, self._odom_cb, callback_args=role, queue_size=10)
            if self.use_speed_override:
                override_topic = f"/carla/{role}/vehicle_control_cmd_override"
                rospy.Subscriber(
                    override_topic,
                    AckermannDrive,
                    self._speed_override_cb,
                    callback_args=role,
                    queue_size=5,
                )

            self.control_publishers[role] = rospy.Publisher(cmd_topic, AckermannDrive, queue_size=1)
            self.pose_publishers[role] = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
            
            # Deadlock/token 기반 충돌 무시 플래그
            override_col_topic = f"/collision_override/{role}"
            self._collision_override: Dict[str, bool] = getattr(self, "_collision_override", {})
            self._collision_override[role] = False
            rospy.Subscriber(
                override_col_topic,
                Bool,
                self._collision_override_cb,
                callback_args=role,
                queue_size=5,
            )

        # Subscribe traffic light phase if message is available
        if TrafficLightPhase is not None:
            rospy.Subscriber("/traffic_phase", TrafficLightPhase, self._tl_cb, queue_size=5)
        
        # Uplink (voltage) subscriber: topic configurable
        self.uplink_topic = str(rospy.get_param("~uplink_topic", "/uplink")).strip()
        if self.uplink_topic:
            rospy.Subscriber(self.uplink_topic, Uplink, self._uplink_cb, queue_size=10)

        rospy.Subscriber("/emergency_stop", Bool, self._e_stop_cb, queue_size=5)
        rospy.Timer(rospy.Duration(1.0 / 20.0), self._refresh_vehicles)
        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.control_frequency)), self._control_loop)

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _refresh_vehicles(self, _evt) -> None:
        actors = self.world.get_actors().filter("vehicle.*")
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role in self.states:
                self.vehicles[role] = actor

    def _path_cb(self, msg: Path, role: str) -> None:
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        # 항상 0..N-1로 정규화된 인덱스 사용 (플래툰 안정성 우선)
        idx_profile = list(range(len(points)))
        s_profile, total_len = self._compute_path_profile(points)
        
        with self.state_locks[role]:  # Lock for thread-safe state update
            st = self.states[role]
            st["path"] = points
            st["path_idx"] = idx_profile
            st["current_index"] = 0
            st["current_seq"] = idx_profile[0] if idx_profile else None
            st["s_profile"] = s_profile
            st["path_length"] = total_len
            st["idx_min"] = idx_profile[0] if idx_profile else 0
            st["idx_max"] = idx_profile[-1] if idx_profile else 0
            st["progress_s"] = 0.0
            st["progress_idx_ratio"] = 0.0
            st["progress_fail_count"] = 0
            
        if self.path_log_enable:
            rospy.loginfo(
                f"{role}: path recv len={len(points)} idx_range=[{idx_profile[0] if idx_profile else 0},{idx_profile[-1] if idx_profile else 0}] stamp={msg.header.stamp.to_sec():.3f}",
            )
        # rospy.loginfo_throttle(1.0, f"{role}: planned_path received ({len(points)} pts, len={total_len:.1f} m)")
        vehicle = self.vehicles.get(role)
        rear = self._rear_point(self.states[role], vehicle)
        if rear is not None:
            rx, ry, _ = rear
            proj = self._project_progress(
                points,
                s_profile,
                idx_profile,
                rx,
                ry,
                0.0,
                self.progress_backtrack_window_m,
                prev_idx=None,
            )
            if proj is not None:
                s_now, idx, seq_val = proj
                with self.state_locks[role]:
                    st = self.states[role]
                    st["progress_s"] = s_now
                    st["current_index"] = idx
                    st["current_seq"] = seq_val
            else:
                # 초기에 경로 투영 실패 시 바로 근처 점으로 스냅
                snap = self._force_snap_progress(self.states[role], rx, ry)
                if snap is not None:
                    s_now, idx = snap
                    with self.state_locks[role]:
                        st = self.states[role]
                        st["progress_s"] = s_now
                        st["current_index"] = idx
                        st["progress_fail_count"] = 0
                    # rospy.loginfo(f"{role}: progress reset to s={s_now:.1f} at path receive")

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        st = self.states[role]
        st["position"] = msg.pose.pose.position
        st["orientation"] = msg.pose.pose.orientation
        st["twist"] = msg.twist.twist
        # also publish pose for tools
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pose_publishers[role].publish(pose_msg)

    def _speed_override_cb(self, msg: AckermannDrive, role: str) -> None:
        """
        외부에서 들어온 속도 오버라이드(AckermannDrive)의 speed 필드만 사용한다.
        steering은 기본 컨트롤러 값을 그대로 사용한다.
        """
        self._speed_override[role] = {
            "speed": float(getattr(msg, "speed", 0.0)),
            "stamp": rospy.Time.now().to_sec(),
        }

    def _collision_override_cb(self, msg: Bool, role: str) -> None:
        try:
            self._collision_override[role] = bool(msg.data)
        except Exception:
            self._collision_override[role] = False

    def _compute_path_profile(self, points: List[Tuple[float, float]]):
        if len(points) < 2:
            return [], 0.0
        cumulative = [0.0]
        total = 0.0
        for i in range(1, len(points)):
            step = math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            total += step
            cumulative.append(total)
        return cumulative, total

    def _rear_point(self, st, vehicle):
        """
        Pure pursuit은 보통 rear-axle를 기준점으로 사용한다.
        CARLA odom은 차량 중심 기준이므로, bbox 절반 길이나 wheelbase/2를 뒤로 보정해 rear를 추정한다.
        """
        pos = st.get("position")
        ori = st.get("orientation")
        if pos is None or ori is None:
            return None
        yaw = quaternion_to_yaw(ori)
        # rear axle 위치: 차량 전체 길이의 7/8 지점(= center에서 뒤로 0.75 * length/2 = 0.75*extent.x)
        back_offset = max(0.0, self.wheelbase * 0.5)
        if vehicle is not None:
            bb = getattr(vehicle, "bounding_box", None)
            if bb is not None and getattr(bb, "extent", None) is not None:
                length_half = float(bb.extent.x)
                back_offset = max(back_offset, 0.75 * length_half)
        rx = pos.x - math.cos(yaw) * back_offset
        ry = pos.y - math.sin(yaw) * back_offset
        return rx, ry, yaw

    def _project_progress(
        self,
        path,
        s_profile,
        idx_profile,
        px,
        py,
        prev_s: float,
        backtrack_window: float,
        heading: float = None,
        prev_idx=None,
        prev_index=None,
        idx_window=None,
    ):
        attempts = []
        seq_window = max(0.0, getattr(self, "progress_sequential_window_m", 0.0))
        if seq_window > 1e-3:
            attempts.append(seq_window)
        attempts.append(None)  # Fall back to unrestricted search
        for forward_window in attempts:
            result = self._project_progress_limited(
                path,
                s_profile,
                idx_profile,
                px,
                py,
                prev_s,
                backtrack_window,
                forward_window=forward_window,
                heading=heading,
                prev_idx=prev_idx,
                prev_index=prev_index,
                idx_window=idx_window,
            )
            if result is not None:
                return result
        return None

    def _project_progress_limited(
        self,
        path,
        s_profile,
        idx_profile,
        px,
        py,
        prev_s: float,
        backtrack_window: float,
        forward_window: float = None,
        heading: float = None,
        prev_idx=None,
        prev_index=None,
        idx_window=None,
    ):
        # 방어: s_profile이 비었거나 길이가 path와 다르면 즉시 재계산
        if len(path) < 2:
            return None
        if not s_profile or len(s_profile) != len(path):
            s_profile, _ = self._compute_path_profile(path)
            if not s_profile or len(s_profile) != len(path):
                return None
        min_s = max(0.0, float(prev_s) - max(0.0, float(backtrack_window)))
        max_s = float("inf")
        if forward_window is not None and math.isfinite(float(forward_window)):
            max_s = float(prev_s) + max(0.0, float(forward_window))
        best_dist_sq = float("inf")
        best_index = None
        best_t = 0.0
        best_s = None
        best_seq = None
        for idx in range(len(path) - 1):
            x1, y1 = path[idx]
            x2, y2 = path[idx + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue
            if self.platoon_mode and prev_index is not None and idx_window is not None and idx_window > 0:
                if idx < prev_index - idx_window or idx > prev_index + idx_window:
                    continue
            if idx_profile and len(idx_profile) > idx and prev_idx is not None:
                seg_seq = idx_profile[idx]
                if seg_seq < prev_idx - self.idx_backtrack_allow:
                    continue
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + dx * t
            proj_y = y1 + dy * t
            dist_sq = (proj_x - px) ** 2 + (proj_y - py) ** 2
            seg_length = math.sqrt(seg_len_sq)
            cand_s = s_profile[idx] + t * seg_length
            if cand_s < min_s:
                continue
            if cand_s > max_s:
                continue
            if heading is not None and self._progress_heading_limit_rad < math.pi - 1e-6:
                seg_heading = math.atan2(dy, dx)
                diff = abs((seg_heading - heading + math.pi) % (2.0 * math.pi) - math.pi)
                if diff > self._progress_heading_limit_rad:
                    continue
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_index = idx
                best_t = t
                best_s = cand_s
                best_seq = idx_profile[idx] if idx_profile and len(idx_profile) > idx else None
        if best_index is None:
            return None
        # 추가 방어: 인덱스 경계 확인
        if best_index < 0 or best_index >= len(path) - 1 or best_index >= len(s_profile):
            return None
        return float(best_s), best_index, best_seq

    def _project_point_on_path(self, path, px: float, py: float):
        if len(path) < 2:
            return None
        best_dist_sq = float("inf")
        best_index = None
        best_t = 0.0
        best_seg = (0.0, 0.0)
        for idx in range(len(path) - 1):
            x1, y1 = path[idx]
            x2, y2 = path[idx + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + dx * t
            proj_y = y1 + dy * t
            dist_sq = (proj_x - px) ** 2 + (proj_y - py) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_index = idx
                best_t = t
                best_seg = (dx, dy)
        if best_index is None:
            return None
        proj_x = path[best_index][0] + best_seg[0] * best_t
        proj_y = path[best_index][1] + best_seg[1] * best_t
        return best_index, best_t, proj_x, proj_y, best_seg[0], best_seg[1]

    def _estimate_curvature(self, path, idx: int):
        """세 점(앞/현재/뒤)을 사용해 단순 곡률 κ(1/m)를 추정한다."""
        if len(path) < 3:
            return None
        i = max(1, min(len(path) - 2, idx))
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        x3, y3 = path[i + 1]
        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x3 - x1, y3 - y1)
        if a < 1e-3 or b < 1e-3 or c < 1e-3:
            return None
        area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        denom = a * b * c
        if denom < 1e-6:
            return None
        kappa = 4.0 * area / denom  # 2R = abc/area -> k = 1/R = 4*area/(abc)
        
        return kappa

    def _force_snap_progress(self, st, px: float, py: float):
        path = st.get("path") or []
        s_profile = st.get("s_profile") or []
        proj = self._project_point_on_path(path, px, py)
        if proj is None:
            return None
        idx, t, proj_x, proj_y, seg_dx, seg_dy = proj
        seg_len = math.hypot(seg_dx, seg_dy)
        if not s_profile or len(s_profile) != len(path):
            s_profile, _ = self._compute_path_profile(path)
        if idx >= len(s_profile):
            return None
        if seg_len < 1e-6:
            s_now = s_profile[idx]
        else:
            s_now = s_profile[idx] + t * seg_len
        return s_now, idx

    def _update_idx_progress(self, st) -> float:
        """웨이포인트 seq 기반 진행 비율(0~1)을 계산한다."""
        idx_min = int(st.get("idx_min", 0))
        idx_max = int(st.get("idx_max", 0))
        seq = st.get("current_seq", None)
        if seq is None:
            pi = st.get("path_idx") or []
            ci = int(st.get("current_index", 0))
            if 0 <= ci < len(pi):
                seq = pi[ci]
        denom = max(1, idx_max - idx_min)
        ratio = 0.0
        if seq is not None and idx_max > idx_min:
            ratio = float(seq - idx_min) / float(denom)
        else:
            # seq 정보가 없거나 범위가 0이면 arc-length 기반 fallback
            path_len = float(st.get("path_length", 0.0))
            prog_s = float(st.get("progress_s", 0.0))
            if path_len > 1e-3:
                ratio = prog_s / path_len
        ratio = max(0.0, min(1.0, ratio))
        st["progress_idx_ratio"] = ratio
        return ratio

    def _sample_path_at_s(self, path, s_profile, s_target: float, min_index: int = 0):
        """Arc-length 보간으로 정확한 목표점을 얻는다."""
        if len(path) < 2 or not s_profile or len(s_profile) != len(path):
            return None
        # 진행 방향 유지: 검색 시작 인덱스를 최소값으로 제한
        start_i = max(0, min(len(s_profile) - 1, int(min_index)))
        if s_target <= s_profile[start_i]:
            return path[start_i][0], path[start_i][1], start_i, 0.0
        if s_target >= s_profile[-1]:
            return path[-1][0], path[-1][1], len(path) - 1, 0.0
        # 찾기 (min_index 이후에서만 진행, 세그먼트 길이 0은 건너뜀)
        for i in range(start_i, len(s_profile) - 1):
            if s_profile[i + 1] >= s_target:
                ds_raw = s_profile[i + 1] - s_profile[i]
                if abs(ds_raw) < 1e-6:
                    continue
                ds = max(1e-6, ds_raw)
                t = (s_target - s_profile[i]) / ds
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                tx = x1 + (x2 - x1) * t
                ty = y1 + (y2 - y1) * t
                return tx, ty, i, t
        return None

    def _select_target(self, st, x, y, lookahead_override: float = None) -> Tuple[float, float]:
        # Local copy to avoid race condition during state update
        path = list(st.get("path") or [])
        s_profile = list(st.get("s_profile") or [])
        if len(path) < 2 or len(s_profile) != len(path):
            return x, y
        # arc-length target selection with interpolation
        s_now = float(st.get("progress_s", 0.0))
        # 진행 방향 보존: 현재 인덱스 기준으로 뒤로 가지 않도록 보정
        cur_idx = max(0, min(len(path) - 1, int(st.get("current_index", 0))))
        if s_profile and cur_idx < len(s_profile):
            s_now = max(s_now, s_profile[cur_idx])
        base_ld = float(self.lookahead_distance) if lookahead_override is None else float(lookahead_override)
        ld = base_ld
        # 곡률 기반 LD 축소: 직선(κ≈0)에서 ld≈curv_ld_max, 곡률 커지면 최소 curv_ld_min
        if self.curv_ld_enable:
            kappa = self._estimate_curvature(path, st.get("current_index", 0))
            if kappa is not None:
                ld_curv = self.curv_ld_max / (1.0 + self.curv_ld_gain * abs(kappa))
                ld = max(self.curv_ld_min, min(self.curv_ld_max, ld_curv))
        # 헤딩 오차 기반 추가 축소 (LPF/데드존 반영)
        heading_err = float(st.get("heading_err_filt", st.get("heading_err", 0.0)))
        ld = ld / (1.0 + self.heading_ld_gain * abs(heading_err))
        ld = max(self.min_lookahead_m, min(self.max_lookahead_m, ld))
        # 디버그: 최종 ld 출력
        # rospy.loginfo_throttle(
        #     0.5,
        #     f"{st.get('role','')}: ld={ld:.2f} (curv={kappa if 'kappa' in locals() else None}, head_err={heading_err:.3f})",
        # )
        s_target = s_now + max(0.05, ld)
        sample = self._sample_path_at_s(path, s_profile, s_target, min_index=cur_idx)
        if sample is None:
            return x, y
        tx, ty, idx, _t = sample
        st["current_index"] = idx
        return tx, ty

    def _compute_control(self, st, vehicle, role: str):
        path = st.get("path") or []
        pos = st.get("position")
        ori = st.get("orientation")
        if not path or len(path) < 2 or pos is None or ori is None or vehicle is None:
            return 0.0, 0.0
        ref = self._rear_point(st, vehicle)
        if ref is None:
            return None, None
        fx, fy, yaw = ref
        prev_s = float(st.get("progress_s", 0.0))
        proj = self._project_progress(
            path,
            st.get("s_profile") or [],
            st.get("path_idx") or [],
            fx,
            fy,
            prev_s,
            self.progress_backtrack_window_m,
            heading=yaw,
            prev_idx=st.get("current_seq"),
            prev_index=st.get("current_index"),
            idx_window=(self.platoon_idx_window if self.platoon_mode else None),
        )
        if proj is not None:
            s_now, idx, seq_val = proj
            st["progress_s"] = s_now
            st["current_index"] = idx
            st["current_seq"] = seq_val
            # seq가 이전보다 크게 후퇴하면 강제 스냅
            prev_seq = st.get("last_seq")
            if prev_seq is not None and seq_val is not None and seq_val < prev_seq - self.idx_backtrack_allow:
                reset = self._force_snap_progress(st, fx, fy)
                if reset is not None:
                    s_now, idx = reset
                    st["progress_s"] = s_now
                    st["current_index"] = idx
                    seq_val = st["path_idx"][idx] if st.get("path_idx") and idx < len(st["path_idx"]) else seq_val
                    st["current_seq"] = seq_val
            st["last_seq"] = seq_val
            self._update_idx_progress(st)
            # 현재 진행 세그먼트 헤딩으로 헤딩 오차 추정
            path_idx = max(0, min(len(path) - 2, idx))
            seg_dx = path[path_idx + 1][0] - path[path_idx][0]
            seg_dy = path[path_idx + 1][1] - path[path_idx][1]
            if abs(seg_dx) + abs(seg_dy) < 1e-6:
                st["heading_err"] = 0.0
            else:
                path_heading = math.atan2(seg_dy, seg_dx)
                err = path_heading - yaw
                while err > math.pi:
                    err -= 2.0 * math.pi
                while err < -math.pi:
                    err += 2.0 * math.pi
                # 데드존 및 LPF 적용
                if abs(err) < self.heading_deadzone_rad:
                    err_adj = 0.0
                else:
                    err_adj = math.copysign(abs(err) - self.heading_deadzone_rad, err)
                prev_filt = float(st.get("heading_err_filt", err_adj))
                filt = prev_filt + self.heading_lpf_alpha * (err_adj - prev_filt)
                st["heading_err_raw"] = err
                st["heading_err"] = err_adj
                st["heading_err_filt"] = filt
        else:
            count = st.setdefault("progress_fail_count", 0) + 1
            st["progress_fail_count"] = count
            # rospy.logwarn_throttle(
            #     2.0,
            #     f"{role}: progress projection failed #{count} (fx={fx:.1f}, fy={fy:.1f}); keeping previous s={prev_s:.1f}",
            # )
            # 즉시 경로로 스냅하여 조향이 바로 경로를 향하도록
            if count >= 2:
                reset = self._force_snap_progress(st, fx, fy)
                if reset is not None:
                    s_now, idx = reset
                    st["progress_s"] = s_now
                    st["current_index"] = idx
                    st["progress_fail_count"] = 0
                    seq_val = st["path_idx"][idx] if st.get("path_idx") and idx < len(st["path_idx"]) else None
                    st["current_seq"] = seq_val
                    st["last_seq"] = seq_val
                    self._update_idx_progress(st)
                    # rospy.loginfo(f"{role}: progress reset to s={s_now:.1f} after projection failure")
        if proj is not None:
            st["progress_fail_count"] = 0
        else:
            # 헤딩 오차 정보를 사용할 수 없으므로 리셋
            st["heading_err"] = 0.0
            st["heading_err_filt"] = 0.0
        tx, ty = self._select_target(st, fx, fy, lookahead_override=None)
        dx = tx - fx
        dy = ty - fy
        alpha_raw = math.atan2(dy, dx) - yaw
        while alpha_raw > math.pi:
            alpha_raw -= 2.0 * math.pi
        while alpha_raw < -math.pi:
            alpha_raw += 2.0 * math.pi
        alpha = alpha_raw
        Ld = math.hypot(dx, dy)
        if Ld < 1e-3:
            return 0.0, 0.0
        steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)
        # 명령 상한으로 클램프 (라디안)
        steer = max(-self.cmd_max_steer, min(self.cmd_max_steer, steer))
        speed = self.target_speed
        # Apply traffic light gating
        speed = self._apply_tl_gating(role, vehicle, st, fx, fy, speed)
        # Apply forward collision gating (vehicles in front cone within distance)
        speed = self._apply_collision_gating(role, fx, fy, yaw, speed)
        # Apply parking slowdown when low-voltage path ends at parking dest
        speed = self._apply_parking_speed_limit(role, st, speed)
        # rospy.loginfo_throttle(
        #     0.5,
        #     "%s steer=%.3f Ld=%.2f alpha=%.3f",
        #     role,
        #     steer,
        #     Ld,
        #     alpha,
        # )
        return steer, speed

    def _get_edge_at_location(self, x: float, y: float) -> Optional[tuple]:
        """주어진 위치의 edge를 반환"""
        if self.route_planner is None:
            return None
        try:
            loc = carla.Location(x=x, y=y, z=0.0)
            edge = self.route_planner._localize(loc)
            return edge
        except Exception:
            return None

    def _is_vehicle_registered_at_tl(self, role: str) -> Optional[str]:
        """해당 차량이 신호등에 등록되어 있는지 확인 (자료구조 기반)
        
        Returns:
            등록된 신호등 이름, 없으면 None
        """
        for signal_name, roles in self._vehicles_at_tl.items():
            if role in roles:
                return signal_name
        return None

    def _should_ignore_for_traffic_light(self, my_edge: tuple, other_edge: tuple, other_role: str) -> bool:
        """
        신호대기 차량을 무시해야 하는지 판단
        
        조건:
        1. 상대 차량이 신호등에 등록되어 있음 (_vehicles_at_tl)
        2. 내 edge가 traffic_interference의 key 중 하나에 속해 있음
        3. 상대 edge가 해당 key의 value에 속해 있음
        
        Returns:
            True if should ignore (proceed without stopping)
        """
        if my_edge is None or other_edge is None:
            return False
        
        # 상대 차량이 신호등에 등록되어 있는지 확인 (자료구조 기반)
        stopped_signal = self._is_vehicle_registered_at_tl(other_role)
        if stopped_signal is None:
            return False
        
        # traffic_interference 맵 확인
        for key_edges, value_edges in self.traffic_interference.items():
            # 내 edge가 key 중 하나에 속해 있고
            if my_edge in key_edges and other_edge in value_edges:
                # 상대 edge가 해당 value 중 하나에 속해 있으면
                rospy.loginfo_throttle(
                    0.1,
                    f"Ignoring {other_role} at {stopped_signal}: my_edge={my_edge} -> other_edge={other_edge}"
                )
                return True
        
        return False

    def _apply_collision_gating(self, role: str, fx: float, fy: float, yaw: float, speed_cmd: float) -> float:
        # Deadlock token으로 충돌 정지 해제 요청 시 그대로 통과
        if self._collision_override.get(role, False):
            return speed_cmd
        
        # 플래툰/외부 속도오버라이드 사용 시 충돌 정지 비활성화 (간섭 방지)
        if self.use_speed_override:
            return speed_cmd
        
        if not self.collision_stop_enable:
            return speed_cmd
        
        angle_th = abs(float(self.collision_stop_angle_deg)) * math.pi / 180.0
        dist_th = max(0.0, float(self.collision_stop_distance_m))
        
        # 내 위치의 edge 확인
        my_edge = self._get_edge_at_location(fx, fy)
        
        # Iterate over other controlled vehicles (based on odom states)
        for other_role, ost in self.states.items():
            if other_role == role:
                continue

            op = ost.get("position")
            if op is None:
                continue
            
            # 상대 차량의 heading 확인 (반대 방향 차량 필터링용)
            other_ori = ost.get("orientation")
            if other_ori is not None:
                other_yaw = quaternion_to_yaw(other_ori)
                yaw_diff = abs(yaw - other_yaw)
                
                # 정규화: 0 ~ pi 범위로
                while yaw_diff > math.pi:
                    yaw_diff = abs(yaw_diff - 2.0 * math.pi)

                # 반대 방향 (약 100도 이상 차이) → 무시
                if yaw_diff > math.pi * 100 / 180:  # ~100도
                    continue
            
            other_x = float(op.x)
            other_y = float(op.y)
            dx = other_x - fx
            dy = other_y - fy
            dist = math.hypot(dx, dy)
            
            if dist <= 1e-3:
                return 0.0
            
            # Bearing difference to heading
            ang = math.atan2(dy, dx)
            rel = ang - yaw
            while rel > math.pi:
                rel -= 2.0 * math.pi

            while rel < -math.pi:
                rel += 2.0 * math.pi
            
            if abs(rel) <= angle_th and dist <= dist_th:
                # 신호대기 차량 무시 로직
                other_edge = self._get_edge_at_location(other_x, other_y)
                if self._should_ignore_for_traffic_light(my_edge, other_edge, other_role):
                    continue  # 해당 차량 무시하고 다음 차량 검사
                
                signal = self._is_vehicle_registered_at_tl(other_role)
                if signal and role not in self._vehicles_at_tl[signal]:
                    self._vehicles_at_tl[signal].append(role)
                    rospy.loginfo(f"[TL] {role} registered at {signal} due to {other_role} stopping")
                    self._publish_vehicles_at_tl()

                return 0.0
        return speed_cmd

    def _get_vehicle_max_steer(self, vehicle) -> float:
        # 차량 물리 최대 조향(rad) 조회; 실패 시 파라미터 max_steer 사용
        try:
            pc = vehicle.get_physics_control()
            wheels = getattr(pc, "wheels", [])
            if wheels:
                deg = max([float(getattr(w, "max_steer_angle", 0.0)) for w in wheels])
                if deg > 0.0:
                    return deg * math.pi / 180.0
        except Exception:
            pass
        return max(1e-3, float(self.max_steer))

    def _apply_carla_control(self, vehicle, steer, speed):
        if self.emergency_stop_active:
            steer = 0.0
            speed = 0.0
        control = carla.VehicleControl()
        # 차량 물리 최대각으로 정규화하여 CARLA [-1,1]에 매핑
        veh_max = self._get_vehicle_max_steer(vehicle)
        control.steer = max(-1.0, min(1.0, float(steer) / max(1e-3, veh_max)))
        v = vehicle.get_velocity()
        current_speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        speed_error = speed - current_speed
        if speed_error > 0:
            control.throttle = max(0.2, min(1.0, speed_error / max(1.0, speed)))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(1.0, -speed_error / max(1.0, self.target_speed))
        vehicle.apply_control(control)

    def _tl_cb(self, msg: "TrafficLightPhase") -> None:
        # Cache approaches for quick gating
        iid = msg.intersection_id or "default"
        try:
            approaches = []
            for ap in msg.approaches:
                approaches.append({
                    "name": ap.name,
                    "color": int(ap.color),
                    "xmin": float(ap.xmin), "xmax": float(ap.xmax),
                    "ymin": float(ap.ymin), "ymax": float(ap.ymax),
                    "sx": float(ap.stopline_x), "sy": float(ap.stopline_y),
                })
            self._tl_phase[iid] = {"stamp": msg.header.stamp, "approaches": approaches}

            if int(ap.color) == 2:
                # 녹색불인 경우 → 해당 신호등에서 모든 차량 제거
                for ap in approaches:
                    signal_name = ap.get("name", "")
                    if signal_name in self._vehicles_at_tl:
                        self._vehicles_at_tl[signal_name] = []
                        # rospy.loginfo(f"[TL] All vehicles cleared from {signal_name} (color=green)")
                        self._publish_vehicles_at_tl()

        except Exception:
            self._tl_phase[iid] = {"stamp": rospy.Time.now(), "approaches": []}

    def _uplink_cb(self, msg: "Uplink") -> None:
        try:
            hwid_or_logical = int(msg.vehicle_id)
            logical_id = self._map_hwid_to_logical(hwid_or_logical)
            if logical_id is None:
                rospy.logwarn_throttle(
                    2.0,
                    "[Controller] logical id not found for hwid=%d (ip=%s); skip voltage",
                    hwid_or_logical,
                    self._hwid_to_ip(hwid_or_logical),
                )
                return
            self._voltage[logical_id] = float(msg.voltage)
        except Exception:
            pass

    def _e_stop_cb(self, msg: Bool) -> None:
        try:
            self.emergency_stop_active = bool(msg.data)
        except Exception:
            self.emergency_stop_active = False

    def _is_low_voltage(self, role: str) -> bool:
        try:
            vid = int(role.split("_")[-1])
            v = self._voltage.get(vid, float("inf"))
            # print(f"Voltage: {v}")
            return v <= float(self.low_voltage_threshold)
        except Exception:
            return False

    def _apply_tl_gating(self, role: str, vehicle, st, fx: float, fy: float, speed_cmd: float) -> float:
        if not self._tl_phase:
            return speed_cmd
        
        pos = st.get("position")
        # rear 기준뿐 아니라 차량 중심 좌표도 함께 검사해 영역 누락 방지
        check_points = [("rear", fx, fy)]
        if pos is not None:
            check_points.append(("center", float(pos.x), float(pos.y)))
        
        convert_role_name_to_color = lambda r: ["red", "yellow", "green", "black", "white"][int(r[-1]) - 1]

        for data in self._tl_phase.values():
            approaches = data.get("approaches", [])
            for ap in approaches:
                signal_name = ap.get("name", "")

                for label, px, py in check_points:
                    if px >= ap["xmin"] and px <= ap["xmax"] and py >= ap["ymin"] and py <= ap["ymax"]:
                        color = int(ap.get("color", 0))  # 0=R,1=Y,2=G
                        
                        # Region-based hard stop (빨간불 또는 황색)
                        if color == 0 or (color == 1 and self.tl_yellow_policy.lower() != "permissive"):
                            # 신호등에 차량 등록
                            if role not in self._vehicles_at_tl[signal_name]:
                                self._vehicles_at_tl[signal_name].append(role)
                                # rospy.loginfo(f"[TL] {convert_role_name_to_color(role)} registered at {signal_name} (color=red/yellow)")
                                self._publish_vehicles_at_tl()

                            return 0.0
                        
                        # 녹색불인 경우 → 해당 신호등에서 차량 제거
                        elif color == 2:
                            if role in self._vehicles_at_tl[signal_name]:
                                self._vehicles_at_tl[signal_name] = []
                                # rospy.loginfo(f"[TL] {convert_role_name_to_color(role)} left {signal_name} (color=green)")
                                self._publish_vehicles_at_tl()
                        break  # 해당 approach에서 hit 확인됨
        
        return speed_cmd

    def _publish_vehicles_at_tl(self) -> None:
        """신호등별 정지 차량 정보를 JSON 문자열로 발행"""
        try:
            msg = String()
            msg.data = json.dumps(self._vehicles_at_tl)
            self._vehicles_at_tl_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[TL] Failed to publish vehicles_at_tl: {e}")

    def _apply_parking_speed_limit(self, role: str, st, speed_cmd: float) -> float:
        if not self._is_low_voltage(role):
            return speed_cmd
        # Distance to parking destination (vehicle position preferred)
        pos = st.get("position")
        if pos is not None:
            dist = math.hypot(float(pos.x) - float(self.parking_dest_x), float(pos.y) - float(self.parking_dest_y))
        else:
            path = st.get("path") or []
            if len(path) < 2:
                return speed_cmd
            px, py = path[-1]
            dist = math.hypot(px - float(self.parking_dest_x), py - float(self.parking_dest_y))
        # Distance to end of current path (arc-length)
        dist_end = self._distance_to_path_end(st)
        stop_r = max(0.0, float(self.parking_stop_radius))
        slow_r = max(stop_r, float(self.parking_speed_radius))
        if (dist_end is not None and dist_end <= stop_r) or dist <= stop_r:
            return 0.0
        if (dist_end is not None and dist_end <= slow_r) or dist <= slow_r:
            return min(speed_cmd, float(self.parking_speed))
        return speed_cmd

    def _distance_to_path_end(self, st) -> float:
        path = st.get("path") or []
        s_profile = st.get("s_profile") or []
        if len(path) < 2 or not s_profile or len(s_profile) != len(path):
            return None
        path_len = float(st.get("path_length", 0.0))
        prog = float(st.get("progress_s", 0.0))
        rem = path_len - prog
        if rem < 0.0:
            rem = 0.0
        return rem

    def _load_vehicle_ips(self, num_vehicles: int):
        ips_param = rospy.get_param("~vehicle_ips", None)
        if isinstance(ips_param, list) and ips_param:
            return [str(ip) for ip in ips_param][:num_vehicles]
        if isinstance(ips_param, str):
            txt = ips_param.strip()
            if txt:
                try:
                    loaded = yaml.safe_load(txt)
                    if isinstance(loaded, list):
                        return [str(ip) for ip in loaded][:num_vehicles]
                except Exception:
                    pass
                parts = [p.strip() for p in txt.split(",") if p.strip()]
                if parts:
                    return parts[:num_vehicles]

        ips = []
        for i in range(1, num_vehicles + 1):
            val = rospy.get_param(f"~vehicle_{i}_ip", None)
            if val is None:
                val = rospy.get_param(f"/vehicle_{i}_ip", None)
            if val is not None:
                ips.append(str(val))
        return ips

    def _hwid_to_ip(self, hwid: int) -> str:
        return f"{self.hw_ip_prefix}{hwid + self.hw_ip_offset}"

    def _map_hwid_to_logical(self, hwid_or_logical: int):
        # 이미 논리 ID 범위라면 그대로 사용
        if 1 <= hwid_or_logical <= self.num_vehicles:
            return hwid_or_logical

        ip = self._hwid_to_ip(hwid_or_logical)
        lid = self.ip_to_logical.get(ip)
        if lid is not None:
            return lid

        if self.allow_hwid_fallback:
            return hwid_or_logical

        return None
    
    def _publish_ackermann(self, role, steer, speed):
        msg = AckermannDrive()
        msg.steering_angle = float(0.0 if self.emergency_stop_active else steer)
        msg.speed = float(0.0 if self.emergency_stop_active else speed)
        self.control_publishers[role].publish(msg)

    def _get_speed_override(self, role: str) -> float:
        if not self.use_speed_override:
            return None
        data = self._speed_override.get(role)
        if not data:
            return None
        now = rospy.Time.now().to_sec()
        if now - float(data.get("stamp", 0.0)) > max(0.0, self.speed_override_timeout):
            return None
        try:
            return float(data.get("speed", 0.0))
        except Exception:
            return None

    def _control_loop(self, _evt) -> None:
        for role, st in self.states.items():
            vehicle = self.vehicles.get(role)
            if vehicle is None:
                continue
            steer, speed = self._compute_control(st, vehicle, role)
            if steer is None:
                continue
            override_speed = self._get_speed_override(role)
            if override_speed is not None:
                speed = override_speed
            self._apply_carla_control(vehicle, steer, speed)
            self._publish_ackermann(role, steer, speed)

            # colors = ["red", "yellow", "green", "black", "white"]
            # rospy.loginfo_throttle(
            #     0.5,
            #     f"{colors[int(role[-1]) - 1]}: cmd steer={steer:.3f} rad speed={speed:.2f} m/s",
            # )


if __name__ == "__main__":
    try:
        SimpleMultiVehicleController()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleMultiVehicleController crashed: {e}")
        raise