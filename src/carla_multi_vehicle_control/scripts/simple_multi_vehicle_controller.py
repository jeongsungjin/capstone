#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple

import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header
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
        self.min_lookahead_m = float(rospy.get_param("~min_lookahead_m", 2.0))
        self.max_lookahead_m = float(rospy.get_param("~max_lookahead_m", 5.0))
        self.heading_ld_gain = float(rospy.get_param("~heading_ld_gain", 2.0))  # ld / (1 + gain * |heading_err|)
        self.heading_deadzone_rad = abs(float(rospy.get_param("~heading_deadzone_deg", 2.0))) * math.pi / 180.0
        self.heading_lpf_alpha = float(rospy.get_param("~heading_lpf_alpha", 0.3))  # 0~1, 0=hold, 1=no filter
        self.heading_lpf_alpha = max(0.0, min(1.0, self.heading_lpf_alpha))
        self.wheelbase = float(rospy.get_param("~wheelbase", 1.74))
        # max_steer: 차량의 물리적 최대 조향각(rad) – CARLA 정규화에 사용 (fallback)
        self.max_steer = float(rospy.get_param("~max_steer", 0.5))
        # cmd_max_steer: 명령으로 허용할 최대 조향(rad) – Ackermann/제어 내부 클램프
        self.cmd_max_steer = float(rospy.get_param("~cmd_max_steer", 0.5))
        self.target_speed = float(rospy.get_param("~target_speed", 6.0))
        self.control_frequency = float(rospy.get_param("~control_frequency", 30.0))
        # Low-voltage speed gating + parking slowdown
        self.low_voltage_threshold = float(rospy.get_param("~low_voltage_threshold", 5.0))
        self.parking_dest_x = float(rospy.get_param("~parking_dest_x", -23.0))
        self.parking_dest_y = float(rospy.get_param("~parking_dest_y", -16.5))
        self.parking_speed = float(rospy.get_param("~parking_speed", 3.0))
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

        # State
        self.vehicles: Dict[str, carla.Actor] = {}
        self.states: Dict[str, Dict] = {}
        self.control_publishers: Dict[str, rospy.Publisher] = {}
        self.pose_publishers: Dict[str, rospy.Publisher] = {}
        self._tl_phase: Dict[str, Dict] = {}
        self._voltage: Dict[int, float] = {}
        self._speed_override: Dict[str, Dict[str, float]] = {}

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            self.states[role] = {
                "path": [],  # List[(x, y)]
                "position": None,
                "orientation": None,
                "twist": None,
                "current_index": 0,
                "s_profile": [],
                "progress_s": 0.0,
                "path_length": 0.0,
            }
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
        rospy.Subscriber("/imu_uplink", Uplink, self._uplink_cb, queue_size=10)
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
        s_profile, total_len = self._compute_path_profile(points)
        st = self.states[role]
        st["path"] = points
        st["current_index"] = 0
        st["s_profile"] = s_profile
        st["path_length"] = total_len
        st["progress_s"] = 0.0
        st["progress_fail_count"] = 0
        # rospy.loginfo_throttle(1.0, f"{role}: planned_path received ({len(points)} pts, len={total_len:.1f} m)")
        vehicle = self.vehicles.get(role)
        rear = self._rear_point(st, vehicle)
        if rear is not None:
            rx, ry, _ = rear
            proj = self._project_progress(
                points,
                s_profile,
                rx,
                ry,
                0.0,
                self.progress_backtrack_window_m,
            )
            if proj is not None:
                s_now, idx = proj
                st["progress_s"] = s_now
                st["current_index"] = idx
            else:
                # 초기에 경로 투영 실패 시 바로 근처 점으로 스냅
                snap = self._force_snap_progress(st, rx, ry)
                if snap is not None:
                    s_now, idx = snap
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
        px,
        py,
        prev_s: float,
        backtrack_window: float,
        heading: float = None,
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
                px,
                py,
                prev_s,
                backtrack_window,
                forward_window=forward_window,
                heading=heading,
            )
            if result is not None:
                return result
        return None

    def _project_progress_limited(
        self,
        path,
        s_profile,
        px,
        py,
        prev_s: float,
        backtrack_window: float,
        forward_window: float = None,
        heading: float = None,
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
        if best_index is None:
            return None
        # 추가 방어: 인덱스 경계 확인
        if best_index < 0 or best_index >= len(path) - 1 or best_index >= len(s_profile):
            return None
        return float(best_s), best_index

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

    def _sample_path_at_s(self, path, s_profile, s_target: float):
        """Arc-length 보간으로 정확한 목표점을 얻는다."""
        if len(path) < 2 or not s_profile or len(s_profile) != len(path):
            return None
        if s_target <= s_profile[0]:
            return path[0][0], path[0][1], 0, 0.0
        if s_target >= s_profile[-1]:
            return path[-1][0], path[-1][1], len(path) - 1, 0.0
        # 찾기
        for i in range(len(s_profile) - 1):
            if s_profile[i + 1] >= s_target:
                ds = max(1e-6, s_profile[i + 1] - s_profile[i])
                t = (s_target - s_profile[i]) / ds
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                tx = x1 + (x2 - x1) * t
                ty = y1 + (y2 - y1) * t
                return tx, ty, i, t
        return None

    def _select_target(self, st, x, y, lookahead_override: float = None) -> Tuple[float, float]:
        path = st.get("path") or []
        s_profile = st.get("s_profile") or []
        if len(path) < 2:
            return x, y
        # arc-length target selection with interpolation
        s_now = float(st.get("progress_s", 0.0))
        base_ld = float(self.lookahead_distance) if lookahead_override is None else float(lookahead_override)
        heading_err = float(st.get("heading_err_filt", st.get("heading_err", 0.0)))
        # 조향/헤딩 오차 기반 lookahead 축소 (LPF/데드존 반영)
        ld = base_ld / (1.0 + self.heading_ld_gain * abs(heading_err))
        ld = max(self.min_lookahead_m, min(self.max_lookahead_m, ld))
        s_target = s_now + max(0.05, ld)
        sample = self._sample_path_at_s(path, s_profile, s_target)
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
            fx,
            fy,
            prev_s,
            self.progress_backtrack_window_m,
            heading=yaw,
        )
        if proj is not None:
            s_now, idx = proj
            st["progress_s"] = s_now
            st["current_index"] = idx
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
            rospy.logwarn_throttle(
                2.0,
                f"{role}: progress projection failed #{count} (fx={fx:.1f}, fy={fy:.1f}); keeping previous s={prev_s:.1f}",
            )
            # 즉시 경로로 스냅하여 조향이 바로 경로를 향하도록
            reset = self._force_snap_progress(st, fx, fy)
            if reset is not None:
                s_now, idx = reset
                st["progress_s"] = s_now
                st["current_index"] = idx
                st["progress_fail_count"] = 0
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
        # Iterate over other controlled vehicles (based on odom states)
        for other_role, ost in self.states.items():
            if other_role == role:
                continue
            op = ost.get("position")
            if op is None:
                continue
            dx = float(op.x) - fx
            dy = float(op.y) - fy
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
                rospy.logwarn_throttle(
                    1.0,
                    f"{role}: stopping for {other_role} (dist={dist:.1f} m, rel={math.degrees(rel):.1f} deg)",
                )
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
        except Exception:
            self._tl_phase[iid] = {"stamp": rospy.Time.now(), "approaches": []}

    def _uplink_cb(self, msg: "Uplink") -> None:
        try:
            self._voltage[int(msg.vehicle_id)] = float(msg.voltage)
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
        hit = False
        min_gap = float("inf")
        min_info = None
        for data in self._tl_phase.values():
            approaches = data.get("approaches", [])
            for ap in approaches:
                for label, px, py in check_points:
                    if px >= ap["xmin"] and px <= ap["xmax"] and py >= ap["ymin"] and py <= ap["ymax"]:
                        color = int(ap.get("color", 0))  # 0=R,1=Y,2=G
                        color_name = "red" if color == 0 else ("yellow" if color == 1 else "green")
                        # Region-based hard stop
                        if color == 0 or (color == 1 and self.tl_yellow_policy.lower() != "permissive"):
                            rospy.loginfo_throttle(
                                0.5,
                                f"{role}: TL HIT ({label}) color={color_name} region=({ap['xmin']:.2f},{ap['xmax']:.2f},{ap['ymin']:.2f},{ap['ymax']:.2f}) pos=({px:.2f},{py:.2f}) speed_in={speed_cmd:.2f} -> 0",
                            )
                            hit = True
                            return 0.0
                        # In region but not stopping (green or permissive yellow)
                        rospy.loginfo_throttle(
                            0.5,
                            f"{role}: TL IN ({label}) color={color_name} region=({ap['xmin']:.2f},{ap['xmax']:.2f},{ap['ymin']:.2f},{ap['ymax']:.2f}) pos=({px:.2f},{py:.2f}) keep speed={speed_cmd:.2f}",
                        )
                        hit = True
                    # gap to bbox (0 if inside); use L2 of outside deltas
                    dx = 0.0
                    if px < ap["xmin"]:
                        dx = ap["xmin"] - px
                    elif px > ap["xmax"]:
                        dx = px - ap["xmax"]
                    dy = 0.0
                    if py < ap["ymin"]:
                        dy = ap["ymin"] - py
                    elif py > ap["ymax"]:
                        dy = py - ap["ymax"]
                    gap = math.hypot(dx, dy)
                    if gap < min_gap:
                        min_gap = gap
                        min_info = (label, px, py, ap, ap.get("color", 0))
        if not hit and min_info is not None:
            label, px, py, ap, color = min_info
            rospy.logdebug_throttle(
                1.0,
                f"{role}: TL no-hit ({label}) pos=({px:.2f},{py:.2f}) closest region=({ap['xmin']:.2f},{ap['xmax']:.2f},{ap['ymin']:.2f},{ap['ymax']:.2f}) gap={min_gap:.2f} color={int(color)} phase_cache={len(self._tl_phase)}",
            )
        return speed_cmd

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
            rospy.loginfo_throttle(
                0.5,
                f"{role}: cmd steer={steer:.3f} rad speed={speed:.2f} m/s",
            )


if __name__ == "__main__":
    try:
        SimpleMultiVehicleController()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleMultiVehicleController crashed: {e}")
        raise