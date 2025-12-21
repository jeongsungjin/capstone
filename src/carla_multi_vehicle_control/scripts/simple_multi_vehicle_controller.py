#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple

import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header, Float32
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

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.lookahead_distance = float(rospy.get_param("~lookahead_distance", 1.0))
        # Curvature-adaptive lookahead (straight: large, sharp turn: small)
        self.ld_min = float(rospy.get_param("~ld_min", 0.5))
        self.ld_max = float(rospy.get_param("~ld_max", 3.0))
        self.ld_kappa_max = float(rospy.get_param("~ld_kappa_max", 0.3))  # rad per meter
        self.ld_window_m = float(rospy.get_param("~ld_window_m", 3.0))
        self.wheelbase = float(rospy.get_param("~wheelbase", 1.74))
        # max_steer: 차량의 물리적 최대 조향각(rad) – CARLA 정규화에 사용 (fallback)
        self.max_steer = float(rospy.get_param("~max_steer", 0.5))
        # cmd_max_steer: 명령으로 허용할 최대 조향(rad) – Ackermann/제어 내부 클램프
        self.cmd_max_steer = float(rospy.get_param("~cmd_max_steer", 0.5))
        self.target_speed = float(rospy.get_param("~target_speed", 20.0))
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

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            self.states[role] = {
                "path": [],  # List[(x, y)]
                "position": None,
                "orientation": None,
                "current_index": 0,
                "s_profile": [],
                "progress_s": 0.0,
                "path_length": 0.0,
                "obstacle_stop_s": -1.0,  # 장애물 정지 arc-length (-1.0 = 정지 없음)
            }
            path_topic = f"/planned_path_{role}"
            odom_topic = f"/carla/{role}/odometry"
            cmd_topic = f"/carla/{role}/vehicle_control_cmd"
            pose_topic = f"/{role}/pose"
            stop_topic = f"/obstacle_stop_{role}"
            rospy.Subscriber(path_topic, Path, self._path_cb, callback_args=role, queue_size=1)
            rospy.Subscriber(odom_topic, Odometry, self._odom_cb, callback_args=role, queue_size=10)
            rospy.Subscriber(stop_topic, Float32, self._obstacle_stop_cb, callback_args=role, queue_size=1)
            self.control_publishers[role] = rospy.Publisher(cmd_topic, AckermannDrive, queue_size=1)
            self.pose_publishers[role] = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)

        # Subscribe traffic light phase if message is available
        if TrafficLightPhase is not None:
            rospy.Subscriber("/traffic_phase", TrafficLightPhase, self._tl_cb, queue_size=5)
        rospy.Subscriber("/uplink", Uplink, self._uplink_cb, queue_size=10)

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

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        st = self.states[role]
        st["position"] = msg.pose.pose.position
        st["orientation"] = msg.pose.pose.orientation
        # also publish pose for tools
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pose_publishers[role].publish(pose_msg)

    def _obstacle_stop_cb(self, msg: Float32, role: str) -> None:
        """장애물 정지 arc-length 콜백"""
        st = self.states[role]
        stop_s = float(msg.data)
        st["obstacle_stop_s"] = stop_s
        if stop_s >= 0:
            rospy.loginfo_throttle(2.0, f"{role}: obstacle stop_s = {stop_s:.1f}m")
        else:
            rospy.loginfo_throttle(2.0, f"{role}: obstacle stop cleared")

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
        ld = float(self.lookahead_distance) if lookahead_override is None else float(lookahead_override)
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
        else:
            count = st.setdefault("progress_fail_count", 0) + 1
            st["progress_fail_count"] = count
            rospy.logwarn_throttle(
                2.0,
                f"{role}: progress projection failed #{count} (fx={fx:.1f}, fy={fy:.1f}); keeping previous s={prev_s:.1f}",
            )
            # If we have too many consecutive failures, try to hard-reset onto the path
            if count >= 3:
                reset = self._force_snap_progress(st, fx, fy)
                if reset is not None:
                    s_now, idx = reset
                    st["progress_s"] = s_now
                    st["current_index"] = idx
                    st["progress_fail_count"] = 0
                    rospy.loginfo(f"{role}: progress reset to s={s_now:.1f} after repeated projection failures")
        if proj is not None:
            st["progress_fail_count"] = 0
        # Curvature-adaptive lookahead: reduce Ld on sharp turns, increase on straights
        ld_dynamic = self.lookahead_distance
        try:
            ld_dynamic = self._compute_adaptive_lookahead(st)
        except Exception:
            ld_dynamic = self.lookahead_distance
        tx, ty = self._select_target(st, fx, fy, lookahead_override=ld_dynamic)
        dx = tx - fx
        dy = ty - fy
        alpha = math.atan2(dy, dx) - yaw
        while alpha > math.pi:
            alpha -= 2.0 * math.pi
        while alpha < -math.pi:
            alpha += 2.0 * math.pi
        Ld = math.hypot(dx, dy)
        if Ld < 1e-3:
            return 0.0, 0.0
        steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)
        # 명령 상한으로 클램프 (라디안)
        steer = max(-self.cmd_max_steer, min(self.cmd_max_steer, steer))
        speed = self.target_speed
        # Apply traffic light gating
        speed = self._apply_tl_gating(vehicle, fx, fy, speed)
        # Apply forward collision gating (vehicles in front cone within distance)
        speed = self._apply_collision_gating(role, fx, fy, yaw, speed)
        # Apply parking slowdown when low-voltage path ends at parking dest
        speed = self._apply_parking_speed_limit(role, st, speed)
        
        # 장애물 정지 arc-length 체크
        obstacle_stop_s = float(st.get("obstacle_stop_s", -1.0))
        current_progress = float(st.get("progress_s", 0.0))
        if obstacle_stop_s >= 0 and current_progress >= obstacle_stop_s - 1.0:  # 1m 마진
            rospy.logwarn_throttle(2.0, f"{role}: stopping at obstacle (progress={current_progress:.1f}m >= stop_s={obstacle_stop_s:.1f}m)")
            speed = 0.0
        
        # 경로 끝 도달 시 정지
        dist_to_end = self._distance_to_path_end(st)
        if dist_to_end is not None and dist_to_end <= 2.0:
            speed = 0.0
        
        return steer, speed

    def _apply_collision_gating(self, role: str, fx: float, fy: float, yaw: float, speed_cmd: float) -> float:
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

    def _compute_adaptive_lookahead(self, st) -> float:
        """
        Compute curvature-adaptive lookahead within [ld_min, ld_max].
        Map kappa (rad/m) to lookahead: L = Lmax - (Lmax-Lmin) * clamp(kappa/kappa_max, 0, 1)
        """
        path = st.get("path") or []
        s_profile = st.get("s_profile") or []
        idx = st.get("current_index", 0)
        if len(path) < 3 or not s_profile:
            return max(self.ld_min, min(self.ld_max, self.lookahead_distance))
        # Find a forward index at least ld_window_m ahead in arc-length
        s_now = float(st.get("progress_s", 0.0))
        target_s = s_now + max(0.5, float(self.ld_window_m))
        j = idx
        while j + 1 < len(s_profile) and float(s_profile[j]) < target_s:
            j += 1
        j = min(j, len(path) - 1)
        if j <= idx:
            return max(self.ld_min, min(self.ld_max, self.lookahead_distance))
        # Estimate heading change between segments around idx and j
        def heading(p, q):
            return math.atan2(q[1] - p[1], q[0] - p[0])
        i2 = min(idx + 1, len(path) - 1)
        j1 = max(j - 1, 0)
        h0 = heading(path[idx], path[i2])
        h1 = heading(path[j1], path[j])
        dtheta = (h1 - h0 + math.pi) % (2.0 * math.pi) - math.pi
        ds = max(1e-3, float(s_profile[j]) - float(s_profile[idx]))
        kappa = abs(dtheta) / ds
        ratio = max(0.0, min(1.0, kappa / max(1e-6, float(self.ld_kappa_max))))
        L = float(self.ld_max) - (float(self.ld_max) - float(self.ld_min)) * ratio
        return max(float(self.ld_min), min(float(self.ld_max), L))

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

    def _is_low_voltage(self, role: str) -> bool:
        try:
            vid = int(role.split("_")[-1])
            v = self._voltage.get(vid, float("inf"))
            return v <= float(self.low_voltage_threshold)
        except Exception:
            return False

    def _apply_tl_gating(self, vehicle, fx: float, fy: float, speed_cmd: float) -> float:
        if not self._tl_phase:
            return speed_cmd
        for data in self._tl_phase.values():
            approaches = data.get("approaches", [])
            for ap in approaches:
                if fx >= ap["xmin"] and fx <= ap["xmax"] and fy >= ap["ymin"] and fy <= ap["ymax"]:
                    color = int(ap.get("color", 0))  # 0=R,1=Y,2=G
                    # Region-based hard stop
                    if color == 0:
                        return 0.0
                    if color == 1 and self.tl_yellow_policy.lower() != "permissive":
                        return 0.0
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
        msg.steering_angle = float(steer)
        msg.speed = float(speed)
        self.control_publishers[role].publish(msg)

    def _control_loop(self, _evt) -> None:
        for role, st in self.states.items():
            vehicle = self.vehicles.get(role)
            if vehicle is None:
                continue
            steer, speed = self._compute_control(st, vehicle, role)
            if steer is None:
                continue
            self._apply_carla_control(vehicle, steer, speed)
            self._publish_ackermann(role, steer, speed)


if __name__ == "__main__":
    try:
        SimpleMultiVehicleController()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleMultiVehicleController crashed: {e}")
        raise
