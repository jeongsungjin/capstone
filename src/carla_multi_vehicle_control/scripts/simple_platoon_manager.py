#!/usr/bin/env python3

import math
from typing import Dict, List, Optional, Tuple

import rospy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header


class SimplePlatoonManager:
    """
    단순 플래툰:
    - 리더 global_path를 모든 planned_path_*에 그대로 공유
    - 체인: 리더 -> follower1 -> follower2 ...
    - 속도: 선행 속도 복사, 너무 붙으면 감속/정지 (PD 감속은 선택)
    """

    def __init__(self) -> None:
        rospy.init_node("simple_platoon_manager", anonymous=False)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.leader_role = str(rospy.get_param("~leader_role", "ego_vehicle_2"))
        followers_str = str(rospy.get_param("~follower_roles", "ego_vehicle_1,ego_vehicle_3"))
        self.follower_roles: List[str] = [r.strip() for r in followers_str.split(",") if r.strip()][: max(0, self.num_vehicles - 1)]

        self.desired_gap_m = float(rospy.get_param("~desired_gap_m", 6.0))
        self.gap_deadband_m = float(rospy.get_param("~gap_deadband_m", 0.5))
        self.max_speed_mps = float(rospy.get_param("~max_speed_mps", 8.0))
        self.min_follow_speed_mps = float(rospy.get_param("~min_follow_speed_mps", 1.0))
        self.stop_gap_m = float(rospy.get_param("~stop_gap_m", 0.8))
        self.follower_speed_limits: Dict[str, float] = rospy.get_param("~follower_speed_limits", {})
        # 간격 유지용 약한 PD (격차가 벌어지면 가속, 붙으면 감속)
        self.kp_follow = float(rospy.get_param("~kp_follow", 0.3))
        self.kd_follow = float(rospy.get_param("~kd_follow", 0.1))
        # 곡선 포함 아크길이 기반 간격 측정 사용 여부
        self.use_arclength_gap = bool(rospy.get_param("~use_arclength_gap", True))
        self.max_off_path_m = float(rospy.get_param("~max_off_path_m", 3.0))
        self.force_min_if_lead_moves = bool(rospy.get_param("~force_min_if_lead_moves", True))
        self.log_verbose = bool(rospy.get_param("~log_verbose", True))
        # accel_limit_mps2 <= 0 이면 가속 제한 없음
        self.accel_limit_mps2 = float(rospy.get_param("~accel_limit_mps2", 0.0))
        self.control_hz = float(rospy.get_param("~control_hz", 10.0))

        self.path_source_topic = str(rospy.get_param("~path_source_topic", f"/global_path_{self.leader_role}"))
        # 0 또는 음수이면 주기적 재발행을 하지 않는다.
        self.republish_hz = float(rospy.get_param("~republish_hz", 0.0))

        self.publish_leader_path = bool(rospy.get_param("~publish_leader_path", True))

        self.all_roles = [self.leader_role] + self.follower_roles
        # 등간격 체인: 리더(1) -> 2, 2 -> 3
        self.spacing_pairs: List[Tuple[str, str]] = []
        if "ego_vehicle_2" in self.follower_roles:
            self.spacing_pairs.append((self.leader_role, "ego_vehicle_2"))
        if "ego_vehicle_3" in self.follower_roles:
            self.spacing_pairs.append(("ego_vehicle_2", "ego_vehicle_3"))
        if not self.spacing_pairs:
            self.spacing_pairs = [(self.leader_role, r) for r in self.follower_roles]

        self.path_msg: Optional[Path] = None
        self.path_pts: List[Tuple[float, float]] = []
        self.path_s: List[float] = []
        self.odom: Dict[str, Odometry] = {}
        self.cmd_cache: Dict[str, AckermannDrive] = {}
        self._last_speed_cmd: Dict[str, float] = {}
        self._last_cmd_time: Dict[str, float] = {}
        self._last_gap: Dict[Tuple[str, str], float] = {}

        pub_roles = self.all_roles if self.publish_leader_path else self.follower_roles
        self.path_pubs: Dict[str, rospy.Publisher] = {
            r: rospy.Publisher(f"/planned_path_{r}", Path, queue_size=1, latch=True) for r in pub_roles
        }
        # RViz에서 global_path_* 를 그릴 때도 모두 동일 경로를 보도록 추가 퍼블리셔
        self.global_path_pubs: Dict[str, rospy.Publisher] = {
            r: rospy.Publisher(f"/global_path_{r}", Path, queue_size=1, latch=True) for r in pub_roles
        }
        self.override_pubs: Dict[str, rospy.Publisher] = {
            r: rospy.Publisher(f"/carla/{r}/vehicle_control_cmd_override", AckermannDrive, queue_size=5)
            for r in self.follower_roles
        }

        rospy.Subscriber(self.path_source_topic, Path, self._path_cb, queue_size=1)
        for role in self.all_roles:
            rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role, queue_size=10)
            rospy.Subscriber(f"/carla/{role}/vehicle_control_cmd", AckermannDrive, self._cmd_cb, callback_args=role, queue_size=5)

        rospy.Timer(rospy.Duration(1.0 / max(0.1, self.control_hz)), self._tick_control)
        if self.republish_hz > 0.0:
            rospy.Timer(rospy.Duration(1.0 / max(0.1, self.republish_hz)), self._republish_path)

        rospy.loginfo("[platoon] leader=%s followers=%s path=%s", self.leader_role, self.follower_roles, self.path_source_topic)

    def _path_cb(self, msg: Path) -> None:
        if not msg.poses:
            return
        fixed = self._ensure_orientations(msg)
        self.path_msg = fixed
        self._prepare_path_data(fixed)
        # if self.log_verbose:
        #     rospy.loginfo("[platoon] shared leader path (%d pts)", len(msg.poses))
        self._publish_path_to_all(fixed)

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        self.odom[role] = msg

    def _cmd_cb(self, msg: AckermannDrive, role: str) -> None:
        self.cmd_cache[role] = msg

    def _republish_path(self, _evt=None) -> None:
        if self.path_msg is None or not self.path_msg.poses:
            return
        self._publish_path_to_all(self.path_msg)

    def _publish_path_to_all(self, msg: Path) -> None:
        for _role, pub in self.path_pubs.items():
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=msg.header.frame_id)
            path_copy.poses = msg.poses
            pub.publish(path_copy)
        # RViz에서 global_path_* 를 그릴 때도 동일 경로를 보게 복제
        for _role, pub in self.global_path_pubs.items():
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=msg.header.frame_id)
            path_copy.poses = msg.poses
            pub.publish(path_copy)

    def _tick_control(self, _evt) -> None:
        if self.path_msg is None:
            return
        if self.odom.get(self.leader_role) is None:
            return

        for pred_role, role in self.spacing_pairs:
            pred_odom = self.odom.get(pred_role)
            foll_odom = self.odom.get(role)
            if pred_odom is None or foll_odom is None:
                continue

            prev_gap = self._last_gap.get((pred_role, role))
            if self.use_arclength_gap and self.path_pts:
                gap, rel_speed = self._gap_and_rel_speed_arclength(pred_odom, foll_odom)
            else:
                gap, rel_speed = self._gap_and_rel_speed_simple(pred_odom, foll_odom)
            # arclength 실패/유클리드에서의 튀는 값을 완화하기 위해 이전 gap과 블렌딩
            if prev_gap is not None and gap is not None:
                gap = 0.6 * gap + 0.4 * prev_gap
            self._last_gap[(pred_role, role)] = gap
            base_speed = self._base_speed(pred_role, pred_odom)

            # 기본: 리더 속도를 복사하고, 등간격 오차에 약한 PD 보정
            err = gap - self.desired_gap_m
            rel = rel_speed
            if abs(err) < self.gap_deadband_m:
                err = 0.0
            # 과도한 오차로 인한 급가감속 방지
            err = max(-5.0, min(5.0, err))
            desired = base_speed + self.kp_follow * err + self.kd_follow * rel

            if gap <= max(0.0, self.stop_gap_m):
                if base_speed > 0.2 and self.force_min_if_lead_moves:
                    desired = max(self.min_follow_speed_mps, desired)
                else:
                    desired = 0.0
            else:
                desired = max(desired, self.min_follow_speed_mps)
                desired = min(desired, self._role_speed_limit(role))

            now = rospy.Time.now().to_sec()
            last_v = float(self._last_speed_cmd.get(role, 0.0))
            last_t = float(self._last_cmd_time.get(role, now))
            if self.accel_limit_mps2 > 0.0:
                dv_max = self.accel_limit_mps2 * max(1e-3, now - last_t)
                desired = max(last_v - dv_max, min(last_v + dv_max, desired))

            cmd = AckermannDrive()
            cmd.steering_angle = 0.0
            cmd.speed = desired
            pub = self.override_pubs.get(role)
            if pub is not None:
                pub.publish(cmd)
            if self.log_verbose:
                rospy.loginfo("[platoon-simple] role=%s gap=%.2f base=%.2f cmd=%.2f rel=%.2f", role, gap, base_speed, desired, rel_speed)
            self._last_speed_cmd[role] = desired
            self._last_cmd_time[role] = now

    def _base_speed(self, role: str, odom: Odometry) -> float:
        cmd = self.cmd_cache.get(role)
        v_cmd = 0.0
        if cmd is not None:
            try:
                v_cmd = max(0.0, float(cmd.speed))
            except Exception:
                v_cmd = 0.0
        lin = odom.twist.twist.linear
        v_odom = math.hypot(lin.x, lin.y)
        # 실제 주행 속도가 명령보다 빠르면 odom 속도를 따름
        return max(v_cmd, v_odom)

    def _role_speed_limit(self, role: str) -> float:
        try:
            if role in self.follower_speed_limits:
                return float(self.follower_speed_limits[role])
        except Exception:
            pass
        return self.max_speed_mps

    def _ensure_orientations(self, msg: Path) -> Path:
        """경로에 orientation이 비어 있으면 좌표 기울기로 보완."""
        pts: List[Tuple[float, float]] = []
        for ps in msg.poses:
            pts.append((ps.pose.position.x, ps.pose.position.y))
        need_fill = True
        for ps in msg.poses:
            q = ps.pose.orientation
            if abs(q.w) > 1e-3 or abs(q.z) > 1e-3 or abs(q.x) > 1e-3 or abs(q.y) > 1e-3:
                need_fill = False
                break
        if len(pts) < 2 or not need_fill:
            return msg
        new_msg = Path()
        new_msg.header = msg.header
        for i, ps in enumerate(msg.poses):
            x, y = pts[i]
            if i < len(pts) - 1:
                nx, ny = pts[i + 1]
            else:
                nx, ny = pts[i]
            dx = nx - x
            dy = ny - y
            yaw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else 0.0
            qz = math.sin(0.5 * yaw)
            qw = math.cos(0.5 * yaw)
            p = ps
            p.pose.orientation.x = 0.0
            p.pose.orientation.y = 0.0
            p.pose.orientation.z = qz
            p.pose.orientation.w = qw
            new_msg.poses.append(p)
        return new_msg

    def _prepare_path_data(self, msg: Path) -> None:
        pts: List[Tuple[float, float]] = []
        for ps in msg.poses:
            pts.append((ps.pose.position.x, ps.pose.position.y))
        if len(pts) < 2:
            self.path_pts = []
            self.path_s = []
            return
        s: List[float] = [0.0]
        total = 0.0
        for i in range(1, len(pts)):
            step = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            total += step
            s.append(total)
        self.path_pts = pts
        self.path_s = s

    def _project_s(self, x: float, y: float) -> Optional[Tuple[float, float, int]]:
        if len(self.path_pts) < 2 or len(self.path_s) != len(self.path_pts):
            return None
        best_dist2 = None
        best_s = None
        best_idx = None
        for i in range(len(self.path_pts) - 1):
            x1, y1 = self.path_pts[i]
            x2, y2 = self.path_pts[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg = dx * dx + dy * dy
            if seg < 1e-9:
                continue
            t = ((x - x1) * dx + (y - y1) * dy) / seg
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy
            dist2 = (x - px) * (x - px) + (y - py) * (y - py)
            s_proj = self.path_s[i] + math.sqrt(seg) * t
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_s = s_proj
                best_idx = i
        if best_s is None or best_dist2 is None or best_idx is None:
            return None
        return best_s, math.sqrt(best_dist2), best_idx

    def _gap_and_rel_speed_arclength(self, lead: Odometry, foll: Odometry) -> Tuple[float, float]:
        lead_pos = lead.pose.pose.position
        foll_pos = foll.pose.pose.position
        lead_proj = self._project_s(lead_pos.x, lead_pos.y)
        foll_proj = self._project_s(foll_pos.x, foll_pos.y)
        if lead_proj is None or foll_proj is None:
            return self._gap_and_rel_speed_simple(lead, foll)
        lead_s, lead_d, lead_idx = lead_proj
        foll_s, foll_d, foll_idx = foll_proj
        if lead_d > self.max_off_path_m or foll_d > self.max_off_path_m:
            return self._gap_and_rel_speed_simple(lead, foll)
        gap = lead_s - foll_s
        # path tangent (follower segment 기준)
        idx = min(foll_idx, len(self.path_pts) - 2)
        x1, y1 = self.path_pts[idx]
        x2, y2 = self.path_pts[idx + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return gap, 0.0
        tx = dx / seg_len
        ty = dy / seg_len
        lv = lead.twist.twist.linear
        fv = foll.twist.twist.linear
        v_lead = lv.x * tx + lv.y * ty
        v_foll = fv.x * tx + fv.y * ty
        rel_speed = v_lead - v_foll
        return gap, rel_speed

    @staticmethod
    def _gap_and_rel_speed_simple(lead: Odometry, foll: Odometry) -> Tuple[float, float]:
        dx = lead.pose.pose.position.x - foll.pose.pose.position.x
        dy = lead.pose.pose.position.y - foll.pose.pose.position.y
        gap = math.hypot(dx, dy)
        if gap < 1e-3:
            return 0.0, 0.0
        ux = dx / gap
        uy = dy / gap
        lv = lead.twist.twist.linear
        fv = foll.twist.twist.linear
        rel_vx = lv.x - fv.x
        rel_vy = lv.y - fv.y
        rel_speed = rel_vx * ux + rel_vy * uy
        return gap, rel_speed


def main() -> None:
    try:
        SimplePlatoonManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

