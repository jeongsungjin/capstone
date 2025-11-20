#!/usr/bin/env python3

import bisect
import math
import sys
from collections import deque
from typing import Deque, Dict, Tuple, Optional, List

import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

CARLA_EGG = "/home/ctrl/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg"
if CARLA_EGG not in sys.path:
    sys.path.insert(0, CARLA_EGG)

try:
    import carla
except ImportError as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


class PlatoonManager:
    def __init__(self) -> None:
        rospy.init_node("platoon_manager", anonymous=False)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        # Parameters
        self.leader_role = str(rospy.get_param("~leader_role", "ego_vehicle_1"))
        self.follower_roles = [s.strip() for s in str(rospy.get_param("~follower_roles", "ego_vehicle_2,ego_vehicle_3")).split(",") if s.strip()]
        self.desired_gap_m = float(rospy.get_param("~desired_gap_m", 6.0))
        self.teleport_on_start = bool(rospy.get_param("~teleport_on_start", True))
        self.max_speed = float(rospy.get_param("~max_speed", 6.0))
        self.kp_gap = float(rospy.get_param("~kp_gap", 0.4))
        self.kd_gap = float(rospy.get_param("~kd_gap", 0.6))
        self.kp_gap_close = float(rospy.get_param("~kp_gap_close", 0.7))
        self.kd_gap_close = float(rospy.get_param("~kd_gap_close", 0.9))
        self.gap_deadband_m = float(rospy.get_param("~gap_deadband_m", 0.5))
        self.rel_speed_deadband = float(rospy.get_param("~rel_speed_deadband", 0.2))
        self.path_switch_max_dist_m = float(rospy.get_param("~path_switch_max_dist_m", 8.0))
        self.accel_limit_mps2 = float(rospy.get_param("~accel_limit_mps2", 1.2))
        history_default = max(60.0, self.desired_gap_m * (len(self.follower_roles) + 3))
        self.leader_track_history_m = float(rospy.get_param("~leader_track_history_m", history_default))
        self.leader_track_sample_ds = float(rospy.get_param("~leader_track_sample_ds", 0.5))
        self.leader_track_min_sample_ds = float(rospy.get_param("~leader_track_min_sample_ds", 0.2))
        self.follow_path_preview_m = float(rospy.get_param("~follow_path_preview_m", 15.0))
        self.follow_path_backfill_m = float(rospy.get_param("~follow_path_backfill_m", 1.0))
        self.follow_path_step_m = float(rospy.get_param("~follow_path_step_m", 0.8))
        self.follow_path_min_points = int(rospy.get_param("~follow_path_min_points", 3))
        self.min_follow_gap_m = float(rospy.get_param("~min_follow_gap_m", 2.0))
        self.gap_warmup_margin_m = float(rospy.get_param("~gap_warmup_margin_m", 1.0))
        self.start_gap_history_m = float(rospy.get_param("~start_gap_history_m", 5.0))
        self.start_gap_follow_m = float(rospy.get_param("~start_gap_follow_m", 0.5))
        # (simplified) no rolling tail / no grace switching

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # State
        self.actors: Dict[str, carla.Actor] = {}
        self.odom: Dict[str, Odometry] = {}
        self.leader_cmd: Optional[AckermannDrive] = None

        # Publishers
        # Publish directly to planned_path_* to avoid being overwritten by planner->relay
        self.path_pubs: Dict[str, rospy.Publisher] = {r: rospy.Publisher(f"/planned_path_{r}", Path, queue_size=1, latch=True) for r in self.follower_roles}
        self.override_pubs: Dict[str, rospy.Publisher] = {r: rospy.Publisher(f"/carla/{r}/vehicle_control_cmd_override", AckermannDrive, queue_size=1) for r in self.follower_roles}

        # Subscriptions
        rospy.Subscriber(f"/carla/{self.leader_role}/vehicle_control_cmd", AckermannDrive, self._cmd_cb, callback_args=self.leader_role, queue_size=1)
        # Odometry for leader and followers
        rospy.Subscriber(f"/carla/{self.leader_role}/odometry", Odometry, self._odom_cb, callback_args=self.leader_role, queue_size=10)
        for r in self.follower_roles:
            rospy.Subscriber(f"/carla/{r}/odometry", Odometry, self._odom_cb, callback_args=r, queue_size=10)
        # Subscribe predecessor commands for chain feed-forward
        for pred in [self.leader_role] + self.follower_roles[:-1]:
            rospy.Subscriber(f"/carla/{pred}/vehicle_control_cmd", AckermannDrive, self._cmd_cb, callback_args=pred, queue_size=1)

        rospy.Timer(rospy.Duration(0.1), self._tick)
        rospy.Timer(rospy.Duration(1.0), self._refresh_actors, oneshot=True)
        rospy.Timer(rospy.Duration(2.0), self._maybe_teleport_followers, oneshot=True)
        self._cmd_cache: Dict[str, AckermannDrive] = {}
        self._last_speed_cmd: Dict[str, float] = {}
        self._last_cmd_time: Dict[str, float] = {}
        self._leader_track: Deque[Tuple[float, float, float, float]] = deque()
        self._leader_track_header: Optional[Header] = None
        self._leader_track_last_s: float = 0.0
        self._leader_track_last_xy: Optional[Tuple[float, float]] = None
        self._leader_track_min_gap = min(self.follow_path_preview_m + self.follow_path_backfill_m, self.leader_track_history_m)
        # no last-valid hold in simplified version

    # ----------------- CARLA helpers -----------------
    def _refresh_actors(self, _evt=None) -> None:
        actors = self.world.get_actors().filter("vehicle.*")
        by_role: Dict[str, carla.Actor] = {}
        for a in actors:
            role = a.attributes.get("role_name", "")
            if not role:
                continue
            by_role[role] = a
        self.actors = by_role

    def _maybe_teleport_followers(self, _evt=None) -> None:
        if not self.teleport_on_start:
            return
        leader = self.actors.get(self.leader_role)
        if leader is None:
            self._refresh_actors()
            leader = self.actors.get(self.leader_role)
        if leader is None:
            rospy.logwarn("platoon_manager: leader actor not found; skip teleport")
            return
        try:
            base = leader.get_transform()
        except Exception:
            return
        yaw_rad = math.radians(base.rotation.yaw)
        fx = math.cos(yaw_rad)
        fy = math.sin(yaw_rad)
        # Place each follower behind the leader along -forward
        for idx, role in enumerate(self.follower_roles, start=1):
            actor = self.actors.get(role)
            if actor is None:
                continue
            offset = self.desired_gap_m * float(idx)
            loc = carla.Location(
                x=base.location.x - fx * offset,
                y=base.location.y - fy * offset,
                z=base.location.z,
            )
            tf = carla.Transform(location=loc, rotation=base.rotation)
            try:
                actor.set_transform(tf)
            except Exception:
                pass

    def _append_leader_track(self, odom: Odometry) -> None:
        pos = odom.pose.pose.position
        yaw = self._yaw_from_quat(odom.pose.pose.orientation)
        if self._leader_track_header is None:
            frame_id = odom.header.frame_id if odom.header and odom.header.frame_id else "map"
            self._leader_track_header = Header(frame_id=frame_id)
        if not self._leader_track:
            self._leader_track.append((0.0, float(pos.x), float(pos.y), float(yaw)))
            self._leader_track_last_xy = (float(pos.x), float(pos.y))
            self._leader_track_last_s = 0.0
            return
        if self._leader_track_last_xy is None:
            self._leader_track_last_xy = (float(pos.x), float(pos.y))
        dx = float(pos.x) - self._leader_track_last_xy[0]
        dy = float(pos.y) - self._leader_track_last_xy[1]
        dist = math.hypot(dx, dy)
        if dist < max(1e-3, self.leader_track_sample_ds):
            return
        last_s = self._leader_track[-1][0]
        s_val = last_s + dist
        self._leader_track.append((s_val, float(pos.x), float(pos.y), float(yaw)))
        self._leader_track_last_xy = (float(pos.x), float(pos.y))
        self._leader_track_last_s = s_val
        self._trim_leader_track(s_val)

    def _trim_leader_track(self, current_s: float) -> None:
        min_s = current_s - max(0.0, self.leader_track_history_m)
        while self._leader_track and self._leader_track[0][0] < min_s:
            self._leader_track.popleft()

    def _current_history_span(self) -> float:
        if len(self._leader_track) < 2:
            return 0.0
        return self._leader_track[-1][0] - self._leader_track[0][0]

    def _active_gap_for_index(self, index: float, history_span: Optional[float] = None) -> float:
        if index <= 0.0:
            return 0.0
        if history_span is None:
            history_span = self._current_history_span()
        base_gap = max(0.0, self.desired_gap_m * index)
        min_gap = max(0.0, self.min_follow_gap_m * index)
        if history_span < max(0.0, self.start_gap_history_m):
            return max(0.0, self.start_gap_follow_m)
        available = max(0.0, history_span - max(0.0, self.gap_warmup_margin_m))
        if available <= 1e-3:
            return min_gap if min_gap > 0.0 else 0.0
        return max(min_gap, min(base_gap, available))

    def _publish_follower_paths(self) -> None:
        if len(self._leader_track) < 2:
            return
        latest_s = self._leader_track[-1][0]
        earliest_s = self._leader_track[0][0]
        available = latest_s - earliest_s
        min_required = max(
            0.5,
            self.follow_path_step_m * max(1, self.follow_path_min_points - 1),
        )
        if available < min_required:
            return
        history_span = available
        for idx, role in enumerate(self.follower_roles, start=1):
            pub = self.path_pubs.get(role)
            if pub is None:
                continue
            gap = self._active_gap_for_index(float(idx), history_span)
            path_msg = self._build_follower_path(gap)
            if path_msg is None or not path_msg.poses:
                continue
            pub.publish(path_msg)

    def _build_follower_path(self, gap_m: float) -> Optional[Path]:
        if len(self._leader_track) < 2:
            return None
        track_list = list(self._leader_track)
        earliest_s = track_list[0][0]
        latest_s = track_list[-1][0]
        history_span = latest_s - earliest_s
        effective_gap = min(
            gap_m,
            max(0.0, history_span - max(0.2, self.follow_path_step_m)),
        )
        target_s = latest_s - effective_gap
        target_s = max(target_s, earliest_s + 1e-3)
        start_s = max(earliest_s, target_s - max(0.0, self.follow_path_backfill_m))
        end_s = min(latest_s, target_s + max(0.0, self.follow_path_preview_m))
        if end_s - start_s < 0.5:
            return None
        samples = self._sample_track_segment(track_list, start_s, end_s, max(0.1, self.follow_path_step_m))
        if len(samples) < max(2, self.follow_path_min_points):
            return None
        header = Header(stamp=rospy.Time.now())
        if self._leader_track_header is not None and self._leader_track_header.frame_id:
            header.frame_id = self._leader_track_header.frame_id
        else:
            header.frame_id = "map"
        path = Path(header=header)
        for x, y, yaw in samples:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            path.poses.append(pose)
        return path

    def _sample_track_segment(
        self,
        track: List[Tuple[float, float, float, float]],
        start_s: float,
        end_s: float,
        step: float,
    ) -> List[Tuple[float, float, float]]:
        if not track:
            return []
        step = max(0.05, step)
        samples: List[Tuple[float, float, float]] = []
        s = start_s
        while s <= end_s + 1e-6:
            point = self._interpolate_track(track, s)
            if point is not None:
                samples.append(point)
            s += step
        # ensure final point at end_s
        if samples:
            last = samples[-1]
            if abs(end_s - last[0]) > 1e-3:
                final_point = self._interpolate_track(track, end_s)
                if final_point is not None:
                    samples.append(final_point)
        return [(x, y, yaw) for (_s, x, y, yaw) in samples]

    def _interpolate_track(
        self,
        track: List[Tuple[float, float, float, float]],
        target_s: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        if not track:
            return None
        if target_s <= track[0][0]:
            return track[0]
        if target_s >= track[-1][0]:
            return track[-1]
        s_values = [item[0] for item in track]
        idx = bisect.bisect_left(s_values, target_s)
        if idx <= 0:
            return track[0]
        if idx >= len(track):
            return track[-1]
        s1, x1, y1, yaw1 = track[idx - 1]
        s2, x2, y2, yaw2 = track[idx]
        span = s2 - s1
        if span <= 1e-6:
            return track[idx]
        ratio = (target_s - s1) / span
        x = x1 + (x2 - x1) * ratio
        y = y1 + (y2 - y1) * ratio
        yaw = self._interp_angle(yaw1, yaw2, ratio)
        return (target_s, x, y, yaw)

    def _cmd_cb(self, msg: AckermannDrive, role: str) -> None:
        self._cmd_cache[role] = msg
        if role == self.leader_role:
            self.leader_cmd = msg

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        self.odom[role] = msg
        if role == self.leader_role:
            self._append_leader_track(msg)

    # ----------------- Spacing control -----------------
    def _tick(self, _evt) -> None:
        if self.leader_cmd is None:
            return
        lead_odom = self.odom.get(self.leader_role)
        if lead_odom is None:
            return
        self._publish_follower_paths()
        # Leader forward unit vector from odometry orientation yaw
        # Predecessor chain: leader -> follower_1 -> follower_2 -> ...
        chain = [self.leader_role] + self.follower_roles
        for idx, role in enumerate(self.follower_roles):
            foll_odom = self.odom.get(role)
            if foll_odom is None:
                continue
            pred_role = chain[idx]  # leader for first follower, previous follower otherwise
            pred_odom = self.odom.get(pred_role)
            if pred_odom is None:
                continue
            pyaw = self._yaw_from_quat(pred_odom.pose.pose.orientation)
            fwd = (math.cos(pyaw), math.sin(pyaw))
            gap, rel_speed = self._gap_and_rel_speed(pred_odom, foll_odom, fwd)
            desired_gap = self._active_gap_for_index(1.0)
            err = gap - desired_gap
            # Deadband to reduce chattering
            if abs(err) < max(0.0, self.gap_deadband_m):
                err = 0.0
            if abs(rel_speed) < max(0.0, self.rel_speed_deadband):
                rel_speed = 0.0
            # Feed-forward base speed from predecessor command if available; fallback to predecessor odometry
            pred_cmd = self._cmd_cache.get(pred_role)
            if pred_cmd is not None:
                base = float(pred_cmd.speed)
            elif self.leader_cmd is not None and pred_role == self.leader_role:
                base = float(self.leader_cmd.speed)
            else:
                pv = pred_odom.twist.twist.linear
                base = math.hypot(pv.x, pv.y)
            base = max(0.0, min(self.max_speed, base))
            # PD correction relative to predecessor (asymmetric)
            if err < 0.0:
                kp = self.kp_gap_close
                kd = self.kd_gap_close
            else:
                kp = self.kp_gap
                kd = self.kd_gap
            corr = kp * err + kd * rel_speed
            desired = base + corr
            desired = max(0.0, min(self.max_speed, desired))
            # Rate limiting (accel/decel)
            now = rospy.Time.now().to_sec()
            last_v = float(self._last_speed_cmd.get(role, 0.0))
            last_t = float(self._last_cmd_time.get(role, now))
            dt = max(1e-3, now - last_t)
            dv_max = self.accel_limit_mps2 * dt
            speed = max(last_v - dv_max, min(last_v + dv_max, desired))
            # Clamp
            if speed < 0.0:
                speed = 0.0
            if speed > self.max_speed:
                speed = self.max_speed
            cmd = AckermannDrive()
            # Speed-only override: do not override steering
            cmd.steering_angle = 0.0
            cmd.speed = float(speed)
            pub = self.override_pubs.get(role)
            if pub is not None:
                pub.publish(cmd)
            self._last_speed_cmd[role] = speed
            self._last_cmd_time[role] = now

    @staticmethod
    def _yaw_from_quat(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _angle_diff_rad(a: float, b: float) -> float:
        diff = a - b
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        return abs(diff)

    @staticmethod
    def _interp_angle(a: float, b: float, ratio: float) -> float:
        diff = b - a
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        return a + diff * ratio

    @staticmethod
    def _gap_and_rel_speed(lead: Odometry, foll: Odometry, fwd: Tuple[float, float]) -> Tuple[float, float]:
        dx = lead.pose.pose.position.x - foll.pose.pose.position.x
        dy = lead.pose.pose.position.y - foll.pose.pose.position.y
        # Longitudinal projection along leader forward
        gap = max(0.0, dx * fwd[0] + dy * fwd[1])
        lv = lead.twist.twist.linear
        fv = foll.twist.twist.linear
        rel_vx = (lv.x - fv.x)
        rel_vy = (lv.y - fv.y)
        rel_speed = rel_vx * fwd[0] + rel_vy * fwd[1]
        return gap, rel_speed


if __name__ == "__main__":
    try:
        PlatoonManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


