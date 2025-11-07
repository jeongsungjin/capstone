#!/usr/bin/env python3

import math
import sys
from typing import Dict, Tuple, Optional

import rospy
from ackermann_msgs.msg import AckermannDrive
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
        self.desired_gap_m = float(rospy.get_param("~desired_gap_m", 9.0))
        self.teleport_on_start = bool(rospy.get_param("~teleport_on_start", True))
        self.max_speed = float(rospy.get_param("~max_speed", 10.0))
        self.kp_gap = float(rospy.get_param("~kp_gap", 0.4))
        self.kd_gap = float(rospy.get_param("~kd_gap", 0.6))
        self.kp_gap_close = float(rospy.get_param("~kp_gap_close", 0.7))
        self.kd_gap_close = float(rospy.get_param("~kd_gap_close", 0.9))
        self.gap_deadband_m = float(rospy.get_param("~gap_deadband_m", 0.5))
        self.rel_speed_deadband = float(rospy.get_param("~rel_speed_deadband", 0.2))
        self.path_switch_max_dist_m = float(rospy.get_param("~path_switch_max_dist_m", 8.0))
        self.accel_limit_mps2 = float(rospy.get_param("~accel_limit_mps2", 1.2))
        self.path_publish_min_dt = float(rospy.get_param("~path_publish_min_dt", 0.6))
        self.path_publish_min_index_advance = int(rospy.get_param("~path_publish_min_index_advance", 6))

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
        rospy.Subscriber(f"/global_path_{self.leader_role}", Path, self._path_cb, queue_size=1)
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
        self._leader_path: Optional[Path] = None
        self._cmd_cache: Dict[str, AckermannDrive] = {}
        self._last_speed_cmd: Dict[str, float] = {}
        self._last_cmd_time: Dict[str, float] = {}
        self._last_path_idx: Dict[str, int] = {}
        self._last_path_pub_time: Dict[str, float] = {}
        self._path_initialized: Dict[str, bool] = {r: False for r in self.follower_roles}

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

    # ----------------- ROS callbacks -----------------
    def _path_cb(self, msg: Path) -> None:
        # Cache leader path; follower paths will be trimmed per follower in _tick
        self._leader_path = msg
        # Initial publish: immediately seed followers with leader path once available
        for role, pub in self.path_pubs.items():
            if role not in self._path_initialized:
                self._path_initialized[role] = False
            odom = self.odom.get(role)
            if odom is None:
                continue
            follower_path = self._trim_path_from_projection(self._leader_path, odom)
            if follower_path is not None and follower_path.poses:
                pub.publish(follower_path)
                idx = self._project_index_on_path(self._leader_path, odom.pose.pose.position.x, odom.pose.pose.position.y)
                now = rospy.Time.now().to_sec()
                self._last_path_idx[role] = idx
                self._last_path_pub_time[role] = now
                self._path_initialized[role] = True

    def _cmd_cb(self, msg: AckermannDrive, role: str) -> None:
        self._cmd_cache[role] = msg
        if role == self.leader_role:
            self.leader_cmd = msg

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        self.odom[role] = msg

    # ----------------- Spacing control -----------------
    def _tick(self, _evt) -> None:
        if self.leader_cmd is None:
            return
        lead_odom = self.odom.get(self.leader_role)
        if lead_odom is None:
            return
        # Publish follower-specific planned paths aligned to each follower's projection on leader path
        if self._leader_path is not None and self._leader_path.poses:
            for role, pub in self.path_pubs.items():
                odom = self.odom.get(role)
                if odom is None:
                    continue
                # If not initialized yet (e.g., odom arrived after path), seed once without distance gating
                if not self._path_initialized.get(role, False):
                    follower_path = self._trim_path_from_projection(self._leader_path, odom)
                    if follower_path is not None and follower_path.poses:
                        pub.publish(follower_path)
                        idx0 = self._project_index_on_path(self._leader_path, odom.pose.pose.position.x, odom.pose.pose.position.y)
                        t0 = rospy.Time.now().to_sec()
                        self._last_path_idx[role] = idx0
                        self._last_path_pub_time[role] = t0
                        self._path_initialized[role] = True
                        continue
                idx, dist = self._project_on_path(self._leader_path, odom.pose.pose.position.x, odom.pose.pose.position.y)
                now = rospy.Time.now().to_sec()
                last_idx = int(self._last_path_idx.get(role, -9999))
                last_t = float(self._last_path_pub_time.get(role, 0.0))
                if (dist <= max(0.0, self.path_switch_max_dist_m)) and ((idx - last_idx) >= max(1, self.path_publish_min_index_advance) or (now - last_t) >= self.path_publish_min_dt):
                    follower_path = self._trim_path_from_projection(self._leader_path, odom)
                    if follower_path is not None:
                        pub.publish(follower_path)
                        self._last_path_idx[role] = idx
                        self._last_path_pub_time[role] = now
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
            err = gap - self.desired_gap_m
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
            # Keep steering aligned with leader command to minimize oscillations; controller will use its own steering
            leader_cmd = self._cmd_cache.get(self.leader_role)
            cmd.steering_angle = float(leader_cmd.steering_angle) if leader_cmd is not None else 0.0
            cmd.speed = float(speed)
            pub = self.override_pubs.get(role)
            if pub is not None:
                pub.publish(cmd)
            self._last_speed_cmd[role] = speed
            self._last_cmd_time[role] = now

    @staticmethod
    def _project_index_on_path(path: Path, px: float, py: float) -> int:
        best_dist_sq = float("inf")
        best_index = 0
        for idx in range(len(path.poses) - 1):
            x1 = path.poses[idx].pose.position.x
            y1 = path.poses[idx].pose.position.y
            x2 = path.poses[idx + 1].pose.position.x
            y2 = path.poses[idx + 1].pose.position.y
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + dx * t
            proj_y = y1 + dy * t
            d2 = (proj_x - px) * (proj_x - px) + (proj_y - py) * (proj_y - py)
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best_index = idx
        return best_index

    @staticmethod
    def _project_on_path(path: Path, px: float, py: float) -> Tuple[int, float]:
        best_dist_sq = float("inf")
        best_index = 0
        for idx in range(len(path.poses) - 1):
            x1 = path.poses[idx].pose.position.x
            y1 = path.poses[idx].pose.position.y
            x2 = path.poses[idx + 1].pose.position.x
            y2 = path.poses[idx + 1].pose.position.y
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + dx * t
            proj_y = y1 + dy * t
            d2 = (proj_x - px) * (proj_x - px) + (proj_y - py) * (proj_y - py)
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best_index = idx
        return best_index, math.sqrt(best_dist_sq)

    def _trim_path_from_projection(self, leader_path: Path, odom: Odometry) -> Optional[Path]:
        if leader_path is None or not leader_path.poses:
            return None
        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        start_idx = self._project_index_on_path(leader_path, px, py)
        trimmed = Path()
        trimmed.header = Header(stamp=rospy.Time.now(), frame_id=leader_path.header.frame_id or "map")
        trimmed.poses = leader_path.poses[start_idx:]
        if not trimmed.poses:
            trimmed.poses = leader_path.poses[-1:]
        return trimmed

    @staticmethod
    def _yaw_from_quat(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

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


