#!/usr/bin/env python3

import math
import sys
from typing import Dict, Tuple, Optional, List

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
        self.desired_gap_m = float(rospy.get_param("~desired_gap_m", 9.0))
        self.teleport_on_start = bool(rospy.get_param("~teleport_on_start", True))
        self.max_speed = float(rospy.get_param("~max_speed", 10.0))
        self.kp_gap = float(rospy.get_param("~kp_gap", 0.4))
        self.kd_gap = float(rospy.get_param("~kd_gap", 0.6))
        self.kp_gap_close = float(rospy.get_param("~kp_gap_close", 0.7))
        self.kd_gap_close = float(rospy.get_param("~kd_gap_close", 0.9))
        self.gap_deadband_m = float(rospy.get_param("~gap_deadband_m", 3.0))
        self.rel_speed_deadband = float(rospy.get_param("~rel_speed_deadband", 0.2))
        self.accel_limit_mps2 = float(rospy.get_param("~accel_limit_mps2", 1.2))
        self.path_connection_max_dist_m = float(rospy.get_param("~path_connection_max_dist_m", 20.0))
        self.path_connection_step_m = float(rospy.get_param("~path_connection_step_m", 2.0))

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # State
        self.actors: Dict[str, carla.Actor] = {}
        self.odom: Dict[str, Odometry] = {}
        self.leader_cmd: Optional[AckermannDrive] = None
        # Use first follower's (ego_vehicle_2) global path as the shared path for the platoon
        # 이 경로를 1,2,3번 차량이 동일하게(투영 위치 기준으로 잘라서) 추종한다.
        self.shared_path_role: Optional[str] = self.follower_roles[0] if self.follower_roles else None
        self.shared_global_path: Optional[Path] = None
        # Arc-length profile for path-based gap calculation
        self._path_arc_length_profile: Optional[List[float]] = None
        self._path_points_cache: Optional[List[Tuple[float, float, float]]] = None  # (x, y, yaw)

        # Publishers
        # 리더 + 모든 팔로워(1,2,3)에 대해 동일한 공유 경로 기반 planned_path_* 를 발행한다.
        all_platoon_roles = [self.leader_role] + self.follower_roles
        self.path_pubs: Dict[str, rospy.Publisher] = {
            r: rospy.Publisher(f"/planned_path_{r}", Path, queue_size=1, latch=True) 
            for r in all_platoon_roles
        }
        self.override_pubs: Dict[str, rospy.Publisher] = {r: rospy.Publisher(f"/carla/{r}/vehicle_control_cmd_override", AckermannDrive, queue_size=1) for r in self.follower_roles}

        # Subscriptions
        # 첫 번째 팔로워(보통 ego_vehicle_2)의 global path 를 받아서 플래툰 공유 경로로 사용
        if self.shared_path_role:
            rospy.Subscriber(f"/global_path_{self.shared_path_role}", Path, self._shared_path_cb, queue_size=1)
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
        rospy.Timer(rospy.Duration(0.5), self._update_all_follower_paths)  # Periodic path update
        self._cmd_cache: Dict[str, AckermannDrive] = {}
        self._last_speed_cmd: Dict[str, float] = {}
        self._last_cmd_time: Dict[str, float] = {}
        
        # Log initialization
        rospy.loginfo(f"platoon_manager: initialized with leader={self.leader_role}, followers={self.follower_roles}, shared_path_role={self.shared_path_role}")
        rospy.loginfo(f"platoon_manager: publishing paths to {list(self.path_pubs.keys())}")
        rospy.loginfo(f"platoon_manager: publishing speed overrides to {list(self.override_pubs.keys())}")

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

    def _shared_path_cb(self, msg: Path) -> None:
        """첫 번째 팔로워(보통 ego_vehicle_2)의 global path 를 공유 경로로 받아서,
        리더/팔로워 모두 같은 전역 경로를 (각자 현재 위치에서 투영한 지점부터) 추종하게 만든다.
        """
        self.shared_global_path = msg
        # Extract path points with yaw
        if not msg.poses:
            rospy.logwarn("platoon_manager: received empty shared path (no poses)")
            return
        
        rospy.loginfo(f"platoon_manager: received shared path from {self.shared_path_role} with {len(msg.poses)} points")
        path_points_xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        
        # Build path points with yaw and compute arc-length profile
        self._path_points_cache = []
        for p in msg.poses:
            x = float(p.pose.position.x)
            y = float(p.pose.position.y)
            yaw = self._yaw_from_quat(p.pose.orientation)
            self._path_points_cache.append((x, y, yaw))
        
        # Compute arc-length profile for accurate gap calculation (공유 경로 기준 arclength)
        self._path_arc_length_profile = self._compute_path_arc_length_profile(self._path_points_cache)
        rospy.loginfo(f"platoon_manager: computed arc-length profile, total path length: {self._path_arc_length_profile[-1]:.2f}m")
        
        # 리더 + 모든 팔로워에 대해, "동일한" 공유 경로 그대로를 planned_path_* 로 발행한다.
        # (각 차량의 위치에 따라 sub-path 를 따로 만들지 않고, 2번 차량의 전역 경로를 1,2,3 모두 공유)
        all_platoon_roles = [self.leader_role] + self.follower_roles
        rospy.loginfo(f"platoon_manager: publishing shared full path for {all_platoon_roles}")
        
        for role in all_platoon_roles:
            pub = self.path_pubs.get(role)
            if pub is None:
                continue
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=msg.header.frame_id)
            path_copy.poses = msg.poses
            pub.publish(path_copy)
            rospy.loginfo(f"platoon_manager: published shared full path ({len(path_copy.poses)} points) to {role}")

    def _build_subpath_from_projection(
        self, path_points: List[Tuple[float, float]], shared_path: Path, odom: Odometry, role: str
    ) -> Optional[Path]:
        """공유 경로 상에 현재 차량 위치를 투영한 지점부터 끝까지의 순수 sub-path 생성.
        - 추가적인 '뒤로 이어지는' 연결선 없이, 기존 전역 경로의 일부만 잘라서 사용한다.
        """
        if not path_points or len(path_points) < 2:
            return None

        # 현재 차량 위치
        px = float(odom.pose.pose.position.x)
        py = float(odom.pose.pose.position.y)

        # 공유 경로에 투영
        proj_result = self._project_to_path(path_points, px, py)
        if proj_result is None:
            # 투영 실패 시, 공유 경로 전체를 그대로 사용
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=shared_path.header.frame_id)
            path_copy.poses = shared_path.poses
            rospy.logwarn_throttle(2.0, f"platoon_manager: {role} projection failed, using full shared path")
            return path_copy

        proj_index, proj_t, proj_dist = proj_result

        # 투영 거리가 너무 멀면(예: 완전히 다른 도로), 그대로 전체 경로 사용
        if proj_dist > self.path_connection_max_dist_m:
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=shared_path.header.frame_id)
            path_copy.poses = shared_path.poses
            rospy.logwarn_throttle(2.0, f"platoon_manager: {role} too far from shared path (dist={proj_dist:.1f}m), using full shared path")
            return path_copy

        # projection segment 기준으로 시작 index 결정
        start_idx = proj_index
        if proj_t > 0.5:
            start_idx = min(proj_index + 1, len(shared_path.poses) - 1)

        # projection 지점부터 끝까지의 순수 sub-path 생성
        new_path = Path()
        new_path.header = Header(stamp=rospy.Time.now(), frame_id=shared_path.header.frame_id)
        for i in range(start_idx, len(shared_path.poses)):
            new_path.poses.append(shared_path.poses[i])

        # 너무 짧으면 전체 경로 사용
        if len(new_path.poses) < 10:
            rospy.logwarn_throttle(2.0, f"platoon_manager: {role} sub-path too short ({len(new_path.poses)} pts), using full shared path")
            full_path = Path()
            full_path.header = Header(stamp=rospy.Time.now(), frame_id=shared_path.header.frame_id)
            full_path.poses = shared_path.poses
            return full_path

        return new_path
    
    def _project_to_path(
        self, path_points: List[Tuple[float, float]], px: float, py: float
    ) -> Optional[Tuple[int, float, float]]:
        """
        Project point onto path and return (segment_index, t_parameter, distance).
        Returns None if projection fails.
        """
        if not path_points or len(path_points) < 2:
            return None
        
        best_dist_sq = float("inf")
        best_index = None
        best_t = 0.0
        
        for idx in range(len(path_points) - 1):
            x1, y1 = path_points[idx]
            x2, y2 = path_points[idx + 1]
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
        
        if best_index is None:
            return None
        
        return best_index, best_t, math.sqrt(best_dist_sq)
    
    def _generate_connection_waypoints(
        self,
        follower_pos: Tuple[float, float],
        path_points: List[Tuple[float, float]],
        proj_index: int,
        proj_t: float,
        proj_dist: float,
    ) -> List[Tuple[float, float, float]]:
        """Generate waypoints from follower position to path projection point."""
        if proj_index >= len(path_points) - 1:
            return []
        
        fx, fy = follower_pos
        x1, y1 = path_points[proj_index]
        x2, y2 = path_points[proj_index + 1]
        
        # Projected point on path segment
        dx = x2 - x1
        dy = y2 - y1
        proj_x = x1 + dx * proj_t
        proj_y = y1 + dy * proj_t
        
        # Calculate direction from follower to projection point
        to_proj_dx = proj_x - fx
        to_proj_dy = proj_y - fy
        to_proj_dist = math.hypot(to_proj_dx, to_proj_dy)
        
        if to_proj_dist < 1e-3:
            return []
        
        # Path direction at projection point
        path_yaw = math.atan2(dy, dx)
        
        waypoints = []
        num_steps = max(2, int(to_proj_dist / self.path_connection_step_m))
        
        for i in range(1, num_steps + 1):
            ratio = float(i) / float(num_steps)
            x = fx + to_proj_dx * ratio
            y = fy + to_proj_dy * ratio
            
            # Interpolate yaw: start with direction to path, end with path direction
            start_yaw = math.atan2(to_proj_dy, to_proj_dx)
            yaw_diff = path_yaw - start_yaw
            # Normalize angle difference
            while yaw_diff > math.pi:
                yaw_diff -= 2.0 * math.pi
            while yaw_diff < -math.pi:
                yaw_diff += 2.0 * math.pi
            yaw = start_yaw + yaw_diff * ratio
            
            waypoints.append((x, y, yaw))
        
        return waypoints

    def _cmd_cb(self, msg: AckermannDrive, role: str) -> None:
        self._cmd_cache[role] = msg
        if role == self.leader_role:
            self.leader_cmd = msg

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        self.odom[role] = msg
    
    def _update_all_follower_paths(self, _evt=None) -> None:
        """주기적으로 리더/팔로워의 planned_path_* 를 갱신.
        현재는 2번 차량의 전역 경로 전체를 그대로 공유하므로, 여기서는
        단순히 공유 경로가 있을 때 최근 버전을 다시 발행해주는 역할만 수행한다.
        """
        if self.shared_global_path is None or not self.shared_global_path.poses:
            return
        
        # Arc-length 캐시가 없다면 생성 (간격 계산용)
        if self._path_points_cache is None or self._path_arc_length_profile is None:
            self._path_points_cache = []
            for p in self.shared_global_path.poses:
                x = float(p.pose.position.x)
                y = float(p.pose.position.y)
                yaw = self._yaw_from_quat(p.pose.orientation)
                self._path_points_cache.append((x, y, yaw))
            self._path_arc_length_profile = self._compute_path_arc_length_profile(self._path_points_cache)
        
        all_platoon_roles = [self.leader_role] + self.follower_roles
        for role in all_platoon_roles:
            pub = self.path_pubs.get(role)
            if pub is None:
                continue
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=self.shared_global_path.header.frame_id)
            path_copy.poses = self.shared_global_path.poses
            pub.publish(path_copy)

    # ----------------- Spacing control -----------------
    def _tick(self, _evt) -> None:
        # Check prerequisites
        lead_odom = self.odom.get(self.leader_role)
        if lead_odom is None:
            rospy.logwarn_throttle(1.0, f"platoon_manager: leader {self.leader_role} odometry not available")
            return
        
        # If leader command is not available, use odometry speed as fallback
        if self.leader_cmd is None:
            # Use leader's current speed from odometry as fallback
            lv = lead_odom.twist.twist.linear
            leader_speed = math.hypot(lv.x, lv.y)
            if leader_speed < 0.1:  # If leader is not moving, don't start followers
                rospy.logwarn_throttle(2.0, f"platoon_manager: leader {self.leader_role} command not available and leader is stationary")
                return
            # Create a dummy command for fallback
            rospy.logdebug_throttle(1.0, f"platoon_manager: using leader odometry speed {leader_speed:.2f} as fallback")
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
            # Use arc-length based gap calculation for accurate curved path following
            gap, rel_speed = self._gap_and_rel_speed_arclength(pred_odom, foll_odom)
            desired_gap = self.desired_gap_m  # Simple fixed gap
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
            elif pred_role == self.leader_role:
                if self.leader_cmd is not None:
                    base = float(self.leader_cmd.speed)
                else:
                    # Fallback to leader odometry speed
                    pv = pred_odom.twist.twist.linear
                    base = math.hypot(pv.x, pv.y)
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
                rospy.logdebug_throttle(1.0, f"platoon_manager: {role} speed={speed:.2f} m/s (gap={gap:.2f}m, err={err:.2f}m, base={base:.2f})")
            else:
                rospy.logwarn_throttle(2.0, f"platoon_manager: no override publisher for {role}")
            self._last_speed_cmd[role] = speed
            self._last_cmd_time[role] = now

    @staticmethod
    def _yaw_from_quat(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _compute_path_arc_length_profile(self, path_points: List[Tuple[float, float, float]]) -> List[float]:
        """Compute cumulative arc-length profile for the path."""
        if not path_points or len(path_points) < 2:
            return []
        
        arc_lengths = [0.0]
        cumulative = 0.0
        
        for i in range(1, len(path_points)):
            x1, y1, _ = path_points[i - 1]
            x2, y2, _ = path_points[i]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            cumulative += seg_len
            arc_lengths.append(cumulative)
        
        return arc_lengths
    
    def _project_to_path_arclength(self, px: float, py: float) -> Optional[Tuple[float, int, float]]:
        """
        Project point onto path and return (arc_length, segment_index, distance_to_path).
        Returns None if projection fails.
        """
        if not self._path_points_cache or not self._path_arc_length_profile:
            return None
        
        if len(self._path_points_cache) < 2:
            return None
        
        best_dist_sq = float("inf")
        best_index = None
        best_t = 0.0
        
        for idx in range(len(self._path_points_cache) - 1):
            x1, y1, _ = self._path_points_cache[idx]
            x2, y2, _ = self._path_points_cache[idx + 1]
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
        
        if best_index is None:
            return None
        
        # Compute arc-length at projection point
        seg_len = math.hypot(
            self._path_points_cache[best_index + 1][0] - self._path_points_cache[best_index][0],
            self._path_points_cache[best_index + 1][1] - self._path_points_cache[best_index][1]
        )
        arclength = self._path_arc_length_profile[best_index] + seg_len * best_t
        
        return arclength, best_index, math.sqrt(best_dist_sq)
    
    def _get_path_direction_at_arclength(self, arclength: float) -> Optional[Tuple[float, float]]:
        """Get path direction vector (unit vector) at given arc-length."""
        if not self._path_points_cache or not self._path_arc_length_profile:
            return None
        
        if len(self._path_points_cache) < 2:
            return None
        
        # Find segment containing this arc-length
        if arclength <= self._path_arc_length_profile[0]:
            x1, y1, yaw1 = self._path_points_cache[0]
            return (math.cos(yaw1), math.sin(yaw1))
        
        if arclength >= self._path_arc_length_profile[-1]:
            x2, y2, yaw2 = self._path_points_cache[-1]
            return (math.cos(yaw2), math.sin(yaw2))
        
        # Binary search for segment
        for idx in range(len(self._path_arc_length_profile) - 1):
            s1 = self._path_arc_length_profile[idx]
            s2 = self._path_arc_length_profile[idx + 1]
            
            if s1 <= arclength <= s2:
                # Interpolate direction
                x1, y1, yaw1 = self._path_points_cache[idx]
                x2, y2, yaw2 = self._path_points_cache[idx + 1]
                
                seg_len = s2 - s1
                if seg_len < 1e-6:
                    return (math.cos(yaw1), math.sin(yaw1))
                
                ratio = (arclength - s1) / seg_len
                
                # Interpolate yaw angle
                yaw_diff = yaw2 - yaw1
                while yaw_diff > math.pi:
                    yaw_diff -= 2.0 * math.pi
                while yaw_diff < -math.pi:
                    yaw_diff += 2.0 * math.pi
                yaw = yaw1 + yaw_diff * ratio
                
                return (math.cos(yaw), math.sin(yaw))
        
        # Fallback
        x1, y1, yaw1 = self._path_points_cache[0]
        return (math.cos(yaw1), math.sin(yaw1))
    
    def _gap_and_rel_speed_arclength(self, lead: Odometry, foll: Odometry) -> Tuple[float, float]:
        """
        Calculate gap and relative speed using arc-length along path.
        This accounts for path curvature.
        """
        if not self._path_points_cache or not self._path_arc_length_profile:
            # Fallback to simple 2D projection
            return self._gap_and_rel_speed_simple(lead, foll)
        
        # Project both vehicles onto path
        lx = float(lead.pose.pose.position.x)
        ly = float(lead.pose.pose.position.y)
        fx = float(foll.pose.pose.position.x)
        fy = float(foll.pose.pose.position.y)
        
        lead_proj = self._project_to_path_arclength(lx, ly)
        foll_proj = self._project_to_path_arclength(fx, fy)
        
        if lead_proj is None or foll_proj is None:
            # Fallback if projection fails
            return self._gap_and_rel_speed_simple(lead, foll)
        
        lead_s, lead_idx, lead_dist = lead_proj
        foll_s, foll_idx, foll_dist = foll_proj
        
        # 기본 arc-length gap/relative speed 계산
        raw_gap = lead_s - foll_s
        
        # Get path direction at follower position for velocity projection
        path_dir = self._get_path_direction_at_arclength(foll_s)
        if path_dir is None:
            # Fallback
            pyaw = self._yaw_from_quat(lead.pose.pose.orientation)
            path_dir = (math.cos(pyaw), math.sin(pyaw))
        
        # Project velocities onto path direction
        lv = lead.twist.twist.linear
        fv = foll.twist.twist.linear
        
        # Transform velocities to path direction
        lead_speed = lv.x * path_dir[0] + lv.y * path_dir[1]
        foll_speed = fv.x * path_dir[0] + fv.y * path_dir[1]
        
        rel_speed = lead_speed - foll_speed
        
        # 단순 거리 기반 gap/rel_speed 계산 (검증 및 fallback 용)
        simple_gap, simple_rel_speed = self._gap_and_rel_speed_simple(lead, foll)
        
        gap = raw_gap
        # 1) arc-length gap 가 심하게 작으면서(simple_gap 보다 훨씬 작으면) 이상치로 간주
        if gap < 1.0 and simple_gap > gap + 2.0:
            rospy.logwarn_throttle(
                2.0,
                "platoon_manager: arc-length gap suspiciously small (%.2f m) vs simple gap (%.2f m), "
                "falling back to simple gap for spacing control",
                gap,
                simple_gap,
            )
            gap = simple_gap
            rel_speed = simple_rel_speed
        # 2) arc-length 상 follower 가 앞에 있는 것으로 나오면(음수 gap), simple gap 사용
        elif gap < 0.0:
            rospy.logwarn_throttle(
                2.0,
                "platoon_manager: negative arc-length gap (%.2f m), falling back to simple gap (%.2f m)",
                gap,
                simple_gap,
            )
            gap = max(0.0, simple_gap)
            rel_speed = simple_rel_speed
        
        # 최소 gap 하한 (너무 작은 값에 대한 보호)
        if gap < 0.5:
            gap = 0.5
        
        return gap, rel_speed

    @staticmethod
    def _gap_and_rel_speed_simple(lead: Odometry, foll: Odometry) -> Tuple[float, float]:
        """Fallback: Simple 2D projection along leader heading (original method)."""
        pyaw = PlatoonManager._yaw_from_quat(lead.pose.pose.orientation)
        fwd = (math.cos(pyaw), math.sin(pyaw))
        dx = lead.pose.pose.position.x - foll.pose.pose.position.x
        dy = lead.pose.pose.position.y - foll.pose.pose.position.y
        gap = max(0.0, dx * fwd[0] + dy * fwd[1])
        lv = lead.twist.twist.linear
        fv = foll.twist.twist.linear
        rel_vx = (lv.x - fv.x)
        rel_vy = (lv.y - fv.y)
        rel_speed = rel_vx * fwd[0] + rel_vy * fwd[1]
        return gap, rel_speed

    @staticmethod
    def _gap_and_rel_speed(lead: Odometry, foll: Odometry, fwd: Tuple[float, float]) -> Tuple[float, float]:
        """Legacy method kept for compatibility - now uses arc-length method."""
        # This method signature is kept for backward compatibility but should use arc-length
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