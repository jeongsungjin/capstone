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

CARLA_EGG = "/home/jamie/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg"
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
        self.accel_limit_mps2 = float(rospy.get_param("~accel_limit_mps2", 1.2))
        self.path_connection_max_dist_m = float(rospy.get_param("~path_connection_max_dist_m", 20.0))
        self.path_connection_step_m = float(rospy.get_param("~path_connection_step_m", 2.0))
        self.path_switch_max_heading_diff_deg = float(rospy.get_param("~path_switch_max_heading_diff_deg", 60.0))
        self.path_switch_max_position_diff_m = float(rospy.get_param("~path_switch_max_position_diff_m", 25.0))
        self.path_continuity_check_points = int(rospy.get_param("~path_continuity_check_points", 10))

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # State
        self.actors: Dict[str, carla.Actor] = {}
        self.odom: Dict[str, Odometry] = {}
        self.leader_cmd: Optional[AckermannDrive] = None
        # Use middle follower's path as the shared path
        # For followers [ego_vehicle_2, ego_vehicle_3], use ego_vehicle_2 (first follower) as shared path
        # This is explicitly set to ego_vehicle_2 for platoon path sharing
        if self.follower_roles:
            # Use first follower (ego_vehicle_2) as shared path role
            # Previously used middle_idx which gave ego_vehicle_3, now explicitly use first follower
            self.shared_path_role = self.follower_roles[0]
            rospy.loginfo(f"platoon_manager: using {self.shared_path_role} as shared path role (first follower from {self.follower_roles})")
        else:
            self.shared_path_role = None
        self.shared_global_path: Optional[Path] = None
        # Arc-length profile for path-based gap calculation
        self._path_arc_length_profile: Optional[List[float]] = None
        self._path_points_cache: Optional[List[Tuple[float, float, float]]] = None  # (x, y, yaw)
        # Store previous paths per vehicle to check continuity
        self._previous_paths: Dict[str, Path] = {}
        self._previous_path_points: Dict[str, List[Tuple[float, float, float]]] = {}
        # Flag: shared loop path initialized from middle follower
        self._shared_path_initialized: bool = False

        # Publishers
        # CRITICAL: All platoon vehicles (1, 2, 3) should follow the same shared loop path
        # Publish to leader and ALL followers (including middle vehicle) - all use the same shared path
        all_platoon_roles = [self.leader_role] + self.follower_roles
        self.path_pubs: Dict[str, rospy.Publisher] = {
            r: rospy.Publisher(f"/planned_path_{r}", Path, queue_size=1, latch=True) 
            for r in all_platoon_roles
        }
        self.override_pubs: Dict[str, rospy.Publisher] = {r: rospy.Publisher(f"/carla/{r}/vehicle_control_cmd_override", AckermannDrive, queue_size=1) for r in self.follower_roles}

        # Subscriptions
        # Subscribe to middle follower's global path (this becomes the shared path for all)
        if self.shared_path_role:
            rospy.Subscriber(f"/global_path_{self.shared_path_role}", Path, self._middle_path_cb, queue_size=1)
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
        rospy.loginfo(f"platoon_manager: initialized with leader={self.leader_role}, followers={self.follower_roles}, shared_path_vehicle={self.shared_path_role}")
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

    def _middle_path_cb(self, msg: Path) -> None:
        """Callback when middle follower's global path is received. This becomes the shared path for leader and other followers."""
        # If shared loop path is already initialized, just update raw message and return.
        # 이 이후에는 follower 경로를 다시 만들지 않고, 초기 정렬 상태를 유지한다.
        if self._shared_path_initialized:
            self.shared_global_path = msg
            return

        self.shared_global_path = msg
        # Extract path points with yaw
        if not msg.poses:
            rospy.logwarn("platoon_manager: received empty path from middle follower")
            return
        
        rospy.loginfo(f"platoon_manager: received path from {self.shared_path_role} with {len(msg.poses)} points")
        path_points_xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        
        # Build path points with yaw and compute arc-length profile
        self._path_points_cache = []
        for p in msg.poses:
            x = float(p.pose.position.x)
            y = float(p.pose.position.y)
            yaw = self._yaw_from_quat(p.pose.orientation)
            self._path_points_cache.append((x, y, yaw))
        
        # Compute arc-length profile for accurate gap calculation
        self._path_arc_length_profile = self._compute_path_arc_length_profile(self._path_points_cache)
        rospy.loginfo(f"platoon_manager: computed arc-length profile, total path length: {self._path_arc_length_profile[-1]:.2f}m")
        
        # CRITICAL: Generate path for ALL platoon vehicles (1, 2, 3) - all use the same shared loop path
        # All vehicles follow the same shared loop path from ego_vehicle_2
        all_platoon_roles = [self.leader_role] + self.follower_roles
        rospy.loginfo(f"platoon_manager: publishing shared loop path for all platoon vehicles: {all_platoon_roles}")
        
        for role in all_platoon_roles:
            vehicle_odom = self.odom.get(role)
            if vehicle_odom is None:
                # If no odometry, publish original path anyway
                rospy.logwarn(f"platoon_manager: {role} odometry not available, publishing original path")
                pub = self.path_pubs.get(role)
                if pub is not None:
                    path_copy = Path()
                    path_copy.header = Header(stamp=rospy.Time.now(), frame_id=msg.header.frame_id)
                    path_copy.poses = msg.poses
                    pub.publish(path_copy)
                    rospy.loginfo(f"platoon_manager: published original path to {role}")
                continue

            # Check path continuity before switching
            should_switch = self._should_switch_path(role, vehicle_odom, self._path_points_cache)
            
            if not should_switch:
                # Keep previous path if switch is not safe
                prev_path = self._previous_paths.get(role)
                if prev_path is not None:
                    rospy.logwarn(f"platoon_manager: {role} path switch rejected due to discontinuity, keeping previous path")
                    pub = self.path_pubs.get(role)
                    if pub is not None:
                        pub.publish(prev_path)
                continue
                # No previous path, use new one anyway
                rospy.logwarn(f"platoon_manager: {role} no previous path available, using new path despite discontinuity")
            
            # CRITICAL: For platooning, all vehicles use the same pure loop path
            # Simply wrap the path starting from the nearest point - NO connection waypoints
            vehicle_path = self._build_pure_loop_path(msg, vehicle_odom, role)
            pub = self.path_pubs.get(role)
            if pub is not None and vehicle_path:
                # Store for next comparison
                self._previous_paths[role] = vehicle_path
                self._previous_path_points[role] = self._path_points_cache.copy()
                pub.publish(vehicle_path)
                rospy.loginfo(f"platoon_manager: published path with {len(vehicle_path.poses)} points to {role}")
            elif vehicle_path is None:
                rospy.logwarn(f"platoon_manager: failed to build path for {role}")

        # Mark shared loop path as initialized so subsequent updates don't rebuild follower paths
        self._shared_path_initialized = True

    def _build_pure_loop_path(self, loop_path: Path, vehicle_odom: Odometry, role: str) -> Optional[Path]:
        """
        Build pure loop path starting from nearest point to vehicle.
        NO connection waypoints - just the original loop path wrapped from nearest point.
        """
        if not loop_path.poses or len(loop_path.poses) < 2:
            return None
        
        # Get vehicle current position
        vx = float(vehicle_odom.pose.pose.position.x)
        vy = float(vehicle_odom.pose.pose.position.y)
        
        # Find nearest point on loop path
        path_points = [(p.pose.position.x, p.pose.position.y) for p in loop_path.poses]
        proj_result = self._project_to_path(path_points, vx, vy)
        
        if proj_result is None:
            # Cannot project, use original loop path as-is
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=loop_path.header.frame_id)
            path_copy.poses = loop_path.poses
            return path_copy
        
        proj_index, proj_t, proj_dist = proj_result
        
        # Determine start index
        start_idx = proj_index
        if proj_t > 0.5:  # If closer to next point, start from next point
            start_idx = min(proj_index + 1, len(loop_path.poses) - 1)
        
        # Build pure loop path - wrap around starting from nearest point
        new_path = Path()
        new_path.header = Header(stamp=rospy.Time.now(), frame_id=loop_path.header.frame_id)
        
        # Add path points from start_idx to end
        for i in range(start_idx, len(loop_path.poses)):
            new_path.poses.append(loop_path.poses[i])
        
        # Wrap around - add points from start to start_idx
        for i in range(0, start_idx):
            new_path.poses.append(loop_path.poses[i])
        
        # Add first point again to complete the loop
        new_path.poses.append(loop_path.poses[0])
        
        rospy.logdebug_throttle(5.0, f"platoon_manager: {role} pure loop path (start_idx={start_idx}, total={len(new_path.poses)} points)")
        
        return new_path if new_path.poses else None

    def _build_follower_path_with_connection(
        self, path_points: List[Tuple[float, float]], leader_path: Path, foll_odom: Odometry, role: str
    ) -> Optional[Path]:
        """Build follower path by projecting follower position onto leader path and adding connection waypoints if needed."""
        if not path_points or len(path_points) < 2:
            return None
        
        # Get follower current position
        fx = float(foll_odom.pose.pose.position.x)
        fy = float(foll_odom.pose.pose.position.y)
        
        # Project follower position onto path
        proj_result = self._project_to_path(path_points, fx, fy)
        if proj_result is None:
            # Cannot project, use original path
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=leader_path.header.frame_id)
            path_copy.poses = leader_path.poses
            return path_copy
        
        proj_index, proj_t, proj_dist = proj_result
        
        # Check if connection waypoints are needed
        if proj_dist > self.path_connection_max_dist_m:
            # Too far from path, don't add connection (will use original path)
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=leader_path.header.frame_id)
            path_copy.poses = leader_path.poses
            return path_copy
        
        # Create new path starting from projection point
        new_path = Path()
        new_path.header = Header(stamp=rospy.Time.now(), frame_id=leader_path.header.frame_id)
        
        # Add connection waypoints if follower is significantly off the path
        if proj_dist > 2.0:  # More than 2m away from path
            connection_waypoints = self._generate_connection_waypoints(
                (fx, fy), path_points, proj_index, proj_t, proj_dist
            )
            for x, y, yaw in connection_waypoints:
                pose = PoseStamped()
                pose.header = new_path.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.orientation.w = math.cos(yaw * 0.5)
                pose.pose.orientation.z = math.sin(yaw * 0.5)
                new_path.poses.append(pose)
        
        # Add path points from projection point onwards
        # CRITICAL: For loop paths (YAML fixed loop path), always wrap around to ensure full path coverage
        start_idx = proj_index
        if proj_t > 0.5:  # If closer to next point, start from next point
            start_idx = min(proj_index + 1, len(leader_path.poses) - 1)
        
        # Check if path is a loop (first and last points are close - within 15m for YAML loop path)
        # This is ALWAYS the case for platooning vehicles using fixed YAML loop path
        first_pose = leader_path.poses[0]
        last_pose = leader_path.poses[-1]
        first_x = first_pose.pose.position.x
        first_y = first_pose.pose.position.y
        last_x = last_pose.pose.position.x
        last_y = last_pose.pose.position.y
        loop_dist = math.hypot(last_x - first_x, last_y - first_y)
        is_loop = loop_dist < 15.0  # Increased threshold for loop detection
        
        # CRITICAL: For platooning, the fixed YAML path is always a loop
        # Always wrap around to ensure followers have complete loop path regardless of position
        if is_loop or len(leader_path.poses) > 100:  # Assume loop if path is long (YAML loop path has 330 points)
            # Add path points from start_idx to end
            for i in range(start_idx, len(leader_path.poses)):
                new_path.poses.append(leader_path.poses[i])
            
            # CRITICAL: Wrap around - add points from start to start_idx
            # This ensures followers always have a complete loop path regardless of their position
            for i in range(0, start_idx):
                new_path.poses.append(leader_path.poses[i])
            
            # Add first point again to complete the loop
            new_path.poses.append(leader_path.poses[0])
            
            rospy.loginfo(f"platoon_manager: {role} wrapped loop path (start_idx={start_idx}, remaining={len(leader_path.poses) - start_idx}, wrapped={start_idx}, total={len(new_path.poses)} points)")
        else:
            # Not a loop, add path points from start_idx to end
            for i in range(start_idx, len(leader_path.poses)):
                new_path.poses.append(leader_path.poses[i])
            
            # If remaining path is too short, use full path
            if len(new_path.poses) < 20:
                rospy.logwarn_throttle(2.0, f"platoon_manager: {role} path too short ({len(new_path.poses)} points) starting from index {start_idx}, using full path")
                full_path = Path()
                full_path.header = Header(stamp=rospy.Time.now(), frame_id=leader_path.header.frame_id)
                full_path.poses = leader_path.poses
                return full_path
        
        # Ensure minimum path length
        if len(new_path.poses) < 20:
            rospy.logwarn_throttle(2.0, f"platoon_manager: {role} generated path too short ({len(new_path.poses)} points), using full path")
            full_path = Path()
            full_path.header = Header(stamp=rospy.Time.now(), frame_id=leader_path.header.frame_id)
            full_path.poses = leader_path.poses
            return full_path
        
        return new_path if new_path.poses else None
    
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
    
    def _should_switch_path(self, role: str, vehicle_odom: Odometry, new_path_points: List[Tuple[float, float, float]]) -> bool:
        """
        Check if it's safe to switch to new path based on continuity.
        Returns True if switch is safe, False if should keep previous path.
        """
        if not self._previous_path_points.get(role):
            # No previous path, always accept first path
            return True
        
        prev_path_points = self._previous_path_points[role]
        if not prev_path_points or len(prev_path_points) < 2:
            return True
        
        # Get vehicle current state
        vx = float(vehicle_odom.pose.pose.position.x)
        vy = float(vehicle_odom.pose.pose.position.y)
        vyaw = self._yaw_from_quat(vehicle_odom.pose.pose.orientation)
        
        # Project vehicle onto previous path to find current progress
        prev_proj = self._project_point_to_path_points(vx, vy, prev_path_points)
        if prev_proj is None:
            return True  # Can't project, accept new path
        
        prev_s, prev_idx, prev_dist = prev_proj
        
        # Project vehicle onto new path
        new_proj = self._project_point_to_path_points(vx, vy, new_path_points)
        if new_proj is None:
            return False  # Can't project to new path, keep old
        
        new_s, new_idx, new_dist = new_proj
        
        # Check 1: Position difference - new path should be reasonably close
        if new_dist > self.path_switch_max_position_diff_m:
            rospy.logwarn(f"platoon_manager: {role} new path too far ({new_dist:.2f}m > {self.path_switch_max_position_diff_m}m)")
            return False
        
        # Check 2: Compare path segments ahead of vehicle position
        # Check next N points in both paths for continuity
        check_count = min(self.path_continuity_check_points, len(prev_path_points) - prev_idx, len(new_path_points) - new_idx)
        
        if check_count < 3:
            # Not enough points to compare, accept if close enough
            return new_dist < self.path_switch_max_position_diff_m * 0.5
        
        # Compare heading directions at similar distances ahead
        max_heading_diff_rad = math.radians(self.path_switch_max_heading_diff_deg)
        similar_count = 0
        
        for i in range(1, min(check_count, 5)):  # Check first 5 points ahead
            if prev_idx + i >= len(prev_path_points) or new_idx + i >= len(new_path_points):
                break
            
            # Get headings at these points
            prev_x1, prev_y1, prev_yaw1 = prev_path_points[prev_idx + i - 1]
            prev_x2, prev_y2, prev_yaw2 = prev_path_points[prev_idx + i]
            prev_seg_yaw = math.atan2(prev_y2 - prev_y1, prev_x2 - prev_x1)
            
            new_x1, new_y1, new_yaw1 = new_path_points[new_idx + i - 1]
            new_x2, new_y2, new_yaw2 = new_path_points[new_idx + i]
            new_seg_yaw = math.atan2(new_y2 - new_y1, new_x2 - new_x1)
            
            # Calculate heading difference
            heading_diff = abs(new_seg_yaw - prev_seg_yaw)
            # Normalize to [-pi, pi]
            while heading_diff > math.pi:
                heading_diff -= 2.0 * math.pi
            heading_diff = abs(heading_diff)
            
            if heading_diff < max_heading_diff_rad:
                similar_count += 1
        
        # If most segments ahead are similar, safe to switch
        similarity_ratio = float(similar_count) / float(min(check_count - 1, 4))
        
        if similarity_ratio < 0.6:  # Less than 60% similarity
            rospy.logwarn(f"platoon_manager: {role} path heading mismatch (similarity={similarity_ratio:.2f}), rejecting switch")
            return False
        
        # Check 3: Compare vehicle heading with new path heading at projection point
        if new_idx < len(new_path_points) - 1:
            new_x1, new_y1, _ = new_path_points[new_idx]
            new_x2, new_y2, _ = new_path_points[new_idx + 1]
            new_path_yaw = math.atan2(new_y2 - new_y1, new_x2 - new_x1)
            
            heading_diff = abs(new_path_yaw - vyaw)
            while heading_diff > math.pi:
                heading_diff -= 2.0 * math.pi
            heading_diff = abs(heading_diff)
            
            if heading_diff > max_heading_diff_rad:
                rospy.logwarn(f"platoon_manager: {role} vehicle heading mismatch with new path ({math.degrees(heading_diff):.1f}deg > {self.path_switch_max_heading_diff_deg}deg)")
                return False
        
        return True
    
    def _project_point_to_path_points(self, px: float, py: float, path_points: List[Tuple[float, float, float]]) -> Optional[Tuple[float, int, float]]:
        """
        Project point onto path points and return (arc_length_estimate, segment_index, distance_to_path).
        Similar to _project_to_path_arclength but works with path_points directly.
        """
        if not path_points or len(path_points) < 2:
            return None
        
        best_dist_sq = float("inf")
        best_index = None
        best_t = 0.0
        
        cumulative_dist = 0.0
        segment_distances = []
        
        for idx in range(len(path_points) - 1):
            x1, y1, _ = path_points[idx]
            x2, y2, _ = path_points[idx + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            
            if seg_len_sq < 1e-6:
                segment_distances.append(0.0)
                continue
            
            seg_len = math.sqrt(seg_len_sq)
            segment_distances.append(seg_len)
            
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
        
        # Estimate arc-length
        arclength = sum(segment_distances[:best_index])
        if best_index < len(segment_distances):
            arclength += segment_distances[best_index] * best_t
        
        return arclength, best_index, math.sqrt(best_dist_sq)

    def _cmd_cb(self, msg: AckermannDrive, role: str) -> None:
        self._cmd_cache[role] = msg
        if role == self.leader_role:
            self.leader_cmd = msg


    def _odom_cb(self, msg: Odometry, role: str) -> None:
        self.odom[role] = msg
    
    def _update_all_follower_paths(self, _evt=None) -> None:
        """Periodically update all paths (leader and followers except middle) based on current odometry."""
        if self.shared_global_path is None or not self.shared_global_path.poses:
            return
        
        # Rebuild path cache and arc-length profile if needed
        if self._path_points_cache is None or self._path_arc_length_profile is None:
            self._path_points_cache = []
            for p in self.shared_global_path.poses:
                x = float(p.pose.position.x)
                y = float(p.pose.position.y)
                yaw = self._yaw_from_quat(p.pose.orientation)
                self._path_points_cache.append((x, y, yaw))
            self._path_arc_length_profile = self._compute_path_arc_length_profile(self._path_points_cache)
        
        path_points = [(p.pose.position.x, p.pose.position.y) for p in self.shared_global_path.poses]
        # CRITICAL: Update paths for ALL platoon vehicles (1, 2, 3) - all use the same shared loop path
        all_platoon_roles = [self.leader_role] + self.follower_roles
        
        for role in all_platoon_roles:
            vehicle_odom = self.odom.get(role)
            if vehicle_odom is None:
                continue

            # 이미 이 차량에 대해 경로가 한 번 설정되었다면, 재생성하지 않는다.
            # 초기 정렬 이후에는 동일한 폐경로를 유지하면서 필요 시 재발행만 한다.
            prev_path = self._previous_paths.get(role)
            if prev_path is not None:
                pub = self.path_pubs.get(role)
                if pub is not None:
                    pub.publish(prev_path)
                continue
            
            # Check path continuity before switching (same as in _middle_path_cb)
            should_switch = self._should_switch_path(role, vehicle_odom, self._path_points_cache)
            
            if not should_switch:
                # Keep previous path if switch is not safe
                prev_path = self._previous_paths.get(role)
                if prev_path is not None:
                    # Check if previous path is still valid (not reached end)
                    # If close to end, accept new path anyway
                    continue
                # No previous path, use new one anyway
            
            # CRITICAL: For platooning, all vehicles use the same pure loop path
            # NO connection waypoints - just pure loop path wrapped from nearest point
            vehicle_path = self._build_pure_loop_path(self.shared_global_path, vehicle_odom, role)
            pub = self.path_pubs.get(role)
            if pub is not None and vehicle_path:
                # Store for next comparison
                self._previous_paths[role] = vehicle_path
                self._previous_path_points[role] = self._path_points_cache.copy()
                pub.publish(vehicle_path)

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
                rospy.logwarn_throttle(2.0, f"platoon_manager: {role} odometry not available, skipping speed command")
                continue
            pred_role = chain[idx]  # leader for first follower, previous follower otherwise
            pred_odom = self.odom.get(pred_role)
            if pred_odom is None:
                rospy.logwarn_throttle(2.0, f"platoon_manager: {role} predecessor {pred_role} odometry not available, skipping speed command")
                continue
            # Use arc-length based gap calculation for accurate curved path following
            gap, rel_speed = self._gap_and_rel_speed_arclength(pred_odom, foll_odom)
            
            # Check if gap calculation failed (None or invalid values)
            if gap is None or rel_speed is None:
                rospy.logwarn_throttle(1.0, f"platoon_manager: {role} gap calculation failed (gap={gap}, rel_speed={rel_speed}), using fallback")
                # Fallback: use simple gap calculation
                gap, rel_speed = self._gap_and_rel_speed_simple(pred_odom, foll_odom)
            
            # CRITICAL: Validate gap - if gap is too small (< 1.0m) or negative, use simple distance
            # Arc-length gap can be wrong if projection fails or vehicles are on different path segments
            simple_gap, simple_rel_speed = self._gap_and_rel_speed_simple(pred_odom, foll_odom)
            
            # If arc-length gap is suspiciously small (< 1.0m) but simple distance is larger, use simple distance
            if gap < 1.0 and simple_gap > gap + 2.0:
                rospy.logwarn_throttle(2.0, f"platoon_manager: {role} arc-length gap too small ({gap:.2f}m) but simple gap larger ({simple_gap:.2f}m), using simple gap")
                gap = simple_gap
                rel_speed = simple_rel_speed
            elif gap < 0.0:
                # If gap is negative, use simple distance
                gap = simple_gap
                rel_speed = simple_rel_speed
                rospy.logwarn_throttle(2.0, f"platoon_manager: {role} negative gap, using simple gap ({gap:.2f}m)")
            
            # Ensure gap is reasonable (at least 0.5m to avoid division issues)
            if gap < 0.5:
                gap = 0.5
            
            desired_gap = self.desired_gap_m  # Simple fixed gap
            err = gap - desired_gap
            # Deadband to reduce chattering
            if abs(err) < max(0.0, self.gap_deadband_m):
                err = 0.0
            if abs(rel_speed) < max(0.0, self.rel_speed_deadband):
                rel_speed = 0.0
            # Feed-forward base speed from predecessor command if available; fallback to predecessor odometry
            # CRITICAL: Always ensure base speed reflects predecessor's actual movement
            pred_cmd = self._cmd_cache.get(pred_role)
            pv = pred_odom.twist.twist.linear
            pred_actual_speed = math.hypot(pv.x, pv.y)  # Actual speed from odometry
            
            if pred_cmd is not None:
                base = float(pred_cmd.speed)
                # CRITICAL: If command speed is 0 but predecessor is actually moving, use odometry speed
                if base < 0.1 and pred_actual_speed > 0.1:
                    base = pred_actual_speed
            elif pred_role == self.leader_role:
                if self.leader_cmd is not None:
                    base = float(self.leader_cmd.speed)
                    # CRITICAL: If command speed is 0 but leader is actually moving, use odometry speed
                    if base < 0.1 and pred_actual_speed > 0.1:
                        base = pred_actual_speed
                else:
                    # Fallback to leader odometry speed
                    base = pred_actual_speed
            else:
                base = pred_actual_speed
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
            
            # CRITICAL: Prevent speed from dropping to 0 if predecessor is moving
            # If predecessor is moving but desired speed is 0, use minimum speed
            pred_speed = math.hypot(pred_odom.twist.twist.linear.x, pred_odom.twist.twist.linear.y)
            
            # CRITICAL: If correction is too negative (follower too close), limit correction
            # to prevent speed from dropping below reasonable minimum
            if pred_speed > 0.1 and desired < 0.1:
                rospy.logwarn_throttle(2.0, f"platoon_manager: {role} desired speed too low ({desired:.2f}) but predecessor {pred_role} moving ({pred_speed:.2f}). base={base:.2f}, corr={corr:.2f}, gap={gap:.2f}, err={err:.2f}")
                
                # If gap calculation seems wrong (gap=0 or very small), limit correction
                if gap < 1.0:
                    # Use predecessor speed as minimum, limit negative correction
                    desired = max(pred_speed * 0.5, base * 0.5, 0.3)
                    rospy.logwarn_throttle(2.0, f"platoon_manager: {role} gap too small ({gap:.2f}m), limiting correction, desired={desired:.2f}")
                else:
                    # Normal case: ensure minimum speed
                    if base < 0.1 and pred_speed > 0.1:
                        base = pred_speed
                        desired = base + corr
                        desired = max(0.0, min(self.max_speed, desired))
                    # Ensure minimum speed
                    if desired < 0.3:
                        desired = 0.3
            
            # Rate limiting (accel/decel)
            now = rospy.Time.now().to_sec()
            last_v = float(self._last_speed_cmd.get(role, 0.0))
            last_t = float(self._last_cmd_time.get(role, now))
            dt = max(1e-3, now - last_t)
            dv_max = self.accel_limit_mps2 * dt
            speed = max(last_v - dv_max, min(last_v + dv_max, desired))
            
            # CRITICAL: Prevent speed from dropping to 0 if predecessor is moving
            if pred_speed > 0.1 and speed < 0.1:
                speed = max(0.3, speed)  # Use at least 0.3 m/s if predecessor is moving
            
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
                rospy.logdebug_throttle(1.0, f"platoon_manager: {role} speed={speed:.2f} m/s (gap={gap:.2f}m, err={err:.2f}m, base={base:.2f}, pred_speed={pred_speed:.2f})")
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
        
        # CRITICAL: Gap is arc-length difference along path
        # If follower is ahead of leader on path (foll_s > lead_s), this means they're
        # on a loop path and follower is in the next loop cycle - adjust gap accordingly
        gap_arclength = lead_s - foll_s
        
        # If gap is negative (follower ahead), check if it's a loop path
        # For loop paths, add path length to get correct gap
        if gap_arclength < 0.0 and self._path_arc_length_profile:
            total_path_length = self._path_arc_length_profile[-1]
            # If follower is more than half path length ahead, they're in next cycle
            if abs(gap_arclength) > total_path_length * 0.5:
                gap_arclength = total_path_length + gap_arclength
        
        gap = max(0.0, gap_arclength)
        
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


