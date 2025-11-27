#!/usr/bin/env python3

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import rospy
import yaml
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool

# Prefer centralized CARLA path setup if available
try:
    from setup_carla_path import CARLA_BUILD_PATH  # noqa: F401
except Exception:
    _env = os.environ.get("CARLA_PYTHON_PATH")
    _default = os.path.expanduser("~/carla/PythonAPI/carla/build/lib.linux-x86_64-cpython-38")
    CARLA_BUILD_PATH = _env if _env else _default
if CARLA_BUILD_PATH and CARLA_BUILD_PATH not in sys.path:
    sys.path.insert(0, CARLA_BUILD_PATH)

try:
    import carla
except ImportError as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


def quaternion_to_yaw(orientation):
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MultiVehicleController:
    def __init__(self):
        rospy.init_node("multi_vehicle_controller", anonymous=True)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.num_vehicles = rospy.get_param("~num_vehicles", 3)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 5.0)
        self.wheelbase = rospy.get_param("~wheelbase", 2.7)
        self.max_steer = rospy.get_param("~max_steer", 1.0)
        self.target_speed = rospy.get_param("~target_speed", 2.0)
        self.control_frequency = rospy.get_param("~control_frequency", 30.0)
        # Vehicle dimensions for path-based collision detection
        self.vehicle_length = float(rospy.get_param("~vehicle_length", 6.0))
        self.vehicle_width = float(rospy.get_param("~vehicle_width", 2.5))
        self.vehicle_half_length = self.vehicle_length / 2.0
        self.vehicle_half_width = self.vehicle_width / 2.0
        # Path-based collision detection parameters
        self.enable_path_collision_check = bool(rospy.get_param("~enable_path_collision_check", True))
        self.path_collision_lookahead_time = float(rospy.get_param("~path_collision_lookahead_time", 5.0))  # seconds
        self.path_collision_check_interval = float(rospy.get_param("~path_collision_check_interval", 0.5))  # seconds
        self.path_collision_threshold = float(rospy.get_param("~path_collision_threshold", 4.0))  # meters (considering vehicle size)
        # Target selection policy: use arc-length along path (preserves waypoint order)
        self.target_select_by_arclength = bool(rospy.get_param("~target_select_by_arclength", True))
        # Region-based control gains
        self.regions_config_file = rospy.get_param("~regions_config_file", "/home/jamie/capstone/src/carla_multi_vehicle_planner/scripts/config/regions.yaml")
        self.default_speed_gain = float(rospy.get_param("~default_speed_gain", 1.0))
        self.default_steering_gain = float(rospy.get_param("~default_steering_gain", 0.5))
        self.regions: List[Dict] = []
        if self.regions_config_file:
            self._load_regions_config()
        # (removed) projection stabilization params

        # Safety stop parameters (area-limited)
        self.enable_safety_stop = bool(rospy.get_param("~enable_safety_stop", True))
        self.safety_x_min = float(rospy.get_param("~safety_x_min", -13.0))
        self.safety_x_max = float(rospy.get_param("~safety_x_max", 13.0))
        self.safety_y_min = float(rospy.get_param("~safety_y_min", -40.0))
        self.safety_y_max = float(rospy.get_param("~safety_y_max", 5.0))
        self.safety_distance = float(rospy.get_param("~safety_distance", 18.0))
        # Consider only vehicles within my front cone and approaching
        self.safety_front_cone_deg = float(rospy.get_param("~safety_front_cone_deg", 100.0))
        self.safety_require_closing = bool(rospy.get_param("~safety_require_closing", True))
        # Deadlock escape tuning
        self.safety_deadlock_timeout_sec = float(rospy.get_param("~safety_deadlock_timeout_sec", 2.0))
        self.safety_deadlock_escape_speed = float(rospy.get_param("~safety_deadlock_escape_speed", 2.0))
        self.safety_deadlock_escape_duration_sec = float(rospy.get_param("~safety_deadlock_escape_duration_sec", 1.0))
        # Opposite-lane ignore within intersection area
        self.enable_opposite_lane_ignore = bool(rospy.get_param("~enable_opposite_lane_ignore", True))
        self.opposite_lane_heading_thresh_deg = float(rospy.get_param("~opposite_lane_heading_thresh_deg", 100.0))

        # Intersection dynamic priority (entry-order based)
        self.intersection_dynamic_priority = bool(rospy.get_param("~intersection_dynamic_priority", True))
        self.intersection_x_min = float(rospy.get_param("~intersection_x_min", self.safety_x_min))
        self.intersection_x_max = float(rospy.get_param("~intersection_x_max", self.safety_x_max))
        self.intersection_y_min = float(rospy.get_param("~intersection_y_min", self.safety_y_min))
        self.intersection_y_max = float(rospy.get_param("~intersection_y_max", self.safety_y_max))
        # Optional second intersection area
        self.intersection2_enabled = bool(rospy.get_param("~intersection2_enabled", False))
        self.intersection2_x_min = float(rospy.get_param("~intersection2_x_min", self.intersection_x_min))
        self.intersection2_x_max = float(rospy.get_param("~intersection2_x_max", self.intersection_x_max))
        self.intersection2_y_min = float(rospy.get_param("~intersection2_y_min", self.intersection_y_min))
        self.intersection2_y_max = float(rospy.get_param("~intersection2_y_max", self.intersection_y_max))

        # Intersection stability (hysteresis + grace-time) to avoid flickering entry-order
        self.intersection_hysteresis_margin = float(rospy.get_param("~intersection_hysteresis_margin", 5.0))
        self.intersection_exit_grace_sec = float(rospy.get_param("~intersection_exit_grace_sec", 1.5))

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        self.vehicles = {}
        self.states = {}
        self.control_publishers = {}
        self.pose_publishers = {}
        # Optional external override for AckermannDrive (e.g., platoon manager)
        self.enable_external_control_override = bool(rospy.get_param("~enable_external_control_override", False))
        self._override_cmds = {}
        self._override_stamp = {}
        self.override_speed_only = bool(rospy.get_param("~override_speed_only", True))
        # Emergency stop (global)
        self.emergency_hold_sec = float(rospy.get_param("~emergency_hold_sec", 1.5))
        self._emergency_until = 0.0
        # Person stop (spawned via rviz_person_spawner)
        self.enable_person_stop = bool(rospy.get_param("~enable_person_stop", True))
        self.person_stop_distance_m = float(rospy.get_param("~person_stop_distance_m", 8.0))
        self.person_front_cone_deg = float(rospy.get_param("~person_front_cone_deg", 100.0))
        self.person_topic = str(rospy.get_param("~person_topic", "/spawned_person"))
        self._person_xy = None  # type: ignore
        self._person_stamp = None  # type: ignore
        # Global emergency stop (applies to all vehicles regardless of area)
        self.enable_global_emergency_stop = bool(rospy.get_param("~enable_global_emergency_stop", True))
        self.global_emergency_distance_m = float(rospy.get_param("~global_emergency_distance_m", 10.0))
        self.global_emergency_cone_deg = float(rospy.get_param("~global_emergency_cone_deg", 30.0))  # ±60 degrees
        # Tracking hold (pause control when BEV tracking unavailable)
        self.enable_tracking_hold = bool(rospy.get_param("~enable_tracking_hold", True))
        self.tracking_hold_timeout_sec = float(rospy.get_param("~tracking_hold_timeout_sec", 0.5))
        self._tracking_ok = {}
        self._tracking_stamp = {}
        # Track per-role intersection entry order
        self.intersection_orders = {0: {}, 1: {}}
        self._intersection_counters = {0: 0, 1: 0}
        # Track last time a role was observed inside each area (with hysteresis margin)
        self._intersection_last_inside = {0: {}, 1: {}}

        # Platoon priority inheritance (optional)
        self.platoon_enable = bool(rospy.get_param("~platoon_enable", False))
        self.platoon_leader = str(rospy.get_param("~platoon_leader", "ego_vehicle_1"))
        followers_str = str(rospy.get_param("~platoon_followers", "")).strip()
        self.platoon_followers = [s.strip() for s in followers_str.split(",") if s.strip()] if followers_str else []
        if self.platoon_enable:
            self._platoon_roles = {self.platoon_leader, *self.platoon_followers}
        else:
            self._platoon_roles = set()

        # Dynamic lookahead distance based on path curvature (optional)
        self.dynamic_lookahead_enable = bool(rospy.get_param("~dynamic_lookahead_enable", True))
        default_ld_min = max(0.5, self.lookahead_distance * 0.5)
        self.lookahead_min = float(rospy.get_param("~lookahead_min", default_ld_min))
        self.lookahead_max = float(rospy.get_param("~lookahead_max", max(self.lookahead_distance, self.lookahead_min)))
        if self.lookahead_min < 0.05:
            rospy.logwarn("multi_vehicle_controller: lookahead_min too small (%.3f); clamping to 0.05", self.lookahead_min)
            self.lookahead_min = 0.05
        if self.lookahead_max < self.lookahead_min:
            rospy.logwarn(
                "multi_vehicle_controller: lookahead_max (%.3f) < lookahead_min (%.3f); clamping max to min",
                self.lookahead_max,
                self.lookahead_min,
            )
            self.lookahead_max = self.lookahead_min
        self.lookahead_kappa_gain = float(rospy.get_param("~lookahead_kappa_gain",0.0))

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            self.states[role] = {
                "path": [],  # List of (x, y) tuples for control
                "path_msg": None,  # Original Path message for collision detection
                "current_index": 0,
                "position": None,
                "orientation": None,
                "s_profile": [],
                "path_length": 0.0,
                "progress_s": 0.0,
                "remaining_distance": None,
                # safety/deadlock fields
                "safety_stop_since": None,
                "deadlock_escape_until": None,
                "tracking_hold": False,
            }
            # Consume controller-compatible planned paths (global planner output relayed or platoon-trimmed)
            path_topic = f"/planned_path_{role}"
            odom_topic = f"/carla/{role}/odometry"
            control_topic = f"/carla/{role}/vehicle_control_cmd"
            pose_topic = f"/{role}/pose"
            rospy.Subscriber(path_topic, Path, self._path_cb, callback_args=role)
            rospy.Subscriber(odom_topic, Odometry, self._odom_cb, callback_args=role)
            self.control_publishers[role] = rospy.Publisher(control_topic, AckermannDrive, queue_size=1)
            self.pose_publishers[role] = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
            if self.enable_external_control_override:
                override_topic = f"/carla/{role}/vehicle_control_cmd_override"
                rospy.Subscriber(override_topic, AckermannDrive, self._override_cb, callback_args=role, queue_size=1)
            if self.enable_tracking_hold:
                tracking_topic = f"/carla/{role}/bev_tracking_ok"
                rospy.Subscriber(tracking_topic, Bool, self._tracking_cb, callback_args=role, queue_size=1)
                self._tracking_ok[role] = False
                self._tracking_stamp[role] = 0.0

        # Global emergency stop topic
        rospy.Subscriber("/emergency_stop", Bool, self._emergency_cb, queue_size=1)
        # Person position topic
        if self.enable_person_stop:
            rospy.Subscriber(self.person_topic, PoseStamped, self._person_cb, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / 50.0), self._refresh_vehicles)
        rospy.Timer(rospy.Duration(1.0 / 50.0), self._control_loop)
        rospy.on_shutdown(self._shutdown)

    def _role_name(self, index):
        return f"ego_vehicle_{index + 1}"

    def _refresh_vehicles(self, _event):
        actors = self.world.get_actors().filter("vehicle.*")
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role in self.states:
                self.vehicles[role] = actor

    def _load_regions_config(self):
        """Load region-based control gains from YAML file."""
        try:
            config_path = self.regions_config_file
            if not config_path:
                rospy.logwarn("Region config file path not specified")
                return
            
            # Support both absolute and relative paths
            if not os.path.isabs(config_path):
                # Try to find relative to package
                try:
                    import rospkg
                    rospack = rospkg.RosPack()
                    pkg_path = rospack.get_path("carla_multi_vehicle_planner")
                    config_path = os.path.join(pkg_path, config_path)
                except Exception:
                    # Fallback to relative to current working directory
                    pass
            
            if not os.path.exists(config_path):
                rospy.logwarn(f"Region config file not found: {config_path}")
                return
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.regions = config.get("regions", [])
            rospy.loginfo(f"Loaded {len(self.regions)} regions from {config_path}")
            
            # Validate and set default gains if not specified
            for region in self.regions:
                if "speed_gain" not in region:
                    region["speed_gain"] = self.default_speed_gain
                if "steering_gain" not in region:
                    region["steering_gain"] = self.default_steering_gain
                rospy.loginfo(f"  {region.get('id', 'unknown')}: speed_gain={region['speed_gain']:.2f}, steering_gain={region['steering_gain']:.2f}")
        
        except Exception as e:
            rospy.logerr(f"Failed to load region config: {e}")
            self.regions = []
    
    def _point_in_rectangle(self, px: float, py: float, corners: Dict) -> bool:
        """Check if a point is inside a rectangle defined by 4 corners."""
        try:
            tl = corners.get("top_left", [0, 0])
            tr = corners.get("top_right", [0, 0])
            br = corners.get("bottom_right", [0, 0])
            bl = corners.get("bottom_left", [0, 0])
            
            # Get rectangle bounds
            x_min = min(tl[0], tr[0], br[0], bl[0])
            x_max = max(tl[0], tr[0], br[0], bl[0])
            y_min = min(tl[1], tr[1], br[1], bl[1])
            y_max = max(tl[1], tr[1], br[1], bl[1])
            
            # Check if point is inside rectangle
            return x_min <= px <= x_max and y_min <= py <= y_max
        
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error checking point in rectangle: {e}")
            return False
    
    def _get_region_gains(self, x: float, y: float, role: Optional[str] = None) -> Tuple[float, float]:
        """Get speed and steering gains for current position.

        Also emits a clear debug log indicating which region (if any) the vehicle is in.
        """
        # Check each region (first match wins)
        for region in self.regions:
            corners = region.get("corners", {})
            if self._point_in_rectangle(x, y, corners):
                speed_gain = float(region.get("speed_gain", self.default_speed_gain))
                steering_gain = float(region.get("steering_gain", self.default_steering_gain))
                region_id = region.get("id", "unknown")
                # 눈에 잘 보이는 디버깅 로그 (1초에 한 번씩)
                rospy.loginfo_throttle(
                    1.0,
                    "[REGION] %s in %s: speed_gain=%.2f, steering_gain=%.2f at (x=%.2f, y=%.2f)",
                    role or "unknown",
                    str(region_id),
                    speed_gain,
                    steering_gain,
                    x,
                    y,
                )
                return speed_gain, steering_gain

        # Not in any explicit region
        rospy.logdebug_throttle(
            2.0,
            "[REGION] %s in NO_REGION: using default gains speed=%.2f, steer=%.2f at (x=%.2f, y=%.2f)",
            role or "unknown",
            self.default_speed_gain,
            self.default_steering_gain,
            x,
            y,
        )
        return self.default_speed_gain, self.default_steering_gain
    
    def _role_index_from_name(self, role):
        try:
            return max(1, int(role.split("_")[-1]))
        except Exception:
            return 9999

    def _in_safety_area(self, position):
        if position is None:
            return False
        x = position.x
        y = position.y
        return (self.safety_x_min <= x <= self.safety_x_max) and (self.safety_y_min <= y <= self.safety_y_max)
    
    def _in_safety_area_with_margin(self, position, margin):
        """Check if position is within safety area with margin (for hysteresis)."""
        if position is None:
            return False
        x = position.x
        y = position.y
        return (
            (self.safety_x_min - margin) <= x <= (self.safety_x_max + margin)
            and (self.safety_y_min - margin) <= y <= (self.safety_y_max + margin)
        )

    def _which_intersection_area(self, position):
        if position is None:
            return None
        x = position.x
        y = position.y
        # Area 0
        if (self.intersection_x_min <= x <= self.intersection_x_max) and (self.intersection_y_min <= y <= self.intersection_y_max):
            return 0
        # Area 1 (optional)
        if self.intersection2_enabled:
            if (self.intersection2_x_min <= x <= self.intersection2_x_max) and (self.intersection2_y_min <= y <= self.intersection2_y_max):
                return 1
        return None

    def _in_area_with_margin(self, position, area_idx, margin):
        if position is None:
            return False
        x = position.x
        y = position.y
        if area_idx == 0:
            return (
                (self.intersection_x_min - margin) <= x <= (self.intersection_x_max + margin)
                and (self.intersection_y_min - margin) <= y <= (self.intersection_y_max + margin)
            )
        if area_idx == 1 and self.intersection2_enabled:
            return (
                (self.intersection2_x_min - margin) <= x <= (self.intersection2_x_max + margin)
                and (self.intersection2_y_min - margin) <= y <= (self.intersection2_y_max + margin)
            )
        return False

    def _in_intersection_area(self, position):
        return self._which_intersection_area(position) is not None

    def _normalize_angle(self, ang):
        while ang > math.pi:
            ang -= 2.0 * math.pi
        while ang < -math.pi:
            ang += 2.0 * math.pi
        return ang

    def _is_opposite_lane(self, my_position, other_position):
        try:
            if self.carla_map is None or my_position is None or other_position is None:
                return False
            my_loc = carla.Location(x=my_position.x, y=my_position.y, z=0.0)
            ot_loc = carla.Location(x=other_position.x, y=other_position.y, z=0.0)
            my_wp = self.carla_map.get_waypoint(my_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            ot_wp = self.carla_map.get_waypoint(ot_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if my_wp is None or ot_wp is None:
                return False
            # Primary: same road and lane_id sign differs (typical opposite direction)
            try:
                if my_wp.road_id == ot_wp.road_id and (my_wp.lane_id * ot_wp.lane_id) < 0:
                    return True
            except Exception:
                pass
            # Fallback: opposite heading
            my_yaw = math.radians(my_wp.transform.rotation.yaw)
            ot_yaw = math.radians(ot_wp.transform.rotation.yaw)
            diff = abs(self._normalize_angle(my_yaw - ot_yaw))
            if diff >= math.radians(max(0.0, min(175.0, self.opposite_lane_heading_thresh_deg))):
                return True
        except Exception:
            return False
        return False

    def _check_path_overlap(self, my_role, other_role, my_path_points, other_path_points, lookahead_dist_m: float = 50.0) -> bool:
        """Check if paths overlap considering vehicle dimensions.
        
        IMPORTANT: Only checks near-future overlap. Paths must be close enough in space
        and time for collision to be relevant. Far future overlaps are ignored.
        
        Args:
            my_role: My vehicle role
            other_role: Other vehicle role
            my_path_points: List of (x, y, yaw) tuples for my path
            other_path_points: List of (x, y, yaw) tuples for other path
            lookahead_dist_m: Maximum distance to check ahead (meters) - should be limited (e.g., 10-15m)
            
        Returns:
            True if paths overlap within threshold distance considering vehicle size
        """
        if not my_path_points or not other_path_points:
            return False
        
        # Limit lookahead to reasonable distance (don't check too far ahead)
        lookahead_dist_m = min(lookahead_dist_m, 15.0)  # Maximum 15m lookahead
        
        # Get current positions and speeds
        my_state = self.states.get(my_role, {})
        other_state = self.states.get(other_role, {})
        my_pos = my_state.get("position")
        other_pos = other_state.get("position")
        
        if my_pos is None or other_pos is None:
            return False
        
        my_actor = self.vehicles.get(my_role)
        other_actor = self.vehicles.get(other_role)
        my_speed = 0.0
        other_speed = 0.0
        if my_actor is not None:
            v = my_actor.get_velocity()
            my_speed = math.hypot(v.x, v.y)
        if other_actor is not None:
            v = other_actor.get_velocity()
            other_speed = math.hypot(v.x, v.y)
        
        # Find my current position on my path
        my_current_dist = float('inf')
        my_start_idx = 0
        for i, (x, y, _) in enumerate(my_path_points):
            dist = math.hypot(x - my_pos.x, y - my_pos.y)
            if dist < my_current_dist:
                my_current_dist = dist
                my_start_idx = i
        
        # Find other's current position on their path
        other_current_dist = float('inf')
        other_start_idx = 0
        for i, (x, y, _) in enumerate(other_path_points):
            dist = math.hypot(x - other_pos.x, y - other_pos.y)
            if dist < other_current_dist:
                other_current_dist = dist
                other_start_idx = i
        
        # Check path segments ahead
        threshold_dist = self.path_collision_threshold
        check_step = 0.5  # Check every 0.5m along path
        max_iterations = int(lookahead_dist_m / check_step)
        
        for step in range(0, max_iterations):
            check_dist = step * check_step
            
            # Estimate my future position on path
            my_future_idx = my_start_idx
            accumulated_dist = 0.0
            for i in range(my_start_idx, len(my_path_points) - 1):
                x1, y1, _ = my_path_points[i]
                x2, y2, _ = my_path_points[i + 1]
                seg_dist = math.hypot(x2 - x1, y2 - y1)
                if accumulated_dist + seg_dist >= check_dist:
                    # Interpolate
                    t = (check_dist - accumulated_dist) / max(seg_dist, 1e-6)
                    my_future_x = x1 + t * (x2 - x1)
                    my_future_y = y1 + t * (y2 - y1)
                    my_future_idx = i
                    break
                accumulated_dist += seg_dist
                my_future_idx = i + 1
            else:
                # Reached end of path
                if my_future_idx >= len(my_path_points):
                    break
                my_future_x, my_future_y, _ = my_path_points[-1]
            
            # Estimate other's future position on path
            other_future_idx = other_start_idx
            accumulated_dist = 0.0
            for i in range(other_start_idx, len(other_path_points) - 1):
                x1, y1, _ = other_path_points[i]
                x2, y2, _ = other_path_points[i + 1]
                seg_dist = math.hypot(x2 - x1, y2 - y1)
                if accumulated_dist + seg_dist >= check_dist:
                    # Interpolate
                    t = (check_dist - accumulated_dist) / max(seg_dist, 1e-6)
                    other_future_x = x1 + t * (x2 - x1)
                    other_future_y = y1 + t * (y2 - y1)
                    other_future_idx = i
                    break
                accumulated_dist += seg_dist
                other_future_idx = i + 1
            else:
                # Reached end of path
                if other_future_idx >= len(other_path_points):
                    break
                other_future_x, other_future_y, _ = other_path_points[-1]
            
            # Check distance considering vehicle dimensions
            dist_between = math.hypot(my_future_x - other_future_x, my_future_y - other_future_y)
            # Subtract vehicle half-lengths (front and back)
            effective_dist = dist_between - self.vehicle_half_length - self.vehicle_half_length
            
            if effective_dist < threshold_dist:
                # Paths overlap within threshold
                rospy.logdebug_throttle(1.0, "[PATH_COLLISION] %s and %s paths overlap: dist=%.2fm (effective=%.2fm) at %.1fm ahead", 
                                       my_role, other_role, dist_between, effective_dist, check_dist)
                return True
            
            # Early exit if paths are diverging
            if check_dist > 0 and dist_between > lookahead_dist_m:
                break
        
        return False
    
    def _has_path_collision_with_higher_priority(self, my_role, my_position) -> bool:
        """[DISABLED] Path-based collision prediction is disabled to reduce deadlocks.

        Kept for backward compatibility, but always returns False.
        """
        return False

    def _has_imminent_collision(self, my_role, my_position, my_yaw):
        """Check if there's a vehicle in very close range (emergency stop).
        
        This applies globally to all vehicles regardless of intersection area.
        Checks for vehicles within a narrow front cone at very close distance.
        Only lower priority vehicle stops to avoid deadlock, except in extreme close range (< 5m).
        """
        if my_position is None or my_yaw is None:
            rospy.logdebug_throttle(2.0, "[EMERGENCY] %s: missing position or yaw", my_role)
            return False
        
        forward_x = math.cos(my_yaw)
        forward_y = math.sin(my_yaw)
        # Narrow cone: ±30 degrees (total 60 degrees by default)
        half_cone_rad = math.radians(self.global_emergency_cone_deg / 2.0)
        cos_limit = math.cos(half_cone_rad)
        
        my_index = self._role_index_from_name(my_role)
        my_area = self._which_intersection_area(my_position)
        in_intersection = my_area is not None
        my_order = None
        if in_intersection:
            my_order = self.intersection_orders.get(my_area, {}).get(my_role)
        
        checked_count = 0
        for other_role, other_state in self.states.items():
            if other_role == my_role:
                continue
            
            checked_count += 1
            
            # Skip platoon members if platooning is enabled
            if (
                self.platoon_enable
                and self._platoon_roles
                and (my_role in self._platoon_roles)
                and (other_role in self._platoon_roles)
            ):
                rospy.logdebug_throttle(2.0, "[EMERGENCY] %s: skipping platoon member %s", my_role, other_role)
                continue
            
            other_pos = other_state.get("position")
            if other_pos is None:
                rospy.logdebug_throttle(2.0, "[EMERGENCY] %s: %s has no position", my_role, other_role)
                continue
            
            # Check if opposite lane (ignore if opposite direction)
            if self.enable_opposite_lane_ignore:
                try:
                    if self._is_opposite_lane(my_position, other_pos):
                        rospy.logdebug_throttle(2.0, "[EMERGENCY] %s: skipping opposite lane %s", my_role, other_role)
                        continue
                except Exception as e:
                    rospy.logdebug_throttle(2.0, "[EMERGENCY] %s: opposite lane check failed: %s", my_role, str(e))
                    pass
            
            dx = other_pos.x - my_position.x
            dy = other_pos.y - my_position.y
            dist = math.hypot(dx, dy)
            
            # Check distance threshold first
            if dist > self.global_emergency_distance_m:
                continue
            
            # Check if in front cone
            if dist > 1e-3:
                dir_dot = (dx * forward_x + dy * forward_y) / dist
                if dir_dot < cos_limit:
                    continue
            else:
                # Very close, assume in front cone
                dir_dot = 1.0
            
            # CRITICAL: If other vehicle is behind me (opposite direction), ignore it regardless of priority
            # No matter how high priority (platoon, entry order), if it's behind, don't yield
            if dir_dot < -0.1:  # Other vehicle is clearly behind me
                rospy.logdebug_throttle(2.0, "[EMERGENCY] %s: ignoring %s (behind me, dir_dot=%.2f) - continuing", 
                                       my_role, other_role, dir_dot)
                continue
            
            # For very close distances (< 5m), ignore priority to avoid deadlock
            # Both vehicles should stop if extremely close
            emergency_threshold = 5.0  # meters
            use_priority_check = dist >= emergency_threshold
            
            if use_priority_check:
                # Check priority - only lower priority vehicle stops
                # Inside intersection: use entry order
                # Outside intersection: use position-based priority (front vehicle has priority)
                other_area = self._which_intersection_area(other_pos)
                higher = False
                
                if self.intersection_dynamic_priority and in_intersection and (other_area == my_area):
                    # Inside intersection: use entry order priority
                    other_order = self.intersection_orders.get(my_area, {}).get(other_role)
                    if other_order is not None and my_order is not None:
                        higher = other_order < my_order
                        rospy.logdebug_throttle(1.0, "[EMERGENCY] %s vs %s: orders %d vs %d, higher=%s", 
                                               my_role, other_role, my_order, other_order, higher)
                    elif other_order is not None and my_order is None:
                        higher = True
                        rospy.logdebug_throttle(1.0, "[EMERGENCY] %s (no order) vs %s (order %d): other has priority", 
                                               my_role, other_role, other_order)
                    elif other_order is None and my_order is not None:
                        higher = False
                        rospy.logdebug_throttle(1.0, "[EMERGENCY] %s (order %d) vs %s (no order): I have priority", 
                                               my_role, my_order, other_role)
                    else:
                        # Both have no order yet: check if platoon vs non-platoon
                        # If one is platoon and other is not, non-platoon has lower priority (should stop)
                        my_is_platoon = self.platoon_enable and my_role in self._platoon_roles
                        other_is_platoon = self.platoon_enable and other_role in self._platoon_roles
                        
                        if my_is_platoon and not other_is_platoon:
                            # I'm in platoon, other is not - other should stop (I have priority)
                            higher = False
                            rospy.logdebug_throttle(1.0, "[EMERGENCY] %s (platoon, no order) vs %s (non-platoon, no order): platoon has priority", 
                                                   my_role, other_role)
                        elif not my_is_platoon and other_is_platoon:
                            # I'm not in platoon, other is - I should stop
                            higher = True
                            rospy.logdebug_throttle(1.0, "[EMERGENCY] %s (non-platoon, no order) vs %s (platoon, no order): platoon has priority", 
                                                   my_role, other_role)
                        else:
                            # Both same type (both platoon or both non-platoon), use role index as fallback
                            other_index = self._role_index_from_name(other_role)
                            higher = other_index < my_index
                            rospy.logdebug_throttle(1.0, "[EMERGENCY] %s vs %s: both no order, using role index: %d < %d = %s", 
                                                   my_role, other_role, other_index, my_index, higher)
                else:
                    # Outside intersection or different area: use position-based priority
                    # Front vehicle (closer to my direction) has higher priority
                    if dir_dot > 0.1:  # Other is clearly ahead
                        higher = True
                        rospy.logdebug_throttle(1.0, "[EMERGENCY] %s vs %s: other is ahead (dir_dot=%.2f), other has priority", 
                                               my_role, other_role, dir_dot)
                    elif dir_dot < -0.1:  # Other is clearly behind
                        higher = False
                        rospy.logdebug_throttle(1.0, "[EMERGENCY] %s vs %s: other is behind (dir_dot=%.2f), I have priority", 
                                               my_role, other_role, dir_dot)
                    else:
                        # Similar direction: check if platoon vs non-platoon
                        my_is_platoon = self.platoon_enable and my_role in self._platoon_roles
                        other_is_platoon = self.platoon_enable and other_role in self._platoon_roles
                        
                        if my_is_platoon and not other_is_platoon:
                            # I'm in platoon, other is not - other should stop (I have priority)
                            higher = False
                            rospy.logdebug_throttle(1.0, "[EMERGENCY] %s (platoon) vs %s (non-platoon): platoon has priority", 
                                                   my_role, other_role)
                        elif not my_is_platoon and other_is_platoon:
                            # I'm not in platoon, other is - I should stop
                            higher = True
                            rospy.logdebug_throttle(1.0, "[EMERGENCY] %s (non-platoon) vs %s (platoon): platoon has priority", 
                                                   my_role, other_role)
                        else:
                            # Both same type, use role index
                            other_index = self._role_index_from_name(other_role)
                            higher = other_index < my_index
                            rospy.logdebug_throttle(1.0, "[EMERGENCY] %s vs %s: similar direction, using role index: %d < %d = %s", 
                                                   my_role, other_role, other_index, my_index, higher)
                
                # Only stop if other vehicle has higher priority
                if not higher:
                    rospy.logdebug_throttle(1.0, "[EMERGENCY] %s: skipping %s (I have higher priority) at %.2fm, dir_dot=%.2f", 
                                           my_role, other_role, dist, dir_dot if dist > 1e-3 else 1.0)
                    continue
            
            # Vehicle found in narrow front cone at very close range
            priority_msg = f"(higher priority)" if use_priority_check else "(emergency: both should stop)"
            rospy.logwarn_throttle(0.5, "[EMERGENCY] %s: vehicle %s %s detected at %.2fm in front cone (%.1f deg)", 
                                   my_role, other_role, priority_msg, dist, self.global_emergency_cone_deg)
            return True
        
        if checked_count == 0:
            rospy.logdebug_throttle(2.0, "[EMERGENCY] %s: no other vehicles to check", my_role)
        return False

    def _has_nearby_higher_priority(self, my_role, my_position, my_yaw):
        my_index = self._role_index_from_name(my_role)
        if my_position is None or my_index is None or my_yaw is None:
            return False
        forward_x = math.cos(my_yaw)
        forward_y = math.sin(my_yaw)
        cos_limit = math.cos(math.radians(max(0.0, min(175.0, self.safety_front_cone_deg)) / 2.0))
        my_actor = self.vehicles.get(my_role)
        my_vx = 0.0
        my_vy = 0.0
        if my_actor is not None:
            v = my_actor.get_velocity()
            my_vx, my_vy = v.x, v.y
        my_area = self._which_intersection_area(my_position)
        in_intersection = my_area is not None
        my_order = None
        if in_intersection:
            my_order = self.intersection_orders.get(my_area, {}).get(my_role)
        for other_role, other_state in self.states.items():
            if other_role == my_role:
                continue
            if (
                self.platoon_enable
                and self._platoon_roles
                and (my_role in self._platoon_roles)
                and (other_role in self._platoon_roles)
            ):
                continue
            other_pos = other_state.get("position")
            if other_pos is None:
                continue
            # Decide priority basis
            higher = False
            other_area = self._which_intersection_area(other_pos)
            if self.intersection_dynamic_priority and in_intersection and (other_area == my_area):
                other_order = self.intersection_orders.get(my_area, {}).get(other_role)
                # If both have orders, compare them
                if other_order is not None and my_order is not None:
                    higher = other_order < my_order
                # If other has order but I don't, other has higher priority (entered first)
                elif other_order is not None and my_order is None:
                    higher = True
                    rospy.logdebug_throttle(1.0, "[PRIORITY] %s (no order) yields to %s (order %d)", 
                                          my_role, other_role, other_order)
                # If I have order but other doesn't, I have higher priority (entered first)
                elif other_order is None and my_order is not None:
                    higher = False
                    rospy.logdebug_throttle(1.0, "[PRIORITY] %s (order %d) has priority over %s (no order)", 
                                          my_role, my_order, other_role)
                # If neither has order yet, use role index as fallback (to avoid deadlock)
                else:
                    other_index = self._role_index_from_name(other_role)
                    higher = other_index < my_index
                    rospy.logdebug_throttle(1.0, "[PRIORITY] %s and %s both have no order, using role index: %d < %d", 
                                          my_role, other_role, other_index, my_index)
            else:
                # Outside intersection or dynamic priority disabled, use role index
                other_index = self._role_index_from_name(other_role)
                higher = other_index < my_index

            if not higher:
                continue
            
            # CRITICAL: Check relative direction - if other vehicle is behind me, ignore it
            # No matter how high priority (platoon, entry order), if it's behind, don't yield
            dx = (other_pos.x - my_position.x)
            dy = (other_pos.y - my_position.y)
            dist = math.hypot(dx, dy)
            
            if dist > 1e-3:
                dir_dot = (dx * forward_x + dy * forward_y) / dist
                if dir_dot < -0.1:  # Other vehicle is clearly behind me
                    rospy.logdebug_throttle(2.0, "[PRIORITY] %s: ignoring %s with higher priority (behind me, dir_dot=%.2f) - continuing", 
                                           my_role, other_role, dir_dot)
                    continue
            
            # 단순 거리/각도 기반 안전 정지: 경로 기반 충돌 예측은 사용하지 않는다.
            # Consider vehicle dimensions in distance calculation
            vehicle_half_length_sum = self.vehicle_half_length + self.vehicle_half_length
            effective_dist = dist - vehicle_half_length_sum
            if effective_dist > self.safety_distance:
                continue
            if self.enable_opposite_lane_ignore and self._in_safety_area(my_position):
                try:
                    if self._is_opposite_lane(my_position, other_pos):
                        continue
                except Exception:
                    pass
            if dist > 1e-3:
                dir_dot = (dx * forward_x + dy * forward_y) / dist
                if dir_dot < cos_limit:
                    continue
            if self.safety_require_closing:
                other_actor = self.vehicles.get(other_role)
                ovx = ovy = 0.0
                if other_actor is not None:
                    ov = other_actor.get_velocity()
                    ovx, ovy = ov.x, ov.y
                closing_rate = (dx * (ovx - my_vx) + dy * (ovy - my_vy)) / max(dist, 1e-3)
                if closing_rate >= 0.0:
                    continue
            return True
        return False

    def _estimate_path_curvature(self, points, center_index):
        # Estimate curvature (1/R) using three path points around current index.
        try:
            if not points or len(points) < 3:
                return 0.0
            i = max(1, min(center_index, len(points) - 2))
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            x3, y3 = points[i + 1]
            a = math.hypot(x2 - x1, y2 - y1)
            b = math.hypot(x3 - x2, y3 - y2)
            c = math.hypot(x3 - x1, y3 - y1)
            if a < 1e-6 or b < 1e-6 or c < 1e-6:
                return 0.0
            twice_area = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            curvature = (2.0 * twice_area) / max(1e-6, a * b * c)
            return max(0.0, float(curvature))
        except Exception:
            return 0.0

    def _compute_effective_lookahead(self, state):
        base_ld = float(self.lookahead_distance)
        if not self.dynamic_lookahead_enable:
            return base_ld
        path_points = state.get("path") or []
        current_index = int(state.get("current_index", 0))
        kappa = self._estimate_path_curvature(path_points, current_index)
        # Shrink lookahead with curvature: ld_eff = base / (1 + gain * kappa), clamped
        denom = 1.0 + max(0.0, self.lookahead_kappa_gain) * max(0.0, kappa)
        ld_eff = base_ld / max(1e-6, denom)
        ld_eff = max(self.lookahead_min, min(self.lookahead_max, ld_eff))
        return ld_eff

    def _path_cb(self, msg, role):
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        s_profile, total_len = self._compute_path_profile(points)
        # Assign in a tight block to reduce race with control loop
        st = self.states[role]
        st["path"] = points
        st["path_msg"] = msg  # original for yaw/collision checks
        st["current_index"] = 0
        st["s_profile"] = s_profile
        st["path_length"] = total_len
        st["progress_s"] = 0.0
        st["remaining_distance"] = total_len if total_len > 0.0 else None
        # rospy.loginfo(f"{role}: received path with {len(points)} points")

    def _odom_cb(self, msg, role):
        self.states[role]["position"] = msg.pose.pose.position
        self.states[role]["orientation"] = msg.pose.pose.orientation
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pose_publishers[role].publish(pose_msg)

    def _compute_path_profile(self, points):
        if len(points) < 2:
            return [], 0.0
        cumulative = [0.0]
        total = 0.0
        for idx in range(1, len(points)):
            step = math.hypot(
                points[idx][0] - points[idx - 1][0],
                points[idx][1] - points[idx - 1][1],
            )
            total += step
            cumulative.append(total)
        return cumulative, total

    def _front_point(self, role, state, vehicle):
        position = state.get("position")
        orientation = state.get("orientation")
        if position is None or orientation is None:
            return None
        yaw = quaternion_to_yaw(orientation)
        offset = 2.0
        if vehicle is not None:
            bb = getattr(vehicle, "bounding_box", None)
            if bb is not None and getattr(bb, "extent", None) is not None:
                offset = bb.extent.x + 0.3
        fx = position.x + math.cos(yaw) * offset
        fy = position.y + math.sin(yaw) * offset
        return fx, fy

    def _project_progress(self, path, s_profile, px, py):
        if len(path) < 2 or not s_profile:
            return None
        best_dist_sq = float("inf")
        best_index = None
        best_t = 0.0
        for idx in range(len(path) - 1):
            x1, y1 = path[idx]
            x2, y2 = path[idx + 1]
            seg_dx = x2 - x1
            seg_dy = y2 - y1
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len_sq < 1e-6:
                continue
            t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + seg_dx * t
            proj_y = y1 + seg_dy * t
            dist_sq = (proj_x - px) ** 2 + (proj_y - py) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_index = idx
                best_t = t
        if best_index is None:
            return None
        seg_length = math.hypot(path[best_index + 1][0] - path[best_index][0], path[best_index + 1][1] - path[best_index][1])
        if seg_length < 1e-6:
            s_now = s_profile[best_index]
        else:
            s_now = s_profile[best_index] + best_t * seg_length
        return s_now, best_index

    def _update_progress(self, role, state, vehicle):
        path = state.get("path")
        s_profile = state.get("s_profile")
        if not path or not s_profile:
            return
        front = self._front_point(role, state, vehicle)
        if front is None:
            return
        projection = self._project_progress(path, s_profile, front[0], front[1])
        if projection is None:
            return
        s_now, index = projection
        state["progress_s"] = s_now
        total_len = state.get("path_length", 0.0)
        if total_len > 0.0:
            state["remaining_distance"] = max(0.0, total_len - s_now)
        else:
            state["remaining_distance"] = None
        state["current_index"] = min(index, len(path) - 1)

    def _control_loop(self, _event):
        for role, state in self.states.items():
            vehicle = self.vehicles.get(role)
            if vehicle is None:
                continue
            now_sec = float(rospy.Time.now().to_sec())
            tracking_hold = False
            if self.enable_tracking_hold:
                ok_flag = self._tracking_ok.get(role, False)
                last_stamp = self._tracking_stamp.get(role, 0.0)
                if (not ok_flag) or ((now_sec - last_stamp) > max(0.0, self.tracking_hold_timeout_sec)):
                    tracking_hold = True
                state["tracking_hold"] = tracking_hold
            else:
                state["tracking_hold"] = False
            self._update_progress(role, state, vehicle)
            steer, speed = self._compute_control(role, state)
            if steer is None:
                continue
            if tracking_hold:
                speed = 0.0
                rospy.loginfo_throttle(1.0, "%s: waiting for reliable BEV tracking", role)
            # Apply external override if enabled and recent
            if self.enable_external_control_override:
                cmd = self._override_cmds.get(role)
                stamp = self._override_stamp.get(role)
                if cmd is not None and stamp is not None:
                    if (rospy.Time.now() - stamp).to_sec() < 0.5:
                        if self.override_speed_only:
                            speed = cmd.speed
                        else:
                            steer = cmd.steering_angle
                            speed = cmd.speed
            # Global emergency stop: check for imminent collision regardless of area
            position = state.get("position")
            orientation = state.get("orientation")
            my_yaw = quaternion_to_yaw(orientation) if orientation is not None else None
            
            # Global emergency stop: check for imminent collision regardless of area
            # CRITICAL: Platoon followers should NOT stop due to emergency stop logic
            # They are controlled by platoon_manager for speed, and should only stop if leader stops
            is_platoon_follower = (self.platoon_enable and 
                                  self._platoon_roles and 
                                  role in self._platoon_roles and 
                                  role != self.platoon_leader)
            
            if not is_platoon_follower:
                # This should work even if path-based check fails
                if self.enable_global_emergency_stop and position is not None and my_yaw is not None:
                    has_imminent = self._has_imminent_collision(role, position, my_yaw)
                    if has_imminent:
                        speed = 0.0
                        rospy.logwarn_throttle(0.5, "[EMERGENCY] %s: EMERGENCY STOP due to imminent collision at (%.2f, %.2f)", 
                                              role, position.x, position.y)
            else:
                # Platoon follower: emergency stop logic is handled by platoon_manager
                # Only allow emergency stop for truly critical situations (very close, < 3m)
                if self.enable_global_emergency_stop and position is not None and my_yaw is not None:
                    has_imminent = self._has_imminent_collision(role, position, my_yaw)
                    if has_imminent:
                        # Check distance - only stop if extremely close (< 3m)
                        # This prevents platoon followers from stopping unnecessarily
                        rospy.logdebug_throttle(2.0, "[EMERGENCY] %s (platoon follower): emergency check triggered, but speed controlled by platoon_manager", role)
            
            # Area-limited safety stop for lower-priority vehicles near other vehicles
            # CRITICAL: Platoon followers should NOT stop due to safety stop logic
            # They are controlled by platoon_manager for speed, and should only stop if leader stops
            in_area = self._in_safety_area(position)
            # Area-limited safety stop for lower-priority vehicles near other vehicles
            # Skip safety stop for platoon followers (speed controlled by platoon_manager)
            if self.enable_safety_stop and in_area and not is_platoon_follower:
                now = rospy.Time.now().to_sec()
                base_area = None
                if self.intersection_dynamic_priority:
                    base_area = self._which_intersection_area(position)
                # Assign order on first entry to base (tight) area only
                if self.intersection_dynamic_priority and base_area is not None:
                    # Initialize dictionary if needed
                    if base_area not in self.intersection_orders:
                        self.intersection_orders[base_area] = {}
                    if base_area not in self._intersection_counters:
                        self._intersection_counters[base_area] = 0
                    
                    # Assign priority only if not already assigned
                    if role not in self.intersection_orders[base_area]:
                        self._intersection_counters[base_area] += 1
                        assigned_order = self._intersection_counters[base_area]
                        self.intersection_orders[base_area][role] = assigned_order
                        rospy.loginfo("[INTERSECTION] %s assigned order %d in area %d", role, assigned_order, base_area)
                    # Platoon: inherit leader's order for followers in same area
                    if self.platoon_enable and role in self.platoon_followers:
                        lead_order = self.intersection_orders.get(base_area, {}).get(self.platoon_leader)
                        if lead_order is not None:
                            current = self.intersection_orders.get(base_area, {}).get(role)
                            if current is None or lead_order < current:
                                self.intersection_orders[base_area][role] = lead_order
                                rospy.loginfo("[INTERSECTION] %s inherited order %d from leader %s in area %d", 
                                             role, lead_order, self.platoon_leader, base_area)
                # Update last-inside timestamps using hysteresis margin for both areas
                for a_idx in (0, 1):
                    if a_idx == 1 and not self.intersection2_enabled:
                        continue
                    if self._in_area_with_margin(position, a_idx, self.intersection_hysteresis_margin):
                        self._intersection_last_inside[a_idx][role] = now
                # Graceful clearing when outside expanded box for longer than grace time
                for a_idx in (0, 1):
                    if a_idx == 1 and not self.intersection2_enabled:
                        continue
                    if role in self.intersection_orders.get(a_idx, {}):
                        inside = self._in_area_with_margin(position, a_idx, self.intersection_hysteresis_margin)
                        if not inside:
                            last_ts = self._intersection_last_inside[a_idx].get(role)
                            if last_ts is None or (now - last_ts) >= self.intersection_exit_grace_sec:
                                self.intersection_orders[a_idx].pop(role, None)
                                self._intersection_last_inside[a_idx].pop(role, None)
                has_higher_front = self._has_nearby_higher_priority(role, position, my_yaw) if in_area else False
                escape_until = state.get("deadlock_escape_until")
                if escape_until is not None and now_sec < escape_until:
                    # In escape window: allow creeping
                    speed = min(speed, self.safety_deadlock_escape_speed)
                elif has_higher_front:
                    # Apply safety stop with deadlock timer
                    since = state.get("safety_stop_since")
                    if since is None:
                        state["safety_stop_since"] = now_sec
                        speed = 0.0
                    else:
                        if (now_sec - since) >= self.safety_deadlock_timeout_sec:
                            # Start escape window and creep
                            state["deadlock_escape_until"] = now_sec + self.safety_deadlock_escape_duration_sec
                            speed = min(speed, self.safety_deadlock_escape_speed)
                        else:
                            speed = 0.0
                else:
                    # Reset timers if not actively safety stopping
                    state["safety_stop_since"] = None
                    # Do not reset escape_until so window can complete
            # Person stop: if approaching spawned person and within distance, force stop
            if self.enable_person_stop and state.get("position") is not None and state.get("orientation") is not None and self._person_xy is not None:
                px = float(state["position"].x)
                py = float(state["position"].y)
                tx, ty = self._person_xy
                dx = tx - px
                dy = ty - py
                dist = math.hypot(dx, dy)
                if dist <= max(0.0, self.person_stop_distance_m):
                    yaw = quaternion_to_yaw(state["orientation"]) if state.get("orientation") is not None else None
                    if yaw is not None and dist > 1e-3:
                        fx = math.cos(yaw)
                        fy = math.sin(yaw)
                        cos_half = math.cos(math.radians(max(0.0, min(175.0, self.person_front_cone_deg)) / 2.0))
                        dir_dot = (dx * fx + dy * fy) / dist
                        if dir_dot >= cos_half:
                            speed = 0.0
            # Global emergency stop takes precedence
            now_f = rospy.Time.now().to_sec()
            if now_f < float(self._emergency_until):
                speed = 0.0
            self._apply_carla_control(role, vehicle, steer, speed)
            self._publish_ackermann(role, steer, speed)

    def _override_cb(self, msg, role):
        self._override_cmds[role] = msg
        self._override_stamp[role] = rospy.Time.now()

    def _emergency_cb(self, msg):
        active = bool(getattr(msg, "data", False))
        if active:
            self._emergency_until = rospy.Time.now().to_sec() + max(0.1, self.emergency_hold_sec)
        else:
            self._emergency_until = 0.0

    def _person_cb(self, msg: PoseStamped):
        try:
            self._person_xy = (float(msg.pose.position.x), float(msg.pose.position.y))
            self._person_stamp = rospy.Time.now()
        except Exception:
            self._person_xy = None
            self._person_stamp = None

    def _tracking_cb(self, msg: Bool, role: str) -> None:
        self._tracking_ok[role] = bool(getattr(msg, "data", False))
        self._tracking_stamp[role] = rospy.Time.now().to_sec()

    def _compute_control(self, role, state):
        path = state["path"]
        position = state["position"]
        orientation = state["orientation"]
        # Strong guard against empty/too-short path or missing pose
        if not path or len(path) < 2 or position is None or orientation is None:
            return None, None

        x = position.x
        y = position.y
        yaw = quaternion_to_yaw(orientation)

        # Determine effective lookahead distance considering path curvature
        effective_lookahead = self._compute_effective_lookahead(state)

        target = None
        if self.target_select_by_arclength and state.get("s_profile"):
            # Choose the point whose arc-length >= s_now + lookahead_distance
            s_profile = state["s_profile"]
            # Guard against race: if s_profile length mismatches path, recompute quickly
            if len(s_profile) != len(path):
                s_profile, _ = self._compute_path_profile(path)
                state["s_profile"] = s_profile
            s_now = float(state.get("progress_s", 0.0))
            s_target = s_now + float(effective_lookahead)
            # Binary search could be used; linear scan is fine for typical sizes
            target_idx = None
            for i, s in enumerate(s_profile):
                if s >= s_target:
                    target_idx = i
                    break
            if target_idx is None:
                target_idx = len(path) - 1
            # Final clamp to avoid IndexError on races
            if target_idx < 0 or target_idx >= len(path):
                if not path:
                    return None, None
                target_idx = len(path) - 1
            tx, ty = path[target_idx]
            target = (tx, ty)
            state["current_index"] = target_idx
        else:
            # Fallback: distance-based selection forward from current_index
            index = state["current_index"]
            while index < len(path):
                px, py = path[index]
                if math.hypot(px - x, py - y) > effective_lookahead * 0.5:
                    break
                index += 1
            state["current_index"] = min(index, len(path) - 1)
            for offset in range(len(path)):
                candidate_index = (state["current_index"] + offset) % len(path)
                px, py = path[candidate_index]
                dist = math.hypot(px - x, py - y)
                if dist >= effective_lookahead:
                    target = (px, py)
                    break
            if target is None:
                target = path[-1]

        tx, ty = target
        dx = tx - x
        dy = ty - y
        alpha = math.atan2(dy, dx) - yaw
        while alpha > math.pi:
            alpha -= 2.0 * math.pi
        while alpha < -math.pi:
            alpha += 2.0 * math.pi

        Ld = math.hypot(dx, dy)
        if Ld < 1e-3:
            return 0.0, 0.0

        steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)
        steer = max(-self.max_steer, min(self.max_steer, steer))

        speed = self.target_speed
        # if abs(steer) > 0.2:
        #     speed *= 0.7
        # if Ld < effective_lookahead:
        #     speed *= max(0.0, Ld / max(1e-6, effective_lookahead))
        
        # Apply region-based gains
        if self.regions:
            speed_gain, steering_gain = self._get_region_gains(x, y, role)
            speed *= speed_gain
            steer *= steering_gain
            # Re-clamp steering after gain application
            steer = max(-self.max_steer, min(self.max_steer, steer))

        return steer, speed

    def _apply_carla_control(self, role, vehicle, steer, speed):
        control = carla.VehicleControl()
        control.steer = max(-1.0, min(1.0, steer / max(1e-3, self.max_steer)))
        current_velocity = vehicle.get_velocity()
        current_speed = math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2 + current_velocity.z ** 2)
        speed_error = speed - current_speed
        if speed_error > 0:
            control.throttle = max(0.2, min(1.0, speed_error / max(1.0, speed)))
            control.brake = 0.0 
        else:
            control.throttle = 0.0
            control.brake = min(1.0, -speed_error / max(1.0, self.target_speed))
        vehicle.apply_control(control)


    def _publish_ackermann(self, role, steer, speed):
        msg = AckermannDrive()
        msg.steering_angle = steer
        msg.speed = speed
        self.control_publishers[role].publish(msg)

    def _stop_flag_cb(self, msg: Bool, role: str):
        self.stop_flags[role] = bool(msg.data)

    def _shutdown(self):
        pass


if __name__ == "__main__":
    try:
        MultiVehicleController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
