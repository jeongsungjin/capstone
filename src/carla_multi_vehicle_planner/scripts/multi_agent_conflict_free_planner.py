#!/usr/bin/env python3

import heapq
import math
import os
import random
import threading
from typing import Dict, List, Optional, Tuple

import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from std_msgs.msg import String
from typing import Any

# Ensure CARLA Python API is available on sys.path before import
try:
    from setup_carla_path import CARLA_EGG, AGENTS_ROOT  # type: ignore
except Exception:
    CARLA_EGG = None
    AGENTS_ROOT = None

try:
    import carla  # type: ignore
except ImportError as exc:
    rospy.logfatal(f"Failed to import CARLA package: {exc}")
    carla = None
    GlobalRoutePlanner = None
    GlobalRoutePlannerDAO = None
else:
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
    except Exception as exc:
        rospy.logfatal(f"Failed to import CARLA GlobalRoutePlanner: {exc}")
        GlobalRoutePlanner = None
    try:
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO  # type: ignore
    except Exception:
        GlobalRoutePlannerDAO = None


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MultiAgentConflictFreePlanner:
    """
    새 다중 차량 글로벌 플래너(시간 스케줄링 없이) – 경로 충돌을 간단한 공간/방향 제약으로 회피.

    - 각 차량의 현재 전방 위치를 기준으로 시작 웨이포인트를 정렬 선택
    - 목적지 후보는 스폰포인트 기반, 경로 길이 범위 내에서 랜덤 샘플
    - 이미 선택된 다른 차량 경로와 초기 구간(지평선)에서 근접/역방향 충돌을 피하도록 후보를 필터
    - 경로는 `/global_path_{role}`로 게시(기존 컨트롤러/리레이 호환)
    """

    def __init__(self) -> None:
        rospy.init_node("multi_agent_conflict_free_planner", anonymous=True)
        if carla is None or GlobalRoutePlanner is None:
            raise RuntimeError("CARLA navigation modules unavailable")

        # Parameters
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.global_route_resolution = float(rospy.get_param("~global_route_resolution", 1.0))
        self.path_sampling = float(rospy.get_param("~path_sampling", 0.3))
        self.min_destination_distance = float(rospy.get_param("~min_destination_distance", 30.0))
        self.max_destination_distance = float(rospy.get_param("~max_destination_distance", 140.0))
        self.destination_retry_limit = int(rospy.get_param("~destination_retry_limit", 80))
        self.replan_interval = float(rospy.get_param("~replan_interval", 1.0))
        self.replan_remaining_m = float(rospy.get_param("~replan_remaining_m", 50.0))
        
        # Force outer lane for specific roles (comma-separated, e.g., "ego_vehicle_2,ego_vehicle_3")
        outer_lane_roles_str = str(rospy.get_param("~force_outer_lane_roles", "")).strip()
        self.force_outer_lane_roles = set()
        if outer_lane_roles_str:
            self.force_outer_lane_roles = {role.strip() for role in outer_lane_roles_str.split(",") if role.strip()}
            rospy.loginfo("force_outer_lane_roles: %s", self.force_outer_lane_roles)
        
        # Fixed loop path for platoon vehicles
        self.fixed_loop_path_file = rospy.get_param("~fixed_loop_path_file", "")
        # Platoon roles: leader (ego_vehicle_1) + followers (ego_vehicle_2, ego_vehicle_3)
        self.platoon_roles = set()
        # Shared-path role for platooning (주로 ego_vehicle_2, follower 중 첫 번째)
        self.platoon_shared_role: Optional[str] = None
        platoon_enable = rospy.get_param("~platoon_enable", False)
        if platoon_enable:
            # Get platoon roles from launch file (default: ego_vehicle_1, ego_vehicle_2, ego_vehicle_3)
            leader_role = str(rospy.get_param("~platoon_leader_role", "ego_vehicle_1"))
            follower_roles_str = str(rospy.get_param("~platoon_follower_roles", "ego_vehicle_2,ego_vehicle_3")).strip()
            follower_roles = [r.strip() for r in follower_roles_str.split(",") if r.strip()]
            self.platoon_roles = {leader_role} | set(follower_roles)
            # Shared path role: use first follower if available, else leader
            if follower_roles:
                self.platoon_shared_role = follower_roles[0]
            else:
                self.platoon_shared_role = leader_role
            rospy.loginfo("Platoon enabled: roles=%s, shared_path_role=%s", sorted(self.platoon_roles), self.platoon_shared_role)
        
        self.fixed_loop_path_points: Optional[List[Tuple[float, float, float]]] = None
        if self.fixed_loop_path_file:
            self.fixed_loop_path_points = self._load_fixed_loop_path(self.fixed_loop_path_file)
            if self.fixed_loop_path_points:
                rospy.loginfo("Loaded fixed loop path: %d points from %s", 
                             len(self.fixed_loop_path_points), self.fixed_loop_path_file)
                if self.platoon_roles:
                    rospy.loginfo("Fixed loop path will be used for platoon vehicles: %s", sorted(self.platoon_roles))
        
        # Safety area bounds for intersection detection (only apply outer lane filter outside intersections)
        # These should match the safety area bounds in multi_vehicle_controller
        self.safety_x_min = float(rospy.get_param("~safety_x_min", -18.0))
        self.safety_x_max = float(rospy.get_param("~safety_x_max", 18.0))
        self.safety_y_min = float(rospy.get_param("~safety_y_min", -35.0))
        self.safety_y_max = float(rospy.get_param("~safety_y_max", 3.0))

        # 시작 웨이포인트 선택 제약
        self.start_heading_deg = float(rospy.get_param("~start_heading_deg", 30.0))
        self.start_search_radius = float(rospy.get_param("~start_search_radius", 5.0))
        self.start_search_radius_max = float(rospy.get_param("~start_search_radius_max", 15.0))
        self.start_k_candidates = int(rospy.get_param("~start_k_candidates", 40))
        self.start_offset_m = float(rospy.get_param("~start_offset_m", 2.0))
        self.start_join_max_gap_m = float(rospy.get_param("~start_join_max_gap_m", 6.0))

        # 간단 충돌 회피 파라미터
        self.conflict_horizon_m = float(rospy.get_param("~conflict_horizon_m", 60.0))
        self.conflict_key_resolution = float(rospy.get_param("~conflict_key_resolution", 1.0))
        self.opposite_heading_thresh_deg = float(rospy.get_param("~opposite_heading_thresh_deg", 100.0))
        # 초기 추종 가능성(헤딩 정합) 체크 파라미터
        self.heading_compat_deg = float(rospy.get_param("~heading_compat_deg", 45.0))
        self.heading_compat_dist_m = float(rospy.get_param("~heading_compat_dist_m", 8.0))

        # 사용자가 편집한 글로벌 루트 YAML(예: carla_global_route_editor.py 결과)을
        # "동적 전역 경로 계획의 후보 목적지(웨이포인트 집합)" 및
        # "GlobalRoutePlanner 경로의 기하 보정(waypoint 위치 미세 조정)"에 사용할 수 있도록 하는 옵션
        # use_custom_goal_points: on/off 스위치
        self.use_custom_goal_points: bool = bool(
            rospy.get_param("~use_custom_goal_points", False)
        )
        self.custom_goal_points_file = rospy.get_param("~custom_goal_points_file", "")
        # carla.Location 리스트로 보관 (도로 위로 snap 된 위치)
        self.custom_goal_locs: List[carla.Location] = []
        # 기하 보정을 위한 (grid cell -> (x, y)) 맵
        # GlobalRoutePlanner 가 생성하는 route 의 waypoint 들을, 이 좌표들로 미세 수정할 때 사용
        self.custom_geom_points: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # 전역 경로에서 특정 영역에만 좌우 / X / Y 오프셋을 적용하기 위한 영역 설정 파일 (선택)
        # carla_path_offset_region_editor.py 로 생성한 path_offset_regions.yaml 사용
        self.offset_regions_file = rospy.get_param("~offset_regions_file", "")
        # 각 영역: {x_min, x_max, y_min, y_max, offset_lateral_m, offset_x_m, offset_y_m}
        self.offset_regions: List[Dict[str, float]] = []
        # 영역 경계에서 오프셋을 부드럽게 0으로 줄여가는 blending 폭 (m)
        self.offset_blend_width_m = float(rospy.get_param("~offset_blend_width_m", 5.0))

        # Visualisation
        self.enable_visualization = bool(rospy.get_param("~enable_visualization", True))
        self.visualization_lifetime = float(rospy.get_param("~visualization_lifetime", 30.0))

        # CARLA world/map
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        # Route planner
        if GlobalRoutePlannerDAO is not None:
            dao = GlobalRoutePlannerDAO(self.carla_map, self.global_route_resolution)
            self.route_planner = GlobalRoutePlanner(dao)
        else:
            self.route_planner = GlobalRoutePlanner(self.carla_map, self.global_route_resolution)
        if hasattr(self.route_planner, "setup"):
            self.route_planner.setup()

        # Waypoint cache for nearest queries
        waypoint_resolution = max(0.5, min(2.0, float(self.global_route_resolution)))
        self._waypoint_cache = [
            wp
            for wp in self.carla_map.generate_waypoints(waypoint_resolution)
            if wp.lane_type == carla.LaneType.Driving
        ]

        self.spawn_points = self.carla_map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        # 커스텀 목적지 후보 로드 (옵션 on 이고 파일이 지정된 경우)
        if self.use_custom_goal_points and self.custom_goal_points_file:
            self._load_custom_goal_points(self.custom_goal_points_file)

        # 좌우 오프셋 영역 로드 (있다면)
        if self.offset_regions_file:
            self._load_offset_regions(self.offset_regions_file)

        # State
        self.vehicles: List[carla.Actor] = []
        self.active_destinations: set[int] = set()
        self.vehicle_paths: Dict[str, List[Tuple[float, float]]] = {}
        self.vehicle_path_s: Dict[str, List[float]] = {}
        self.vehicle_path_len: Dict[str, float] = {}
        self._planning_lock = threading.RLock()
        # Mode: when True, platoon vehicles use fixed loop path only (no auto planning)
        # when False, platoon vehicles use normal/dynamic global planning (and ego_vehicle_2 path is shared via platoon_manager)
        self.fixed_loop_mode: bool = bool(self.fixed_loop_path_points and self.platoon_roles)

        # Override goal support (manual goal from RViz via rviz_goal_mux)
        # Map role -> (x, y, stamp)
        self.override_goal: Dict[str, Optional[Tuple[float, float, rospy.Time]]] = {
            self._role_name(i): None for i in range(self.num_vehicles)
        }
        self.override_subscribers = []
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/override_goal/{role}"
            sub = rospy.Subscriber(topic, PoseStamped, self._make_override_cb(role), queue_size=1)
            self.override_subscribers.append(sub)

        # External replan trigger (e.g., after teleport)
        rospy.Subscriber("/force_replan", String, self._force_replan_cb, queue_size=10)

        # Publishers per role
        self.path_publishers: Dict[str, rospy.Publisher] = {}
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/global_path_{role}"
            self.path_publishers[role] = rospy.Publisher(topic, Path, queue_size=1, latch=True)

        # Finished event publisher (override goal reached)
        self.finished_pub = rospy.Publisher("/override_goal_finished", String, queue_size=10)

        # Kickoff
        rospy.sleep(1.0)
        self._refresh_vehicles()
        self._plan_for_all()
        rospy.Timer(rospy.Duration(self.replan_interval), self._timer_cb)

    # -------------------------------------------------------------
    # Timers
    # -------------------------------------------------------------
    def _timer_cb(self, _event) -> None:
        with self._planning_lock:
            self._refresh_vehicles()
            self._plan_for_all()

    # -------------------------------------------------------------
    # Vehicle discovery
    # -------------------------------------------------------------
    def _refresh_vehicles(self) -> None:
        actors = self.world.get_actors().filter("vehicle.*")
        vehicles: List[carla.Actor] = []
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role.startswith("ego_vehicle_"):
                vehicles.append(actor)
        vehicles.sort(key=lambda veh: veh.attributes.get("role_name", ""))
        self.vehicles = vehicles
        rospy.loginfo_throttle(5.0, f"ConflictFreePlanner tracking {len(self.vehicles)} ego vehicles")

    # -------------------------------------------------------------
    # Planning
    # -------------------------------------------------------------
    def _plan_for_all(self) -> None:
        # Build conflict keys from already chosen paths this cycle
        taken_keys: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # Update fixed-loop vs dynamic mode for platoon vehicles based on override goals
        platoon_override_active = False
        if self.platoon_roles:
            for role in self.platoon_roles:
                ov = self.override_goal.get(role)
                if ov is not None:
                    platoon_override_active = True
                    break

        if platoon_override_active and self.fixed_loop_mode:
            rospy.loginfo_throttle(2.0, "MultiAgentPlanner: platoon override active -> switching to DYNAMIC path mode (no fixed loop)")
            self.fixed_loop_mode = False
        elif (not platoon_override_active) and self.fixed_loop_path_points and self.platoon_roles and not self.fixed_loop_mode:
            # No active override for platoon roles: return to fixed loop mode
            rospy.loginfo_throttle(2.0, "MultiAgentPlanner: no platoon override -> switching back to FIXED LOOP mode")
            self.fixed_loop_mode = True

        for index, vehicle in enumerate(self.vehicles[: self.num_vehicles]):
            role = self._role_name(index)
            # If override goal is active for this role, prioritize it
            ov = self.override_goal.get(role)
            if ov is not None:
                # If override goal already reached, clear and continue normal planning
                if self._check_override_goal(vehicle, role, ov):
                    continue
                self._plan_override(vehicle, index, ov)
                # Accumulate conflict keys for subsequent vehicles using the just planned route if present
                points = self.vehicle_paths.get(role)
                if points and len(points) >= 2:
                    # Synthesize pseudo-route by snapping points to cells for conflict keys
                    for x, y in points:
                        key = (int(round(x / max(0.25, self.conflict_key_resolution))),
                               int(round(y / max(0.25, self.conflict_key_resolution))))
                        taken_keys[key] = (x, y)
                continue
            front_loc, front_xy, yaw_rad = self._vehicle_front(vehicle)

            # CRITICAL: If fixed-loop mode is enabled and this is a platoon vehicle, ONLY use fixed loop path
            # DO NOT allow any automatic path planning or replanning for platoon vehicles in this mode
            # This check MUST be before any replanning logic to completely bypass path planning
            if self.fixed_loop_mode and self.platoon_roles and role in self.platoon_roles and self.fixed_loop_path_points:
                # Use fixed loop path for platoon vehicle - NO automatic path planning allowed
                # Skip ALL replanning checks - always use fixed loop path
                # Even if existing path is far from destination, use fixed loop path
                points = self._get_fixed_loop_path_from_current_position(front_xy)
                if points and len(points) >= 2:
                    # Ensure path starts at vehicle (or close enough)
                    points = self._ensure_path_starts_at_vehicle(points, front_xy)
                    
                    self.vehicle_paths[role] = points
                    self._store_path_geometry(role, points)
                    self._publish_path(points, role)
                    # Accumulate conflict keys for subsequent vehicles
                    for x, y in points:
                        key = (int(round(x / max(0.25, self.conflict_key_resolution))),
                               int(round(y / max(0.25, self.conflict_key_resolution))))
                        taken_keys[key] = (x, y)
                    rospy.logdebug_throttle(5.0, "%s: using fixed loop path (%d points) - NO auto planning/replanning", 
                                          role, len(points))
                    continue
                else:
                    rospy.logerr("%s: failed to generate fixed loop path - platoon vehicle cannot plan automatically!", role)
                    # Don't fall back to normal planning - just skip and use existing path if any
                    if self.vehicle_paths.get(role):
                        rospy.logwarn("%s: keeping existing path due to loop path failure (NO replanning allowed)", role)
                        continue
                    # Only if absolutely no path exists, allow fallback (should not happen in platooning)
                    rospy.logerr("%s: no path available - THIS SHOULD NOT HAPPEN for platoon vehicles", role)
                    # Still skip normal planning - let platoon_manager handle it
                    continue

            # Skip replanning if far from destination on existing path
            existing = self.vehicle_paths.get(role)
            if existing and len(existing) >= 2:
                remaining = self._compute_remaining_distance(front_xy, existing, self.vehicle_path_s.get(role), self.vehicle_path_len.get(role, 0.0))
                if remaining is not None and remaining > self.replan_remaining_m:
                    continue
            start_wp, start_meta = self._select_start_waypoint(front_loc, yaw_rad)
            if start_wp is None:
                rospy.logwarn("%s: no valid start waypoint", role)
                continue

            dest_index, route, points = self._sample_destination_route(start_wp, role)
            if dest_index is None or not route or len(points) < 2:
                rospy.logwarn("%s: destination sampling failed", role)
                continue
            
            # For platoon vehicles: use outer lane everywhere (including intersections)
            # This prevents turning inside intersections by forcing outer lane usage
            if role in self.force_outer_lane_roles:
                # Apply outer lane filtering everywhere (including intersections)
                # This forces the vehicle to use outer lane even inside intersections,
                # effectively preventing turns and making it go around
                route = self._filter_route_to_outer_lane(route, skip_intersections=False)
                if not route or len(route) < 2:
                    rospy.logwarn("%s: outer lane filtering produced invalid route", role)
                    continue
                points = self._route_to_points(route)
                if len(points) < 2:
                    rospy.logwarn("%s: outer lane route produced insufficient points", role)
                    continue

            # For platoon vehicles: check if route goes through intersection AFTER outer lane filtering
            if role in self.force_outer_lane_roles:
                if self._route_goes_through_intersection(route):
                    rospy.logwarn("%s: route still goes through intersection after outer lane filtering, retrying", role)
                    # Retry to find a route that avoids intersection
                    success = False
                    for retry in range(5):
                        d2, r2, p2 = self._sample_destination_route(start_wp, role)
                        if d2 is None or not r2 or len(p2) < 2:
                            continue
                        # Apply outer lane filtering
                        r2 = self._filter_route_to_outer_lane(r2, skip_intersections=False)
                        if not r2 or len(r2) < 2:
                            continue
                        p2 = self._route_to_points(r2)
                        if len(p2) < 2:
                            continue
                        # Check if this route avoids intersection
                        if not self._route_goes_through_intersection(r2):
                            dest_index, route, points = d2, r2, p2
                            success = True
                            break
                    if not success:
                        rospy.logwarn("%s: could not find route avoiding intersection, using best-effort path", role)
            
            # Conflict filtering: avoid overlapping initial horizon with already planned ones
            ok_points = self._filter_conflicts(route, taken_keys)
            if not ok_points:
                # Retry a few times for this vehicle
                success = False
                for _ in range(8):
                    d2, r2, p2 = self._sample_destination_route(start_wp, role)
                    if d2 is None or not r2 or len(p2) < 2:
                        continue
                    # For platoon vehicles: apply outer lane filtering and check intersection
                    if role in self.force_outer_lane_roles:
                        r2 = self._filter_route_to_outer_lane(r2, skip_intersections=False)
                        if not r2 or len(r2) < 2:
                            continue
                        p2 = self._route_to_points(r2)
                        if len(p2) < 2:
                            continue
                        # Prefer routes that avoid intersection
                        if self._route_goes_through_intersection(r2) and attempts < 5:
                            continue
                    ok_points = self._filter_conflicts(r2, taken_keys)
                    if ok_points:
                        dest_index, route, points = d2, r2, p2
                        success = True
                        break
                if not success:
                    rospy.logwarn("%s: conflict-free destination not found; using best-effort path", role)

            # Heading compatibility: ensure initial route segment aligns with current yaw
            if not self._route_heading_compatible(route, yaw_rad):
                compatible_found = False
                for _ in range(8):
                    d2, r2, p2 = self._sample_destination_route(start_wp, role)
                    if d2 is None or not r2 or len(p2) < 2:
                        continue
                    if not self._filter_conflicts(r2, taken_keys):
                        continue
                    if self._route_heading_compatible(r2, yaw_rad):
                        dest_index, route, points = d2, r2, p2
                        compatible_found = True
                        break
                if not compatible_found:
                    # Keep existing path if any, skip replanning this cycle
                    if self.vehicle_paths.get(role):
                        rospy.loginfo("%s: heading-incompatible route; keeping existing path", role)
                        continue

            # Start join guard
            points = self._ensure_path_starts_at_vehicle(points, front_xy)

            # Publish
            self.vehicle_paths[role] = points
            self._store_path_geometry(role, points)
            self._publish_path(points, role)

            # Update conflict keys for next vehicles
            self._accumulate_conflict_keys(route, taken_keys)

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"
    
    def _load_fixed_loop_path(self, yaml_file: str) -> Optional[List[Tuple[float, float, float]]]:
        """Load fixed loop path from YAML file.
        
        Returns list of (x, y, yaw_rad) tuples, or None if loading fails.
        """
        if not yaml_file or not os.path.exists(yaml_file):
            rospy.logwarn("Fixed loop path file not found: %s", yaml_file)
            return None
        
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data or "path" not in data:
                rospy.logwarn("Invalid YAML format: missing 'path' key")
                return None
            
            path_points = []
            for point in data["path"]:
                if "x" in point and "y" in point:
                    x = float(point["x"])
                    y = float(point["y"])
                    # Use yaw_rad if available, otherwise compute from yaw_deg
                    if "yaw_rad" in point:
                        yaw = float(point["yaw_rad"])
                    elif "yaw_deg" in point:
                        yaw = math.radians(float(point["yaw_deg"]))
                    else:
                        yaw = 0.0
                    path_points.append((x, y, yaw))
            
            if len(path_points) < 2:
                rospy.logwarn("Path has less than 2 points")
                return None
            
            # Make it a closed loop by connecting last point to first point
            # Check if first and last points are close (already a loop)
            first_x, first_y, _ = path_points[0]
            last_x, last_y, _ = path_points[-1]
            dist_to_close = math.hypot(last_x - first_x, last_y - first_y)
            
            if dist_to_close > 5.0:  # If gap > 5m, add connection point
                # Add first point at the end to close the loop
                path_points.append(path_points[0])
                rospy.loginfo("Closed loop path by connecting last to first (gap: %.2fm)", dist_to_close)
            else:
                rospy.loginfo("Path is already a closed loop (gap: %.2fm)", dist_to_close)
            
            rospy.loginfo("Loaded fixed loop path: %d points, total length: %.2fm", 
                         len(path_points), data.get("total_length_m", 0.0))
            return path_points
            
        except Exception as e:
            rospy.logerr("Failed to load fixed loop path from %s: %s", yaml_file, str(e))
            return None
    
    def _find_nearest_path_index(self, current_pos: Tuple[float, float], path_points: List[Tuple[float, float, float]]) -> int:
        """Find the index of the nearest path point to current position.
        
        Returns the index of the closest point.
        """
        if not path_points:
            return 0
        
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, (x, y, _) in enumerate(path_points):
            dist = math.hypot(x - current_pos[0], y - current_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _get_fixed_loop_path_from_current_position(self, current_pos: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """Get fixed loop path starting from the nearest point to current position.
        
        This ensures the vehicle can join the loop path from any starting position.
        Returns list of (x, y) tuples for pure pursuit.
        """
        if not self.fixed_loop_path_points:
            return None
        
        # Find nearest point on the loop path
        nearest_idx = self._find_nearest_path_index(current_pos, self.fixed_loop_path_points)
        
        # Rotate the path so it starts from the nearest point
        # Create a new path that starts from nearest_idx and wraps around
        loop_path = []
        
        # Add points from nearest_idx to end
        for i in range(nearest_idx, len(self.fixed_loop_path_points)):
            x, y, _ = self.fixed_loop_path_points[i]
            loop_path.append((x, y))
        
        # Add points from start to nearest_idx (wrap around)
        for i in range(nearest_idx):
            x, y, _ = self.fixed_loop_path_points[i]
            loop_path.append((x, y))
        
        # Ensure we have a complete loop by adding the first point again at the end
        # This helps pure pursuit with modulo indexing
        if loop_path and len(loop_path) > 0:
            loop_path.append(loop_path[0])
        
        rospy.logdebug("Fixed loop path: starting from index %d (nearest to current position %.2f, %.2f), total points: %d", 
                      nearest_idx, current_pos[0], current_pos[1], len(loop_path))
        
        return loop_path if len(loop_path) >= 2 else None

    def _vehicle_front(self, vehicle) -> Tuple[carla.Location, Tuple[float, float], float]:
        transform = vehicle.get_transform()
        yaw_rad = math.radians(transform.rotation.yaw)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        offset = self.start_offset_m
        bb = getattr(vehicle, "bounding_box", None)
        if bb is not None and getattr(bb, "extent", None) is not None:
            offset = bb.extent.x + 0.3
        front_loc = carla.Location(
            x=transform.location.x + forward_x * offset,
            y=transform.location.y + forward_y * offset,
            z=transform.location.z,
        )
        return front_loc, (front_loc.x, front_loc.y), yaw_rad

    def _make_override_cb(self, role: str):
        def _cb(msg: PoseStamped):
            stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
            goal_info = (float(msg.pose.position.x), float(msg.pose.position.y), stamp)
            with self._planning_lock:
                # If this is a platoon role, redirect override to shared-path role (e.g., ego_vehicle_2)
                target_role = role
                if self.platoon_roles and role in self.platoon_roles and self.platoon_shared_role:
                    target_role = self.platoon_shared_role
                self.override_goal[target_role] = goal_info
                # Drop stored path so visualizers/controllers get the new one on next publish
                self.vehicle_paths.pop(target_role, None)
                self.vehicle_path_s.pop(target_role, None)
                self.vehicle_path_len.pop(target_role, None)
                # When manual override is set for platoon, leave fixed_loop_mode False (dynamic mode)
                if self.platoon_roles:
                    self.fixed_loop_mode = False
            rospy.loginfo("%s: override set to (%.2f, %.2f) (stored for %s)", role, goal_info[0], goal_info[1], target_role)
        return _cb

    def _force_replan_cb(self, msg: String) -> None:
        role = msg.data.strip()
        if not role:
            return
        with self._planning_lock:
            # Clear any override and stored path so next cycle plans fresh from current pose
            self.override_goal[role] = None
            self.vehicle_paths.pop(role, None)
            self.vehicle_path_s.pop(role, None)
            self.vehicle_path_len.pop(role, None)
        rospy.loginfo("%s: force replan requested", role)

    def _snap_to_waypoint(self, xy: Tuple[float, float]):
        loc = carla.Location(x=xy[0], y=xy[1], z=0.5)
        return self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    def _plan_override(self, vehicle, index: int, goal_info: Tuple[float, float, rospy.Time]) -> None:
        role = self._role_name(index)
        goal_xy = (goal_info[0], goal_info[1])

        front_loc, front_xy, yaw_rad = self._vehicle_front(vehicle)
        start_wp, start_meta = self._select_start_waypoint(front_loc, yaw_rad)
        if start_wp is None:
            rospy.logwarn("%s: override planning failed; no valid start waypoint", role)
            return

        goal_wp = self._snap_to_waypoint(goal_xy)
        if goal_wp is None:
            rospy.logwarn("%s: override goal off-lane; ignoring", role)
            return

        route = self.route_planner.trace_route(start_wp.transform.location, goal_wp.transform.location)
        print(f"route: {route}")
        if not route or len(route) < 2:
            rospy.logwarn("%s: override route trace failed", role)
            return
        
        # Filter route to outer lane if required for this role
        # For platoon vehicles: use outer lane everywhere (including intersections) to prevent turns
        if role in self.force_outer_lane_roles:
            # Apply outer lane filtering everywhere (including intersections)
            # This forces the vehicle to use outer lane even inside intersections,
            # effectively preventing turns and making it go around
            route = self._filter_route_to_outer_lane(route, skip_intersections=False)
            if not route or len(route) < 2:
                rospy.logwarn("%s: override outer lane filtering failed", role)
                return
        
        if not self._route_heading_compatible(route, yaw_rad):
            rospy.logwarn("%s: override route heading-incompatible; ignoring", role)
            return

        points = self._route_to_points(route)
        if len(points) < 2:
            rospy.logwarn("%s: override route produced insufficient samples", role)
            return
        points = self._ensure_path_starts_at_vehicle(points, front_xy)

        self.vehicle_paths[role] = points
        self._store_path_geometry(role, points)
        self._publish_path(points, role)

    def _route_heading_compatible(self, route: List[Tuple], yaw_rad: float) -> bool:
        if not route or len(route) < 2:
            return False
        max_deg = 45.0
        thresh = math.radians(max_deg)
        acc = 0.0
        prev_wp = None
        for wp, _ in route:
            if prev_wp is not None:
                a = prev_wp.transform.location
                b = wp.transform.location
                dx = b.x - a.x
                dy = b.y - a.y
                seg_len = math.hypot(dx, dy)
                if seg_len > 1e-3:
                    heading = math.atan2(dy, dx)
                    if abs(_normalize_angle(heading - yaw_rad)) > thresh:
                        return False
                    acc += seg_len
                    if acc >= 8.0:
                        break
            prev_wp = wp
        return True

    def _check_override_goal(self, vehicle, role: str, goal_info: Tuple[float, float, rospy.Time]) -> bool:
        goal_xy = (goal_info[0], goal_info[1])
        curr = vehicle.get_location()
        dist = math.hypot(curr.x - goal_xy[0], curr.y - goal_xy[1])
        # Reuse destination_reached semantics similar to other planners
        threshold = 5.0
        if dist <= threshold:
            rospy.loginfo("%s: override goal reached; clearing override", role)
            self.override_goal[role] = None
            # Notify listeners that this role has finished its override goal
            try:
                self.finished_pub.publish(String(data=role))
            except Exception as exc:
                rospy.logwarn("%s: failed to publish finished event: %s", role, exc)
            # Keep existing path; next cycle will replan as needed
            return True
        return False

    def _select_start_waypoint(self, position: carla.Location, yaw_rad: float):
        if not self._waypoint_cache:
            return None, {"source": "no_waypoints"}

        forward = (math.cos(yaw_rad), math.sin(yaw_rad))
        primary_angle = math.radians(max(0.0, min(self.start_heading_deg, 85.0)))
        secondary_angle = math.radians(75.0)
        primary_radius = max(1.0, self.start_search_radius)
        secondary_radius = max(primary_radius, self.start_search_radius_max)
        k = max(1, int(self.start_k_candidates))

        def _distance_sq(wp: carla.Waypoint) -> float:
            loc = wp.transform.location
            dx = loc.x - position.x
            dy = loc.y - position.y
            return dx * dx + dy * dy

        candidates = heapq.nsmallest(k * 3, self._waypoint_cache, key=_distance_sq)

        def _select(cands: List[carla.Waypoint], max_radius: float, max_angle_rad: float, label: str):
            best_wp = None
            best_dist = float("inf")
            for wp in cands:
                loc = wp.transform.location
                dx = loc.x - position.x
                dy = loc.y - position.y
                dist = math.hypot(dx, dy)
                if dist < 1e-2 or dist > max_radius:
                    continue
                dot = dx * forward[0] + dy * forward[1]
                if dot <= 0.0:
                    continue
                cos_angle = dot / max(dist, 1e-3)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                heading_angle = math.acos(cos_angle)
                if heading_angle > max_angle_rad:
                    continue
                lane_yaw = math.radians(wp.transform.rotation.yaw)
                lane_delta = abs(_normalize_angle(lane_yaw - yaw_rad))
                if lane_delta > max_angle_rad + math.radians(5.0):
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_wp = wp
            if best_wp is None:
                return None, {"source": f"no_{label}"}
            return best_wp, {"source": label, "distance": best_dist}

        primary = _select(candidates, primary_radius, primary_angle, "primary")
        if primary[0] is not None:
            return primary
        secondary = _select(candidates, secondary_radius, secondary_angle, "expanded")
        if secondary[0] is not None:
            return secondary
        fallback_wp = self.carla_map.get_waypoint(position, project_to_road=True, lane_type=carla.LaneType.Driving)
        if fallback_wp is not None:
            for ds in (1.0, 1.5, 2.0, 3.0):
                next_wps = fallback_wp.next(ds)
                for candidate in next_wps:
                    loc = candidate.transform.location
                    dx = loc.x - position.x
                    dy = loc.y - position.y
                    dist = math.hypot(dx, dy)
                    if dist < 1e-2:
                        continue
                    dot = dx * forward[0] + dy * forward[1]
                    if dot <= 0.0:
                        continue
                    lane_yaw = math.radians(candidate.transform.rotation.yaw)
                    lane_delta = abs(_normalize_angle(lane_yaw - yaw_rad))
                    if lane_delta > math.radians(95.0):
                        continue
                    return candidate, {"source": f"fallback_next_{ds:.1f}", "distance": dist}
            return fallback_wp, {"source": "fallback_base", "distance": 0.0}
        return None, {"source": "no_match"}

    def _route_to_points(self, route: List[Tuple]) -> List[Tuple[float, float]]:
        waypoints = [item[0] for item in route]
        if not waypoints:
            return []
        pts: List[Tuple[float, float]] = []
        last: Optional[Tuple[float, float]] = None

        # geometry 보정용 grid resolution
        try:
            geom_res = max(0.25, float(self.conflict_key_resolution))
        except Exception:
            geom_res = 1.0

        for wp in waypoints:
            loc = wp.transform.location
            x = loc.x
            y = loc.y

            # 사용자가 편집한 글로벌 루트 YAML 좌표로 waypoint 위치를 미세 조정
            # (같은 grid cell 에 대응되는 edited_global_route 점이 있으면 그 좌표로 치환)
            if self.use_custom_goal_points and self.custom_geom_points:
                key = (int(round(x / geom_res)), int(round(y / geom_res)))
                override = self.custom_geom_points.get(key)
                if override is not None:
                    x, y = override

            # 2) 선택된 offset 영역에 있는 경우 lateral/X/Y offset 적용
            if self.offset_regions:
                off_lat, off_x, off_y = self._get_offsets_for_point(x, y)
                if abs(off_lat) > 1e-4 or abs(off_x) > 1e-4 or abs(off_y) > 1e-4:
                    # 차량 진행 방향 기준 좌/우 법선으로 lateral 적용
                    yaw = math.radians(wp.transform.rotation.yaw)
                    nx = -math.sin(yaw)
                    ny = math.cos(yaw)
                    x += nx * off_lat + off_x
                    y += ny * off_lat + off_y

            curr = (x, y)
            if last is None:
                pts.append(curr)
                last = curr
                continue
            seg = math.hypot(curr[0] - last[0], curr[1] - last[1])
            if seg < 1e-3:
                continue
            steps = max(1, int(seg // self.path_sampling))
            for step in range(1, steps + 1):
                ratio = min(1.0, (step * self.path_sampling) / seg)
                interp = (
                    last[0] + (curr[0] - last[0]) * ratio,
                    last[1] + (curr[1] - last[1]) * ratio,
                )
                if not pts or math.hypot(interp[0] - pts[-1][0], interp[1] - pts[-1][1]) > 0.05:
                    pts.append(interp)
            if math.hypot(curr[0] - pts[-1][0], curr[1] - pts[-1][1]) > 0.05:
                pts.append(curr)
            last = curr
        return pts

    def _route_heading_compatible(self, route: List[Tuple], yaw_rad: float) -> bool:
        if not route or len(route) < 2:
            return False
        max_deg = max(0.0, min(180.0, self.heading_compat_deg))
        thresh = math.radians(max_deg)
        acc = 0.0
        prev_wp = None
        for wp, _ in route:
            if prev_wp is not None:
                a = prev_wp.transform.location
                b = wp.transform.location
                dx = b.x - a.x
                dy = b.y - a.y
                seg_len = math.hypot(dx, dy)
                if seg_len > 1e-3:
                    heading = math.atan2(dy, dx)
                    if abs(_normalize_angle(heading - yaw_rad)) > thresh:
                        return False
                    acc += seg_len
                    if acc >= max(0.5, self.heading_compat_dist_m):
                        break
            prev_wp = wp
        return True

    def _route_length(self, route: List[Tuple]) -> float:
        total = 0.0
        prev = None
        for waypoint, _ in route:
            loc = waypoint.transform.location
            if prev is not None:
                total += loc.distance(prev)
            prev = loc
        return total

    def _store_path_geometry(self, role: str, points: List[Tuple[float, float]]) -> None:
        if not points or len(points) < 2:
            self.vehicle_path_s.pop(role, None)
            self.vehicle_path_len[role] = 0.0
            return
        s_profile: List[float] = [0.0]
        total = 0.0
        for i in range(1, len(points)):
            step = math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            total += step
            s_profile.append(total)
        self.vehicle_path_s[role] = s_profile
        self.vehicle_path_len[role] = total

    def _compute_remaining_distance(
        self,
        front_xy: Tuple[float, float],
        points: List[Tuple[float, float]],
        s_profile: Optional[List[float]],
        total_len: float,
    ) -> Optional[float]:
        if not points or len(points) < 2 or not s_profile or total_len <= 0.0:
            return None
        px, py = front_xy
        best_dist_sq = float("inf")
        best_idx = None
        best_t = 0.0
        for idx in range(len(points) - 1):
            x1, y1 = points[idx]
            x2, y2 = points[idx + 1]
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
                best_idx = idx
                best_t = t
        if best_idx is None:
            return None
        seg_len = math.hypot(points[best_idx + 1][0] - points[best_idx][0], points[best_idx + 1][1] - points[best_idx][1])
        if seg_len < 1e-6:
            s_now = s_profile[best_idx]
        else:
            s_now = s_profile[best_idx] + best_t * seg_len
        return max(0.0, total_len - s_now)

    def _is_in_safety_area(self, location: carla.Location) -> bool:
        """Check if location is within safety area (intersection)."""
        x = location.x
        y = location.y
        return (self.safety_x_min <= x <= self.safety_x_max and 
                self.safety_y_min <= y <= self.safety_y_max)
    
    def _filter_route_avoid_intersections(self, route: List[Tuple]) -> List[Tuple]:
        """Filter route to avoid going through intersections.
        
        For platoon vehicles, we want to go around intersections using outer lanes
        rather than turning inside intersections.
        This function keeps waypoints outside intersections and uses outer lane waypoints
        inside intersections to prevent turns.
        """
        if not route or len(route) < 2:
            return route
        
        # Don't remove waypoints - instead, we'll use outer lane filtering to prevent turns
        # This ensures route continuity while forcing outer lane usage
        return route
    
    def _filter_route_to_outer_lane(self, route: List[Tuple], skip_intersections: bool = True) -> List[Tuple]:
        """Filter route to use only outer lane waypoints (largest absolute lane_id).
        
        This helps avoid intersections by staying on the outer lane.
        Uses CARLA's waypoint API to find the outermost lane at each waypoint location.
        
        Args:
            route: List of (waypoint, road_option) tuples
            skip_intersections: If True, skip filtering inside safety area (allows turns in intersections)
        """
        if not route or len(route) < 2:
            return route
        
        filtered = []
        for wp, road_option in route:
            # Skip filtering if inside safety area (intersection) and skip_intersections is True
            if skip_intersections and self._is_in_safety_area(wp.transform.location):
                # Keep original waypoint inside intersection to allow turns
                filtered.append((wp, road_option))
                continue
            
            # Find outermost lane by checking adjacent lanes
            # CARLA: lane_id positive = forward direction, negative = backward
            # Larger absolute value = outer lane
            outer_wp = wp
            max_lane_id_abs = abs(wp.lane_id)
            current_wp = wp
            
            # Try to get right lane (outer lane for right-hand traffic)
            # Keep going right until we reach the outermost lane
            for _ in range(5):  # Maximum 5 lane changes (safety limit)
                try:
                    right_wp = current_wp.get_right_lane()
                    if right_wp is None or right_wp.lane_type != carla.LaneType.Driving:
                        break
                    if abs(right_wp.lane_id) > max_lane_id_abs:
                        outer_wp = right_wp
                        max_lane_id_abs = abs(right_wp.lane_id)
                        current_wp = right_wp
                    else:
                        break
                except Exception:
                    break
            
            # Also check left lane in case we're on the wrong side
            # But prioritize right lane (outer lane for right-hand traffic)
            current_wp = wp
            for _ in range(5):
                try:
                    left_wp = current_wp.get_left_lane()
                    if left_wp is None or left_wp.lane_type != carla.LaneType.Driving:
                        break
                    if abs(left_wp.lane_id) > max_lane_id_abs:
                        # Only use left lane if it's truly outer (shouldn't happen in RHT)
                        outer_wp = left_wp
                        max_lane_id_abs = abs(left_wp.lane_id)
                        current_wp = left_wp
                    else:
                        break
                except Exception:
                    break
            
            filtered.append((outer_wp, road_option))
        
        return filtered
    
    def _route_goes_through_intersection(self, route: List[Tuple]) -> bool:
        """Check if route goes through intersection (safety area).
        
        Returns True if route has significant portion inside safety area (intersection).
        For platoon vehicles, we want to avoid planning routes that go through intersections.
        """
        if not route:
            return False
        intersection_count = 0
        total_count = len(route)
        
        for wp, _ in route:
            if self._is_in_safety_area(wp.transform.location):
                intersection_count += 1
        
        # If more than 5% of waypoints are in intersection, consider it going through
        # This is stricter than before to avoid planning routes through intersections
        intersection_ratio = float(intersection_count) / float(total_count) if total_count > 0 else 0.0
        return intersection_ratio > 0.05 or intersection_count > 5
    
    def _sample_destination_route(self, start_waypoint, role: str = None) -> Tuple[Optional[int], Optional[List[Tuple]], Optional[List[Tuple[float, float]]]]:
        start_loc = start_waypoint.transform.location
        attempts = 0
        # For platoon vehicles, prefer routes that don't go through intersections
        avoid_intersections = role in self.force_outer_lane_roles if role else False
        
        while attempts < self.destination_retry_limit:
            attempts += 1
            # 1) 사용자가 편집한 글로벌 루트 YAML에서 목적지 후보를 가져오는 모드 (옵션 on + 로드 성공 시)
            if self.use_custom_goal_points and self.custom_goal_locs:
                dest_index = random.randrange(len(self.custom_goal_locs))
                if dest_index in self.active_destinations:
                    continue
                dest_loc = self.custom_goal_locs[dest_index]
                rospy.loginfo_throttle(
                    5.0,
                    "MultiAgentPlanner: using CUSTOM goal idx=%d at (x=%.2f, y=%.2f) for role=%s",
                    dest_index,
                    dest_loc.x,
                    dest_loc.y,
                    role or "unknown",
                )
            # 2) 기본 spawn point 기반 목적지 샘플링
            else:
                dest_index = random.randrange(len(self.spawn_points))
                if dest_index in self.active_destinations:
                    continue
                dest_loc = self.spawn_points[dest_index].location
            euclid = start_loc.distance(dest_loc)
            if euclid < self.min_destination_distance or euclid > self.max_destination_distance * 1.5:
                continue
            route = self.route_planner.trace_route(start_loc, dest_loc)
            if not route or len(route) < 2:
                continue
            
            # For platoon vehicles: STRICTLY avoid routes that go through intersections
            if avoid_intersections and self._route_goes_through_intersection(route):
                # Skip routes that go through intersections (try next destination)
                # Only allow intersection routes as absolute last resort (last 10% of attempts)
                if attempts < self.destination_retry_limit * 0.9:
                    continue
                rospy.logwarn_throttle(1.0, "%s: using intersection route as last resort (attempt %d/%d)", 
                                       role, attempts, self.destination_retry_limit)
            
            length = self._route_length(route)
            if length < self.min_destination_distance or length > self.max_destination_distance * 1.5:
                continue
            points = self._route_to_points(route)
            if len(points) < 2:
                continue
            return dest_index, route, points
        # Fallback: broaden distance bounds deterministically over all spawn points
        ordered = sorted(range(len(self.spawn_points)), key=lambda i: start_loc.distance(self.spawn_points[i].location))
        for dest_index in ordered:
            if dest_index in self.active_destinations:
                continue
            dest_loc = self.spawn_points[dest_index].location
            euclid = start_loc.distance(dest_loc)
            if euclid < max(5.0, self.min_destination_distance * 0.5) or euclid > max(self.max_destination_distance * 2.0, self.min_destination_distance + 10.0):
                continue
            route = self.route_planner.trace_route(start_loc, dest_loc)
            if not route or len(route) < 2:
                continue
            points = self._route_to_points(route)
            if len(points) < 2:
                continue
            return dest_index, route, points
        return None, None, None

    def _key_for_waypoint(self, wp: carla.Waypoint) -> Tuple[int, int]:
        loc = wp.transform.location
        res = max(0.25, self.conflict_key_resolution)
        return (int(round(loc.x / res)), int(round(loc.y / res)))

    def _accumulate_conflict_keys(self, route: List[Tuple], taken: Dict[Tuple[int, int], Tuple[float, float]]) -> None:
        acc = 0.0
        last_loc = None
        for wp, _ in route:
            loc = wp.transform.location
            if last_loc is not None:
                acc += loc.distance(last_loc)
            last_loc = loc
            key = self._key_for_waypoint(wp)
            taken[key] = (loc.x, loc.y)
            if self.conflict_horizon_m > 0.0 and acc >= self.conflict_horizon_m:
                break

    def _filter_conflicts(self, route: List[Tuple], taken: Dict[Tuple[int, int], Tuple[float, float]]) -> bool:
        if not taken:
            return True
        acc = 0.0
        last_loc = None
        for wp, _ in route:
            loc = wp.transform.location
            if last_loc is not None:
                acc += loc.distance(last_loc)
            last_loc = loc
            key = self._key_for_waypoint(wp)
            if key in taken:
                # Same cell occupied in horizon – consider conflict
                return False
            if self.conflict_horizon_m > 0.0 and acc >= self.conflict_horizon_m:
                break
        return True

    # -------------------------------------------------------------
    # 사용자 편집 글로벌 루트 YAML 로딩 (목적지 후보 웨이포인트 집합)
    # -------------------------------------------------------------
    def _load_custom_goal_points(self, yaml_file: str) -> None:
        """carla_global_route_editor.py 가 생성한 YAML에서 x,y 를 읽어 도로 위 carla.Location 리스트로 변환."""
        if not yaml_file or not os.path.exists(yaml_file):
            rospy.logwarn("MultiAgentPlanner: custom_goal_points_file not found: %s", yaml_file)
            return
        if self.carla_map is None:
            rospy.logwarn("MultiAgentPlanner: carla_map is None, cannot load custom goal points")
            return
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            rospy.logwarn("MultiAgentPlanner: failed to load custom goal YAML %s: %s", yaml_file, exc)
            return

        path_list = data.get("path") if isinstance(data, dict) else None
        if not path_list:
            rospy.logwarn("MultiAgentPlanner: custom goal YAML %s has no 'path' key or is empty", yaml_file)
            return

        goal_locs: List[carla.Location] = []
        geom_points: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # geometry 보정용 grid resolution (conflict_key_resolution 과 동일 계열 사용)
        try:
            geom_res = max(0.25, float(self.conflict_key_resolution))
        except Exception:
            geom_res = 1.0
        for idx, pt in enumerate(path_list):
            try:
                x = float(pt.get("x"))
                y = float(pt.get("y"))
            except Exception:
                continue
            # 1) geometry 보정을 위한 raw 좌표를 grid cell 에 매핑
            key = (int(round(x / geom_res)), int(round(y / geom_res)))
            geom_points[key] = (x, y)

            # 2) 목적지 후보용으로는 도로 위 waypoint 로 snap
            loc = carla.Location(x=x, y=y, z=0.5)
            try:
                wp = self.carla_map.get_waypoint(
                    loc, project_to_road=True, lane_type=carla.LaneType.Driving
                )
            except Exception:
                wp = self.carla_map.get_waypoint(loc, project_to_road=True)
            if wp is None:
                continue
            goal_locs.append(wp.transform.location)

        if not goal_locs:
            rospy.logwarn("MultiAgentPlanner: no valid custom goal points found in %s", yaml_file)
            return

        self.custom_goal_locs = goal_locs
        self.custom_geom_points = geom_points
        rospy.loginfo(
            "MultiAgentPlanner: loaded %d custom goal points from %s (used as destination candidates)",
            len(self.custom_goal_locs),
            yaml_file,
        )

    # -------------------------------------------------------------
    # Offset regions (좌우 오프셋 영역) 로딩 및 조회
    # -------------------------------------------------------------
    def _load_offset_regions(self, yaml_file: str) -> None:
        """carla_path_offset_region_editor.py 가 생성한 YAML에서 영역/offset 정보를 읽어온다."""
        if not yaml_file or not os.path.exists(yaml_file):
            rospy.logwarn("MultiAgentPlanner: offset_regions_file not found: %s", yaml_file)
            return
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            rospy.logwarn("MultiAgentPlanner: failed to load offset regions YAML %s: %s", yaml_file, exc)
            return

        regions_list = data.get("regions") if isinstance(data, dict) else None
        if not regions_list:
            rospy.logwarn("MultiAgentPlanner: offset regions YAML %s has no 'regions' key or is empty", yaml_file)
            return

        loaded: List[Dict[str, float]] = []
        for reg in regions_list:
            # 새 포맷: points (브러쉬로 찍은 world 좌표 리스트)
            pts = reg.get("points")
            corners = reg.get("corners") if pts is None else None
            if pts:
                try:
                    xs = [float(p[0]) for p in pts]
                    ys = [float(p[1]) for p in pts]
                except Exception:
                    continue
            elif corners:
                if len(corners) < 4:
                    continue
                try:
                    xs = [float(c[0]) for c in corners]
                    ys = [float(c[1]) for c in corners]
                except Exception:
                    continue
            else:
                continue
            try:
                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)
                offset_lat = float(reg.get("offset_lateral_m", 0.0))
                offset_x = float(reg.get("offset_x_m", 0.0))
                offset_y = float(reg.get("offset_y_m", 0.0))
            except Exception:
                continue
            loaded.append(
                {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "offset_lateral_m": offset_lat,
                    "offset_x_m": offset_x,
                    "offset_y_m": offset_y,
                }
            )

        if not loaded:
            rospy.logwarn("MultiAgentPlanner: no valid offset regions found in %s", yaml_file)
            return

        self.offset_regions = loaded
        rospy.loginfo(
            "MultiAgentPlanner: loaded %d offset regions from %s",
            len(self.offset_regions),
            yaml_file,
        )

    def _get_offsets_for_point(self, x: float, y: float) -> Tuple[float, float, float]:
        """해당 (x,y)가 어떤 offset 영역 안에 있으면 그 영역의 (lateral, x, y) offset 을 반환.

        영역 경계에서는 offset_blend_width_m 내에서 선형 블렌딩(0→1)을 적용하여
        경로가 부자연스럽게 꺾이지 않도록 한다.
        """
        if not self.offset_regions:
            return 0.0, 0.0, 0.0
        lat = 0.0
        ox = 0.0
        oy = 0.0
        blend_w = max(0.0, float(self.offset_blend_width_m))
        for reg in self.offset_regions:
            if (
                reg["x_min"] <= x <= reg["x_max"]
                and reg["y_min"] <= y <= reg["y_max"]
            ):
                # 영역 내부에서 가장 가까운 경계까지의 거리
                dx_min = x - reg["x_min"]
                dx_max = reg["x_max"] - x
                dy_min = y - reg["y_min"]
                dy_max = reg["y_max"] - y
                dist_edge = min(dx_min, dx_max, dy_min, dy_max)
                if blend_w > 1e-3:
                    w = max(0.0, min(1.0, dist_edge / blend_w))
                else:
                    w = 1.0
                lat += w * float(reg.get("offset_lateral_m", 0.0))
                ox += w * float(reg.get("offset_x_m", 0.0))
                oy += w * float(reg.get("offset_y_m", 0.0))
        return lat, ox, oy

    def _ensure_path_starts_at_vehicle(self, path_points: List[Tuple[float, float]], current_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        if not path_points:
            return []
        d0 = math.hypot(current_xy[0] - path_points[0][0], current_xy[1] - path_points[0][1])
        if d0 > max(0.0, self.start_join_max_gap_m):
            return path_points
        if d0 < 0.1:
            path_points[0] = current_xy
            return path_points
        return [current_xy] + path_points

    def _publish_path(self, path_points: List[Tuple[float, float]], role: str) -> None:
        publisher = self.path_publishers.get(role)
        if publisher is None:
            return
        header = Header(stamp=rospy.Time.now(), frame_id="map")
        msg = Path(header=header)
        for x, y in path_points:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        publisher.publish(msg)


if __name__ == "__main__":
    try:
        MultiAgentConflictFreePlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


