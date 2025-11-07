#!/usr/bin/env python3

import heapq
import math
import random
import threading
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
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

        # State
        self.vehicles: List[carla.Actor] = []
        self.active_destinations: set[int] = set()
        self.vehicle_paths: Dict[str, List[Tuple[float, float]]] = {}
        self.vehicle_path_s: Dict[str, List[float]] = {}
        self.vehicle_path_len: Dict[str, float] = {}
        self._planning_lock = threading.RLock()

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

        # Publishers per role
        self.path_publishers: Dict[str, rospy.Publisher] = {}
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/global_path_{role}"
            self.path_publishers[role] = rospy.Publisher(topic, Path, queue_size=1, latch=True)

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

            dest_index, route, points = self._sample_destination_route(start_wp)
            if dest_index is None or not route or len(points) < 2:
                rospy.logwarn("%s: destination sampling failed", role)
                continue

            # Conflict filtering: avoid overlapping initial horizon with already planned ones
            ok_points = self._filter_conflicts(route, taken_keys)
            if not ok_points:
                # Retry a few times for this vehicle
                success = False
                for _ in range(8):
                    d2, r2, p2 = self._sample_destination_route(start_wp)
                    if d2 is None or not r2 or len(p2) < 2:
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
                    d2, r2, p2 = self._sample_destination_route(start_wp)
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
                self.override_goal[role] = goal_info
                # Drop stored path so visualizers/controllers get the new one on next publish
                self.vehicle_paths.pop(role, None)
                self.vehicle_path_s.pop(role, None)
                self.vehicle_path_len.pop(role, None)
            rospy.loginfo("%s: override set to (%.2f, %.2f)", role, goal_info[0], goal_info[1])
        return _cb

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
        if not route or len(route) < 2:
            rospy.logwarn("%s: override route trace failed", role)
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
        for wp in waypoints:
            loc = wp.transform.location
            curr = (loc.x, loc.y)
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

    def _sample_destination_route(self, start_waypoint) -> Tuple[Optional[int], Optional[List[Tuple]], Optional[List[Tuple[float, float]]]]:
        start_loc = start_waypoint.transform.location
        attempts = 0
        while attempts < self.destination_retry_limit:
            attempts += 1
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


