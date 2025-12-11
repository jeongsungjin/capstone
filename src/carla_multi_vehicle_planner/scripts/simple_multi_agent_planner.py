#!/usr/bin/env python3
import heapq
import math
import random
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header

# Ensure CARLA Python API and Agents are on sys.path (side-effect import)
try:
    import setup_carla_path  # noqa: F401
except Exception:
    setup_carla_path = None  # type: ignore

try:
    import carla  # type: ignore
except Exception as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA package: {exc}")

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
except Exception:
    GlobalRoutePlanner = None  # type: ignore


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class SimpleMultiAgentPlanner:
    """
    최소 기능 다중 차량 글로벌 플래너.
    - 각 ego 차량의 현재 전방 위치를 시작점으로 사용
    - 스폰 포인트 중 거리 제약을 만족하는 목적지 임의 선택
    - CARLA GlobalRoutePlanner로 경로 생성 후 /global_path_{role} 퍼블리시
    - 충돌 회피, 수동 목표, 오프셋/시각화 등 부가 기능은 제거
    """

    def __init__(self) -> None:
        rospy.init_node("multi_agent_conflict_free_planner", anonymous=True)

        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 5))
        self.global_route_resolution = float(rospy.get_param("~global_route_resolution", 1.0))
        # Fallback lane-follow tuning (for when GRP is unavailable)
        self.lane_follow_step_m = float(rospy.get_param("~lane_follow_step_m", 0.3))      # default denser than 2.0
        self.lane_follow_length_m = float(rospy.get_param("~lane_follow_length_m", 150.0))
        self.path_thin_min_m = float(rospy.get_param("~path_thin_min_m", 0.1))            # default denser than 0.2
        # Replan policy: distance-gated (no fixed-period replanning)
        self.replan_gate_distance_m = float(rospy.get_param("~replan_gate_distance_m", 8.0))
        self.replan_soft_distance_m = float(rospy.get_param("~replan_soft_distance_m", 20.0))
        self.replan_check_interval = float(rospy.get_param("~replan_check_interval", 1.0))
        # Align first path segment with vehicle heading by looking slightly ahead when replanning
        self.heading_align_lookahead_m = float(rospy.get_param("~heading_align_lookahead_m", 2.5))
        self.heading_align_max_diff_deg = float(rospy.get_param("~heading_align_max_diff_deg", 60.0))
        # Start waypoint selection parameters
        self.start_heading_deg = float(rospy.get_param("~start_heading_deg", 30.0))
        self.start_search_radius = float(rospy.get_param("~start_search_radius", 5.0))
        self.start_search_radius_max = float(rospy.get_param("~start_search_radius_max", 15.0))
        self.start_k_candidates = int(rospy.get_param("~start_k_candidates", 40))
        self.start_join_max_gap_m = float(rospy.get_param("~start_join_max_gap_m", 12.0))
        self.start_offset_m = float(rospy.get_param("~start_offset_m", 3.0))
        self.path_extension_overlap_m = float(rospy.get_param("~path_extension_overlap_m", 8.0))
        self.heading_compat_deg = float(rospy.get_param("~heading_compat_deg", 60.0))
        self.heading_compat_dist_m = float(rospy.get_param("~heading_compat_dist_m", 5.0))
        self.min_destination_distance = float(rospy.get_param("~min_destination_distance", 80.0))
        self.max_destination_distance = float(rospy.get_param("~max_destination_distance", 180.0))

        # CARLA world/map
        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        timeout = float(rospy.get_param("~carla_timeout", 10.0))
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        waypoint_resolution = max(0.5, min(2.0, float(self.global_route_resolution)))
        self._waypoint_cache = [
            wp
            for wp in self.carla_map.generate_waypoints(waypoint_resolution)
            if getattr(wp, "lane_type", carla.LaneType.Driving) == carla.LaneType.Driving
        ]

        # Route planner (prefer CARLA agents; fallback to simple lane-follow)
        self.route_planner = None
        if GlobalRoutePlanner is not None:
            try:
                self.route_planner = GlobalRoutePlanner(self.carla_map, self.global_route_resolution)
                if hasattr(self.route_planner, "setup"):
                    self.route_planner.setup()
                rospy.loginfo("SimpleMultiAgentPlanner: using CARLA GlobalRoutePlanner")
            except Exception as exc:
                self.route_planner = None
                rospy.logwarn(f"GlobalRoutePlanner unavailable, falling back to lane-follow routing: {exc}")
        else:
            rospy.logwarn("GlobalRoutePlanner module not found; using lane-follow routing fallback")

        self.spawn_points = self.carla_map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        # Publishers
        self.path_publishers: Dict[str, rospy.Publisher] = {}
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/global_path_{role}"
            self.path_publishers[role] = rospy.Publisher(topic, Path, queue_size=1, latch=True)

        # Destination memory per role (for distance-gated replanning)
        self._current_dest: Dict[str, Optional[carla.Location]] = {self._role_name(i): None for i in range(self.num_vehicles)}
        self._active_paths: Dict[str, List[Tuple[float, float]]] = {}
        self._active_path_s: Dict[str, List[float]] = {}
        self._active_path_len: Dict[str, float] = {}

        rospy.sleep(0.5)
        self._plan_once()
        rospy.Timer(rospy.Duration(self.replan_check_interval), self._replan_check_cb)

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _replan_check_cb(self, _evt) -> None:
        # Distance-gated replanning per vehicle
        vehicles = self._get_ego_vehicles()
        if not vehicles:
            return
        for index, vehicle in enumerate(vehicles[: self.num_vehicles]):
            role = self._role_name(index)
            front_loc = self._vehicle_front(vehicle)
            current_path = self._active_paths.get(role)
            if not current_path or len(current_path) < 2:
                self._plan_for_role(vehicle, role, front_loc)
                continue
            remaining = self._remaining_path_distance(role, front_loc)
            if remaining is None:
                self._plan_for_role(vehicle, role, front_loc)
                continue
            if remaining <= max(0.0, float(self.replan_soft_distance_m)):
                self._extend_path(vehicle, role, front_loc)

    def _extend_path(self, vehicle, role: str, front_loc: carla.Location) -> None:
        current = self._active_paths.get(role)
        if not current or len(current) < 2:
            self._plan_for_role(vehicle, role, front_loc)
            return
        s_profile = self._active_path_s.get(role)
        suffix = current
        if s_profile and len(s_profile) == len(current):
            progress = self._project_progress_on_path(current, s_profile, front_loc.x, front_loc.y)
            if progress is not None:
                overlap_target = max(0.0, progress - float(self.path_extension_overlap_m))
                start_idx = 0
                for i, s in enumerate(s_profile):
                    if s >= overlap_target:
                        start_idx = i
                        break
                suffix = current[start_idx:]
        prefix_copy = list(suffix)
        if len(prefix_copy) < 2:
            self._plan_for_role(vehicle, role, front_loc)
            return
        self._plan_for_role(vehicle, role, front_loc, prefix_points=prefix_copy)

    def _get_ego_vehicles(self) -> List[carla.Actor]:
        actors = self.world.get_actors().filter("vehicle.*")
        vehicles: List[carla.Actor] = []
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role.startswith("ego_vehicle_"):
                vehicles.append(actor)
        vehicles.sort(key=lambda v: v.attributes.get("role_name", ""))
        return vehicles

    def _vehicle_front(self, vehicle: carla.Actor) -> carla.Location:
        tf = vehicle.get_transform()
        yaw_rad = math.radians(tf.rotation.yaw)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        offset = max(0.5, float(self.start_offset_m))
        bb = getattr(vehicle, "bounding_box", None)
        if bb is not None and getattr(bb, "extent", None) is not None:
            offset = bb.extent.x + 0.3
        return carla.Location(
            x=tf.location.x + forward_x * offset,
            y=tf.location.y + forward_y * offset,
            z=tf.location.z,
        )


    def _choose_destination(self, start: carla.Location, max_trials: int = 80) -> Optional[carla.Location]:
        for _ in range(max_trials):
            cand = random.choice(self.spawn_points).location
            dist = math.hypot(cand.x - start.x, cand.y - start.y)
            if self.min_destination_distance <= dist <= self.max_destination_distance:
                return cand
        return None

    def _trace_route(self, start: carla.Location, dest: carla.Location):
        if self.route_planner is not None:
            try:
                return self.route_planner.trace_route(start, dest)
            except Exception as exc:
                rospy.logwarn(f"trace_route failed: {exc}")
                return None
        # Fallback: generate a forward route by lane-follow (parameterized density)
        return self._lane_follow_route(
            start,
            length_m=float(self.lane_follow_length_m),
            step_m=max(0.05, float(self.lane_follow_step_m)),
        )

    def _lane_follow_route(self, start: carla.Location, length_m: float = 120.0, step_m: float = 2.0):
        try:
            wp = self.carla_map.get_waypoint(start, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                return None
            route = []
            traveled = 0.0
            current = wp
            # Seed with start waypoint
            route.append((current, None))
            while traveled < max(5.0, float(length_m)):
                nxt = current.next(float(step_m))
                if not nxt:
                    break
                # Choose the next waypoint that keeps heading smooth and lane consistent
                chosen = None
                try:
                    cur_yaw_deg = float(current.transform.rotation.yaw)
                    cur_yaw = math.radians(cur_yaw_deg)
                    cur_road = getattr(current, "road_id", None)
                    cur_lane = getattr(current, "lane_id", None)
                    best_score = float("inf")
                    for cand in nxt:
                        yaw_deg = float(cand.transform.rotation.yaw)
                        yaw = math.radians(yaw_deg)
                        # Smaller yaw change preferred
                        yaw_diff = abs((yaw - cur_yaw + math.pi) % (2.0 * math.pi) - math.pi)
                        same_lane = 1 if (getattr(cand, "road_id", None) == cur_road and getattr(cand, "lane_id", None) == cur_lane) else 0
                        # Score: prioritize same lane, then minimal heading change
                        score = (0 if same_lane else 1) * 10.0 + yaw_diff
                        if score < best_score:
                            best_score = score
                            chosen = cand
                except Exception:
                    chosen = None
                if chosen is None:
                    chosen = nxt[0]
                current = chosen
                route.append((current, None))
                traveled += float(step_m)
            return route if len(route) >= 2 else None
        except Exception as exc:
            rospy.logwarn(f"lane-follow routing failed: {exc}")
            return None

    def _route_to_points(self, route) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        if not route:
            return points
        last_x, last_y = None, None
        for wp, _ in route:
            loc = wp.transform.location
            x, y = float(loc.x), float(loc.y)
            if last_x is None:
                points.append((x, y))
                last_x, last_y = x, y
                continue
            dx, dy = x - last_x, y - last_y
            # Thin only when spacing exceeds configured threshold
            if dx * dx + dy * dy >= float(self.path_thin_min_m) * float(self.path_thin_min_m):
                points.append((x, y))
                last_x, last_y = x, y
        if len(points) >= 2:
            return points
        return points

    def _route_heading_compatible(self, route, yaw_rad: float) -> bool:
        if not route or len(route) < 2:
            return False
        thresh = math.radians(max(0.0, min(180.0, float(self.heading_compat_deg))))
        accum = 0.0
        prev_wp = None
        for wp, _ in route:
            if prev_wp is not None:
                a = prev_wp.transform.location
                b = wp.transform.location
                dx = b.x - a.x
                dy = b.y - a.y
                seg = math.hypot(dx, dy)
                if seg > 1e-3:
                    heading = math.atan2(dy, dx)
                    if abs(_normalize_angle(heading - yaw_rad)) > thresh:
                        return False
                    accum += seg
                    if accum >= max(0.5, float(self.heading_compat_dist_m)):
                        break
            prev_wp = wp
        return True

    def _ensure_path_starts_at_vehicle(self, path_points: List[Tuple[float, float]], front_xy: Tuple[float, float]):
        if not path_points:
            return []
        gap = math.hypot(front_xy[0] - path_points[0][0], front_xy[1] - path_points[0][1])
        max_gap = max(0.0, float(self.start_join_max_gap_m))
        if gap < 0.1:
            path_points[0] = front_xy
            return path_points
        if gap > max_gap:
            # Insert an intermediate point along heading direction before original path
            first_x, first_y = path_points[0]
            vec_x = first_x - front_xy[0]
            vec_y = first_y - front_xy[1]
            dist = math.hypot(vec_x, vec_y)
            if dist > 1e-3:
                span = min(dist, float(self.heading_align_lookahead_m))
                yaw = math.atan2(vec_y, vec_x)
                mid = (front_xy[0] + math.cos(yaw) * span, front_xy[1] + math.sin(yaw) * span)
                return [front_xy, mid] + path_points
        return [front_xy] + path_points

    def _snap_points_to_lane(self, points: List[Tuple[float, float]]):
        snapped: List[Tuple[float, float]] = []
        for x, y in points:
            loc = carla.Location(x=x, y=y, z=0.0)
            try:
                wp = self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            except Exception:
                wp = None
            if wp is not None:
                lane_loc = wp.transform.location
                snapped.append((lane_loc.x, lane_loc.y))
            else:
                snapped.append((x, y))
        return snapped

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

    def _project_progress_on_path(self, points: List[Tuple[float, float]], s_profile: List[float], px: float, py: float):
        if len(points) < 2:
            return None
        if not s_profile or len(s_profile) != len(points):
            s_profile, _ = self._compute_path_profile(points)
            if not s_profile or len(s_profile) != len(points):
                return None
        best_dist_sq = float("inf")
        best_index = None
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
            dist_sq = (proj_x - px) ** 2 + (proj_y - py) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_index = idx
                best_t = t
        if best_index is None:
            return None
        seg_length = math.hypot(points[best_index + 1][0] - points[best_index][0], points[best_index + 1][1] - points[best_index][1])
        if seg_length < 1e-6:
            s_now = s_profile[best_index]
        else:
            s_now = s_profile[best_index] + best_t * seg_length
        return s_now

    def _remaining_path_distance(self, role: str, front_loc: carla.Location):
        points = self._active_paths.get(role)
        if not points or len(points) < 2:
            return None
        s_profile = self._active_path_s.get(role) or []
        progress = self._project_progress_on_path(points, s_profile, front_loc.x, front_loc.y)
        if progress is None:
            return None
        path_len = self._active_path_len.get(role, 0.0)
        remaining = path_len - progress
        if remaining < 0.0:
            remaining = 0.0
        return remaining

    def _store_active_path(self, role: str, points: List[Tuple[float, float]]):
        if len(points) < 2:
            self._active_paths.pop(role, None)
            self._active_path_s.pop(role, None)
            self._active_path_len.pop(role, None)
            return
        s_profile, total_len = self._compute_path_profile(points)
        self._active_paths[role] = points
        self._active_path_s[role] = s_profile
        self._active_path_len[role] = total_len

    def _publish_path(self, points: List[Tuple[float, float]], role: str) -> None:
        if role not in self.path_publishers:
            return
        msg = Path()
        msg.header = Header(frame_id="map", stamp=rospy.Time.now())
        for x, y in points:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = 0.0
            msg.poses.append(p)
        self.path_publishers[role].publish(msg)

    def _plan_once(self) -> None:
        vehicles = self._get_ego_vehicles()
        if not vehicles:
            rospy.loginfo_throttle(5.0, "No ego vehicles found")
            return
        for index, vehicle in enumerate(vehicles[: self.num_vehicles]):
            role = self._role_name(index)
            front_loc = self._vehicle_front(vehicle)
            self._plan_for_role(vehicle, role, front_loc)

    def _plan_for_role(self, vehicle, role: str, front_loc: carla.Location, prefix_points: Optional[List[Tuple[float, float]]] = None) -> None:
        # Sample (or resample) destination and publish a fresh path
        dest_loc = self._choose_destination(front_loc)
        if dest_loc is None:
            rospy.logwarn_throttle(5.0, f"{role}: destination not found within distance bounds")
            return
        tf = vehicle.get_transform()
        yaw_rad = math.radians(tf.rotation.yaw)
        start_loc = None
        start_heading = yaw_rad
        if prefix_points is not None and len(prefix_points) >= 2:
            dx = prefix_points[-1][0] - prefix_points[-2][0]
            dy = prefix_points[-1][1] - prefix_points[-2][1]
            if dx * dx + dy * dy > 1e-3:
                start_heading = math.atan2(dy, dx)
        if prefix_points is not None and len(prefix_points) >= 1:
            last = prefix_points[-1]
            start_loc = carla.Location(x=last[0], y=last[1], z=front_loc.z)
        if start_loc is None:
            start_loc = carla.Location(x=front_loc.x + math.cos(start_heading) * float(self.heading_align_lookahead_m),
                                       y=front_loc.y + math.sin(start_heading) * float(self.heading_align_lookahead_m),
                                       z=front_loc.z)
        start_loc.z = front_loc.z
        route = None
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            route = self._trace_route(start_loc, dest_loc)
            if route and len(route) >= 2:
                if self.route_planner is None or self._route_heading_compatible(route, yaw_rad):
                    break
            dest_loc = self._choose_destination(front_loc)
            if dest_loc is None:
                route = None
                break
            attempts += 1
        if not route or len(route) < 2:
            rospy.logwarn_throttle(5.0, f"{role}: route trace failed")
            return
        new_points = self._route_to_points(route)
        if prefix_points is not None and len(prefix_points) >= 1:
            combined = prefix_points[:-1] + new_points
            points = self._snap_points_to_lane(combined)
        else:
            points = self._ensure_path_starts_at_vehicle(new_points, (front_loc.x, front_loc.y))
            points = self._snap_points_to_lane(points)
        if len(points) < 2:
            rospy.logwarn_throttle(5.0, f"{role}: insufficient path points")
            return
        self._publish_path(points, role)
        self._store_active_path(role, points)
        # Remember current destination for distance-gated replanning
        self._current_dest[role] = dest_loc


if __name__ == "__main__":
    try:
        SimpleMultiAgentPlanner()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleMultiAgentPlanner crashed: {e}")
        raise
