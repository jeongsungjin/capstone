#!/usr/bin/env python3

import heapq
import math
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header

from setup_carla_path import CARLA_EGG, AGENTS_ROOT

try:
    import carla
except ImportError as exc:
    rospy.logfatal(f"Failed to import CARLA package: {exc}")
    carla = None
    GlobalRoutePlanner = None
    GlobalRoutePlannerDAO = None
else:
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ImportError as exc:
        rospy.logfatal(f"Failed to import CARLA GlobalRoutePlanner: {exc}")
        GlobalRoutePlanner = None
    try:
        from agents.navigation.global_route_planner_dao import (
            GlobalRoutePlannerDAO,
        )
    except ImportError:
        GlobalRoutePlannerDAO = None


class NetworkXPathPlanner:
    """Route planner that leverages CARLA's GlobalRoutePlanner to stay on-lane."""

    def __init__(self):
        rospy.init_node("networkx_path_planner", anonymous=True)
        if carla is None or GlobalRoutePlanner is None:
            raise RuntimeError("CARLA navigation modules unavailable")

        # Parameters
        self.num_vehicles = rospy.get_param("~num_vehicles", 3)
        self.path_sampling = rospy.get_param("~path_sampling", 1.0)
        self.path_update_interval = rospy.get_param("~path_update_interval", 0.5)
        self.min_destination_distance = rospy.get_param("~min_destination_distance", 80.0)
        self.max_destination_distance = rospy.get_param("~max_destination_distance", 400.0)
        self.destination_retry_limit = rospy.get_param("~destination_retry_limit", 20)
        self.destination_reached_threshold = rospy.get_param("~destination_reached_threshold", 5.0)
        self.visualization_lifetime = rospy.get_param("~visualization_lifetime", 60.0)
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        self.global_route_resolution = rospy.get_param("~global_route_resolution", 2.0)
        self.override_preempt_min_dist = float(
            rospy.get_param("~override_preempt_min_dist", 0.0)
        )
        if self.override_preempt_min_dist < 0.0:
            rospy.logwarn("override_preempt_min_dist < 0, clamping to 0.0")
            self.override_preempt_min_dist = 0.0

        self.start_heading_deg = float(rospy.get_param("~start_heading_deg", 45.0))
        self.start_search_radius = float(rospy.get_param("~start_search_radius", 10.0))
        self.start_search_radius_max = float(
            rospy.get_param("~start_search_radius_max", 45.0)
        )
        self.start_k_candidates = int(rospy.get_param("~start_k_candidates", 10))
        self.start_offset_m = float(rospy.get_param("~start_offset_m", 2.0))
        if self.start_k_candidates <= 0:
            self.start_k_candidates = 1
        if self.start_search_radius <= 0.0:
            self.start_search_radius = 5.0
        if self.start_search_radius_max < self.start_search_radius:
            self.start_search_radius_max = self.start_search_radius

        self.replan_remaining_m = float(rospy.get_param("~replan_remaining_m", 30.0))
        self.replan_cooldown_sec = float(rospy.get_param("~replan_cooldown_sec", 0.5))
        if self.replan_remaining_m < 1.0:
            self.replan_remaining_m = 1.0
        if self.replan_cooldown_sec < 0.0:
            self.replan_cooldown_sec = 0.0

        # Connect to CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        waypoint_resolution = max(1.0, min(2.0, float(self.global_route_resolution)))
        self._waypoint_cache = [
            wp
            for wp in self.carla_map.generate_waypoints(waypoint_resolution)
            if wp.lane_type == carla.LaneType.Driving
        ]

        # Prepare global route planner
        if GlobalRoutePlannerDAO is not None:
            dao = GlobalRoutePlannerDAO(self.carla_map, self.global_route_resolution)
            self.route_planner = GlobalRoutePlanner(dao)
        else:
            self.route_planner = GlobalRoutePlanner(
                self.carla_map, self.global_route_resolution
            )
        if hasattr(self.route_planner, "setup"):
            self.route_planner.setup()

        # Destination candidates
        self.spawn_points: List = self.carla_map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        # Vehicle state
        self.vehicles: List = []
        self.vehicle_paths: Dict[str, List[Tuple[float, float]]] = {}
        self.vehicle_path_s: Dict[str, List[float]] = {}
        self.vehicle_path_lengths: Dict[str, float] = {}
        self.vehicle_destinations: Dict[str, int] = {}
        self.previous_paths: Dict[str, List[Tuple[float, float]]] = {}
        self.active_destinations: set[int] = set()
        self.override_goal: Dict[str, Optional[Tuple[float, float, rospy.Time]]] = {
            self._role_name(i): None for i in range(self.num_vehicles)
        }
        self.path_goal_signature: Dict[str, Tuple[str, Tuple[float, ...]]] = {}
        self.replan_pending: Dict[str, bool] = {
            self._role_name(i): False for i in range(self.num_vehicles)
        }
        self.last_replan_time: Dict[str, rospy.Time] = {
            self._role_name(i): rospy.Time(0) for i in range(self.num_vehicles)
        }
        self.remaining_cache: Dict[str, float] = {}
        self._planning_lock = threading.RLock()

        # Publishers
        self.path_publishers: Dict[str, rospy.Publisher] = {}
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/planned_path_{role}"
            self.path_publishers[role] = rospy.Publisher(topic, Path, queue_size=1, latch=True)

        self.override_subscribers = []
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/override_goal/{role}"
            sub = rospy.Subscriber(topic, PoseStamped, self._make_override_cb(role), queue_size=1)
            self.override_subscribers.append(sub)

        self.colors = [
            carla.Color(r=255, g=0, b=0),
            carla.Color(r=0, g=255, b=0),
            carla.Color(r=0, g=0, b=255),
            carla.Color(r=255, g=255, b=0),
            carla.Color(r=255, g=0, b=255),
            carla.Color(r=0, g=255, b=255),
        ]

        # Initial planning
        rospy.sleep(1.0)
        self._refresh_vehicles()
        self._plan_for_all()

        rospy.Timer(rospy.Duration(self.path_update_interval), self._timer_cb)
        rospy.Timer(rospy.Duration(0.1), self._progress_timer_cb)
        rospy.on_shutdown(self._cleanup)

    # ------------------------------------------------------------------
    # Core planning loop
    # ------------------------------------------------------------------
    def _timer_cb(self, _event):
        with self._planning_lock:
            self._refresh_vehicles()
            self._plan_for_all()

    def _progress_timer_cb(self, _event):
        with self._planning_lock:
            self._refresh_vehicles()
            now = rospy.Time.now()
            for index, vehicle in enumerate(self.vehicles[: self.num_vehicles]):
                role = self._role_name(index)
                path = self.vehicle_paths.get(role)
                if not path or len(path) < 2:
                    continue
                remaining = self._compute_remaining_distance(vehicle, role)
                if remaining is None:
                    continue
                self.remaining_cache[role] = remaining
                rospy.loginfo_throttle(2.0, f"{role}: remaining distance {remaining:.1f} m")
                last_time = self.last_replan_time.get(role, rospy.Time(0))
                if self.replan_pending.get(role, False):
                    continue
                if remaining > self.replan_remaining_m:
                    continue
                if (now - last_time).to_sec() < self.replan_cooldown_sec:
                    continue
                goal_signature = self.path_goal_signature.get(role)
                if goal_signature is None:
                    continue
                rospy.loginfo(
                    "%s: remaining %.1fm ≤ %.1fm, starting pre-emptive replan",
                    role,
                    remaining,
                    self.replan_remaining_m,
                )
                self.replan_pending[role] = True
                override_goal = self.override_goal.get(role)
                if override_goal is not None:
                    self._plan_override(vehicle, index, override_goal)
                else:
                    self._plan_for_vehicle(vehicle, index, reuse_destination=True)

    def _refresh_vehicles(self):
        actors = self.world.get_actors().filter("vehicle.*")
        vehicles = []
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role.startswith("ego_vehicle_"):
                vehicles.append(actor)
        vehicles.sort(key=lambda veh: veh.attributes.get("role_name", ""))
        self.vehicles = vehicles
        rospy.loginfo_throttle(5.0, f"Tracking {len(self.vehicles)} ego vehicles")

    def _plan_for_all(self):
        now = rospy.Time.now()
        for index, vehicle in enumerate(self.vehicles[: self.num_vehicles]):
            role = self._role_name(index)
            override_goal = self.override_goal.get(role)
            if override_goal is not None:
                if self._check_override_goal(vehicle, role, override_goal):
                    continue
                if not self.replan_pending.get(role, False):
                    last_time = self.last_replan_time.get(role, rospy.Time(0))
                    if (now - last_time).to_sec() < self.replan_cooldown_sec:
                        continue
                    self.replan_pending[role] = True
                    self._plan_override(vehicle, index, override_goal)
                continue

            if (
                role not in self.vehicle_paths
                and not self.replan_pending.get(role, False)
            ):
                last_time = self.last_replan_time.get(role, rospy.Time(0))
                if (now - last_time).to_sec() < self.replan_cooldown_sec:
                    continue
                self.replan_pending[role] = True
                self._plan_for_vehicle(vehicle, index)
                continue

            if self._check_destination(vehicle, role):
                if not self.replan_pending.get(role, False):
                    last_time = self.last_replan_time.get(role, rospy.Time(0))
                    if (now - last_time).to_sec() < self.replan_cooldown_sec:
                        continue
                    self.replan_pending[role] = True
                    self._plan_for_vehicle(vehicle, index)

    # ------------------------------------------------------------------
    # Per vehicle planning using GlobalRoutePlanner
    # ------------------------------------------------------------------
    def _plan_for_vehicle(self, vehicle, index: int, reuse_destination: bool = False):
        role = self._role_name(index)

        dest_index = None
        if reuse_destination:
            dest_index = self.vehicle_destinations.get(role)
            if dest_index is None or dest_index >= len(self.spawn_points):
                reuse_destination = False
        if not reuse_destination:
            self._reset_destination_tracking(role)

        front_loc, front_offset, yaw_rad = self._vehicle_front_location(vehicle)
        start_waypoint, start_meta = self.get_forward_aligned_start_node(
            front_loc,
            yaw_rad,
            self._waypoint_cache,
            self.start_k_candidates,
            self.start_heading_deg,
            self.start_search_radius,
            self.start_search_radius_max,
        )
        if start_waypoint is None:
            self._handle_plan_failure(
                role, f"{role}: could not resolve heading-aligned start waypoint"
            )
            return

        front_xy = (front_loc.x, front_loc.y)

        if reuse_destination and dest_index is not None:
            dest_loc = self.spawn_points[dest_index].location
            route = self.route_planner.trace_route(
                start_waypoint.transform.location, dest_loc
            )
            if not route or len(route) < 2:
                self._handle_plan_failure(
                    role,
                    f"{role}: failed to trace route to existing destination {dest_index}",
                )
                return
            sampled_points = self._route_to_points(route)
            if len(sampled_points) < 2:
                self._handle_plan_failure(
                    role,
                    f"{role}: existing destination route produced insufficient samples",
                )
                return
            sampled_points = self._ensure_path_starts_at_vehicle(sampled_points, front_xy)
            goal_signature = ("spawn", (float(dest_index),))
            self._commit_path(
                role,
                index,
                start_waypoint,
                start_meta,
                sampled_points,
                dest_index,
                front_offset,
                goal_signature,
            )
            return

        dest_index, route_points = self._sample_destination_route(start_waypoint)
        if dest_index is None or not route_points:
            self._handle_plan_failure(
                role, f"{role}: failed to find valid destination route"
            )
            return

        route_points = self._ensure_path_starts_at_vehicle(route_points, front_xy)
        goal_signature = ("spawn", (float(dest_index),))
        self._commit_path(
            role,
            index,
            start_waypoint,
            start_meta,
            route_points,
            dest_index,
            front_offset,
            goal_signature,
        )

    def _sample_destination_route(
        self, start_waypoint
    ) -> Tuple[Optional[int], Optional[List[Tuple[float, float]]]]:
        start_loc = start_waypoint.transform.location
        attempts = 0
        while attempts < self.destination_retry_limit:
            attempts += 1
            dest_index = random.randrange(len(self.spawn_points))
            if dest_index in self.active_destinations:
                continue

            dest_loc = self.spawn_points[dest_index].location
            euclid = start_loc.distance(dest_loc)
            if euclid < self.min_destination_distance or euclid > self.max_destination_distance:
                continue

            route = self.route_planner.trace_route(start_loc, dest_loc)
            if not route or len(route) < 2:
                continue

            route_length = self._route_length(route)
            if route_length < self.min_destination_distance:
                continue

            # Allow the planner to exceed max_distance slightly since road length > Euclid
            if route_length > self.max_destination_distance * 1.5:
                continue

            sampled_points = self._route_to_points(route)
            if len(sampled_points) < 2:
                continue

            return dest_index, sampled_points

        return None, None

    def _plan_override(
        self,
        vehicle,
        index: int,
        goal_info: Tuple[float, float, rospy.Time],
    ):
        role = self._role_name(index)
        goal_xy = (goal_info[0], goal_info[1])
        front_loc, front_offset, yaw_rad = self._vehicle_front_location(vehicle)
        start_waypoint, start_meta = self.get_forward_aligned_start_node(
            front_loc,
            yaw_rad,
            self._waypoint_cache,
            self.start_k_candidates,
            self.start_heading_deg,
            self.start_search_radius,
            self.start_search_radius_max,
        )
        if start_waypoint is None:
            self._handle_plan_failure(
                role,
                f"{role}: override planning failed to resolve heading-aligned start waypoint",
            )
            return

        goal_waypoint = self._snap_to_waypoint(goal_xy)
        if goal_waypoint is None:
            self._handle_plan_failure(
                role,
                "%s: override goal (%.1f, %.1f) is off road; ignoring"
                % (role, goal_xy[0], goal_xy[1]),
            )
            return

        start_loc = start_waypoint.transform.location
        goal_loc = goal_waypoint.transform.location
        route = self.route_planner.trace_route(start_loc, goal_loc)
        if not route or len(route) < 2:
            self._handle_plan_failure(role, f"{role}: failed to trace override route")
            return

        sampled_points = self._route_to_points(route)
        if len(sampled_points) < 2:
            self._handle_plan_failure(
                role, f"{role}: override route produced insufficient samples"
            )
            return

        sampled_points = self._ensure_path_starts_at_vehicle(
            sampled_points, (front_loc.x, front_loc.y)
        )

        goal_signature = ("override", (goal_xy[0], goal_xy[1]))
        if self._commit_path(
            role,
            index,
            start_waypoint,
            start_meta,
            sampled_points,
            None,
            front_offset,
            goal_signature,
        ):
            rospy.loginfo(
                "override path published for %s (%d poses)",
                role,
                len(sampled_points),
            )

    # ------------------------------------------------------------------
    # Destination monitoring
    # ------------------------------------------------------------------
    def _check_destination(self, vehicle, role: str) -> bool:
        dest_index = self.vehicle_destinations.get(role)
        if dest_index is None or dest_index >= len(self.spawn_points):
            return False

        remaining = self._compute_remaining_distance(vehicle, role)
        if remaining is None:
            return False

        if remaining <= self.destination_reached_threshold:
            rospy.loginfo(
                f"{role}: destination reached (remaining {remaining:.2f} m)"
            )
            self._clear_path(role)
            self.vehicle_paths.pop(role, None)
            self.vehicle_path_s.pop(role, None)
            self.vehicle_path_lengths.pop(role, None)
            self.remaining_cache.pop(role, None)
            self.path_goal_signature.pop(role, None)
            self._reset_destination_tracking(role)
            return True
        return False

    def _check_override_goal(
        self,
        vehicle,
        role: str,
        goal_info: Tuple[float, float, rospy.Time],
    ) -> bool:
        goal_xy = (goal_info[0], goal_info[1])
        current_location = vehicle.get_location()
        distance = math.hypot(
            current_location.x - goal_xy[0], current_location.y - goal_xy[1]
        )
        threshold = max(
            self.destination_reached_threshold, self.override_preempt_min_dist
        )
        if distance <= threshold:
            rospy.loginfo("override goal reached for %s, clearing override", role)
            self._clear_path(role)
            self.vehicle_paths.pop(role, None)
            self.vehicle_path_s.pop(role, None)
            self.vehicle_path_lengths.pop(role, None)
            self.remaining_cache.pop(role, None)
            self.path_goal_signature.pop(role, None)
            self.override_goal[role] = None
            return True
        return False

    # ------------------------------------------------------------------
    # ROS / Visualization helpers
    # ------------------------------------------------------------------
    def _publish_path(self, path_points: List[Tuple[float, float]], role: str):
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

    def _draw_path(self, path_points: List[Tuple[float, float]], index: int):
        if not self.enable_visualization or len(path_points) < 2:
            return
        role = self._role_name(index)
        self._clear_path(role)
        color = self.colors[index % len(self.colors)]
        for i in range(len(path_points) - 1):
            p1 = carla.Location(x=path_points[i][0], y=path_points[i][1], z=0.5)
            p2 = carla.Location(x=path_points[i + 1][0], y=path_points[i + 1][1], z=0.5)
            self.world.debug.draw_line(
                p1,
                p2,
                thickness=0.3,
                color=color,
                life_time=self.visualization_lifetime,
            )
        self.previous_paths[role] = path_points[:]

    def _clear_path(self, role: str):
        if role not in self.previous_paths:
            return
        old_path = self.previous_paths.pop(role)
        clear_color = carla.Color(r=0, g=0, b=0)
        for i in range(len(old_path) - 1):
            p1 = carla.Location(x=old_path[i][0], y=old_path[i][1], z=0.5)
            p2 = carla.Location(x=old_path[i + 1][0], y=old_path[i + 1][1], z=0.5)
            self.world.debug.draw_line(
                p1,
                p2,
                thickness=0.3,
                color=clear_color,
                life_time=0.1,
            )

    def _cleanup(self):
        for role in list(self.previous_paths.keys()):
            self._clear_path(role)
        self.previous_paths.clear()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _make_override_cb(self, role: str):
        def _callback(msg: PoseStamped):
            self._handle_override(role, msg)

        return _callback

    def _handle_override(self, role: str, msg: PoseStamped):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        goal_info = (msg.pose.position.x, msg.pose.position.y, stamp)
        with self._planning_lock:
            if role not in self.override_goal:
                self.override_goal[role] = None
            self.override_goal[role] = goal_info
            self._reset_destination_tracking(role)
            self.vehicle_paths.pop(role, None)
            self.vehicle_path_s.pop(role, None)
            self.vehicle_path_lengths.pop(role, None)
            self.remaining_cache.pop(role, None)
            self.path_goal_signature.pop(role, None)
            self._clear_path(role)
            rospy.loginfo(
                "override set for %s: (%.1f, %.1f)", role, goal_info[0], goal_info[1]
            )

            self._refresh_vehicles()
            self.replan_pending[role] = True
            for index, vehicle in enumerate(self.vehicles[: self.num_vehicles]):
                if self._role_name(index) == role:
                    self._plan_override(vehicle, index, goal_info)
                    break

    def _reset_destination_tracking(self, role: str):
        previous_dest = self.vehicle_destinations.pop(role, None)
        if previous_dest is not None:
            self.active_destinations.discard(previous_dest)

    def _compute_remaining_distance(self, vehicle, role: str) -> Optional[float]:
        path = self.vehicle_paths.get(role)
        s_profile = self.vehicle_path_s.get(role)
        total_length = self.vehicle_path_lengths.get(role, 0.0)
        if not path or not s_profile or total_length <= 0.0 or len(path) < 2:
            return None

        front_loc, _, _ = self._vehicle_front_location(vehicle)
        px = front_loc.x
        py = front_loc.y

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

        segment_length = math.hypot(
            path[best_index + 1][0] - path[best_index][0],
            path[best_index + 1][1] - path[best_index][1],
        )
        if segment_length < 1e-6:
            s_now = s_profile[best_index]
        else:
            s_now = s_profile[best_index] + best_t * segment_length
        remaining = max(0.0, total_length - s_now)
        return remaining

    def _vehicle_front_location(self, vehicle) -> Tuple[carla.Location, float, float]:
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
        return front_loc, offset, yaw_rad

    def _store_path_geometry(self, role: str, path_points: List[Tuple[float, float]]):
        if not path_points:
            self.vehicle_path_s.pop(role, None)
            self.vehicle_path_lengths[role] = 0.0
            return
        cumulative: List[float] = [0.0]
        total = 0.0
        for idx in range(1, len(path_points)):
            step = math.hypot(
                path_points[idx][0] - path_points[idx - 1][0],
                path_points[idx][1] - path_points[idx - 1][1],
            )
            total += step
            cumulative.append(total)
        self.vehicle_path_s[role] = cumulative
        self.vehicle_path_lengths[role] = total

    def _destination_reached(self, vehicle, role: str) -> bool:
        remaining = self._compute_remaining_distance(vehicle, role)
        if remaining is None:
            return False
        if remaining <= self.destination_reached_threshold:
            rospy.loginfo(
                "%s: destination arc-length %.2fm ≤ threshold %.2fm",
                role,
                remaining,
                self.destination_reached_threshold,
            )
            return True
        return False

    def _handle_plan_failure(self, role: str, reason: str):
        rospy.logwarn(reason)
        self.replan_pending[role] = False
        self.last_replan_time[role] = rospy.Time.now()

    def _commit_path(
        self,
        role: str,
        index: int,
        start_waypoint,
        start_meta: Optional[Dict[str, Any]],
        path_points: List[Tuple[float, float]],
        dest_index: Optional[int],
        front_offset: float,
        goal_signature: Tuple[str, Tuple[float, ...]],
    ) -> bool:
        if len(path_points) < 2:
            self._handle_plan_failure(role, f"{role}: planned path invalid (len={len(path_points)})")
            return False

        self.vehicle_paths[role] = path_points
        self._store_path_geometry(role, path_points)
        self.remaining_cache[role] = self.vehicle_path_lengths.get(role, 0.0)

        if dest_index is not None:
            self.vehicle_destinations[role] = dest_index
            self.active_destinations.add(dest_index)

        self.path_goal_signature[role] = goal_signature
        self.replan_pending[role] = False
        self.last_replan_time[role] = rospy.Time.now()

        meta = start_meta.copy() if start_meta else {}
        meta["front_offset"] = front_offset
        waypoint_id = getattr(start_waypoint, "id", None)
        rospy.loginfo(
            "%s: start waypoint id=%s road=%d lane=%d source=%s front_offset=%.2fm",
            role,
            waypoint_id if waypoint_id is not None else "n/a",
            start_waypoint.road_id,
            start_waypoint.lane_id,
            meta.get("source", "unknown"),
            front_offset,
        )

        path_len = self._path_length(path_points)
        rospy.loginfo(
            "%s: planned CARLA route with %d samples (%.1fm)",
            role,
            len(path_points),
            path_len,
        )

        # self._draw_path(path_points, index)
        self._publish_path(path_points, role)
        rospy.loginfo("%s: new path swapped (%d poses)", role, len(path_points))
        return True

    def get_forward_aligned_start_node(
        self,
        position: carla.Location,
        yaw_rad: float,
        waypoint_candidates: List[carla.Waypoint],
        k: int,
        heading_deg: float,
        radius_primary: float,
        radius_secondary: float,
    ) -> Tuple[Optional[carla.Waypoint], Dict[str, Any]]:
        if not waypoint_candidates:
            return None, {"source": "no_waypoints"}

        forward = (math.cos(yaw_rad), math.sin(yaw_rad))
        primary_angle = math.radians(max(0.0, min(heading_deg, 85.0)))
        secondary_angle = math.radians(75.0)
        primary_radius = max(1.0, radius_primary)
        secondary_radius = max(primary_radius, radius_secondary)
        candidate_count = max(1, k)

        nearest_candidates = self._nearest_waypoints(
            position, candidate_count * 3
        )

        def _select(
            candidates: List[carla.Waypoint],
            max_radius: float,
            max_angle_rad: float,
            label: str,
        ) -> Optional[Tuple[carla.Waypoint, Dict[str, Any]]]:
            best_wp = None
            best_dist = float("inf")
            best_angle = 0.0
            best_lane_delta = 0.0
            for wp in candidates:
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
                lane_delta = abs(self._normalize_angle(lane_yaw - yaw_rad))
                if lane_delta > max_angle_rad + math.radians(5.0):
                    continue
                if dist < best_dist:
                    best_wp = wp
                    best_dist = dist
                    best_angle = heading_angle
                    best_lane_delta = lane_delta
            if best_wp is None:
                return None
            metadata = {
                "source": label,
                "distance": best_dist,
                "angle": math.degrees(best_angle),
                "lane_delta": math.degrees(best_lane_delta),
            }
            return best_wp, metadata

        primary_candidate = _select(nearest_candidates, primary_radius, primary_angle, "primary")
        if primary_candidate is not None:
            return primary_candidate

        secondary_candidate = _select(
            nearest_candidates, secondary_radius, secondary_angle, "expanded"
        )
        if secondary_candidate is not None:
            return secondary_candidate

        fallback_wp = self.carla_map.get_waypoint(
            position,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
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
                    lane_delta = abs(self._normalize_angle(lane_yaw - yaw_rad))
                    if lane_delta > math.radians(95.0):
                        continue
                    meta = {
                        "source": f"fallback_next_{ds:.1f}",
                        "distance": dist,
                        "lane_delta": math.degrees(lane_delta),
                    }
                    return candidate, meta
            lane_yaw = math.radians(fallback_wp.transform.rotation.yaw)
            lane_delta = abs(self._normalize_angle(lane_yaw - yaw_rad))
            meta = {
                "source": "fallback_base",
                "distance": 0.0,
                "lane_delta": math.degrees(lane_delta),
            }
            return fallback_wp, meta

        return None, {"source": "no_match"}

    def _ensure_path_starts_at_vehicle(
        self, path_points: List[Tuple[float, float]], current_xy: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        if not path_points:
            return []
        if self._distance(current_xy, path_points[0]) < 0.1:
            path_points[0] = current_xy
            return path_points
        return [current_xy] + path_points

    def _nearest_waypoints(
        self, position: carla.Location, max_candidates: int
    ) -> List[carla.Waypoint]:
        if max_candidates <= 0 or not self._waypoint_cache:
            return []

        def _distance_sq(wp: carla.Waypoint) -> float:
            loc = wp.transform.location
            dx = loc.x - position.x
            dy = loc.y - position.y
            return dx * dx + dy * dy

        return heapq.nsmallest(max_candidates, self._waypoint_cache, key=_distance_sq)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _snap_to_waypoint(self, point_xy: Tuple[float, float]):
        location = carla.Location(x=point_xy[0], y=point_xy[1], z=0.5)
        return self.carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    def _route_to_points(
        self, route: List[Tuple]
    ) -> List[Tuple[float, float]]:
        waypoints = [item[0] for item in route]
        if not waypoints:
            return []
        points: List[Tuple[float, float]] = []
        last_point: Optional[Tuple[float, float]] = None
        for wp in waypoints:
            loc = wp.transform.location
            current = (loc.x, loc.y)
            if last_point is None:
                points.append(current)
                last_point = current
                continue
            segment_length = self._distance(current, last_point)
            if segment_length < 1e-3:
                continue
            steps = max(1, int(segment_length // self.path_sampling))
            for step in range(1, steps + 1):
                ratio = min(1.0, (step * self.path_sampling) / segment_length)
                interp = (
                    last_point[0] + (current[0] - last_point[0]) * ratio,
                    last_point[1] + (current[1] - last_point[1]) * ratio,
                )
                if not points or self._distance(interp, points[-1]) > 0.05:
                    points.append(interp)
            if self._distance(current, points[-1]) > 0.05:
                points.append(current)
            last_point = current
        return points

    @staticmethod
    def _route_length(route: List[Tuple]) -> float:
        total = 0.0
        previous = None
        for waypoint, _ in route:
            loc = waypoint.transform.location
            if previous is not None:
                total += loc.distance(previous)
            previous = loc
        return total

    @staticmethod
    def _distance(pt_a: Tuple[float, float], pt_b: Tuple[float, float]) -> float:
        return math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])

    @staticmethod
    def _path_length(points: List[Tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(points) - 1):
            total += math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        return total

    @staticmethod
    def _role_name(index: int) -> str:
        return f"ego_vehicle_{index + 1}"


if __name__ == "__main__":
    try:
        planner = NetworkXPathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
