#!/usr/bin/env python3
import math
import random
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from std_msgs.msg import String

try:
    from capstone_msgs.msg import Uplink  # type: ignore
except Exception:
    Uplink = None  # type: ignore

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
        self.path_thin_min_m = float(rospy.get_param("~path_thin_min_m", 0.1))            # default denser than 0.2
        self.replan_soft_distance_m = float(rospy.get_param("~replan_soft_distance_m", 60.0))
        self.replan_check_interval = float(rospy.get_param("~replan_check_interval", 1.0))
        # Align first path segment with vehicle heading by looking slightly ahead when replanning
        self.heading_align_lookahead_m = float(rospy.get_param("~heading_align_lookahead_m", 2.5))
        # Start waypoint/path stitch parameters
        self.start_join_max_gap_m = float(rospy.get_param("~start_join_max_gap_m", 12.0))
        self.start_offset_m = float(rospy.get_param("~start_offset_m", 3.0))
        self.path_extension_overlap_m = float(rospy.get_param("~path_extension_overlap_m", 30.0))
        self.max_extend_attempts = int(rospy.get_param("~max_extend_attempts", 3))
        self.min_destination_distance = float(rospy.get_param("~min_destination_distance", 70.0))
        self.max_destination_distance = float(rospy.get_param("~max_destination_distance", 100.0))
        # Low-voltage handling (forced destination / parking)
        self.low_voltage_threshold = float(rospy.get_param("~low_voltage_threshold", 5.0))
        self.low_voltage_dest_x = float(rospy.get_param("~low_voltage_dest_x", -12.5))
        self.low_voltage_dest_y = float(rospy.get_param("~low_voltage_dest_y", -16.5))
        self.parking_dest_x = float(rospy.get_param("~parking_dest_x", -27.0))
        self.parking_dest_y = float(rospy.get_param("~parking_dest_y", -16.5))
        self.parking_trigger_distance_m = float(rospy.get_param("~parking_trigger_distance_m", 0.5))

        # CARLA world/map
        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        timeout = float(rospy.get_param("~carla_timeout", 10.0))
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        # Route planner (prefer CARLA agents; fallback to simple lane-follow)
        self.route_planner = None
        if GlobalRoutePlanner is None:
            raise RuntimeError("CARLA GlobalRoutePlanner module unavailable")
        try:
            self.route_planner = GlobalRoutePlanner(self.carla_map, self.global_route_resolution)
            if hasattr(self.route_planner, "setup"):
                self.route_planner.setup()
            rospy.loginfo("SimpleMultiAgentPlanner: using CARLA GlobalRoutePlanner")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize GlobalRoutePlanner: {exc}")

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
        self._voltage: Dict[int, float] = {}
        # Low-voltage 단계 관리: idle -> to_buffer(low_voltage_dest) -> to_parking(parking_dest) -> parked
        self._lv_stage: Dict[str, str] = {self._role_name(i): "idle" for i in range(self.num_vehicles)}
        # 수동 목적지 오버라이드 (PoseStamped에서 받음)
        self._override_goal: Dict[str, Optional[carla.Location]] = {self._role_name(i): None for i in range(self.num_vehicles)}
        self._override_active: Dict[str, bool] = {self._role_name(i): False for i in range(self.num_vehicles)}
        self.override_clear_radius = float(rospy.get_param("~override_clear_radius", 3.0))
        self.override_hold_sec = float(rospy.get_param("~override_hold_sec", 5.0))
        self._override_hold_until: Dict[str, float] = {self._role_name(i): 0.0 for i in range(self.num_vehicles)}

        # Uplink subscriber (voltage)
        self.uplink_topic = str(rospy.get_param("~uplink_topic", "/uplink"))
        if Uplink is not None and self.uplink_topic:
            rospy.Subscriber(self.uplink_topic, Uplink, self._uplink_cb, queue_size=10)

        # Override goal subscribers per vehicle
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/override_goal/{role}"
            rospy.Subscriber(topic, PoseStamped, self._override_goal_cb, callback_args=role, queue_size=1)

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
            # hold 중이면 스킵 (정지 유지)
            hold_until = self._override_hold_until.get(role, 0.0)
            if hold_until > 0.0:
                if rospy.Time.now().to_sec() < hold_until:
                    continue
                else:
                    self._override_hold_until[role] = 0.0
                    # hold 종료 후 기존 path 제거 → 재계획 유도
                    self._active_paths.pop(role, None)
                    self._active_path_s.pop(role, None)
                    self._active_path_len.pop(role, None)
            # override goal가 있고 도착했다면 해제
            if self._override_goal.get(role) is not None:
                if front_loc.distance(self._override_goal[role]) <= max(0.5, float(self.override_clear_radius)):
                    self._override_goal[role] = None
                    self._override_active[role] = False
                    hold_t = rospy.Time.now().to_sec() + max(0.0, float(self.override_hold_sec))
                    self._override_hold_until[role] = hold_t
                    # 정지용 path 퍼블리시(1포인트만 넣어 컨트롤러 속도 0 유도)
                    stop_pts = [(front_loc.x, front_loc.y)]
                    self._publish_path(stop_pts, role)
                    self._store_active_path(role, stop_pts)
                    rospy.loginfo(f"{role}: override goal reached -> hold for {self.override_hold_sec:.1f}s")
                    continue
            dest_override = None
            if self._override_active.get(role, False) and self._override_goal.get(role) is not None:
                dest_override = self._override_goal[role]
            # override가 활성인 동안에는 일반/저전압 재계획을 막고, 오버라이드 경로만 유지
            if dest_override is not None:
                current_path = self._active_paths.get(role)
                # 이미 목적지가 path 끝에 충분히 가까우면 재계획하지 않음
                if current_path and len(current_path) >= 2:
                    last = current_path[-1]
                    if math.hypot(last[0] - dest_override.x, last[1] - dest_override.y) <= max(1.0, self.start_join_max_gap_m):
                        continue
                    # 끝점이 멀면, 기존 경로의 마지막 점을 제외한 prefix로 한번만 append
                    prefix = current_path[:-1] if len(current_path) > 1 else current_path
                    if not self._plan_for_role(vehicle, role, front_loc, prefix_points=prefix, dest_override=dest_override, force_direct=False):
                        rospy.logwarn_throttle(5.0, f"{role}: failed to append override path")
                    continue
                else:
                    # 경로가 없으면 현재 위치 기준으로 계획
                    if not self._plan_for_role(vehicle, role, front_loc, dest_override=dest_override, force_direct=False):
                        rospy.logwarn_throttle(5.0, f"{role}: failed to plan override path")
                    continue

            # override가 없을 때만 저전압/일반 로직 수행
            stage = self._lv_stage.get(role, "idle")
            is_low = self._is_low_voltage(role)
            if is_low:
                # Distance to buffer and parking targets (front of vehicle)
                dist_buffer = math.hypot(front_loc.x - float(self.low_voltage_dest_x), front_loc.y - float(self.low_voltage_dest_y))
                dist_parking = math.hypot(front_loc.x - float(self.parking_dest_x), front_loc.y - float(self.parking_dest_y))
                # 1) idle -> buffer dest (low_voltage_dest)
                if stage == "idle":
                    dest_override = carla.Location(x=float(self.low_voltage_dest_x), y=float(self.low_voltage_dest_y), z=front_loc.z)
                    if self._plan_for_role(vehicle, role, front_loc, dest_override=dest_override, force_direct=False):
                        rospy.loginfo(f"{role}: LOW-V start -> buffer ({self.low_voltage_dest_x:.2f},{self.low_voltage_dest_y:.2f})")
                        self._lv_stage[role] = "to_buffer"
                    else:
                        rospy.logwarn_throttle(5.0, f"{role}: low-voltage buffer plan failed")
                    continue
                # 2) heading to buffer: 모니터링하다가 여유거리 도달 시 최종 주차 목적지로 재계획
                if stage == "to_buffer":
                    remaining = self._remaining_path_distance(role, front_loc)
                    # Use either remaining arc-length or actual distance-to-target to trigger parking leg
                    trigger_dist = max(0.0, float(self.parking_trigger_distance_m))
                    if remaining is None:
                        dest_override = carla.Location(x=float(self.low_voltage_dest_x), y=float(self.low_voltage_dest_y), z=front_loc.z)
                        self._plan_for_role(vehicle, role, front_loc, dest_override=dest_override, force_direct=False)
                    elif remaining <= trigger_dist or dist_buffer <= trigger_dist:
                        dest_override = carla.Location(x=float(self.parking_dest_x), y=float(self.parking_dest_y), z=front_loc.z)
                        if self._plan_for_role(vehicle, role, front_loc, dest_override=dest_override, force_direct=True):
                            rospy.loginfo(f"{role}: LOW-V buffer reached (remaining {remaining:.1f} m, dist {dist_buffer:.1f} m) -> parking ({self.parking_dest_x:.2f},{self.parking_dest_y:.2f})")
                            self._lv_stage[role] = "to_parking"
                        else:
                            rospy.logwarn_throttle(5.0, f"{role}: parking plan failed after buffer")
                    continue
                # 3) heading to final parking: 도착 감지, 필요 시 복구
                if stage == "to_parking":
                    remaining = self._remaining_path_distance(role, front_loc)
                    if remaining is None:
                        dest_override = carla.Location(x=float(self.parking_dest_x), y=float(self.parking_dest_y), z=front_loc.z)
                        self._plan_for_role(vehicle, role, front_loc, dest_override=dest_override, force_direct=True)
                    elif remaining <= 1.0 or dist_parking <= 1.5:
                        rospy.loginfo(f"{role}: LOW-V parking reached (remaining {remaining:.1f} m, dist {dist_parking:.1f} m) -> parked")
                        self._lv_stage[role] = "parked"
                    continue
                # 4) parked: 더 이상 계획 안 함
                if stage == "parked":
                    continue
            current_path = self._active_paths.get(role)
            if not current_path or len(current_path) < 2:
                if not self._plan_for_role(vehicle, role, front_loc, dest_override=dest_override):
                    rospy.logwarn_throttle(5.0, f"{role}: failed to plan initial path")
                continue
            remaining = self._remaining_path_distance(role, front_loc)
            if remaining is None:
                if not self._plan_for_role(vehicle, role, front_loc, dest_override=dest_override):
                    rospy.logwarn_throttle(5.0, f"{role}: failed to replan after progress loss")
                continue
            if remaining <= max(0.0, float(self.replan_soft_distance_m)):
                rospy.loginfo(f"{role}: remaining {remaining:.1f} m <= soft {self.replan_soft_distance_m:.1f} m -> path extension")
                if not self._extend_path(vehicle, role, front_loc, dest_override=dest_override):
                    rospy.logwarn_throttle(5.0, f"{role}: path extension failed; forcing fresh plan")
                    if self._plan_for_role(vehicle, role, front_loc, dest_override=dest_override):
                        pass

    def _extend_path(self, vehicle, role: str, front_loc: carla.Location, dest_override: Optional[carla.Location] = None) -> bool:
        current = self._active_paths.get(role)
        if not current or len(current) < 2:
            rospy.logwarn_throttle(5.0, f"{role}: no active path to extend; full replan")
            return self._plan_for_role(vehicle, role, front_loc, dest_override=dest_override)
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
        attempts = 0
        while attempts < max(1, int(self.max_extend_attempts)):
            prefix_copy = list(suffix)
            if len(prefix_copy) < 2:
                rospy.logwarn_throttle(5.0, f"{role}: prefix too short during extension; replanning")
                return self._plan_for_role(vehicle, role, front_loc)
            remaining = self._remaining_path_distance(role, front_loc)
            if remaining is not None:
                rospy.loginfo(f"{role}: extending path (remaining {remaining:.1f} m, overlap {self.path_extension_overlap_m} m, attempt {attempts + 1})")
            if self._plan_for_role(vehicle, role, front_loc, prefix_points=prefix_copy):
                return True
            attempts += 1
            rospy.logwarn_throttle(5.0, f"{role}: path extension attempt {attempts} failed; retrying")
        rospy.logwarn_throttle(5.0, f"{role}: all extension attempts failed; falling back to fresh plan")
        return self._plan_for_role(vehicle, role, front_loc)

    def _get_ego_vehicles(self) -> List[carla.Actor]:
        actors = self.world.get_actors().filter("vehicle.*")
        vehicles: List[carla.Actor] = []
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role.startswith("ego_vehicle_"):
                vehicles.append(actor)
        vehicles.sort(key=lambda v: v.attributes.get("role_name", ""))
        return vehicles

    def _uplink_cb(self, msg: "Uplink") -> None:
        try:
            self._voltage[int(msg.vehicle_id)] = float(msg.voltage)
        except Exception:
            pass

    def _is_low_voltage(self, role: str) -> bool:
        try:
            vid = int(role.split("_")[-1])
            v = self._voltage.get(vid, float("inf"))
            return v <= float(self.low_voltage_threshold)
        except Exception:
            return False

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
        try:
            return self.route_planner.trace_route(start, dest)
        except Exception as exc:
            rospy.logwarn(f"trace_route failed: {exc}")
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

    def _straight_line_points(self, start_loc: carla.Location, dest_loc: carla.Location, spacing: float = 1.0) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        dx = float(dest_loc.x) - float(start_loc.x)
        dy = float(dest_loc.y) - float(start_loc.y)
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return [(float(start_loc.x), float(start_loc.y)), (float(dest_loc.x), float(dest_loc.y))]
        steps = max(1, int(dist / max(0.1, spacing)))
        for i in range(steps + 1):
            t = float(i) / float(steps)
            x = float(start_loc.x) + dx * t
            y = float(start_loc.y) + dy * t
            points.append((x, y))
        return points

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
            rospy.logwarn_throttle(2.0, "SimpleMultiAgentPlanner: unable to project progress (disjoint path)")
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
            rospy.logwarn_throttle(2.0, f"{role}: progress projection failed; forcing replan fallback")
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

    def _plan_for_role(self, vehicle, role: str, front_loc: carla.Location, prefix_points: Optional[List[Tuple[float, float]]] = None, dest_override: Optional[carla.Location] = None, force_direct: bool = False) -> bool:
        # Sample (or resample) destination and publish a fresh path
        dest_loc = dest_override if dest_override is not None else self._choose_destination(front_loc)
        if dest_loc is None:
            rospy.logwarn_throttle(5.0, f"{role}: destination not found within distance bounds")
            return False
        # Detect off-road override target (e.g., parking off the lane)
        offroad_override = False
        if dest_override is not None:
            try:
                wp_test = self.carla_map.get_waypoint(dest_loc, project_to_road=False)
                if wp_test is None:
                    offroad_override = True
            except Exception:
                offroad_override = True
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
        if not force_direct:
            while attempts < max_attempts:
                route = self._trace_route(start_loc, dest_loc)
                if route and len(route) >= 2:
                    break
                dest_loc = self._choose_destination(front_loc)
                if dest_loc is None:
                    route = None
                    break
                attempts += 1
        new_points: List[Tuple[float, float]] = []
        if not force_direct and route and len(route) >= 2:
            new_points = self._route_to_points(route)
        else:
            # Always use direct line when dest_override is present (e.g., off-road parking), or when GRP fails
            new_points = self._straight_line_points(start_loc, dest_loc, spacing=max(0.5, float(self.path_thin_min_m)))
            rospy.logwarn_throttle(2.0, f"{role}: using direct line to dest ({dest_loc.x:.2f},{dest_loc.y:.2f}) (force_direct={force_direct}, route_ok={route is not None and len(route)>=2})")
        if prefix_points is not None and len(prefix_points) >= 1:
            combined = prefix_points[:-1] + new_points
            points = combined if (offroad_override or force_direct) else self._snap_points_to_lane(combined)
        else:
            points = self._ensure_path_starts_at_vehicle(new_points, (front_loc.x, front_loc.y))
            points = points if (offroad_override or force_direct) else self._snap_points_to_lane(points)
        if len(points) < 2:
            rospy.logwarn_throttle(5.0, f"{role}: insufficient path points")
            return False
        rospy.logdebug(f"{role}: publishing path with {len(points)} points (prefix={'yes' if prefix_points else 'no'})")
        if offroad_override or force_direct:
            rospy.logwarn_throttle(2.0, f"{role}: off-road/direct destination; path not snapped to lane (dest=({dest_loc.x:.2f},{dest_loc.y:.2f}))")
        self._publish_path(points, role)
        self._store_active_path(role, points)
        # Remember current destination for distance-gated replanning
        self._current_dest[role] = dest_loc
        return True

    def _override_goal_cb(self, msg: PoseStamped, role: str) -> None:
        try:
            loc = carla.Location(
                x=float(msg.pose.position.x),
                y=float(msg.pose.position.y),
                z=float(msg.pose.position.z),
            )
            self._override_goal[role] = loc
            self._override_active[role] = True
            rospy.loginfo(f"{role}: override goal set to ({loc.x:.2f},{loc.y:.2f})")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, f"{role}: failed to set override goal: {exc}")

if __name__ == "__main__":
    try:
        SimpleMultiAgentPlanner()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleMultiAgentPlanner crashed: {e}")
        raise
