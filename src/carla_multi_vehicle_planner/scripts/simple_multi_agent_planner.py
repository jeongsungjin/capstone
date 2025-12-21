#!/usr/bin/env python3
import math
import random
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import Header

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
    from agents.navigation.global_route_planner import GlobalRoutePlanner as CARLAGlobalRoutePlanner  # type: ignore
except Exception:
    CARLAGlobalRoutePlanner = None  # type: ignore

try:
    from global_planner import GlobalPlanner
except Exception:
    GlobalPlanner = None  # type: ignore

try:
    from obstacle_planner import ObstaclePlanner
except Exception:
    ObstaclePlanner = None  # type: ignore


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
        self.parking_dest_x = float(rospy.get_param("~parking_dest_x", -25.0))
        self.parking_dest_y = float(rospy.get_param("~parking_dest_y", -16.5))
        self.parking_trigger_distance_m = float(rospy.get_param("~parking_trigger_distance_m", 0.5))
        
        # Obstacle avoidance parameters
        self.obstacle_radius = float(rospy.get_param("~obstacle_radius", 5.0))
        self.obstacle_replan_threshold = float(rospy.get_param("~obstacle_replan_threshold", 30.0))

        # CARLA world/map
        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        timeout = float(rospy.get_param("~carla_timeout", 10.0))
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        # Route planner (GlobalPlanner 우선, 없으면 CARLA GlobalRoutePlanner)
        self.route_planner = None
        self._obstacle_planner = None
        
        if GlobalPlanner is not None:
            try:
                self.route_planner = GlobalPlanner(self.carla_map, self.global_route_resolution)
                rospy.loginfo("SimpleMultiAgentPlanner: using GlobalPlanner")
                
                # ObstaclePlanner 초기화
                if ObstaclePlanner is not None:
                    self._obstacle_planner = ObstaclePlanner(self.route_planner)
                    rospy.loginfo("SimpleMultiAgentPlanner: ObstaclePlanner initialized")
            except Exception as exc:
                rospy.logwarn(f"Failed to init GlobalPlanner: {exc}, falling back to CARLAGlobalRoutePlanner")
        
        if self.route_planner is None:
            if CARLAGlobalRoutePlanner is None:
                raise RuntimeError("CARLA GlobalRoutePlanner module unavailable")
            try:
                self.route_planner = CARLAGlobalRoutePlanner(self.carla_map, self.global_route_resolution)
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
        
        # 장애물로 인해 정지 중인 차량 추적 (role -> obstacle_pos)
        self._obstacle_blocked_roles: Dict[str, Tuple[float, float]] = {}
        # Low-voltage 단계 관리: idle -> to_buffer(low_voltage_dest) -> to_parking(parking_dest) -> parked
        self._lv_stage: Dict[str, str] = {self._role_name(i): "idle" for i in range(self.num_vehicles)}

        # Uplink subscriber
        if Uplink is not None:
            rospy.Subscriber("/uplink", Uplink, self._uplink_cb, queue_size=10)
        
        # Obstacle subscriber
        self._obstacles: List[Tuple[float, float, float]] = []
        rospy.Subscriber("/obstacles", PoseArray, self._obstacle_cb, queue_size=1)
        
        # Vehicle edges tracking for overlap detection
        self._vehicle_route_edges: Dict[str, List[Tuple[int, int]]] = {}

        rospy.sleep(0.5)
        self._plan_once()
        rospy.Timer(rospy.Duration(self.replan_check_interval), self._replan_check_cb)
        
        # Overlap detection logging timer (for debugging)
        self._overlap_log_interval = float(rospy.get_param("~overlap_log_interval", 5.0))
        rospy.Timer(rospy.Duration(self._overlap_log_interval), self._log_overlap_detection_cb)

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    # ─────────────────────────────────────────────────────────────────────────
    # Obstacle Handling
    # ─────────────────────────────────────────────────────────────────────────

    def _obstacle_cb(self, msg: PoseArray) -> None:
        """장애물 토픽 콜백 - 장애물 변화 시 노드 차단 업데이트"""
        new_obstacles = []
        for pose in msg.poses:
            new_obstacles.append((pose.position.x, pose.position.y, pose.position.z))
        
        # 장애물 변화 감지
        if self._obstacles_changed(new_obstacles):
            rospy.loginfo(f"Obstacles changed: {len(self._obstacles)} -> {len(new_obstacles)}")
            self._obstacles = new_obstacles
            self._update_blocked_nodes()
            # 필요 시 재계획 트리거
            self._on_obstacle_change()

    def _obstacles_changed(self, new_obstacles: List[Tuple[float, float, float]]) -> bool:
        """장애물 목록 변화 여부 확인"""
        if len(new_obstacles) != len(self._obstacles):
            return True
        for new, old in zip(sorted(new_obstacles), sorted(self._obstacles)):
            if math.hypot(new[0] - old[0], new[1] - old[1]) > 1.0:
                return True
        return False

    def _update_blocked_nodes(self) -> None:
        """장애물 위치로 차단 노드/엣지 업데이트"""
        if not hasattr(self.route_planner, 'clear_obstacles'):
            rospy.logwarn("route_planner does not have clear_obstacles method")
            return
        self.route_planner.clear_obstacles()
        for ox, oy, oz in self._obstacles:
            loc = carla.Location(x=ox, y=oy, z=oz)
            # 노드뿐만 아니라 엣지도 차단
            blocked = self.route_planner.add_obstacle_on_road(loc, self.obstacle_radius)
            edge = self.route_planner._localize(loc)
            rospy.loginfo(f"Obstacle at ({ox:.1f}, {oy:.1f}): blocked {blocked} nodes, edge={edge}")

    def _on_obstacle_change(self) -> None:
        """
        장애물 변화 시 호출되는 콜백
        - 영향받는 차량의 경로를 장애물 5m 전까지 자름
        - 장애물 해제 시 replan 트리거
        """
        vehicles = self._get_ego_vehicles()
        stop_distance = 5.0  # 장애물 앞 정지 거리
        
        if not self._obstacles:
            # 장애물 없음 → 정지 중인 차량들 replan
            if self._obstacle_blocked_roles:
                rospy.loginfo("Obstacle removed - replanning blocked vehicles")
                for role in list(self._obstacle_blocked_roles.keys()):
                    self._obstacle_blocked_roles.pop(role, None)
                    # 해당 차량 replan
                    for index, vehicle in enumerate(vehicles[:self.num_vehicles]):
                        if self._role_name(index) == role:
                            front_loc = self._vehicle_front(vehicle)
                            if self._plan_for_role(vehicle, role, front_loc):
                                rospy.loginfo(f"{role}: replanned after obstacle removal")
                            break
            return
        
        for index, vehicle in enumerate(vehicles[:self.num_vehicles]):
            role = self._role_name(index)
            try:
                front_loc = self._vehicle_front(vehicle)
            except Exception:
                continue
            
            # 해당 차량의 경로 가져오기
            path = self._active_paths.get(role)
            if not path or len(path) < 2:
                continue
            
            # 경로상 장애물과 가장 가까운 지점 찾기
            obstacle_on_path = None
            obstacle_path_idx = None
            min_dist_to_path = float('inf')
            
            for ox, oy, oz in self._obstacles:
                for i, (px, py) in enumerate(path):
                    dist_to_path = math.hypot(ox - px, oy - py)
                    if dist_to_path < self.obstacle_radius + 2.0:
                        if dist_to_path < min_dist_to_path:
                            min_dist_to_path = dist_to_path
                            obstacle_on_path = (ox, oy)
                            obstacle_path_idx = i
            
            if obstacle_on_path is None:
                # 이 차량 경로에 장애물 없음
                if role in self._obstacle_blocked_roles:
                    # 이전에 막혀있었으면 replan
                    self._obstacle_blocked_roles.pop(role)
                    if self._plan_for_role(vehicle, role, front_loc):
                        rospy.loginfo(f"{role}: replanned - obstacle no longer on path")
                continue
            
            # 차량에서 장애물까지 거리
            dist_to_obstacle = math.hypot(obstacle_on_path[0] - front_loc.x, 
                                          obstacle_on_path[1] - front_loc.y)
            
            # 이미 이 장애물로 정지 중이면 스킵
            if role in self._obstacle_blocked_roles:
                continue
            
            rospy.logwarn(f"[OBSTACLE] {role}: obstacle at ({obstacle_on_path[0]:.1f}, {obstacle_on_path[1]:.1f}), "
                         f"dist={dist_to_obstacle:.1f}m, path_idx={obstacle_path_idx}")
            
            # 정지점 계산: 장애물 위치에서 stop_distance만큼 뒤
            stop_idx = max(0, obstacle_path_idx - int(stop_distance / 0.5))  # 0.5m 간격 가정
            
            # 경로 자르기
            trimmed_path = path[:stop_idx + 1]
            if len(trimmed_path) >= 2:
                self._publish_path(trimmed_path, role)
                self._store_active_path(role, trimmed_path)
                self._obstacle_blocked_roles[role] = obstacle_on_path
                rospy.logwarn(f"[OBSTACLE] {role}: path trimmed to {len(trimmed_path)} points (stop before obstacle)")

    def _generate_avoidance_path(self, role: str, obstacle_location: Tuple[float, float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        장애물 회피 경로 생성
        TODO: 회피 알고리즘 구현
        - FrenetPath 활용
        - 좌/우 회피 방향 결정
        - 회피 경로 생성
        
        Args:
            role: 차량 역할명
            obstacle_location: 장애물 위치 (x, y, z)
            
        Returns:
            회피 경로 [(x, y), ...] 또는 None
        """
        # TODO: 실제 회피 알고리즘 구현
        rospy.logdebug(f"{role}: generate_avoidance_path placeholder for obstacle at {obstacle_location}")
        return None

    def _should_replan_for_obstacle(self, role: str, vehicle_loc: carla.Location) -> bool:
        """
        장애물로 인해 재계획이 필요한지 판단
        TODO: 판단 로직 구현
        - 현재 경로와 장애물 충돌 확인
        - 거리 임계값 확인
        
        Args:
            role: 차량 역할명
            vehicle_loc: 차량 현재 위치
            
        Returns:
            재계획 필요 여부
        """
        if not self._obstacles:
            return False
        
        # 가장 가까운 장애물 거리
        min_dist = float('inf')
        for ox, oy, oz in self._obstacles:
            dist = math.hypot(vehicle_loc.x - ox, vehicle_loc.y - oy)
            min_dist = min(min_dist, dist)
        
        # TODO: 더 정교한 판단 로직 추가
        return min_dist < self.obstacle_replan_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Overlap Detection (for debugging)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_edges_from_route(self, route) -> List[Tuple[int, int]]:
        """
        GlobalPlanner route에서 엣지 리스트 추출
        
        Args:
            route: [(waypoint, RoadOption), ...]
            
        Returns:
            [(n1, n2), ...] 엣지 리스트
        """
        if not route or not hasattr(self.route_planner, '_localize'):
            return []
        
        edges = []
        for wp, _ in route:
            loc = wp.transform.location
            edge = self.route_planner._localize(loc)
            if edge is not None and edge not in edges:
                edges.append(edge)
        return edges

    def _update_vehicle_edges_from_route(self, role: str, route) -> None:
        """route에서 엣지 추출하여 저장"""
        edges = self._extract_edges_from_route(route)
        if edges:
            self._vehicle_route_edges[role] = edges
            
            # 엣지 리스트와 차선 정보 로깅
            rospy.loginfo(f"[EDGES] {role}: {len(edges)} edges in route")
            lane_info = []
            for i, edge in enumerate(edges[:10]):  # 최대 10개만 출력
                lane_key = self.route_planner._edge_to_lane.get(edge, None)
                if lane_key:
                    lane_info.append(f"{i}:{lane_key}")
            rospy.loginfo(f"  Lanes (first 10): {', '.join(lane_info)}")

    def _log_overlap_detection_cb(self, _evt) -> None:
        """주기적으로 반대 차선 중복 상태 로깅"""
        if not hasattr(self.route_planner, 'find_opposite_lane_overlaps'):
            return
        
        roles = list(self._vehicle_route_edges.keys())
        if len(roles) < 2:
            return
        
        for i, role_a in enumerate(roles):
            edges_a = self._vehicle_route_edges.get(role_a, [])
            if not edges_a:
                continue
            
            for role_b in roles[i+1:]:
                edges_b = self._vehicle_route_edges.get(role_b, [])
                if not edges_b:
                    continue
                
                overlaps = self.route_planner.find_opposite_lane_overlaps(edges_a, edges_b)
                if overlaps:
                    rospy.loginfo(
                        f"[OVERLAP] {role_a} vs {role_b}: {len(overlaps)} opposite-lane overlaps detected"
                    )
                    for idx_a, edge_a, idx_b, edge_b in overlaps[:3]:  # 최대 3개만 로깅
                        lane_a = self.route_planner._edge_to_lane.get(edge_a, "?")
                        lane_b = self.route_planner._edge_to_lane.get(edge_b, "?")
                        rospy.loginfo(
                            f"  - {role_a}[{idx_a}] lane{lane_a} <-> {role_b}[{idx_b}] lane{lane_b}"
                        )

    def _replan_check_cb(self, _evt) -> None:
        # Distance-gated replanning per vehicle
        vehicles = self._get_ego_vehicles()
        if not vehicles:
            return
        for index, vehicle in enumerate(vehicles[: self.num_vehicles]):
            role = self._role_name(index)
            
            # 장애물로 정지 중인 차량은 replan 스킵 (우회 불가 시 정지 유지)
            if role in self._obstacle_blocked_roles:
                continue
            
            front_loc = self._vehicle_front(vehicle)
            dest_override = None
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
            # Update vehicle edges for overlap detection
            self._update_vehicle_edges_from_route(role, route)
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

if __name__ == "__main__":
    try:
        SimpleMultiAgentPlanner()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleMultiAgentPlanner crashed: {e}")
        raise
