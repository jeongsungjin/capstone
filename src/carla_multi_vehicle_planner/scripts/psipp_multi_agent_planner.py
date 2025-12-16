#!/usr/bin/env python3
"""
PSIPP 기반 다중 차량 충돌 회피 경로 플래너
- CARLA waypoint에서 로드맵 생성
- PSIPP로 충돌 회피 경로 계획
- nav_msgs/Path로 /global_path_{role} 토픽에 발행
"""

import math
from collections import deque
from typing import Dict, List, Tuple, Optional

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header

import psipp

try:
    import setup_carla_path  # noqa: F401
except Exception:
    pass

try:
    import carla
except Exception as exc:
    carla = None
    print(f"Failed to import CARLA: {exc}")


class PSIPPMultiAgentPlanner:
    """PSIPP 기반 다중 차량 충돌 회피 플래너"""
    
    def __init__(self):
        rospy.init_node("psipp_multi_agent_planner", anonymous=True)
        
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")
        
        # Parameters
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.waypoint_spacing = float(rospy.get_param("~waypoint_spacing", 0.1))  # 0.1m 간격
        self.vehicle_radius = float(rospy.get_param("~vehicle_radius", 2.0))
        self.replan_interval = float(rospy.get_param("~replan_interval", 5.0))
        self.min_destination_distance = float(rospy.get_param("~min_destination_distance", 10.0))
        self.max_destination_distance = float(rospy.get_param("~max_destination_distance", 500.0))
        self.goal_reached_threshold = float(rospy.get_param("~goal_reached_threshold", 2.0))  # 목표 도달 판정 거리 (m)
        self.check_interval = float(rospy.get_param("~check_interval", 0.01))  # 도달 체크 주기 (s)
        self.roadmap_file = rospy.get_param("~roadmap_file", "")  # JSON 로드맵 파일 경로
        self.target_speed = float(rospy.get_param("~target_speed", 6.0))  # 차량 목표 속도 (m/s)
        
        # CARLA connection
        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        timeout = float(rospy.get_param("~carla_timeout", 10.0))
        
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.spawn_points = self.carla_map.get_spawn_points()
        
        rospy.loginfo(f"Connected to CARLA. Map: {self.carla_map.name}")
        
        # Load or build roadmap
        # if self.roadmap_file and self._load_roadmap_from_file(self.roadmap_file):
        #     rospy.loginfo(f"Loaded roadmap from {self.roadmap_file}")
        # else:
        self._build_roadmap()
        
        # PSIPP planner
        self.planner = psipp.Planner()
        self.planner.set_roadmap(self.vertices, self.edges, self.vehicle_radius, self.target_speed)
        rospy.loginfo(f"PSIPP roadmap: {len(self.vertices)} vertices, {len(self.edges)} edges, velocity={self.target_speed}m/s")
        
        # Publishers
        self.path_publishers: Dict[str, rospy.Publisher] = {}
        for i in range(self.num_vehicles):
            role = self._role_name(i)
            topic = f"/global_path_{role}"
            self.path_publishers[role] = rospy.Publisher(topic, Path, queue_size=1, latch=True)
        
        # State
        self._current_dest: Dict[str, Optional[int]] = {}      # 현재 목표 vertex_id
        self._active_plans: Dict[str, List[int]] = {}          # 현재 경로 (vertex path)
        self._goal_positions: Dict[str, Tuple[float, float]] = {}  # 목표 좌표
        
        # Timing tracking (for logging planned vs actual arrival times)
        self._program_start_time: Optional[rospy.Time] = None  # 프로그램 시작 시간
        self._expected_times: Dict[str, Dict[int, float]] = {}  # role -> {vertex_id -> expected_time}
        self._last_logged_vertex: Dict[str, int] = {}  # 마지막으로 로깅한 vertex (중복 방지)
        self._plan_start_times: Dict[str, float] = {}  # role -> plan 시작 시 상대 시간
        
        rospy.sleep(0.5)
        
        # Record program start time for relative timing
        self._program_start_time = rospy.Time.now()
        rospy.loginfo("[timing] Program start time recorded for relative timing")
        
        # Initial planning
        self._plan_all_vehicles()
        
        # Periodic goal-reaching check (faster interval)
        rospy.Timer(rospy.Duration(self.check_interval), self._check_goal_reached_cb)
        
        rospy.loginfo(f"PSIPP Multi-Agent Planner started! (goal_threshold={self.goal_reached_threshold}m)")
    
    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"
    
    def _load_roadmap_from_file(self, filepath: str) -> bool:
        """JSON 파일에서 로드맵 로드"""
        import json
        import os
        
        if not os.path.exists(filepath):
            rospy.logwarn(f"Roadmap file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.vertices = [tuple(v) for v in data["vertices"]]
            self.edges = [tuple(e) for e in data["edges"]]
            
            rospy.loginfo(f"Loaded from JSON: {len(self.vertices)} vertices, {len(self.edges)} edges")
            return True
        except Exception as e:
            rospy.logwarn(f"Failed to load roadmap: {e}")
            return False
    
    def _build_roadmap(self):
        """토폴로지 노드만으로 PSIPP 로드맵 생성 (고속 계획용)"""
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        
        rospy.loginfo("Building coarse topology roadmap for fast planning...")
        
        # GlobalRoutePlanner 초기화
        self.grp = GlobalRoutePlanner(self.carla_map, 0.1)  # 보간용으로 저장
        topology_graph = self.grp._graph
        
        rospy.loginfo(f"Topology graph: {topology_graph.number_of_nodes()} nodes, {topology_graph.number_of_edges()} edges")
        
        # 토폴로지 노드만 vertex로 사용
        coord_precision = 1
        coord_to_id = {}
        self.vertices: List[Tuple[float, float]] = []
        edges_set = set()
        
        # 엣지별 waypoint 경로 저장 (보간용)
        self.edge_paths: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        
        def add_vertex(loc) -> int:
            """좌표에서 vertex ID 가져오기 (없으면 생성)"""
            key = (round(loc.x, coord_precision), round(loc.y, coord_precision))
            if key not in coord_to_id:
                vertex_id = len(self.vertices)
                coord_to_id[key] = vertex_id
                self.vertices.append((loc.x, loc.y))
            return coord_to_id[key]
        
        # 토폴로지 엣지만 사용 (entry -> exit)
        for u, v, edge_data in topology_graph.edges(data=True):
            entry_wp = edge_data.get('entry_waypoint')
            exit_wp = edge_data.get('exit_waypoint')
            path_waypoints = edge_data.get('path', [])
            
            if entry_wp is None or exit_wp is None:
                continue
            
            # entry와 exit만 vertex로 추가
            entry_vid = add_vertex(entry_wp.transform.location)
            exit_vid = add_vertex(exit_wp.transform.location)
            
            if entry_vid != exit_vid:
                edges_set.add((entry_vid, exit_vid))
                
                # 이 엣지의 세부 경로 저장 (나중에 보간용)
                path_coords = []
                path_coords.append((entry_wp.transform.location.x, entry_wp.transform.location.y))
                for wp in path_waypoints:
                    path_coords.append((wp.transform.location.x, wp.transform.location.y))
                path_coords.append((exit_wp.transform.location.x, exit_wp.transform.location.y))
                self.edge_paths[(entry_vid, exit_vid)] = path_coords
        
        self.edges = list(edges_set)
        rospy.loginfo(f"Created {len(self.vertices)} vertices, {len(self.edges)} edges (topology nodes only)")
    
    def _find_closest_vertex(self, x: float, y: float) -> int:
        """가장 가까운 vertex 찾기"""
        min_dist = float('inf')
        closest_id = 0
        for vid, (vx, vy) in enumerate(self.vertices):
            dist = math.sqrt((x - vx) ** 2 + (y - vy) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_id = vid
        return closest_id
    
    def _interpolate_path(self, vertex_path: List[int]) -> List[Tuple[float, float]]:
        """PSIPP vertex path를 캐시된 edge_paths로 빠르게 보간 (trace_route 없음)"""
        if len(vertex_path) < 2:
            return [self.vertices[v] for v in vertex_path]
        
        detailed_path = []
        
        for i in range(len(vertex_path) - 1):
            from_vid = vertex_path[i]
            to_vid = vertex_path[i + 1]
            
            # 캐시된 edge_paths 사용 (trace_route 호출 없음)
            edge_key = (from_vid, to_vid)
            if edge_key in self.edge_paths:
                edge_path = self.edge_paths[edge_key]
                if i == 0:
                    detailed_path.extend(edge_path)
                else:
                    detailed_path.extend(edge_path[1:])  # 시작점 중복 제거
            else:
                # fallback: 직선 연결
                if i == 0:
                    detailed_path.append(self.vertices[from_vid])
                detailed_path.append(self.vertices[to_vid])
        
        return detailed_path
    
    
    def _get_ego_vehicles(self) -> List:
        """ego 차량 목록 가져오기"""
        actors = self.world.get_actors().filter("vehicle.*")
        vehicles = []
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role.startswith("ego_vehicle_"):
                vehicles.append(actor)
        vehicles.sort(key=lambda v: v.attributes.get("role_name", ""))
        return vehicles
    
    def _vehicle_front(self, vehicle) -> Tuple[float, float]:
        """차량 전방 위치 계산"""
        tf = vehicle.get_transform()
        yaw_rad = math.radians(tf.rotation.yaw)
        offset = 3.0  # 차량 전방 오프셋
        bb = getattr(vehicle, "bounding_box", None)
        if bb is not None and getattr(bb, "extent", None) is not None:
            offset = bb.extent.x + 0.3
        
        front_x = tf.location.x + math.cos(yaw_rad) * offset
        front_y = tf.location.y + math.sin(yaw_rad) * offset
        return front_x, front_y
    
    def _choose_destination(self, start_x: float, start_y: float) -> Optional[int]:
        """거리 조건 만족하는 임의 목적지 선택"""
        import random
        for _ in range(50):
            sp = random.choice(self.spawn_points)
            dest_x, dest_y = sp.location.x, sp.location.y
            dist = math.hypot(dest_x - start_x, dest_y - start_y)
            if self.min_destination_distance <= dist <= self.max_destination_distance:
                return self._find_closest_vertex(dest_x, dest_y)
        # Fallback: 가장 먼 spawn point
        max_dist = 0
        best_vid = 0
        for sp in self.spawn_points:
            dist = math.hypot(sp.location.x - start_x, sp.location.y - start_y)
            if dist > max_dist:
                max_dist = dist
                best_vid = self._find_closest_vertex(sp.location.x, sp.location.y)
        return best_vid
    
    def _plan_all_vehicles(self):
        """모든 차량의 경로를 동시에 계획"""
        vehicles = self._get_ego_vehicles()
        if not vehicles:
            rospy.logwarn_throttle(5.0, "No ego vehicles found")
            return
        
        vehicles = vehicles[:self.num_vehicles]
        
        # Collect tasks
        tasks = []
        vehicle_info = []
        
        for i, vehicle in enumerate(vehicles):
            role = self._role_name(i)
            front_x, front_y = self._vehicle_front(vehicle)
            start_vertex = self._find_closest_vertex(front_x, front_y)
            goal_vertex = self._choose_destination(front_x, front_y)
            
            if goal_vertex is None:
                rospy.logwarn(f"{role}: Could not find destination")
                continue
            
            tasks.append((start_vertex, goal_vertex, 0.0))  # 초기 계획: t=0에서 시작
            vehicle_info.append((vehicle, role, start_vertex, goal_vertex))
            
            start_pos = self.vertices[start_vertex]
            goal_pos = self.vertices[goal_vertex]
            rospy.loginfo(f"{role}: ({start_pos[0]:.1f}, {start_pos[1]:.1f}) -> ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
        
        if not tasks:
            return
        
        # Plan with PSIPP
        import time
        rospy.loginfo(f"Planning {len(tasks)} vehicle paths with PSIPP...")
        start_time = time.time()
        plans = self.planner.plan(tasks)
        elapsed_ms = (time.time() - start_time) * 1000
        rospy.loginfo(f"PSIPP planning completed in {elapsed_ms:.1f}ms")
        
        # Publish paths and store timing info
        plan_base_time = self._get_relative_time()  # 현재 상대 시간 (계획 시작점)
        
        for plan, (vehicle, role, start_v, goal_v) in zip(plans, vehicle_info):
            if plan.success:
                # PSIPP vertex path -> GlobalRoutePlanner 보간 경로
                path_points = self._interpolate_path(plan.vertex_path)
                self._publish_path(path_points, role)
                self._active_plans[role] = plan.vertex_path
                self._current_dest[role] = goal_v
                self._goal_positions[role] = self.vertices[goal_v]  # 목표 좌표 저장
                
                # Store expected times from PSIPP plan
                self._store_expected_times(role, plan, plan_base_time)
                
                total_dist = sum(
                    math.hypot(
                        self.vertices[plan.vertex_path[i+1]][0] - self.vertices[plan.vertex_path[i]][0],
                        self.vertices[plan.vertex_path[i+1]][1] - self.vertices[plan.vertex_path[i]][1]
                    )
                    for i in range(len(plan.vertex_path) - 1)
                )
                # PSIPP 시간 → 실제 시간 (target_speed 적용)
                actual_arrival_time = total_dist / self.target_speed
                rospy.loginfo(f"{role}: SUCCESS - {len(plan.vertex_path)} waypoints, {total_dist:.1f}m, ETA {actual_arrival_time:.1f}s (@{self.target_speed}m/s)")
                
                # Log planned wait points (where PSIPP expects waiting)
                self._log_wait_points(role, plan, plan_base_time)
            else:
                rospy.logwarn(f"{role}: FAILED to find collision-free path")
        
        # Visualize in CARLA
        self._visualize_paths(plans, vehicle_info)
    
    def _publish_path(self, points: List[Tuple[float, float]], role: str) -> None:
        """nav_msgs/Path 발행 (simple_multi_agent_planner와 동일한 형식)"""
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
        rospy.logdebug(f"{role}: Published path with {len(points)} poses")
    
    def _visualize_paths(self, plans, vehicle_info):
        return

        """CARLA에서 경로 시각화"""
        debug = self.world.debug
        
        colors = [
            carla.Color(255, 0, 0),    # Red - Vehicle 1
            carla.Color(0, 255, 0),    # Green - Vehicle 2
            carla.Color(0, 0, 255),    # Blue - Vehicle 3
            carla.Color(255, 255, 0),  # Yellow - Vehicle 4
            carla.Color(255, 0, 255),  # Magenta - Vehicle 5
            carla.Color(0, 255, 255),  # Cyan - Vehicle 6
        ]
        
        for plan, (vehicle, role, start_v, goal_v) in zip(plans, vehicle_info):
            if not plan.success:
                continue
            
            color = colors[plan.agent_id % len(colors)]
            path = plan.vertex_path
            z_offset = 0.5 + plan.agent_id * 0.2
            
            # Draw path
            for i in range(len(path) - 1):
                v1 = self.vertices[path[i]]
                v2 = self.vertices[path[i + 1]]
                debug.draw_line(
                    carla.Location(x=v1[0], y=v1[1], z=z_offset),
                    carla.Location(x=v2[0], y=v2[1], z=z_offset),
                    thickness=0.15, color=color, life_time=self.replan_interval + 1.0
                )
            
            # Start/Goal markers
            debug.draw_point(
                carla.Location(x=self.vertices[path[0]][0], y=self.vertices[path[0]][1], z=1.0),
                size=0.3, color=carla.Color(255, 255, 255), life_time=self.replan_interval + 1.0
            )
            debug.draw_point(
                carla.Location(x=self.vertices[path[-1]][0], y=self.vertices[path[-1]][1], z=1.0),
                size=0.3, color=color, life_time=self.replan_interval + 1.0
            )
    
    def _get_relative_time(self) -> float:
        """프로그램 시작 이후 상대 시간 (초) 반환"""
        if self._program_start_time is None:
            return 0.0
        return (rospy.Time.now() - self._program_start_time).to_sec()
    
    def _store_expected_times(self, role: str, plan, plan_base_time: float) -> None:
        """PSIPP plan의 moves에서 각 vertex 도착 예상 시간 저장"""
        self._expected_times[role] = {}
        self._plan_start_times[role] = plan_base_time
        self._last_logged_vertex[role] = -1
        
        # 시작 vertex
        if plan.vertex_path:
            self._expected_times[role][plan.vertex_path[0]] = plan_base_time
        
        # moves에서 각 target_vertex의 end_time 저장
        for move in plan.moves:
            # end_time은 PSIPP 내부 시간 (plan 시작 기준)
            # plan_base_time을 더해 절대 상대 시간으로 변환
            expected_arrival = plan_base_time + move.end_time
            self._expected_times[role][move.target_vertex] = expected_arrival
        
        rospy.logdebug(f"[timing] {role}: Stored expected times for {len(self._expected_times[role])} vertices")
    
    def _log_wait_points(self, role: str, plan, plan_base_time: float) -> None:
        """PSIPP가 대기를 계획한 지점 로깅 (start_time > 이전 end_time인 경우)"""
        prev_end_time = 0.0
        for move in plan.moves:
            wait_duration = move.start_time - prev_end_time
            if wait_duration > 0.1:  # 0.1초 이상 대기인 경우만 로깅
                vertex_coords = self.vertices[move.target_vertex]
                rospy.loginfo(
                    f"[timing][wait] {role}: WAIT {wait_duration:.2f}s before vertex {move.target_vertex} "
                    f"({vertex_coords[0]:.1f}, {vertex_coords[1]:.1f}) at t={plan_base_time + move.start_time:.2f}s"
                )
            prev_end_time = move.end_time
    
    def _log_vertex_timing(self, role: str, vertex_id: int, actual_time: float) -> None:
        """특정 vertex 도착 시 예상 시간과 실제 시간 비교 로깅"""
        expected_times = self._expected_times.get(role, {})
        if vertex_id not in expected_times:
            return
        
        expected_time = expected_times[vertex_id]
        diff = actual_time - expected_time
        vertex_coords = self.vertices[vertex_id]
        
        # Log with clear formatting
        sign = "+" if diff >= 0 else ""
        rospy.loginfo(
            f"[timing] {role} | vertex={vertex_id} ({vertex_coords[0]:.1f}, {vertex_coords[1]:.1f}) | "
            f"expected={expected_time:.2f}s, actual={actual_time:.2f}s, diff={sign}{diff:.2f}s"
        )
    
    def _check_goal_reached_cb(self, event):
        """주기적으로 각 차량의 목표 도달 여부 체크 + 웨이포인트 통과 로깅"""
        vehicles = self._get_ego_vehicles()
        if not vehicles:
            return
        
        vehicles_to_replan = []
        current_time = self._get_relative_time()
        
        for i, vehicle in enumerate(vehicles[:self.num_vehicles]):
            role = self._role_name(i)
            goal_vid = self._current_dest.get(role)
            
            if goal_vid is None:
                # 목표가 없으면 재계획 필요
                vehicles_to_replan.append((i, vehicle, role))
                continue
            
            # 차량 현재 위치
            front_x, front_y = self._vehicle_front(vehicle)
            
            # 현재 경로에서 가장 가까운 vertex 찾기 (timing logging용)
            active_plan = self._active_plans.get(role, [])
            if active_plan:
                closest_vid = self._find_closest_vertex_in_path(front_x, front_y, active_plan)
                last_logged = self._last_logged_vertex.get(role, -1)
                
                # 새로운 vertex에 도달한 경우에만 로깅
                if closest_vid != last_logged and closest_vid in self._expected_times.get(role, {}):
                    self._log_vertex_timing(role, closest_vid, current_time)
                    self._last_logged_vertex[role] = closest_vid
            
            # 목표 위치
            goal_x, goal_y = self.vertices[goal_vid]
            
            # 목표까지 거리
            dist_to_goal = math.hypot(front_x - goal_x, front_y - goal_y)
            
            if dist_to_goal <= self.goal_reached_threshold:
                # 최종 목표 도달 시에도 timing 로깅
                self._log_vertex_timing(role, goal_vid, current_time)
                rospy.loginfo(f"{role}: Goal reached! (dist={dist_to_goal:.1f}m <= threshold={self.goal_reached_threshold}m)")
                vehicles_to_replan.append((i, vehicle, role))
        
        # 도달한 차량들 재계획
        if vehicles_to_replan:
            self._replan_vehicles(vehicles_to_replan)
    
    def _find_closest_vertex_in_path(self, x: float, y: float, path: List[int]) -> int:
        """경로 내에서 가장 가까운 vertex 찾기"""
        min_dist = float('inf')
        closest_vid = path[0] if path else -1
        for vid in path:
            vx, vy = self.vertices[vid]
            dist = math.hypot(x - vx, y - vy)
            if dist < min_dist:
                min_dist = dist
                closest_vid = vid
        return closest_vid
    
    def _replan_vehicles(self, vehicles_to_replan: List[Tuple[int, any, str]]):
        """특정 차량들만 재계획 (다른 차량 경로 고려)"""
        if not vehicles_to_replan:
            return
        
        all_vehicles = self._get_ego_vehicles()[:self.num_vehicles]
        
        # 재계획 대상 차량만 처리 (다른 차량은 기존 경로 유지)
        replan_indices = {v[0] for v in vehicles_to_replan}
        
        for idx, vehicle, role in vehicles_to_replan:
            front_x, front_y = self._vehicle_front(vehicle)
            
            # 현재 위치에서 가장 가까운 vertex
            start_vertex = self._find_closest_vertex(front_x, front_y)
            
            # 새 목적지 선택
            goal_vertex = self._choose_destination(front_x, front_y)
            
            if goal_vertex is None:
                rospy.logwarn(f"{role}: Could not find destination")
                continue
            
            # 이 차량만 재계획 (t=0에서 시작)
            tasks = [(start_vertex, goal_vertex, 0.0)]
            
            import time
            plan_start = time.time()
            plans = self.planner.plan(tasks)
            elapsed_ms = (time.time() - plan_start) * 1000
            
            if plans and plans[0].success:
                plan = plans[0]
                
                # 새 경로만 사용 (PSIPP가 이미 현재 위치에서 새 목표까지 계획함)
                # 이전에 remaining_path를 prepend하던 로직은 폐루프를 만들어 제거함
                combined_vertex_path = plan.vertex_path
                
                # 경로 발행
                path_points = self._interpolate_path(combined_vertex_path)
                self._publish_path(path_points, role)
                self._active_plans[role] = combined_vertex_path
                self._current_dest[role] = goal_vertex
                self._goal_positions[role] = self.vertices[goal_vertex]
                
                # Store expected times for timing logging
                plan_base_time = self._get_relative_time()
                self._store_expected_times(role, plan, plan_base_time)
                self._log_wait_points(role, plan, plan_base_time)
                
                total_dist = sum(
                    math.hypot(
                        self.vertices[combined_vertex_path[i+1]][0] - self.vertices[combined_vertex_path[i]][0],
                        self.vertices[combined_vertex_path[i+1]][1] - self.vertices[combined_vertex_path[i]][1]
                    )
                    for i in range(len(combined_vertex_path) - 1)
                )
                actual_arrival_time = total_dist / self.target_speed
                rospy.loginfo(f"{role}: NEW PATH - {len(combined_vertex_path)} waypoints, {total_dist:.1f}m, ETA {actual_arrival_time:.1f}s (plan took {elapsed_ms:.1f}ms)")
            else:
                rospy.logwarn(f"{role}: FAILED to find collision-free path")
    
    def _visualize_single_paths(self, plan_info_list):
        """개별 차량 경로 시각화"""
        debug = self.world.debug
        colors = [
            carla.Color(255, 0, 0),    # Red - Vehicle 1
            carla.Color(0, 255, 0),    # Green - Vehicle 2
            carla.Color(0, 0, 255),    # Blue - Vehicle 3
            carla.Color(255, 255, 0),  # Yellow - Vehicle 4
            carla.Color(255, 0, 255),  # Magenta - Vehicle 5
            carla.Color(0, 255, 255),  # Cyan - Vehicle 6
        ]
        
        for plan, (vehicle, role, start_v, goal_v, _) in plan_info_list:
            if not plan.success:
                continue
            
            color = colors[plan.agent_id % len(colors)]
            path = plan.vertex_path
            z_offset = 0.5 + plan.agent_id * 0.2
            
            for i in range(len(path) - 1):
                v1 = self.vertices[path[i]]
                v2 = self.vertices[path[i + 1]]
                debug.draw_line(
                    carla.Location(x=v1[0], y=v1[1], z=z_offset),
                    carla.Location(x=v2[0], y=v2[1], z=z_offset),
                    thickness=0.15, color=color, life_time=30.0
                )
            
            # Goal marker
            debug.draw_point(
                carla.Location(x=self.vertices[path[-1]][0], y=self.vertices[path[-1]][1], z=1.0),
                size=0.4, color=color, life_time=30.0
            )


if __name__ == "__main__":
    try:
        planner = PSIPPMultiAgentPlanner()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"PSIPP Multi-Agent Planner crashed: {e}")
        raise
