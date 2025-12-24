#!/usr/bin/env python3
"""
ObstaclePlanner: 장애물 회피 전담 플래너

기능:
- 회피 경로 생성 (FrenetPath 활용)
- 경로 점유 확인
- 회피/정지 판단
"""

import math
from typing import List, Tuple, Optional, Dict

import setup_carla_path  # noqa: F401
import carla

import rospy
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header

from frenet_path import FrenetPath


class ObstaclePlanner:
    """
    장애물 회피 전담 플래너.
    
    GlobalPlanner와 함께 사용하여 장애물 회피 경로 생성 및 판단.
    """

    def __init__(self, global_planner, num_vehicles, on_obstacle_change_callback=None):
        """
        Args:
            global_planner: GlobalPlanner 인스턴스
            num_vehicles: 차량 수
            on_obstacle_change_callback: 장애물 변화 시 호출할 콜백 함수
        """
        self.route_planner = global_planner
        self._on_obstacle_change_callback = on_obstacle_change_callback

        self.obstacle_radius = float(rospy.get_param("~obstacle_radius", 5.0))  # 경로 차단용 반경
        self.obstacle_collision_radius = float(rospy.get_param("~obstacle_collision_radius", 1.0))  # 회피 경로 충돌 검사용 반경
        self.obstacle_replan_threshold = float(rospy.get_param("~obstacle_replan_threshold", 30.0))
        self.obstacle_stop_distance = float(rospy.get_param("~obstacle_stop_distance", 1.0))  # 장애물 앞 정지 거리
        self.avoidance_margin_after = float(rospy.get_param("~avoidance_margin_after", 1.0))  # 장애물 뒤 복귀 마진

        self.num_vehicles = num_vehicles
        self._role_name = lambda index: f"ego_vehicle_{index + 1}"

        self._avoidance_path_pubs: Dict[str, Dict[float, rospy.Publisher]] = {}  # role -> {d: Publisher}
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            self._avoidance_path_pubs[role] = {}
            for d in range(-20, 40, 5):
                d /= 10
                avoidance_topic = f"/avoidance_path_{role}_d{d:.1f}".replace("-", "n").replace(".", "_")
                self._avoidance_path_pubs[role][d] = rospy.Publisher(avoidance_topic, Path, queue_size=1, latch=True)

        # 장애물로 인해 정지 중인 차량 추적 (role -> obstacle_pos)
        self._obstacle_blocked_roles: Dict[str, Tuple[float, float]] = {}
        
        # Obstacle subscriber
        self.is_obstacle_list_changed = False
        self._obstacles: List[Tuple[float, float, float]] = []
        rospy.Subscriber("/obstacles", PoseArray, self._obstacle_cb, queue_size=1)

        # Overlap detection logging timer (for debugging)
        self._overlap_log_interval = float(rospy.get_param("~overlap_log_interval", 5.0))
        rospy.Timer(rospy.Duration(self._overlap_log_interval), self._log_overlap_detection_cb)
        
        # Vehicle route edges tracking (role -> [(n1, n2), ...])
        self._vehicle_route_edges: Dict[str, List[Tuple[int, int]]] = {}

    def _obstacle_cb(self, msg: PoseArray) -> None:
        """장애물 토픽 콜백 - 장애물 변화 시 노드 차단 업데이트"""
        new_obstacles = []
        for pose in msg.poses:
            new_obstacles.append((pose.position.x, pose.position.y, pose.position.z))
        
        self.is_obstacle_list_changed = self.obstacle_changed(new_obstacles)
        
        # 장애물 변화 감지
        if self.is_obstacle_list_changed:
            rospy.loginfo(f"Obstacles changed: {len(self._obstacles)} -> {len(new_obstacles)}")
            self._obstacles = new_obstacles
            self._update_blocked_nodes()
            
            # 콜백 호출 - 즉시 회피 적용
            if self._on_obstacle_change_callback is not None:
                try:
                    self._on_obstacle_change_callback()
                except Exception as e:
                    rospy.logwarn(f"on_obstacle_change_callback failed: {e}")

    def obstacle_changed(self, new_obstacles):
        if len(new_obstacles) != len(self._obstacles):
            return True
        
        for new, old in zip(sorted(new_obstacles), sorted(self._obstacles)):
            if math.hypot(new[0] - old[0], new[1] - old[1]) > 1.0:
                return True

        return False

    def _update_blocked_nodes(self) -> None:
        """장애물 위치로 차단 노드/엣지 업데이트 (node_list 기반)"""
        if self.route_planner is None:
            return
        if not hasattr(self.route_planner, 'clear_obstacles'):
            rospy.logwarn_throttle(10.0, "route_planner does not have clear_obstacles method")
            return
        
        self.route_planner.clear_obstacles()
        
        # 모든 차량의 경로 엣지와 장애물 비교
        for ox, oy, oz in self._obstacles:
            obstacle_loc = carla.Location(x=ox, y=oy, z=oz)
            blocked_count = 0
            
            # 각 차량의 route edges에서 장애물과 가까운 엣지 찾기
            for role, edges in self._vehicle_route_edges.items():
                for edge in edges:
                    # 엣지의 waypoints와 장애물 거리 확인
                    if self._is_edge_blocked_by_obstacle(edge, obstacle_loc):
                        self.route_planner.mark_edge_blocked(edge)
                        blocked_count += 1
            
            # fallback: edge 기반으로 못 찾으면 기존 방식 사용
            if blocked_count == 0:
                blocked = self.route_planner.add_obstacle_on_road(obstacle_loc, self.obstacle_radius)
                edge = self.route_planner._localize(obstacle_loc)
                rospy.loginfo(f"Obstacle at ({ox:.1f}, {oy:.1f}): blocked {blocked} nodes (fallback), edge={edge}")
            else:
                rospy.loginfo(f"Obstacle at ({ox:.1f}, {oy:.1f}): blocked {blocked_count} edges (node_list based)")

    def _is_edge_blocked_by_obstacle(self, edge: Tuple[int, int], obstacle_loc) -> bool:
        """엣지가 장애물에 의해 막혔는지 확인"""
        if self.route_planner is None or not hasattr(self.route_planner, '_graph'):
            return False
        
        n1, n2 = edge
        graph = self.route_planner._graph
        
        if not graph.has_edge(n1, n2):
            return False
        
        # 엣지의 path (waypoints) 가져오기
        edge_data = graph.edges[n1, n2]
        path = edge_data.get('path', [])
        
        ox, oy = obstacle_loc.x, obstacle_loc.y
        
        for wp in path:
            loc = wp.transform.location
            if math.hypot(loc.x - ox, loc.y - oy) < self.obstacle_radius:
                return True
        
        return False

    def update_vehicle_edges_from_nodes(self, role: str, node_list) -> None:
        """A* 노드 리스트에서 직접 엣지 추출하여 저장"""
        if not node_list:
            return
        
        # GlobalPlanner의 nodes_to_edges 사용
        if hasattr(self.route_planner, 'nodes_to_edges'):
            edges = self.route_planner.nodes_to_edges(node_list)
        else:
            # fallback: 직접 변환
            edges = []
            for i in range(len(node_list) - 1):
                edges.append((node_list[i], node_list[i + 1]))
        
        if edges:
            self._vehicle_route_edges[role] = edges
            rospy.logdebug(f"{role}: updated {len(edges)} edges from node_list")

    # ─────────────────────────────────────────────────────────────────────────
    # Path Avoidance Application (Main Entry Point)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_avoidance_to_path(
        self, 
        role: str, 
        path: List[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], bool, float]:
        """
        경로에 회피 적용 (메인 진입점)
        
        Args:
            role: 차량 역할명
            path: 원본 경로 [(x, y), ...]
            
        Returns:
            (modified_path, need_stop, stop_arc_length)
            - modified_path: 회피 적용된 경로 또는 원본
            - need_stop: 회피 불가능하여 정지 필요 여부
            - stop_arc_length: 정지 필요 시 정지 지점까지의 arc-length (아니면 -1)
        """
        if not path or len(path) < 2:
            return path, False, -1.0
        
        # 1. 경로에서 장애물 찾기
        obstacle_info = self._find_obstacle_on_path(path)
        if obstacle_info is None:
            # 장애물 없음 → 그대로 반환
            if role in self._obstacle_blocked_roles:
                self._obstacle_blocked_roles.pop(role)
            return path, False, -1.0
        
        obstacle_pos, obstacle_idx, dist_to_obstacle = obstacle_info
        rospy.loginfo(f"[AVOIDANCE] {role}: obstacle at ({obstacle_pos[0]:.1f}, {obstacle_pos[1]:.1f}), path_idx={obstacle_idx}")
        
        # 2. 회피 경로 생성 시도
        avoidance_result = self._generate_avoidance_segment(path, obstacle_idx)
        
        if avoidance_result is not None:
            # 회피 성공 → 경로 교체
            combined_path, best_d = avoidance_result
            rospy.loginfo(f"[AVOIDANCE] {role}: avoidance applied with d={best_d:.1f}")
            if role in self._obstacle_blocked_roles:
                self._obstacle_blocked_roles.pop(role)
            return combined_path, False, -1.0
        
        # 3. 회피 불가 → 정지 필요 (경로는 그대로 유지, 컨트롤러에서 정지 처리)
        self._obstacle_blocked_roles[role] = obstacle_pos
        rospy.logfatal(f"[MUST STOP] {role}: CANNOT AVOID OBSTACLE at ({obstacle_pos[0]:.1f}, {obstacle_pos[1]:.1f}) - VEHICLE MUST STOP!")
        
        return path, True, -1.0

    def _find_obstacle_on_path(
        self, 
        path: List[Tuple[float, float]]
    ) -> Optional[Tuple[Tuple[float, float], int, float]]:
        """
        경로에서 장애물 찾기
        
        Returns:
            (obstacle_pos, path_idx, distance) 또는 None
        """
        if not self._obstacles:
            return None
        
        best_match = None
        min_dist = float('inf')
        
        for ox, oy, oz in self._obstacles:
            for i, (px, py) in enumerate(path):
                dist = math.hypot(ox - px, oy - py)
                # 경로에 실제로 가까운 장애물만 감지 (기본 1.0 + 1.5 = 2.5m)
                if dist < self.obstacle_collision_radius + 1.5 and dist < min_dist:
                    min_dist = dist
                    best_match = ((ox, oy), i, dist)
        
        return best_match

    def _generate_avoidance_segment(
        self, 
        path: List[Tuple[float, float]], 
        obstacle_idx: int
    ) -> Optional[Tuple[List[Tuple[float, float]], float]]:
        """
        장애물 구간을 Frenet 회피 경로로 교체
        - 가까운 장애물들을 그룹으로 병합
        - 각 그룹마다 하나의 회피 세그먼트
        
        Returns:
            (combined_path, best_d) 또는 None (회피 불가)
        """
        if len(path) < 10:
            return None
        
        try:
            # 경로에 가까운 모든 장애물의 s 좌표 수집
            obstacle_s_values = []
            for ox, oy, oz in self._obstacles:
                min_dist = float('inf')
                closest_idx = 0
                for i, (px, py) in enumerate(path):
                    dist = math.hypot(ox - px, oy - py)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                # 경로에 실제로 가까운 장애물만 고려
                if min_dist < self.obstacle_collision_radius + 1.5:
                    s_approx = 0.0
                    for i in range(min(closest_idx, len(path) - 1)):
                        dx = path[i+1][0] - path[i][0]
                        dy = path[i+1][1] - path[i][1]
                        s_approx += math.hypot(dx, dy)
                    obstacle_s_values.append((s_approx, (ox, oy)))
            
            if not obstacle_s_values:
                return None
            
            obstacle_s_values.sort(key=lambda x: x[0])
            
            # 가까운 장애물들을 그룹으로 병합
            min_return_gap = max(3.0, self.avoidance_margin_after * 2 + self.obstacle_stop_distance)
            groups = []  # [(s_start, s_end, [(ox, oy), ...]), ...]
            
            for s_obs, (ox, oy) in obstacle_s_values:
                if not groups:
                    # 첫 그룹 생성
                    groups.append([s_obs, s_obs, [(ox, oy)]])
                else:
                    last_group = groups[-1]
                    # 이전 그룹의 끝과 현재 장애물 거리
                    gap = s_obs - last_group[1]
                    
                    if gap < min_return_gap:
                        # 기존 그룹에 병합
                        last_group[1] = s_obs  # s_end 확장
                        last_group[2].append((ox, oy))
                    else:
                        # 새 그룹 생성
                        groups.append([s_obs, s_obs, [(ox, oy)]])
            
            rospy.loginfo(f"[AVOIDANCE] {len(obstacle_s_values)} obstacles -> {len(groups)} groups")
            
            # 전체 FrenetPath 생성
            frenet_full = FrenetPath(path)
            
            # 최적 d값 찾기 (모든 장애물에 대해 충돌 없는 최소 d)
            best_d = None
            for d_int in range(-40, 65, 5):  # d = -4.0 ~ 6.0m
                d = d_int / 10.0
                if d == 0:
                    continue
                
                all_clear = True
                for s_obs, (ox, oy) in obstacle_s_values:
                    x_avoid, y_avoid = frenet_full.frenet_to_cartesian(s_obs, d)
                    if math.hypot(x_avoid - ox, y_avoid - oy) < self.obstacle_collision_radius:
                        all_clear = False
                        break
                
                if all_clear:
                    if best_d is None or abs(d) < abs(best_d):
                        best_d = d
            
            if best_d is None:
                rospy.logwarn(f"[AVOIDANCE] No valid d offset found")
                return None
            
            rospy.loginfo(f"[AVOIDANCE] Using d={best_d:.1f}")
            
            # 그룹별로 회피 세그먼트 생성
            combined_path = []
            last_end_idx = 0
            
            for group_s_start, group_s_end, group_obstacles in groups:
                # 그룹의 회피 구간 (앞뒤 마진 포함)
                s_start = max(0.5, group_s_start - self.obstacle_stop_distance - 1.0)
                s_end = min(frenet_full.total_length - 0.5, group_s_end + self.avoidance_margin_after + self.obstacle_radius + 1.0)
                
                # s 좌표 → path 인덱스 변환
                start_idx = 0
                end_idx = len(path) - 1
                cumulative_s = 0.0
                for j in range(len(path) - 1):
                    if cumulative_s <= s_start:
                        start_idx = j
                    if cumulative_s >= s_end:
                        end_idx = j + 1
                        break
                    dx = path[j+1][0] - path[j][0]
                    dy = path[j+1][1] - path[j][1]
                    cumulative_s += math.hypot(dx, dy)
                
                # 겹침 방지: start_idx가 last_end_idx보다 작으면 조정
                start_idx = max(start_idx, last_end_idx)
                
                # 이전 구간의 원본 경로 추가
                if start_idx > last_end_idx:
                    combined_path.extend(path[last_end_idx:start_idx])
                
                # 회피 세그먼트 생성
                avoidance_segment = path[start_idx:end_idx + 1]
                if len(avoidance_segment) >= 5:
                    frenet = FrenetPath(avoidance_segment)
                    avoidance_path = frenet.generate_avoidance_path(
                        s_start=0.5,
                        s_end=max(1.0, frenet.total_length - 0.5),
                        d_offset=best_d,
                        num_points=max(20, int(frenet.total_length / 0.5)),
                        smooth_ratio=0.25
                    )
                    if avoidance_path:
                        combined_path.extend(avoidance_path)
                    else:
                        combined_path.extend(avoidance_segment)
                else:
                    combined_path.extend(avoidance_segment)
                
                last_end_idx = end_idx + 1
            
            # 남은 원본 경로 추가
            if last_end_idx < len(path):
                combined_path.extend(path[last_end_idx:])
            
            if len(combined_path) < 2:
                return None
            
            return combined_path, best_d
            
        except Exception as e:
            rospy.logwarn(f"[AVOIDANCE] failed to generate avoidance: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _compute_arc_length(self, path: List[Tuple[float, float]]) -> float:
        """경로의 총 arc-length 계산"""
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total += math.hypot(dx, dy)
        return total

    def _publish_avoidance_candidates(self, role: str, original_path: List[Tuple[float, float]], stop_idx: int, obstacle_idx: int) -> None:
        """
        회피 후보 경로들을 RViz에 시각화용으로 발행
        - 정지점(stop_idx)에서 시작해서 장애물을 지나 복귀
        
        Args:
            role: 차량 역할명
            original_path: 원본 경로
            stop_idx: 정지점 인덱스 (잘린 경로 끝)
            obstacle_idx: 장애물이 있는 경로 인덱스
        """
        if len(original_path) < 10:
            return
        
        try:            
            # 회피 구간: 정지점 ~ 장애물 뒤 마진
            margin_after_idx = int(self.avoidance_margin_after / 0.5)  # m → index (0.5m 간격)
            start_idx = stop_idx
            end_idx = min(len(original_path), obstacle_idx + margin_after_idx)
            
            # 정지점부터 장애물 지난 후까지 경로 추출
            avoidance_segment = original_path[start_idx:end_idx]
            
            if len(avoidance_segment) < 5:
                return
            
            # FrenetPath 생성
            frenet = FrenetPath(avoidance_segment)
            
            # 회피 구간 (전체 구간 사용)
            s_start = 0.5
            s_end = frenet.total_length - 0.5
            
            if s_end <= s_start:
                return
            
            # 회피 후보 생성 및 평가
            candidates = []  # [(cost, d, path), ...]
            
            for d in range(-20, 40, 5):
                d /= 10
                avoidance_path = frenet.generate_avoidance_path(
                    s_start=s_start,
                    s_end=s_end,
                    d_offset=d,
                    num_points=max(20, int((s_end - s_start) / 0.5)),
                    smooth_ratio=0.3
                )
                
                if not avoidance_path or len(avoidance_path) < 2:
                    continue
                
                # 충돌 검사: 경로가 장애물과 충돌하는지 확인
                collision = False
                for px, py in avoidance_path:
                    for ox, oy, _ in self._obstacles:
                        if math.hypot(px - ox, py - oy) < self.obstacle_collision_radius:
                            collision = True
                            break
                    if collision:
                        break
                
                # 코스트 계산: |d| / 0.5 (d=0이 가장 저렴)
                d_cost = abs(d) / 0.5
                
                if not collision:
                    candidates.append((d_cost, d, avoidance_path))
                
                # 시각화용 발행
                msg = Path()
                msg.header = Header(frame_id="map", stamp=rospy.Time.now())
                for x, y in avoidance_path:
                    p = PoseStamped()
                    p.header = msg.header
                    p.pose.position.x = x
                    p.pose.position.y = y
                    p.pose.position.z = 0.0
                    msg.poses.append(p)
                
                if role in self._avoidance_path_pubs and d in self._avoidance_path_pubs[role]:
                    self._avoidance_path_pubs[role][d].publish(msg)
            
            # 최적 경로 선택 (가장 낮은 코스트)
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_cost, best_d, best_path = candidates[0]
                rospy.loginfo(f"[AVOIDANCE] {role}: best path d={best_d:.1f} (cost={best_cost:.1f}, {len(candidates)} valid)")
                # TODO: 선택된 경로를 실제 주행에 사용
            else:
                rospy.logwarn(f"[AVOIDANCE] {role}: no collision-free path found!")
            
            rospy.loginfo(f"[AVOIDANCE] {role}: avoidance paths from stop_idx={stop_idx} to {end_idx} (obstacle at {obstacle_idx})")
            
        except Exception as e:
            rospy.logwarn(f"[AVOIDANCE] {role}: failed to generate candidates: {e}")

    def _log_overlap_detection_cb(self, _evt) -> None:
        """주기적으로 반대 차선 중복 상태 로깅"""
        if self.route_planner is None or not hasattr(self.route_planner, 'find_opposite_lane_overlaps'):
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

    # ─────────────────────────────────────────────────────────────────────────
    # Avoidance Path Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_avoidance_path(
        self,
        obstacle_location,
        d_offset: float = None,
        lookahead_m: float = None,
        lookbehind_m: float = None
    ) -> Optional[List[Tuple[float, float]]]:
        """
        장애물 회피 경로 생성
        
        Args:
            obstacle_location: 장애물 위치 (carla.Location 또는 (x, y, z) 튜플)
            d_offset: 횡방향 오프셋 (None이면 기본값 사용)
            lookahead_m: 장애물 앞 거리
            lookbehind_m: 장애물 뒤 거리
            
        Returns:
            [(x, y), ...] 회피 경로 또는 None
        """
        d_offset = d_offset or self.d_offset
        lookahead_m = lookahead_m or self.lookahead_m
        lookbehind_m = lookbehind_m or self.lookbehind_m
        
        # 장애물 위치의 FrenetPath 가져오기
        if self.route_planner is None or not hasattr(self.route_planner, 'get_frenet_path_for_location'):
            return None
        frenet = self.route_planner.get_frenet_path_for_location(obstacle_location)
        if frenet is None:
            return None
        
        # 좌표 추출
        if hasattr(obstacle_location, 'x'):
            ox, oy = obstacle_location.x, obstacle_location.y
        else:
            ox, oy = obstacle_location[0], obstacle_location[1]
        
        # 장애물의 s 좌표
        s_obs, d_obs = frenet.cartesian_to_frenet(ox, oy)
        
        # 회피 방향 결정
        direction = frenet.get_avoidance_direction(ox, oy)
        if direction == 'left':
            actual_offset = -abs(d_offset)
        else:
            actual_offset = abs(d_offset)
        
        # 회피 구간
        s_start = max(0, s_obs - lookbehind_m)
        s_end = min(frenet.total_length, s_obs + lookahead_m)
        
        return frenet.generate_avoidance_path(s_start, s_end, actual_offset)

    def get_avoidance_direction(self, obstacle_location) -> Optional[str]:
        """
        장애물에 대한 회피 방향 결정
        
        Returns:
            'left', 'right', 또는 None
        """
        if self.route_planner is None or not hasattr(self.route_planner, 'get_frenet_path_for_location'):
            return None
        frenet = self.route_planner.get_frenet_path_for_location(obstacle_location)
        if frenet is None:
            return None
        
        if hasattr(obstacle_location, 'x'):
            ox, oy = obstacle_location.x, obstacle_location.y
        else:
            ox, oy = obstacle_location[0], obstacle_location[1]
        
        return frenet.get_avoidance_direction(ox, oy)

    # ─────────────────────────────────────────────────────────────────────────
    # Path Clearance Check
    # ─────────────────────────────────────────────────────────────────────────

    def is_avoidance_path_clear(
        self,
        avoidance_path: List[Tuple[float, float]],
        other_vehicle_locations: List[Tuple[float, float]] = None,
        safety_radius: float = None
    ) -> bool:
        """
        회피 경로 점유 확인
        
        TODO: 실제 구현 예정 (현재는 항상 True 반환)
        
        Args:
            avoidance_path: 회피 경로 [(x, y), ...]
            other_vehicle_locations: 다른 차량 위치 목록
            safety_radius: 안전 반경
            
        Returns:
            경로가 비어있으면 True
        """
        # TODO: 실제 점유 확인 로직 구현
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Decision Making
    # ─────────────────────────────────────────────────────────────────────────

    def should_avoid_or_stop(
        self,
        obstacle_location,
        other_vehicle_locations: List[Tuple[float, float]] = None
    ) -> str:
        """
        회피/정지 판단
        
        Returns:
            'AVOID' - 회피 경로로 진행
            'STOP' - 정지 (회피 경로 점유됨)
            'PROCEED' - 회피 없이 진행 (회피 경로 생성 불가)
        """
        # 회피 경로 생성
        avoidance_path = self.generate_avoidance_path(obstacle_location)
        
        if not avoidance_path:
            return 'PROCEED'  # 회피 경로 없음 → 그냥 진행
        
        # 회피 경로 점유 확인
        if self.is_avoidance_path_clear(avoidance_path, other_vehicle_locations):
            return 'AVOID'
        else:
            return 'STOP'

    def check_path_conflict(
        self,
        path_a: List[Tuple[float, float]],
        path_b: List[Tuple[float, float]],
        threshold: float = 3.0
    ) -> Optional[int]:
        """
        두 경로의 충돌 지점 확인
        
        Args:
            path_a: 첫 번째 경로
            path_b: 두 번째 경로
            threshold: 충돌 판정 거리
            
        Returns:
            path_a에서 충돌 발생 인덱스, 없으면 None
        """
        for i, (ax, ay) in enumerate(path_a):
            for bx, by in path_b:
                if math.hypot(ax - bx, ay - by) < threshold:
                    return i
        return None
