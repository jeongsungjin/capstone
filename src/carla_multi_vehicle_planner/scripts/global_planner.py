#!/usr/bin/env python3
"""
GlobalPlanner: CARLA GlobalRoutePlanner 상속 클래스
- 장애물 위치의 노드를 제외하고 경로 탐색
- 반대 차선 맵 관리
"""

import math
from typing import Set, List, Tuple, Optional

import networkx as nx
import setup_carla_path  # noqa: F401
import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner

import rospy

class GlobalPlanner(GlobalRoutePlanner):
    """
    CARLA GlobalRoutePlanner 확장.
    
    기능:
    - 장애물 위치를 노드로 변환하여 차단
    - _path_search 오버라이드로 차단 노드 제외 탐색
    - 반대 차선 맵 관리
    - FrenetPath 캐시
    """

    def __init__(self, wmap, sampling_resolution: float = 1.0):
        """
        Args:
            wmap: CARLA Map 객체
            sampling_resolution: 경로 샘플링 해상도 (미터)
        """
        super().__init__(wmap, sampling_resolution)
        self._blocked_nodes: Set[int] = set()
        self._blocked_edges: Set[Tuple[int, int]] = set()
        self._obstacle_locations: List[Tuple[float, float, float]] = []

        self.opposite_lane_edge: Dict[Tuple[int, int], Tuple[int, int]] = {
            (24, 23): (25, 26), (25, 26): (24, 23),
            (18, 21): (22, 6), (22, 6): (18, 21),
            (19, 5): (17, 20), (17, 20): (19, 5),
            (15, 0): (3, 16), (3, 16): (15, 0),
            (11, 9): (4, 8), (4, 8): (11, 9),
            (12, 10): (7, 13), (7, 13): (12, 10),
            (10, 11): (8, 7), (8, 7): (10, 11),
            (6, 7): (10, 18), (10, 18): (6, 7),
            (8, 18): (6, 11), (6, 11): (8, 18),
            (16, 22): (21, 15), (21, 15): (16, 22),
            (26, 15): (16, 24), (16, 24): (26, 15),
            (26, 22): (21, 24), (21, 24): (26, 22),
            (20, 25): (23, 19), (23, 19): (20, 25),
            (20, 12): (13, 19), (13, 19): (20, 12),
            (13, 25): (23, 12), (23, 12): (13, 25)
            
            # 회전 교차로
            # (5, 3),  
            # (0, 1), (1, 14), (14, 2), (2, 17),
            # (2, 4), (9, 1), 
            # (5, 1), (2, 3), (2, 1) 
        }

        self.lane_direction: Dict[Tuple[int, int], float] = {}
        self._cache_edge_directions()

    def _cache_edge_directions(self) -> None:
        """모든 엣지의 진행 방향(yaw)을 미리 계산하여 캐싱"""
        skipped = []
        for n1, n2, data in self._graph.edges(data=True):
            entry_wp = data.get('entry_waypoint')
            exit_wp = data.get('exit_waypoint')
            if entry_wp and exit_wp:
                dx = exit_wp.transform.location.x - entry_wp.transform.location.x + 1e-6
                dy = exit_wp.transform.location.y - entry_wp.transform.location.y + 1e-6
                yaw = math.atan2(dy, dx)
                self.lane_direction[(n1, n2)] = yaw
            else:
                skipped.append((n1, n2))
        
        rospy.loginfo(f'[GlobalPlanner] edges : {self.lane_direction.items()}')
        rospy.loginfo(f"[GlobalPlanner] Cached {len(self.lane_direction)} edge directions, skipped {len(skipped)}: {skipped}")

    def get_edge_direction(self, edge: Tuple[int, int]) -> Optional[float]:
        """엣지의 진행 방향(yaw) 반환 (캐시됨)"""
        return self.lane_direction.get(edge)

    def is_opposite_direction(self, edge_a: Tuple[int, int], edge_b: Tuple[int, int]) -> bool:
        """
        두 엣지가 반대 방향인지 확인 (각도 차이 > 90도)
        
        Returns:
            True if angle difference > 90 degrees (opposite direction)
        """
        # 특수 케이스: 이 엣지들끼리는 같은 차선으로 간주 (반대 방향 아님 -> 정지)
        special_edges = [(2, 4), (4, 8), (8, 7)]
        if edge_a in special_edges and edge_b in special_edges:
            return False

        dir_a = self.get_edge_direction(edge_a)
        dir_b = self.get_edge_direction(edge_b)
        if dir_a is None or dir_b is None:
            return False
        
        diff = abs(dir_a - dir_b)
        while diff > math.pi:
            diff = abs(diff - 2 * math.pi)
        
        return diff > math.pi / 180 * 85  # 90도 초과면 반대 방향

    def add_obstacle(self, location, radius: float = 5.0) -> int:
        """
        장애물 위치 추가 → 근처 노드 차단
        
        Args:
            location: carla.Location 또는 (x, y, z) 튜플
            radius: 차단 반경 (미터)
            
        Returns:
            차단된 노드 수
        """
        if hasattr(location, 'x'):
            ox, oy, oz = location.x, location.y, location.z
        else:
            ox, oy, oz = location[0], location[1], location[2] if len(location) > 2 else 0.0
        
        self._obstacle_locations.append((ox, oy, oz))
        
        # 근처 노드 찾아서 차단
        blocked_count = 0
        for xyz, node_id in self._id_map.items():
            dist = math.hypot(xyz[0] - ox, xyz[1] - oy)
            if dist <= radius:
                self._blocked_nodes.add(node_id)
                blocked_count += 1
        
        return blocked_count

    def add_obstacle_on_road(self, location, radius: float = 5.0) -> int:
        """
        도로 위 장애물 추가 - 해당 도로의 엣지도 차단
        
        Args:
            location: carla.Location
            radius: 차단 반경
            
        Returns:
            차단된 노드 수
        """
        blocked = self.add_obstacle(location, radius)
        
        # 해당 위치의 도로 엣지도 차단
        edge = self._localize(location)
        if edge is not None:
            self._blocked_edges.add(edge)
        
        return blocked

    def clear_obstacles(self) -> None:
        """모든 장애물 및 차단 노드 제거"""
        self._blocked_nodes.clear()
        self._blocked_edges.clear()
        self._obstacle_locations.clear()

    def mark_edge_blocked(self, edge: Tuple[int, int]) -> None:
        """엣지를 차단 상태로 마킹"""
        self._blocked_edges.add(edge)

    def is_edge_blocked(self, node1: int, node2: int) -> bool:
        """특정 엣지가 장애물로 차단되었는지 확인"""
        if (node1, node2) in self._blocked_edges:
            return True
        if node1 in self._blocked_nodes or node2 in self._blocked_nodes:
            return True
        return False

    def _blocked_weight(self, u: int, v: int, edge_data: dict) -> float:
        """
        차단 노드 포함 엣지는 무한대 가중치 반환
        A* weight 함수로 사용
        """
        if u in self._blocked_nodes or v in self._blocked_nodes:
            return 10000
        
        if (u, v) in self._blocked_edges:
            return 10000
        
        return edge_data.get('length', 1)

    def get_id_for_edge(self, edge: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if not self._graph.has_edge(edge[0], edge[1]):
            return None
    
        entry_wp = self._graph.edges[edge[0], edge[1]].get('entry_waypoint')
        if not entry_wp:
            return None

        return (entry_wp.road_id, entry_wp.lane_id)

    def get_edges_at_location(self, x: float, y: float, radius: float = 1.0) -> List[Tuple[int, int]]:
        """해당 위치를 지나는 모든 edge 반환"""
        edges = []
        loc = carla.Location(x=x, y=y, z=0.0)
        
        for n1, n2, data in self._graph.edges(data=True):
            path = data.get('path', [])
            for wp in path:
                if wp.transform.location.distance(loc) < radius:
                    edges.append((n1, n2))
                    break  # 이 edge는 찾았으니 다음 edge로
        
        return edges

    def are_edges_on_opposite_lanes(self, edge_a: Tuple[int, int], edge_b: Tuple[int, int]) -> bool:
        """
        두 엣지가 반대 차선 관계인지 확인
        
        Returns:
            True if edge_a and edge_b are on opposite lanes
        """
        id_a = self.get_id_for_edge(edge_a)
        id_b = self.get_id_for_edge(edge_b)
        
        if id_a is None or id_b is None:
            return False
        
        road_a, lane_a = id_a
        road_b, lane_b = id_b
        
        # 같은 road_id + lane_id 부호가 반대 = 반대 차선
        return (road_a == road_b) and (lane_a * lane_b < 0)

    def _path_search(self, origin, destination) -> Optional[List[int]]:
        """
        차단 노드를 제외하고 A* 경로 탐색 (오버라이드)
        - Custom weight function으로 차단 처리 (그래프 복사 없음!)
        
        Returns:
            경로 노드 ID 리스트, 또는 경로 없으면 None
        """

        start = self._localize(origin)
        end = self._localize(destination)
        
        if start is None or end is None:
            return None
        
        # 시작/끝 노드가 차단되었으면 경로 없음
        if start[0] in self._blocked_nodes or end[0] in self._blocked_nodes:
            return None
        
        try:
            # Custom weight function으로 차단 노드 우회
            route = nx.astar_path(
                self._graph,
                source=start[0],
                target=end[0],
                heuristic=self._distance_heuristic,
                weight=self._blocked_weight  # 함수 전달!
            )
            route.append(end[1])
            return route

        except nx.NetworkXNoPath:
            return None

        except nx.NodeNotFound:
            return None

    def trace_route_with_nodes(self, origin, destination):
        """
        경로 탐색 + A* 노드 리스트 반환
        
        Returns:
            (route, node_list) 또는 (None, None)
            - route: [(waypoint, RoadOption), ...]
            - node_list: [node_id, ...] A* 경로의 노드 순서
        """
        try:
            node_list = self._path_search(origin, destination)
            if node_list is None:
                return None, None
            route = self.trace_route(origin, destination)
            return route, node_list
            
        except Exception:
            return None, None

    def nodes_to_edges(self, node_list: List[int]) -> List[Tuple[int, int]]:
        """
        노드 리스트를 엣지 리스트로 변환
        
        Args:
            node_list: [n0, n1, n2, ...] A* 경로 노드 순서
            
        Returns:
            [(n0, n1), (n1, n2), ...] 엣지 리스트
        """
        if not node_list or len(node_list) < 2:
            return []
        
        return [(node_list[i], node_list[i + 1]) for i in range(len(node_list) - 1)]
