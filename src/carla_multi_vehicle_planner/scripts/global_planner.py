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
        rospy.loginfo('악!!!!!!!!!!!!!!!')

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
