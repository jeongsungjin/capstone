#!/usr/bin/env python3
"""
GlobalPlanner: CARLA GlobalRoutePlanner 상속 클래스
- 장애물 위치의 노드를 제외하고 경로 탐색
- 반대 차선 맵 관리
"""

import math
from typing import Set, List, Tuple, Optional, Dict

import networkx as nx
import setup_carla_path  # noqa: F401
import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from frenet_path import FrenetPath


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
        
        # 엣지별 FrenetPath 캐시
        self._edge_frenet_cache: dict = {}
        
        # 반대 차선 맵: (road_id, lane_id) -> opposite (road_id, lane_id) or None
        self._opposite_lane_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        self._build_opposite_lane_map()

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

    def is_blocked(self, node_id: int) -> bool:
        """노드가 차단되었는지 확인"""
        return node_id in self._blocked_nodes

    def get_blocked_nodes(self) -> Set[int]:
        """차단된 노드 집합 반환"""
        return self._blocked_nodes.copy()

    def get_obstacle_locations(self) -> List[Tuple[float, float, float]]:
        """등록된 장애물 위치 목록 반환"""
        return self._obstacle_locations.copy()

    def get_occupied_edge(self, location) -> Optional[Tuple[int, int]]:
        """
        장애물 좌표가 점유 중인 엣지 반환 (Local Planner용)
        
        Args:
            location: carla.Location 또는 (x, y, z) 튜플
            
        Returns:
            (node_id_1, node_id_2) 엣지 튜플, 또는 None
        """
        if hasattr(location, 'x'):
            loc = location
        else:
            if carla is None:
                return None
            loc = carla.Location(x=location[0], y=location[1], 
                                 z=location[2] if len(location) > 2 else 0.0)
        
        return self._localize(loc)

    def get_occupied_edge_info(self, location) -> Optional[dict]:
        """
        장애물 좌표가 점유 중인 엣지의 상세 정보 반환
        
        Args:
            location: carla.Location 또는 (x, y, z) 튜플
            
        Returns:
            dict with keys:
                - 'edge': (node1, node2)
                - 'entry_waypoint': carla.Waypoint
                - 'exit_waypoint': carla.Waypoint
                - 'road_id': int
                - 'lane_id': int
                - 'length': float
            또는 None
        """
        edge = self.get_occupied_edge(location)
        if edge is None:
            return None
        
        n1, n2 = edge
        if not self._graph.has_edge(n1, n2):
            return None
        
        edge_data = self._graph.edges[n1, n2]
        entry_wp = edge_data.get('entry_waypoint')
        
        return {
            'edge': edge,
            'entry_waypoint': entry_wp,
            'exit_waypoint': edge_data.get('exit_waypoint'),
            'road_id': entry_wp.road_id if entry_wp else None,
            'lane_id': entry_wp.lane_id if entry_wp else None,
            'length': edge_data.get('length', 0),
            'path': edge_data.get('path', []),
        }

    def get_all_obstacle_edges(self) -> List[dict]:
        """
        등록된 모든 장애물이 점유 중인 엣지 정보 목록 반환
        
        Returns:
            List of edge info dicts (see get_occupied_edge_info)
        """
        result = []
        for ox, oy, oz in self._obstacle_locations:
            info = self.get_occupied_edge_info((ox, oy, oz))
            if info is not None:
                info['obstacle_location'] = (ox, oy, oz)
                result.append(info)
        return result

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
            return float('inf')
        if (u, v) in self._blocked_edges:
            return float('inf')
        return edge_data.get('length', 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Opposite Lane (반대 차선) Management
    # ─────────────────────────────────────────────────────────────────────────

    def _build_opposite_lane_map(self) -> None:
        """
        맵 초기화 시 반대 차선 관계 미리 빌드
        (road_id, lane_id) -> opposite (road_id, lane_id)
        """
        topology = self._wmap.get_topology()
        visited = set()
        
        for wp_start, wp_end in topology:
            key = (wp_start.road_id, wp_start.lane_id)
            if key in visited:
                continue
            visited.add(key)
            
            # 반대 차선 찾기 (lane_id 부호가 다른 차선)
            opposite_wp = self._find_opposite_lane_wp(wp_start)
            if opposite_wp:
                opposite_key = (opposite_wp.road_id, opposite_wp.lane_id)
                self._opposite_lane_map[key] = opposite_key
                # 양방향 등록
                self._opposite_lane_map[opposite_key] = key
            else:
                self._opposite_lane_map[key] = None

    def _find_opposite_lane_wp(self, wp):
        """waypoint의 반대 차선 찾기 (lane_id 부호가 다른 주행 차선)"""
        # 왼쪽 차선 확인
        left = wp.get_left_lane()
        if left and left.lane_id * wp.lane_id < 0:
            # 부호 다름 = 반대 방향
            if left.lane_type == carla.LaneType.Driving:
                return left
        
        # 오른쪽 차선 확인
        right = wp.get_right_lane()
        if right and right.lane_id * wp.lane_id < 0:
            if right.lane_type == carla.LaneType.Driving:
                return right
        
        return None

    def get_lane_key(self, location) -> Optional[Tuple[int, int]]:
        """
        위치에서 (road_id, lane_id) 반환
        
        Args:
            location: carla.Location 또는 (x, y, z) 튜플
        """
        if hasattr(location, 'x'):
            loc = location
        else:
            loc = carla.Location(x=location[0], y=location[1], 
                                 z=location[2] if len(location) > 2 else 0.0)
        
        wp = self._wmap.get_waypoint(loc, lane_type=carla.LaneType.Driving)
        if wp:
            return (wp.road_id, wp.lane_id)
        return None

    def get_opposite_lane_key(self, location) -> Optional[Tuple[int, int]]:
        """
        위치의 반대 차선 (road_id, lane_id) 반환
        
        Returns:
            (road_id, lane_id) 또는 None (반대 차선 없음)
        """
        key = self.get_lane_key(location)
        if key:
            return self._opposite_lane_map.get(key)
        return None

    def has_opposite_lane(self, location) -> bool:
        """반대 차선 존재 여부"""
        return self.get_opposite_lane_key(location) is not None

    def get_opposite_lane_waypoint(self, location):
        """
        반대 차선의 waypoint 반환
        
        Returns:
            carla.Waypoint 또는 None
        """
        if hasattr(location, 'x'):
            loc = location
        else:
            loc = carla.Location(x=location[0], y=location[1], 
                                 z=location[2] if len(location) > 2 else 0.0)
        
        wp = self._wmap.get_waypoint(loc, lane_type=carla.LaneType.Driving)
        if wp:
            return self._find_opposite_lane_wp(wp)
        return None

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

    def trace_route_safe(self, origin, destination):
        """
        안전한 경로 탐색 - 실패 시 None 반환
        
        Returns:
            [(waypoint, RoadOption), ...] 또는 None
        """
        try:
            route = self._path_search(origin, destination)
            if route is None:
                return None
            return self.trace_route(origin, destination)
        except Exception:
            return None

    def can_reach(self, origin, destination) -> bool:
        """
        목적지까지 도달 가능한지 확인 (차단 노드 고려)
        """
        route = self._path_search(origin, destination)
        return route is not None

    # ─────────────────────────────────────────────────────────────────────────
    # FrenetPath 관련 메서드
    # ─────────────────────────────────────────────────────────────────────────

    def get_frenet_path_for_edge(self, edge: Tuple[int, int]) -> Optional['FrenetPath']:
        """
        엣지에 대한 FrenetPath 반환 (캐싱)
        
        Args:
            edge: (node1, node2) 엣지 튜플
            
        Returns:
            FrenetPath 객체 또는 None
        """
        # 캐시 확인
        if edge in self._edge_frenet_cache:
            return self._edge_frenet_cache[edge]
        
        # 엣지의 waypoint path 추출
        n1, n2 = edge
        if not self._graph.has_edge(n1, n2):
            return None
        
        edge_data = self._graph.edges[n1, n2]
        path_wps = edge_data.get('path', [])
        entry_wp = edge_data.get('entry_waypoint')
        exit_wp = edge_data.get('exit_waypoint')
        
        # waypoint → (x, y) 리스트
        points = []
        if entry_wp:
            loc = entry_wp.transform.location
            points.append((loc.x, loc.y))
        for wp in path_wps:
            loc = wp.transform.location
            points.append((loc.x, loc.y))
        if exit_wp:
            loc = exit_wp.transform.location
            points.append((loc.x, loc.y))
        
        if len(points) < 2:
            return None
        
        # FrenetPath 생성 및 캐싱
        frenet = FrenetPath(points)
        self._edge_frenet_cache[edge] = frenet
        return frenet

    def get_frenet_path_for_location(self, location) -> Optional['FrenetPath']:
        """
        위치가 속한 엣지의 FrenetPath 반환
        
        Args:
            location: carla.Location 또는 (x, y, z) 튜플
        """
        edge = self.get_occupied_edge(location)
        if edge is None:
            return None
        return self.get_frenet_path_for_edge(edge)

    def clear_frenet_cache(self) -> None:
        """FrenetPath 캐시 초기화"""
        self._edge_frenet_cache.clear()

