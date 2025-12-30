#!/usr/bin/env python3
"""
GlobalPlanner: CARLA GlobalRoutePlanner 상속 클래스
- 장애물 위치의 노드를 제외하고 경로 탐색
- 반대 차선 맵 관리
"""

import math
import os
import json
import re
from typing import Set, List, Dict, Tuple, Optional

import numpy as np
import networkx as nx
import setup_carla_path  # noqa: F401
import carla
import rospkg

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

    class RawPathPoint:
        """도로 스냅/웨이포인트 변환 없이, 주어진 좌표 그대로 넘기는 최소 래퍼."""

        def __init__(self, x: float, y: float, z: float, yaw_rad: float):
            self.transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(yaw=math.degrees(yaw_rad)),
            )
            # 도로/차선 정보 없음
            self.road_id = 0
            self.lane_id = 0
            self.lane_type = carla.LaneType.Any
            self.is_junction = False

    def __init__(self, wmap, sampling_resolution: float = 1.0):
        """
        Args:
            wmap: CARLA Map 객체
            sampling_resolution: 경로 샘플링 해상도 (미터)
        """
        super().__init__(wmap, sampling_resolution)
        self._map = wmap  # carla.Map reference for custom path snapping/logging
        self._blocked_nodes: Set[int] = set()
        self._blocked_edges: Set[Tuple[int, int]] = set()
        self._obstacle_locations: List[Tuple[float, float, float]] = []

        self._custom_blocked_edge: Tuple[Tuple[int, int]] = (
            (26, 15),
            (16, 22),
            (13, 19),
            (20, 25)
        )

        for u, v in self._custom_blocked_edge:
            self._graph.remove_edge(u, v)

        ### 커스텀 Path 반영하기 ###
        try:
            map_name = getattr(self._map, "name", "unknown")
        except Exception:
            map_name = "unknown"
        rospy.loginfo(f"[GlobalPlanner] init: map={map_name}, sampling_res={sampling_resolution}")
        self._load_custom_edge_paths()

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

    def _load_custom_edge_paths(self) -> None:
        """
        perception/scripts/custom_paths/custom_path_{u}to{v}.json 읽어 edge path 교체
        포맷: {"points": [[x,y], ...]}  (추가 필드는 무시)
        - entry/exit waypoint에 가장 가까운 구간만 슬라이스하여 사용
        - 스냅 실패 시 해당 포인트는 제외, usable waypoint 2개 미만이면 스킵
        - 거리/실패 카운트를 INFO/WARN으로 로깅
        """
        try:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("perception")
        except Exception as exc:
            rospy.logwarn(f"[GlobalPlanner] rospack get_path(perception) failed: {exc}")
            return

        custom_dir = os.path.join(pkg_path, "scripts", "custom_paths")
        if not os.path.isdir(custom_dir):
            rospy.loginfo(f"[GlobalPlanner] custom path dir not found: {custom_dir}")
            return

        files = [f for f in os.listdir(custom_dir) if f.startswith("custom_path_") and f.endswith(".json")]
        if not files:
            rospy.loginfo("[GlobalPlanner] no custom_path_*.json found")
            return

        pattern = re.compile(r"custom_path_(\d+)to(\d+)\.json")
        applied = 0
        rospy.loginfo(f"[GlobalPlanner] custom path load start (map={getattr(self._map, 'name', 'unknown')}, dir={custom_dir})")
        for fname in files:
            m = pattern.match(fname)
            if not m:
                continue
            u = int(m.group(1))
            v = int(m.group(2))
            if not self._graph.has_edge(u, v):
                rospy.logwarn(f"[GlobalPlanner] edge ({u},{v}) not in graph; skip {fname}")
                continue
            full_path = os.path.join(custom_dir, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                rospy.logwarn(f"[GlobalPlanner] failed to load {fname}: {exc}")
                continue

            if not isinstance(data, dict) or "points" not in data or not isinstance(data["points"], list):
                rospy.logwarn(f"[GlobalPlanner] custom path {fname} invalid format (need dict with 'points')")
                continue

            points_raw = data["points"]
            if points_raw is None or len(points_raw) < 2:
                rospy.logwarn(f"[GlobalPlanner] custom path {fname} invalid or too short")
                continue

            try:
                entry_wp = self._graph[u][v]["entry_waypoint"]
                exit_wp = self._graph[u][v]["exit_waypoint"]
            except Exception:
                rospy.logwarn(f"[GlobalPlanner] edge ({u},{v}) missing waypoints; skip")
                continue

            entry_loc = entry_wp.transform.location
            exit_loc = exit_wp.transform.location

            # 포인트 정리 (dict 또는 [x,y])
            clean_pts: List[Tuple[float, float]] = []
            for pt in points_raw:
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    clean_pts.append((float(pt["x"]), float(pt["y"])))
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    clean_pts.append((float(pt[0]), float(pt[1])))
            if len(clean_pts) < 2:
                rospy.logwarn(f"[GlobalPlanner] custom path {fname} invalid after cleaning")
                continue

            # 엔트리/익시트에 가장 가까운 구간으로 슬라이스
            def _dist_sq(pt, loc):
                return (pt[0] - loc.x) ** 2 + (pt[1] - loc.y) ** 2

            entry_idx = min(range(len(clean_pts)), key=lambda i: _dist_sq(clean_pts[i], entry_loc))
            exit_idx = min(range(len(clean_pts)), key=lambda i: _dist_sq(clean_pts[i], exit_loc))
            if entry_idx <= exit_idx:
                segment = clean_pts[entry_idx: exit_idx + 1]
            else:
                segment = list(reversed(clean_pts[exit_idx: entry_idx + 1]))
            if len(segment) < 2:
                rospy.logwarn(
                    f"[GlobalPlanner] custom path {fname} trimmed too short (entry_idx={entry_idx}, exit_idx={exit_idx})"
                )
                continue

            # 엔트리/익시트 좌표로 첫/마지막 점을 덮어쓰기
            # 첫/마지막 점을 엔트리/익시트 좌표로 대체
            if segment:
                segment[0] = (entry_loc.x, entry_loc.y)
                segment[-1] = (exit_loc.x, exit_loc.y)

            # 아크길이 기반 리샘플링으로 매끈하게 (도로 스냅 없음)
            resampled = []
            try:
                seg_arr = np.array(segment, dtype=float)
                deltas = np.diff(seg_arr, axis=0)
                seg_len = np.sqrt((deltas ** 2).sum(axis=1))
                s = np.concatenate(([0.0], np.cumsum(seg_len)))
                total = s[-1] if len(s) > 0 else 0.0
                if total > 0.0:
                    t = np.linspace(0.0, total, num=len(segment))
                    # 균일 간격으로 다시 보간 (포인트 수는 원본과 동일)
                    xs = np.interp(t, s, seg_arr[:, 0])
                    ys = np.interp(t, s, seg_arr[:, 1])
                    resampled = list(zip(xs, ys))
            except Exception:
                resampled = segment
            if not resampled:
                resampled = segment

            custom_wps: List[carla.Waypoint] = []
            for idx_pt, (x, y) in enumerate(resampled):
                # yaw 근사: 다음 점 기준, 마지막은 이전 점
                if idx_pt < len(resampled) - 1:
                    nx_pt = resampled[idx_pt + 1]
                else:
                    nx_pt = resampled[idx_pt - 1]
                dx = nx_pt[0] - x
                dy = nx_pt[1] - y
                yaw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else 0.0
                # 도로 스냅 전혀 안 함: raw 좌표 그대로
                custom_wps.append(self.RawPathPoint(x, y, entry_loc.z, yaw))

            if len(custom_wps) < 2:
                rospy.logwarn(
                    f"[GlobalPlanner] custom path {fname} produced insufficient usable points; skip "
                    f"(points={len(resampled)})"
                )
                continue

            self._graph[u][v]["path"] = [entry_wp] + custom_wps + [exit_wp]
            applied += 1
            rospy.loginfo(
                f"[GlobalPlanner] applied custom path {fname} (len_raw={len(points_raw)}, "
                f"seg_len={len(resampled)}, points={len(custom_wps)})"
            )

        rospy.loginfo(f"[GlobalPlanner] applied {applied} custom edge paths from {custom_dir}")


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
            if edge not in self._blocked_edges:
                u, v = edge
                self._graph[u][v]['length'] ^= (2 << 8)
            
            self._blocked_edges.add(edge)
        
        return blocked

    def clear_obstacles(self) -> None:
        """모든 장애물 및 차단 노드 제거"""
        self._blocked_nodes.clear()
        for u, v in self._blocked_edges:
            self._graph[u][v]['length'] ^= (2 << 8)
        
        self._blocked_edges.clear()
        self._obstacle_locations.clear()

    def mark_edge_blocked(self, edge: Tuple[int, int]) -> None:
        """엣지를 차단 상태로 마킹"""    
        if edge not in self._blocked_edges:
            u, v = edge
            self._graph[u][v]['length'] ^= (2 << 8)

        self._blocked_edges.add(edge)
    
    def is_edge_blocked(self, node1: int, node2: int) -> bool:
        """특정 엣지가 장애물로 차단되었는지 확인"""
        if (node1, node2) in self._blocked_edges:
            return True

        if node1 in self._blocked_nodes or node2 in self._blocked_nodes:
            return True
        
        return False

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
            start, end = self._localize(origin), self._localize(destination)

            route = nx.astar_path(
                self._graph, source=start[0], target=end[0],
                heuristic=self._distance_heuristic, weight='length')
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
            return route, None
            
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
