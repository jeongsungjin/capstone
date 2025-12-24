#!/usr/bin/env python3
"""
CARLA 토폴로지 기반 PSIPP 로드맵 생성기

사용법:
    python3 roadmap_generator.py [--spacing 0.1] [--output roadmap.json] [--visualize]

커스텀 가능한 항목:
    - waypoint_spacing: waypoint 간격 (기본값: 0.1m)
    - coord_precision: 좌표 반올림 정밀도 (기본값: 2)
    - 제외할 도로 타입
    - 수동 엣지 추가/제거
"""

import argparse
import json
import math
from typing import Dict, List, Tuple, Set, Optional

try:
    import setup_carla_path  # noqa: F401
except Exception:
    pass

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


class RoadmapGenerator:
    """CARLA 토폴로지 기반 PSIPP 로드맵 생성기"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 10.0,
        waypoint_spacing: float = 0.1,
        coord_precision: int = 2,
    ):
        """
        Args:
            host: CARLA 서버 호스트
            port: CARLA 서버 포트
            timeout: 연결 타임아웃
            waypoint_spacing: waypoint 간격 (미터)
            coord_precision: 좌표 반올림 소수점 자리수
        """
        self.waypoint_spacing = waypoint_spacing
        self.coord_precision = coord_precision
        
        # CARLA 연결
        print(f"Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        print(f"Connected! Map: {self.carla_map.name}")
        
        # 결과 저장
        self.vertices: List[Tuple[float, float]] = []
        self.edges: List[Tuple[int, int]] = []
        self.coord_to_id: Dict[Tuple[float, float], int] = {}
        
        # 제외할 도로 타입 (커스텀 가능)
        self.excluded_road_types: Set[int] = set()
        # RoadOption 값들:
        # 0: VOID, 1: LEFT, 2: RIGHT, 3: STRAIGHT, 4: LANEFOLLOW
        # 5: CHANGELANELEFT, 6: CHANGELANERIGHT
    
    def add_vertex(self, loc) -> int:
        """좌표에서 vertex ID 가져오기 (없으면 생성)"""
        key = (
            round(loc.x, self.coord_precision),
            round(loc.y, self.coord_precision)
        )
        if key not in self.coord_to_id:
            vertex_id = len(self.vertices)
            self.coord_to_id[key] = vertex_id
            self.vertices.append((loc.x, loc.y))
        return self.coord_to_id[key]
    
    def generate(self) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """
        토폴로지 기반 로드맵 생성
        
        Returns:
            (vertices, edges): 정점 좌표 리스트, 엣지 (from, to) 리스트
        """
        print(f"Building roadmap with {self.waypoint_spacing}m spacing...")
        
        # GlobalRoutePlanner 초기화
        grp = GlobalRoutePlanner(self.carla_map, self.waypoint_spacing)
        topology_graph = grp._graph
        
        print(f"Topology graph: {topology_graph.number_of_nodes()} nodes, {topology_graph.number_of_edges()} edges")
        
        # 초기화
        self.vertices = []
        self.edges = []
        self.coord_to_id = {}
        edges_set: Set[Tuple[int, int]] = set()
        
        # 토폴로지 그래프의 모든 엣지 처리
        for u, v, edge_data in topology_graph.edges(data=True):
            # 도로 타입 확인 (제외 대상이면 스킵)
            road_type = edge_data.get('type', 0)
            if road_type in self.excluded_road_types:
                continue
            
            path_waypoints = edge_data.get('path', [])
            if not path_waypoints:
                continue
            
            # 이 엣지의 모든 waypoint를 vertex로 변환
            prev_vertex_id = None
            for wp in path_waypoints:
                vertex_id = self.add_vertex(wp.transform.location)
                
                # 연속된 waypoint들을 엣지로 연결
                if prev_vertex_id is not None and prev_vertex_id != vertex_id:
                    edges_set.add((prev_vertex_id, vertex_id))
                
                prev_vertex_id = vertex_id
            
            # entry/exit waypoint 연결
            entry_wp = edge_data.get('entry_waypoint')
            exit_wp = edge_data.get('exit_waypoint')
            
            if entry_wp and path_waypoints:
                entry_vid = self.add_vertex(entry_wp.transform.location)
                first_path_vid = self.add_vertex(path_waypoints[0].transform.location)
                if entry_vid != first_path_vid:
                    edges_set.add((entry_vid, first_path_vid))
            
            if exit_wp and path_waypoints:
                last_path_vid = self.add_vertex(path_waypoints[-1].transform.location)
                exit_vid = self.add_vertex(exit_wp.transform.location)
                if last_path_vid != exit_vid:
                    edges_set.add((last_path_vid, exit_vid))
        
        self.edges = list(edges_set)
        
        print(f"Generated: {len(self.vertices)} vertices, {len(self.edges)} edges")
        return self.vertices, self.edges
    
    def add_custom_edge(self, from_x: float, from_y: float, to_x: float, to_y: float):
        """수동으로 엣지 추가 (도로 규칙 무시하고 연결해야 할 때)"""
        from_loc = type('Location', (), {'x': from_x, 'y': from_y})()
        to_loc = type('Location', (), {'x': to_x, 'y': to_y})()
        
        from_id = self.add_vertex(from_loc)
        to_id = self.add_vertex(to_loc)
        
        if from_id != to_id:
            edge = (from_id, to_id)
            if edge not in self.edges:
                self.edges.append(edge)
                print(f"Added custom edge: {from_id} -> {to_id}")
    
    def remove_edge(self, from_id: int, to_id: int):
        """엣지 제거"""
        edge = (from_id, to_id)
        if edge in self.edges:
            self.edges.remove(edge)
            print(f"Removed edge: {from_id} -> {to_id}")
    
    def find_closest_vertex(self, x: float, y: float) -> Tuple[int, float]:
        """가장 가까운 vertex 찾기"""
        min_dist = float('inf')
        closest_id = 0
        for vid, (vx, vy) in enumerate(self.vertices):
            dist = math.hypot(x - vx, y - vy)
            if dist < min_dist:
                min_dist = dist
                closest_id = vid
        return closest_id, min_dist
    
    def save(self, filepath: str):
        """로드맵을 JSON 파일로 저장"""
        data = {
            "map_name": self.carla_map.name,
            "waypoint_spacing": self.waypoint_spacing,
            "coord_precision": self.coord_precision,
            "vertices": self.vertices,
            "edges": self.edges,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved roadmap to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """JSON 파일에서 로드맵 로드"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        vertices = [tuple(v) for v in data["vertices"]]
        edges = [tuple(e) for e in data["edges"]]
        print(f"Loaded roadmap: {len(vertices)} vertices, {len(edges)} edges")
        return vertices, edges
    
    def visualize_in_carla(self, life_time: float = 60.0):
        """CARLA에서 로드맵 시각화"""
        debug = self.world.debug
        
        print(f"Visualizing roadmap for {life_time}s...")
        
        # 엣지 그리기
        for from_id, to_id in self.edges:
            v1 = self.vertices[from_id]
            v2 = self.vertices[to_id]
            debug.draw_line(
                carla.Location(x=v1[0], y=v1[1], z=0.5),
                carla.Location(x=v2[0], y=v2[1], z=0.5),
                thickness=0.05,
                color=carla.Color(0, 255, 0),
                life_time=life_time
            )
        
        # 정점 그리기 (샘플링)
        step = max(1, len(self.vertices) // 100)  # 최대 100개만 표시
        for i in range(0, len(self.vertices), step):
            v = self.vertices[i]
            debug.draw_point(
                carla.Location(x=v[0], y=v[1], z=0.5),
                size=0.1,
                color=carla.Color(255, 0, 0),
                life_time=life_time
            )
        
        print("Visualization complete!")


def main():
    parser = argparse.ArgumentParser(description="CARLA 토폴로지 기반 PSIPP 로드맵 생성기")
    parser.add_argument("--host", default="localhost", help="CARLA 서버 호스트")
    parser.add_argument("--port", type=int, default=2000, help="CARLA 서버 포트")
    parser.add_argument("--spacing", type=float, default=0.1, help="Waypoint 간격 (m)")
    parser.add_argument("--precision", type=int, default=2, help="좌표 반올림 정밀도")
    parser.add_argument("--output", type=str, default=None, help="출력 JSON 파일 경로")
    parser.add_argument("--visualize", action="store_true", help="CARLA에서 시각화")
    
    args = parser.parse_args()
    
    # 로드맵 생성
    generator = RoadmapGenerator(
        host=args.host,
        port=args.port,
        waypoint_spacing=args.spacing,
        coord_precision=args.precision,
    )
    
    vertices, edges = generator.generate()
    
    # 저장
    if args.output:
        generator.save(args.output)
    
    # 시각화
    if args.visualize:
        generator.visualize_in_carla()
    
    # 통계 출력
    print("\n=== Roadmap Statistics ===")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Spacing: {args.spacing}m")
    
    # 샘플 출력
    print("\n=== Sample Vertices (first 5) ===")
    for i, v in enumerate(vertices[:5]):
        print(f"  [{i}] ({v[0]:.2f}, {v[1]:.2f})")
    
    print("\n=== Sample Edges (first 5) ===")
    for e in edges[:5]:
        print(f"  {e[0]} -> {e[1]}")


if __name__ == "__main__":
    main()
