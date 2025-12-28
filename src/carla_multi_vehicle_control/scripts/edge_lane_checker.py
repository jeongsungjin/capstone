#!/usr/bin/env python3
"""
엣지 (n1, n2) 튜플 리스트를 받아서 (road_id, lane_id) 출력하는 유틸리티

사용 예시:
    python3 edge_lane_checker.py

또는 Python에서:
    from edge_lane_checker import get_lane_ids_for_edges
    results = get_lane_ids_for_edges([(n1, n2), (n1, n2), ...])
"""

import sys
import os

# CARLA path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../carla_multi_vehicle_planner/scripts'))

try:
    import setup_carla_path  # noqa: F401
    import carla
except ImportError:
    print("CARLA Python API not found. Please ensure it's installed.")
    sys.exit(1)

try:
    from global_planner import GlobalPlanner
except ImportError as e:
    print(f"GlobalPlanner not found: {e}")
    sys.exit(1)


def get_lane_ids_for_edges(edges: list, host: str = "localhost", port: int = 2000) -> list:
    """
    엣지 (n1, n2) 튜플 리스트를 받아서 (road_id, lane_id) 튜플 리스트 반환
    
    Args:
        edges: [(n1, n2), (n1, n2), ...] 형태의 노드 ID 튜플 리스트
        host: CARLA 서버 호스트
        port: CARLA 서버 포트
    
    Returns:
        [(road_id, lane_id), ...] 또는 None (실패 시)
    """
    try:
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        world = client.get_world()
        carla_map = world.get_map()
        
        # GlobalPlanner 초기화
        planner = GlobalPlanner(carla_map, sampling_resolution=1.0)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return None
    
    results = []
    for i, edge in enumerate(edges):
        if len(edge) < 2:
            results.append((None, None))
            continue
        
        n1, n2 = int(edge[0]), int(edge[1])
        
        try:
            lane_id = planner.get_id_for_edge((n1, n2))
            if lane_id is not None:
                road_id, lid = lane_id
                results.append((n1, n2, road_id, lid))
                print(f"[{i}] edge ({n1}, {n2}) -> road_id={road_id}, lane_id={lid}")
            else:
                results.append((n1, n2, None, None))
                print(f"[{i}] edge ({n1}, {n2}) -> No lane ID found")
        except Exception as e:
            results.append((None, None))
            print(f"[{i}] edge ({n1}, {n2}) -> Error: {e}")
    
    return results


if __name__ == "__main__":
    # 테스트용 예시 엣지 (노드 ID 튜플)

    test_edges = [(a, b) for a in range(27) for b in range(27) if a != b]
    edge_ids = [edge_id for edge_id in get_lane_ids_for_edges(test_edges) if edge_id[-1] is not None]
    edge_ids.sort(key=lambda x: (x[2], x[3], x[0], x[1]))

    print(*edge_ids, sep='\n')
