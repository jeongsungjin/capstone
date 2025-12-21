#!/usr/bin/env python3
"""
전체 맵 Waypoint 시각화 + FrenetPath Offset

1. 전체 맵의 모든 waypoint를 선으로 연결
2. 각 엣지에 대해 d=-2~+2 offset 시각화
"""

import sys
import matplotlib.pyplot as plt

# CARLA Python API 경로 추가
carla_path = "/home/ctrl/carla/PythonAPI/carla/build/lib.linux-x86_64-cpython-38"
if carla_path not in sys.path:
    sys.path.insert(0, carla_path)

script_path = "/home/ctrl/capstone/src/carla_multi_vehicle_planner/scripts"
if script_path not in sys.path:
    sys.path.insert(0, script_path)

import carla
from frenet_path import FrenetPath


def main():
    # CARLA 연결
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    carla_map = world.get_map()
    print(f"Connected: {carla_map.name}")
    
    # 전체 맵 topology 가져오기
    topology = carla_map.get_topology()
    print(f"Total edges in map: {len(topology)}")
    
    # 시각화
    fig, ax = plt.subplots(figsize=(16, 16))
    
    colors = {-1.0: 'darkblue', -0.5: 'blue', 0: 'black', 0.5: 'red', 1.0: 'darkred'}
    
    all_lane_widths = []
    edge_count = 0
    
    for wp_start, wp_end in topology:
        # 시작점에서 끝점까지 waypoint 샘플링
        current_wp = wp_start
        segment_wps = [current_wp]
        
        for _ in range(200):  # 최대 200개
            next_wps = current_wp.next(0.5)
            if not next_wps:
                break
            current_wp = next_wps[0]
            segment_wps.append(current_wp)
            
            # 끝점 근처에 도달하면 종료
            dist_to_end = current_wp.transform.location.distance(wp_end.transform.location)
            if dist_to_end < 1.0:
                break
        
        if len(segment_wps) < 3:
            continue
        
        edge_count += 1
        all_lane_widths.extend([wp.lane_width for wp in segment_wps])
        
        # 좌표 추출
        coords = [(wp.transform.location.x, wp.transform.location.y) for wp in segment_wps]
        
        # FrenetPath 생성
        try:
            frenet = FrenetPath(coords)
            
            # d=-2, -1, 0, +1, +2 offset 경로 그리기
            for d in [-1.0, -0.5, 0, 0.5, 1.0]:
                offset_path = frenet.generate_lane_offset_path(d, num_points=len(coords))
                if offset_path and len(offset_path) > 1:
                    xs, ys = zip(*offset_path)
                    ax.plot(xs, ys, color=colors[d], linewidth=0.5 if d != 0 else 1.0, alpha=0.6)
                    
                    # 중심 경로(d=0)에만 화살표 추가 (중간 지점)
                    if d == 0 and len(offset_path) > 10:
                        mid = len(offset_path) // 2
                        dx = offset_path[mid+1][0] - offset_path[mid][0]
                        dy = offset_path[mid+1][1] - offset_path[mid][1]
                        ax.annotate('', xy=(offset_path[mid+1][0], offset_path[mid+1][1]),
                                    xytext=(offset_path[mid][0], offset_path[mid][1]),
                                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        except Exception as e:
            # FrenetPath 생성 실패 시 원본만 그리기
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color='gray', linewidth=0.5, alpha=0.3)
    
    print(f"Processed edges: {edge_count}")
    
    # 도로폭 통계
    if all_lane_widths:
        min_w = min(all_lane_widths)
        max_w = max(all_lane_widths)
        avg_w = sum(all_lane_widths) / len(all_lane_widths)
        print(f"Lane width: min={min_w:.2f}m, max={max_w:.2f}m, avg={avg_w:.2f}m")
    
    # 스폰 포인트 표시
    spawn_points = carla_map.get_spawn_points()
    sp_xs = [sp.location.x for sp in spawn_points]
    sp_ys = [sp.location.y for sp in spawn_points]
    ax.scatter(sp_xs, sp_ys, c='green', s=80, marker='*', label='Spawn Points', zorder=10)
    
    # 범례 추가
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color='darkblue', lw=2, label='d=-2m'),
        Line2D([0], [0], color='blue', lw=2, label='d=-1m'),
        Line2D([0], [0], color='black', lw=2, label='d=0 (center)'),
        Line2D([0], [0], color='red', lw=2, label='d=+1m'),
        Line2D([0], [0], color='darkred', lw=2, label='d=+2m'),
    ]
    ax.legend(handles=legend_lines, loc='upper right')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Full Map with FrenetPath Offsets: {carla_map.name}\n({edge_count} edges, d=-2 to +2)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
