#!/usr/bin/env python3
"""
ObstaclePlanner: 장애물 회피 전담 플래너

기능:
- 회피 경로 생성 (FrenetPath 활용)
- 경로 점유 확인
- 회피/정지 판단
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict

import setup_carla_path  # noqa: F401
import carla

import rospy
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path

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

        self.obstacle_radius = float(rospy.get_param("~obstacle_radius", 1.25))  # 경로 차단용 반경
        self.obstacle_collision_radius = float(rospy.get_param("~obstacle_collision_radius", 1.25))  # 회피 경로 충돌 검사용 반경
        self.obstacle_stop_distance = float(rospy.get_param("~obstacle_stop_distance", 1.0))  # 장애물 앞 정지 거리
        self.avoidance_margin_after = float(rospy.get_param("~avoidance_margin_after", 1.0))  # 장애물 뒤 복귀 마진

        # 회피 d_offset 범위 (0.5 간격으로 탐색)
        self.d_offset_scale = 10
        self.min_d_offset = int(self.d_offset_scale * float(rospy.get_param("~min_d_offset", -1.5)))   # 왼쪽 최대
        self.max_d_offset = int(self.d_offset_scale * float(rospy.get_param("~max_d_offset", 4.0)))    # 오른쪽 최대
        self.d_offset_step = int(self.d_offset_scale * float(rospy.get_param("~d_offset_step", 0.5)))  # 탐색 간격
        self.lane_width = float(rospy.get_param("~lane_width", 4.0))                              # 차선 폭
        self.obstacle_padding = 0 * float(rospy.get_param("~obstacle_padding", 0.25))                 # 장애물 여유 간격
        self.vehicle_padding  = 0 * float(rospy.get_param("~vehicle_padding", 0.25))                   # 차량 여유 간격

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
        self._obstacle_blocked_roles: Dict[str, carla.Location] = {}
        
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
        
        self.is_obstacle_list_changed = self._obstacle_changed(new_obstacles)
        
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

    def _obstacle_changed(self, new_obstacles):
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

    def apply_avoidance_to_path(
        self, 
        role: str, 
        path: List[Tuple[float, float]],
        obstacles_on_path: List[Tuple[Tuple[float, float], int]],
    ) -> List[Tuple[float, float]]:
        if not path or len(path) < 10:
            return path, [(0, 0)]
        
        frenet_path = FrenetPath(path)

        # 1. 경로에서 장애물 찾기
        if not obstacles_on_path:
            if role in self._obstacle_blocked_roles:
                self._obstacle_blocked_roles.pop(role)

            return path, [(0, 0)]
        
        rospy.loginfo(f"[AVOIDANCE] {role}: obstacle detected {len(obstacles_on_path)} on path")
        
        # 2. 회피 경로 생성 시도
        combined_path, best_d_offsets, stop_poses, need_stop_idx = self._generate_avoidance_segment(frenet_path, obstacles_on_path)
        rospy.loginfo(f"[AVOIDANCE] {role}: avoidance applied with d={best_d_offsets}")
        if need_stop_idx != -1:
            self._obstacle_blocked_roles[role] = stop_poses[need_stop_idx]
            rospy.logfatal(f"[MUST STOP] {role}: CANNOT AVOID OBSTACLE at ({stop_poses[need_stop_idx].x:.1f}, {stop_poses[need_stop_idx].y:.1f}) - VEHICLE MUST STOP!")

        return combined_path, stop_poses
    
    def _find_obstacle_on_path(self, frenet_path, is_frenet=True) -> List[Tuple[Tuple[float, float], int]]:
        """
        경로에서 장애물 찾기
        
        Returns:
            [(obstacle_pos, path_idx), ...]
        """
        ret = []
        if not self._obstacles:
            return ret
        
        if not is_frenet:
            frenet_path = FrenetPath(frenet_path)

        for ox, oy, _ in self._obstacles:
            # TODO : 회전 교차로 같은 중복으로 겹칠 수 있는 영역에 대한 로직을 구현해야 함.
            # 아직 어떻게 할지 몰라서 일단 한 개만 처리하도록 구현함.
            collision_indices = frenet_path.get_multiple_s_values(ox, oy, self.obstacle_radius, self.lane_width)
            if len(collision_indices) > 0:
                s_obs, d_obs = frenet_path.cartesian_to_frenet(ox, oy)
                ret.append((collision_indices[0], (s_obs, d_obs), (ox, oy)))
        
        return ret

    def _generate_avoidance_segment(self, frenet_path, obstacles):
        try:
            best_d_offsets, stop_poses, need_stop_idx = [], [], -1
            for idx, (path_idx, (s_obs, d_obs), (ox, oy)) in enumerate(obstacles):
                # 횡방향 경로 계획
                safe_d_offsets = []
                for d_offset in range(self.min_d_offset, self.max_d_offset, self.d_offset_step):
                    d_offset /= self.d_offset_scale
                    
                    avoid_x, avoid_y = frenet_path.frenet_to_cartesian(frenet_path._s_profile[path_idx], d_offset)
                    dist = np.hypot(avoid_x - ox, avoid_y - oy)

                    if dist > self.obstacle_radius + self.lane_width / 2 + self.obstacle_padding + self.vehicle_padding:
                        safe_d_offsets.append(d_offset)

                safe_d_offsets = np.array(safe_d_offsets)
                if len(safe_d_offsets) == 0:
                    rospy.logfatal(f'[AVOIDANCE] failed to find safe d-offset for obstacle at ({ox:.1f}, {oy:.1f})')
                    need_stop_idx = idx
                    break

                best_d_offset = safe_d_offsets[np.abs(safe_d_offsets).argmin()]
                best_d_offsets.append(best_d_offset)

                # 종방향 경로 계획
                wheelbase = 1.74
                delta_max_rad = math.radians(17.5)
                r_min = wheelbase / math.tan(delta_max_rad)
    
                look_ahead = math.sqrt(2 * r_min * abs(best_d_offset))
                look_behind = wheelbase * 3

                s_start = max(0.5, s_obs - (self.obstacle_radius + look_ahead))
                stop_pos = frenet_path.frenet_to_cartesian(s_start, 0)
                stop_poses.append(carla.Location(
                    x=stop_pos[0],
                    y=stop_pos[1]
                ))
            
                s_end = min(frenet_path._s_profile[-1] - 0.5, s_obs + (self.obstacle_radius + look_behind))

                rospy.loginfo(f'[AVOIDANCE] s_start: {s_start:.1f}, s_end: {s_end:.1f}, d_offset: {best_d_offset:.1f} total_length: {frenet_path.total_length:.1f}')

                s_start_idx = np.searchsorted(frenet_path._s_profile, s_start)
                s_end_idx = np.searchsorted(frenet_path._s_profile, s_end)

                frenet_path.update_d_offset(s_start_idx, s_end_idx, best_d_offset)

            avoidance_path = frenet_path.generate_avoidance_path()
            return avoidance_path, best_d_offsets, stop_poses, need_stop_idx

        # TODO: 여기서 None 이 반환되는 건에 대해 이후 로직에 대한 처리하 한 곳도 되어 있지 않음!!
        except Exception as e:
            rospy.logwarn(f"[AVOIDANCE] failed to generate avoidance: {e}")
            import traceback
            traceback.print_exc()
            return None, [], [], -1

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

    def _log_overlap_detection_cb(self, _evt) -> None:
        """주기적으로 반대 차선 중복 상태 로깅 (현재 비활성화)"""
        pass

