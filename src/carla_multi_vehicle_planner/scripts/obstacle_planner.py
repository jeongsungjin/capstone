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

        self.obstacle_radius = float(rospy.get_param("~obstacle_radius", 0.8))  # 경로 차단용 반경
        self.obstacle_collision_radius = float(rospy.get_param("~obstacle_collision_radius", 0.8))  # 회피 경로 충돌 검사용 반경
        self.obstacle_stop_distance = float(rospy.get_param("~obstacle_stop_distance", 1.0))  # 장애물 앞 정지 거리
        self.avoidance_margin_after = float(rospy.get_param("~avoidance_margin_after", 1.0))  # 장애물 뒤 복귀 마진

        # 회피 d_offset 범위 (0.5 간격으로 탐색)
        self.d_offset_scale = 10
        self.min_d_offset = int(self.d_offset_scale * float(rospy.get_param("~min_d_offset", 0.0)))   # 왼쪽 최대
        self.max_d_offset = int(self.d_offset_scale * float(rospy.get_param("~max_d_offset", 4.0)))    # 오른쪽 최대
        self.d_offset_step = int(self.d_offset_scale * float(rospy.get_param("~d_offset_step", 0.1)))  # 탐색 간격
        self.lane_width = float(rospy.get_param("~lane_width", 4.0))                              # 차선 폭
        self.obstacle_padding = 0 * float(rospy.get_param("~obstacle_padding", 0.25))                 # 장애물 여유 간격
        self.vehicle_padding  = 0 * float(rospy.get_param("~vehicle_padding", 0.25))                   # 차량 여유 간격

        self.num_vehicles = num_vehicles
        self._role_name = lambda index: f"ego_vehicle_{index + 1}"

        # 장애물로 인해 정지 중인 차량 추적 (role -> obstacle_pos)
        self._obstacle_blocked_roles: Dict[str, List[Tuple[carla.Location, float]]] = {}
        
        # Obstacle subscriber
        self.is_obstacle_list_changed = False
        self._obstacles: List[Tuple[float, float, float]] = []
        rospy.Subscriber("/obstacles", PoseArray, self._obstacle_cb, queue_size=1)

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

    def get_stop_pos(self, role: str) -> carla.Location:
        return self._obstacle_blocked_roles[role][0]

    def stop_done(self, role: str):
        self._obstacle_blocked_roles[role].pop(0)
        if not self._obstacle_blocked_roles[role]:
            self._obstacle_blocked_roles.pop(role)

    def apply_avoidance_to_path(
        self, 
        role: str, 
        path: List[Tuple[float, float]],
        obstacles_on_path: List[Tuple[Tuple[float, float], int]],
    ) -> Tuple[Optional[List[Tuple[float, float]]], List[float], List[float]]:
        """
        경로에 장애물 회피를 적용
        
        Returns:
            (avoidance_path, s_starts, s_ends) 또는 (None, [], []) 실패 시
        """
        if not path or len(path) < 10:
            return path, [], []
        
        try:
            frenet_path = FrenetPath(path)

            # 1. 경로에서 장애물 찾기
            if not obstacles_on_path:
                if role in self._obstacle_blocked_roles:
                    self._obstacle_blocked_roles.pop(role)

                return path, [], []
            
            rospy.loginfo(f"[AVOIDANCE] {role}: obstacle detected {len(obstacles_on_path)} on path")
            
            # 2. 회피 경로 생성 시도
            combined_path, best_d_offsets, stop_poses, s_starts, s_ends = self._generate_avoidance_segment(frenet_path, obstacles_on_path)
            
            # 생성 실패 시 원본 경로 반환
            if combined_path is None:
                rospy.logwarn(f"[AVOIDANCE] {role}: avoidance generation failed, returning original path")
                return path, [], []
            
            self._obstacle_blocked_roles[role] = stop_poses
            rospy.loginfo(f"[AVOIDANCE] {role}: avoidance applied with d={best_d_offsets}")
            
            return combined_path, s_starts, s_ends
        
        except Exception as e:
            rospy.logwarn(f"[AVOIDANCE] {role}: exception in apply_avoidance_to_path: {e}")
            import traceback
            traceback.print_exc()
            return None, [], []
    
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
            collision_indices = frenet_path.get_multiple_s_values(ox, oy, self.obstacle_radius, self.lane_width)
            for collision_idx in collision_indices:
                s_obs, d_obs = frenet_path.cartesian_to_frenet(ox, oy)
                ret.append((collision_idx, (s_obs, d_obs), (ox, oy)))
        
        return ret

    def _generate_avoidance_segment(self, frenet_path, obstacles):
        try:
            best_d_offsets, stop_poses = [], []
            s_starts, s_ends = [], []
            for (path_idx, (s_obs, d_obs), (ox, oy)) in obstacles:
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

                best_d_offset = safe_d_offsets[np.abs(safe_d_offsets).argmin()] if len(safe_d_offsets) > 0 else None
                best_d_offsets.append(best_d_offset)

                # 종방향 경로 계획
                wheelbase = 1.74
                delta_max_rad = math.radians(17.5)
                r_min = wheelbase / math.tan(delta_max_rad)
    
                look_ahead = math.sqrt(2 * r_min * abs(best_d_offset if best_d_offset else (self.max_d_offset / self.d_offset_scale)))
                look_behind = wheelbase * 6

                s_start = max(0.5, s_obs - (self.obstacle_radius + look_ahead))
                s_starts.append(s_start)
                stop_pos = frenet_path.frenet_to_cartesian(s_start, 0)

                # 회피 경로 길이 계산 (s_end - s_start)
                s_end = min(frenet_path._s_profile[-1] - 0.5, s_obs + (self.obstacle_radius + look_behind))
                avoidance_length = s_end - s_start
                rospy.logfatal(f'[AVOIDACNE] avoidance_length: {avoidance_length:.1f} {look_ahead:.1f} {look_behind:.1f}')
                
                # 연속 장애물 구간 처리를 어떻게 해야할까?
                stop_poses.append((
                    carla.Location(
                        x=stop_pos[0],
                        y=stop_pos[1]
                    ),
                    best_d_offset,
                    s_start, s_end                   
                ))
            
                s_ends.append(s_end)

                if best_d_offset:
                    s_start_idx = np.searchsorted(frenet_path._s_profile, s_start)
                    s_apex_idx = np.searchsorted(frenet_path._s_profile, s_obs)
                    s_end_idx = np.searchsorted(frenet_path._s_profile, s_end)
                    rospy.loginfo(f'[AVOIDANCE] s_start: {s_start:.1f}, s_end: {s_end:.1f}, d_offset: {best_d_offset:.1f} total_length: {frenet_path.total_length:.1f}')
                    frenet_path.update_d_offset_two_stage_quintic(s_start_idx, s_apex_idx, s_end_idx, best_d_offset)

            avoidance_path = frenet_path.generate_avoidance_path()
            return avoidance_path, best_d_offsets, stop_poses, s_starts, s_ends

        # 예외 발생 시 None, 빈 리스트 반환
        except Exception as e:
            rospy.logwarn(f"[AVOIDANCE] failed to generate avoidance: {e}")
            import traceback
            traceback.print_exc()
            return None, [], [], [], []  # 5개의 반환값 맞춤

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
