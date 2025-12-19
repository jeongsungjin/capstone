#!/usr/bin/env python3
"""
ObstaclePlanner: 장애물 회피 전담 플래너

기능:
- 회피 경로 생성 (FrenetPath 활용)
- 경로 점유 확인
- 회피/정지 판단
"""

import math
from typing import List, Tuple, Optional

import setup_carla_path  # noqa: F401
import carla

from frenet_path import FrenetPath


class ObstaclePlanner:
    """
    장애물 회피 전담 플래너.
    
    GlobalPlanner와 함께 사용하여 장애물 회피 경로 생성 및 판단.
    """

    def __init__(self, global_planner):
        """
        Args:
            global_planner: GlobalPlanner 인스턴스
        """
        self._global_planner = global_planner
        
        # 회피 파라미터
        self.d_offset = 3.5  # 횡방향 오프셋 (차선 폭의 절반 정도)
        self.lookahead_m = 15.0  # 장애물 앞 거리
        self.lookbehind_m = 5.0  # 장애물 뒤 거리
        self.safety_radius = 2.0  # 점유 확인 시 안전 반경

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
        frenet = self._global_planner.get_frenet_path_for_location(obstacle_location)
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
        frenet = self._global_planner.get_frenet_path_for_location(obstacle_location)
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
