#!/usr/bin/env python3
"""
FrenetPath: Reference Path 기반 Frenet 좌표계 경로 표현
- 엣지별 Frenet 경로 생성
- 장애물 회피 방향 결정 및 경로 생성
- Local Planning 지원
- scipy 스플라인 보간
"""

from typing import List, Tuple, Optional
import numpy as np

from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree


class FrenetPath:
    """
    Reference Path 기반 Frenet Frame 경로
    
    Frenet 좌표계:
    - s: 경로 따라 누적 거리 (arc-length)
    - d: 경로에서 횡방향 오프셋 (좌측 음수, 우측 양수)
    
    Usage:
        frenet_path = FrenetPath(edge_waypoints)
        s, d = frenet_path.cartesian_to_frenet(x, y)
        direction = frenet_path.get_avoidance_direction(obs_x, obs_y)
        avoidance = frenet_path.generate_avoidance_path(...)
    """

    def __init__(self, reference_path: List[Tuple[float, float]]):
        """
        Args:
            reference_path: [(x, y), ...] 경로점 리스트
        """
        if len(reference_path) < 2:
            raise ValueError("Reference path must have at least 2 points")
        
        self._path = np.array(reference_path)
        self._s_profile = self._compute_arc_length()
        self._total_length = self._s_profile[-1] if len(self._s_profile) > 0 else 0.0
        
        # 세그먼트 방향 벡터 미리 계산
        self._segment_headings = self._compute_segment_headings()
        
        # scipy 스플라인 빌드
        self._spline_x = None
        self._spline_y = None
        self._kdtree = None
        self._build_scipy_splines()

    def _compute_arc_length(self) -> np.ndarray:
        """누적 arc-length 계산"""
        diff_x = np.diff(self._path[:, 0])
        diff_y = np.diff(self._path[:, 1])

        s = np.cumsum(np.hypot(diff_x, diff_y))
        s = np.insert(s, 0, 0.0)

        return s

    def _compute_segment_headings(self) -> np.ndarray:
        """각 세그먼트의 heading 계산"""
        diff_x = np.diff(self._path[:, 0])
        diff_y = np.diff(self._path[:, 1])

        headings = np.arctan2(diff_y, diff_x)
        headings = np.append(headings, headings[-1])

        return headings

    def _build_scipy_splines(self) -> None:
        """scipy CubicSpline 생성"""
        # Cubic Spline 보간 (s → x, y)
        self._spline_x = CubicSpline(self._s_profile, self._path[:, 0])
        self._spline_y = CubicSpline(self._s_profile, self._path[:, 1])
        
        # KD-Tree for fast nearest point search
        self._kdtree = cKDTree(self._path)

    @property
    def total_length(self) -> float:
        """경로 총 길이"""
        return self._total_length

    @property
    def path(self) -> np.ndarray:
        """Reference path 반환"""
        return self._path.copy()

    def get_point_at_s(self, s: float) -> Tuple[float, float, float]:
        """
        s 위치의 (x, y, heading) 반환 (스플라인 보간)
        """
        s = max(0.0, min(s, self._total_length))
        
        x = float(self._spline_x(s))
        y = float(self._spline_y(s))
        
        # heading = atan2(dy/ds, dx/ds)
        dx_ds = float(self._spline_x(s, 1))  # 1차 미분
        dy_ds = float(self._spline_y(s, 1))
        heading = np.arctan2(dy_ds, dx_ds)
        
        return x, y, heading

    def get_curvature_at_s(self, s: float) -> float:
        """
        s 위치의 곡률 계산
        
        Returns:
            곡률 (1/m)
        """
        s = max(0.0, min(s, self._total_length))
        
        # 곡률 = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        dx = float(self._spline_x(s, 1))
        dy = float(self._spline_y(s, 1))
        ddx = float(self._spline_x(s, 2))
        ddy = float(self._spline_y(s, 2))
        
        denom = (dx ** 2 + dy ** 2) ** 1.5
        if denom < 1e-6:
            return 0.0
        
        return abs(dx * ddy - dy * ddx) / denom

    def cartesian_to_frenet(self, x: float, y: float) -> Tuple[float, float]:
        """
        Cartesian (x, y) → Frenet (s, d) 변환
        KD-Tree로 빠른 탐색 + 정밀 보정
        """
        # 가장 가까운 점 찾기
        dist, idx = self._kdtree.query([x, y])
        
        # 주변 세그먼트에서 정밀 투영
        best_s = self._s_profile[idx]
        best_d = dist
        
        # 인접 세그먼트 검사
        for i in [max(0, idx - 1), idx]:
            if i >= len(self._path) - 1:
                continue
            
            x1, y1 = self._path[i]
            x2, y2 = self._path[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            
            if seg_len_sq < 1e-6:
                continue
            
            t = ((x - x1) * dx + (y - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
            
            if dist_sq < best_d ** 2:
                seg_len = np.sqrt(seg_len_sq)
                best_s = self._s_profile[i] + t * seg_len
                cross = dx * (y - y1) - dy * (x - x1)
                best_d = np.sqrt(dist_sq) * (-1.0 if cross > 0 else 1.0)
        
        return best_s, best_d

    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float]:
        """
        Frenet (s, d) → Cartesian (x, y) 변환
        """
        px, py, heading = self.get_point_at_s(s)
        
        # d만큼 횡방향(수직) 이동
        nx = np.sin(heading)
        ny = -np.cos(heading)
        
        x = px + d * nx
        y = py + d * ny
        
        return x, y

    def obstacle_to_frenet(self, obs_x: float, obs_y: float) -> Tuple[float, float]:
        """장애물 좌표 → Frenet 변환"""
        return self.cartesian_to_frenet(obs_x, obs_y)

    def generate_avoidance_path(
        self,
        s_start: float,
        s_end: float,
        d_offset: float,
        num_points: int = 20,
        smooth_ratio: float = 0.3
    ) -> List[Tuple[float, float]]:
        """
        회피 경로 생성
        
        Args:
            s_start: 회피 시작 s
            s_end: 회피 종료 s
            d_offset: 횡방향 오프셋 (양수=우측, 음수=좌측)
            num_points: 경로점 수
            smooth_ratio: 진입/복귀 구간 비율 (0~0.5)
        """
        path = []
        smooth_ratio = max(0.1, min(0.5, smooth_ratio))
        
        for i in range(num_points):
            t = i / (num_points - 1)
            s = s_start + t * (s_end - s_start)
            
            if t < smooth_ratio:
                d_ratio = t / smooth_ratio
                d = d_offset * self._smooth_step(d_ratio)
            elif t > (1.0 - smooth_ratio):
                d_ratio = (1.0 - t) / smooth_ratio
                d = d_offset * self._smooth_step(d_ratio)
            else:
                d = d_offset
            
            x, y = self.frenet_to_cartesian(s, d)
            path.append((x, y))
        
        return path

    def _smooth_step(self, t: float) -> float:
        """Smoothstep 함수 (부드러운 0→1 전환)"""
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)

    def generate_lane_offset_path(
        self,
        d_offset: float,
        s_start: Optional[float] = None,
        s_end: Optional[float] = None,
        num_points: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """
        전체 경로를 d만큼 오프셋한 경로 생성
        """
        s_start = s_start if s_start is not None else 0.0
        s_end = s_end if s_end is not None else self._total_length
        
        if num_points is None:
            num_points = max(10, int((s_end - s_start) / 0.5))
        
        path = []
        for i in range(num_points):
            t = i / (num_points - 1)
            s = s_start + t * (s_end - s_start)
            x, y = self.frenet_to_cartesian(s, d_offset)
            path.append((x, y))
        
        return path

    def is_obstacle_in_lane(
        self,
        obs_x: float,
        obs_y: float,
        lane_width: float = 3.5
    ) -> bool:
        """장애물이 경로 차선 내에 있는지 확인"""
        s, d = self.cartesian_to_frenet(obs_x, obs_y)
        return abs(d) <= lane_width / 2

    def get_avoidance_direction(
        self,
        obs_x: float,
        obs_y: float,
        default_direction: str = "right"
    ) -> str:
        """장애물 회피 방향 결정"""
        s, d = self.cartesian_to_frenet(obs_x, obs_y)
        
        if abs(d) < 0.5:
            return default_direction
        elif d > 0:
            return "left"
        else:
            return "right"

    def sample_path(self, ds: float = 0.5) -> List[Tuple[float, float]]:
        """
        경로를 일정 간격으로 샘플링 (스플라인 사용으로 부드러운 경로)
        
        Args:
            ds: 샘플링 간격 (m)
        """
        num_points = max(2, int(self._total_length / ds) + 1)
        path = []
        
        for i in range(num_points):
            s = i * ds
            if s > self._total_length:
                s = self._total_length
            x, y, _ = self.get_point_at_s(s)
            path.append((x, y))
        
        return path
