#!/usr/bin/env python3

import math
import os
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import rospy
import yaml

try:
    from setup_carla_path import CARLA_EGG, AGENTS_ROOT  # type: ignore
except Exception:
    CARLA_EGG = None
    AGENTS_ROOT = None

try:
    import carla  # type: ignore
except Exception as exc:  # pragma: no cover
    carla = None
    rospy.logfatal(f"[OFFSET_EDITOR] Failed to import CARLA: {exc}")


class CarlaPathOffsetRegionEditor:
    """
    CARLA 맵 위에서 '좌표 오프셋을 줄 영역'을 브러쉬로 칠해서 지정하고
    각 영역마다 lateral + x/y 방향 오프셋 값을 설정하는 도구.

    - CARLA 맵 로드 → driving lane waypoint 를 2D로 시각화
    - 마우스 드래그로 브러쉬 영역을 칠함 (곡선 커브도 쉽게 지정)
    - 콘솔에서 영역 id, 기본 오프셋(lateral / x / y)을 입력
    - 방향키(←→↑↓)로 선택된 영역의 x/y 오프셋을 nudge_step_m 단위로 조정
    - YAML 로 저장

    이 YAML 은 이후 multi_agent_conflict_free_planner 의
    _route_to_points() 같은 곳에서 좌표 오프셋을 적용할 때 사용할 수 있다.
    """

    def __init__(self) -> None:
        rospy.init_node("carla_path_offset_region_editor", anonymous=True)

        # 패키지 내부 scripts/config/ 로 기본 출력 경로 고정 (툴이 tools/ 로 이동해도 유지되도록)
        script_dir = os.path.dirname(__file__)
        pkg_config_dir = os.path.abspath(os.path.join(script_dir, "..", "config"))
        default_output = os.path.join(pkg_config_dir, "path_offset_regions.yaml")
        self.output_yaml = rospy.get_param("~output_yaml", default_output)
        # 키보드 방향키로 영역의 x/y 오프셋을 조정할 때 한 번에 움직이는 거리(m)
        self.nudge_step_m = float(rospy.get_param("~nudge_step_m", 0.4))
        self.carla_host = rospy.get_param("~carla_host", "localhost")
        self.carla_port = int(rospy.get_param("~carla_port", 2000))
        self.map_waypoint_resolution = float(
            rospy.get_param("~map_waypoint_resolution", 1.0)
        )

        if carla is None:
            raise RuntimeError("[OFFSET_EDITOR] CARLA Python API not available")

        # Connect to CARLA
        client = carla.Client(self.carla_host, self.carla_port)
        client.set_timeout(5.0)
        self.world = client.get_world()
        self.carla_map = self.world.get_map()

        # Generate base waypoints for visualization
        waypoints = self.carla_map.generate_waypoints(self.map_waypoint_resolution)
        self.map_points: List[Tuple[float, float]] = []
        for wp in waypoints:
            try:
                if wp.lane_type != carla.LaneType.Driving:
                    continue
            except Exception:
                pass
            loc = wp.transform.location
            self.map_points.append((float(loc.x), float(loc.y)))

        if not self.map_points:
            raise RuntimeError("[OFFSET_EDITOR] No driving waypoints found on map")

        # Image/transform parameters
        self.margin = 40
        self.img_size = 1000
        self._compute_bounds()

        # Infinite grid (world coordinates) – 8x8m 기본, 원점/위치는 드래그로 조정
        self.grid_cell_m = float(rospy.get_param("~grid_cell_m", 8.0))
        # grid origin in world coords (기본: (0,0))
        self.grid_origin_x = float(rospy.get_param("~grid_origin_x", 0.0))
        self.grid_origin_y = float(rospy.get_param("~grid_origin_y", 0.0))
        # Grid drag state (우클릭 드래그)
        self.dragging_grid = False
        self.grid_drag_start_px: Tuple[int, int] = (0, 0)
        self.grid_origin_start: Tuple[float, float] = (self.grid_origin_x, self.grid_origin_y)

        # Brush painting state
        self.painting = False
        self.brush_radius_px = int(rospy.get_param("~brush_radius_px", 20))
        # 현재 브러쉬로 그리고 있는 world 좌표 포인트들
        self.current_brush_points: List[Tuple[float, float]] = []

        # Collected regions
        # Each region: {id, points (world coords), center, offset_lateral_m, offset_x_m, offset_y_m}
        self.regions: List[Dict[str, Any]] = []

        # 현재 선택/조정 대상 영역 인덱스 (기본: 마지막 영역)
        self.selected_region_idx: int = -1

        self._run_ui()

    # -------------------------------------------------------------
    # 좌표 변환
    # -------------------------------------------------------------
    def _compute_bounds(self) -> None:
        xs = [p[0] for p in self.map_points]
        ys = [p[1] for p in self.map_points]
        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)

        width = max(self.max_x - self.min_x, 1e-3)
        height = max(self.max_y - self.min_y, 1e-3)
        span = max(width, height)
        available = self.img_size - 2 * self.margin
        self.scale = available / span

    def world_to_image(self, x: float, y: float) -> Tuple[int, int]:
        u = int((x - self.min_x) * self.scale) + self.margin
        # OpenCV: y-axis downwards
        v = int((self.max_y - y) * self.scale) + self.margin
        return u, v

    def image_to_world(self, u: int, v: int) -> Tuple[float, float]:
        x = (u - self.margin) / self.scale + self.min_x
        y = self.max_y - (v - self.margin) / self.scale
        return float(x), float(y)

    # -------------------------------------------------------------
    # UI 루프
    # -------------------------------------------------------------
    def _run_ui(self) -> None:
        win_name = "CARLA Path Offset Region Editor"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self._on_mouse)

        rospy.loginfo(
            "[OFFSET_EDITOR] UI started. "
            "좌클릭 드래그=직사각형 영역, 우클릭 드래그=격자 위치 조정, 방향키=off_x/off_y 조정, S=YAML 저장, Q/ESC=종료"
        )

        while not rospy.is_shutdown():
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 40

            # Draw infinite grid in world coordinates
            self._draw_grid(img)

            # Draw map waypoints
            for x, y in self.map_points:
                u, v = self.world_to_image(x, y)
                cv2.circle(img, (u, v), 1, (100, 100, 100), -1)

            # Draw existing regions (브러쉬 포인트 + 현재 offset 반영)
            for idx, region in enumerate(self.regions):
                pts = region.get("points", [])
                if not pts:
                    continue
                color = (0, 255, 255) if idx == self.selected_region_idx else (255, 255, 0)
                off_x = float(region.get("offset_x_m", 0.0))
                off_y = float(region.get("offset_y_m", 0.0))
                for (cx, cy) in pts[:: max(1, len(pts) // 500)]:  # 많을 때는 샘플링
                    u, v = self.world_to_image(cx + off_x, cy + off_y)
                    cv2.circle(img, (u, v), 2, color, -1)
                center = region.get("center", [0.0, 0.0])
                oid = str(region.get("id", ""))
                off_lat = float(region.get("offset_lateral_m", 0.0))
                cu, cv = self.world_to_image(center[0] + off_x, center[1] + off_y)
                cv2.putText(
                    img,
                    f"{oid}: lat={off_lat:.2f}, dx={off_x:.2f}, dy={off_y:.2f}",
                    (cu + 5, cv - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Draw current rectangle selection (preview)
            if getattr(self, "drawing_rect", False):
                x1, y1 = self.rect_start_uv
                x2, y2 = self.rect_end_uv
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            help_text = "L-Drag: rect region, R-Drag: move grid, Arrows: adjust dx/dy, S: save YAML, Q/ESC: quit"
            cv2.putText(
                img,
                help_text,
                (10, self.img_size - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(win_name, img)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord("q"):
                rospy.loginfo("[OFFSET_EDITOR] exit requested")
                break
            elif key == ord("s"):
                self._save_yaml()
            # 방향키로 마지막/선택된 영역을 world 좌표 기준으로 이동
            elif key == 81:  # left
                self._nudge_selected_region(dx=-self.nudge_step_m, dy=0.0)
            elif key == 83:  # right
                self._nudge_selected_region(dx=self.nudge_step_m, dy=0.0)
            elif key == 82:  # up
                self._nudge_selected_region(dx=0.0, dy=self.nudge_step_m)
            elif key == 84:  # down
                self._nudge_selected_region(dx=0.0, dy=-self.nudge_step_m)

        cv2.destroyAllWindows()

    # -------------------------------------------------------------
    # 마우스 콜백
    # -------------------------------------------------------------
    def _on_mouse(self, event, u, v, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # 직사각형 시작점 기록 (image coords)
            self.drawing_rect = True
            self.rect_start_uv = (u, v)
            self.rect_end_uv = (u, v)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_rect:
            # 드래그 중: 현재 마우스 위치까지 직사각형 업데이트
            self.rect_end_uv = (u, v)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing_rect:
            # 직사각형 완료 → world 좌표로 변환하여 영역 등록
            self.drawing_rect = False
            x1, y1 = self.rect_start_uv
            x2, y2 = self.rect_end_uv
            if x1 == x2 or y1 == y2:
                rospy.loginfo("[OFFSET_EDITOR] rectangle too thin, ignored")
            else:
                wx1, wy1 = self.image_to_world(x1, y1)
                wx2, wy2 = self.image_to_world(x2, y2)
                x_min = min(wx1, wx2)
                x_max = max(wx1, wx2)
                y_min = min(wy1, wy2)
                y_max = max(wy1, wy2)
                # 시각화/오프셋 적용용으로 4개의 꼭짓점을 points 로 저장
                points_world = [
                    (x_min, y_max),  # top-left
                    (x_max, y_max),  # top-right
                    (x_max, y_min),  # bottom-right
                    (x_min, y_min),  # bottom-left
                ]
                self._register_rect_region(points_world)
        # 우클릭: 격자 드래그 시작/종료
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging_grid = True
            self.grid_drag_start_px = (u, v)
            self.grid_origin_start = (self.grid_origin_x, self.grid_origin_y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_grid:
            du = u - self.grid_drag_start_px[0]
            dv = v - self.grid_drag_start_px[1]
            # 이미지 → 월드 좌표 변환 (world_to_image 의 역방향)
            dx_world = du / self.scale
            dy_world = -dv / self.scale  # 이미지 y는 아래로 증가
            self.grid_origin_x = self.grid_origin_start[0] + dx_world
            self.grid_origin_y = self.grid_origin_start[1] + dy_world
        elif event == cv2.EVENT_RBUTTONUP and self.dragging_grid:
            self.dragging_grid = False

    def _register_rect_region(self, points_world: List[Tuple[float, float]]) -> None:
        """직사각형 모서리 4점을 world 좌표로 받아 영역 등록."""
        if not points_world or len(points_world) < 4:
            rospy.loginfo("[OFFSET_EDITOR] rectangle has insufficient points, ignored")
            return

        # center 계산 (4개 꼭짓점 평균)
        xs = [p[0] for p in points_world]
        ys = [p[1] for p in points_world]
        center_x = float(sum(xs) / len(xs))
        center_y = float(sum(ys) / len(ys))

        print("\n[OFFSET_EDITOR] 새로운 직사각형 영역이 선택되었습니다.")
        print(f"  corners: {len(points_world)}")
        print(f"  center(world): ({center_x:.2f},{center_y:.2f})")

        # 콘솔에서 id, offset 값 입력
        try:
            region_id = input("  영역 id를 입력하세요 (예: 'R1' 또는 '5'): ").strip()
        except Exception:
            region_id = ""
        if not region_id:
            region_id = f"region_{len(self.regions)+1}"

        offset_lat = 0.0
        try:
            raw = input("  lateral offset [m] (기본 0.0, +면 왼쪽, -면 오른쪽으로 생각): ").strip()
            if raw:
                offset_lat = float(raw)
        except Exception:
            offset_lat = 0.0

        offset_x = 0.0
        try:
            raw = input("  world X offset [m] (기본 0.0, +면 +X 방향): ").strip()
            if raw:
                offset_x = float(raw)
        except Exception:
            offset_x = 0.0

        offset_y = 0.0
        try:
            raw = input("  world Y offset [m] (기본 0.0, +면 +Y 방향): ").strip()
            if raw:
                offset_y = float(raw)
        except Exception:
            offset_y = 0.0

        region = {
            "id": region_id,
            "center": [float(center_x), float(center_y)],
            "points": [[float(x), float(y)] for (x, y) in points_world],
            "offset_lateral_m": float(offset_lat),
            "offset_x_m": float(offset_x),
            "offset_y_m": float(offset_y),
        }
        self.regions.append(region)
        # 새로 추가된 영역을 선택 대상으로 설정
        self.selected_region_idx = len(self.regions) - 1
        rospy.loginfo(
            "[OFFSET_EDITOR] Added region %s with offset(lat=%.2f, dx=%.2f, dy=%.2f) (center=%.2f,%.2f)",
            region_id,
            offset_lat,
            offset_x,
            offset_y,
            center_x,
            center_y,
        )

    def _nudge_selected_region(self, dx: float, dy: float) -> None:
        """선택된 영역의 world X/Y 오프셋 값을 조정."""
        if not self.regions:
            return
        idx = self.selected_region_idx
        if idx < 0 or idx >= len(self.regions):
            idx = len(self.regions) - 1
            self.selected_region_idx = idx
        region = self.regions[idx]
        region["offset_x_m"] = float(region.get("offset_x_m", 0.0)) + dx
        region["offset_y_m"] = float(region.get("offset_y_m", 0.0)) + dy
        rospy.loginfo(
            "[OFFSET_EDITOR] Nudged region %s offset to (dx=%.2f, dy=%.2f)",
            region.get("id", "unknown"),
            region.get("offset_x_m", 0.0),
            region.get("offset_y_m", 0.0),
        )

    # -------------------------------------------------------------
    # YAML 저장
    # -------------------------------------------------------------
    def _save_yaml(self) -> None:
        if not self.regions:
            rospy.logwarn("[OFFSET_EDITOR] No regions to save")
            return
        data = {
            "regions": self.regions,
            "description": "Regions for geometric offset along global paths. "
            "offset_lateral_m is applied along lane normal; offset_x_m, offset_y_m are world X/Y shifts.",
        }
        out_dir = os.path.dirname(self.output_yaml)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(self.output_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        rospy.loginfo(
            "[OFFSET_EDITOR] Saved %d regions to %s", len(self.regions), self.output_yaml
        )

    # -------------------------------------------------------------
    # Grid drawing
    # -------------------------------------------------------------
    def _draw_grid(self, img: np.ndarray) -> None:
        """무한 8x8m(기본) 격자를 world 좌표계 기준으로 그리고, 우클릭 드래그로 origin 을 조정."""
        cell = max(0.1, float(self.grid_cell_m))

        # world bounds of current view
        x_min = self.min_x
        x_max = self.max_x
        y_min = self.min_y
        y_max = self.max_y

        # vertical lines (고정 x)
        # k such that x = grid_origin_x + k*cell in [x_min, x_max]
        start_k = int(math.floor((x_min - self.grid_origin_x) / cell))
        end_k = int(math.ceil((x_max - self.grid_origin_x) / cell))
        for k in range(start_k, end_k + 1):
            gx = self.grid_origin_x + k * cell
            p1u, p1v = self.world_to_image(gx, y_min)
            p2u, p2v = self.world_to_image(gx, y_max)
            cv2.line(img, (p1u, p1v), (p2u, p2v), (60, 60, 60), 1)

        # horizontal lines (고정 y)
        start_l = int(math.floor((y_min - self.grid_origin_y) / cell))
        end_l = int(math.ceil((y_max - self.grid_origin_y) / cell))
        for l in range(start_l, end_l + 1):
            gy = self.grid_origin_y + l * cell
            p1u, p1v = self.world_to_image(x_min, gy)
            p2u, p2v = self.world_to_image(x_max, gy)
            cv2.line(img, (p1u, p1v), (p2u, p2v), (60, 60, 60), 1)


if __name__ == "__main__":
    try:
        CarlaPathOffsetRegionEditor()
    except rospy.ROSInterruptException:
        pass


