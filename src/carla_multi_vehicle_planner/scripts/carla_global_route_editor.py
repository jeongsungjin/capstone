#!/usr/bin/env python3

import math
import os
from typing import List, Tuple

import cv2
import numpy as np
import rospy
import yaml
from nav_msgs.msg import Path

# CARLA 맵 / GlobalRoutePlanner 를 직접 써서 전체 도로 네트워크와 샘플 경로를 시각화하기 위한 의존성
try:
    from setup_carla_path import CARLA_EGG, AGENTS_ROOT  # type: ignore
except Exception:
    CARLA_EGG = None
    AGENTS_ROOT = None

try:
    import carla  # type: ignore
except Exception:
    carla = None

try:
    from agents.navigation.global_route_planner import (  # type: ignore
        GlobalRoutePlanner,
    )
    from agents.navigation.global_route_planner_dao import (  # type: ignore
        GlobalRoutePlannerDAO,
    )
except Exception:
    GlobalRoutePlanner = None
    GlobalRoutePlannerDAO = None


class CarlaGlobalRouteEditor:
    """
    CARLA 글로벌 경로(Path)를 시각화해서 점 단위로 상하좌우 이동시키고
    YAML 파일로 저장하는 간단한 편집기.

    - 입력: ROS Path (예: /global_path_ego_vehicle_2)
    - UI: OpenCV 2D 화면에서 경로 폴리라인 표시
      * 마우스 왼클릭: 가장 가까운 점 선택
      * 방향키 ←↑→↓: 선택된 점을 월드 좌표 기준으로 이동
      * s: YAML 저장
      * q 또는 ESC: 종료
    """

    def __init__(self) -> None:
        rospy.init_node("carla_global_route_editor", anonymous=True)

        # path_topic 이 빈 문자열이면 ROS Path 없이 빈 경로에서 시작 (수동으로 점을 찍어 생성)
        # 기본값은 "" 로 두어서 /global_path_* 를 전혀 안 쓰는 모드가 기본이 되도록 함
        self.path_topic = str(rospy.get_param("~path_topic", ""))
        self.output_yaml = rospy.get_param(
            "~output_yaml",
            os.path.join(
                os.path.dirname(__file__),
                "edited_global_path.yaml",
            ),
        )
        # 한 번에 움직이는 거리 (m)
        self.edit_step_m = float(rospy.get_param("~edit_step_m", 0.3))

        # 편집 대상 경로 (x, y) 리스트
        self.points: List[Tuple[float, float]] = []

        # 맵 전체를 시각화하기 위한 CARLA waypoint 리스트
        self.map_points: List[Tuple[float, float]] = []

        # CARLA 맵을 배경으로 쓸지 여부
        self.use_carla_map = bool(rospy.get_param("~use_carla_map", True))

        # CARLA route 플래너를 사용해서 초기 경로를 불러올지 여부
        self.use_carla_route = bool(rospy.get_param("~use_carla_route", False))
        # CARLA map waypoint 자체를 편집 대상으로 쓸지 여부 (true 면 generate_waypoints 결과를 편집)
        self.edit_map_waypoints = bool(rospy.get_param("~edit_map_waypoints", True))
        # spawn 포인트 인덱스로 시작/종료 위치 지정
        self.start_spawn_index = int(rospy.get_param("~start_spawn_index", 0))
        self.end_spawn_index = int(rospy.get_param("~end_spawn_index", 1))
        self.route_resolution = float(rospy.get_param("~route_resolution", 0.3))

        # 화면 좌표 변환 파라미터 (bounds 계산에 사용)
        self.margin = 40
        self.img_size = 1000  # 정사각형 캔버스

        # (옵션1) CARLA 맵 전체 waypoint 를 로드해서 배경으로 표시
        if self.use_carla_map:
            self._load_carla_map_points()

        # 옵션: 별도 초기 경로가 없고, 맵 waypoint 를 직접 수정하고 싶다면,
        # map_points 를 편집용 points 로 복사
        if self.edit_map_waypoints and not self.points and self.map_points:
            self.points = list(self.map_points)
            rospy.loginfo(
                "CarlaGlobalRouteEditor: using CARLA map waypoints (%d pts) as editable points",
                len(self.points),
            )

        # (옵션2) CARLA GlobalRoutePlanner 를 사용해서 spawn→spawn 경로를 초기 편집 경로로 로드
        if self.use_carla_route and not self.points:
            self._load_route_from_carla()

        # (옵션3) ROS Path 에서 초기 경로 로드 (carla_route 가 없을 때 보조)
        if (not self.use_carla_route) and self.path_topic:
            # 빈 문자열인 경우는 사용하지 않음
            if not self.path_topic.strip():
                rospy.loginfo(
                    "CarlaGlobalRouteEditor: path_topic is empty string -> skip Path loading"
                )
            else:
                try:
                    rospy.loginfo(
                        "CarlaGlobalRouteEditor: waiting for Path on %s",
                        self.path_topic,
                    )
                    path_msg: Path = rospy.wait_for_message(
                        self.path_topic, Path, timeout=5.0
                    )
                    rospy.loginfo(
                        "CarlaGlobalRouteEditor: received path with %d poses",
                        len(path_msg.poses),
                    )

                    for pose in path_msg.poses:
                        self.points.append(
                            (float(pose.pose.position.x), float(pose.pose.position.y))
                        )
                except rospy.ROSException:
                    rospy.logwarn(
                        "CarlaGlobalRouteEditor: timeout waiting for Path on %s, starting without initial path",
                        self.path_topic,
                    )

        if not self.points:
            rospy.loginfo(
                "CarlaGlobalRouteEditor: no initial path loaded. "
                "마우스 왼클릭으로 점을 추가하여 새 경로를 만들 수 있습니다."
            )

        # 편집/맵 포인트를 반영해서 스케일 계산
        self._compute_bounds()

        # 선택된 점 인덱스
        self.selected_idx = 0 if self.points else -1

        # 마우스 위치 기록
        self.last_mouse_pos = (0, 0)

        self._run_ui()

    # ------------------------------------------------------------------
    # 좌표 변환
    # ------------------------------------------------------------------
    def _compute_bounds(self) -> None:
        # 편집 경로와 CARLA 맵 waypoint 를 모두 고려해서 화면 bounds 설정
        all_x: List[float] = []
        all_y: List[float] = []
        if self.points:
            all_x.extend(p[0] for p in self.points)
            all_y.extend(p[1] for p in self.points)
        if self.map_points:
            all_x.extend(p[0] for p in self.map_points)
            all_y.extend(p[1] for p in self.map_points)

        if not all_x:
            # 기본 좌표계 (0,0)을 중심으로 작은 스케일
            self.min_x = -10.0
            self.max_x = 10.0
            self.min_y = -10.0
            self.max_y = 10.0
        else:
            self.min_x = min(all_x)
            self.max_x = max(all_x)
            self.min_y = min(all_y)
            self.max_y = max(all_y)

        width = max(self.max_x - self.min_x, 1e-3)
        height = max(self.max_y - self.min_y, 1e-3)
        span = max(width, height)

        available = self.img_size - 2 * self.margin
        self.scale = available / span

    def world_to_image(self, x: float, y: float) -> Tuple[int, int]:
        """월드 좌표(x, y)를 이미지 픽셀 좌표(u, v)로 변환."""
        u = int((x - self.min_x) * self.scale) + self.margin
        # OpenCV 좌표계는 y가 아래로 증가하므로 뒤집기
        v = int((self.max_y - y) * self.scale) + self.margin
        return u, v

    def image_to_world(self, u: int, v: int) -> Tuple[float, float]:
        """이미지 픽셀 좌표(u, v)를 월드 좌표(x, y)로 근사 역변환."""
        x = (u - self.margin) / self.scale + self.min_x
        y = self.max_y - (v - self.margin) / self.scale
        return float(x), float(y)

    # ------------------------------------------------------------------
    # UI 루프
    # ------------------------------------------------------------------
    def _run_ui(self) -> None:
        win_name = "CARLA Global Route Editor"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self._on_mouse)

        rospy.loginfo(
            "CarlaGlobalRouteEditor: UI started. "
            "마우스 왼클릭=점 선택, 방향키=이동, s=YAML 저장, q/ESC=종료"
        )

        while not rospy.is_shutdown():
            # 진한 회색 배경으로 변경 (도로/경로가 더 잘 보이도록)
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 40

            # CARLA 맵 전체 waypoint 를 연한 회색 점으로 표시 (도로 네트워크 확인용)
            if self.map_points:
                for x, y in self.map_points:
                    u, v = self.world_to_image(x, y)
                    # 점 크기를 1 → 3 으로 키워서 가시성 향상
                    cv2.circle(img, (u, v), 3, (120, 120, 120), -1)

            # 경로 폴리라인 그리기
            if self.points:
                pts_img = [self.world_to_image(x, y) for x, y in self.points]
                if len(pts_img) >= 2:
                    cv2.polylines(
                        img,
                        [np.array(pts_img, dtype=np.int32)],
                        isClosed=False,
                        color=(0, 0, 0),
                        thickness=2,
                    )

            # 모든 점 표시
            for i, (x, y) in enumerate(self.points):
                u, v = self.world_to_image(x, y)
                # 선택된 점: 노란색, 나머지: 시안색, 더 굵게 표시
                color = (0, 255, 255) if i == self.selected_idx else (255, 255, 0)
                radius = 7 if i == self.selected_idx else 4
                cv2.circle(img, (u, v), radius, color, -1)

            # 선택 인덱스/좌표 텍스트
            if 0 <= self.selected_idx < len(self.points):
                sel_x, sel_y = self.points[self.selected_idx]
                text = f"idx={self.selected_idx}  x={sel_x:.2f}, y={sel_y:.2f}"
            else:
                text = "no point selected"
            cv2.putText(
                img,
                text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 간단한 사용법 도움말
            help_text = "LClick: select/move point, Arrows: move, S: save YAML, Q/ESC: quit"
            cv2.putText(
                img,
                help_text,
                (10, self.img_size - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(win_name, img)
            key = cv2.waitKey(30) & 0xFF

            if key == 27 or key == ord("q"):
                # ESC 또는 q
                rospy.loginfo("CarlaGlobalRouteEditor: exit requested")
                break
            elif key == ord("s"):
                self._save_yaml()
            elif key == 81:  # 왼쪽 화살표
                self._nudge_selected(dx=-self.edit_step_m, dy=0.0)
            elif key == 83:  # 오른쪽 화살표
                self._nudge_selected(dx=self.edit_step_m, dy=0.0)
            elif key == 82:  # 위쪽 화살표
                self._nudge_selected(dx=0.0, dy=self.edit_step_m)
            elif key == 84:  # 아래쪽 화살표
                self._nudge_selected(dx=0.0, dy=-self.edit_step_m)

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # 마우스 / 키보드 핸들러
    # ------------------------------------------------------------------
    def _on_mouse(self, event, u, v, flags, param) -> None:
        self.last_mouse_pos = (u, v)
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.points:
                # 첫 점 추가
                x, y = self.image_to_world(u, v)
                self.points.append((x, y))
                self.selected_idx = 0
                self._compute_bounds()
                rospy.loginfo("Added first point at (x=%.2f, y=%.2f)", x, y)
            else:
                # 가장 가까운 점 선택
                min_dist = float("inf")
                best_idx = self.selected_idx if self.selected_idx >= 0 else 0
                for i, (x, y) in enumerate(self.points):
                    px, py = self.world_to_image(x, y)
                    d2 = (px - u) ** 2 + (py - v) ** 2
                    if d2 < min_dist:
                        min_dist = d2
                        best_idx = i
                self.selected_idx = best_idx
                sel_x, sel_y = self.points[self.selected_idx]
                rospy.loginfo(
                    "Selected index %d at (x=%.2f, y=%.2f)",
                    self.selected_idx,
                    sel_x,
                    sel_y,
                )

    def _nudge_selected(self, dx: float, dy: float) -> None:
        """선택된 점을 월드 좌표 기준으로 이동."""
        if not (0 <= self.selected_idx < len(self.points)):
            return
        x, y = self.points[self.selected_idx]
        new_x = x + dx
        new_y = y + dy
        self.points[self.selected_idx] = (new_x, new_y)
        self._compute_bounds()
        rospy.loginfo(
            "Moved index %d to (x=%.2f, y=%.2f)", self.selected_idx, new_x, new_y
        )

    # ------------------------------------------------------------------
    # CARLA 맵에서 전체 글로벌 루트(waypoint) 시각화용 포인트 로드
    # ------------------------------------------------------------------
    def _load_carla_map_points(self) -> None:
        """CARLA 맵의 driving lane waypoint 를 전역적으로 샘플링해서 self.map_points 로 저장."""
        if carla is None:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: CARLA Python API not available, cannot load global map waypoints"
            )
            return

        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        resolution = float(rospy.get_param("~map_waypoint_resolution", 4.0))

        try:
            client = carla.Client(host, port)
            client.set_timeout(5.0)
            world = client.get_world()
            carla_map = world.get_map()
        except Exception as exc:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: failed to connect to CARLA (%s:%d): %s",
                host,
                port,
                exc,
            )
            return

        try:
            waypoints = carla_map.generate_waypoints(resolution)
        except Exception as exc:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: generate_waypoints failed: %s", exc
            )
            return

        pts: List[Tuple[float, float]] = []
        for wp in waypoints:
            try:
                if wp.lane_type != carla.LaneType.Driving:
                    continue
            except Exception:
                # 오래된 CARLA 버전 호환
                pass
            loc = wp.transform.location
            pts.append((float(loc.x), float(loc.y)))

        self.map_points = pts
        rospy.loginfo(
            "CarlaGlobalRouteEditor: loaded %d CARLA map waypoints for visualization",
            len(self.map_points),
        )

    # ------------------------------------------------------------------
    # CARLA GlobalRoutePlanner 를 사용해서 spawn 간 경로를 초기 편집 경로로 로드
    # ------------------------------------------------------------------
    def _load_route_from_carla(self) -> None:
        """CARLA GlobalRoutePlanner 를 사용하여 spawn→spawn 경로를 생성해 self.points 로 저장."""
        if carla is None or GlobalRoutePlanner is None:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: GlobalRoutePlanner not available, cannot load route"
            )
            return

        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))

        try:
            client = carla.Client(host, port)
            client.set_timeout(5.0)
            world = client.get_world()
            carla_map = world.get_map()
        except Exception as exc:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: failed to connect to CARLA (%s:%d) for route: %s",
                host,
                port,
                exc,
            )
            return

        spawn_points = carla_map.get_spawn_points()
        if not spawn_points or len(spawn_points) < 2:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: not enough spawn points to build route"
            )
            return

        start_idx = max(0, min(self.start_spawn_index, len(spawn_points) - 1))
        end_idx = max(0, min(self.end_spawn_index, len(spawn_points) - 1))
        if start_idx == end_idx:
            end_idx = (start_idx + 1) % len(spawn_points)

        start_loc = spawn_points[start_idx].location
        end_loc = spawn_points[end_idx].location

        try:
            if GlobalRoutePlannerDAO is not None:
                dao = GlobalRoutePlannerDAO(carla_map, 2.0)
                grp = GlobalRoutePlanner(dao)
            else:
                grp = GlobalRoutePlanner(carla_map, 2.0)
            if hasattr(grp, "setup"):
                grp.setup()
        except Exception as exc:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: failed to setup GlobalRoutePlanner: %s", exc
            )
            return

        try:
            route = grp.trace_route(start_loc, end_loc)
        except Exception as exc:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: trace_route failed from spawn %d to %d: %s",
                start_idx,
                end_idx,
                exc,
            )
            return

        if not route or len(route) < 2:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: GlobalRoutePlanner returned empty/short route"
            )
            return

        pts = self._route_to_points(route)
        if len(pts) < 2:
            rospy.logwarn(
                "CarlaGlobalRouteEditor: sampled route has less than 2 points"
            )
            return

        self.points = pts
        rospy.loginfo(
            "CarlaGlobalRouteEditor: loaded CARLA route with %d points from spawn %d to %d",
            len(self.points),
            start_idx,
            end_idx,
        )

    # ------------------------------------------------------------------
    # CARLA route → (x, y) 샘플 포인트 변환 (multi_agent_conflict_free_planner 와 동일 개념)
    # ------------------------------------------------------------------
    def _route_to_points(self, route) -> List[Tuple[float, float]]:
        """[(Waypoint, RoadOption)] 경로를 self.route_resolution 간격 (x, y) 포인트로 샘플링."""
        waypoints = [item[0] for item in route]
        if not waypoints:
            return []
        pts: List[Tuple[float, float]] = []
        last: Tuple[float, float] = None  # type: ignore
        for wp in waypoints:
            loc = wp.transform.location
            curr = (float(loc.x), float(loc.y))
            if last is None:
                pts.append(curr)
                last = curr
                continue
            seg = math.hypot(curr[0] - last[0], curr[1] - last[1])
            if seg < 1e-3:
                continue
            steps = max(1, int(seg // self.route_resolution))
            for step in range(1, steps + 1):
                ratio = min(1.0, (step * self.route_resolution) / seg)
                interp = (
                    last[0] + (curr[0] - last[0]) * ratio,
                    last[1] + (curr[1] - last[1]) * ratio,
                )
                if not pts or math.hypot(
                    interp[0] - pts[-1][0], interp[1] - pts[-1][1]
                ) > 0.05:
                    pts.append(interp)
            if math.hypot(curr[0] - pts[-1][0], curr[1] - pts[-1][1]) > 0.05:
                pts.append(curr)
            last = curr
        return pts

    # ------------------------------------------------------------------
    # YAML 저장
    # ------------------------------------------------------------------
    def _save_yaml(self) -> None:
        """현재 편집된 경로를 YAML 파일로 저장."""
        # yaw, 총 길이 계산
        path_entries = []
        total_length = 0.0
        n = len(self.points)
        for i, (x, y) in enumerate(self.points):
            if i < n - 1:
                nx, ny = self.points[i + 1]
            else:
                # 마지막 점은 이전 점과의 방향 사용
                nx, ny = self.points[i][0], self.points[i][1]
            dx = nx - x
            dy = ny - y
            yaw = math.atan2(dy, dx) if (dx != 0.0 or dy != 0.0) else 0.0
            if i > 0:
                px, py = self.points[i - 1]
                total_length += math.hypot(x - px, y - py)
            path_entries.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "yaw_rad": float(yaw),
                    "yaw_deg": float(math.degrees(yaw)),
                }
            )

        data = {
            "num_points": len(self.points),
            "total_length_m": float(total_length),
            "loop_path": False,
            "path": path_entries,
        }

        out_dir = os.path.dirname(self.output_yaml)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(self.output_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        rospy.loginfo("Saved edited path to %s (num_points=%d, total_length=%.2fm)",
                      self.output_yaml, len(self.points), total_length)


if __name__ == "__main__":
    try:
        CarlaGlobalRouteEditor()
    except rospy.ROSInterruptException:
        pass


