#!/usr/bin/env python3

import math
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header


class SimplePlatoonManager:
    """
    단순 플래툰 (경로만 공유):
    - 모든 차량이 동일한 경로를 사용하도록 path를 복제해서 배포
    - 속도 오버라이드/등간격 제어는 하지 않음 (각 차량이 자기 컨트롤러 목표속도 사용)
    """

    def __init__(self) -> None:
        rospy.init_node("simple_platoon_manager", anonymous=False)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.leader_role = str(rospy.get_param("~leader_role", "ego_vehicle_2"))
        followers_str = str(rospy.get_param("~follower_roles", "ego_vehicle_1,ego_vehicle_3"))
        self.follower_roles: List[str] = [r.strip() for r in followers_str.split(",") if r.strip()][: max(0, self.num_vehicles - 1)]

        self.path_source_topic = str(rospy.get_param("~path_source_topic", f"/global_path_{self.leader_role}"))
        # 0 또는 음수이면 주기적 재발행을 하지 않는다.
        self.republish_hz = float(rospy.get_param("~republish_hz", 0.0))
        # 차량 위치 기준으로 경로 시작부를 잘라낼 때 허용할 최대 오프셋 (m)
        self.max_trim_distance_m = float(rospy.get_param("~max_trim_distance_m", 30.0))
        # 트림 시 뒤로 허용할 최대 백트랙 (m) – 지나간 구간으로 점프 방지
        self.trim_backtrack_window_m = float(rospy.get_param("~trim_backtrack_window_m", 8.0))
        # 리더 재계획 시 후미 보호용 이전 경로 유지/병합 설정
        self.merge_keep_dist_m = float(rospy.get_param("~merge_keep_dist_m", 30.0))

        self.publish_leader_path = bool(rospy.get_param("~publish_leader_path", True))

        self.all_roles = [self.leader_role] + self.follower_roles

        self.path_msg: Optional[Path] = None
        self.path_pts: List[Tuple[float, float]] = []
        self.path_s: List[float] = []
        self.odom: Dict[str, Odometry] = {}
        self._last_s: Dict[str, float] = {}

        pub_roles = self.all_roles if self.publish_leader_path else self.follower_roles
        self.path_pubs: Dict[str, rospy.Publisher] = {
            r: rospy.Publisher(f"/planned_path_{r}", Path, queue_size=1, latch=True) for r in pub_roles
        }
        # RViz에서 global_path_* 를 그릴 때도 모두 동일 경로를 보도록 추가 퍼블리셔
        self.global_path_pubs: Dict[str, rospy.Publisher] = {
            r: rospy.Publisher(f"/global_path_{r}", Path, queue_size=1, latch=True) for r in pub_roles
        }

        rospy.Subscriber(self.path_source_topic, Path, self._path_cb, queue_size=1)
        for role in self.all_roles:
            rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role, queue_size=10)

        if self.republish_hz > 0.0:
            rospy.Timer(rospy.Duration(1.0 / max(0.1, self.republish_hz)), self._republish_path)

        rospy.loginfo("[platoon] leader=%s followers=%s path=%s", self.leader_role, self.follower_roles, self.path_source_topic)

    def _path_cb(self, msg: Path) -> None:
        if not msg.poses:
            return
        merged = self._merge_with_previous(msg)
        # 이후 로직은 병합 결과를 기준으로 동작
        msg = merged
        fixed = self._ensure_orientations(msg)
        self.path_msg = fixed
        self._prepare_path_data(fixed)
        # if self.log_verbose:
        #     rospy.loginfo("[platoon] shared leader path (%d pts)", len(msg.poses))
        self._publish_path_to_all(fixed)

    def _republish_path(self, _evt=None) -> None:
        if self.path_msg is None or not self.path_msg.poses:
            return
        self._publish_path_to_all(self.path_msg)

    def _publish_path_to_all(self, msg: Path) -> None:
        for role, pub in self.path_pubs.items():
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=msg.header.frame_id)
            path_copy.poses = self._trim_poses_for_role(role, msg)
            if path_copy.poses:
                rospy.logfatal(
                    "[PLATOON PUB] role=%s n=%d first=(%.2f,%.2f) last=(%.2f,%.2f)",
                    role,
                    len(path_copy.poses),
                    path_copy.poses[0].pose.position.x,
                    path_copy.poses[0].pose.position.y,
                    path_copy.poses[-1].pose.position.x,
                    path_copy.poses[-1].pose.position.y,
                )
            pub.publish(path_copy)
        # RViz에서 global_path_* 를 그릴 때도 동일 경로를 보게 복제 (trim 동일 적용)
        for role, pub in self.global_path_pubs.items():
            path_copy = Path()
            path_copy.header = Header(stamp=rospy.Time.now(), frame_id=msg.header.frame_id)
            path_copy.poses = self._trim_poses_for_role(role, msg)
            if path_copy.poses:
                rospy.logfatal(
                    "[PLATOON PUB] global role=%s n=%d first=(%.2f,%.2f) last=(%.2f,%.2f)",
                    role,
                    len(path_copy.poses),
                    path_copy.poses[0].pose.position.x,
                    path_copy.poses[0].pose.position.y,
                    path_copy.poses[-1].pose.position.x,
                    path_copy.poses[-1].pose.position.y,
                )
            pub.publish(path_copy)

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        self.odom[role] = msg

    def _ensure_orientations(self, msg: Path) -> Path:
        """경로에 orientation이 비어 있으면 좌표 기울기로 보완."""
        pts: List[Tuple[float, float]] = []
        for ps in msg.poses:
            pts.append((ps.pose.position.x, ps.pose.position.y))
        need_fill = True
        for ps in msg.poses:
            q = ps.pose.orientation
            if abs(q.w) > 1e-3 or abs(q.z) > 1e-3 or abs(q.x) > 1e-3 or abs(q.y) > 1e-3:
                need_fill = False
                break
        if len(pts) < 2 or not need_fill:
            return msg
        new_msg = Path()
        new_msg.header = msg.header
        for i, ps in enumerate(msg.poses):
            x, y = pts[i]
            if i < len(pts) - 1:
                nx, ny = pts[i + 1]
            else:
                nx, ny = pts[i]
            dx = nx - x
            dy = ny - y
            yaw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else 0.0
            qz = math.sin(0.5 * yaw)
            qw = math.cos(0.5 * yaw)
            p = ps
            p.pose.orientation.x = 0.0
            p.pose.orientation.y = 0.0
            p.pose.orientation.z = qz
            p.pose.orientation.w = qw
            new_msg.poses.append(p)
        return new_msg

    def _prepare_path_data(self, msg: Path) -> None:
        pts: List[Tuple[float, float]] = []
        for ps in msg.poses:
            pts.append((ps.pose.position.x, ps.pose.position.y))
        if len(pts) < 2:
            self.path_pts = []
            self.path_s = []
            return
        s: List[float] = [0.0]
        total = 0.0
        for i in range(1, len(pts)):
            step = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            total += step
            s.append(total)
        self.path_pts = pts
        self.path_s = s

    def _project_s(self, x: float, y: float) -> Optional[Tuple[float, float, int, float, float, float]]:
        if len(self.path_pts) < 2 or len(self.path_s) != len(self.path_pts):
            return None
        best_dist2 = None
        best_s = None
        best_idx = None
        best_t = 0.0
        best_px = 0.0
        best_py = 0.0
        for i in range(len(self.path_pts) - 1):
            x1, y1 = self.path_pts[i]
            x2, y2 = self.path_pts[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg = dx * dx + dy * dy
            if seg < 1e-9:
                continue
            t = ((x - x1) * dx + (y - y1) * dy) / seg
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy
            dist2 = (x - px) * (x - px) + (y - py) * (y - py)
            s_proj = self.path_s[i] + math.sqrt(seg) * t
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_s = s_proj
                best_idx = i
                best_t = t
                best_px = px
                best_py = py
        if best_s is None or best_dist2 is None or best_idx is None:
            return None
        return best_s, math.sqrt(best_dist2), best_idx, best_t, best_px, best_py

    def _trim_poses_for_role(self, role: str, msg: Path):
        """
        차량 현재 위치를 경로에 투영해 그 지점부터의 서픽스를 전달한다.
        교차로 재계획 등으로 경로가 크게 달라도 각 차량이 자기 위치에서
        가까운 지점부터 추종하도록 보정한다.
        """
        if len(msg.poses) < 2:
            return msg.poses
        odom = self.odom.get(role)
        if odom is None:
            return msg.poses
        pos = odom.pose.pose.position
        proj = self._project_s(pos.x, pos.y)
        if proj is None:
            return msg.poses
        _s, dist, idx, t, px, py = proj
        # 너무 멀리 떨어진 경우 원본 유지
        if dist > self.max_trim_distance_m:
            return msg.poses
        # 지나간 구간으로 점프하지 않도록 최근 진행 s보다 크게 유지
        last_s = self._last_s.get(role)
        if last_s is not None and _s < (last_s - self.trim_backtrack_window_m):
            return msg.poses
        # idx는 세그먼트 시작 인덱스이므로 그대로 사용 (마지막 안전)
        idx = max(0, min(idx, len(msg.poses) - 2))
        # 보간된 시작점을 삽입해 차량 위치에서 바로 시작하도록 보정
        start_pose = PoseStamped()
        start_pose.header = msg.header
        # 세그먼트 방향 기반 yaw
        seg_x1 = msg.poses[idx].pose.position.x
        seg_y1 = msg.poses[idx].pose.position.y
        seg_x2 = msg.poses[idx + 1].pose.position.x
        seg_y2 = msg.poses[idx + 1].pose.position.y
        seg_dx = seg_x2 - seg_x1
        seg_dy = seg_y2 - seg_y1
        yaw = math.atan2(seg_dy, seg_dx) if abs(seg_dx) + abs(seg_dy) > 1e-9 else 0.0
        start_pose.pose.position.x = px
        start_pose.pose.position.y = py
        start_pose.pose.position.z = msg.poses[idx].pose.position.z
        start_pose.pose.orientation.x = 0.0
        start_pose.pose.orientation.y = 0.0
        start_pose.pose.orientation.z = math.sin(0.5 * yaw)
        start_pose.pose.orientation.w = math.cos(0.5 * yaw)
        trimmed = [start_pose] + msg.poses[idx + 1 :]
        # 최소 두 점은 유지
        if len(trimmed) < 2:
            return msg.poses
        self._last_s[role] = _s
        return trimmed

    def _merge_with_previous(self, new_msg: Path) -> Path:
        """
        리더가 재계획한 새 경로가 들어올 때,
        이전 경로의 리더 진행 지점 이후 구간을 일정 거리(merge_keep_dist_m)만큼 유지해
        하나의 경로로 이어붙인다.
        """
        # 이전 경로가 없거나 유지 거리가 0이면 그대로 반환
        if self.path_msg is None or not self.path_msg.poses or self.merge_keep_dist_m <= 0.0:
            return new_msg
        odom = self.odom.get(self.leader_role)
        if odom is None:
            return new_msg
        # 이전 경로 기준 투영 (self.path_pts/self.path_s는 이전 경로 데이터)
        proj = self._project_s(odom.pose.pose.position.x, odom.pose.pose.position.y)
        if proj is None:
            return new_msg
        s_proj, dist, idx, t, px, py = proj

        prev = self.path_msg
        if len(prev.poses) < 2:
            return new_msg

        # 이전 경로 suffix 생성 (리더 진행 지점부터 keep_dist까지)
        suffix: List[PoseStamped] = []
        # 시작 보간점
        start_pose = PoseStamped()
        start_pose.header = prev.header
        seg_x1 = prev.poses[idx].pose.position.x
        seg_y1 = prev.poses[idx].pose.position.y
        seg_x2 = prev.poses[idx + 1].pose.position.x if idx + 1 < len(prev.poses) else seg_x1
        seg_y2 = prev.poses[idx + 1].pose.position.y if idx + 1 < len(prev.poses) else seg_y1
        seg_dx, seg_dy = seg_x2 - seg_x1, seg_y2 - seg_y1
        yaw = math.atan2(seg_dy, seg_dx) if abs(seg_dx) + abs(seg_dy) > 1e-9 else 0.0
        start_pose.pose.position.x = px
        start_pose.pose.position.y = py
        start_pose.pose.position.z = prev.poses[idx].pose.position.z
        start_pose.pose.orientation.x = 0.0
        start_pose.pose.orientation.y = 0.0
        start_pose.pose.orientation.z = math.sin(0.5 * yaw)
        start_pose.pose.orientation.w = math.cos(0.5 * yaw)
        suffix.append(start_pose)

        # s_profile이 이전 path_s에 담겨 있으므로 이를 활용
        keep_limit = s_proj + max(0.0, self.merge_keep_dist_m)
        for i in range(idx + 1, len(prev.poses)):
            if i >= len(self.path_s):
                break
            if self.path_s[i] > keep_limit:
                break
            suffix.append(prev.poses[i])

        # suffix 길이가 충분하지 않으면 병합 이득이 거의 없으므로 건너뜀
        if len(suffix) < 2:
            return new_msg

        # 새 경로와의 연결을 부드럽게: 간격이 크면 직선 보간 점 삽입
        merged = Path()
        merged.header = new_msg.header
        merged.poses = []

        merged.poses.extend(suffix)
        if new_msg.poses:
            merged.poses.extend(new_msg.poses)
        # 새 경로가 비었으면 suffix만이라도 유지

        # 최소 두 점 유지
        if len(merged.poses) < 2:
            return new_msg
        return merged


def main() -> None:
    try:
        SimplePlatoonManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

