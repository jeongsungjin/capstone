#!/usr/bin/env python3
"""
단순 데드락 해소 토큰 매니저 (ID 우선순위 기반).

- 각 차량 pose(/<role>/pose)와 글로벌 경로(/global_path_<role>)를 구독.
- 정지 상태(저속 지속) 차량 중 ID가 가장 낮은 차량에 토큰을 부여.
- 토큰 차량은 경로 전방 lookahead 구간에 다른 차량이 d_safe 이내 없을 때만
  /collision_override/<role> = True 를 퍼블리시하여 컨트롤러의 충돌 정지 게이트를 해제.
- 토큰 유효 시간 초과, 전방 점유 발견 시 토큰 회수.
"""

import math
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool


class DeadlockTokenManager:
    def __init__(self) -> None:
        rospy.init_node("deadlock_token_manager", anonymous=True)

        self.dest_frame = str(rospy.get_param("~frame_id", "map"))
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 5))
        self.vehicle_ids = self._parse_vehicle_ids()
        self.lookahead_m = float(rospy.get_param("~lookahead_m", 8.0))
        self.safe_distance_m = float(rospy.get_param("~safe_distance_m", 2.5))
        self.stop_speed_thresh = float(rospy.get_param("~stop_speed_thresh", 0.3))
        self.stop_time_thresh = float(rospy.get_param("~stop_time_thresh", 2.0))
        self.token_timeout = float(rospy.get_param("~token_timeout", 5.0))
        self.timer_hz = float(rospy.get_param("~check_hz", 5.0))
        self.path_topic_prefix = str(rospy.get_param("~path_topic_prefix", "/global_path_"))
        self.pose_topic_prefix = str(rospy.get_param("~pose_topic_prefix", "/"))
        self.heading_gate_rad = abs(float(rospy.get_param("~heading_gate_deg", 110.0))) * math.pi / 180.0

        # 상태 캐시
        self._poses: Dict[int, Tuple[float, float, float]] = {}  # x, y, stamp_sec
        self._speeds: Dict[int, float] = {}
        self._last_move_time: Dict[int, float] = {vid: 0.0 for vid in self.vehicle_ids}
        self._paths: Dict[int, List[Tuple[float, float]]] = {}
        self._publishers: Dict[int, rospy.Publisher] = {}

        # 토큰
        self._token_vid: Optional[int] = None
        self._token_stamp: float = 0.0

        # 구독/퍼블리셔 설정
        for vid in self.vehicle_ids:
            role = self._role_name(vid)
            pose_topic = f"{self.pose_topic_prefix}{role}/pose" if self.pose_topic_prefix.endswith("/") else f"{self.pose_topic_prefix}{role}/pose"
            path_topic = f"{self.path_topic_prefix}{role}"
            rospy.Subscriber(pose_topic, PoseStamped, self._pose_cb, callback_args=vid, queue_size=10)
            rospy.Subscriber(path_topic, Path, self._path_cb, callback_args=vid, queue_size=1)
            self._publishers[vid] = rospy.Publisher(f"/collision_override/{role}", Bool, queue_size=1, latch=True)
            # 초기 False
            self._publishers[vid].publish(Bool(data=False))

        rospy.Timer(rospy.Duration(1.0 / max(0.1, self.timer_hz)), self._timer_cb)
        # rospy.loginfo(
        #     "[deadlock-token] vehicles=%s lookahead=%.1f safe=%.1f stop(v<=%.2f for %.1fs) timeout=%.1fs",
        #     self.vehicle_ids,
        #     self.lookahead_m,
        #     self.safe_distance_m,
        #     self.stop_speed_thresh,
        #     self.stop_time_thresh,
        #     self.token_timeout,
        # )

    def _parse_vehicle_ids(self):
        raw = rospy.get_param("~vehicle_ids", None)
        ids = []
        if isinstance(raw, list):
            for v in raw:
                try:
                    ids.append(int(v))
                except Exception:
                    pass
        elif isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            for p in parts:
                try:
                    ids.append(int(p))
                except Exception:
                    pass
        if ids:
            return tuple(sorted(set(ids)))
        return tuple(range(1, max(1, self.num_vehicles) + 1))

    def _role_name(self, vid: int) -> str:
        return f"ego_vehicle_{vid}"

    def _pose_cb(self, msg: PoseStamped, vid: int) -> None:
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        t = msg.header.stamp.to_sec()
        if not math.isfinite(t) or t <= 0.0:
            t = rospy.Time.now().to_sec()
        prev = self._poses.get(vid)
        if prev:
            dt = max(1e-3, t - prev[2])
            dist = math.hypot(x - prev[0], y - prev[1])
            spd = dist / dt
            self._speeds[vid] = spd
            if spd > self.stop_speed_thresh:
                self._last_move_time[vid] = t
        else:
            self._last_move_time[vid] = t
        self._poses[vid] = (x, y, t)

    def _path_cb(self, msg: Path, vid: int) -> None:
        pts: List[Tuple[float, float]] = []
        for pose in msg.poses:
            pts.append((float(pose.pose.position.x), float(pose.pose.position.y)))
        self._paths[vid] = pts

    def _is_stopped(self, vid: int, now: float) -> bool:
        spd = self._speeds.get(vid, 0.0)
        last_move = self._last_move_time.get(vid, 0.0)
        return spd <= self.stop_speed_thresh and (now - last_move) >= self.stop_time_thresh

    def _project_progress(self, path: List[Tuple[float, float]], px: float, py: float):
        if len(path) < 2:
            return None
        best_dist_sq = float("inf")
        best_idx = 0
        best_t = 0.0
        s_profile = [0.0]
        total = 0.0
        for i in range(1, len(path)):
            seg = math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            total += seg
            s_profile.append(total)
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + dx * t
            proj_y = y1 + dy * t
            dist_sq = (proj_x - px) ** 2 + (proj_y - py) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = i
                best_t = t
        seg_len = max(1e-6, math.hypot(path[best_idx + 1][0] - path[best_idx][0], path[best_idx + 1][1] - path[best_idx][1]))
        s = s_profile[best_idx] + best_t * seg_len
        return s, s_profile

    def _ahead_samples(self, path: List[Tuple[float, float]], s_profile: List[float], s_now: float, lookahead: float):
        samples: List[Tuple[float, float, float]] = []
        if len(path) < 2 or not s_profile:
            return samples
        target = s_now
        end_s = min(s_profile[-1], s_now + max(0.1, lookahead))
        i = 0
        while i < len(s_profile) - 1 and s_profile[i + 1] < s_now:
            i += 1
        while target <= end_s and i < len(s_profile) - 1:
            while i < len(s_profile) - 1 and s_profile[i + 1] < target:
                i += 1
            s0, s1 = s_profile[i], s_profile[i + 1]
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            seg_heading = math.atan2(y1 - y0, x1 - x0)
            if s1 - s0 < 1e-6:
                samples.append((x0, y0, seg_heading))
                target += 0.5
                continue
            t = (target - s0) / (s1 - s0)
            t = max(0.0, min(1.0, t))
            samples.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, seg_heading))
            target += 0.5  # 0.5m 간격 샘플
        if samples:
            end_idx = min(len(path) - 1, i + 1)
            end_heading = math.atan2(path[end_idx][1] - path[end_idx - 1][1], path[end_idx][0] - path[end_idx - 1][0]) if end_idx >= 1 else 0.0
            samples.append((path[end_idx][0], path[end_idx][1], end_heading))
        return samples

    def _path_clear(self, vid: int) -> bool:
        pose = self._poses.get(vid)
        path = self._paths.get(vid)
        if pose is None or not path:
            return False
        proj = self._project_progress(path, pose[0], pose[1])
        if proj is None:
            return False
        s_now, s_profile = proj
        samples = self._ahead_samples(path, s_profile, s_now, self.lookahead_m)
        if not samples:
            return False
        for other_vid, other_pose in self._poses.items():
            if other_vid == vid:
                continue
            ox, oy, _ = other_pose
            for sx, sy, shdg in samples:
                dist = math.hypot(ox - sx, oy - sy)
                if dist <= self.safe_distance_m:
                    # 앞쪽 진행 방향 기준으로 헤딩 게이트 내에 있을 때만 차단
                    bearing = math.atan2(oy - sy, ox - sx)
                    diff = abs((bearing - shdg + math.pi) % (2.0 * math.pi) - math.pi)
                    if diff <= self.heading_gate_rad:
                        return False
        return True

    def _publish_override(self, vid: int, value: bool) -> None:
        pub = self._publishers.get(vid)
        if pub is not None:
            pub.publish(Bool(data=value))

    def _revoke_all(self):
        for vid in self.vehicle_ids:
            self._publish_override(vid, False)

    def _timer_cb(self, _evt) -> None:
        now = rospy.Time.now().to_sec()
        # 토큰 유지 검사
        if self._token_vid is not None:
            # 만료 또는 전방 점유 시 회수
            timed_out = (now - self._token_stamp) >= self.token_timeout
            clear = self._path_clear(self._token_vid)
            if timed_out or not clear:
                # rospy.loginfo("[deadlock-token] revoke vid=%d reason=%s", self._token_vid, "timeout" if timed_out else "blocked")
                self._publish_override(self._token_vid, False)
                self._token_vid = None
        # 토큰 없으면 선택
        if self._token_vid is None:
            stopped = [vid for vid in self.vehicle_ids if self._is_stopped(vid, now)]
            if len(stopped) >= 1:
                cand = sorted(stopped)[0]
                if self._path_clear(cand):
                    self._token_vid = cand
                    self._token_stamp = now
                    # rospy.loginfo("[deadlock-token] grant vid=%d", cand)
                    # 다른 차량은 False
                    self._revoke_all()
                    self._publish_override(cand, True)
        # 안전: 토큰이 없는 차량은 False 유지
        if self._token_vid is None:
            self._revoke_all()


def main() -> None:
    DeadlockTokenManager()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

