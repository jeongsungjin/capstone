#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header


class PathRelay:
    def __init__(self):
        rospy.init_node("path_relay", anonymous=True)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.leader_role = str(rospy.get_param("~leader_role", "ego_vehicle_1"))
        self.platoon_enable = bool(rospy.get_param("~platoon_enable", False))
        # follower 위치 기준 트림 파라미터
        self.max_trim_distance_m = float(rospy.get_param("~max_trim_distance_m", 50.0))
        self.backtrack_window_m = float(rospy.get_param("~backtrack_window_m", 8.0))
        ignore_str = str(rospy.get_param("~ignore_roles", "")).strip()
        self.ignore_roles = set([s.strip() for s in ignore_str.split(",") if s.strip()]) if ignore_str else set()

        self._publishers = {}
        self._odom = {}
        for index in range(self.num_vehicles):
            role = f"ego_vehicle_{index + 1}"
            if role in self.ignore_roles:
                continue
            dst_topic = f"/planned_path_{role}"
            pub = rospy.Publisher(dst_topic, Path, queue_size=1, latch=True)
            self._publishers[role] = pub
            rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role, queue_size=10)
            # follower별 개별 경로 전달은 platoon 비활성 시에만 사용
            if not self.platoon_enable:
                src_topic = f"/global_path_{role}"
                rospy.Subscriber(src_topic, Path, self._follower_cb, callback_args=role, queue_size=1)

        # platoon 모드일 때만 리더 경로를 모든 차량에 복제
        if self.platoon_enable:
            src_topic = f"/global_path_{self.leader_role}"
            rospy.Subscriber(src_topic, Path, self._leader_cb, queue_size=1)
            rospy.loginfo("path_relay platoon mode ON: %s -> planned_path_*", src_topic)
        else:
            rospy.loginfo("path_relay platoon mode OFF: /global_path_* -> /planned_path_* (per role)")

    def _leader_cb(self, msg: Path):
        for role, pub in self._publishers.items():
            out = msg if role == self.leader_role else self._trim_for_role(msg, role)
            repub = self._renumber_path(out)
            if repub.poses:
                rospy.logfatal(
                    "[RELAY] src=%s dst=%s n=%d seq_min=%d seq_max=%d first=(%.2f,%.2f) last=(%.2f,%.2f)",
                    f"/global_path_{self.leader_role}",
                    f"/planned_path_{role}",
                    len(repub.poses),
                    repub.poses[0].header.seq,
                    repub.poses[-1].header.seq,
                    repub.poses[0].pose.position.x,
                    repub.poses[0].pose.position.y,
                    repub.poses[-1].pose.position.x,
                    repub.poses[-1].pose.position.y,
                )
            pub.publish(repub)

    def _follower_cb(self, msg: Path, role: str):
        if role in self.ignore_roles:
            return
        pub = self._publishers.get(role)
        if pub is None:
            return
        trimmed = self._trim_for_role(msg, role)
        repub = self._renumber_path(trimmed)
        if repub.poses:
            rospy.logfatal(
                "[RELAY] src=%s dst=%s n=%d seq_min=%d seq_max=%d first=(%.2f,%.2f) last=(%.2f,%.2f)",
                f"/global_path_{role}",
                f"/planned_path_{role}",
                len(repub.poses),
                repub.poses[0].header.seq,
                repub.poses[-1].header.seq,
                repub.poses[0].pose.position.x,
                repub.poses[0].pose.position.y,
                repub.poses[-1].pose.position.x,
                repub.poses[-1].pose.position.y,
            )
        pub.publish(repub)

    def _odom_cb(self, msg: Odometry, role: str):
        self._odom[role] = msg

    def _project_s(self, pts, s_profile, x, y):
        if len(pts) < 2 or len(s_profile) != len(pts):
            return None
        best = None
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg = dx * dx + dy * dy
            if seg < 1e-9:
                continue
            t = ((x - x1) * dx + (y - y1) * dy) / seg
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy
            dist2 = (x - px) ** 2 + (y - py) ** 2
            s_proj = s_profile[i] + math.sqrt(seg) * t
            if best is None or dist2 < best[0]:
                best = (dist2, s_proj, i, t, px, py)
        return best

    def _compute_s_profile(self, pts):
        if len(pts) < 2:
            return []
        s = [0.0]
        total = 0.0
        for i in range(1, len(pts)):
            step = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            total += step
            s.append(total)
        return s

    def _trim_for_role(self, msg: Path, role: str) -> Path:
        if len(msg.poses) < 2:
            return msg
        odom = self._odom.get(role)
        if odom is None:
            return msg
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        s_profile = self._compute_s_profile(pts)
        proj = self._project_s(pts, s_profile, odom.pose.pose.position.x, odom.pose.pose.position.y)
        if proj is None:
            return msg
        dist2, s_proj, idx, t, px, py = proj
        dist = math.sqrt(dist2)
        if dist > self.max_trim_distance_m:
            return msg
        idx = max(0, min(idx, len(msg.poses) - 2))
        start_pose = PoseStamped()
        start_pose.header = msg.header
        x1, y1 = pts[idx]
        x2, y2 = pts[idx + 1]
        dx = x2 - x1
        dy = y2 - y1
        yaw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else 0.0
        start_pose.pose.position.x = px
        start_pose.pose.position.y = py
        start_pose.pose.position.z = msg.poses[idx].pose.position.z
        start_pose.pose.orientation.x = 0.0
        start_pose.pose.orientation.y = 0.0
        start_pose.pose.orientation.z = math.sin(0.5 * yaw)
        start_pose.pose.orientation.w = math.cos(0.5 * yaw)
        # seq는 기존 세그먼트의 seq를 사용해 연속성 유지
        start_pose.header.seq = msg.poses[idx].header.seq
        trimmed = Path()
        trimmed.header = msg.header
        trimmed.poses = [start_pose] + msg.poses[idx + 1 :]
        # 최소 3포인트 미만이면 트림하지 않고 원본 유지 (투영 불안정 방지)
        if len(trimmed.poses) < 5:
            return msg
        return trimmed

    def _renumber_path(self, msg: Path) -> Path:
        """seq가 큰 에폭 값을 갖고 있을 때 진행률 계산이 꼬이지 않도록 0..N-1로 재부여."""
        if not msg.poses:
            return msg
        out = Path()
        out.header = msg.header
        for i, ps in enumerate(msg.poses):
            p = PoseStamped()
            # 각 pose마다 독립 Header를 생성해 seq를 안전하게 부여
            p.header = Header()
            p.header.frame_id = msg.header.frame_id
            p.header.stamp = msg.header.stamp
            p.header.seq = i
            p.pose = ps.pose
            out.poses.append(p)
        return out


if __name__ == "__main__":
    try:
        PathRelay()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


