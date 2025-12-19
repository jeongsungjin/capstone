#!/usr/bin/env python3
"""
RViz Click Obstacle Manager
- RViz에서 클릭하여 장애물 추가/삭제
- /obstacles 토픽으로 장애물 목록 발행
"""

import math
import threading
from typing import List, Tuple

import rospy
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

# Ensure CARLA Python API is available on sys.path before import
try:
    from setup_carla_path import CARLA_EGG  # type: ignore  # noqa: F401
except Exception:
    pass

try:
    import carla  # type: ignore
except Exception as exc:
    rospy.logfatal(f"Failed to import CARLA package: {exc}")
    carla = None


class RvizObstacleManager:
    """
    RViz에서 클릭하여 장애물을 추가/삭제하는 노드
    
    모드:
    - 'add': 클릭 위치에 장애물 추가
    - 'remove': 클릭 위치 근처 장애물 삭제
    """

    def __init__(self) -> None:
        rospy.init_node("rviz_obstacle_manager", anonymous=False)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        # Parameters
        self.carla_host: str = rospy.get_param("~carla_host", "localhost")
        self.carla_port: int = int(rospy.get_param("~carla_port", 2000))
        self.snap_to_road: bool = bool(rospy.get_param("~snap_to_road", True))
        self.default_z: float = float(rospy.get_param("~default_z", 0.5))
        self.remove_radius: float = float(rospy.get_param("~remove_radius", 5.0))
        self.obstacle_height: float = float(rospy.get_param("~obstacle_height", 2.0))
        self.obstacle_radius: float = float(rospy.get_param("~obstacle_radius", 1.0))
        self.publish_rate: float = float(rospy.get_param("~publish_rate", 1.0))

        # CARLA client
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        # State
        self._lock = threading.RLock()
        self._obstacles: List[Tuple[float, float, float]] = []
        self._mode: str = "add"  # 'add' or 'remove'

        # Publishers
        self._obstacle_pub = rospy.Publisher("/obstacles", PoseArray, queue_size=1, latch=True)
        self._marker_pub = rospy.Publisher("/obstacle_markers", MarkerArray, queue_size=1, latch=True)

        # Subscribers
        rospy.Subscriber("/obstacle_mode", String, self._mode_cb, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._goal_cb, queue_size=10)
        rospy.Subscriber("/clicked_point", PoseStamped, self._click_cb, queue_size=10)

        # 주기적 발행
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self._publish_cb)

        rospy.loginfo("RvizObstacleManager initialized (mode: %s)", self._mode)
        rospy.loginfo("  - Click on RViz to add obstacles")
        rospy.loginfo("  - Publish to /obstacle_mode: 'add' or 'remove'")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _snap_to_road_height(self, x: float, y: float) -> float:
        """도로 높이에 맞춤"""
        if not self.snap_to_road:
            return self.default_z
        try:
            loc = carla.Location(x=x, y=y, z=0.5)
            wp = self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                return float(wp.transform.location.z) + 0.1
        except Exception:
            pass
        return self.default_z

    def _find_nearest_obstacle(self, x: float, y: float) -> int:
        """가장 가까운 장애물 인덱스 반환 (-1 if none)"""
        best_idx = -1
        best_dist = float('inf')
        for i, (ox, oy, oz) in enumerate(self._obstacles):
            dist = math.hypot(x - ox, y - oy)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist > self.remove_radius:
            return -1
        return best_idx

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _mode_cb(self, msg: String) -> None:
        """모드 전환 콜백"""
        mode = msg.data.strip().lower()
        if mode in ("add", "remove", "clear"):
            with self._lock:
                if mode == "clear":
                    self._obstacles.clear()
                    rospy.loginfo("RvizObstacleManager: cleared all obstacles")
                    self._publish_obstacles()
                else:
                    self._mode = mode
                    rospy.loginfo("RvizObstacleManager: mode = %s", mode)

    def _goal_cb(self, msg: PoseStamped) -> None:
        """2D Nav Goal 클릭 콜백"""
        self._handle_click(msg.pose.position.x, msg.pose.position.y)

    def _click_cb(self, msg: PoseStamped) -> None:
        """Publish Point 클릭 콜백"""
        self._handle_click(msg.pose.position.x, msg.pose.position.y)

    def _handle_click(self, x: float, y: float) -> None:
        """클릭 처리"""
        with self._lock:
            mode = self._mode

        if mode == "add":
            z = self._snap_to_road_height(x, y)
            with self._lock:
                self._obstacles.append((x, y, z))
            rospy.loginfo("RvizObstacleManager: added obstacle at (%.1f, %.1f, %.1f)", x, y, z)
        elif mode == "remove":
            with self._lock:
                idx = self._find_nearest_obstacle(x, y)
                if idx >= 0:
                    removed = self._obstacles.pop(idx)
                    rospy.loginfo("RvizObstacleManager: removed obstacle at (%.1f, %.1f)", removed[0], removed[1])
                else:
                    rospy.logwarn("RvizObstacleManager: no obstacle near (%.1f, %.1f) to remove", x, y)

        self._publish_obstacles()

    def _publish_cb(self, _evt) -> None:
        """주기적 장애물 발행"""
        self._publish_obstacles()

    def _publish_obstacles(self) -> None:
        """장애물 토픽 & 마커 발행"""
        with self._lock:
            obstacles = list(self._obstacles)

        # PoseArray 발행
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        for ox, oy, oz in obstacles:
            pose = Pose()
            pose.position.x = ox
            pose.position.y = oy
            pose.position.z = oz
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self._obstacle_pub.publish(msg)

        # MarkerArray 발행 (RViz 시각화)
        markers = MarkerArray()
        
        # 기존 마커 삭제
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)
        
        # 새 마커 추가
        for i, (ox, oy, oz) in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = ox
            marker.pose.position.y = oy
            marker.pose.position.z = oz + self.obstacle_height / 2
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.obstacle_radius * 2
            marker.scale.y = self.obstacle_radius * 2
            marker.scale.z = self.obstacle_height
            marker.color.r = 1.0
            marker.color.g = 0.3
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker.lifetime = rospy.Duration(0)
            markers.markers.append(marker)

        self._marker_pub.publish(markers)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API (for external use)
    # ─────────────────────────────────────────────────────────────────────────

    def add_obstacle(self, x: float, y: float, z: float = None) -> None:
        """프로그래밍으로 장애물 추가"""
        if z is None:
            z = self._snap_to_road_height(x, y)
        with self._lock:
            self._obstacles.append((x, y, z))
        self._publish_obstacles()

    def remove_obstacle(self, x: float, y: float) -> bool:
        """프로그래밍으로 장애물 삭제"""
        with self._lock:
            idx = self._find_nearest_obstacle(x, y)
            if idx >= 0:
                self._obstacles.pop(idx)
                self._publish_obstacles()
                return True
        return False

    def clear_obstacles(self) -> None:
        """모든 장애물 삭제"""
        with self._lock:
            self._obstacles.clear()
        self._publish_obstacles()

    def get_obstacles(self) -> List[Tuple[float, float, float]]:
        """장애물 목록 반환"""
        with self._lock:
            return list(self._obstacles)


if __name__ == "__main__":
    try:
        RvizObstacleManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
