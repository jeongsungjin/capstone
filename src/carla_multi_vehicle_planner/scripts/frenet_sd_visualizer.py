#!/usr/bin/env python3
import math
import threading
from typing import List, Tuple

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker

# Ensure local package scripts are importable
try:
    from frenet import CubicSpline2D, catesian_to_frenet
except Exception:
    # Fallback when executed without proper PYTHONPATH
    from .frenet import CubicSpline2D, catesian_to_frenet  # type: ignore


class FrenetSDVisualizer:
    def __init__(self):
        rospy.init_node("frenet_sd_visualizer", anonymous=True)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.interval = float(rospy.get_param("~spline_interval", 0.2))
        self.frame_id = rospy.get_param("~frame_id", "map")

        self._lock = threading.RLock()

        # Per-vehicle state
        self._csp = {}
        self._last_xy_path = {}

        # Publishers and Subscribers per role
        self._sd_marker_pubs = {}
        self._xy_marker_pubs = {}
        self._sd_path_pubs = {}

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            xy_topic = f"/global_path_{role}"
            rospy.Subscriber(xy_topic, Path, self._path_cb, callback_args=role, queue_size=1)

            self._sd_marker_pubs[role] = rospy.Publisher(f"/frenet/{role}/sd_markers", Marker, queue_size=1, latch=True)
            self._xy_marker_pubs[role] = rospy.Publisher(f"/frenet/{role}/xy_markers", Marker, queue_size=1, latch=True)
            self._sd_path_pubs[role] = rospy.Publisher(f"/frenet/{role}/sd_path", Path, queue_size=1, latch=True)

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _path_cb(self, msg: Path, role: str) -> None:
        with self._lock:
            points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            if len(points) < 2:
                return
            self._last_xy_path[role] = points

            # Build spline centerline from XY path
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            try:
                csp = CubicSpline2D(xs, ys, interval=self.interval)
            except Exception as exc:
                rospy.logwarn(f"{role}: failed to build spline: {exc}")
                return
            self._csp[role] = csp

            # Convert the same XY samples to Frenet s,d
            sd_points: List[Tuple[float, float]] = []
            for (x, y) in points:
                s, d = catesian_to_frenet(x, y, csp)
                sd_points.append((s, d))

            # Publish visualizations
            self._publish_xy_marker(role, points)
            self._publish_sd_path(role, sd_points)
            self._publish_sd_marker(role, sd_points)

    def _publish_xy_marker(self, role: str, points: List[Tuple[float, float]]) -> None:
        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        marker.ns = f"xy_path_{role}"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.color = self._role_color_rgba(role, alpha=0.9)
        marker.lifetime = rospy.Duration(0)
        marker.points = [Point(x=x, y=y, z=0.1) for (x, y) in points]
        self._xy_marker_pubs[role].publish(marker)

    def _publish_sd_marker(self, role: str, sd_points: List[Tuple[float, float]]) -> None:
        # Visualize s,d in an auxiliary frame projected into map as (s, d)
        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        marker.ns = f"sd_curve_{role}"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.color = self._role_color_rgba(role, alpha=0.9)
        marker.lifetime = rospy.Duration(0)
        marker.points = [Point(x=s, y=d, z=0.3) for (s, d) in sd_points]
        self._sd_marker_pubs[role].publish(marker)

    def _publish_sd_path(self, role: str, sd_points: List[Tuple[float, float]]) -> None:
        # Publish Path where poses use (s,d) as x,y in frame_id=map so it can be viewed in RViz
        header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        path_msg = Path(header=header)
        poses: List[PoseStamped] = []
        for (s, d) in sd_points:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = s
            pose.pose.position.y = d
            pose.pose.position.z = 0.0
            poses.append(pose)
        path_msg.poses = poses
        self._sd_path_pubs[role].publish(path_msg)

    @staticmethod
    def _role_color_rgba(role: str, alpha: float = 0.9) -> ColorRGBA:
        palette_bgr = {
            "ego_vehicle_1": (0, 255, 0),
            "ego_vehicle_2": (255, 0, 0),
            "ego_vehicle_3": (0, 0, 255),
            "ego_vehicle_4": (255, 255, 0),
            "ego_vehicle_5": (255, 0, 255),
            "ego_vehicle_6": (0, 255, 255),
        }
        b, g, r = palette_bgr.get(role, (200, 200, 200))
        return ColorRGBA(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=alpha)


def main() -> None:
    FrenetSDVisualizer()
    rospy.loginfo("frenet_sd_visualizer started")
    rospy.spin()


if __name__ == "__main__":
    main()


