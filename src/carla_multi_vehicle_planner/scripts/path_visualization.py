#!/usr/bin/env python3

import threading
from typing import Dict, Optional

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String

try:
    # Ensure CARLA API on path via helper (side-effect inserts sys.path)
    from setup_carla_path import CARLA_BUILD_PATH  # noqa: F401
except Exception:
    pass

try:
    import carla
except ImportError as exc:
    rospy.logfatal(f"CARLA import failed in path_visualization: {exc}")
    carla = None


class ClickReplanPathVisualizer:
    def __init__(self):
        rospy.init_node("path_visualization", anonymous=False)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        # Params
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 4))
        self.life_time = float(rospy.get_param("~life_time", 10.0))  # seconds to keep path visible
        self.override_to_path_window = float(rospy.get_param("~override_to_path_window", 3.0))
        self.thickness = float(rospy.get_param("~thickness", 0.4))
        # Constant altitude mode (legacy)
        self.altitude_z = float(rospy.get_param("~z", 0.5))
        # Road-elevation following mode
        self.z_follow_road = bool(rospy.get_param("~z_follow_road", True))
        self.z_offset_m = float(rospy.get_param("~z_offset_m", 0.5))  # meters above road when following

        # CARLA connection
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # State
        self._lock = threading.RLock()
        self.selected_role: Optional[str] = None
        # role -> rospy.Time when override received
        self.override_time: Dict[str, rospy.Time] = {}

        # Color palette (RGB) – match bev_visualizer/networkx settings
        self.colors = [
            carla.Color(r=0, g=255, b=0),      # 1: green
            carla.Color(r=0, g=0, b=255),      # 2: red
            carla.Color(r=255, g=0, b=0),       # 3: blue
            carla.Color(r=255, g=255, b=0),    # 4: yellow
            carla.Color(r=255, g=0, b=255),    # 5: magenta
            carla.Color(r=0, g=255, b=255),    # 6: cyan
        ]

        # Subs
        rospy.Subscriber("/selected_vehicle", String, self._selected_cb)
        for i in range(1, self.num_vehicles + 1):
            role = f"ego_vehicle_{i}"
            rospy.Subscriber(f"/override_goal/{role}", PoseStamped, self._override_cb, callback_args=role, queue_size=1)
            rospy.Subscriber(f"/global_path_{role}", Path, self._path_cb, callback_args=role, queue_size=1)

        rospy.loginfo("path_visualization: ready (life_time=%.1fs, window=%.1fs)", self.life_time, self.override_to_path_window)

    def _selected_cb(self, msg: String):
        role = (msg.data or "").strip()
        if not role:
            return
        with self._lock:
            self.selected_role = role

    def _override_cb(self, _msg: PoseStamped, role: str):
        # Record override only if it matches current selection
        with self._lock:
            if self.selected_role == role:
                self.override_time[role] = rospy.Time.now()
                rospy.loginfo("path_visualization: override for %s registered", role)

    def _path_cb(self, msg: Path, role: str):
        # Draw only if a recent override happened for this role
        with self._lock:
            t_override = self.override_time.get(role)
        if t_override is None:
            return
        if (msg.header.stamp - t_override) > rospy.Duration(self.override_to_path_window):
            return

        pts = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        if len(pts) < 2:
            return

        color = self.colors[(int(role.rsplit("_", 1)[-1]) - 1) % len(self.colors)]
        # Optionally follow road elevation per point
        if self.z_follow_road:
            carla_map = self.world.get_map()
            def loc_with_road_z(x, y):
                wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is None:
                    return carla.Location(x=x, y=y, z=self.altitude_z)
                return carla.Location(x=x, y=y, z=wp.transform.location.z + self.z_offset_m)
            for i in range(len(pts) - 1):
                p1 = loc_with_road_z(pts[i][0], pts[i][1])
                p2 = loc_with_road_z(pts[i + 1][0], pts[i + 1][1])
                self.world.debug.draw_line(p1, p2, thickness=self.thickness, color=color, life_time=self.life_time)
        else:
            for i in range(len(pts) - 1):
                p1 = carla.Location(x=pts[i][0], y=pts[i][1], z=self.altitude_z)
                p2 = carla.Location(x=pts[i + 1][0], y=pts[i + 1][1], z=self.altitude_z)
                self.world.debug.draw_line(p1, p2, thickness=self.thickness, color=color, life_time=self.life_time)
        rospy.loginfo("path_visualization: drew replan path for %s (%d poses) for %.1fs", role, len(pts), self.life_time)


def main():
    try:
        ClickReplanPathVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


