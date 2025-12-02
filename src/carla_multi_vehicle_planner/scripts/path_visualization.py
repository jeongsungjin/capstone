#!/usr/bin/env python3

import threading
from typing import Dict, Optional, List, Tuple

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
        self.thickness = float(rospy.get_param("~thickness", 0.2))
        self.redraw_period = float(rospy.get_param("~redraw_period", 0.5))  # seconds; keep lines alive until finished
        # Visual intensity controls
        self.brightness_scale = float(rospy.get_param("~brightness_scale", 0.0))  # 0.0..1.0 → closer to white
        self.intensity_scale = float(rospy.get_param("~intensity_scale", 1.0))  # 0.0..1.0 → darker
        self.life_multiplier = float(rospy.get_param("~life_multiplier", 1.2))  # line life relative to redraw period
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
        # role -> cached path points
        self.current_paths: Dict[str, List[Tuple[float, float]]] = {}
        # roles currently redrawn until finished event
        self.active_roles: Dict[str, bool] = {}

        # Color palette (RGB) – canonical mapping
        self._color_map = {
            "yellow": carla.Color(r=255, g=255, b=0),
            "green": carla.Color(r=0, g=255, b=0),
            "red": carla.Color(r=255, g=0, b=0),
            "purple": carla.Color(r=128, g=0, b=128),
            "pink": carla.Color(r=255, g=192, b=203),
            "white": carla.Color(r=255, g=255, b=255),
        }
        # Default index-based colors (legacy fallback)
        self._default_colors = [
            self._color_map["yellow"],
            self._color_map["green"],
            self._color_map["red"],
            self._color_map["purple"],
            self._color_map["pink"],
            self._color_map["white"],
        ]
        # Optional: honor color_list order so UI path colors match spawned vehicle colors
        raw_colors = str(rospy.get_param("~color_list", "")).strip()
        active_colors = [c.strip().lower() for c in raw_colors.split(",") if c.strip()] if raw_colors else []
        if active_colors:
            self._role_colors = [self._color_map.get(c, self._default_colors[i % len(self._default_colors)])
                                 for i, c in enumerate(active_colors)]
        else:
            self._role_colors = []

        # Subs
        rospy.Subscriber("/selected_vehicle", String, self._selected_cb)
        rospy.Subscriber("/override_goal_finished", String, self._finished_cb, queue_size=10)
        for i in range(1, self.num_vehicles + 1):
            role = f"ego_vehicle_{i}"
            rospy.Subscriber(f"/override_goal/{role}", PoseStamped, self._override_cb, callback_args=role, queue_size=1)
            # UI 시각화는 오프셋 적용 전 경로를 구독
            rospy.Subscriber(f"/global_path_ui_{role}", Path, self._path_cb, callback_args=role, queue_size=1)

        # Periodic redraw to persist visualization until finished signal
        rospy.Timer(rospy.Duration(max(0.05, self.redraw_period)), self._timer_cb)

        rospy.loginfo("path_visualization: ready (life_time=%.1fs, window=%.1fs, redraw=%.2fs, life_mult=%.2f)",
                      self.life_time, self.override_to_path_window, self.redraw_period, self.life_multiplier)

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

        # Cache path and activate continuous redraw until finished signal
        with self._lock:
            self.current_paths[role] = pts
            self.active_roles[role] = True
        rospy.loginfo("path_visualization: path cached for %s (%d poses); persisting until finished", role, len(pts))

    def _finished_cb(self, msg: String):
        role = (msg.data or "").strip()
        if not role:
            return
        with self._lock:
            if self.active_roles.get(role):
                self.active_roles[role] = False
                # Optionally drop cache so redraw stops immediately
                # Keep last lines visible for one redraw period to fade out
                self.current_paths.pop(role, None)
        rospy.loginfo("path_visualization: received finished for %s; stopping redraw", role)

    def _timer_cb(self, _event):
        # Redraw active roles frequently with short lifetime to keep visible
        with self._lock:
            roles = [r for r, active in self.active_roles.items() if active]
            paths = {r: self.current_paths.get(r) for r in roles}
        for role in roles:
            pts = paths.get(role)
            if not pts or len(pts) < 2:
                continue
            self._draw_path(role, pts, life=max(0.01, self.redraw_period * max(0.1, self.life_multiplier)))

    def _draw_path(self, role: str, pts: List[Tuple[float, float]], life: float):
        color = self._color_for_role(role)
        if self.intensity_scale < 1.0:
            color = self._darken(color, self.intensity_scale)
        if self.brightness_scale > 0.0:
            color = self._brighten(color, self.brightness_scale)
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
                self.world.debug.draw_line(p1, p2, thickness=self.thickness, color=color, life_time=life)
        else:
            for i in range(len(pts) - 1):
                p1 = carla.Location(x=pts[i][0], y=pts[i][1], z=self.altitude_z)
                p2 = carla.Location(x=pts[i + 1][0], y=pts[i + 1][1], z=self.altitude_z)
                self.world.debug.draw_line(p1, p2, thickness=self.thickness, color=color, life_time=life)

    def _color_for_role(self, role: str) -> "carla.Color":
        try:
            idx = max(0, int(role.rsplit("_", 1)[-1]) - 1)
        except Exception:
            idx = 0
        if self._role_colors:
            return self._role_colors[idx % len(self._role_colors)]
        return self._default_colors[idx % len(self._default_colors)]

    @staticmethod
    def _brighten(color: "carla.Color", scale: float) -> "carla.Color":
        """
        Scale color towards white for higher perceived brightness.
        scale in [0,1]: 0=no change, 1=white.
        """
        s = max(0.0, min(1.0, float(scale)))
        r = int(color.r + s * (255 - color.r))
        g = int(color.g + s * (255 - color.g))
        b = int(color.b + s * (255 - color.b))
        r = 0 if r < 0 else (255 if r > 255 else r)
        g = 0 if g < 0 else (255 if g > 255 else g)
        b = 0 if b < 0 else (255 if b > 255 else b)
        return carla.Color(r=r, g=g, b=b)

    @staticmethod
    def _darken(color: "carla.Color", scale: float) -> "carla.Color":
        """
        Multiply color intensity to make it darker.
        scale in [0,1]: 1=no change, 0=black.
        """
        s = max(0.0, min(1.0, float(scale)))
        r = int(color.r * s)
        g = int(color.g * s)
        b = int(color.b * s)
        r = 0 if r < 0 else (255 if r > 255 else r)
        g = 0 if g < 0 else (255 if g > 255 else g)
        b = 0 if b < 0 else (255 if b > 255 else b)
        return carla.Color(r=r, g=g, b=b)


def main():
    try:
        ClickReplanPathVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


