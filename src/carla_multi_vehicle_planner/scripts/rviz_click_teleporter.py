#!/usr/bin/env python3

import math
import threading
from typing import Dict, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

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


class RvizClickTeleporter:
    def __init__(self) -> None:
        rospy.init_node("rviz_click_teleporter", anonymous=False)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        # Parameters
        self.carla_host: str = rospy.get_param("~carla_host", "localhost")
        self.carla_port: int = int(rospy.get_param("~carla_port", 2000))
        self.use_waypoint_heading: bool = bool(rospy.get_param("~use_waypoint_heading", True))
        self.snap_to_waypoint_height: bool = bool(rospy.get_param("~snap_to_waypoint_height", True))
        self.default_z: float = float(rospy.get_param("~default_z", 0.5))
        self.z_offset: float = float(rospy.get_param("~z_offset", 0.0))
        self.disable_collision_check: bool = bool(rospy.get_param("~disable_collision_check", True))

        # CARLA client/world/map
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        # State
        self._lock = threading.RLock()
        self._current_role: str = rospy.get_param("~default_vehicle", "ego_vehicle_1")
        self._actors_by_role: Dict[str, carla.Actor] = {}
        self._last_refresh_sec: float = 0.0
        self._refresh_interval_sec: float = 1.0

        # ROS I/O
        self._force_pub = rospy.Publisher("/force_replan", String, queue_size=1)
        rospy.Subscriber("/selected_vehicle", String, self._selection_cb, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._goal_cb, queue_size=10)

        rospy.loginfo("rviz_click_teleporter: current role = %s", self._current_role)

    # --------------- Helpers ---------------
    def _refresh_actors_if_needed(self) -> None:
        now = rospy.get_time()
        if now - self._last_refresh_sec < self._refresh_interval_sec:
            return
        self._last_refresh_sec = now
        try:
            actors = self.world.get_actors().filter("vehicle.*")
        except Exception:
            return
        by_role: Dict[str, carla.Actor] = {}
        for actor in actors:
            try:
                role = actor.attributes.get("role_name", "")
            except Exception:
                continue
            if role:
                by_role[role] = actor
        with self._lock:
            self._actors_by_role = by_role

    def _resolve_actor(self, role: str) -> Optional[carla.Actor]:
        self._refresh_actors_if_needed()
        with self._lock:
            actor = self._actors_by_role.get(role)
        return actor

    def _snap_height_and_heading(self, x: float, y: float, yaw_from_msg_deg: float) -> Tuple[float, float]:
        z = self.default_z + self.z_offset
        yaw_deg = yaw_from_msg_deg
        try:
            loc = carla.Location(x=x, y=y, z=0.5)
            wp = self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            wp = None
        if wp is not None:
            if self.snap_to_waypoint_height:
                try:
                    z = float(wp.transform.location.z) + self.z_offset
                except Exception:
                    z = self.default_z + self.z_offset
            if self.use_waypoint_heading:
                try:
                    yaw_deg = float(wp.transform.rotation.yaw)
                except Exception:
                    pass
        return z, yaw_deg

    def _apply_transform(self, actor: carla.Actor, transform: carla.Transform, role: str) -> bool:
        disabled = False
        if self.disable_collision_check and hasattr(actor, "set_simulate_physics"):
            try:
                actor.set_simulate_physics(False)
                disabled = True
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "rviz_click_teleporter: failed to disable physics for %s: %s", role, exc)
        try:
            actor.set_transform(transform)
            try:
                if hasattr(actor, "set_target_velocity"):
                    actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                if hasattr(actor, "set_target_angular_velocity"):
                    actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            except Exception:
                pass
            return True
        except Exception as exc:
            rospy.logwarn("rviz_click_teleporter: teleport failed for %s: %s", role, exc)
            return False
        finally:
            if disabled and hasattr(actor, "set_simulate_physics"):
                try:
                    actor.set_simulate_physics(True)
                except Exception as exc:
                    rospy.logwarn_throttle(5.0, "rviz_click_teleporter: failed to re-enable physics for %s: %s", role, exc)

    # --------------- Callbacks ---------------
    def _selection_cb(self, msg: String) -> None:
        role = msg.data.strip()
        if not role:
            return
        with self._lock:
            self._current_role = role
        rospy.loginfo("rviz_click_teleporter: switched to %s", role)

    def _goal_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            role = self._current_role
        if not role:
            rospy.logwarn("rviz_click_teleporter: no role selected; ignoring click")
            return
        actor = self._resolve_actor(role)
        if actor is None:
            rospy.logwarn("rviz_click_teleporter: actor for %s not found", role)
            return

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        # Orientation from RViz (if provided)
        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        # yaw from quaternion
        yaw_from_msg_deg = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

        z, yaw_deg = self._snap_height_and_heading(x, y, yaw_from_msg_deg)

        tf = carla.Transform(
            location=carla.Location(x=x, y=y, z=z),
            rotation=carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg),
        )
        if not self._apply_transform(actor, tf, role):
            return
        rospy.loginfo("rviz_click_teleporter: teleported %s to (%.1f, %.1f, %.1f) yaw=%.1f°", role, x, y, z, yaw_deg)

        # Force replan from new position by notifying planner
        try:
            self._force_pub.publish(String(data=role))
        except Exception:
            pass


if __name__ == "__main__":
    try:
        RvizClickTeleporter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


