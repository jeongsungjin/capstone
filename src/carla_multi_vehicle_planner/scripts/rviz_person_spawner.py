#!/usr/bin/env python3

import math
import sys
from typing import Optional

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


# Ensure CARLA egg on path (align with other scripts)
CARLA_EGG = "/home/ctrl/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg"
if CARLA_EGG not in sys.path:
    sys.path.insert(0, CARLA_EGG)

try:
    import carla  # type: ignore
except ImportError as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RvizPersonSpawner:
    def __init__(self) -> None:
        rospy.init_node("rviz_person_spawner", anonymous=False)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.enabled: bool = False
        self.world: Optional[carla.World] = None
        self.map: Optional[carla.Map] = None
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Parameters
        self.blueprint_id = str(rospy.get_param("~blueprint", "walker.pedestrian.0001"))
        self.z_offset = float(rospy.get_param("~z_offset", 0.2))
        self.output_topic = str(rospy.get_param("~output_topic", "/spawned_person"))

        self.pose_pub = rospy.Publisher(self.output_topic, PoseStamped, queue_size=1, latch=True)
        rospy.Subscriber("/spawn_person_mode", Bool, self._mode_cb, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._goal_cb, queue_size=10)

        rospy.loginfo("rviz_person_spawner ready. Press 'o' in key_selector to toggle mode.")

    def _mode_cb(self, msg: Bool) -> None:
        self.enabled = bool(msg.data)
        rospy.loginfo("rviz_person_spawner: mode %s", "ON" if self.enabled else "OFF")

    def _goal_cb(self, msg: PoseStamped) -> None:
        if not self.enabled:
            return
        pose = msg.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        yaw = yaw_from_quat(pose.orientation)
        # Snap to nearest sidewalk/shoulder/parking (fallback: driving) for proper ground Z
        lane_mask = carla.LaneType.Sidewalk | carla.LaneType.Shoulder | carla.LaneType.Parking | carla.LaneType.Driving
        wp = self.map.get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True, lane_type=lane_mask)
        if wp is not None:
            base_loc = wp.transform.location
            sx = float(base_loc.x)
            sy = float(base_loc.y)
            sz = float(base_loc.z) + self.z_offset
            if math.isnan(yaw) or abs(yaw) < 1e-6:
                yaw = math.radians(float(wp.transform.rotation.yaw))
        else:
            sx = x
            sy = y
            sz = float(pose.position.z) if pose.position.z != 0.0 else self.z_offset
        try:
            bp_lib = self.world.get_blueprint_library()
            if self.blueprint_id:
                bp = bp_lib.find(self.blueprint_id)
            else:
                # fallback to first walker
                walkers = bp_lib.filter("walker.pedestrian.*") or bp_lib.filter("walker.*")
                bp = walkers[0]
            tf = carla.Transform(location=carla.Location(x=sx, y=sy, z=sz), rotation=carla.Rotation(yaw=math.degrees(yaw)))
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is None:
                rospy.logwarn("rviz_person_spawner: spawn failed at (%.2f, %.2f)", sx, sy)
                return
            # publish spawned pose
            out = PoseStamped()
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = msg.header.frame_id or "map"
            out.pose.position.x = sx
            out.pose.position.y = sy
            out.pose.position.z = sz
            out.pose.orientation = pose.orientation
            self.pose_pub.publish(out)
            try:
                self.world.debug.draw_string(tf.location, "PEDESTRIAN", life_time=2.0, color=carla.Color(0, 255, 0))
            except Exception:
                pass
            rospy.loginfo("rviz_person_spawner: spawned %s id=%s at (%.2f, %.2f, %.2f)", bp.id, actor.id, sx, sy, sz)
        except Exception as exc:
            rospy.logwarn("rviz_person_spawner: exception during spawn: %s", exc)


if __name__ == "__main__":
    try:
        RvizPersonSpawner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


