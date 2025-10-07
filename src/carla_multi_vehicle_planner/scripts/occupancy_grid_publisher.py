#!/usr/bin/env python3

import math
import os
import sys
from typing import TYPE_CHECKING

import rospy
from geometry_msgs.msg import Pose, Quaternion
from nav_msgs.msg import OccupancyGrid

# Ensure CARLA Python API is available on sys.path before import
try:
    import setup_carla_path  # pylint: disable=unused-import
except ImportError:
    setup_carla_path = None  # noqa: F841 so linters do not complain

default_egg = "/home/ctrl/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg"
CARLA_EGG_PATH = os.environ.get("CARLA_EGG") or getattr(setup_carla_path, "CARLA_EGG", None) or default_egg

if CARLA_EGG_PATH and CARLA_EGG_PATH not in sys.path:
    sys.path.insert(0, CARLA_EGG_PATH)

try:
    import carla  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime with ROS logging
    rospy.logfatal(
        "occupancy_grid_publisher: failed to import CARLA egg %s under Python %s: %s",
        CARLA_EGG_PATH,
        sys.version.replace("\n", " "),
        exc,
    )
    sys.exit(1)

if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    import carla as carla_types


class CarlaOccupancyGridPublisher:
    """Publishes an occupancy grid representation of CARLA roads."""

    def __init__(self) -> None:
        self._node_name = rospy.get_name() or "occupancy_grid_publisher"
        self._resolution = float(rospy.get_param("~resolution", 0.25))
        if self._resolution <= 0:
            rospy.logwarn("%s: resolution <= 0 requested, falling back to 0.25", self._node_name)
            self._resolution = 0.25

        self._update_rate = float(rospy.get_param("~update_rate", 1.0))
        if self._update_rate <= 0:
            rospy.logwarn("%s: update_rate <= 0 requested, falling back to 1.0", self._node_name)
            self._update_rate = 1.0

        self._publisher = rospy.Publisher("/map", OccupancyGrid, queue_size=1, latch=True)
        self._client = None
        self._map_msg = None
        # Sampling distance should not be too small to avoid enormous waypoint lists.
        self._sample_distance = max(self._resolution, 1.0)
        # Slight inflation radius to make roads look continuous.
        self._expansion_radius = max(1, int(math.ceil(0.5 / self._resolution)))

    def start(self) -> None:
        rate = rospy.Rate(self._update_rate)
        while not rospy.is_shutdown():
            if self._map_msg is None:
                self._map_msg = self._build_map_message()
                if self._map_msg is None:
                    rate.sleep()
                    continue
                rospy.loginfo(
                    "%s: generated occupancy grid %dx%d at %.2f m/cell",
                    self._node_name,
                    self._map_msg.info.width,
                    self._map_msg.info.height,
                    self._map_msg.info.resolution,
                )

            timestamp = rospy.Time.now()
            self._map_msg.header.stamp = timestamp
            self._map_msg.info.map_load_time = timestamp
            self._publisher.publish(self._map_msg)
            rate.sleep()

    def _ensure_client(self):
        if self._client is None:
            try:
                self._client = carla.Client("localhost", 2000)
                self._client.set_timeout(5.0)
            except RuntimeError as exc:
                rospy.logerr("%s: failed to create CARLA client: %s", self._node_name, exc)
                self._client = None
        return self._client

    def _build_map_message(self) -> OccupancyGrid:
        client = self._ensure_client()
        if client is None:
            return None

        try:
            world = client.get_world()
        except RuntimeError as exc:
            rospy.logerr("%s: failed to access CARLA world: %s", self._node_name, exc)
            self._client = None
            return None

        carla_map = world.get_map()
        if carla_map is None:
            rospy.logwarn("%s: CARLA world has no map loaded", self._node_name)
            return None

        waypoints = self._collect_waypoints(carla_map)
        if not waypoints:
            rospy.logwarn("%s: no waypoints available from CARLA map", self._node_name)
            return None

        return self._create_grid_from_waypoints(waypoints)

    def _collect_waypoints(self, carla_map) -> list:
        try:
            waypoints = carla_map.generate_waypoints(self._sample_distance)
        except RuntimeError as exc:
            rospy.logwarn("%s: waypoint generation failed (%s); falling back to topology", self._node_name, exc)
            waypoints = []

        if not waypoints:
            topology = carla_map.get_topology()
            for start_wp, end_wp in topology:
                if start_wp is not None:
                    waypoints.append(start_wp)
                if end_wp is not None:
                    waypoints.append(end_wp)

        # Deduplicate by location index to keep the grid manageable.
        unique = {}
        for wp in waypoints:
            loc = wp.transform.location
            key = (round(loc.x, 2), round(loc.y, 2))
            unique[key] = wp
        return list(unique.values())

    def _create_grid_from_waypoints(self, waypoints: list) -> OccupancyGrid:
        xs = [wp.transform.location.x for wp in waypoints]
        ys = [wp.transform.location.y for wp in waypoints]

        if not xs or not ys:
            rospy.logwarn("%s: waypoint list missing coordinates", self._node_name)
            return None

        margin = max(self._resolution * 4.0, 5.0)
        min_x = min(xs) - margin
        max_x = max(xs) + margin
        min_y = min(ys) - margin
        max_y = max(ys) + margin

        span_x = max(max_x - min_x, self._resolution)
        span_y = max(max_y - min_y, self._resolution)

        width = max(1, int(math.ceil(span_x / self._resolution)))
        height = max(1, int(math.ceil(span_y / self._resolution)))

        data = [-1] * (width * height)
        free_cells = set()

        for wp in waypoints:
            loc = wp.transform.location
            col_f = (loc.x - min_x) / self._resolution
            row_f = (loc.y - min_y) / self._resolution
            if not (math.isfinite(col_f) and math.isfinite(row_f)):
                continue
            col = int(col_f)
            row = int(row_f)
            col = min(max(col, 0), width - 1)
            row = min(max(row, 0), height - 1)
            free_cells.add((row, col))

        expanded_cells = set()
        radius_sq = self._expansion_radius * self._expansion_radius
        for row, col in free_cells:
            for d_row in range(-self._expansion_radius, self._expansion_radius + 1):
                for d_col in range(-self._expansion_radius, self._expansion_radius + 1):
                    if d_row * d_row + d_col * d_col > radius_sq:
                        continue
                    new_row = row + d_row
                    new_col = col + d_col
                    if 0 <= new_row < height and 0 <= new_col < width:
                        expanded_cells.add((new_row, new_col))

        for row, col in expanded_cells:
            index = row * width + col
            data[index] = 0  # free cell

        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.info.resolution = self._resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin = Pose()
        grid.info.origin.position.x = min_x
        grid.info.origin.position.y = min_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        grid.data = data
        return grid


def main() -> None:
    rospy.init_node("occupancy_grid_publisher", anonymous=False)
    rospy.loginfo(
        "occupancy_grid_publisher: Python runtime %s", sys.version.replace("\n", " ")
    )
    for idx, path_entry in enumerate(sys.path[:5]):
        rospy.loginfo("occupancy_grid_publisher: sys.path[%d]=%s", idx, path_entry)
    if len(sys.path) > 5:
        rospy.loginfo("occupancy_grid_publisher: ... (%d entries total)", len(sys.path))
    publisher = CarlaOccupancyGridPublisher()
    publisher.start()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
