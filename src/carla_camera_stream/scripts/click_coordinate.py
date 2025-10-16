#!/usr/bin/env python3

import math
import os
import sys

import rospy
from geometry_msgs.msg import Pose, Quaternion, PoseStamped
from nav_msgs.msg import OccupancyGrid

# CARLA Python API 경로 설정
default_build_path = "/home/jamie/carla/PythonAPI/carla/build/lib.linux-x86_64-cpython-38"
CARLA_BUILD_PATH = os.environ.get("CARLA_BUILD_PATH", default_build_path)

if CARLA_BUILD_PATH and CARLA_BUILD_PATH not in sys.path:
    sys.path.insert(0, CARLA_BUILD_PATH)

try:
    import carla
except ImportError as exc:
    rospy.logfatal("Failed to import CARLA: %s", exc)
    sys.exit(1)


class SimpleGoalLogger:
    def __init__(self):
        rospy.init_node("simple_goal_logger", anonymous=True)
        
        # CARLA 클라이언트 설정
        self._client = None
        self._map_msg = None
        self._resolution = 0.25  # 맵 해상도
        self._sample_distance = 2.0  # waypoint 샘플링 거리
        self._expansion_radius = 2  # 도로 확장 반경
        
        # RViz의 2D Nav Goal topic을 구독
        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", 
            PoseStamped, 
            self.goal_callback, 
            queue_size=10
        )
        
        # Occupancy Grid Publisher
        self.map_publisher = rospy.Publisher("/map", OccupancyGrid, queue_size=1, latch=True)
        
        rospy.loginfo("Simple Goal Logger: Ready to log mouse click coordinates")
        rospy.loginfo("Click '2D Nav Goal' in RViz and click anywhere to see coordinates")
        
        # CARLA 맵 생성 및 발행
        self._build_and_publish_map()

    def goal_callback(self, msg):
        """RViz에서 2D Nav Goal 클릭 시 호출되는 콜백 함수"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        # x, y 좌표만 출력
        rospy.loginfo("Mouse clicked at: (%.2f, %.2f)", x, y)

    def _ensure_client(self):
        """CARLA 클라이언트 생성"""
        if self._client is None:
            try:
                self._client = carla.Client("localhost", 2000)
                self._client.set_timeout(5.0)
                rospy.loginfo("Connected to CARLA server")
            except RuntimeError as exc:
                rospy.logerr("Failed to create CARLA client: %s", exc)
                self._client = None
        return self._client

    def _build_and_publish_map(self):
        """CARLA 맵을 생성하고 발행"""
        client = self._ensure_client()
        if client is None:
            rospy.logwarn("CARLA client not available, skipping map generation")
            return

        try:
            world = client.get_world()
            carla_map = world.get_map()
            
            if carla_map is None:
                rospy.logwarn("CARLA world has no map loaded")
                return
                
            waypoints = self._collect_waypoints(carla_map)
            if not waypoints:
                rospy.logwarn("No waypoints available from CARLA map")
                return
                
            self._map_msg = self._create_grid_from_waypoints(waypoints)
            if self._map_msg is not None:
                timestamp = rospy.Time.now()
                self._map_msg.header.stamp = timestamp
                self._map_msg.info.map_load_time = timestamp
                self.map_publisher.publish(self._map_msg)
                
                rospy.loginfo("Published CARLA map: %dx%d at %.2f m/cell", 
                             self._map_msg.info.width, 
                             self._map_msg.info.height, 
                             self._map_msg.info.resolution)
            else:
                rospy.logwarn("Failed to create occupancy grid from waypoints")
                
        except RuntimeError as exc:
            rospy.logerr("Failed to access CARLA world: %s", exc)
            self._client = None

    def _collect_waypoints(self, carla_map):
        """CARLA 맵에서 waypoint 수집"""
        try:
            waypoints = carla_map.generate_waypoints(self._sample_distance)
        except RuntimeError as exc:
            rospy.logwarn("Waypoint generation failed (%s); falling back to topology", exc)
            waypoints = []

        if not waypoints:
            topology = carla_map.get_topology()
            for start_wp, end_wp in topology:
                if start_wp is not None:
                    waypoints.append(start_wp)
                if end_wp is not None:
                    waypoints.append(end_wp)

        # 중복 제거
        unique = {}
        for wp in waypoints:
            loc = wp.transform.location
            key = (round(loc.x, 2), round(loc.y, 2))
            unique[key] = wp
        return list(unique.values())

    def _create_grid_from_waypoints(self, waypoints):
        """Waypoint에서 occupancy grid 생성"""
        xs = [wp.transform.location.x for wp in waypoints]
        ys = [wp.transform.location.y for wp in waypoints]

        if not xs or not ys:
            rospy.logwarn("Waypoint list missing coordinates")
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

        # 도로 확장
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

    def run(self):
        """노드 실행"""
        rospy.spin()


if __name__ == "__main__":
    try:
        logger = SimpleGoalLogger()
        logger.run()
    except rospy.ROSInterruptException:
        pass
