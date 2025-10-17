#!/usr/bin/env python3

import threading
import sys

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# =======================
# Vehicle Visualization Settings
# =======================
VEHICLE_RECT_W = 16    # 직사각형 가로 (픽셀)
VEHICLE_RECT_H = 12    # 직사각형 세로 (픽셀)

VEHICLE_MARKER_SCALE = 6.0    # RViz 마커 큐브 크기 (meter 단위)
VEHICLE_MARKER_TYPE = Marker.CUBE  # RViz 마커: CUBE 로 변경

CARLA_EGG = "/home/ctrl/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg"
if CARLA_EGG not in sys.path:
    sys.path.insert(0, CARLA_EGG)

try:
    import carla
except ImportError as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


class BEVVisualizer:
    def __init__(self):
        rospy.init_node("bev_visualizer", anonymous=True)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.scale = rospy.get_param("~map_scale", 0.25)
        self.update_rate = rospy.get_param("~update_rate", 2.0)
        self.max_vehicle_count = rospy.get_param("~max_vehicle_count", 6)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        self.base_image, self.offset, self.extent = self._generate_base_map()
        self.vehicle_states = {}
        self.vehicle_paths = {}
        self.lock = threading.Lock()

        self.image_pub = rospy.Publisher("/carla_multi_vehicle_planner/bev_image", Image, queue_size=1)
        self.vehicle_markers_pub = rospy.Publisher("/carla_multi_vehicle_planner/vehicle_markers", MarkerArray, queue_size=1)
        self.path_markers_pub = rospy.Publisher("/carla_multi_vehicle_planner/path_markers", MarkerArray, queue_size=1)

        for index in range(1, self.max_vehicle_count + 1):
            role = f"ego_vehicle_{index}"
            rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role)
            rospy.Subscriber(f"/planned_path_{role}", Path, self._path_cb, callback_args=role)

        rospy.Timer(rospy.Duration(1.0 / max(self.update_rate, 0.1)), self._timer_cb)

    def _generate_base_map(self):
        waypoints = self.carla_map.generate_waypoints(2.0)
        if not waypoints:
            raise RuntimeError("Could not generate waypoints for BEV map")

        xs = [wp.transform.location.x for wp in waypoints]
        ys = [wp.transform.location.y for wp in waypoints]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max(1, int((max_x - min_x) / self.scale) + 1)
        height = max(1, int((max_y - min_y) / self.scale) + 1)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        for wp in waypoints:
            px = int((wp.transform.location.x - min_x) / self.scale)
            py = int((wp.transform.location.y - min_y) / self.scale)
            py = height - py - 1  # 상하 반전 교정
            if 0 <= px < width and 0 <= py < height:
                image[py, px] = (60, 60, 60)

        return image, (min_x, min_y), (max_x - min_x, max_y - min_y)

    def _world_to_pixel(self, x, y, offset=None):
        origin_x, origin_y = offset if offset else self.offset
        px = int((x - origin_x) / self.scale)
        py = int((y - origin_y) / self.scale)
        py = self.base_image.shape[0] - py - 1  # 상하 반전
        px = np.clip(px, 0, self.base_image.shape[1] - 1)
        py = np.clip(py, 0, self.base_image.shape[0] - 1)
        return px, py

    def _pixel_to_world(self, px, py):
        x = px * self.scale + self.offset[0]
        y = (self.base_image.shape[0] - py - 1) * self.scale + self.offset[1]
        return x, y

    def _odom_cb(self, msg, role):
        with self.lock:
            self.vehicle_states[role] = msg.pose.pose

    def _path_cb(self, msg, role):
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        with self.lock:
            self.vehicle_paths[role] = points

    def _draw_rect(self, img, cx, cy, w, h, color):
        x1, y1 = int(cx - w // 2), int(cy - h // 2)
        x2, y2 = int(cx + w // 2), int(cy + h // 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)

    def _timer_cb(self, _event):
        with self.lock:
            image = self.base_image.copy()
            markers = MarkerArray()
            path_markers = MarkerArray()
            stamp = rospy.Time.now()

            for index, role in enumerate(sorted(self.vehicle_states.keys())):
                color_bgr = self._role_color(role)
                pose = self.vehicle_states[role]

                # ===== 2D Map 직사각형 표시 =====
                px, py = self._world_to_pixel(pose.position.x, pose.position.y)
                self._draw_rect(image, px, py, VEHICLE_RECT_W, VEHICLE_RECT_H, color_bgr)

                # ===== RViz Cube Marker =====
                marker = Marker()
                marker.header = Header(stamp=stamp, frame_id="map")
                marker.ns = "vehicles"
                marker.id = index
                marker.type = VEHICLE_MARKER_TYPE  # CUBE
                marker.action = Marker.ADD
                marker.pose = pose
                marker.scale.x = VEHICLE_MARKER_SCALE
                marker.scale.y = VEHICLE_MARKER_SCALE/2
                marker.scale.z = VEHICLE_MARKER_SCALE
                marker.color = ColorRGBA(
                    r=color_bgr[2] / 255.0,
                    g=color_bgr[1] / 255.0,
                    b=color_bgr[0] / 255.0,
                    a=0.9,
                )
                markers.markers.append(marker)

                # ===== Path는 기존 그대로 (circle 그대로 둠) =====
                path = self.vehicle_paths.get(role, [])
                if len(path) >= 2:
                    path_marker = Marker()
                    path_marker.header = Header(stamp=stamp, frame_id="map")
                    path_marker.ns = "paths"
                    path_marker.id = index
                    path_marker.type = Marker.LINE_STRIP
                    path_marker.action = Marker.ADD
                    path_marker.scale.x = 0.5  # 원래 path 굵기 그대로 유지
                    path_marker.color = ColorRGBA(
                        r=color_bgr[2] / 255.0,
                        g=color_bgr[1] / 255.0,
                        b=color_bgr[0] / 255.0,
                        a=0.8,
                    )

                    for x, y in path:
                        point = Point(x=x, y=y, z=0.1)
                        path_marker.points.append(point)
                        px2, py2 = self._world_to_pixel(x, y)
                        cv2.circle(image, (px2, py2), 2, color_bgr, -1)  # 원래 path 점 유지

                    path_markers.markers.append(path_marker)

            # ===== publish image =====
            img_msg = Image()
            img_msg.header = Header(stamp=stamp, frame_id="map")
            img_msg.height, img_msg.width = image.shape[:2]
            img_msg.encoding = "bgr8"
            img_msg.step = image.shape[1] * 3
            img_msg.data = image.tobytes()

        self.image_pub.publish(img_msg)
        self.vehicle_markers_pub.publish(markers)
        self.path_markers_pub.publish(path_markers)

    @staticmethod
    def _role_color(role):
        palette = {
            # BGR palette (converted to RGB when publishing markers)
            # 1: green, 2: red, 3: blue, 4: yellow, 5: magenta, 6: cyan
            "ego_vehicle_1": (0, 255, 0),     # green -> BGR(0,255,0)
            "ego_vehicle_2": (0, 0, 255),     # red   -> BGR(0,0,255)
            "ego_vehicle_3": (255, 0, 0),     # blue  -> BGR(255,0,0)
            "ego_vehicle_4": (0, 255, 255),   # yellow-> BGR(0,255,255)
            "ego_vehicle_5": (255, 0, 255),   # magenta->BGR(255,0,255)
            "ego_vehicle_6": (255, 255, 0),   # cyan  -> BGR(255,255,0)
        }
        return palette.get(role, (200, 200, 200))


if __name__ == "__main__":
    try:
        BEVVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
