#!/usr/bin/env python3
import threading
from typing import Dict, List, Tuple

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class ScheduleVisualizer:
    def __init__(self):
        rospy.init_node("schedule_visualizer", anonymous=True)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.text_scale = float(rospy.get_param("~text_scale", 5.2))
        self.line_width = float(rospy.get_param("~line_width", 0.35))
        self.time_window = float(rospy.get_param("~time_window", 20.0))  # seconds of future to show; <=0 to disable
        self.hide_past = bool(rospy.get_param("~hide_past", True))
        self.clear_each_tick = bool(rospy.get_param("~clear_each_tick", True))
        self.label_count = int(rospy.get_param("~label_count", 12))  # approx labels per vehicle

        self._lock = threading.RLock()
        # role -> (samples, base_stamp_sec)
        self._schedules: Dict[str, Tuple[List[Tuple[float, float, float]], float]] = {}

        self._pub = rospy.Publisher("/carla_multi_vehicle_planner/schedule_markers", MarkerArray, queue_size=1)

        for index in range(self.num_vehicles):
            role = f"ego_vehicle_{index + 1}"
            topic = f"/global_path_schedule_{role}"
            rospy.Subscriber(topic, Path, self._schedule_cb, callback_args=role, queue_size=1)

        rospy.Timer(rospy.Duration(0.5), self._timer_cb)

    def _schedule_cb(self, msg: Path, role: str) -> None:
        with self._lock:
            samples: List[Tuple[float, float, float]] = []
            for pose in msg.poses:
                x = pose.pose.position.x
                y = pose.pose.position.y
                t = pose.pose.position.z  # arrival time encoded by planner
                samples.append((x, y, t))
            base_stamp_sec = float(msg.header.stamp.to_sec()) if msg.header and msg.header.stamp else float(rospy.Time.now().to_sec())
            self._schedules[role] = (samples, base_stamp_sec)

    def _timer_cb(self, _event):
        with self._lock:
            markers = MarkerArray()
            now = rospy.Time.now()
            now_sec = float(now.to_sec())

            # Optionally clear previously published markers on our topic
            if self.clear_each_tick:
                clear = Marker()
                clear.header = Header(stamp=now, frame_id=self.frame_id)
                clear.action = Marker.DELETEALL
                markers.markers.append(clear)

            for idx, role in enumerate(sorted(self._schedules.keys())):
                color = self._role_color_rgba(role, alpha=0.95)
                entry = self._schedules.get(role)
                if not entry:
                    continue
                samples, base_stamp_sec = entry
                if len(samples) == 0:
                    continue

                # Filter to show only future (relative to the message base stamp)
                rel_now = now_sec - base_stamp_sec
                filtered: List[Tuple[float, float, float]] = []
                for x, y, t in samples:
                    if self.hide_past and t < rel_now - 1e-6:
                        continue
                    if self.time_window > 0.0 and t > rel_now + self.time_window:
                        continue
                    filtered.append((x, y, t))

                if len(filtered) < 2:
                    # Show at least labels if one point remains
                    pass
                else:
                    # Polyline of scheduled geometry (future window only)
                    line = Marker()
                    line.header = Header(stamp=now, frame_id=self.frame_id)
                    line.ns = f"schedule_poly_{role}"
                    line.id = idx * 1000
                    line.type = Marker.LINE_STRIP
                    line.action = Marker.ADD
                    line.scale.x = self.line_width
                    line.color = color
                    line.lifetime = rospy.Duration(0.6)
                    for x, y, _t in filtered:
                        line.points.append(Point(x=x, y=y, z=0.2))
                    markers.markers.append(line)

                # Text labels: evenly sampled across filtered points
                if len(filtered) >= 1:
                    step = max(1, len(filtered) // max(1, self.label_count))
                    label_id = idx * 1000 + 1
                    for i in range(0, len(filtered), step):
                        x, y, t = filtered[i]
                        text = Marker()
                        text.header = Header(stamp=now, frame_id=self.frame_id)
                        text.ns = f"schedule_time_{role}"
                        text.id = label_id
                        label_id += 1
                        text.type = Marker.TEXT_VIEW_FACING
                        text.action = Marker.ADD
                        text.scale.z = self.text_scale
                        text.color = color
                        text.color.a = 1.0
                        text.pose.position.x = x
                        text.pose.position.y = y
                        text.pose.position.z = 1.5
                        text.text = f"t={t:.1f}s"
                        text.lifetime = rospy.Duration(0.6)
                        markers.markers.append(text)

        self._pub.publish(markers)

    @staticmethod
    def _role_color_rgba(role: str, alpha: float = 1.0) -> ColorRGBA:
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


if __name__ == "__main__":
    try:
        ScheduleVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


