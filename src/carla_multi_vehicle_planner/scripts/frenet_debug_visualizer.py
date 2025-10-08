#!/usr/bin/env python3

import math
import threading
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def color_rgba(r, g, b, a):
    return ColorRGBA(r=r, g=g, b=b, a=a)


class FrenetDebugVisualizer:
    def __init__(self) -> None:
        rospy.init_node("frenet_debug_visualizer", anonymous=True)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.pred_horizon_s = float(rospy.get_param("~pred_horizon_s", 4.0))
        self.pred_dt = float(rospy.get_param("~pred_dt", 0.2))
        self.assumed_speed_mps = float(rospy.get_param("~assumed_speed_mps", 6.0))
        self.r_safe_R0 = float(rospy.get_param("~R0", 0.5))
        self.r_safe_tau = float(rospy.get_param("~tau", 1.0))

        self._lock = threading.RLock()
        self._globals: Dict[str, List[Tuple[float, float]]] = {}
        self._locals: Dict[str, List[Tuple[float, float]]] = {}
        self._frenet: Dict[str, Tuple[Tuple[float, float], float]] = {}
        self._speed: Dict[str, float] = {}

        self.pub = rospy.Publisher("/carla_multi_vehicle_planner/frenet_debug", MarkerArray, queue_size=1)

        for i in range(self.num_vehicles):
            role = self._role(i)
            rospy.Subscriber(f"/global_path_{role}", Path, self._cb_global, callback_args=role)
            rospy.Subscriber(f"/local_path_{role}", Path, self._cb_local, callback_args=role)
            rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._cb_odom, callback_args=role)

        rospy.Timer(rospy.Duration(0.2), self._timer)

    def _role(self, i: int) -> str:
        return f"ego_vehicle_{i + 1}"

    def _cb_global(self, msg: Path, role: str) -> None:
        with self._lock:
            pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            self._globals[role] = pts
            self._update_frenet(role, pts)

    def _cb_local(self, msg: Path, role: str) -> None:
        with self._lock:
            pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            self._locals[role] = pts

    def _cb_odom(self, msg: Odometry, role: str) -> None:
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.hypot(vx, vy)
        with self._lock:
            self._speed[role] = speed

    def _update_frenet(self, role: str, pts: List[Tuple[float, float]]) -> None:
        if len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            heading = math.atan2(y1 - y0, x1 - x0)
            self._frenet[role] = ((x0, y0), heading)

    def _timer(self, _event) -> None:
        markers = MarkerArray()
        stamp = rospy.Time.now()
        with self._lock:
            roles = sorted(set(list(self._globals.keys()) + list(self._locals.keys())))
            for idx, role in enumerate(roles):
                base_color = self._role_color(idx)
                frame_id = "map"
                header = Header(stamp=stamp, frame_id=frame_id)

                # Frenet origin + heading arrow
                origin_heading = self._frenet.get(role)
                if origin_heading is not None:
                    (ox, oy), heading = origin_heading
                    m = Marker()
                    m.header = header
                    m.ns = f"{role}/frenet"
                    m.id = idx * 1000 + 1
                    m.type = Marker.ARROW
                    m.action = Marker.ADD
                    m.scale.x = 2.0
                    m.scale.y = 0.4
                    m.scale.z = 0.4
                    m.color = color_rgba(1.0, 1.0, 1.0, 0.8)
                    m.points = [
                        Point(x=ox, y=oy, z=0.2),
                        Point(x=ox + math.cos(heading) * 3.0, y=oy + math.sin(heading) * 3.0, z=0.2),
                    ]
                    markers.markers.append(m)

                # Global path line
                gpts = self._globals.get(role, [])
                if len(gpts) >= 2:
                    m = Marker()
                    m.header = header
                    m.ns = f"{role}/global"
                    m.id = idx * 1000 + 2
                    m.type = Marker.LINE_STRIP
                    m.action = Marker.ADD
                    m.scale.x = 0.4
                    m.color = color_rgba(base_color[0], base_color[1], base_color[2], 0.5)
                    for x, y in gpts:
                        m.points.append(Point(x=x, y=y, z=0.05))
                    markers.markers.append(m)

                # Local path line
                lpts = self._locals.get(role, [])
                if len(lpts) >= 2:
                    m = Marker()
                    m.header = header
                    m.ns = f"{role}/local"
                    m.id = idx * 1000 + 3
                    m.type = Marker.LINE_STRIP
                    m.action = Marker.ADD
                    m.scale.x = 0.6
                    m.color = color_rgba(base_color[0], base_color[1], base_color[2], 0.9)
                    for x, y in lpts:
                        m.points.append(Point(x=x, y=y, z=0.07))
                    markers.markers.append(m)

                # Near-term predictions with safety radius
                if len(gpts) >= 2 and origin_heading is not None:
                    speed = max(self.assumed_speed_mps, self._speed.get(role, 0.0))
                    (ox, oy), heading = origin_heading
                    # Build a simple s-profile along global for drawing samples
                    s_profile = [0.0]
                    for i in range(1, len(gpts)):
                        s_profile.append(s_profile[-1] + math.hypot(gpts[i][0] - gpts[i - 1][0], gpts[i][1] - gpts[i - 1][1]))
                    total_s = s_profile[-1]
                    t = 0.0
                    pred_id_base = idx * 1000 + 10
                    while t <= self.pred_horizon_s:
                        target_s = min(total_s, speed * t)
                        # find position along polyline at target_s
                        px, py = gpts[0]
                        for i in range(len(s_profile) - 1):
                            if target_s <= s_profile[i + 1]:
                                ratio = 0.0
                                seg = s_profile[i + 1] - s_profile[i]
                                if seg > 1e-6:
                                    ratio = (target_s - s_profile[i]) / seg
                                px = gpts[i][0] + (gpts[i + 1][0] - gpts[i][0]) * ratio
                                py = gpts[i][1] + (gpts[i + 1][1] - gpts[i][1]) * ratio
                                break

                        rsafe = self.r_safe_R0 + self.r_safe_tau * speed
                        c = Marker()
                        c.header = header
                        c.ns = f"{role}/pred"
                        c.id = pred_id_base + int(t * 10)
                        c.type = Marker.CYLINDER
                        c.action = Marker.ADD
                        c.pose.position.x = px
                        c.pose.position.y = py
                        c.pose.position.z = 0.1
                        c.scale.x = rsafe * 2.0
                        c.scale.y = rsafe * 2.0
                        c.scale.z = 0.05
                        c.color = color_rgba(base_color[0], base_color[1], base_color[2], 0.25)
                        markers.markers.append(c)

                        t += self.pred_dt

        self.pub.publish(markers)

    def _role_color(self, i: int) -> Tuple[float, float, float]:
        palette = [
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
        return palette[i % len(palette)]


if __name__ == "__main__":
    try:
        FrenetDebugVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


