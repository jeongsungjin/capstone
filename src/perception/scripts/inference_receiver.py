#!/usr/bin/env python3

import json
import math
import select
import socket
from typing import List, Optional, Set, Tuple, Union

import rospy
from capstone_msgs.msg import BEVInfo

from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import MarkerArray, Marker

class InferenceReceiverNode:
    """Receives global_tracks inference over UDP and publishes BEVInfo."""

    def __init__(self) -> None:
        rospy.init_node("inference_receiver", anonymous=True)

        self.udp_ip = str(rospy.get_param("~udp_ip", "0.0.0.0"))
        self.udp_port = int(rospy.get_param("~udp_port", 60050))
        ports_csv = str(rospy.get_param("~udp_ports", "")).strip()
        if ports_csv:
            try:
                self.udp_ports = [int(p.strip()) for p in ports_csv.split(",") if p.strip()]
            except Exception:
                self.udp_ports = [self.udp_port]
        else:
            self.udp_ports = [self.udp_port]

        self.input_yaw_degrees = bool(rospy.get_param("~input_yaw_degrees", True))
        self.max_items = int(rospy.get_param("~max_items", 64))
        allowed_classes_csv = str(rospy.get_param("~allowed_classes", "0")).strip()
        if allowed_classes_csv:
            try:
                self.allowed_classes: Optional[Set[int]] = {
                    int(c.strip()) for c in allowed_classes_csv.split(",") if c.strip()
                }
            except Exception:
                self.allowed_classes = None
                rospy.logwarn("InferenceReceiver: failed to parse allowed_classes; disabling class filter")
        else:
            self.allowed_classes = None
        self.topic = str(rospy.get_param("~topic", "/bev_info_raw"))
        self.frame_id = str(rospy.get_param("~frame_id", "map"))

        self.socks: List[socket.socket] = []
        for port in self.udp_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except Exception:
                pass
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            sock.bind((self.udp_ip, int(port)))
            sock.setblocking(False)
            self.socks.append(sock)

        self.obstacle_pub = rospy.Publisher("/obstacles", PoseArray, queue_size=1)
        self.marker_pub = rospy.Publisher("/obstacle_markers", MarkerArray, queue_size=1, latch=True)

        self.obstacle_height = 2.0

        self.pub = rospy.Publisher(self.topic, BEVInfo, queue_size=1)
        rospy.loginfo(
            "[Inference UDP] listening on %s ports %s -> %s (allowed_classes=%s)",
            self.udp_ip,
            ",".join(str(p) for p in self.udp_ports),
            self.topic,
            ",".join(str(c) for c in sorted(self.allowed_classes)) if self.allowed_classes else "ANY",
        )
        rospy.on_shutdown(self._on_shutdown)

        self.frame_seq: int = 0

    def _on_shutdown(self) -> None:
        for s in self.socks:
            try:
                s.close()
            except Exception:
                pass

    def _extract_stamp(self, payload: dict) -> rospy.Time:
        ts_fields = ["timestamp", "stamp", "ts", "time"]
        for key in ts_fields:
            if key not in payload:
                continue
            raw = payload.get(key)
            if raw is None:
                continue
            try:
                return rospy.Time.from_sec(float(raw))
            except (TypeError, ValueError):
                continue
        return rospy.Time.now()

    def _safe_float(self, v, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _extract_speed_mps(self, it: dict) -> float:
        """
        Extract speed (m/s) from one item.
        Priority:
          1) scalar keys: speed_mps, speed, speed_ms, v_mps
          2) vector keys: velocity, vel, v  -> if list/tuple len>=2 => sqrt(vx^2+vy^2)
          3) fallback 0.0
        """
        # 1) scalar
        for k in ("speed_mps", "speed", "speed_ms", "v_mps"):
            if k in it and it.get(k) is not None:
                return self._safe_float(it.get(k), 0.0)

        # 2) vector
        for k in ("velocity", "vel", "v"):
            vv = it.get(k)
            if isinstance(vv, (list, tuple)) and len(vv) >= 2:
                vx = self._safe_float(vv[0], 0.0)
                vy = self._safe_float(vv[1], 0.0)
                return float(math.sqrt(vx * vx + vy * vy))

        return 0.0

    def _extract_class_id(self, it: dict) -> Optional[int]:
        """Read class id from common keys."""
        for k in ("class_id", "class", "cls", "label", "label_id"):
            if k not in it:
                continue
            try:
                return int(it.get(k))
            except Exception:
                continue
        return None

    def _to_bevinfo(self, items: List[dict], stamp: rospy.Time, frame_seq: int) -> BEVInfo:
        ids: List[int] = []
        xs: List[float] = []
        ys: List[float] = []
        yaws: List[float] = []
        colors: List[str] = []
        speeds_mps: List[float] = []

        for it in items[: max(0, self.max_items)]:
            # Optional class filter to drop non-vehicle detections (e.g., cones)
            cls_id = self._extract_class_id(it)
            if self.allowed_classes is not None and cls_id is not None:
                if cls_id not in self.allowed_classes:
                    continue

            try:
                vid = int(it.get("id"))
            except Exception:
                continue

            center = it.get("center", [0.0, 0.0, 0.0])
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue

            x = float(center[0])
            y = float(center[1])

            yaw_val = self._safe_float(it.get("yaw", 0.0), 0.0)
            yaw_rad = math.radians(yaw_val) if self.input_yaw_degrees else yaw_val

            # speed (m/s) - NEW
            spd = self._extract_speed_mps(it)

            color_val = it.get("color")
            if color_val is None:
                color_val = it.get("color_name")
            if color_val is None:
                color_val = it.get("color_id")
            if color_val is None:
                color_val = ""
            try:
                color_str = str(color_val)
            except Exception:
                color_str = ""

            ids.append(vid)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw_rad)
            colors.append(color_str)
            speeds_mps.append(float(spd))

        msg = BEVInfo()
        msg.header.stamp = stamp
        msg.header.seq = frame_seq
        msg.header.frame_id = self.frame_id

        msg.frame_seq = frame_seq
        msg.detCounts = len(ids)
        msg.ids = ids
        msg.center_xs = xs
        msg.center_ys = ys
        msg.yaws = yaws
        msg.colors = colors

        # NEW FIELD (requires BEVInfo.msg update)
        msg.speeds_mps = speeds_mps

        return msg

    def _to_obstacle_info(self, items, stamp, frame_seq):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.seq = frame_seq
        msg.header.frame_id = self.frame_id
        widths = []

        for it in items:
            cls_id = self._extract_class_id(it)
            if cls_id == 0:
                continue
            
            center_pt = it.get("center")
            widths.append(it.get("width"))

            pose = Pose()
            pose.position.x = center_pt[0]
            pose.position.y = center_pt[1]
            pose.position.z = center_pt[2]
            pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.obstacle_pub.publish(msg)

        markers = MarkerArray()
        
        # 기존 마커 삭제
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)
        
        # 새 마커 추가
        for i, pose in enumerate(msg.poses):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = stamp
            marker.header.seq = frame_seq
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = pose.position.x
            marker.pose.position.y = pose.position.y
            marker.pose.position.z = pose.position.z + self.obstacle_height / 2
            marker.pose.orientation.w = 1.0
            marker.scale.x = widths[i] * 2
            marker.scale.y = widths[i] * 2
            marker.scale.z = self.obstacle_height
            marker.color.r = 1.0
            marker.color.g = 0.3
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker.lifetime = rospy.Duration(0)
            markers.markers.append(marker)

        self.marker_pub.publish(markers)

        return len(msg.poses)

    def spin(self) -> None:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if not self.socks:
                rospy.sleep(0.1)
                continue

            readable, _, _ = select.select(self.socks, [], [], 0.5)
            if not readable:
                continue

            for sock in readable:
                try:
                    data, _addr = sock.recvfrom(65507)
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "UDP recv error: %s", exc)
                    continue

                try:
                    payload = json.loads(data.decode("utf-8"))
                except Exception:
                    rospy.logwarn_throttle(2.0, "Non-JSON or invalid payload; ignored")
                    continue

                if payload.get("type") != "global_tracks":
                    continue

                items = payload.get("items", [])
                stamp = self._extract_stamp(payload)

                frame_seq = self.frame_seq
                bev_msg = self._to_bevinfo(items, stamp, frame_seq)
                obstacle_count = self._to_obstacle_info(items, stamp, frame_seq)
                self.pub.publish(bev_msg)

                now = rospy.Time.now()
                latency = (now - stamp).to_sec()
                rospy.loginfo_throttle(
                    1.0,
                    "[Frame %d] published detCounts=%d latency=%.3fs",
                    frame_seq,
                    bev_msg.detCounts + obstacle_count,
                    latency,
                )

                self.frame_seq += 1

            rate.sleep()


if __name__ == "__main__":
    try:
        node = InferenceReceiverNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
