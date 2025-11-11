#!/usr/bin/env python3

import socket
import json
import math
from typing import List, Tuple

import rospy
from capstone_msgs.msg import BEVInfo


class InferenceReceiverNode:
    def __init__(self) -> None:
        rospy.init_node("inference_receiver", anonymous=True)

        # Params
        self.udp_ip = str(rospy.get_param("~udp_ip", "0.0.0.0"))
        self.udp_port = int(rospy.get_param("~udp_port", 60200))
        # If incoming yaw unit is degrees, convert to radians for BEVInfo to match teleporter default
        self.input_yaw_degrees = bool(rospy.get_param("~input_yaw_degrees", True))
        self.max_items = int(rospy.get_param("~max_items", 64))
        self.topic = str(rospy.get_param("~topic", "/bev_info"))

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.settimeout(0.5)

        self.pub = rospy.Publisher(self.topic, BEVInfo, queue_size=1)
        rospy.loginfo("[UDP] Listening on %s:%d -> publishing %s", self.udp_ip, self.udp_port, self.topic)
        rospy.on_shutdown(self._on_shutdown)

    def _on_shutdown(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def _to_bevinfo(self, items: List[dict]) -> BEVInfo:
        ids: List[int] = []
        xs: List[float] = []
        ys: List[float] = []
        yaws: List[float] = []

        for it in items[: max(0, self.max_items)]:
            try:
                vid = int(it.get("id"))
            except Exception:
                continue
            center = it.get("center", [0.0, 0.0, 0.0])
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            x = float(center[0])
            y = float(center[1])
            yaw_val = float(it.get("yaw", 0.0))
            yaw_rad = math.radians(yaw_val) if self.input_yaw_degrees else yaw_val

            ids.append(vid)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw_rad)

        msg = BEVInfo()
        msg.detCounts = len(ids)
        msg.ids = ids
        msg.center_xs = xs
        msg.center_ys = ys
        msg.yaws = yaws
        return msg

    def spin(self) -> None:
        frame_idx = 0
        rate = rospy.Rate(30)  # event-driven; no sleep if messages arrive
        while not rospy.is_shutdown():
            try:
                data, _addr = self.sock.recvfrom(65507)
            except socket.timeout:
                continue
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
            bev_msg = self._to_bevinfo(items)
            self.pub.publish(bev_msg)

            rospy.loginfo_throttle(1.0, "[Frame %d] published detCounts=%d", frame_idx, bev_msg.detCounts)
            frame_idx += 1
            rate.sleep()


if __name__ == "__main__":
    try:
        node = InferenceReceiverNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass