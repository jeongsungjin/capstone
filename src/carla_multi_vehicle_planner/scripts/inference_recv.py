#!/usr/bin/env python3

import socket
import select
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
        ports_csv = str(rospy.get_param("~udp_ports", "")).strip()
        if ports_csv:
            try:
                self.udp_ports = [int(p.strip()) for p in ports_csv.split(",") if p.strip()]
            except Exception:
                self.udp_ports = [self.udp_port]
        else:
            self.udp_ports = [self.udp_port]
        # If incoming yaw unit is degrees, convert to radians for BEVInfo to match teleporter default
        self.input_yaw_degrees = bool(rospy.get_param("~input_yaw_degrees", True))
        # Pose adjustment before publishing
        self.x_offset_m = float(rospy.get_param("~x_offset_m", 0.0))
        self.y_offset_m = float(rospy.get_param("~y_offset_m", 0.0))
        self.yaw_add_deg = float(rospy.get_param("~yaw_add_deg", 0.0))
        self.max_items = int(rospy.get_param("~max_items", 64))
        self.topic = str(rospy.get_param("~topic", "/bev_info"))

        # UDP sockets (multi-port)
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

        self.pub = rospy.Publisher(self.topic, BEVInfo, queue_size=1)
        rospy.loginfo("[UDP] Listening on %s ports %s -> publishing %s", self.udp_ip, ",".join(str(p) for p in self.udp_ports), self.topic)
        rospy.on_shutdown(self._on_shutdown)

    def _on_shutdown(self) -> None:
        for s in self.socks:
            try:
                s.close()
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
            x = float(center[0]) + self.x_offset_m
            y = float(center[1]) + self.y_offset_m
            yaw_val = float(it.get("yaw", 0.0))
            yaw_rad = math.radians(yaw_val) if self.input_yaw_degrees else yaw_val
            # Add yaw offset (default +180 deg) and normalize
            # yaw_rad += math.radians(self.yaw_add_deg)
            # while yaw_rad > math.pi:
            #     yaw_rad -= 2.0 * math.pi
            # while yaw_rad < -math.pi:
            #     yaw_rad += 2.0 * math.pi

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
        rate = rospy.Rate(30)  # event-driven; still cap logging rate
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