#!/usr/bin/env python3

import json
import math
import select
import socket
from typing import List

import rospy
from capstone_msgs.msg import BEVInfo

# capstone_msgs BEVInfo message에 차량 별 속도 필드 추가
# inference_receiver.py 에 속도 필드 추가 반영
# /bev_info_raw 토픽에 속도 정보 포함하여 발행
# control 패키지에서 BEVInfo 메시지 구독하여 pid 속도 제어기 구현
# simple_bev_teleporter에서 불연속 적인 투영에서도 pid제어기가 잘 동작할지?? (확인 필요)


class InferenceReceiverNode:
    """Receives global_tracks inference over UDP and publishes BEVInfo."""

    def __init__(self) -> None:
        rospy.init_node("inference_receiver", anonymous=True)

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

        self.input_yaw_degrees = bool(rospy.get_param("~input_yaw_degrees", True))
        self.max_items = int(rospy.get_param("~max_items", 64))
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

        self.pub = rospy.Publisher(self.topic, BEVInfo, queue_size=1)
        rospy.loginfo("[Inference UDP] listening on %s ports %s -> %s", self.udp_ip,
                      ",".join(str(p) for p in self.udp_ports), self.topic)
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

    def _to_bevinfo(self, items: List[dict], stamp: rospy.Time, frame_seq: int) -> BEVInfo:
        ids: List[int] = []
        xs: List[float] = []
        ys: List[float] = []
        yaws: List[float] = []
        colors: List[str] = []

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
        return msg

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
                self.pub.publish(bev_msg)

                now = rospy.Time.now()
                latency = (now - stamp).to_sec()
                rospy.loginfo_throttle(
                    1.0,
                    "[Frame %d] published detCounts=%d latency=%.3fs",
                    frame_seq,
                    bev_msg.detCounts,
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

