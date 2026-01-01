#!/usr/bin/env python3
from __future__ import annotations

import select
import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict

import rospy
from capstone_msgs.msg import Uplink

# Sender must match exactly: vehicle_id(int32), voltage(float32)
FMT_UPLINK = "!if"  # 8 bytes
PKT_SIZE = struct.calcsize(FMT_UPLINK)


@dataclass
class LatestPacket:
    vehicle_id: int
    voltage: float
    recv_monotonic: float


class UplinkReceiverNode:
    """Receives binary uplink packets over UDP (vehicle_id, voltage) and publishes /uplink."""

    def __init__(self) -> None:
        rospy.init_node("uplink_receiver", anonymous=True)

        self.udp_ip = str(rospy.get_param("~udp_ip", "0.0.0.0"))
        self.udp_port = int(rospy.get_param("~udp_port", 5560))
        self.topic = str(rospy.get_param("~topic", "/uplink"))
        self.rcvbuf = int(rospy.get_param("~rcvbuf_bytes", 4 * 1024 * 1024))

        # Publishing behavior
        self.publish_hz = float(rospy.get_param("~publish_hz", 30.0))
        self.republish_last = bool(rospy.get_param("~republish_last", False))
        self.max_age_s = float(rospy.get_param("~max_age_s", 0.0))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.setblocking(False)

        self.pub = rospy.Publisher(self.topic, Uplink, queue_size=50)
        self.latest: Dict[int, LatestPacket] = {}

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(
            "[uplink-rx] listen %s:%d (pkt=%dB fmt=%s) -> %s, publish_hz=%.1f republish_last=%s max_age_s=%.3f",
            self.udp_ip,
            self.udp_port,
            PKT_SIZE,
            FMT_UPLINK,
            self.topic,
            self.publish_hz,
            self.republish_last,
            self.max_age_s,
        )

    def _on_shutdown(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def _drain_socket(self) -> None:
        """Read all available packets and keep only the latest per vehicle."""
        while True:
            try:
                data, _addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                return
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "[uplink-rx] recv error: %s", exc)
                return

            if len(data) != PKT_SIZE:
                rospy.logwarn_throttle(
                    2.0, "[uplink-rx] bad packet size=%d expected=%d", len(data), PKT_SIZE
                )
                continue

            try:
                vehicle_id, voltage = struct.unpack(FMT_UPLINK, data)
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "[uplink-rx] unpack error: %s", exc)
                continue

            vid = int(vehicle_id)
            self.latest[vid] = LatestPacket(
                vehicle_id=vid,
                voltage=float(voltage),
                recv_monotonic=time.monotonic(),
            )

    def _publish_latest(self) -> None:
        now_m = time.monotonic()
        for vid, pkt in list(self.latest.items()):
            if self.max_age_s > 0.0 and (now_m - pkt.recv_monotonic) > self.max_age_s:
                continue
            msg = Uplink()
            msg.vehicle_id = pkt.vehicle_id
            msg.voltage = pkt.voltage
            self.pub.publish(msg)
            rospy.logdebug_throttle(
                1.0,
                "[uplink-rx] pub vid=%d V=%.2f age=%.3fs",
                pkt.vehicle_id,
                pkt.voltage,
                now_m - pkt.recv_monotonic,
            )

    def spin(self) -> None:
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            select.select([self.sock], [], [], 0.0)
            self._drain_socket()
            if self.republish_last:
                self._publish_latest()
            else:
                self._publish_latest()
            rate.sleep()


def main() -> None:
    node = UplinkReceiverNode()
    node.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
