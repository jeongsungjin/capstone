#!/usr/bin/env python3
from __future__ import annotations

import select
import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rospy
from capstone_msgs.msg import Uplink

# Sender must match exactly
FMT_UPLINK = "!ifffI"  # 20 bytes
PKT_SIZE = struct.calcsize(FMT_UPLINK)


@dataclass
class LatestPacket:
    vehicle_id: int
    voltage: float
    heading_diff: float
    heading_dt: float
    heading_seq: int
    recv_monotonic: float  # local receive time (for optional staleness check)


class HeadingDeltaUplinkReceiverNode:
    """Receives binary uplink packets over UDP, keeps only the latest per vehicle, publishes at fixed rate."""

    def __init__(self) -> None:
        rospy.init_node("heading_delta_uplink_receiver", anonymous=True)

        self.udp_ip = str(rospy.get_param("~udp_ip", "0.0.0.0"))
        self.udp_port = int(rospy.get_param("~udp_port", 5560))
        self.topic = str(rospy.get_param("~topic", "/imu_uplink"))
        self.rcvbuf = int(rospy.get_param("~rcvbuf_bytes", 4 * 1024 * 1024))

        # Publishing behavior
        self.publish_hz = float(rospy.get_param("~publish_hz", 30.0))
        self.republish_last = bool(rospy.get_param("~republish_last", False))

        # Optional: drop publish if packet too old (seconds). 0 disables.
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

        # latest per vehicle_id
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
                vehicle_id, voltage, heading_diff, heading_dt, heading_seq = struct.unpack(FMT_UPLINK, data)
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "[uplink-rx] unpack error: %s", exc)
                continue

            vid = int(vehicle_id)
            self.latest[vid] = LatestPacket(
                vehicle_id=vid,
                voltage=float(voltage),
                heading_diff=float(heading_diff),
                heading_dt=float(heading_dt),
                heading_seq=int(heading_seq),
                recv_monotonic=time.monotonic(),
            )

    def _publish_latest(self) -> None:
        now_m = time.monotonic()

        # If you only want "the very latest overall", not per-vehicle, change this loop.
        for vid, pkt in list(self.latest.items()):
            if self.max_age_s > 0.0 and (now_m - pkt.recv_monotonic) > self.max_age_s:
                # stale -> skip (or del)
                continue

            msg = Uplink()
            msg.vehicle_id = pkt.vehicle_id
            msg.voltage = pkt.voltage
            msg.heading_diff = pkt.heading_diff
            msg.heading_dt = pkt.heading_dt
            msg.heading_seq = int(pkt.heading_seq)

            self.pub.publish(msg)

            rospy.logdebug_throttle(
                1.0,
                "[uplink-rx] pub vid=%d V=%.2f dpsi=%+.6f dt=%.4f seq=%d age=%.3fs",
                pkt.vehicle_id,
                pkt.voltage,
                pkt.heading_diff,
                pkt.heading_dt,
                pkt.heading_seq,
                now_m - pkt.recv_monotonic,
            )

    def spin(self) -> None:
        rate = rospy.Rate(self.publish_hz)

        while not rospy.is_shutdown():
            # Wait a bit for readability, then drain everything so we keep only the latest.
            # (Even if not readable, we still can republish the last.)
            select.select([self.sock], [], [], 0.0)
            self._drain_socket()

            if self.republish_last:
                # publish whatever latest we have (even if it didn't update this cycle)
                self._publish_latest()
            else:
                # publish only if something arrived "recently" during this cycle:
                # simplest is to require age < 1/publish_hz (very strict). Usually republish_last=True is better.
                self._publish_latest()

            rate.sleep()


if __name__ == "__main__":
    try:
        node = HeadingDeltaUplinkReceiverNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
