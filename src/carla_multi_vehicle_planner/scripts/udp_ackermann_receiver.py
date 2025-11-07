#!/usr/bin/env python3

import math
import socket
import struct
import threading

import rospy
from ackermann_msgs.msg import AckermannDrive

FMT = "!iiI"  # int angle, int speed, uint32 seq


class RCCarUdpReceiver:
    def __init__(self) -> None:
        rospy.init_node("rccar_udp_receiver", anonymous=True)

        self.bind_ip = rospy.get_param("~bind_ip", "0.0.0.0")
        self.bind_port = int(rospy.get_param("~bind_port", 5555))
        self.topic = rospy.get_param("~topic", "/rc/udp_cmd")

        # Decode calibration
        self.angle_scale = float(rospy.get_param("~angle_scale", 500.0))
        self.angle_invert = bool(rospy.get_param("~angle_invert", False))
        # Optional speed scaling if downstream expects different units
        self.speed_scale = float(rospy.get_param("~speed_scale", 1.0))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        except OSError:
            pass
        self.sock.bind((self.bind_ip, self.bind_port))
        self.sock.setblocking(True)

        self.pub = rospy.Publisher(self.topic, AckermannDrive, queue_size=10)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        rospy.on_shutdown(self._shutdown)

        rospy.loginfo(
            "RC UDP receiver bound on %s:%d -> publish %s (angle_scale=%.1f invert=%s speed_scale=%.2f)",
            self.bind_ip,
            self.bind_port,
            self.topic,
            self.angle_scale,
            str(self.angle_invert),
            self.speed_scale,
        )

    def _recv_loop(self) -> None:
        while not rospy.is_shutdown() and not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(64)
                if len(data) < struct.calcsize(FMT):
                    continue
                angle_i, speed_i, seq = struct.unpack(FMT, data[: struct.calcsize(FMT)])

                steer = float(angle_i) / max(1e-6, self.angle_scale)
                if self.angle_invert:
                    steer = -steer
                speed = float(speed_i) * self.speed_scale

                msg = AckermannDrive()
                msg.steering_angle = steer
                msg.speed = speed
                self.pub.publish(msg)

                rospy.loginfo_throttle(
                    0.2,
                    "[RC-UDP-RECV] from %s seq=%d angle_i=%d speed_i=%d -> steer=%.3frad (%.1f°) speed=%.2f",
                    f"{addr[0]}:{addr[1]}",
                    int(seq),
                    int(angle_i),
                    int(speed_i),
                    steer,
                    math.degrees(steer),
                    speed,
                )
            except (OSError, socket.error):
                if not self._stop.is_set():
                    rospy.sleep(0.01)
            except Exception as exc:
                rospy.logwarn_throttle(1.0, f"[RC-UDP-RECV] error: {exc}")

    def _shutdown(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except Exception:
            pass


def main() -> None:
    try:
        RCCarUdpReceiver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


