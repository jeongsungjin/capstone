#!/usr/bin/env python3

import json
import select
import socket
from typing import List, Sequence

import rospy
from sensor_msgs.msg import Imu
from capstone_msgs.msg import Uplink


class ImuUplinkReceiverNode:
    """Receives IMU uplink packets over UDP and republishes as Uplink msg."""

    def __init__(self) -> None:
        rospy.init_node("imu_uplink_receiver", anonymous=True)

        self.udp_ip = str(rospy.get_param("~udp_ip", "0.0.0.0"))
        self.udp_port = int(rospy.get_param("~udp_port", 60100))
        self.topic = str(rospy.get_param("~topic", "/imu_uplink"))
        self.frame_id = str(rospy.get_param("~frame_id", "base_link"))
        self.rcvbuf = int(rospy.get_param("~rcvbuf_bytes", 4 * 1024 * 1024))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.setblocking(False)

        self.pub = rospy.Publisher(self.topic, Uplink, queue_size=5)
        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("[IMU UDP] listening on %s:%d -> %s", self.udp_ip, self.udp_port, self.topic)

    def _on_shutdown(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def _as_float_seq(self, value, length: int) -> List[float]:
        if isinstance(value, Sequence):
            try:
                vals = [float(v) for v in value]
                if len(vals) >= length:
                    return vals[:length]
            except Exception:
                pass
        return [0.0] * length

    def _build_imu(self, payload: dict) -> Imu:
        imu = Imu()
        imu.header.stamp = rospy.Time.now()
        imu.header.frame_id = self.frame_id

        orientation = self._as_float_seq(payload.get("orientation"), 4)
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = orientation

        ang_vel = self._as_float_seq(payload.get("angular_velocity"), 3)
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = ang_vel

        lin_acc = self._as_float_seq(payload.get("linear_acceleration"), 3)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = lin_acc

        return imu

    def spin(self) -> None:
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            readable, _, _ = select.select([self.sock], [], [], 0.2)
            if not readable:
                rate.sleep()
                continue

            for _sock in readable:
                try:
                    data, _addr = _sock.recvfrom(65507)
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "[IMU UDP] recv error: %s", exc)
                    continue

                try:
                    payload = json.loads(data.decode("utf-8"))
                except Exception:
                    rospy.logwarn_throttle(2.0, "[IMU UDP] invalid/non-JSON payload; ignored")
                    continue

                msg = Uplink()
                try:
                    msg.vehicle_id = int(payload.get("vehicle_id", 0))
                except Exception:
                    msg.vehicle_id = 0

                try:
                    msg.voltage = float(payload.get("voltage", 0.0))
                except Exception:
                    msg.voltage = 0.0

                try:
                    msg.heading_diff = float(payload.get("heading_diff", 0.0))
                except Exception:
                    msg.heading_diff = 0.0

                try:
                    msg.heading_dt = float(payload.get("heading_dt", 0.0))
                except Exception:
                    msg.heading_dt = 0.0

                try:
                    msg.heading_seq = int(payload.get("heading_seq", 0))
                except Exception:
                    msg.heading_seq = 0

                msg.imu = self._build_imu(payload.get("imu", payload))

                self.pub.publish(msg)
                rospy.logdebug_throttle(1.0, "[IMU UDP] seq=%d dpsi=%.4f dt=%.3f vid=%d",
                                        msg.heading_seq, msg.heading_diff, msg.heading_dt, msg.vehicle_id)

            rate.sleep()


if __name__ == "__main__":
    try:
        node = ImuUplinkReceiverNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass

