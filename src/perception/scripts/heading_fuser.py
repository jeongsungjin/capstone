#!/usr/bin/env python3

import math
import threading
from typing import Dict

import rospy
from capstone_msgs.msg import BEVInfo, Uplink


def wrap_angle_rad(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class HeadingState:
    __slots__ = ("psi", "last_imu_stamp", "last_imu_seq")

    def __init__(self, psi: float = 0.0, stamp: rospy.Time = rospy.Time(0), seq: int = 0) -> None:
        self.psi = psi
        self.last_imu_stamp = stamp
        self.last_imu_seq = seq


class HeadingFuserNode:
    """Fuses IMU heading increments with vision heading anchors."""

    def __init__(self) -> None:
        rospy.init_node("heading_fuser", anonymous=True)

        self.imu_topic = rospy.get_param("~imu_topic", "/imu_uplink")
        self.bev_topic = rospy.get_param("~bev_topic", "/bev_info_raw")
        self.out_topic = rospy.get_param("~out_topic", "/bev_info_fused")

        self.k_gain = float(rospy.get_param("~k_gain", 0.1))  # correction gain
        self.gate_deg = float(rospy.get_param("~gate_deg", 20.0))
        self.max_skew = float(rospy.get_param("~max_skew", 0.0))  # seconds; 0 or negative disables
        self.default_gate_rad = math.radians(self.gate_deg)

        self.states: Dict[int, HeadingState] = {}
        self.lock = threading.Lock()

        self.pub = rospy.Publisher(self.out_topic, BEVInfo, queue_size=1)
        self.sub_imu = rospy.Subscriber(self.imu_topic, Uplink, self._imu_cb, queue_size=30)
        self.sub_bev = rospy.Subscriber(self.bev_topic, BEVInfo, self._bev_cb, queue_size=5)

        rospy.loginfo("[HeadingFuser] imu=%s bev=%s -> out=%s, K=%.3f gate=%.1fdeg max_skew=%.3fs",
                      self.imu_topic, self.bev_topic, self.out_topic, self.k_gain, self.gate_deg, self.max_skew)

    # IMU increment callback
    def _imu_cb(self, msg: Uplink) -> None:
        vehicle_id = int(msg.vehicle_id)
        dpsi = float(msg.heading_diff)
        stamp = rospy.Time.now()  # msg has no header; use receive time
        seq = int(msg.heading_seq)

        with self.lock:
            st = self.states.get(vehicle_id)
            if st is None:
                st = HeadingState(psi=wrap_angle_rad(dpsi), stamp=stamp, seq=seq)
                self.states[vehicle_id] = st
            else:
                st.psi = wrap_angle_rad(st.psi + dpsi)
                st.last_imu_stamp = stamp
                st.last_imu_seq = seq

    # Vision heading callback
    def _bev_cb(self, msg: BEVInfo) -> None:
        if msg.detCounts <= 0:
            return

        out = BEVInfo()
        out.header = msg.header
        out.frame_seq = msg.frame_seq
        out.detCounts = msg.detCounts
        out.ids = list(msg.ids)
        out.center_xs = list(msg.center_xs)
        out.center_ys = list(msg.center_ys)
        out.colors = list(msg.colors)

        fused_yaws = []
        gate_rad = self.default_gate_rad
        now = rospy.Time.now()

        with self.lock:
            for idx, vid_raw in enumerate(msg.ids):
                try:
                    vid = int(vid_raw)
                except Exception:
                    vid = 0

                yaw_obs = float(msg.yaws[idx]) if idx < len(msg.yaws) else 0.0
                st = self.states.get(vid)

                if st is None:
                    # Initialize state with observation
                    st = HeadingState(psi=wrap_angle_rad(yaw_obs), stamp=now, seq=0)
                    self.states[vid] = st
                    fused_yaws.append(st.psi)
                    continue

                # Optional skew guard (disabled if max_skew<=0)
                if self.max_skew > 0.0 and st.last_imu_stamp != rospy.Time(0):
                    skew = abs((msg.header.stamp - st.last_imu_stamp).to_sec())
                    if skew > self.max_skew:
                        fused_yaws.append(st.psi)
                        rospy.logwarn_throttle(
                            1.0,
                            "[HeadingFuser] skip correction vid=%d skew=%.3fs>%.3fs",
                            vid,
                            skew,
                            self.max_skew,
                        )
                        continue

                # Correction
                e = wrap_angle_rad(yaw_obs - st.psi)
                if abs(e) <= gate_rad:
                    st.psi = wrap_angle_rad(st.psi + self.k_gain * e)
                fused_yaws.append(st.psi)

        out.yaws = fused_yaws
        self.pub.publish(out)

        rospy.logdebug_throttle(
            1.0,
            "[HeadingFuser] frame=%d det=%d",
            out.frame_seq,
            out.detCounts,
        )


if __name__ == "__main__":
    try:
        node = HeadingFuserNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

