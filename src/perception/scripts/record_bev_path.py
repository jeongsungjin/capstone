#!/usr/bin/env python3

import json
import math
import os
from typing import List

import rospy
from capstone_msgs.msg import BEVInfo  # type: ignore


class BEVPathRecorder:
    def __init__(self) -> None:
        rospy.init_node("record_bev_path", anonymous=True)

        self.topic = str(rospy.get_param("~topic", "/bev_info_raw"))
        self.target_id = int(rospy.get_param("~target_id", 1))
        self.output_path = str(
            rospy.get_param("~output", "/home/jamie/capstone/src/perception/scripts/data/recorded_path_id1_21to15_5.json")
        )
        self.dedup_eps = float(rospy.get_param("~dedup_epsilon_m", 0.05))
        self.max_points = int(rospy.get_param("~max_points", 0))  # 0이면 무제한

        self.points: List[List[float]] = []  # [x, y]

        rospy.Subscriber(self.topic, BEVInfo, self._cb, queue_size=10)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "[BEVPathRecorder] listening %s target_id=%d -> %s (dedup=%.2f m, max_points=%d)",
            self.topic,
            self.target_id,
            self.output_path,
            self.dedup_eps,
            self.max_points,
        )

    def _too_close(self, x: float, y: float) -> bool:
        if not self.points:
            return False
        lx, ly = self.points[-1]
        return math.hypot(x - lx, y - ly) < self.dedup_eps

    def _cb(self, msg: BEVInfo) -> None:
        if self.max_points > 0 and len(self.points) >= self.max_points:
            return
        try:
            ids = list(msg.ids)
            xs = list(msg.center_xs)
            ys = list(msg.center_ys)
        except Exception:
            return
        if not ids or len(xs) != len(ids) or len(ys) != len(ids):
            return
        for idx, vid in enumerate(ids):
            if int(vid) != self.target_id:
                continue
            x = float(xs[idx])
            y = float(ys[idx])
            if self._too_close(x, y):
                return
            self.points.append([x, y])
            rospy.loginfo_throttle(
                0.5,
                "[BEVPathRecorder] captured id=%d pts=%d (x=%.2f, y=%.2f)",
                self.target_id,
                len(self.points),
                x,
                y,
            )
            return

    def _on_shutdown(self) -> None:
        if not self.points:
            rospy.loginfo("[BEVPathRecorder] no points captured; skip saving")
            return
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        except Exception:
            pass
        data = {"target_id": self.target_id, "points": self.points}
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            rospy.loginfo("[BEVPathRecorder] saved %d points to %s", len(self.points), self.output_path)
        except Exception as exc:
            rospy.logerr("[BEVPathRecorder] failed to save %s: %s", self.output_path, exc)

    def spin(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    try:
        BEVPathRecorder().spin()
    except rospy.ROSInterruptException:
        pass


