#!/usr/bin/env python3

"""Path multiplexer that chooses between global and local routes per vehicle."""

import threading
from typing import Dict, Optional

import rospy
from nav_msgs.msg import Path


class PathMultiplexer:
    def __init__(self) -> None:
        rospy.init_node("path_mux", anonymous=True)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.local_fresh_timeout = float(rospy.get_param("~local_fresh_timeout", 0.5))
        if self.local_fresh_timeout <= 0.0:
            self.local_fresh_timeout = 0.1

        self._lock = threading.RLock()
        self._global_paths: Dict[str, Optional[Path]] = {}
        self._local_paths: Dict[str, Optional[Path]] = {}
        self._local_stamps: Dict[str, rospy.Time] = {}
        self._publishers: Dict[str, rospy.Publisher] = {}
        self._active_source: Dict[str, Optional[str]] = {}

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            self._publishers[role] = rospy.Publisher(
                f"/planned_path_{role}", Path, queue_size=1, latch=True
            )
            rospy.Subscriber(
                f"/global_path_{role}", Path, self._global_cb, callback_args=role
            )
            rospy.Subscriber(
                f"/local_path_{role}", Path, self._local_cb, callback_args=role
            )
            self._global_paths[role] = None
            self._local_paths[role] = None
            self._local_stamps[role] = rospy.Time(0)
            self._active_source[role] = None

        rospy.Timer(rospy.Duration(0.5), self._timeout_timer)

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _global_cb(self, msg: Path, role: str) -> None:
        with self._lock:
            self._global_paths[role] = msg
            if not self._should_use_local(role):
                self._publish(role, msg, "global")

    def _local_cb(self, msg: Path, role: str) -> None:
        with self._lock:
            if not msg.poses:
                self._local_paths[role] = None
                self._local_stamps[role] = rospy.Time(0)
                global_path = self._global_paths.get(role)
                if global_path is not None:
                    self._publish(role, global_path, "global")
                return

            stamp = msg.header.stamp
            if stamp == rospy.Time():
                stamp = rospy.Time.now()

            self._local_paths[role] = msg
            self._local_stamps[role] = stamp
            age = (rospy.Time.now() - stamp).to_sec()
            if age > self.local_fresh_timeout:
                rospy.logwarn(
                    "%s: received stale local path (%.2fs old), ignoring",
                    role,
                    age,
                )
                self._local_paths[role] = None
                self._local_stamps[role] = rospy.Time(0)
                global_path = self._global_paths.get(role)
                if global_path is not None:
                    self._publish(role, global_path, "global")
                return

            self._publish(role, msg, "local")

    def _timeout_timer(self, _event) -> None:
        with self._lock:
            now = rospy.Time.now()
            for role, stamp in self._local_stamps.items():
                if self._local_paths.get(role) is None:
                    continue
                if (now - stamp).to_sec() > self.local_fresh_timeout:
                    rospy.loginfo(
                        "%s: local path expired (%.2fs old), falling back to global",
                        role,
                        (now - stamp).to_sec(),
                    )
                    self._local_paths[role] = None
                    self._local_stamps[role] = rospy.Time(0)
                    global_path = self._global_paths.get(role)
                    if global_path is not None:
                        self._publish(role, global_path, "global")

    def _should_use_local(self, role: str) -> bool:
        local_path = self._local_paths.get(role)
        if local_path is None:
            return False
        stamp = self._local_stamps.get(role, rospy.Time(0))
        age = (rospy.Time.now() - stamp).to_sec()
        return age <= self.local_fresh_timeout

    def _publish(self, role: str, path: Path, source: str) -> None:
        publisher = self._publishers.get(role)
        if publisher is None:
            return
        publisher.publish(path)
        if self._active_source.get(role) != source:
            rospy.loginfo("%s: planning source → %s", role, source)
            self._active_source[role] = source


def main() -> None:
    PathMultiplexer()
    rospy.spin()


if __name__ == "__main__":
    main()
