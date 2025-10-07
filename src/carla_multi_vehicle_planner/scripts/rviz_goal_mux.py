#!/usr/bin/env python3

import threading

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class RvizGoalMux:
    def __init__(self) -> None:
        self._node_name = rospy.get_name() or "rviz_goal_mux"
        self._lock = threading.Lock()
        self._current_role = rospy.get_param("~default_vehicle", "ego_vehicle_1")
        self._publishers = {}

        rospy.loginfo("%s: default vehicle set to %s", self._node_name, self._current_role)

        self._selection_sub = rospy.Subscriber(
            "/selected_vehicle", String, self._selection_callback, queue_size=1
        )
        self._goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self._goal_callback, queue_size=10
        )

    def _selection_callback(self, msg: String) -> None:
        role = msg.data.strip()
        if not role:
            return
        with self._lock:
            self._current_role = role
        rospy.loginfo("%s: switched to %s", self._node_name, role)

    def _goal_callback(self, msg: PoseStamped) -> None:
        with self._lock:
            role = self._current_role
        publisher = self._get_publisher(role)
        if publisher is None:
            rospy.logerr("%s: unable to publish goal, no role selected", self._node_name)
            return
        publisher.publish(msg)
        rospy.loginfo("%s: forwarded goal to /override_goal/%s", self._node_name, role)

    def _get_publisher(self, role: str):
        if not role:
            return None
        with self._lock:
            if role not in self._publishers:
                topic = f"/override_goal/{role}"
                self._publishers[role] = rospy.Publisher(topic, PoseStamped, queue_size=10)
            return self._publishers[role]


def main() -> None:
    rospy.init_node("rviz_goal_mux", anonymous=False)
    node = RvizGoalMux()
    rospy.spin()


if __name__ == "__main__":
    main()
