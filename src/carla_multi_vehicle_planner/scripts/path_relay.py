#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path


class PathRelay:
    def __init__(self):
        rospy.init_node("path_relay", anonymous=True)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))

        self._publishers = {}
        for index in range(self.num_vehicles):
            role = f"ego_vehicle_{index + 1}"
            src_topic = f"/global_path_{role}"
            dst_topic = f"/planned_path_{role}"

            pub = rospy.Publisher(dst_topic, Path, queue_size=1, latch=True)
            self._publishers[role] = pub
            rospy.Subscriber(src_topic, Path, self._cb, callback_args=role, queue_size=1)

        rospy.loginfo("path_relay running: /global_path_* -> /planned_path_*")

    def _cb(self, msg: Path, role: str):
        pub = self._publishers.get(role)
        if pub is None:
            return
        pub.publish(msg)


if __name__ == "__main__":
    try:
        PathRelay()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


