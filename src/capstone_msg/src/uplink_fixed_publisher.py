#!/usr/bin/env python3
import math

import rospy
from capstone_msgs.msg import Uplink


def main():
    rospy.init_node("uplink_fixed_publisher")
    pub = rospy.Publisher("/uplink", Uplink, queue_size=1)
    rate_hz = rospy.get_param("~rate", 5.0)
    rate = rospy.Rate(rate_hz)
    vehicle_id = 1
    voltage = 4.5
    while not rospy.is_shutdown():
        msg = Uplink()
        msg.vehicle_id = vehicle_id
        msg.voltage = voltage
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

