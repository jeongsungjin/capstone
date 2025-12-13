#!/usr/bin/env python3
import math

import rospy
from sensor_msgs.msg import Imu
from capstone_msgs.msg import Uplink


def make_dummy_imu() -> Imu:
    msg = Imu()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "imu"
    # Arbitrary small values for testing
    msg.orientation.w = 1.0
    msg.angular_velocity.x = 0.01
    msg.angular_velocity.y = -0.01
    msg.angular_velocity.z = 0.02
    msg.linear_acceleration.x = 0.1
    msg.linear_acceleration.y = 0.0
    msg.linear_acceleration.z = -9.7
    return msg


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
        msg.imu = make_dummy_imu()
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

