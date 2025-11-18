#!/usr/bin/env python3

import socket
import time

import rospy


def main():
    rospy.init_node("ui_mode_sender", anonymous=False)

    # Params
    enable_platooning = bool(rospy.get_param("~enable_platooning", False))
    udp_host = str(rospy.get_param("~udp_host", "127.0.0.1"))
    udp_port = int(rospy.get_param("~udp_port", 9002))
    repeat_count = int(rospy.get_param("~repeat_count", 3))
    repeat_delay = float(rospy.get_param("~repeat_delay", 0.1))

    message = f"platooning:{'true' if enable_platooning else 'false'}"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for i in range(max(1, repeat_count)):
            try:
                sock.sendto(message.encode("utf-8"), (udp_host, udp_port))
                rospy.loginfo("ui_mode_sender: sent '%s' to %s:%d (%d/%d)",
                              message, udp_host, udp_port, i + 1, repeat_count)
            except Exception as exc:
                rospy.logwarn("ui_mode_sender: send failed: %s", exc)
            if i + 1 < repeat_count and repeat_delay > 0.0:
                time.sleep(repeat_delay)
    finally:
        try:
            sock.close()
        except Exception:
            pass

    # Exit after sending
    rospy.loginfo("ui_mode_sender: done")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


