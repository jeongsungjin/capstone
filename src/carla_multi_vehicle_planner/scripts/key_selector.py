#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rospy
from std_msgs.msg import String


class KeySelectorNode:
    def __init__(self) -> None:
        self._node_name = rospy.get_name() or "key_selector"
        self._pub = rospy.Publisher("/selected_vehicle", String, queue_size=1, latch=True)
        self._stdin = sys.stdin
        if not self._stdin.isatty():
            rospy.logerr("%s: stdin is not a TTY; cannot capture key presses", self._node_name)
            raise RuntimeError("stdin must be a TTY")

        self._fd = self._stdin.fileno()
        self._orig_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._restored = False
        rospy.on_shutdown(self._restore_terminal)

        rospy.loginfo("Press 1/2/3 to choose vehicle. ESC to exit.")
        self._publish_selection("ego_vehicle_1")

    def spin(self) -> None:
        rate = rospy.Rate(50)
        try:
            while not rospy.is_shutdown():
                if self._stdin in select.select([self._stdin], [], [], 0.1)[0]:
                    key = self._stdin.read(1)
                    if not key:
                        continue
                    if key == "\x1b":  # ESC
                        rospy.loginfo("%s: ESC pressed, exiting", self._node_name)
                        break
                    if key == "\x03":  # Ctrl-C
                        raise KeyboardInterrupt
                    self._handle_key(key)
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("%s: Ctrl-C pressed, exiting", self._node_name)
        finally:
            self._restore_terminal()

    def _handle_key(self, key: str) -> None:
        mapping = {
            "1": "ego_vehicle_1",
            "2": "ego_vehicle_2",
            "3": "ego_vehicle_3",
        }
        if key in mapping:
            self._publish_selection(mapping[key])

    def _publish_selection(self, vehicle_id: str) -> None:
        msg = String(data=vehicle_id)
        self._pub.publish(msg)
        rospy.loginfo("%s: selected %s", self._node_name, vehicle_id)

    def _restore_terminal(self) -> None:
        if not self._restored:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._orig_settings)
            self._restored = True


def main() -> None:
    rospy.init_node("key_selector", anonymous=False)
    try:
        node = KeySelectorNode()
    except RuntimeError as exc:
        rospy.logfatal("key_selector: %s", exc)
        return
    node.spin()


if __name__ == "__main__":
    main()
