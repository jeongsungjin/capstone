#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rospy
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Bool


class KeySelectorNode:
    def __init__(self) -> None:
        self._node_name = rospy.get_name() or "key_selector"
        self._pub = rospy.Publisher("/selected_vehicle", String, queue_size=1, latch=True)
        self._e_stop_pub = rospy.Publisher("/emergency_stop", Bool, queue_size=1, latch=True)
        self._emergency_active = False
        self._e_stop_timer = None
        self._e_stop_hz = float(rospy.get_param("~emergency_burst_hz", 20.0))
        # Emergency stop publishers (speed-only override)
        self._roles = [
            "ego_vehicle_1",
            "ego_vehicle_2",
            "ego_vehicle_3",
            "ego_vehicle_4",
            "ego_vehicle_5",
            "ego_vehicle_6",
        ]
        self._stop_pubs = {
            r: rospy.Publisher(f"/carla/{r}/vehicle_control_cmd_override", AckermannDrive, queue_size=1)
            for r in self._roles
        }
        self._stdin = sys.stdin
        if not self._stdin.isatty():
            rospy.logerr("%s: stdin is not a TTY; cannot capture key presses", self._node_name)
            raise RuntimeError("stdin must be a TTY")

        self._fd = self._stdin.fileno()
        self._orig_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._restored = False
        rospy.on_shutdown(self._restore_terminal)

        rospy.loginfo("Press 1/2/3/4/5/6 to choose vehicle, SPACE to EMERGENCY STOP ALL, ESC to exit.")
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
                    if key == " ":  # SPACE: toggle emergency stop
                        self._toggle_emergency()
                        continue
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
            "4": "ego_vehicle_4",
            "5": "ego_vehicle_5",
            "6": "ego_vehicle_6",
        }
        if key in mapping:
            self._publish_selection(mapping[key])

    def _publish_selection(self, vehicle_id: str) -> None:
        msg = String(data=vehicle_id)
        self._pub.publish(msg)
        rospy.loginfo("%s: selected %s", self._node_name, vehicle_id)

    def _toggle_emergency(self) -> None:
        self._emergency_active = not self._emergency_active
        if self._emergency_active:
            # Activate emergency: latch True and start periodic zero-speed publish
            try:
                self._e_stop_pub.publish(Bool(data=True))
            except Exception:
                pass
            self._start_stop_burst()
            rospy.logwarn("%s: EMERGENCY STOP ON", self._node_name)
        else:
            # Deactivate emergency: latch False and stop periodic publishing
            try:
                self._e_stop_pub.publish(Bool(data=False))
            except Exception:
                pass
            self._stop_stop_burst()
            rospy.logwarn("%s: EMERGENCY STOP OFF", self._node_name)

    def _start_stop_burst(self) -> None:
        if self._e_stop_timer is not None:
            return
        period = 1.0 / max(1.0, self._e_stop_hz)
        self._e_stop_timer = rospy.Timer(rospy.Duration(period), self._e_stop_tick)

    def _stop_stop_burst(self) -> None:
        if self._e_stop_timer is not None:
            try:
                self._e_stop_timer.shutdown()
            except Exception:
                pass
            self._e_stop_timer = None

    def _e_stop_tick(self, _evt) -> None:
        if not self._emergency_active:
            return
        stop = AckermannDrive()
        stop.speed = 0.0
        stop.steering_angle = 0.0
        for role, pub in self._stop_pubs.items():
            try:
                pub.publish(stop)
            except Exception:
                pass

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
