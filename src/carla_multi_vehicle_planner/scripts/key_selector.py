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
        # 선택 퍼블리시는 더 이상 사용하지 않지만 호환을 위해 유지
        self._pub = rospy.Publisher("/selected_vehicle", String, queue_size=1, latch=True)
        self._e_stop_pub = rospy.Publisher("/emergency_stop", Bool, queue_size=1, latch=True)
        self._spawn_person_pub = rospy.Publisher("/spawn_person_mode", Bool, queue_size=1, latch=True)
        self._emergency_active_all = False  # 전체 차량 긴급정지 상태
        self._emergency_roles = set()      # 개별 차량 긴급정지 활성 목록
        self._spawn_person_active = False
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
        self._selected = None  # 선택 상태 사용 안 함
        self._stdin = sys.stdin
        if not self._stdin.isatty():
            rospy.logerr("%s: stdin is not a TTY; cannot capture key presses", self._node_name)
            raise RuntimeError("stdin must be a TTY")

        self._fd = self._stdin.fileno()
        self._orig_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._restored = False
        rospy.on_shutdown(self._restore_terminal)

        rospy.loginfo("Press 1-6 to TOGGLE per-vehicle E-STOP, SPACE for ALL, ESC to exit.")

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
                    if key.lower() == "o":  # toggle RViz person spawn mode
                        self._toggle_spawn_person()
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
        # 숫자 입력 즉시 해당 차량 E-STOP 토글
        if key in mapping:
            self._toggle_role_emergency(mapping[key])

    def _publish_selection(self, vehicle_id: str) -> None:
        # 더 이상 선택 기능을 사용하지 않지만 인터페이스만 유지
        msg = String(data=vehicle_id)
        self._pub.publish(msg)

    def _toggle_emergency(self) -> None:
        self._emergency_active_all = not self._emergency_active_all
        if self._emergency_active_all:
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

    def _toggle_selected_emergency(self) -> None:
        self._toggle_role_emergency(self._selected)

    def _toggle_role_emergency(self, role: str) -> None:
        if role not in self._roles:
            rospy.logwarn("%s: role %s unknown for per-vehicle E-STOP", self._node_name, role)
            return
        if role in self._emergency_roles:
            self._emergency_roles.remove(role)
            rospy.logwarn("%s: E-STOP OFF for %s", self._node_name, role)
        else:
            self._emergency_roles.add(role)
            rospy.logwarn("%s: E-STOP ON for %s", self._node_name, role)
        if self._emergency_active_all or self._emergency_roles:
            self._start_stop_burst()
        else:
            self._stop_stop_burst()

    def _toggle_spawn_person(self) -> None:
        self._spawn_person_active = not self._spawn_person_active
        try:
            self._spawn_person_pub.publish(Bool(data=self._spawn_person_active))
        except Exception:
            pass
        rospy.loginfo("%s: spawn person mode %s", self._node_name, "ON" if self._spawn_person_active else "OFF")

    def _start_stop_burst(self) -> None:
        if self._e_stop_timer is not None:
            return
        period = 1.0 / max(1.0, self._e_stop_hz)
        self._e_stop_timer = rospy.Timer(rospy.Duration(period), self._e_stop_tick)

    def _stop_stop_burst(self) -> None:
        if self._e_stop_timer is not None and not self._emergency_active_all and not self._emergency_roles:
            try:
                self._e_stop_timer.shutdown()
            except Exception:
                pass
            self._e_stop_timer = None

    def _e_stop_tick(self, _evt) -> None:
        if not self._emergency_active_all and not self._emergency_roles:
            return
        stop = AckermannDrive()
        stop.speed = 0.0
        stop.steering_angle = 0.0
        for role, pub in self._stop_pubs.items():
            if self._emergency_active_all or role in self._emergency_roles:
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
