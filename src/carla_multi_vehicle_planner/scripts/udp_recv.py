#!/usr/bin/env python3
import socket
import re
from typing import Dict

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

# === 설정 (ROS 파라미터로도 제어 가능) ===
UDP_IP = rospy.get_param("~udp_ip", "0.0.0.0") if rospy.core.is_initialized() else "0.0.0.0"
UDP_PORT = int(rospy.get_param("~udp_port", 9001)) if rospy.core.is_initialized() else 9001

def _role_name(vid: int) -> str:
    return f"ego_vehicle_{max(1, int(vid))}"


def main():
    rospy.init_node("udp_goal_bridge", anonymous=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    rospy.loginfo("udp_goal_bridge: listening on %s:%d", UDP_IP, UDP_PORT)

    selected_pub = rospy.Publisher("/selected_vehicle", String, queue_size=1, latch=True)
    click_pub = rospy.Publisher("/click_button", String, queue_size=1, latch=False)
    override_publishers: Dict[str, rospy.Publisher] = {}

    def get_override_pub(role: str) -> rospy.Publisher:
        pub = override_publishers.get(role)
        if pub is None:
            topic = f"/override_goal/{role}"
            pub = rospy.Publisher(topic, PoseStamped, queue_size=1, latch=False)
            override_publishers[role] = pub
        return pub

    current_vehicle_id = None

    try:
        while not rospy.is_shutdown():
            data, _ = sock.recvfrom(1024)
            msg = data.decode("utf-8", errors="ignore").strip()
            if not msg:
                continue

            # 0) click 이벤트
            if msg.lower() == "click":
                click_pub.publish(String(data="click"))
                rospy.loginfo("udp_goal_bridge: click → /click_button")
                continue

            # 1) 차량 선택 (숫자만)
            if msg.isdigit():
                current_vehicle_id = int(msg)
                role = _role_name(current_vehicle_id)
                selected_pub.publish(String(data=role))
                rospy.loginfo("udp_goal_bridge: selected %s", role)
                continue

            # 2) 좌표 수신 ("X=..." 형태)
            if msg.startswith("X="):
                coords = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", msg)
                if len(coords) >= 3:
                    x, y, z = map(float, coords[:3])
                    if current_vehicle_id is None:
                        rospy.logwarn_throttle(2.0, "udp_goal_bridge: vehicle_id not set; ignoring coords")
                        continue
                    role = _role_name(current_vehicle_id)
                    pub = get_override_pub(role)
                    # Re-publish selection to ensure downstream nodes latch current role
                    selected_pub.publish(String(data=role))
                    ps = PoseStamped()
                    ps.header.stamp = rospy.Time.now()
                    ps.header.frame_id = "map"
                    ps.pose.position.x = x
                    ps.pose.position.y = y
                    ps.pose.position.z = z
                    ps.pose.orientation.w = 1.0
                    pub.publish(ps)
                    rospy.loginfo("udp_goal_bridge: goal → %s at (%.2f, %.2f)", role, x, y)

                    # Optionally repeat goal after a short delay to avoid first-click miss
                    repeat_count = int(rospy.get_param("~goal_repeat_count", 1))
                    repeat_delay = float(rospy.get_param("~goal_repeat_delay", 0.1))
                    repeat_count = max(0, repeat_count)

                    if repeat_count > 0 and repeat_delay > 0.0:
                        def _republish(_event, rx=x, ry=y, rz=z, rrole=role):
                            repub = get_override_pub(rrole)
                            ps2 = PoseStamped()
                            ps2.header.stamp = rospy.Time.now()
                            ps2.header.frame_id = "map"
                            ps2.pose.position.x = rx
                            ps2.pose.position.y = ry
                            ps2.pose.position.z = rz
                            ps2.pose.orientation.w = 1.0
                            repub.publish(ps2)
                        # schedule repeats
                        for i in range(repeat_count):
                            rospy.Timer(rospy.Duration(repeat_delay * (i + 1)), _republish, oneshot=True)
                continue

            # 3) 기타 메시지
            rospy.logdebug("udp_goal_bridge: unknown msg '%s'", msg)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
