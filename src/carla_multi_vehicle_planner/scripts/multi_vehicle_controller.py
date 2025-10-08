#!/usr/bin/env python3

import math
import sys
import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry

CARLA_EGG = "/home/ctrl/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg"
if CARLA_EGG not in sys.path:
    sys.path.insert(0, CARLA_EGG)

try:
    import carla
except ImportError as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


def quaternion_to_yaw(orientation):
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MultiVehicleController:
    def __init__(self):
        rospy.init_node("multi_vehicle_controller", anonymous=True)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.num_vehicles = rospy.get_param("~num_vehicles", 3)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 5.0)
        self.wheelbase = rospy.get_param("~wheelbase", 2.8)
        self.max_steer = rospy.get_param("~max_steer", 0.7)
        self.target_speed = rospy.get_param("~target_speed", 8.0)
        self.control_frequency = rospy.get_param("~control_frequency", 50.0)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        self.vehicles = {}
        self.states = {}
        self.control_publishers = {}
        self.pose_publishers = {}

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            self.states[role] = {
                "path": [],
                "current_index": 0,
                "position": None,
                "orientation": None,
                "s_profile": [],
                "path_length": 0.0,
                "progress_s": 0.0,
                "remaining_distance": None,
            }
            path_topic = f"/planned_path_{role}"
            odom_topic = f"/carla/{role}/odometry"
            control_topic = f"/carla/{role}/vehicle_control_cmd"
            pose_topic = f"/{role}/pose"
            rospy.Subscriber(path_topic, Path, self._path_cb, callback_args=role)
            rospy.Subscriber(odom_topic, Odometry, self._odom_cb, callback_args=role)
            self.control_publishers[role] = rospy.Publisher(control_topic, AckermannDrive, queue_size=1)
            self.pose_publishers[role] = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / 50.0), self._refresh_vehicles)
        rospy.Timer(rospy.Duration(1.0 / 50.0), self._control_loop)
        rospy.on_shutdown(self._shutdown)

    def _role_name(self, index):
        return f"ego_vehicle_{index + 1}"

    def _refresh_vehicles(self, _event):
        actors = self.world.get_actors().filter("vehicle.*")
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role in self.states:
                self.vehicles[role] = actor

    def _path_cb(self, msg, role):
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.states[role]["path"] = points
        self.states[role]["current_index"] = 0
        s_profile, total_len = self._compute_path_profile(points)
        self.states[role]["s_profile"] = s_profile
        self.states[role]["path_length"] = total_len
        self.states[role]["progress_s"] = 0.0
        self.states[role]["remaining_distance"] = total_len if total_len > 0.0 else None
        rospy.loginfo(f"{role}: received path with {len(points)} points")

    def _odom_cb(self, msg, role):
        self.states[role]["position"] = msg.pose.pose.position
        self.states[role]["orientation"] = msg.pose.pose.orientation
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pose_publishers[role].publish(pose_msg)

    def _compute_path_profile(self, points):
        if len(points) < 2:
            return [], 0.0
        cumulative = [0.0]
        total = 0.0
        for idx in range(1, len(points)):
            step = math.hypot(
                points[idx][0] - points[idx - 1][0],
                points[idx][1] - points[idx - 1][1],
            )
            total += step
            cumulative.append(total)
        return cumulative, total

    def _front_point(self, role, state, vehicle):
        position = state.get("position")
        orientation = state.get("orientation")
        if position is None or orientation is None:
            return None
        yaw = quaternion_to_yaw(orientation)
        offset = 2.0
        if vehicle is not None:
            bb = getattr(vehicle, "bounding_box", None)
            if bb is not None and getattr(bb, "extent", None) is not None:
                offset = bb.extent.x + 0.3
        fx = position.x + math.cos(yaw) * offset
        fy = position.y + math.sin(yaw) * offset
        return fx, fy

    def _project_progress(self, path, s_profile, px, py):
        if len(path) < 2 or not s_profile:
            return None
        best_dist_sq = float("inf")
        best_index = None
        best_t = 0.0
        for idx in range(len(path) - 1):
            x1, y1 = path[idx]
            x2, y2 = path[idx + 1]
            seg_dx = x2 - x1
            seg_dy = y2 - y1
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len_sq < 1e-6:
                continue
            t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + seg_dx * t
            proj_y = y1 + seg_dy * t
            dist_sq = (proj_x - px) ** 2 + (proj_y - py) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_index = idx
                best_t = t
        if best_index is None:
            return None
        seg_length = math.hypot(
            path[best_index + 1][0] - path[best_index][0],
            path[best_index + 1][1] - path[best_index][1],
        )
        if seg_length < 1e-6:
            s_now = s_profile[best_index]
        else:
            s_now = s_profile[best_index] + best_t * seg_length
        return s_now, best_index

    def _update_progress(self, role, state, vehicle):
        path = state.get("path")
        s_profile = state.get("s_profile")
        if not path or not s_profile:
            return
        front = self._front_point(role, state, vehicle)
        if front is None:
            return
        projection = self._project_progress(path, s_profile, front[0], front[1])
        if projection is None:
            return
        s_now, index = projection
        state["progress_s"] = s_now
        total_len = state.get("path_length", 0.0)
        if total_len > 0.0:
            state["remaining_distance"] = max(0.0, total_len - s_now)
        else:
            state["remaining_distance"] = None
        state["current_index"] = min(index, len(path) - 1)

    def _control_loop(self, _event):
        for role, state in self.states.items():
            vehicle = self.vehicles.get(role)
            if vehicle is None:
                continue
            self._update_progress(role, state, vehicle)
            steer, speed = self._compute_control(state)
            if steer is None:
                continue
            self._apply_carla_control(role, vehicle, steer, speed)
            self._publish_ackermann(role, steer, speed)

    def _compute_control(self, state):
        path = state["path"]
        position = state["position"]
        orientation = state["orientation"]
        if not path or position is None or orientation is None:
            return None, None

        index = state["current_index"]
        x = position.x
        y = position.y
        yaw = quaternion_to_yaw(orientation)

        while index < len(path):
            px, py = path[index]
            if math.hypot(px - x, py - y) > self.lookahead_distance * 0.5:
                break
            index += 1
        state["current_index"] = min(index, len(path) - 1)

        target = None
        for offset in range(len(path)):
            candidate_index = (state["current_index"] + offset) % len(path)
            px, py = path[candidate_index]
            dist = math.hypot(px - x, py - y)
            if dist >= self.lookahead_distance:
                target = (px, py)
                break
        if target is None:
            target = path[-1]

        tx, ty = target
        dx = tx - x
        dy = ty - y
        alpha = math.atan2(dy, dx) - yaw
        while alpha > math.pi:
            alpha -= 2.0 * math.pi
        while alpha < -math.pi:
            alpha += 2.0 * math.pi

        Ld = math.hypot(dx, dy)
        if Ld < 1e-3:
            return 0.0, 0.0

        steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)
        steer = max(-self.max_steer, min(self.max_steer, steer))

        speed = self.target_speed
        if abs(steer) > 0.2:
            speed *= 0.7
        if Ld < self.lookahead_distance:
            speed *= max(0.0, Ld / self.lookahead_distance)

        return steer, speed

    def _apply_carla_control(self, role, vehicle, steer, speed):
        control = carla.VehicleControl()
        control.steer = max(-1.0, min(1.0, steer / max(1e-3, self.max_steer)))
        current_velocity = vehicle.get_velocity()
        current_speed = math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2 + current_velocity.z ** 2)
        speed_error = speed - current_speed
        if speed_error > 0:
            control.throttle = max(0.2, min(1.0, speed_error / max(1.0, speed)))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(1.0, -speed_error / max(1.0, self.target_speed))
        vehicle.apply_control(control)


    def _publish_ackermann(self, role, steer, speed):
        msg = AckermannDrive()
        msg.steering_angle = steer
        msg.speed = speed
        self.control_publishers[role].publish(msg)

    def _shutdown(self):
        pass


if __name__ == "__main__":
    try:
        MultiVehicleController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
