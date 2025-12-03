#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple

import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry

try:
    import carla  # type: ignore
except Exception as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


def quaternion_to_yaw(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SimpleMultiVehicleController:
    """
    최소 기능 컨트롤러:
    - /planned_path_{role} 구독, /carla/{role}/odometry 구독
    - Pure Pursuit으로 조향/속도 산출
    - CARLA VehicleControl 적용 + /carla/{role}/vehicle_control_cmd 퍼블리시
    - 안전/우선순위/플래툰/영역 게인/추가 오버라이드 기능 제거
    """

    def __init__(self) -> None:
        rospy.init_node("multi_vehicle_controller", anonymous=True)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.lookahead_distance = float(rospy.get_param("~lookahead_distance", 1.0))
        self.wheelbase = float(rospy.get_param("~wheelbase", 1.75))
        self.max_steer = float(rospy.get_param("~max_steer", 1.0))
        self.target_speed = float(rospy.get_param("~target_speed", 2.0))
        self.control_frequency = float(rospy.get_param("~control_frequency", 30.0))

        # CARLA world
        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        timeout = float(rospy.get_param("~carla_timeout", 5.0))
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()

        # State
        self.vehicles: Dict[str, carla.Actor] = {}
        self.states: Dict[str, Dict] = {}
        self.control_publishers: Dict[str, rospy.Publisher] = {}
        self.pose_publishers: Dict[str, rospy.Publisher] = {}

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            self.states[role] = {
                "path": [],  # List[(x, y)]
                "position": None,
                "orientation": None,
                "current_index": 0,
                "s_profile": [],
                "progress_s": 0.0,
                "path_length": 0.0,
            }
            path_topic = f"/planned_path_{role}"
            odom_topic = f"/carla/{role}/odometry"
            cmd_topic = f"/carla/{role}/vehicle_control_cmd"
            pose_topic = f"/{role}/pose"
            rospy.Subscriber(path_topic, Path, self._path_cb, callback_args=role, queue_size=1)
            rospy.Subscriber(odom_topic, Odometry, self._odom_cb, callback_args=role, queue_size=10)
            self.control_publishers[role] = rospy.Publisher(cmd_topic, AckermannDrive, queue_size=1)
            self.pose_publishers[role] = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / 20.0), self._refresh_vehicles)
        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.control_frequency)), self._control_loop)

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _refresh_vehicles(self, _evt) -> None:
        actors = self.world.get_actors().filter("vehicle.*")
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role in self.states:
                self.vehicles[role] = actor

    def _path_cb(self, msg: Path, role: str) -> None:
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        s_profile, total_len = self._compute_path_profile(points)
        st = self.states[role]
        st["path"] = points
        st["current_index"] = 0
        st["s_profile"] = s_profile
        st["path_length"] = total_len
        st["progress_s"] = 0.0

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        st = self.states[role]
        st["position"] = msg.pose.pose.position
        st["orientation"] = msg.pose.pose.orientation
        # also publish pose for tools
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pose_publishers[role].publish(pose_msg)

    def _compute_path_profile(self, points: List[Tuple[float, float]]):
        if len(points) < 2:
            return [], 0.0
        cumulative = [0.0]
        total = 0.0
        for i in range(1, len(points)):
            step = math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            total += step
            cumulative.append(total)
        return cumulative, total

    def _front_point(self, st, vehicle):
        pos = st.get("position")
        ori = st.get("orientation")
        if pos is None or ori is None:
            return None
        yaw = quaternion_to_yaw(ori)
        offset = 2.0
        if vehicle is not None:
            bb = getattr(vehicle, "bounding_box", None)
            if bb is not None and getattr(bb, "extent", None) is not None:
                offset = bb.extent.x + 0.3
        fx = pos.x + math.cos(yaw) * offset
        fy = pos.y + math.sin(yaw) * offset
        return fx, fy, yaw

    def _project_progress(self, path, s_profile, px, py):
        if len(path) < 2 or not s_profile:
            return None
        best_dist_sq = float("inf")
        best_index = None
        best_t = 0.0
        for idx in range(len(path) - 1):
            x1, y1 = path[idx]
            x2, y2 = path[idx + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + dx * t
            proj_y = y1 + dy * t
            dist_sq = (proj_x - px) ** 2 + (proj_y - py) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_index = idx
                best_t = t
        if best_index is None:
            return None
        seg_length = math.hypot(path[best_index + 1][0] - path[best_index][0], path[best_index + 1][1] - path[best_index][1])
        if seg_length < 1e-6:
            s_now = s_profile[best_index]
        else:
            s_now = s_profile[best_index] + best_t * seg_length
        return s_now, best_index

    def _select_target(self, st, x, y) -> Tuple[float, float]:
        path = st.get("path") or []
        s_profile = st.get("s_profile") or []
        if len(path) < 2:
            return x, y
        # arc-length target selection
        s_now = float(st.get("progress_s", 0.0))
        s_target = s_now + float(self.lookahead_distance)
        target_idx = None
        for i, s in enumerate(s_profile):
            if s >= s_target:
                target_idx = i
                break
        if target_idx is None:
            target_idx = len(path) - 1
        target_idx = max(0, min(target_idx, len(path) - 1))
        st["current_index"] = target_idx
        tx, ty = path[target_idx]
        return tx, ty

    def _compute_control(self, st, vehicle):
        path = st.get("path") or []
        pos = st.get("position")
        ori = st.get("orientation")
        if not path or len(path) < 2 or pos is None or ori is None or vehicle is None:
            return None, None
        front = self._front_point(st, vehicle)
        if front is None:
            return None, None
        fx, fy, yaw = front
        proj = self._project_progress(path, st.get("s_profile") or [], fx, fy)
        if proj is not None:
            s_now, idx = proj
            st["progress_s"] = s_now
            st["current_index"] = idx
        tx, ty = self._select_target(st, fx, fy)
        dx = tx - fx
        dy = ty - fy
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
        return steer, speed

    def _apply_carla_control(self, vehicle, steer, speed):
        control = carla.VehicleControl()
        control.steer = max(-1.0, min(1.0, steer / max(1e-3, self.max_steer)))
        v = vehicle.get_velocity()
        current_speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
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
        msg.steering_angle = float(steer)
        msg.speed = float(speed)
        self.control_publishers[role].publish(msg)

    def _control_loop(self, _evt) -> None:
        for role, st in self.states.items():
            vehicle = self.vehicles.get(role)
            if vehicle is None:
                continue
            steer, speed = self._compute_control(st, vehicle)
            if steer is None:
                continue
            self._apply_carla_control(vehicle, steer, speed)
            self._publish_ackermann(role, steer, speed)


if __name__ == "__main__":
    try:
        SimpleMultiVehicleController()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleMultiVehicleController crashed: {e}")
        raise


