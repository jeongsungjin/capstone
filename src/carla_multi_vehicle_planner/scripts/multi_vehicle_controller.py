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
        self.max_steer = rospy.get_param("~max_steer", 1.0)
        self.target_speed = rospy.get_param("~target_speed", 3.0)
        self.control_frequency = rospy.get_param("~control_frequency", 30.0)

        # Safety stop parameters (area-limited)
        self.enable_safety_stop = bool(rospy.get_param("~enable_safety_stop", True))
        self.safety_x_min = float(rospy.get_param("~safety_x_min", -13.0))
        self.safety_x_max = float(rospy.get_param("~safety_x_max", 13.0))
        self.safety_y_min = float(rospy.get_param("~safety_y_min", -40.0))
        self.safety_y_max = float(rospy.get_param("~safety_y_max", 5.0))
        self.safety_distance = float(rospy.get_param("~safety_distance", 18.0))
        # Consider only vehicles within my front cone and approaching
        self.safety_front_cone_deg = float(rospy.get_param("~safety_front_cone_deg", 100.0))
        self.safety_require_closing = bool(rospy.get_param("~safety_require_closing", True))
        # Deadlock escape tuning
        self.safety_deadlock_timeout_sec = float(rospy.get_param("~safety_deadlock_timeout_sec", 3.0))
        self.safety_deadlock_escape_speed = float(rospy.get_param("~safety_deadlock_escape_speed", 1.0))
        self.safety_deadlock_escape_duration_sec = float(rospy.get_param("~safety_deadlock_escape_duration_sec", 2.0))
        # Opposite-lane ignore within intersection area
        self.enable_opposite_lane_ignore = bool(rospy.get_param("~enable_opposite_lane_ignore", True))
        self.opposite_lane_heading_thresh_deg = float(rospy.get_param("~opposite_lane_heading_thresh_deg", 100.0))

        # Intersection dynamic priority (entry-order based)
        self.intersection_dynamic_priority = bool(rospy.get_param("~intersection_dynamic_priority", True))
        self.intersection_x_min = float(rospy.get_param("~intersection_x_min", self.safety_x_min))
        self.intersection_x_max = float(rospy.get_param("~intersection_x_max", self.safety_x_max))
        self.intersection_y_min = float(rospy.get_param("~intersection_y_min", self.safety_y_min))
        self.intersection_y_max = float(rospy.get_param("~intersection_y_max", self.safety_y_max))

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        self.vehicles = {}
        self.states = {}
        self.control_publishers = {}
        self.pose_publishers = {}
        # Track per-role intersection entry order
        self.intersection_order = {}
        self._intersection_counter = 0

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
                # safety/deadlock fields
                "safety_stop_since": None,
                "deadlock_escape_until": None,
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

    def _role_index_from_name(self, role):
        try:
            return max(1, int(role.split("_")[-1]))
        except Exception:
            return 9999

    def _in_safety_area(self, position):
        if position is None:
            return False
        x = position.x
        y = position.y
        return (self.safety_x_min <= x <= self.safety_x_max) and (self.safety_y_min <= y <= self.safety_y_max)

    def _in_intersection_area(self, position):
        if position is None:
            return False
        x = position.x
        y = position.y
        return (self.intersection_x_min <= x <= self.intersection_x_max) and (self.intersection_y_min <= y <= self.intersection_y_max)

    def _normalize_angle(self, ang):
        while ang > math.pi:
            ang -= 2.0 * math.pi
        while ang < -math.pi:
            ang += 2.0 * math.pi
        return ang

    def _is_opposite_lane(self, my_position, other_position):
        try:
            if self.carla_map is None or my_position is None or other_position is None:
                return False
            my_loc = carla.Location(x=my_position.x, y=my_position.y, z=0.0)
            ot_loc = carla.Location(x=other_position.x, y=other_position.y, z=0.0)
            my_wp = self.carla_map.get_waypoint(my_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            ot_wp = self.carla_map.get_waypoint(ot_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if my_wp is None or ot_wp is None:
                return False
            # Primary: same road and lane_id sign differs (typical opposite direction)
            try:
                if my_wp.road_id == ot_wp.road_id and (my_wp.lane_id * ot_wp.lane_id) < 0:
                    return True
            except Exception:
                pass
            # Fallback: opposite heading
            my_yaw = math.radians(my_wp.transform.rotation.yaw)
            ot_yaw = math.radians(ot_wp.transform.rotation.yaw)
            diff = abs(self._normalize_angle(my_yaw - ot_yaw))
            if diff >= math.radians(max(0.0, min(175.0, self.opposite_lane_heading_thresh_deg))):
                return True
        except Exception:
            return False
        return False

    def _has_nearby_higher_priority(self, my_role, my_position, my_yaw):
        my_index = self._role_index_from_name(my_role)
        if my_position is None or my_index is None or my_yaw is None:
            return False
        forward_x = math.cos(my_yaw)
        forward_y = math.sin(my_yaw)
        cos_limit = math.cos(math.radians(max(0.0, min(175.0, self.safety_front_cone_deg)) / 2.0))
        my_actor = self.vehicles.get(my_role)
        my_vx = 0.0
        my_vy = 0.0
        if my_actor is not None:
            v = my_actor.get_velocity()
            my_vx, my_vy = v.x, v.y
        in_intersection = self._in_intersection_area(my_position)
        my_order = self.intersection_order.get(my_role)
        for other_role, other_state in self.states.items():
            if other_role == my_role:
                continue
            other_pos = other_state.get("position")
            if other_pos is None:
                continue
            # Decide priority basis
            higher = False
            if self.intersection_dynamic_priority and in_intersection and self._in_intersection_area(other_pos):
                other_order = self.intersection_order.get(other_role)
                if other_order is not None and my_order is not None:
                    higher = other_order < my_order
                elif other_order is not None and my_order is None:
                    higher = True
                else:
                    other_index = self._role_index_from_name(other_role)
                    higher = other_index < my_index
            else:
                other_index = self._role_index_from_name(other_role)
                higher = other_index < my_index

            if not higher:
                continue

            dx = (other_pos.x - my_position.x)
            dy = (other_pos.y - my_position.y)
            dist = math.hypot(dx, dy)
            if dist > self.safety_distance:
                continue
            if self.enable_opposite_lane_ignore and self._in_safety_area(my_position):
                try:
                    if self._is_opposite_lane(my_position, other_pos):
                        continue
                except Exception:
                    pass
            if dist > 1e-3:
                dir_dot = (dx * forward_x + dy * forward_y) / dist
                if dir_dot < cos_limit:
                    continue
            if self.safety_require_closing:
                other_actor = self.vehicles.get(other_role)
                ovx = ovy = 0.0
                if other_actor is not None:
                    ov = other_actor.get_velocity()
                    ovx, ovy = ov.x, ov.y
                closing_rate = (dx * (ovx - my_vx) + dy * (ovy - my_vy)) / max(dist, 1e-3)
                if closing_rate >= 0.0:
                    continue
            return True
        return False

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
            # Area-limited safety stop for lower-priority vehicles near other vehicles
            if self.enable_safety_stop:
                position = state.get("position")
                orientation = state.get("orientation")
                my_yaw = quaternion_to_yaw(orientation) if orientation is not None else None
                now_sec = float(rospy.Time.now().to_sec())
                in_area = self._in_safety_area(position)
                # dynamic priority entry tracking inside intersection area
                if self.intersection_dynamic_priority:
                    if self._in_intersection_area(position):
                        if role not in self.intersection_order:
                            self._intersection_counter += 1
                            self.intersection_order[role] = self._intersection_counter
                    else:
                        if role in self.intersection_order:
                            self.intersection_order.pop(role, None)
                has_higher_front = self._has_nearby_higher_priority(role, position, my_yaw) if in_area else False
                escape_until = state.get("deadlock_escape_until")
                if escape_until is not None and now_sec < escape_until:
                    # In escape window: allow creeping
                    speed = min(speed, self.safety_deadlock_escape_speed)
                elif in_area and has_higher_front:
                    # Apply safety stop with deadlock timer
                    since = state.get("safety_stop_since")
                    if since is None:
                        state["safety_stop_since"] = now_sec
                        speed = 0.0
                    else:
                        if (now_sec - since) >= self.safety_deadlock_timeout_sec:
                            # Start escape window and creep
                            state["deadlock_escape_until"] = now_sec + self.safety_deadlock_escape_duration_sec
                            speed = min(speed, self.safety_deadlock_escape_speed)
                        else:
                            speed = 0.0
                else:
                    # Reset timers if not actively safety stopping
                    state["safety_stop_since"] = None
                    # Do not reset escape_until so window can complete
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
