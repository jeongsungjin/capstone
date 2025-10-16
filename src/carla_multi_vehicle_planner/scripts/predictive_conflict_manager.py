#!/usr/bin/env python3

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool


@dataclass
class VehicleState:
    role: str
    vehicle_id: int
    path_points: List[Tuple[float, float]] = field(default_factory=list)
    cumulative: List[float] = field(default_factory=list)
    total_length: float = 0.0
    position: Optional[Tuple[float, float]] = None
    speed: float = 0.0
    current_s: Optional[float] = None
    stop_active: bool = False
    last_stop_published: Optional[bool] = None


class PredictiveConflictManager:
    def __init__(self) -> None:
        rospy.init_node("predictive_conflict_manager", anonymous=True)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.collision_threshold = float(rospy.get_param("~collision_threshold", 4.0))
        self.release_threshold = float(rospy.get_param("~release_threshold", 6.0))
        self.prediction_horizon = int(rospy.get_param("~prediction_horizon", 10))
        if self.prediction_horizon <= 0:
            self.prediction_horizon = 10
        self.loop_rate_hz = float(rospy.get_param("~loop_rate_hz", 10.0))
        if self.loop_rate_hz <= 0.0:
            self.loop_rate_hz = 10.0

        self._lock = threading.RLock()
        self._states: Dict[str, VehicleState] = {}
        self._stop_publishers: Dict[str, rospy.Publisher] = {}
        self._path_subscribers = []
        self._odom_subscribers = []
        self._waiting_logged = False

        self._prediction_steps = [float(i) for i in range(1, self.prediction_horizon + 1)]

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            vehicle_id = self._extract_vehicle_id(role) or (index + 1)
            state = VehicleState(role=role, vehicle_id=vehicle_id)
            self._states[role] = state

            path_topic = f"/planned_path_{role}"
            odom_topic = f"/odom_{role}"
            stop_topic = f"/stop_flag_{role}"

            self._path_subscribers.append(
                rospy.Subscriber(path_topic, Path, self._path_callback, callback_args=role)
            )
            self._odom_subscribers.append(
                rospy.Subscriber(odom_topic, Odometry, self._odom_callback, callback_args=role)
            )

            publisher = rospy.Publisher(stop_topic, Bool, queue_size=1, latch=True)
            self._stop_publishers[role] = publisher
            publisher.publish(Bool(data=False))

        rospy.loginfo("predictive_conflict_manager watching %d vehicles", self.num_vehicles)

    def spin(self) -> None:
        rate = rospy.Rate(self.loop_rate_hz)
        while not rospy.is_shutdown():
            with self._lock:
                predictions = {
                    role: self._predict_future(state)
                    for role, state in self._states.items()
                }

                ready_roles = [role for role, preds in predictions.items() if preds]
                missing_roles = [role for role in self._states if role not in ready_roles]
                if missing_roles:
                    if not self._waiting_logged:
                        rospy.loginfo(
                            "Waiting for path/odom data for: %s",
                            ", ".join(sorted(missing_roles)),
                        )
                        self._waiting_logged = True
                else:
                    if self._waiting_logged:
                        rospy.loginfo("All vehicles ready for predictive conflict checks")
                        self._waiting_logged = False

                stop_requests, min_distances = self._evaluate_conflicts(predictions)
                self._publish_stop_flags(stop_requests, min_distances)
            rate.sleep()

    def _path_callback(self, msg: Path, role: str) -> None:
        points = self._extract_points(msg.poses)
        with self._lock:
            state = self._states.get(role)
            if state is None:
                return
            if not points:
                state.path_points = []
                state.cumulative = []
                state.total_length = 0.0
                state.current_s = None
                return

            cumulative, total_length = self._compute_path_profile(points)
            state.path_points = points
            state.cumulative = cumulative
            state.total_length = total_length

            if state.position is not None:
                state.current_s = self._project_to_path(points, cumulative, state.position)

            rospy.loginfo(
                "%s: path received (%d poses, %.1fm)",
                role,
                len(points),
                total_length,
            )

    def _odom_callback(self, msg: Odometry, role: str) -> None:
        position = msg.pose.pose.position
        twist = msg.twist.twist.linear
        px = float(position.x)
        py = float(position.y)
        speed = math.sqrt(twist.x * twist.x + twist.y * twist.y + twist.z * twist.z)

        with self._lock:
            state = self._states.get(role)
            if state is None:
                return
            state.position = (px, py)
            state.speed = speed
            if state.path_points and state.cumulative:
                state.current_s = self._project_to_path(state.path_points, state.cumulative, state.position)

    def _predict_future(self, state: VehicleState) -> List[Tuple[float, float, float]]:
        if not state.path_points or not state.cumulative:
            return []
        if state.current_s is None:
            return []

        predictions: List[Tuple[float, float, float]] = []
        base_speed = max(0.0, state.speed)
        for step in self._prediction_steps:
            distance = state.current_s + base_speed * step
            sample = self._sample_path(state, distance)
            predictions.append((sample[0], sample[1], step))
        return predictions

    def _evaluate_conflicts(
        self, predictions: Dict[str, List[Tuple[float, float, float]]]
    ) -> Tuple[Dict[str, bool], Dict[str, Optional[float]]]:
        roles = sorted(predictions.keys())
        stop_requests: Dict[str, bool] = {role: False for role in roles}
        min_distances: Dict[str, Optional[float]] = {role: None for role in roles}

        for idx, role_a in enumerate(roles):
            preds_a = predictions.get(role_a, [])
            if not preds_a:
                continue
            for role_b in roles[idx + 1 :]:
                preds_b = predictions.get(role_b, [])
                if not preds_b:
                    continue
                steps = min(len(preds_a), len(preds_b))
                collision_time = None
                min_distance = None

                for step_idx in range(steps):
                    ax, ay, at = preds_a[step_idx]
                    bx, by, bt = preds_b[step_idx]
                    if abs(at - bt) > 1e-3:
                        continue
                    distance = math.hypot(ax - bx, ay - by)
                    if min_distance is None or distance < min_distance:
                        min_distance = distance
                    if distance < self.collision_threshold:
                        collision_time = at
                        break

                if min_distance is not None:
                    self._update_min_distance(min_distances, role_a, min_distance)
                    self._update_min_distance(min_distances, role_b, min_distance)

                if collision_time is None:
                    continue

                state_a = self._states[role_a]
                state_b = self._states[role_b]
                if state_a.vehicle_id == state_b.vehicle_id:
                    continue
                if state_a.vehicle_id < state_b.vehicle_id:
                    higher, lower = state_a, state_b
                else:
                    higher, lower = state_b, state_a

                rospy.logwarn(
                    "Potential collision between vehicle_%d and vehicle_%d at t=%.1fs (distance=%.2fm)",
                    higher.vehicle_id,
                    lower.vehicle_id,
                    collision_time,
                    min_distance if min_distance is not None else self.collision_threshold,
                )
                stop_requests[lower.role] = True

        return stop_requests, min_distances

    def _publish_stop_flags(
        self,
        stop_requests: Dict[str, bool],
        min_distances: Dict[str, Optional[float]],
    ) -> None:
        for role, state in self._states.items():
            requested = stop_requests.get(role, False)
            min_distance = min_distances.get(role)

            if requested:
                if not state.stop_active:
                    rospy.loginfo(
                        "Vehicle_%d stopped due to lower priority risk",
                        state.vehicle_id,
                    )
                state.stop_active = True
            else:
                if state.stop_active:
                    if min_distance is not None and min_distance > self.release_threshold:
                        rospy.loginfo(
                            "Vehicle_%d cleared to proceed (min future distance %.2fm)",
                            state.vehicle_id,
                            min_distance,
                        )
                        state.stop_active = False
                else:
                    state.stop_active = False

            publisher = self._stop_publishers.get(role)
            if publisher is None:
                continue

            if state.last_stop_published != state.stop_active:
                publisher.publish(Bool(data=state.stop_active))
                state.last_stop_published = state.stop_active
            elif state.stop_active:
                # Refresh the latched True signal periodically to avoid stale subs.
                publisher.publish(Bool(data=True))

    def _sample_path(self, state: VehicleState, target_distance: float) -> Tuple[float, float]:
        if not state.path_points:
            return (0.0, 0.0)
        if len(state.path_points) == 1:
            return state.path_points[0]

        if target_distance <= 0.0:
            return state.path_points[0]

        if target_distance >= state.total_length:
            return state.path_points[-1]

        cumulative = state.cumulative
        points = state.path_points

        for index in range(1, len(cumulative)):
            if cumulative[index] >= target_distance:
                prev_dist = cumulative[index - 1]
                segment_len = cumulative[index] - prev_dist
                if segment_len <= 1e-6:
                    return points[index]
                ratio = (target_distance - prev_dist) / segment_len
                x0, y0 = points[index - 1]
                x1, y1 = points[index]
                x = x0 + ratio * (x1 - x0)
                y = y0 + ratio * (y1 - y0)
                return (x, y)

        return state.path_points[-1]

    def _update_min_distance(
        self, registry: Dict[str, Optional[float]], role: str, candidate: float
    ) -> None:
        current = registry.get(role)
        if current is None or candidate < current:
            registry[role] = candidate

    def _compute_path_profile(
        self, points: List[Tuple[float, float]]
    ) -> Tuple[List[float], float]:
        cumulative = [0.0]
        total = 0.0
        for index in range(1, len(points)):
            x0, y0 = points[index - 1]
            x1, y1 = points[index]
            segment = math.hypot(x1 - x0, y1 - y0)
            total += segment
            cumulative.append(total)
        return cumulative, total

    def _extract_points(self, poses: List[PoseStamped]) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for pose in poses:
            position = pose.pose.position
            points.append((float(position.x), float(position.y)))
        return points

    def _project_to_path(
        self,
        points: List[Tuple[float, float]],
        cumulative: List[float],
        position: Tuple[float, float],
    ) -> float:
        if not points:
            return 0.0
        if len(points) == 1:
            return 0.0

        px, py = position
        best_distance = float("inf")
        best_s = 0.0

        for index in range(1, len(points)):
            x0, y0 = points[index - 1]
            x1, y1 = points[index]
            dx = x1 - x0
            dy = y1 - y0
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-9:
                continue
            u = ((px - x0) * dx + (py - y0) * dy) / seg_len_sq
            u = max(0.0, min(1.0, u))
            proj_x = x0 + u * dx
            proj_y = y0 + u * dy
            distance = math.hypot(px - proj_x, py - proj_y)
            if distance < best_distance:
                best_distance = distance
                best_s = cumulative[index - 1] + u * math.sqrt(seg_len_sq)

        return best_s

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _extract_vehicle_id(self, role: str) -> Optional[int]:
        digits = "".join(ch if ch.isdigit() else " " for ch in role).split()
        if not digits:
            return None
        try:
            return int(digits[-1])
        except ValueError:
            return None


def main() -> None:
    manager = PredictiveConflictManager()
    manager.spin()


if __name__ == "__main__":
    main()
