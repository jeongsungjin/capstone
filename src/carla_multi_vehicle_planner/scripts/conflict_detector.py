#!/usr/bin/env python3

"""Predictive conflict detector that issues short local detours when needed."""

import math
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path, Odometry


def quaternion_from_yaw(yaw: float) -> Quaternion:
    quat = Quaternion()
    quat.z = math.sin(yaw * 0.5)
    quat.w = math.cos(yaw * 0.5)
    return quat


class ConflictDetector:
    def __init__(self) -> None:
        rospy.init_node("conflict_detector", anonymous=True)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.horizon_s = float(rospy.get_param("~horizon_s", 6.0))
        self.dt = float(rospy.get_param("~dt", 0.1))
        self.R0 = float(rospy.get_param("~R0", 0.5))
        self.tau = float(rospy.get_param("~tau", 1.0))
        self.detour_length_m = float(rospy.get_param("~detour_length_m", 18.0))
        self.detour_lat_m = float(rospy.get_param("~detour_lat_m", 0.6))
        self.min_switch_cooldown_s = float(
            rospy.get_param("~min_switch_cooldown_s", 1.0)
        )
        self.assumed_speed_mps = float(rospy.get_param("~assumed_speed_mps", 6.0))
        self.detour_sampling_step = float(rospy.get_param("~detour_sampling_step", 0.5))
        self.detour_merge_distance = float(rospy.get_param("~detour_merge_distance", 6.0))
        self.lane_width = float(rospy.get_param("~lane_width", 3.2))
        self.s_safe_long = float(rospy.get_param("~s_safe_long", 6.0))
        self.d_safe_lat = float(rospy.get_param("~d_safe_lat", 3.0))
        self.lc_duration_s = float(rospy.get_param("~lc_duration_s", 2.0))
        self.lc_min_gap_s = float(rospy.get_param("~lc_min_gap_s", 0.8))
        self.use_lane_conflict = bool(rospy.get_param("~use_lane_conflict", False))
        # Detour stability controls
        self.detour_republish_min_s = float(rospy.get_param("~detour_republish_min_s", 0.8))
        self.detour_side_stick_s = float(rospy.get_param("~detour_side_stick_s", 1.0))

        if self.dt <= 0.0:
            self.dt = 0.1
        if self.detour_sampling_step <= 0.1:
            self.detour_sampling_step = 0.3
        # Allow stronger lateral detour up to half lane width
        self.detour_lat_m = max(0.1, min(self.lane_width * 0.5, self.detour_lat_m))

        self._lock = threading.RLock()
        self._tracks: Dict[str, Dict[str, object]] = {}
        self._publishers: Dict[str, rospy.Publisher] = {}
        self._active_detours: Dict[str, Dict[str, object]] = {}
        self._detour_state: Dict[str, Dict[str, object]] = {}

        for index in range(self.num_vehicles):
            role = self._role_name(index)
            self._tracks[role] = {
                "points": [],
                "s_profile": [],
                "path_length": 0.0,
                "frame_id": "map",
                "position": None,
                "speed": 0.0,
                "current_s": None,
                "frenet_origin": None,
                "frenet_heading": 0.0,
                "d_profile": [],
                "lane_change_plan": None,
            }
            self._publishers[role] = rospy.Publisher(
                f"/local_path_{role}", Path, queue_size=1
            )
            rospy.Subscriber(
                f"/global_path_{role}", Path, self._path_cb, callback_args=role
            )
            rospy.Subscriber(
                f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role
            )
            self._detour_state[role] = {
                "active": False,
                "last_switch": rospy.Time(0),
                "offset_sign": None,
            }

        timer_period = max(0.1, self.dt * 0.5)
        rospy.Timer(rospy.Duration(timer_period), self._timer_cb)

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _path_cb(self, msg: Path, role: str) -> None:
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        with self._lock:
            track = self._tracks[role]
            track["points"] = points
            s_profile, total = self._build_profile(points)
            track["s_profile"] = s_profile
            track["path_length"] = total
            if msg.header.frame_id:
                track["frame_id"] = msg.header.frame_id
            self._update_frenet_frame(track)
            track["d_profile"] = self._compute_path_lateral_profile(track)
            track["lane_change_plan"] = self._detect_lane_change(track)
            if track.get("position") is not None:
                projection = self._project_to_path(track, track["position"])
                if projection is not None:
                    track["current_s"] = projection

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        position = msg.pose.pose.position
        px = position.x
        py = position.y
        twist = msg.twist.twist.linear
        speed = math.sqrt(twist.x * twist.x + twist.y * twist.y + twist.z * twist.z)
        with self._lock:
            track = self._tracks[role]
            track["position"] = (px, py)
            track["speed"] = speed
            if track["points"]:
                projection = self._project_to_path(track, (px, py))
                if projection is not None:
                    track["current_s"] = projection

    def _timer_cb(self, _event) -> None:
        now = rospy.Time.now()
        with self._lock:
            role_list = list(self._tracks.keys())
            for idx_a, role_a in enumerate(role_list):
                for role_b in role_list[idx_a + 1 :]:
                    self._handle_pair(role_a, role_b, now)

            for role in list(self._active_detours.keys()):
                self._maybe_clear_detour(role, now)

    def _handle_pair(self, role_a: str, role_b: str, now: rospy.Time) -> None:
        track_a = self._tracks[role_a]
        track_b = self._tracks[role_b]
        predictions_a = self._predict_future(track_a)
        predictions_b = self._predict_future(track_b)
        if not predictions_a or not predictions_b:
            return

        steps = min(len(predictions_a), len(predictions_b))
        if steps == 0:
            return

        conflict_step = None
        pred_a_hit = None
        pred_b_hit = None
        conflict_type = None
        for step in range(steps):
            pred_a = predictions_a[step]
            pred_b = predictions_b[step]
            dist = math.hypot(
                pred_a["position"][0] - pred_b["position"][0],
                pred_a["position"][1] - pred_b["position"][1],
            )
            v_pair = max(pred_a["speed"], pred_b["speed"])
            r_safe = self.R0 + v_pair * self.tau
            # Lane conflict disabled by default; relies on Euclidean safety radius
            lane_conflict = False
            if self.use_lane_conflict:
                s_gap = abs(pred_a["s"] - pred_b["s"])
                d_gap = abs(pred_a["d"] - pred_b["d"])
                lane_conflict = s_gap < self.s_safe_long and d_gap < self.d_safe_lat
            if dist < r_safe or lane_conflict:
                conflict_step = step
                pred_a_hit = pred_a
                pred_b_hit = pred_b
                conflict_type = "lane" if lane_conflict else "distance"
                break

        if conflict_step is None or pred_a_hit is None or pred_b_hit is None:
            return

        if conflict_type == "lane":
            handled = self._resolve_lane_change(
                role_a,
                track_a,
                predictions_a,
                pred_a_hit,
                role_b,
                track_b,
                predictions_b,
                pred_b_hit,
                now,
            )
            if handled:
                return

        detour_roles: List[str] = []
        if role_a in self._active_detours:
            detour_roles.append(role_a)
        if role_b in self._active_detours:
            detour_roles.append(role_b)

        if not detour_roles:
            chosen = self._select_role_for_detour(role_a, role_b)
            if chosen is not None:
                detour_roles.append(chosen)

        for role in detour_roles:
            if role == role_a:
                info = {
                    "position": pred_a_hit["position"],
                    "other_position": pred_b_hit["position"],
                    "target_s": pred_a_hit["s"],
                }
            else:
                info = {
                    "position": pred_b_hit["position"],
                    "other_position": pred_a_hit["position"],
                    "target_s": pred_b_hit["s"],
                }
            self._ensure_detour(role, info, now)

    def _select_role_for_detour(self, role_a: str, role_b: str) -> Optional[str]:
        if role_a in self._active_detours and role_b in self._active_detours:
            return None
        if role_a in self._active_detours:
            return role_a
        if role_b in self._active_detours:
            return role_b

        pri_a = self._role_priority(role_a)
        pri_b = self._role_priority(role_b)
        if pri_a > pri_b:
            return role_a
        if pri_b > pri_a:
            return role_b

        track_a = self._tracks[role_a]
        track_b = self._tracks[role_b]
        s_a = track_a.get("current_s")
        s_b = track_b.get("current_s")
        if s_a is None and s_b is None:
            return role_b
        if s_a is None:
            return role_a
        if s_b is None:
            return role_b
        if s_a <= s_b:
            return role_b
        return role_a

    def _role_priority(self, role: str) -> int:
        try:
            suffix = role.rsplit("_", 1)[1]
            return int(suffix) - 1
        except (ValueError, IndexError):
            return 99

    def _resolve_lane_change(
        self,
        role_a: str,
        track_a: Dict[str, object],
        predictions_a: List[Dict[str, float]],
        pred_a_hit: Dict[str, float],
        role_b: str,
        track_b: Dict[str, object],
        predictions_b: List[Dict[str, float]],
        pred_b_hit: Dict[str, float],
        now: rospy.Time,
    ) -> bool:
        candidates: List[Tuple[str, Dict[str, object], List[Dict[str, float]]]] = []
        state_a = self._lane_change_state(track_a, pred_a_hit["s"])
        state_b = self._lane_change_state(track_b, pred_b_hit["s"])
        if state_a in ("before", "during") and track_a.get("lane_change_plan"):
            candidates.append((role_a, track_a, predictions_a))
        if state_b in ("before", "during") and track_b.get("lane_change_plan"):
            candidates.append((role_b, track_b, predictions_b))
        if not candidates:
            return False

        if len(candidates) == 2:
            lc_role, lc_track, lc_predictions = self._choose_lane_change_candidate(
                candidates
            )
            other_predictions = predictions_b if lc_role == role_a else predictions_a
        else:
            lc_role, lc_track, lc_predictions = candidates[0]
            if lc_role == role_a:
                other_predictions = predictions_b
            else:
                other_predictions = predictions_a

        return self._schedule_lane_change(
            lc_role,
            lc_track,
            lc_predictions,
            other_predictions,
            now,
        )

    def _choose_lane_change_candidate(
        self, candidates: List[Tuple[str, Dict[str, object], List[Dict[str, float]]]]
    ) -> Tuple[str, Dict[str, object], List[Dict[str, float]]]:
        chosen = candidates[0]
        chosen_eta = self._lane_change_eta(chosen[1])
        for candidate in candidates[1:]:
            eta = self._lane_change_eta(candidate[1])
            if eta > chosen_eta:
                chosen = candidate
                chosen_eta = eta
        return chosen

    def _lane_change_eta(self, track: Dict[str, object]) -> float:
        plan = track.get("lane_change_plan")
        if not plan:
            return 0.0
        start_s = float(plan.get("scheduled_start_s", plan.get("start_s", 0.0)))
        return self._estimate_time_to_s(track, start_s)

    def _schedule_lane_change(
        self,
        role: str,
        track: Dict[str, object],
        predictions_self: List[Dict[str, float]],
        predictions_other: List[Dict[str, float]],
        now: rospy.Time,
    ) -> bool:
        plan = track.get("lane_change_plan")
        if not plan:
            return False

        duration = max(self.lc_duration_s, 0.5)
        plan.setdefault("duration_s", duration)
        start_s_nominal = float(plan.get("scheduled_start_s", plan.get("start_s", 0.0)))
        t_nominal = self._estimate_time_to_s(track, start_s_nominal)
        search_limit = max(0.0, self.horizon_s - duration)
        delta_step = max(self.lc_min_gap_s, self.dt)

        best_delta = None
        best_duration = duration
        delta = 0.0
        while delta <= search_limit:
            if self._lane_change_interval_safe(
                track,
                plan,
                predictions_self,
                predictions_other,
                t_nominal,
                delta,
                duration,
            ):
                best_delta = delta
                break
            delta += delta_step

        if best_delta is None:
            extended_duration = min(duration * 1.5, max(duration, self.horizon_s - t_nominal))
            if extended_duration > duration:
                if self._lane_change_interval_safe(
                    track,
                    plan,
                    predictions_self,
                    predictions_other,
                    t_nominal,
                    0.0,
                    extended_duration,
                ):
                    best_delta = 0.0
                    best_duration = extended_duration

        if best_delta is None:
            return False

        start_time = t_nominal + best_delta
        end_time = start_time + best_duration
        start_state = self._prediction_sample(predictions_self, start_time)
        end_state = self._prediction_sample(predictions_self, end_time)
        if start_state is None or end_state is None:
            return False

        plan["scheduled_delay"] = best_delta
        plan["scheduled_start_time"] = now + rospy.Duration.from_sec(start_time)
        plan["scheduled_start_s"] = start_state["s"]
        plan["scheduled_end_s"] = end_state["s"]
        plan["duration_s"] = best_duration

        path_msg = self._build_lane_change_path(track, plan)
        if path_msg is None:
            return False

        self._publishers[role].publish(path_msg)
        state = self._detour_state.get(role)
        if state is not None and not state["active"]:
            state["active"] = True
            state["last_switch"] = now
        self._active_detours[role] = {
            "last_conflict": now,
            "last_publish": now,
            "target_s": plan.get("scheduled_end_s", plan.get("end_s")),
        }
        rospy.loginfo(
            "%s: scheduled lane change with delay %.2fs duration %.2fs",
            role,
            best_delta,
            best_duration,
        )
        return True

    def _lane_change_interval_safe(
        self,
        track: Dict[str, object],
        plan: Dict[str, float],
        predictions_self: List[Dict[str, float]],
        predictions_other: List[Dict[str, float]],
        t_nominal: float,
        delta: float,
        duration: float,
    ) -> bool:
        start_time = t_nominal + delta
        end_time = start_time + duration
        base_d = float(plan.get("base_d", 0.0))
        target_d = float(plan.get("target_d", base_d))
        if not predictions_self:
            return True
        for pred in predictions_self:
            t = pred["time"]
            if t > self.horizon_s + 1e-3:
                break
            s_self = pred["s"]
            if t < start_time:
                d_self = base_d
            elif t <= end_time:
                progress = (t - start_time) / max(duration, 1e-3)
                d_self = base_d + (target_d - base_d) * self._smoothstep(progress)
            else:
                d_self = target_d
            other_pred = self._prediction_sample(predictions_other, t)
            if other_pred is None:
                continue
            s_gap = abs(s_self - other_pred["s"])
            d_gap = abs(d_self - other_pred["d"])
            if s_gap < self.s_safe_long and d_gap < self.d_safe_lat:
                return False
        return True

    def _build_lane_change_path(
        self, track: Dict[str, object], plan: Dict[str, float]
    ) -> Optional[Path]:
        origin = track.get("frenet_origin")
        heading = track.get("frenet_heading")
        if origin is None:
            return None
        current_s = float(track.get("current_s") or 0.0)
        path_length = float(track.get("path_length", 0.0))
        if path_length <= 0.0:
            return None

        start_s = float(plan.get("scheduled_start_s", plan.get("start_s", current_s)))
        end_s = float(plan.get("scheduled_end_s", plan.get("end_s", start_s)))
        base_d = float(plan.get("base_d", 0.0))
        target_d = float(plan.get("target_d", base_d))

        segment_start = max(current_s, min(start_s, end_s) - 2.0)
        segment_end = min(path_length, max(end_s, start_s) + self.detour_merge_distance)
        if segment_end - segment_start < 3.0:
            segment_end = min(path_length, segment_start + 3.0)
            if segment_end - segment_start < 1.0:
                return None

        samples: List[float] = []
        s = segment_start
        step = max(0.5, self.detour_sampling_step)
        while s < segment_end:
            samples.append(s)
            s += step
        samples.append(segment_end)

        sin_h = math.sin(heading)
        cos_h = math.cos(heading)
        ox, oy = origin

        points: List[Tuple[float, float]] = []
        span = max(end_s - start_s, 1e-3)
        for sample_s in samples:
            if sample_s <= start_s:
                offset = base_d
            elif sample_s >= end_s:
                offset = target_d
            else:
                progress = (sample_s - start_s) / span
                offset = base_d + (target_d - base_d) * self._smoothstep(progress)
            x = ox + sample_s * cos_h - offset * sin_h
            y = oy + sample_s * sin_h + offset * cos_h
            points.append((x, y))

        path_msg = Path()
        path_msg.header.frame_id = track.get("frame_id", "map")
        path_msg.header.stamp = rospy.Time.now()

        for idx, (px, py) in enumerate(points):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = px
            pose.pose.position.y = py
            pose.pose.position.z = 0.0
            if idx < len(points) - 1:
                nx = points[idx + 1][0] - px
                ny = points[idx + 1][1] - py
            else:
                if idx > 0:
                    nx = px - points[idx - 1][0]
                    ny = py - points[idx - 1][1]
                else:
                    nx, ny = math.cos(heading), math.sin(heading)
            if abs(nx) < 1e-6 and abs(ny) < 1e-6:
                nx, ny = math.cos(heading), math.sin(heading)
            yaw = math.atan2(ny, nx)
            pose.pose.orientation = quaternion_from_yaw(yaw)
            path_msg.poses.append(pose)

        return path_msg

    def _ensure_detour(
        self, role: str, conflict_info: Dict[str, object], now: rospy.Time
    ) -> None:
        record = self._active_detours.get(role)
        last_publish = None
        if record is not None:
            record["last_conflict"] = now
            last_publish = record.get("last_publish")
            previous_target = record.get("target_s")
            current_target = conflict_info.get("target_s")
            if (
                previous_target is not None
                and current_target is not None
                and abs(float(current_target) - float(previous_target)) < 1.0
            ):
                if last_publish is not None and (now - last_publish).to_sec() < self.detour_republish_min_s:
                    return

        # Determine desired side and apply stickiness to avoid rapid flipping
        track = self._tracks[role]
        desired_sign = self._determine_offset(track, conflict_info)
        state = self._detour_state.get(role)
        effective_sign = desired_sign
        if state is not None and state.get("offset_sign") is not None:
            if (now - state.get("last_switch", rospy.Time(0))).to_sec() < self.detour_side_stick_s:
                effective_sign = float(state["offset_sign"])  # stick to previous side briefly
            elif desired_sign != state["offset_sign"]:
                state["offset_sign"] = desired_sign
                state["last_switch"] = now
        elif state is not None:
            state["offset_sign"] = desired_sign
            state["last_switch"] = now

        path_msg = self._build_detour(role, conflict_info, effective_sign)
        if path_msg is None:
            return

        state = self._detour_state.get(role)
        if state is not None and not state["active"]:
            state["active"] = True
            state["last_switch"] = now

        self._publishers[role].publish(path_msg)
        self._active_detours[role] = {
            "last_conflict": now,
            "last_publish": now,
            "target_s": conflict_info.get("target_s"),
        }

    def _maybe_clear_detour(self, role: str, now: rospy.Time) -> None:
        state = self._detour_state.get(role)
        record = self._active_detours.get(role)
        if state is None or record is None:
            return
        if not state["active"]:
            self._active_detours.pop(role, None)
            return

        since_conflict = (now - record["last_conflict"]).to_sec()
        if since_conflict <= self.min_switch_cooldown_s:
            return
        if (now - state["last_switch"]).to_sec() < self.min_switch_cooldown_s:
            return
        self._clear_detour(role, now)

    def _clear_detour(self, role: str, now: rospy.Time) -> None:
        rospy.loginfo("%s: clearing local detour", role)
        self._active_detours.pop(role, None)
        state = self._detour_state.get(role)
        if state is not None:
            state["active"] = False
            state["last_switch"] = now
        empty = Path()
        empty.header.stamp = now
        empty.header.frame_id = self._tracks[role].get("frame_id", "map")
        self._publishers[role].publish(empty)

    def _build_detour(self, role: str, info: Dict[str, object], forced_sign: Optional[float] = None) -> Optional[Path]:
        track = self._tracks[role]
        if not track["points"] or track["path_length"] <= 0.0:
            return None

        current_s = track.get("current_s")
        if current_s is None:
            return None

        target_s = float(info.get("target_s", current_s))
        start_s = max(current_s, target_s - self.detour_length_m * 0.3)
        end_s = min(track["path_length"], start_s + self.detour_length_m)
        if end_s - start_s < 4.0:
            end_s = min(track["path_length"], start_s + 4.0)
            if end_s - start_s < 1.0:
                return None

        samples = self._sample_segment(track, start_s, end_s, self.detour_sampling_step)
        if len(samples) < 2:
            return None

        offset_sign = forced_sign if forced_sign is not None else self._determine_offset(track, info)
        detour_points: List[Tuple[float, float]] = []
        for idx, (sx, sy, sample_s) in enumerate(samples):
            progress = 0.0
            if end_s > start_s:
                progress = (sample_s - start_s) / (end_s - start_s)
            offset_mag = self.detour_lat_m * self._offset_profile(progress)
            tangent = self._path_direction(track, sample_s)
            normal = (-tangent[1], tangent[0])
            detour_points.append(
                (sx + normal[0] * offset_mag * offset_sign, sy + normal[1] * offset_mag * offset_sign)
            )

        merge_end = min(track["path_length"], end_s + self.detour_merge_distance)
        if merge_end > end_s + 1e-3:
            merge_samples = self._sample_segment(track, end_s, merge_end, self.detour_sampling_step)
            for idx, (mx, my, _ms) in enumerate(merge_samples):
                if idx == 0:
                    continue
                detour_points.append((mx, my))

        path_msg = Path()
        path_msg.header.frame_id = track.get("frame_id", "map")
        path_msg.header.stamp = rospy.Time.now()

        for idx, (px, py) in enumerate(detour_points):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = px
            pose.pose.position.y = py
            pose.pose.position.z = 0.0
            if idx < len(detour_points) - 1:
                nx = detour_points[idx + 1][0] - px
                ny = detour_points[idx + 1][1] - py
            else:
                if idx > 0:
                    nx = px - detour_points[idx - 1][0]
                    ny = py - detour_points[idx - 1][1]
                else:
                    nx, ny = 1.0, 0.0
            if abs(nx) < 1e-6 and abs(ny) < 1e-6:
                nx, ny = 1.0, 0.0
            yaw = math.atan2(ny, nx)
            pose.pose.orientation = quaternion_from_yaw(yaw)
            path_msg.poses.append(pose)

        rospy.loginfo(
            "%s: issuing local detour (%.1fm) offset %.2f",
            role,
            end_s - start_s,
            self.detour_lat_m * offset_sign,
        )
        return path_msg

    def _determine_offset(self, track: Dict[str, object], info: Dict[str, object]) -> float:
        target_s = float(info.get("target_s", track.get("current_s") or 0.0))
        tangent = self._path_direction(track, target_s)
        other_pos = info.get("other_position")
        self_pos = info.get("position")
        if other_pos is None or self_pos is None:
            return 1.0
        vector_to_other = (
            float(other_pos[0]) - float(self_pos[0]),
            float(other_pos[1]) - float(self_pos[1]),
        )
        cross = tangent[0] * vector_to_other[1] - tangent[1] * vector_to_other[0]
        if abs(cross) < 1e-6:
            return 1.0
        return -1.0 if cross > 0.0 else 1.0

    def _smoothstep(self, value: float) -> float:
        x = max(0.0, min(1.0, value))
        return x * x * x * (x * (6.0 * x - 15.0) + 10.0)

    def _offset_profile(self, u: float) -> float:
        u = max(0.0, min(1.0, u))
        ramp = 1.0 / 3.0

        if u <= ramp:
            return self._smoothstep(u / ramp)
        if u >= 1.0 - ramp:
            return self._smoothstep((1.0 - u) / ramp)
        return 1.0

    def _update_frenet_frame(self, track: Dict[str, object]) -> None:
        points: List[Tuple[float, float]] = track.get("points", [])
        if len(points) >= 2:
            x0, y0 = points[0]
            x1, y1 = points[1]
            heading = math.atan2(y1 - y0, x1 - x0)
            track["frenet_origin"] = (x0, y0)
            track["frenet_heading"] = heading
        elif points:
            track["frenet_origin"] = points[0]
            track["frenet_heading"] = 0.0
        else:
            track["frenet_origin"] = None
            track["frenet_heading"] = 0.0

    def _compute_path_lateral_profile(self, track: Dict[str, object]) -> List[float]:
        origin = track.get("frenet_origin")
        heading = float(track.get("frenet_heading", 0.0))
        points: List[Tuple[float, float]] = track.get("points", [])
        if origin is None or not points:
            return []
        sin_h = math.sin(heading)
        cos_h = math.cos(heading)
        d_profile: List[float] = []
        ox, oy = origin
        for px, py in points:
            dx = px - ox
            dy = py - oy
            d_val = -dx * sin_h + dy * cos_h
            d_profile.append(d_val)
        return d_profile

    def _detect_lane_change(self, track: Dict[str, object]) -> Optional[Dict[str, float]]:
        d_profile: List[float] = track.get("d_profile", [])
        s_profile: List[float] = track.get("s_profile", [])
        if len(d_profile) < 10 or len(d_profile) != len(s_profile):
            return None
        window = min(10, len(d_profile))
        base_samples = d_profile[:window]
        target_samples = d_profile[-window:]
        base_d = sum(base_samples) / len(base_samples)
        target_d = sum(target_samples) / len(target_samples)
        if abs(target_d - base_d) < self.lane_width * 0.6:
            return None
        threshold = self.lane_width * 0.25
        start_idx = None
        end_idx = None
        for idx, d_val in enumerate(d_profile):
            if start_idx is None and abs(d_val - base_d) > threshold:
                start_idx = idx
            if abs(d_val - target_d) < threshold:
                end_idx = idx
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            return None
        start_s = s_profile[start_idx]
        end_s = s_profile[end_idx]
        return {
            "base_d": base_d,
            "target_d": target_d,
            "start_s": start_s,
            "end_s": end_s,
            "scheduled_delay": 0.0,
            "scheduled_start_time": None,
            "scheduled_start_s": start_s,
            "scheduled_end_s": end_s,
        }

    def _predict_future(self, track: Dict[str, object]) -> List[Dict[str, object]]:
        points: List[Tuple[float, float]] = track.get("points", [])
        if len(points) < 2:
            return []

        current_s = track.get("current_s")
        if current_s is None:
            return []

        path_length = float(track.get("path_length", 0.0))
        if path_length <= 0.0:
            return []

        raw_speed = max(0.0, float(track.get("speed", 0.0)))
        speed = raw_speed if raw_speed >= 0.5 else max(self.assumed_speed_mps, raw_speed)
        steps = max(1, int(self.horizon_s / self.dt))
        results: List[Dict[str, object]] = []
        last_s = None
        for step in range(1, steps + 1):
            future_time = step * self.dt
            target_s = current_s + speed * future_time
            if target_s > path_length:
                target_s = path_length
            position = self._position_along(track, target_s)
            if position is None:
                break
            results.append(
                {
                    "time": future_time,
                    "position": position,
                    "s": target_s,
                    "speed": speed,
                    "d": self._lateral_offset_along(track, target_s),
                }
            )
            if last_s is not None and abs(target_s - last_s) < 1e-3:
                break
            last_s = target_s
            if target_s >= path_length - 1e-3:
                break
        return results

    def _build_profile(self, points: Sequence[Tuple[float, float]]) -> Tuple[List[float], float]:
        if len(points) < 2:
            return [0.0 for _ in points], 0.0
        s_profile = [0.0]
        total = 0.0
        for idx in range(1, len(points)):
            dx = points[idx][0] - points[idx - 1][0]
            dy = points[idx][1] - points[idx - 1][1]
            step = math.hypot(dx, dy)
            total += step
            s_profile.append(total)
        return s_profile, total

    def _project_to_path(
        self, track: Dict[str, object], position: Tuple[float, float]
    ) -> Optional[float]:
        points = track.get("points", [])
        s_profile = track.get("s_profile", [])
        if len(points) < 2 or len(points) != len(s_profile):
            return None
        px, py = position
        best_dist = float("inf")
        best_index = None
        best_t = 0.0
        for idx in range(len(points) - 1):
            x1, y1 = points[idx]
            x2, y2 = points[idx + 1]
            seg_dx = x2 - x1
            seg_dy = y2 - y1
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len_sq < 1e-8:
                continue
            t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + seg_dx * t
            proj_y = y1 + seg_dy * t
            dist = (proj_x - px) * (proj_x - px) + (proj_y - py) * (proj_y - py)
            if dist < best_dist:
                best_dist = dist
                best_index = idx
                best_t = t
        if best_index is None:
            return None
        seg_length = s_profile[best_index + 1] - s_profile[best_index]
        return s_profile[best_index] + seg_length * best_t

    def _sample_segment(
        self, track: Dict[str, object], start_s: float, end_s: float, step: float
    ) -> List[Tuple[float, float, float]]:
        samples: List[Tuple[float, float, float]] = []
        s = start_s
        while s < end_s:
            position = self._position_along(track, s)
            if position is None:
                break
            samples.append((position[0], position[1], s))
            s += step
        end_position = self._position_along(track, end_s)
        if end_position is not None:
            samples.append((end_position[0], end_position[1], end_s))
        return samples

    def _position_along(
        self, track: Dict[str, object], target_s: float
    ) -> Optional[Tuple[float, float]]:
        points = track.get("points", [])
        s_profile = track.get("s_profile", [])
        if len(points) < 2 or len(points) != len(s_profile):
            return None
        if target_s <= 0.0:
            return points[0]
        path_length = s_profile[-1]
        if target_s >= path_length:
            return points[-1]
        for idx in range(len(points) - 1):
            s0 = s_profile[idx]
            s1 = s_profile[idx + 1]
            if target_s < s0:
                return points[idx]
            if target_s <= s1:
                ratio = 0.0
                if s1 > s0:
                    ratio = (target_s - s0) / (s1 - s0)
                x = points[idx][0] + (points[idx + 1][0] - points[idx][0]) * ratio
                y = points[idx][1] + (points[idx + 1][1] - points[idx][1]) * ratio
                return x, y
        return points[-1]

    def _path_direction(
        self, track: Dict[str, object], target_s: float
    ) -> Tuple[float, float]:
        delta = max(0.5, self.detour_sampling_step)
        prev_pos = self._position_along(track, max(0.0, target_s - delta))
        next_pos = self._position_along(track, target_s + delta)
        if prev_pos is None or next_pos is None:
            return (1.0, 0.0)
        dx = next_pos[0] - prev_pos[0]
        dy = next_pos[1] - prev_pos[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return (1.0, 0.0)
        return (dx / norm, dy / norm)

    def _lateral_offset_along(
        self, track: Dict[str, object], target_s: float
    ) -> float:
        d_profile: List[float] = track.get("d_profile", [])
        s_profile: List[float] = track.get("s_profile", [])
        if len(d_profile) < 2 or len(d_profile) != len(s_profile):
            return 0.0
        if target_s <= s_profile[0]:
            return d_profile[0]
        if target_s >= s_profile[-1]:
            return d_profile[-1]
        for idx in range(len(s_profile) - 1):
            s0 = s_profile[idx]
            s1 = s_profile[idx + 1]
            if target_s < s0:
                return d_profile[idx]
            if target_s <= s1:
                ratio = 0.0
                if s1 > s0:
                    ratio = (target_s - s0) / (s1 - s0)
                return d_profile[idx] + ratio * (d_profile[idx + 1] - d_profile[idx])
        return d_profile[-1]

    def _lane_change_state(self, track: Dict[str, object], target_s: float) -> str:
        plan = track.get("lane_change_plan")
        if not plan:
            return "none"
        start_s = float(plan.get("scheduled_start_s", plan.get("start_s", 0.0)))
        end_s = float(plan.get("scheduled_end_s", plan.get("end_s", start_s)))
        margin = 1.0
        if target_s < start_s - margin:
            return "before"
        if target_s <= end_s + margin:
            return "during"
        return "after"

    def _estimate_time_to_s(self, track: Dict[str, object], target_s: float) -> float:
        current_s = float(track.get("current_s") or 0.0)
        remaining = max(0.0, target_s - current_s)
        speed = max(float(track.get("speed", 0.0)), 0.1)
        if speed < 0.5:
            speed = self.assumed_speed_mps
        return remaining / max(speed, 0.1)

    def _prediction_sample(
        self, predictions: List[Dict[str, float]], time_point: float
    ) -> Optional[Dict[str, float]]:
        if not predictions:
            return None
        if time_point <= predictions[0]["time"]:
            return predictions[0]
        for idx in range(len(predictions) - 1):
            a = predictions[idx]
            b = predictions[idx + 1]
            if time_point <= b["time"]:
                ratio = 0.0
                dt = b["time"] - a["time"]
                if dt > 1e-3:
                    ratio = (time_point - a["time"]) / dt
                return {
                    "time": time_point,
                    "s": a["s"] + ratio * (b["s"] - a["s"]),
                    "d": a["d"] + ratio * (b["d"] - a["d"]),
                    "speed": a.get("speed", 0.0),
                }
        return predictions[-1]


def main() -> None:
    ConflictDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
