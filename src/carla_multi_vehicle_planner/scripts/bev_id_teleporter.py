#!/usr/bin/env python3

import math
import threading
from typing import Dict, List, Optional, Tuple

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

# Ensure CARLA API is on sys.path
try:
	# Prefer centralized CARLA path setup if available
	from setup_carla_path import *  # noqa: F401,F403
except Exception:
	# Fallback: rely on environment
	pass

try:
	import carla  # type: ignore
except Exception as exc:
	carla = None  # type: ignore
	rospy.logfatal("Failed to import CARLA Python API: %s", exc)

from capstone_msgs.msg import BEVInfo


def deg(rad: float) -> float:
	return rad * 180.0 / math.pi


def normalize_angle(angle: float) -> float:
	while angle > math.pi:
		angle -= 2.0 * math.pi
	while angle < -math.pi:
		angle += 2.0 * math.pi
	return angle


def quaternion_to_yaw(q) -> float:
	x = getattr(q, "x", 0.0)
	y = getattr(q, "y", 0.0)
	z = getattr(q, "z", 0.0)
	w = getattr(q, "w", 1.0)
	return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class BevIdTeleporter:
	def __init__(self):
		rospy.init_node("bev_id_teleporter", anonymous=True)
		if carla is None:
			raise RuntimeError("CARLA Python API unavailable")

		# Parameters
		self.host: str = rospy.get_param("~carla_host", "localhost")
		self.port: int = int(rospy.get_param("~carla_port", 2000))
		self.max_vehicle_count: int = int(rospy.get_param("~max_vehicle_count", 1))
		self.max_vehicle_count = max(1, min(6, self.max_vehicle_count))

		self.topic_name: str = rospy.get_param("~topic", "/bev_info")
		self.enable_matching: bool = bool(rospy.get_param("~enable_matching", True))
		# Matching weights and gates
		self.w_pos: float = float(rospy.get_param("~w_pos", 1.0))
		self.w_yaw: float = float(rospy.get_param("~w_yaw", 0.3))
		self.dist_gate_m: float = float(rospy.get_param("~dist_gate_m", 10.0))
		self.yaw_gate_deg: float = float(rospy.get_param("~yaw_gate_deg", 60.0))
		self.hold_sec: float = float(rospy.get_param("~hold_sec", 0.5))
		self.ttl_sec: float = float(rospy.get_param("~ttl_sec", 3.0))
		# Optional SciPy usage
		self.use_scipy: bool = bool(rospy.get_param("~use_scipy", True))
		try:
			if self.use_scipy:
				from scipy.optimize import linear_sum_assignment as _lsa  # type: ignore
				self._lsa = _lsa
			else:
				self._lsa = None
		except Exception:
			self._lsa = None

		# Role name handling
		role_names_csv: str = str(rospy.get_param("~role_names", "")).strip()
		if role_names_csv:
			self.role_names: List[str] = [s.strip() for s in role_names_csv.split(",") if s.strip()]
		else:
			self.role_names = [f"ego_vehicle_{i}" for i in range(1, self.max_vehicle_count + 1)]

		# Yaw interpretation
		self.yaw_in_degrees: bool = bool(rospy.get_param("~yaw_in_degrees", False))
		self.enable_yaw_flip: bool = bool(rospy.get_param("~enable_yaw_flip", True))
		self.yaw_flip_threshold_deg: float = float(rospy.get_param("~yaw_flip_threshold_deg", 150.0))

		# Height / waypoint snapping
		self.snap_to_waypoint_height: bool = bool(rospy.get_param("~snap_to_waypoint_height", True))
		self.default_z: float = float(rospy.get_param("~default_z", 0.5))
		self.z_offset: float = float(rospy.get_param("~z_offset", 0.0))
		# Teleport safety
		self.max_teleport_distance_m: float = float(rospy.get_param("~max_teleport_distance_m", 20.0))
		self.teleport_yaw_gate_deg: float = float(rospy.get_param("~teleport_yaw_gate_deg", 120.0))
		self.teleport_stability_warmup_sec: float = float(rospy.get_param("~teleport_stability_warmup_sec", 0.2))

		# Connect to CARLA
		self.client = carla.Client(self.host, self.port)
		self.client.set_timeout(5.0)
		self.world = self.client.get_world()
		self.carla_map = self.world.get_map()

		# State
		self._lock = threading.Lock()
		self.role_to_actor: Dict[str, carla.Actor] = {}
		self.id_to_role: Dict[int, str] = {}
		self.role_to_id: Dict[str, int] = {}
		self.role_state: Dict[str, Dict[str, object]] = {role: {"pos": None, "yaw": None, "stamp": None} for role in []}
		self.role_last_switch: Dict[str, float] = {}
		self.id_last_seen: Dict[int, float] = {}
		self.role_tracking_ok: Dict[str, bool] = {}
		self.tracking_publishers: Dict[str, rospy.Publisher] = {}

		self._refresh_role_to_actor()

		# ROS I/O
		self.role_state = {role: {"pos": None, "yaw": None, "stamp": None} for role in self.role_names}
		for role in self.role_names:
			rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role, queue_size=10)
			topic = f"/carla/{role}/bev_tracking_ok"
			self.tracking_publishers[role] = rospy.Publisher(topic, Bool, queue_size=1, latch=True)
			self.role_tracking_ok[role] = False
			self._set_tracking_status(role, False, force=True)
		self.sub = rospy.Subscriber(self.topic_name, BEVInfo, self._bev_cb, queue_size=1)
		rospy.loginfo(
			"bev_id_teleporter ready: topic=%s, roles=%s (max=%d)",
			self.topic_name,
			",".join(self.role_names),
			self.max_vehicle_count,
		)

	def _refresh_role_to_actor(self) -> None:
		actors = self.world.get_actors().filter("vehicle.*")
		role_to_actor: Dict[str, carla.Actor] = {}
		for actor in actors:
			try:
				role = actor.attributes.get("role_name", "")
			except Exception:
				continue
			if role in self.role_names:
				role_to_actor[role] = actor
		with self._lock:
			self.role_to_actor = role_to_actor
			# Initialize role_state dictionary for any newly found roles
			for role in role_to_actor.keys():
				if role not in self.role_state:
					self.role_state[role] = {"pos": None, "yaw": None, "stamp": None}
			# Initialize switch timestamps if missing
			for role in role_to_actor.keys():
				if role not in self.role_last_switch:
					self.role_last_switch[role] = float(rospy.Time.now().to_sec())
			for role in role_to_actor.keys():
				if role not in self.tracking_publishers:
					topic = f"/carla/{role}/bev_tracking_ok"
					self.tracking_publishers[role] = rospy.Publisher(topic, Bool, queue_size=1, latch=True)
					self.role_tracking_ok[role] = False
					self._set_tracking_status(role, False, force=True)

	def _resolve_actor_for_role(self, role: str) -> Optional[carla.Actor]:
		with self._lock:
			actor = self.role_to_actor.get(role)
		if actor is not None and actor.is_alive:
			return actor
		# Try to refresh cache once
		self._refresh_role_to_actor()
		with self._lock:
			return self.role_to_actor.get(role)

	def _odom_cb(self, msg: Odometry, role: str) -> None:
		pos = msg.pose.pose.position
		yaw = quaternion_to_yaw(msg.pose.pose.orientation)
		with self._lock:
			state = self.role_state.get(role)
			if state is None:
				state = {"pos": None, "yaw": None, "stamp": None}
				self.role_state[role] = state
			state["pos"] = (float(pos.x), float(pos.y))
			state["yaw"] = float(yaw)
			state["stamp"] = float(msg.header.stamp.to_sec()) if msg.header and msg.header.stamp else float(rospy.Time.now().to_sec())

	def _get_role_prediction(self, role: str) -> Optional[Tuple[float, float, float]]:
		with self._lock:
			state = self.role_state.get(role)
		if state is not None and state.get("pos") is not None and state.get("yaw") is not None:
			x, y = state["pos"]  # type: ignore
			yaw = state["yaw"]  # type: ignore
			return float(x), float(y), float(yaw)
		# fallback to actor transform if odom not yet available
		actor = self._resolve_actor_for_role(role)
		if actor is None:
			return None
		try:
			tf = actor.get_transform()
			yaw = math.radians(tf.rotation.yaw)
			return float(tf.location.x), float(tf.location.y), float(yaw)
		except Exception:
			return None

	def _compute_cost_matrix(self, roles: List[str], ids: List[int], xs: List[float], ys: List[float], yaws: List[float]) -> List[List[float]]:
		big = 1e6
		costs: List[List[float]] = []
		for role in roles:
			pred = self._get_role_prediction(role)
			row: List[float] = []
			for idx, vid in enumerate(ids):
				if pred is None:
					row.append(big)
					continue
				rx, ry, ryaw = pred
				dx = xs[idx] - rx
				dy = ys[idx] - ry
				dist = math.hypot(dx, dy)
				yaw_det = yaws[idx]
				ang = abs(normalize_angle(yaw_det - ryaw))
				ang_deg = abs(math.degrees(ang))
				if (self.dist_gate_m > 0.0 and dist > self.dist_gate_m) or (self.yaw_gate_deg > 0.0 and ang_deg > self.yaw_gate_deg):
					row.append(big)
				else:
					row.append(self.w_pos * dist + self.w_yaw * ang)
			costs.append(row)
		return costs

	def _hungarian_assign(self, costs: List[List[float]]) -> List[Tuple[int, int]]:
		# returns list of (row_index, col_index)
		if not costs or not costs[0]:
			return []
		num_rows = len(costs)
		num_cols = len(costs[0])
		if self._lsa is not None:
			import numpy as np  # type: ignore
			cmat = np.array(costs, dtype=float)
			row_ind, col_ind = self._lsa(cmat)
			return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if c < num_cols and r < num_rows and np.isfinite(cmat[r, c])]
		# Greedy fallback
		assigned_cols = set()
		assignment: List[Tuple[int, int]] = []
		for r in range(num_rows):
			best_c = None
			best_v = float("inf")
			for c in range(num_cols):
				v = costs[r][c]
				if c in assigned_cols:
					continue
				if v < best_v:
					best_v = v
					best_c = c
			if best_c is not None and best_v < 1e6:
				assigned_cols.add(best_c)
				assignment.append((r, best_c))
		return assignment

	def _update_mappings_with_assignment(self, roles: List[str], ids: List[int], assignment: List[Tuple[int, int]], xs: List[float], ys: List[float], yaws: List[float]) -> None:
		now = float(rospy.Time.now().to_sec())
		with self._lock:
			# TTL cleanup for roles whose current id missing for too long
			for role, current_id in list(self.role_to_id.items()):
				last_seen = self.id_last_seen.get(current_id, now)
				if (now - last_seen) >= max(0.0, self.ttl_sec):
					self.role_to_id.pop(role, None)
					self.id_to_role.pop(current_id, None)
			# Mark ids seen this frame
			for vid in ids:
				self.id_last_seen[vid] = now
			# Build proposed role->id from assignment
			proposed: Dict[str, int] = {}
			for r_index, c_index in assignment:
				if 0 <= r_index < len(roles) and 0 <= c_index < len(ids):
					proposed[roles[r_index]] = ids[c_index]
			# Apply with hold hysteresis
			for role in roles:
				new_id = proposed.get(role)
				old_id = self.role_to_id.get(role)
				if new_id is None:
					continue
				if old_id is None:
					# Free role: take new assignment
					self.role_to_id[role] = new_id
					self.id_to_role[new_id] = role
					self.role_last_switch[role] = now
					continue
				if old_id == new_id:
					# no change
					continue
				# hold condition
				last_sw = self.role_last_switch.get(role, 0.0)
				if (now - last_sw) < max(0.0, self.hold_sec):
					# keep old mapping for stability
					continue
				# switch mapping
				# remove reverse for old_id
				self.id_to_role.pop(old_id, None)
				self.role_to_id[role] = new_id
				self.id_to_role[new_id] = role
				self.role_last_switch[role] = now
	def _assign_roles_for_ids(self, ids: List[int]) -> List[Tuple[int, str]]:
		assigned: List[Tuple[int, str]] = []
		with self._lock:
			# Build list of available roles not currently used by any id
			used_roles = set(self.id_to_role.values())
			available_roles = [r for r in self.role_names if r not in used_roles]

			for vid in ids:
				role = self.id_to_role.get(vid)
				if role is None:
					if not available_roles:
						continue  # No free roles left
					role = available_roles.pop(0)
					self.id_to_role[vid] = role
				assigned.append((vid, role))
		return assigned

	def _set_tracking_status(self, role: str, ok: bool, force: bool = False) -> None:
		self.role_tracking_ok[role] = ok
		pub = self.tracking_publishers.get(role)
		if pub is None:
			return
		try:
			pub.publish(Bool(data=ok))
		except Exception:
			pass

	def _pick_height(self, x: float, y: float) -> float:
		if not self.snap_to_waypoint_height or self.carla_map is None:
			return self.default_z + self.z_offset
		try:
			wp = self.carla_map.get_waypoint(
				carla.Location(x=float(x), y=float(y), z=0.0),
				project_to_road=True,
				lane_type=carla.LaneType.Any,
			)
			if wp is not None:
				return float(wp.transform.location.z) + self.z_offset
		except Exception:
			pass
		return self.default_z + self.z_offset

	def _input_yaw_to_rad(self, yaw_val: float) -> float:
		if self.yaw_in_degrees:
			return math.radians(yaw_val)
		return yaw_val

	def _maybe_flip_yaw(self, x: float, y: float, yaw_rad: float) -> float:
		if not self.enable_yaw_flip or self.carla_map is None:
			return yaw_rad
		try:
			wp = self.carla_map.get_waypoint(
				carla.Location(x=float(x), y=float(y), z=0.5),
				project_to_road=True,
				lane_type=carla.LaneType.Driving,
			)
		except Exception:
			wp = None
		if wp is None:
			return yaw_rad
		lane_yaw_rad = math.radians(wp.transform.rotation.yaw)
		delta = abs(normalize_angle(yaw_rad - lane_yaw_rad))
		thresh = math.radians(max(0.0, min(180.0, self.yaw_flip_threshold_deg)))
		if delta > thresh:
			flipped = normalize_angle(yaw_rad + math.pi)
			rospy.loginfo_throttle(
				1.0,
				"Yaw flip applied at (%.1f, %.1f): det %.1f°, lane %.1f°",
				float(x),
				float(y),
				math.degrees(yaw_rad),
				wp.transform.rotation.yaw,
			)
			return flipped
		return yaw_rad

	def _bev_cb(self, msg: BEVInfo) -> None:
		# Validate array lengths
		n = int(msg.detCounts)
		ids = list(msg.ids[:n])
		cxs = list(msg.center_xs[:n])
		cys = list(msg.center_ys[:n])
		yaws = list(msg.yaws[:n]) if len(msg.yaws) >= n else [0.0] * n

		if not ids:
			return

		tracking_ok_map = {role: False for role in self.role_names}
		det_yaws_rad: List[float] = []
		for idx in range(len(ids)):
			raw_yaw = float(yaws[idx]) if idx < len(yaws) else 0.0
			yaw_rad = self._input_yaw_to_rad(raw_yaw)
			yaw_rad = self._maybe_flip_yaw(float(cxs[idx]), float(cys[idx]), yaw_rad)
			det_yaws_rad.append(yaw_rad)

		if self.enable_matching:
			# Hungarian-based robust assignment
			roles = list(self.role_names)[: self.max_vehicle_count]
			if roles and ids:
				costs = self._compute_cost_matrix(roles, ids, cxs, cys, det_yaws_rad)
				assignment = self._hungarian_assign(costs)
				self._update_mappings_with_assignment(roles, ids, assignment, cxs, cys, det_yaws_rad)
			# Teleport using current stable mapping for ids present in this frame
			with self._lock:
				role_to_id_snapshot = dict(self.role_to_id)
			for role, vid in role_to_id_snapshot.items():
				if vid not in ids:
					continue
				try:
					idx = ids.index(vid)
				except ValueError:
					continue
				x = float(cxs[idx])
				y = float(cys[idx])
				yaw_rad = det_yaws_rad[idx] if idx < len(det_yaws_rad) else 0.0
				yaw_deg = math.degrees(yaw_rad)
				actor = self._resolve_actor_for_role(role)
				if actor is None:
					continue
				# Teleport safety checks
				try:
					curr_tf = actor.get_transform()
					curr_x = float(curr_tf.location.x)
					curr_y = float(curr_tf.location.y)
					curr_yaw_deg = float(curr_tf.rotation.yaw)
				except Exception:
					curr_x = x
					curr_y = y
					curr_yaw_deg = yaw_deg
				dist = math.hypot(x - curr_x, y - curr_y)
				if self.max_teleport_distance_m > 0.0 and dist > self.max_teleport_distance_m:
					rospy.logwarn_throttle(1.0, "Teleport skip for %s: jump %.1fm > %.1fm", role, dist, self.max_teleport_distance_m)
					continue
				# Optional yaw gate at teleport stage
				dyaw = abs((yaw_deg - curr_yaw_deg + 180.0) % 360.0 - 180.0)
				if self.teleport_yaw_gate_deg > 0.0 and dyaw > self.teleport_yaw_gate_deg:
					rospy.logwarn_throttle(1.0, "Teleport skip for %s: yaw delta %.1f° > %.1f°", role, dyaw, self.teleport_yaw_gate_deg)
					continue
				# Stability warmup after mapping switches
				now_sec = float(rospy.Time.now().to_sec())
				last_sw = self.role_last_switch.get(role, now_sec)
				if (now_sec - last_sw) < max(0.0, self.teleport_stability_warmup_sec):
					rospy.loginfo_throttle(1.0, "Teleport hold for %s: stabilizing mapping (%.2fs)", role, now_sec - last_sw)
					continue
				if role in tracking_ok_map:
					tracking_ok_map[role] = False
				z = self._pick_height(x, y)
				location = carla.Location(x=x, y=y, z=z)
				rotation = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)
				tf = carla.Transform(location=location, rotation=rotation)
				try:
					actor.set_transform(tf)
					if role in tracking_ok_map:
						tracking_ok_map[role] = True
				except Exception as exc:
					rospy.logwarn("Teleport failed for %s (id=%s): %s", role, str(vid), exc)
		else:
			# Fallback to simple first-come mapping
			assignments = self._assign_roles_for_ids(ids)
			if not assignments:
				return
			for (vid, role) in assignments:
				try:
					idx = ids.index(vid)
				except ValueError:
					continue
				x = float(cxs[idx])
				y = float(cys[idx])
				yaw_rad = det_yaws_rad[idx] if idx < len(det_yaws_rad) else 0.0
				yaw_deg = math.degrees(yaw_rad)
				actor = self._resolve_actor_for_role(role)
				if actor is None:
					continue
				if role in tracking_ok_map:
					tracking_ok_map[role] = False
				z = self._pick_height(x, y)
				location = carla.Location(x=x, y=y, z=z)
				rotation = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)
				tf = carla.Transform(location=location, rotation=rotation)
				try:
					actor.set_transform(tf)
					if role in tracking_ok_map:
						tracking_ok_map[role] = True
				except Exception as exc:
					rospy.logwarn("Teleport failed for %s (id=%s): %s", role, str(vid), exc)

		for role, ok in tracking_ok_map.items():
			self._set_tracking_status(role, ok)


if __name__ == "__main__":
	try:
		BevIdTeleporter()
		rospy.spin()
	except rospy.ROSInterruptException:
		pass


