#!/usr/bin/env python3

import math
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDrive

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

COLOR_ALIASES = {
	"green": {"green", "초록", "초록색", "녹색"},
	"orange": {"orange", "주황", "주황색"},
	"purple": {"purple", "보라", "보라색", "vio렛", "violet"},
	"yellow": {"yellow", "노랑", "노란색"},
	"red": {"red", "빨강", "빨간색"},
	"pink": {"pink", "분홍", "분홍색"},
	"white": {"white", "흰색"},
	"blue": {"blue", "파랑", "파란색"},
}

COLOR_LOOKUP = {alias: canonical for canonical, aliases in COLOR_ALIASES.items() for alias in aliases}


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


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
	half = yaw * 0.5
	return (0.0, 0.0, math.sin(half), math.cos(half))


class BevIdTeleporter:
	def __init__(self):
		rospy.init_node("bev_id_teleporter", anonymous=True)
		if carla is None:
			raise RuntimeError("CARLA Python API unavailable")

		self.bev_time_stamp = None

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
		self.enable_yaw_filter: bool = bool(rospy.get_param("~enable_yaw_filter", True))
		self.yaw_filter_alpha: float = float(rospy.get_param("~yaw_filter_alpha", 0.2))
		self.yaw_filter_jump_deg: float = float(rospy.get_param("~yaw_filter_jump_deg", 60.0))
		self.yaw_filter_timeout_sec: float = float(rospy.get_param("~yaw_filter_timeout_sec", 2.0))
		self.enable_waypoint_heading_guard: bool = bool(rospy.get_param("~enable_waypoint_heading_guard", False))
		self.waypoint_heading_window_deg: float = float(rospy.get_param("~waypoint_heading_window_deg", 35.0))
		self.disable_collision_check: bool = bool(rospy.get_param("~disable_collision_check", True))
		self.snap_to_spawn_heading: bool = bool(rospy.get_param("~snap_to_spawn_heading", False))
		self.spawn_heading_max_distance_m: float = float(rospy.get_param("~spawn_heading_max_distance_m", 8.0))
		self.snap_to_spawn_pose_initial: bool = bool(rospy.get_param("~snap_to_spawn_pose_initial", True))
		self.yaw_blend_enabled: bool = bool(rospy.get_param("~yaw_blend_enabled", True))
		self.yaw_waypoint_weight: float = float(rospy.get_param("~yaw_waypoint_weight", 0.1))
		self.yaw_path_weight: float = float(rospy.get_param("~yaw_path_weight", 0.1))
		self.yaw_stability_alpha: float = float(rospy.get_param("~yaw_stability_alpha", 0.3))
		self.only_correct_yaw_during_alignment: bool = bool(rospy.get_param("~only_correct_yaw_during_alignment", True))
		# Kalman filter configuration
		self.kf_enabled: bool = bool(rospy.get_param("~kf_enabled", True))
		self.kf_default_dt: float = float(rospy.get_param("~kf_default_dt", 0.05))
		self.kf_max_dt: float = float(rospy.get_param("~kf_max_dt", 0.3))
		self.kf_process_noise_pos: float = float(rospy.get_param("~kf_process_noise_pos", 0.3))
		self.kf_process_noise_vel: float = float(rospy.get_param("~kf_process_noise_vel", 0.5))
		self.kf_process_noise_yaw_deg: float = float(rospy.get_param("~kf_process_noise_yaw_deg", 15.0))
		self.kf_process_noise_yaw_rate_deg: float = float(rospy.get_param("~kf_process_noise_yaw_rate_deg", 20.0))
		self.kf_measurement_noise_pos: float = float(rospy.get_param("~kf_measurement_noise_pos", 0.1))
		self.kf_measurement_noise_yaw_deg: float = float(rospy.get_param("~kf_measurement_noise_yaw_deg", 45.0))
		self.kf_init_pos_std: float = float(rospy.get_param("~kf_init_pos_std", 1.0))
		self.kf_init_vel_std: float = float(rospy.get_param("~kf_init_vel_std", 1.0))
		self.kf_init_yaw_deg_std: float = float(rospy.get_param("~kf_init_yaw_deg_std", 45.0))
		self.kf_init_yaw_rate_deg_std: float = float(rospy.get_param("~kf_init_yaw_rate_deg_std", 60.0))
		self.kf_meas_reject_distance_m: float = float(rospy.get_param("~kf_meas_reject_distance_m", 10.0))
		self.kf_meas_reject_yaw_deg: float = float(rospy.get_param("~kf_meas_reject_yaw_deg", 25.0))
		# Control input for Kalman filter prediction (optional)
		self.kf_use_control_input: bool = bool(rospy.get_param("~kf_use_control_input", True))
		self.kf_control_timeout_sec: float = float(rospy.get_param("~kf_control_timeout_sec", 0.5))
		self.kf_wheelbase_m: float = float(rospy.get_param("~kf_wheelbase_m", 2.7))  # Vehicle wheelbase for bicycle model
		# Optional forward prediction horizon (seconds) for teleport target, relative to current time
		# e.g., 1.0이면 "1초 뒤 위치" 칼만 예측값을 텔레포트에 사용
		self.kf_prediction_horizon_sec: float = float(rospy.get_param("~kf_prediction_horizon_sec", 0.3))
		# Motion-based yaw estimation (칼만과 독립적으로 런치 파라미터로만 제어)
		# - launch 에서 주는 motion_yaw_enabled 값 그대로 사용
		self.motion_yaw_enabled: bool = bool(rospy.get_param("~motion_yaw_enabled", True))
		# Tuned for 10-20Hz perception (0.05-0.1s) at 2m/s vehicle speed
		# min_distance: 0.1-0.2m per frame at 2m/s → 0.15m threshold
		self.motion_yaw_min_distance_m: float = float(rospy.get_param("~motion_yaw_min_distance_m", 0.15))
		# timeout: allow 3-7 frame misses (0.3-0.7s at 10-20Hz) → 0.7s
		self.motion_yaw_max_timeout_sec: float = float(rospy.get_param("~motion_yaw_max_timeout_sec", 0.7))
		# max_jump: reasonable for 2m/s speed changes → 35deg
		self.motion_yaw_max_jump_deg: float = float(rospy.get_param("~motion_yaw_max_jump_deg", 35.0))

		# Height / waypoint snapping
		self.snap_to_waypoint_height: bool = bool(rospy.get_param("~snap_to_waypoint_height", True))
		self.default_z: float = float(rospy.get_param("~default_z", 0.5))
		self.z_offset: float = float(rospy.get_param("~z_offset", 0.0))
		# Teleport safety
		self.max_teleport_distance_m: float = float(rospy.get_param("~max_teleport_distance_m", 20.0))
		self.teleport_yaw_gate_deg: float = float(rospy.get_param("~teleport_yaw_gate_deg", 150.0))
		self.teleport_stability_warmup_sec: float = float(rospy.get_param("~teleport_stability_warmup_sec", 0.5))
		# Color-aware matching
		self.enable_color_matching: bool = bool(rospy.get_param("~enable_color_matching", True))
		self.color_gate_on_mismatch: bool = bool(rospy.get_param("~color_gate_on_mismatch", False))
		self.color_match_bonus: float = float(rospy.get_param("~color_match_bonus", 4.0))
		self.color_mismatch_penalty: float = float(rospy.get_param("~color_mismatch_penalty", 15.0))
		self.color_missing_penalty: float = float(rospy.get_param("~color_missing_penalty", 2.0))
		self.role_color_map: Dict[str, str] = self._parse_role_color_map(rospy.get_param("~role_color_map", {}))
		if self.enable_color_matching and self.role_color_map:
			rospy.loginfo("Color hints loaded: %s", self.role_color_map)
		# Initial alignment (relaxed gating) parameters
		self.enable_initial_alignment: bool = bool(rospy.get_param("~enable_initial_alignment", True))
		self.initial_alignment_timeout_sec: float = float(rospy.get_param("~initial_alignment_timeout_sec", 5.0))
		self.initial_alignment_avg_duration_sec: float = float(rospy.get_param("~initial_alignment_avg_duration_sec", 1.0))
		self.initial_alignment_dist_gate_m: float = float(rospy.get_param("~initial_alignment_dist_gate_m", 0.0))
		self.initial_alignment_yaw_gate_deg: float = float(rospy.get_param("~initial_alignment_yaw_gate_deg", 0.0))
		self.initial_alignment_max_teleport_m: float = float(rospy.get_param("~initial_alignment_max_teleport_m", 0.0))
		self.initial_alignment_teleport_yaw_gate_deg: float = float(rospy.get_param("~initial_alignment_teleport_yaw_gate_deg", 0.0))
		self.initial_alignment_skip_warmup: bool = bool(rospy.get_param("~initial_alignment_skip_warmup", True))
		self._initial_alignment_start_sec: float = float(rospy.Time.now().to_sec())
		self._initial_alignment_done_roles: Set[str] = set()
		self._initial_alignment_complete: bool = not self.enable_initial_alignment
		self._initial_alignment_samples: Dict[str, Dict[str, float]] = {}
		self._initial_alignment_avg_results: Dict[str, Dict[str, float]] = {}
		self._initial_alignment_avg_ready: bool = self.initial_alignment_avg_duration_sec <= 0.0
		self._initial_alignment_seed_performed: bool = not self.enable_initial_alignment

		# Connect to CARLA
		self.client = carla.Client(self.host, self.port)
		self.client.set_timeout(5.0)
		self.world = self.client.get_world()
		self.carla_map = self.world.get_map()
		self._spawn_points: List[carla.Transform] = []
		try:
			self._spawn_points = list(self.carla_map.get_spawn_points())
		except Exception:
			self._spawn_points = []

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
		self.bev_pose_publishers: Dict[str, rospy.Publisher] = {}
		self._filtered_yaws: Dict[int, float] = {}
		self._filtered_yaw_stamp: Dict[int, float] = {}
		self._roles_initialized: Set[str] = set()
		self._kf_states: Dict[str, Dict[str, Any]] = {}
		# Previous position for motion-based yaw estimation (role -> (x, y, timestamp, yaw))
		self._prev_position: Dict[str, Tuple[float, float, float, float]] = {}
		# Control commands for Kalman filter prediction (role -> (speed, steering_angle, timestamp))
		self._kf_control_cmds: Dict[str, Tuple[float, float, float]] = {}
		yaw_init_var = math.radians(self.kf_init_yaw_deg_std) ** 2
		yaw_rate_init_var = math.radians(self.kf_init_yaw_rate_deg_std) ** 2
		self._kf_H = np.array(
			[
				[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
				[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
			],
			dtype=float,
		)
		self._kf_I = np.eye(6, dtype=float)
		self._kf_init_cov = np.diag(
			[
				self.kf_init_pos_std ** 2,
				self.kf_init_pos_std ** 2,
				self.kf_init_vel_std ** 2,
				self.kf_init_vel_std ** 2,
				yaw_init_var,
				yaw_rate_init_var,
			]
		)
		self._kf_R = np.diag(
			[
				self.kf_measurement_noise_pos ** 2,
				self.kf_measurement_noise_pos ** 2,
				math.radians(self.kf_measurement_noise_yaw_deg) ** 2,
			]
		)
		# Periodic republish of tracking_ok so monitoring tools see fresh messages
		self.tracking_status_rate_hz: float = float(rospy.get_param("~tracking_status_rate_hz", 20.0))
		self._tracking_status_timer = None

		self._refresh_role_to_actor()

		# ROS I/O
		self.role_state = {role: {"pos": None, "yaw": None, "stamp": None} for role in self.role_names}
		for role in self.role_names:
			rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role, queue_size=10)
			topic = f"/carla/{role}/bev_tracking_ok"
			self.tracking_publishers[role] = rospy.Publisher(topic, Bool, queue_size=1, latch=True)
			self.role_tracking_ok[role] = False
			self._set_tracking_status(role, False, force=True)
			pose_topic = f"/bev_tracking_pose/{role}"
			self.bev_pose_publishers[role] = rospy.Publisher(pose_topic, PoseStamped, queue_size=1, latch=True)
			# Subscribe to control commands for Kalman filter prediction (if enabled)
			if self.kf_use_control_input:
				control_topic = f"/carla/{role}/vehicle_control_cmd"
				rospy.Subscriber(control_topic, AckermannDrive, self._control_cmd_cb, callback_args=role, queue_size=1)
		self.sub = rospy.Subscriber(self.topic_name, BEVInfo, self._bev_cb, queue_size=1)
		rospy.loginfo(
			"bev_id_teleporter ready: topic=%s, roles=%s (max=%d)",
			self.topic_name,
			",".join(self.role_names),
			self.max_vehicle_count,
		)
		# Start periodic republish timer
		if self.tracking_status_rate_hz > 0.0:
			self._tracking_status_timer = rospy.Timer(
				rospy.Duration(1.0 / max(0.1, self.tracking_status_rate_hz)),
				self._tracking_status_tick,
				oneshot=False,
				reset=True,
			)

	def _normalize_color(self, value: Any) -> str:
		if value is None:
			return ""
		try:
			text = str(value)
		except Exception:
			return ""
		text = text.strip().lower()
		if not text:
			return ""
		return COLOR_LOOKUP.get(text, text)

	def _parse_role_color_map(self, param: Any) -> Dict[str, str]:
		result: Dict[str, str] = {}
		items = None
		if isinstance(param, dict):
			items = param.items()
		elif isinstance(param, (list, tuple)):
			items = []
			for entry in param:
				if isinstance(entry, (list, tuple)) and len(entry) >= 2:
					items.append((entry[0], entry[1]))
		elif isinstance(param, str):
			chunks = [chunk.strip() for chunk in param.split(",") if chunk.strip()]
			items = []
			for chunk in chunks:
				if ":" not in chunk:
					continue
				key, value = chunk.split(":", 1)
				items.append((key.strip(), value.strip()))
		if items is None:
			return result
		for key, value in items:
			if not isinstance(key, str):
				continue
			color = self._normalize_color(value)
			if color:
				result[key.strip()] = color
		return result

	def _apply_color_hint(self, role: str, det_color: str, base_cost: float, big: float) -> float:
		expected = self.role_color_map.get(role)
		if not expected:
			return base_cost
		color = det_color or ""
		if not color:
			return base_cost + max(0.0, self.color_missing_penalty)
		if color == expected:
			bonus = max(0.0, self.color_match_bonus)
			return max(0.0, base_cost - bonus)
		if self.color_gate_on_mismatch:
			return big
		return base_cost + max(0.0, self.color_mismatch_penalty)

	def _pick_role_by_color_hint(self, available_roles: List[str], det_color: Optional[str]) -> str:
		if not available_roles:
			return ""
		color = self._normalize_color(det_color)
		if color:
			for role in available_roles:
				if self.role_color_map.get(role) == color:
					return role
		return available_roles[0]

	def _initial_alignment_active(self) -> bool:
		if not self.enable_initial_alignment or self._initial_alignment_complete:
			return False
		if len(self._initial_alignment_done_roles) >= len(self.role_names):
			self._finalize_initial_alignment("all roles initialized")
			return False
		now = float(rospy.Time.now().to_sec())
		if self.initial_alignment_timeout_sec > 0.0 and (now - self._initial_alignment_start_sec) >= self.initial_alignment_timeout_sec:
			self._finalize_initial_alignment(
				"timeout after %.1fs (%d/%d roles aligned)"
				% (
					now - self._initial_alignment_start_sec,
					len(self._initial_alignment_done_roles),
					len(self.role_names),
				)
			)
			return False
		return True

	def _mark_initial_alignment_done(self, role: str) -> None:
		if not self.enable_initial_alignment or self._initial_alignment_complete:
			return
		self._initial_alignment_done_roles.add(role)
		if len(self._initial_alignment_done_roles) >= len(self.role_names):
			self._finalize_initial_alignment("all roles initialized")

	def _finalize_initial_alignment(self, reason: str) -> None:
		if self._initial_alignment_complete:
			return
		self._initial_alignment_complete = True
		rospy.loginfo("Initial alignment complete: %s", reason)

	def _set_role_initialized(self, role: str) -> None:
		if role in self._roles_initialized:
			return
		self._roles_initialized.add(role)

	def _control_cmd_cb(self, msg: AckermannDrive, role: str) -> None:
		"""Callback for vehicle control commands. Stores control input for Kalman filter prediction."""
		if not self.kf_use_control_input:
			return
		with self._lock:
			timestamp = float(rospy.Time.now().to_sec())
			speed = float(msg.speed)
			steering_angle = float(msg.steering_angle)
			self._kf_control_cmds[role] = (speed, steering_angle, timestamp)
	
	def _yaw_corrections_active(self) -> bool:
		if not self.only_correct_yaw_during_alignment:
			return True
		return len(self._roles_initialized) < len(self.role_names)

	def _nearest_spawn_transform(self, x: float, y: float) -> Tuple[Optional[carla.Transform], float]:
		best_tf: Optional[carla.Transform] = None
		best_dist: float = float("inf")
		if not self._spawn_points:
			return None, best_dist
		for tf in self._spawn_points:
			dx = tf.location.x - float(x)
			dy = tf.location.y - float(y)
			dist = math.hypot(dx, dy)
			if dist < best_dist:
				best_dist = dist
				best_tf = tf
		return best_tf, best_dist

	def _apply_spawn_snap(self, role: str, x: float, y: float, yaw_deg: float, pose_snap: bool) -> Tuple[float, float, float]:
		tf, dist = self._nearest_spawn_transform(x, y)
		if (
			tf is None
			or (self.spawn_heading_max_distance_m > 0.0 and dist > self.spawn_heading_max_distance_m)
		):
			return x, y, yaw_deg
		heading_snap_allowed = pose_snap or (self.snap_to_spawn_heading and self._yaw_corrections_active())
		changed = False
		if heading_snap_allowed:
			target_yaw = float(tf.rotation.yaw)
			if abs(target_yaw - yaw_deg) > 1e-3:
				yaw_deg = target_yaw
				changed = True
		if pose_snap:
			x = float(tf.location.x)
			y = float(tf.location.y)
			changed = True
		if changed:
			rospy.loginfo(
				"bev_id_teleporter: %s snapped to spawn heading%s (dist=%.2fm, yaw=%.1f°)",
				role,
				" and pose" if pose_snap else "",
				dist,
				yaw_deg,
			)
		return x, y, yaw_deg

	def _blend_yaw_sources(
		self,
		role: str,
		x: float,
		y: float,
		bev_yaw_deg: float,
	) -> float:
		if not self.yaw_blend_enabled:
			return bev_yaw_deg
		if not self._yaw_corrections_active():
			return bev_yaw_deg
		wp_yaw: Optional[float] = None
		path_yaw = self._get_path_heading(role)
		if self.carla_map is not None:
			try:
				loc = carla.Location(x=float(x), y=float(y), z=0.5)
				wp = self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
				if wp is not None:
					wp_yaw = float(wp.transform.rotation.yaw)
			except Exception:
				wp_yaw = None
		weights: List[float] = []
		values: List[float] = []
		if wp_yaw is not None and self.yaw_waypoint_weight > 0.0:
			weights.append(self.yaw_waypoint_weight)
			values.append(wp_yaw)
		if path_yaw is not None and self.yaw_path_weight > 0.0:
			weights.append(self.yaw_path_weight)
			values.append(path_yaw)
		if not values:
			return bev_yaw_deg
		total_w = sum(weights)
		if total_w <= 1e-6:
			return bev_yaw_deg
		target = sum(w * v for w, v in zip(weights, values)) / total_w
		alpha = max(0.0, min(1.0, self.yaw_stability_alpha))
		return (1.0 - alpha) * bev_yaw_deg + alpha * target

	def _kf_force_state(self, role: str, x: float, y: float, yaw_rad: float, stamp: float) -> None:
		if not self.kf_enabled:
			return
		state_vec = np.zeros((6, 1), dtype=float)
		state_vec[0, 0] = float(x)
		state_vec[1, 0] = float(y)
		state_vec[4, 0] = normalize_angle(float(yaw_rad))
		self._kf_states[role] = {
			"x": state_vec,
			"P": np.array(self._kf_init_cov, copy=True),
			"stamp": float(stamp),
		}

	def _kf_process_noise(self, dt: float) -> np.ndarray:
		dt = max(1e-3, float(dt))
		pos_var = (self.kf_process_noise_pos * dt) ** 2
		vel_var = (self.kf_process_noise_vel) ** 2
		yaw_var = (math.radians(self.kf_process_noise_yaw_deg) * dt) ** 2
		yaw_rate_var = math.radians(self.kf_process_noise_yaw_rate_deg) ** 2
		return np.diag([pos_var, pos_var, vel_var, vel_var, yaw_var, yaw_rate_var])

	def _kf_predict_state(self, role: str, stamp: float) -> None:
		if not self.kf_enabled:
			return
		state = self._kf_states.get(role)
		if state is None:
			return
		last_stamp = state.get("stamp")
		if last_stamp is None:
			dt = self.kf_default_dt
		else:
			dt = float(stamp) - float(last_stamp)
			if dt <= 1e-6:
				dt = self.kf_default_dt
		if self.kf_max_dt > 0.0:
			dt = min(self.kf_max_dt, max(dt, 1e-3))
		else:
			dt = max(dt, 1e-3)
		F = np.array(
			[
				[1.0, 0.0, dt, 0.0, 0.0, 0.0],
				[0.0, 1.0, 0.0, dt, 0.0, 0.0],
				[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 0.0, 1.0, dt],
				[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
			],
			dtype=float,
		)
		state["x"] = F @ state["x"]
		
		# Apply control input if available and enabled
		if self.kf_use_control_input:
			with self._lock:
				control_cmd = self._kf_control_cmds.get(role)
			if control_cmd is not None:
				cmd_speed, cmd_steering, cmd_timestamp = control_cmd
				# Check if control command is recent enough
				if (stamp - cmd_timestamp) <= self.kf_control_timeout_sec:
					current_yaw = float(state["x"][4, 0])
					# Update velocity from control command
					# vx = speed * cos(yaw), vy = speed * sin(yaw)
					state["x"][2, 0] = float(cmd_speed) * math.cos(current_yaw)
					state["x"][3, 0] = float(cmd_speed) * math.sin(current_yaw)
					# Update yaw rate from steering command (bicycle model)
					# yaw_rate = (v * tan(steering)) / wheelbase
					if abs(float(cmd_speed)) > 1e-3 and self.kf_wheelbase_m > 1e-3:
						yaw_rate_cmd = (float(cmd_speed) * math.tan(float(cmd_steering))) / self.kf_wheelbase_m
						state["x"][5, 0] = yaw_rate_cmd
		
		state["x"][4, 0] = normalize_angle(float(state["x"][4, 0]))
		state["P"] = F @ state["P"] @ F.T + self._kf_process_noise(dt)
		state["P"] = 0.5 * (state["P"] + state["P"].T)
		state["stamp"] = float(stamp)

	def _kf_predict_ahead_state(self, role: str, horizon_sec: float) -> Optional[Tuple[float, float, float]]:
		"""
		현재 칼만 상태를 기준으로 horizon_sec 뒤의 (x, y, yaw_rad) 예측값을 계산하되,
		내부 상태(self._kf_states)는 변경하지 않고 리턴만 한다.
		"""
		if not self.kf_enabled:
			return None
		state = self._kf_states.get(role)
		if state is None:
			return None
		dt = max(1e-3, float(horizon_sec))
		F = np.array(
			[
				[1.0, 0.0, dt, 0.0, 0.0, 0.0],
				[0.0, 1.0, 0.0, dt, 0.0, 0.0],
				[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 0.0, 1.0, dt],
				[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
			],
			dtype=float,
		)
		x_vec = F @ state["x"]

		# 제어입력을 horizon 동안 유지된다고 가정한 간단한 예측 보정 (선속/조향 기반)
		if self.kf_use_control_input:
			with self._lock:
				control_cmd = self._kf_control_cmds.get(role)
			if control_cmd is not None:
				cmd_speed, cmd_steering, cmd_timestamp = control_cmd
				now = float(rospy.Time.now().to_sec())
				if (now - cmd_timestamp) <= self.kf_control_timeout_sec:
					current_yaw = float(x_vec[4, 0])
					x_vec[2, 0] = float(cmd_speed) * math.cos(current_yaw)
					x_vec[3, 0] = float(cmd_speed) * math.sin(current_yaw)
					if abs(float(cmd_speed)) > 1e-3 and self.kf_wheelbase_m > 1e-3:
						yaw_rate_cmd = (float(cmd_speed) * math.tan(float(cmd_steering))) / self.kf_wheelbase_m
						x_vec[5, 0] = yaw_rate_cmd

		x_vec[4, 0] = normalize_angle(float(x_vec[4, 0]))
		return float(x_vec[0, 0]), float(x_vec[1, 0]), float(x_vec[4, 0])

	def _kf_should_reject_measurement(self, role: str, meas_x: float, meas_y: float, meas_yaw: float) -> bool:
		if not self.kf_enabled:
			return False
		state = self._kf_states.get(role)
		if state is None:
			return False
		pred_x = float(state["x"][0, 0])
		pred_y = float(state["x"][1, 0])
		dist = math.hypot(meas_x - pred_x, meas_y - pred_y)
		if self.kf_meas_reject_distance_m > 0.0 and dist > self.kf_meas_reject_distance_m:
			rospy.logwarn_throttle(
				1.0,
				"KF measurement reject for %s: dist %.1fm > %.1fm",
				role,
				dist,
				self.kf_meas_reject_distance_m,
			)
			return True
		if self.kf_meas_reject_yaw_deg > 0.0:
			pred_yaw = float(state["x"][4, 0])
			dyaw = abs(math.degrees(normalize_angle(meas_yaw - pred_yaw)))
			if dyaw > self.kf_meas_reject_yaw_deg:
				rospy.logwarn_throttle(
					1.0,
					"KF measurement reject for %s: yaw delta %.1f° > %.1f°",
					role,
					dyaw,
					self.kf_meas_reject_yaw_deg,
				)
				return True
		return False

	def _kf_update_state(self, role: str, meas_x: float, meas_y: float, meas_yaw: float) -> None:
		if not self.kf_enabled:
			return
		state = self._kf_states.get(role)
		if state is None:
			return
		z = np.array([[float(meas_x)], [float(meas_y)], [normalize_angle(float(meas_yaw))]], dtype=float)
		x_vec = state["x"]
		innovation = z - self._kf_H @ x_vec
		innovation[2, 0] = normalize_angle(float(innovation[2, 0]))
		S = self._kf_H @ state["P"] @ self._kf_H.T + self._kf_R
		try:
			S_inv = np.linalg.inv(S)
		except np.linalg.LinAlgError:
			rospy.logwarn_throttle(5.0, "KF update skipped for %s due to singular matrix", role)
			return
		K = state["P"] @ self._kf_H.T @ S_inv
		x_vec = x_vec + K @ innovation
		x_vec[4, 0] = normalize_angle(float(x_vec[4, 0]))
		P = (self._kf_I - K @ self._kf_H) @ state["P"]
		state["P"] = 0.5 * (P + P.T)
		state["x"] = x_vec

	def _kf_fuse_measurement(
		self,
		role: str,
		meas_x: float,
		meas_y: float,
		meas_yaw: float,
		stamp: float,
	) -> Tuple[float, float, float, bool]:
		if not self.kf_enabled:
			return meas_x, meas_y, meas_yaw, False
		if role not in self._kf_states:
			self._kf_force_state(role, meas_x, meas_y, meas_yaw, stamp)
			return meas_x, meas_y, meas_yaw, True
		self._kf_predict_state(role, stamp)
		used = False
		if not self._kf_should_reject_measurement(role, meas_x, meas_y, meas_yaw):
			self._kf_update_state(role, meas_x, meas_y, meas_yaw)
			used = True
		state = self._kf_states.get(role)
		if state is None:
			return meas_x, meas_y, meas_yaw, used
		yaw_est = normalize_angle(float(state["x"][4, 0]))
		return float(state["x"][0, 0]), float(state["x"][1, 0]), yaw_est, used

	def _compute_yaw_from_motion(
		self,
		role: str,
		x: float,
		y: float,
		current_yaw_rad: float,
		timestamp: float,
	) -> Optional[float]:
		"""
		Compute yaw from previous and current position using arctan2.
		Returns None if motion-based yaw cannot be computed.
		"""
		if not self.motion_yaw_enabled:
			return None
		
		prev_data = self._prev_position.get(role)
		if prev_data is None:
			# No previous position, store current position and return None
			self._prev_position[role] = (x, y, timestamp, current_yaw_rad)
			return None
		
		# 여기서 prev_data는 (x, y, timestamp, yaw) 형태의 튜플입니다.
		prev_x, prev_y, prev_timestamp, prev_yaw_rad = prev_data
		rospy.loginfo("prev_x: %.1f, prev_y: %.1f, prev_timestamp: %.1f, prev_yaw_rad: %.1f", prev_x, prev_y, prev_timestamp, math.degrees(prev_yaw_rad))
		# Check timeout
		dt = timestamp - prev_timestamp
		rospy.loginfo("[Motion Yaw] %s: dt=%.3fs (timeout=%.3fs)", role, dt, self.motion_yaw_max_timeout_sec)
		if dt > self.motion_yaw_max_timeout_sec:
			# Previous position too old, update and return None
			rospy.logwarn("[Motion Yaw] %s: Timeout! Updating prev_position", role)
			self._prev_position[role] = (x, y, timestamp, current_yaw_rad)
			return None
		
		# Compute displacement
		dx = x - prev_x
		dy = y - prev_y
		dist = math.hypot(dx, dy)
		rospy.loginfo("[Motion Yaw] %s: dist=%.3fm (min=%.3fm), dx=%.3f, dy=%.3f", role, dist, self.motion_yaw_min_distance_m, dx, dy)
		
		# Check minimum distance
		if dist < self.motion_yaw_min_distance_m:
			# Not enough movement, keep previous position timestamp but return None
			# (don't update position to avoid accumulating small errors)
			rospy.loginfo("[Motion Yaw] %s: Distance too small (%.3fm < %.3fm), skipping", role, dist, self.motion_yaw_min_distance_m)
			return None
		
		# Compute heading from motion

		motion_yaw_rad = math.atan2(dy, dx)
		motion_yaw_rad = normalize_angle(motion_yaw_rad)
		rospy.loginfo("Motion yaw!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: %.1f°", math.degrees(motion_yaw_rad))


		# Check for large yaw jumps (might indicate outlier or teleport)
		yaw_diff = abs(normalize_angle(motion_yaw_rad - prev_yaw_rad))
		max_jump_rad = math.radians(self.motion_yaw_max_jump_deg)
		if yaw_diff > max_jump_rad and yaw_diff < (2.0 * math.pi - max_jump_rad):
			# Large jump detected, use current yaw from perception
			rospy.logwarn_throttle(
				2.0,
				"Motion yaw jump for %s: %.1f° (using perception yaw)",
				role,
				math.degrees(yaw_diff),
			)
			self._prev_position[role] = (x, y, timestamp, current_yaw_rad)
			return None
		
		# Update previous position
		self._prev_position[role] = (x, y, timestamp, motion_yaw_rad)
		
		return motion_yaw_rad
	
	def _get_path_heading(self, role: str) -> Optional[float]:
		state = self.role_state.get(role)
		if not state:
			return None
		yaw = state.get("yaw")
		if yaw is None:
			return None
		return math.degrees(float(yaw))

	def _accumulate_initial_alignment_samples(
		self,
		ids: List[int],
		cxs: List[float],
		cys: List[float],
		yaws: List[float],
		colors: List[str],
	) -> None:
		if not self.enable_initial_alignment or self._initial_alignment_seed_performed:
			return
		for idx, color in enumerate(colors):
			if not color:
				continue
			if idx >= len(cxs) or idx >= len(cys) or idx >= len(yaws) or idx >= len(ids):
				continue
			sample = self._initial_alignment_samples.setdefault(
				color,
				{"count": 0.0, "sum_x": 0.0, "sum_y": 0.0, "sum_sin": 0.0, "sum_cos": 0.0, "last_id": None},
			)
			sample["count"] += 1.0
			sample["sum_x"] += float(cxs[idx])
			sample["sum_y"] += float(cys[idx])
			sample["sum_sin"] += math.sin(float(yaws[idx]))
			sample["sum_cos"] += math.cos(float(yaws[idx]))
			sample["last_id"] = int(ids[idx])

	def _prepare_initial_alignment_avg_results(self) -> None:
		self._initial_alignment_avg_results = {}
		for color, sample in self._initial_alignment_samples.items():
			count = sample.get("count", 0.0)
			if count <= 0.0:
				continue
			sum_sin = sample.get("sum_sin", 0.0)
			sum_cos = sample.get("sum_cos", 0.0)
			yaw = math.atan2(sum_sin, sum_cos) if (sum_sin != 0.0 or sum_cos != 0.0) else 0.0
			self._initial_alignment_avg_results[color] = {
				"x": sample.get("sum_x", 0.0) / count,
				"y": sample.get("sum_y", 0.0) / count,
				"yaw": yaw,
				"id": sample.get("last_id"),
			}
		self._initial_alignment_avg_ready = True
		rospy.loginfo(
			"Initial alignment averaging ready with %d colored samples",
			len(self._initial_alignment_avg_results),
		)

	def _apply_initial_alignment_seed(self) -> bool:
		if not self.enable_initial_alignment or self._initial_alignment_seed_performed:
			return False
		if not self.role_color_map:
			rospy.logwarn_throttle(5.0, "Initial alignment seed skipped: role_color_map empty")
			return False
		if not self._initial_alignment_avg_results:
			return False
		seeded = 0
		for role in self.role_names:
			color = self.role_color_map.get(role)
			if not color:
				continue
			sample = self._initial_alignment_avg_results.get(color)
			if sample is None:
				continue
			if self._teleport_role_to_sample(role, sample):
				seeded += 1
		if seeded > 0:
			self._initial_alignment_seed_performed = True
			self._initial_alignment_samples.clear()
			rospy.loginfo(
				"Initial alignment seeding applied for %d roles (averaged %.1fs)",
				seeded,
				self.initial_alignment_avg_duration_sec,
			)
			return True
		return False

	def _teleport_role_to_sample(self, role: str, sample: Dict[str, float]) -> bool:
		actor = self._resolve_actor_for_role(role)
		if actor is None:
			return False
		
		x = float(sample.get("x", 0.0))
		y = float(sample.get("y", 0.0))

		yaw_rad = float(sample.get("yaw", 0.0))
		yaw_deg = math.degrees(yaw_rad)

		if self.yaw_blend_enabled:
			yaw_deg = self._blend_yaw_sources(role, x, y, yaw_deg)

		pose_snap = self.snap_to_spawn_pose_initial and role not in self._initial_alignment_done_roles
		x, y, yaw_deg = self._apply_spawn_snap(role, x, y, yaw_deg, pose_snap)

		yaw_rad_applied = math.radians(yaw_deg)
		z = self._pick_height(x, y)

		location = carla.Location(x=x, y=y, z=z)
		rotation = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)
		tf = carla.Transform(location=location, rotation=rotation)
		det_id = sample.get("id")

		if self._apply_transform(actor, tf, role=role, det_id=det_id):
			self._kf_force_state(role, x, y, yaw_rad_applied, float(rospy.Time.now().to_sec()))
			if det_id is not None:
				try:
					det_int = int(det_id)
					self.role_to_id[role] = det_int
					self.id_to_role[det_int] = role
				except Exception:
					pass

			self.role_last_switch[role] = float(rospy.Time.now().to_sec())
			self._mark_initial_alignment_done(role)
			self._set_role_initialized(role)
			return True
		return False

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
				if role not in self.bev_pose_publishers:
					pose_topic = f"/bev_tracking_pose/{role}"
					self.bev_pose_publishers[role] = rospy.Publisher(pose_topic, PoseStamped, queue_size=1, latch=True)

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

	def _compute_cost_matrix(
		self,
		roles: List[str],
		ids: List[int],
		xs: List[float],
		ys: List[float],
		yaws: List[float],
		colors: List[str],
		dist_gate_m: float,
		yaw_gate_deg: float,
	) -> List[List[float]]:
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
				if (dist_gate_m > 0.0 and dist > dist_gate_m) or (yaw_gate_deg > 0.0 and ang_deg > yaw_gate_deg):
					row.append(big)
				else:
					cost = self.w_pos * dist + self.w_yaw * ang
					if self.enable_color_matching and self.role_color_map:
						det_color = colors[idx] if idx < len(colors) else ""
						cost = self._apply_color_hint(role, det_color, cost, big)
					row.append(min(cost, big))
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

	def _assign_roles_for_ids(self, ids: List[int], colors: Optional[List[str]] = None) -> List[Tuple[int, str]]:
		assigned: List[Tuple[int, str]] = []
		with self._lock:
			# Build list of available roles not currently used by any id
			used_roles = set(self.id_to_role.values())
			available_roles = [r for r in self.role_names if r not in used_roles]
			color_by_id: Dict[int, str] = {}
			if colors is not None:
				for idx, vid in enumerate(ids):
					if idx < len(colors):
						color_by_id[vid] = colors[idx]

			for vid in ids:
				role = self.id_to_role.get(vid)
				if role is None:
					if not available_roles:
						continue  # No free roles left
					if self.enable_color_matching and self.role_color_map and color_by_id:
						role = self._pick_role_by_color_hint(available_roles, color_by_id.get(vid))
						if role in available_roles:
							available_roles.remove(role)
					else:
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

	def _publish_tracking_pose(self, role: str, x: float, y: float, yaw_rad: float) -> None:
		pub = self.bev_pose_publishers.get(role)
		if pub is None:
			return
		msg = PoseStamped()
		msg.header.stamp = self.bev_time_stamp
		msg.header.frame_id = "map"
		msg.pose.position.x = x
		msg.pose.position.y = y
		msg.pose.position.z = self._pick_height(x, y)
		qx, qy, qz, qw = yaw_to_quaternion(yaw_rad)
		msg.pose.orientation.x = qx
		msg.pose.orientation.y = qy
		msg.pose.orientation.z = qz
		msg.pose.orientation.w = qw
		try:
			pub.publish(msg)
		except Exception:
			pass

	def _apply_transform(
		self,
		actor: carla.Actor,
		transform: carla.Transform,
		role: Optional[str] = None,
		det_id: Optional[int] = None,
	) -> bool:
		if actor is None:
			return False
		disabled = False
		if self.disable_collision_check and hasattr(actor, "set_simulate_physics"):
			try:
				actor.set_simulate_physics(False)
				disabled = True
			except Exception as exc:
				rospy.logwarn_throttle(5.0, "bev_id_teleporter: failed to disable physics for %s: %s", role or actor.id, exc)
		try:
			actor.set_transform(transform)
			try:
				if hasattr(actor, "set_target_velocity"):
					actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
				if hasattr(actor, "set_target_angular_velocity"):
					actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
			except Exception:
				pass
			return True
		except Exception as exc:
			if role is not None:
				rospy.logwarn("Teleport failed for %s (id=%s): %s", role, str(det_id) if det_id is not None else "?", exc)
			else:
				rospy.logwarn("Teleport failed: %s", exc)
			return False
		finally:
			if disabled and hasattr(actor, "set_simulate_physics"):
				try:
					actor.set_simulate_physics(True)
				except Exception as exc:
					rospy.logwarn_throttle(5.0, "bev_id_teleporter: failed to re-enable physics for %s: %s", role or actor.id, exc)

	def _tracking_status_tick(self, _evt) -> None:
		# Periodically republish latest known tracking_ok for each role
		for role, ok in list(self.role_tracking_ok.items()):
			try:
				pub = self.tracking_publishers.get(role)
				if pub is not None:
					pub.publish(Bool(data=ok))
			except Exception:
				continue

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

	def _get_waypoint_heading(self, x: float, y: float) -> Optional[Tuple[float, float]]:
		if self.carla_map is None:
			return None
		try:
			wp = self.carla_map.get_waypoint(
				carla.Location(x=float(x), y=float(y), z=0.5),
				project_to_road=True,
				lane_type=carla.LaneType.Driving,
			)
		except Exception:
			return None
		if wp is None:
			return None
		yaw_deg = float(wp.transform.rotation.yaw)
		return math.radians(yaw_deg), yaw_deg

	def _maybe_flip_yaw(self, x: float, y: float, yaw_rad: float) -> float:
		if not self.enable_yaw_flip:
			return yaw_rad
		wp_heading = self._get_waypoint_heading(x, y)
		if wp_heading is None:
			return yaw_rad
		lane_yaw_rad, lane_yaw_deg = wp_heading
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
				lane_yaw_deg,
			)
			return flipped
		return yaw_rad

	def _apply_waypoint_heading_guard(self, x: float, y: float, yaw_rad: float) -> float:
		if not self.enable_waypoint_heading_guard:
			return yaw_rad
		wp_heading = self._get_waypoint_heading(x, y)
		if wp_heading is None:
			return yaw_rad
		lane_yaw_rad, lane_yaw_deg = wp_heading
		window_deg = max(0.0, min(180.0, self.waypoint_heading_window_deg))
		if window_deg <= 0.0:
			if abs(normalize_angle(yaw_rad - lane_yaw_rad)) > math.radians(1.0):
				rospy.loginfo_throttle(
					1.0,
					"Waypoint yaw snap at (%.1f, %.1f): det %.1f° -> lane %.1f°",
					float(x),
					float(y),
					math.degrees(yaw_rad),
					lane_yaw_deg,
				)
			return lane_yaw_rad
		limit = math.radians(window_deg)
		delta = normalize_angle(yaw_rad - lane_yaw_rad)
		if abs(delta) <= limit:
			return yaw_rad
		clamped = lane_yaw_rad + (limit if delta > 0.0 else -limit)
		rospy.loginfo_throttle(
			1.0,
			"Waypoint yaw clamp at (%.1f, %.1f): det %.1f° -> %.1f° (lane %.1f°)",
			float(x),
			float(y),
			math.degrees(yaw_rad),
			math.degrees(clamped),
			lane_yaw_deg,
		)
		return normalize_angle(clamped)

	def _filter_yaw(self, det_id: int, yaw_rad: float) -> float:
		if not self.enable_yaw_filter:
			return yaw_rad
		now = float(rospy.Time.now().to_sec())
		prev = self._filtered_yaws.get(det_id)
		if prev is None:
			self._filtered_yaws[det_id] = yaw_rad
			self._filtered_yaw_stamp[det_id] = now
			return yaw_rad
		delta = normalize_angle(yaw_rad - prev)
		if self.yaw_filter_jump_deg > 0.0 and abs(math.degrees(delta)) > self.yaw_filter_jump_deg:
			filtered = prev
		else:
			alpha = max(0.0, min(1.0, self.yaw_filter_alpha))
			filtered = normalize_angle(prev + alpha * delta)
		self._filtered_yaws[det_id] = filtered
		self._filtered_yaw_stamp[det_id] = now
		return filtered

	def _cleanup_filtered_yaws(self) -> None:
		if not self.enable_yaw_filter or self.yaw_filter_timeout_sec <= 0.0:
			return
		now = float(rospy.Time.now().to_sec())
		timeout = max(0.0, self.yaw_filter_timeout_sec)
		for det_id, stamp in list(self._filtered_yaw_stamp.items()):
			if (now - stamp) >= timeout:
				self._filtered_yaws.pop(det_id, None)
				self._filtered_yaw_stamp.pop(det_id, None)

	def _bev_cb(self, msg: BEVInfo) -> None:
		# 현재 BEV 메시지의 타임스탬프를 그대로 추적 포즈에 사용
		# header.stamp 가 비어 있으면 now() 로 대체
		if msg.header and msg.header.stamp != rospy.Time():
			self.bev_time_stamp = msg.header.stamp
		else:
			self.bev_time_stamp = rospy.Time.now()

		# rospy.loginfo(f"@@@@@@@ : {(rospy.Time.now() - msg.header.stamp).to_sec()}")

		# Validate array lengths
		n = int(msg.detCounts)
		ids = list(msg.ids[:n])
		cxs = list(msg.center_xs[:n])
		cys = list(msg.center_ys[:n])
		yaws = list(msg.yaws[:n]) if len(msg.yaws) >= n else [0.0] * n
		raw_colors = list(getattr(msg, "colors", []))
		color_list = [raw_colors[idx] if idx < len(raw_colors) else "" for idx in range(n)]
		det_colors = [self._normalize_color(color_list[idx]) if idx < len(color_list) else "" for idx in range(n)]

		if not ids:
			return

		initial_alignment_active = self._initial_alignment_active()
		dist_gate_m = self.dist_gate_m
		yaw_gate_deg = self.yaw_gate_deg

		if initial_alignment_active:
			if self.initial_alignment_dist_gate_m < 0.0:
				pass
			elif self.initial_alignment_dist_gate_m == 0.0:
				dist_gate_m = 0.0
			else:
				dist_gate_m = self.initial_alignment_dist_gate_m
			if self.initial_alignment_yaw_gate_deg < 0.0:
				pass
			elif self.initial_alignment_yaw_gate_deg == 0.0:
				yaw_gate_deg = 0.0
			else:
				yaw_gate_deg = self.initial_alignment_yaw_gate_deg

		max_teleport_distance = self.max_teleport_distance_m
		teleport_yaw_gate_deg = self.teleport_yaw_gate_deg

		if initial_alignment_active:
			if self.initial_alignment_max_teleport_m < 0.0:
				pass
			elif self.initial_alignment_max_teleport_m == 0.0:
				max_teleport_distance = 0.0
			else:
				max_teleport_distance = self.initial_alignment_max_teleport_m
			if self.initial_alignment_teleport_yaw_gate_deg < 0.0:
				pass
			elif self.initial_alignment_teleport_yaw_gate_deg == 0.0:
				teleport_yaw_gate_deg = 0.0
			else:
				teleport_yaw_gate_deg = self.initial_alignment_teleport_yaw_gate_deg

		skip_warmup = initial_alignment_active and self.initial_alignment_skip_warmup

		tracking_ok_map = {role: False for role in self.role_names}
		latest_detection: Dict[str, Tuple[float, float, float]] = {}
		det_yaws_rad: List[float] = []
		yaw_corrections_active = self._yaw_corrections_active()

		for idx in range(len(ids)):
			raw_yaw = float(yaws[idx]) if idx < len(yaws) else 0.0
			yaw_rad = self._input_yaw_to_rad(raw_yaw)
			x = float(cxs[idx])
			y = float(cys[idx])
			if yaw_corrections_active:
				yaw_rad = self._maybe_flip_yaw(x, y, yaw_rad)
				yaw_rad = self._filter_yaw(int(ids[idx]), yaw_rad)
				yaw_rad = self._apply_waypoint_heading_guard(x, y, yaw_rad)
			det_yaws_rad.append(yaw_rad)

		if yaw_corrections_active:
			self._cleanup_filtered_yaws()
		else:
			if self._filtered_yaws:
				self._filtered_yaws.clear()
				self._filtered_yaw_stamp.clear()

		now_sec = float(rospy.Time.now().to_sec())
		
		if (
			initial_alignment_active
			and not self._initial_alignment_seed_performed
			and self.initial_alignment_avg_duration_sec > 0.0
		):
			self._accumulate_initial_alignment_samples(ids, cxs, cys, det_yaws_rad, det_colors)
			if not self._initial_alignment_avg_ready and (now_sec - self._initial_alignment_start_sec) >= self.initial_alignment_avg_duration_sec:
				self._prepare_initial_alignment_avg_results()
			if not self._initial_alignment_avg_ready:
				return
			if self._apply_initial_alignment_seed():
				return

		if self.enable_matching:
			# Hungarian-based robust assignment
			roles = list(self.role_names)[: self.max_vehicle_count]
			if roles and ids:
				costs = self._compute_cost_matrix(roles, ids, cxs, cys, det_yaws_rad, det_colors, dist_gate_m, yaw_gate_deg)
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
				x_raw = float(cxs[idx])
				y_raw = float(cys[idx])
				yaw_rad = det_yaws_rad[idx] if idx < len(det_yaws_rad) else 0.0
				x, y, fused_yaw_rad, _ = self._kf_fuse_measurement(role, x_raw, y_raw, yaw_rad, now_sec)
				# Optional forward prediction for teleport target (예: 1초 뒤 위치)
				if self.kf_enabled and self.kf_prediction_horizon_sec > 0.0:
					ahead = self._kf_predict_ahead_state(role, self.kf_prediction_horizon_sec)
					if ahead is not None:
						x, y, fused_yaw_rad = ahead
				# Motion-based yaw estimation when Kalman filter is off
				if not self.kf_enabled and self.motion_yaw_enabled:
					motion_yaw_rad = self._compute_yaw_from_motion(role, x, y, fused_yaw_rad, now_sec)
					if motion_yaw_rad is not None:
						fused_yaw_rad = motion_yaw_rad
				yaw_deg = math.degrees(fused_yaw_rad)
				if self.yaw_blend_enabled:
					yaw_deg = self._blend_yaw_sources(role, x, y, yaw_deg)
				pose_snap = self.snap_to_spawn_pose_initial and role not in self._initial_alignment_done_roles
				x, y, yaw_deg = self._apply_spawn_snap(role, x, y, yaw_deg, pose_snap)
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
				if max_teleport_distance > 0.0 and dist > max_teleport_distance:
					rospy.logwarn_throttle(1.0, "Teleport skip for %s: jump %.1fm > %.1fm", role, dist, max_teleport_distance)
					continue
				# Optional yaw gate at teleport stage
				dyaw = abs((yaw_deg - curr_yaw_deg + 180.0) % 360.0 - 180.0)
				if teleport_yaw_gate_deg > 0.0 and dyaw > teleport_yaw_gate_deg:
					rospy.logwarn_throttle(1.0, "Teleport skip for %s: yaw delta %.1f° > %.1f°", role, dyaw, teleport_yaw_gate_deg)
					continue
				# Stability warmup after mapping switches
				now_sec = float(rospy.Time.now().to_sec())
				last_sw = self.role_last_switch.get(role, now_sec)
				warmup_horizon = 0.0 if skip_warmup else max(0.0, self.teleport_stability_warmup_sec)
				if (now_sec - last_sw) < warmup_horizon:
					rospy.loginfo_throttle(1.0, "Teleport hold for %s: stabilizing mapping (%.2fs)", role, now_sec - last_sw)
					continue
				if role in tracking_ok_map:
					tracking_ok_map[role] = False
				z = self._pick_height(x, y)
				location = carla.Location(x=x, y=y, z=z)
				rotation = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)
				tf = carla.Transform(location=location, rotation=rotation)
				if self._apply_transform(actor, tf, role=role, det_id=vid):
					if role in tracking_ok_map:
						tracking_ok_map[role] = True
						latest_detection[role] = (x, y, fused_yaw_rad)
						if initial_alignment_active:
							self._mark_initial_alignment_done(role)
						self._set_role_initialized(role)
						self._set_role_initialized(role)
		else:
			# Fallback to simple first-come mapping
			assignments = self._assign_roles_for_ids(ids, det_colors)
			if not assignments:
				return
			for (vid, role) in assignments:
				try:
					idx = ids.index(vid)
				except ValueError:
					continue
				x_raw = float(cxs[idx])
				y_raw = float(cys[idx])
				yaw_rad = det_yaws_rad[idx] if idx < len(det_yaws_rad) else 0.0
				x, y, fused_yaw_rad, _ = self._kf_fuse_measurement(role, x_raw, y_raw, yaw_rad, now_sec)
				# Optional forward prediction for teleport target
				if self.kf_enabled and self.kf_prediction_horizon_sec > 0.0:
					ahead = self._kf_predict_ahead_state(role, self.kf_prediction_horizon_sec)
					if ahead is not None:
						x, y, fused_yaw_rad = ahead
				# Motion-based yaw estimation when Kalman filter is off
				if not self.kf_enabled and self.motion_yaw_enabled:
					motion_yaw_rad = self._compute_yaw_from_motion(role, x, y, fused_yaw_rad, now_sec)
					if motion_yaw_rad is not None:
						fused_yaw_rad = motion_yaw_rad
				yaw_deg = math.degrees(fused_yaw_rad)
				if self.yaw_blend_enabled:
					yaw_deg = self._blend_yaw_sources(role, x, y, yaw_deg)
				pose_snap = self.snap_to_spawn_pose_initial and role not in self._initial_alignment_done_roles
				x, y, yaw_deg = self._apply_spawn_snap(role, x, y, yaw_deg, pose_snap)
				actor = self._resolve_actor_for_role(role)
				if actor is None:
					continue
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
				if max_teleport_distance > 0.0 and dist > max_teleport_distance:
					rospy.logwarn_throttle(1.0, "Teleport skip for %s: jump %.1fm > %.1fm", role, dist, max_teleport_distance)
					continue
				dyaw = abs((yaw_deg - curr_yaw_deg + 180.0) % 360.0 - 180.0)
				if teleport_yaw_gate_deg > 0.0 and dyaw > teleport_yaw_gate_deg:
					rospy.logwarn_throttle(1.0, "Teleport skip for %s: yaw delta %.1f° > %.1f°", role, dyaw, teleport_yaw_gate_deg)
					continue
				now_sec = float(rospy.Time.now().to_sec())
				last_sw = self.role_last_switch.get(role, now_sec)
				warmup_horizon = 0.0 if skip_warmup else max(0.0, self.teleport_stability_warmup_sec)
				if (now_sec - last_sw) < warmup_horizon:
					rospy.loginfo_throttle(1.0, "Teleport hold for %s: stabilizing mapping (%.2fs)", role, now_sec - last_sw)
					continue
				if role in tracking_ok_map:
					tracking_ok_map[role] = False
				z = self._pick_height(x, y)
				location = carla.Location(x=x, y=y, z=z)
				rotation = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)
				tf = carla.Transform(location=location, rotation=rotation)
				if self._apply_transform(actor, tf, role=role, det_id=vid):
					if role in tracking_ok_map:
						tracking_ok_map[role] = True
						latest_detection[role] = (x, y, fused_yaw_rad)
						if initial_alignment_active:
							self._mark_initial_alignment_done(role)

		for role, ok in tracking_ok_map.items():
			self._set_tracking_status(role, ok)
			if ok:
				data = latest_detection.get(role)
				if data:
					self._publish_tracking_pose(role, data[0], data[1], data[2])


if __name__ == "__main__":
	try:
		BevIdTeleporter()
		rospy.spin()
	except rospy.ROSInterruptException:
		pass


