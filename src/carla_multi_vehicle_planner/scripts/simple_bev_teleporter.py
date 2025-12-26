#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple

import rospy
from capstone_msgs.msg import BEVInfo
from nav_msgs.msg import Path

try:
	# side-effect: insert CARLA API path to sys.path if available
	from setup_carla_path import *  # noqa: F401,F403
except Exception:
	pass

try:
	import carla  # type: ignore
except Exception as exc:
	carla = None  # type: ignore
	rospy.logfatal("Failed to import CARLA Python API: %s", exc)


class SimpleBevTeleporter:
	def __init__(self) -> None:
		rospy.init_node("bev_id_teleporter", anonymous=True)
		if carla is None:
			raise RuntimeError("CARLA Python API unavailable")

		# minimal parameters (default to /bev_info_raw to match inference_receiver)
		self.topic = rospy.get_param("~topic", "/bev_info_raw")
		self.host = rospy.get_param("~carla_host", "localhost")
		self.port = int(rospy.get_param("~carla_port", 2000))
		self.max_vehicle_count = max(1, min(6, int(rospy.get_param("~max_vehicle_count", 1))))
		self.default_z = float(rospy.get_param("~default_z", -1.5))
		self.yaw_in_degrees = bool(rospy.get_param("~yaw_in_degrees", False))
		# BEV 지연 보상용 경로 기반 lookahead (m)
		self.lookahead_m = float(rospy.get_param("~lookahead_m", 1.0))
		# ENU(CCW) → CARLA(CW) 변환이 필요하면 True로 설정
		self._last_stamp = None
		self._last_seq = None
		self.disable_collisions = bool(rospy.get_param("~disable_collisions", True))
		self._collision_disabled: Dict[str, bool] = {}

		self.client = carla.Client(self.host, self.port)
		self.client.set_timeout(5.0)
		self.world = self.client.get_world()

		self.role_names: List[str] = [f"ego_vehicle_{i}" for i in range(1, self.max_vehicle_count + 1)]
		self.role_to_actor: Dict[str, carla.Actor] = {}
		self._refresh_role_actors()
		# 경로 캐시 (lookahead 적용용)
		self._paths: Dict[str, List[Tuple[float, float]]] = {}
		self._s_profiles: Dict[str, List[float]] = {}

		# 최신 메시지만 사용하도록 queue_size=1, tcp_nodelay
		rospy.Subscriber(self.topic, BEVInfo, self._bev_cb, queue_size=1, tcp_nodelay=True)
		# planned_path_* 구독
		for idx in range(self.max_vehicle_count):
			role = self.role_names[idx]
			topic = f"/planned_path_{role}"
			rospy.Subscriber(topic, Path, self._path_cb, callback_args=role, queue_size=1)
		rospy.loginfo("SimpleBevTeleporter: listening %s for up to %d roles (%s)",
		              self.topic, self.max_vehicle_count, ",".join(self.role_names))

	def _refresh_role_actors(self) -> None:
		try:
			actors = self.world.get_actors().filter("vehicle.*")
		except Exception as exc:
			rospy.logwarn("SimpleBevTeleporter: failed to fetch actors: %s", exc)
			return
		m: Dict[str, carla.Actor] = {}
		for a in actors:
			try:
				role = a.attributes.get("role_name", "")
			except Exception:
				continue
			if role in self.role_names:
				m[role] = a
		self.role_to_actor = m

	def _get_actor(self, role: str):
		actor = self.role_to_actor.get(role)
		if actor is not None and actor.is_alive:
			return actor
		self._refresh_role_actors()
		return self.role_to_actor.get(role)

	def _path_cb(self, msg: Path, role: str) -> None:
		pts: List[Tuple[float, float]] = []
		for pose in msg.poses:
			pts.append((float(pose.pose.position.x), float(pose.pose.position.y)))
		self._paths[role] = pts
		# s-profile 계산
		s_profile: List[float] = [0.0]
		total = 0.0
		for i in range(1, len(pts)):
			step = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
			total += step
			s_profile.append(total)
		self._s_profiles[role] = s_profile

	def _project_on_path(self, role: str, x: float, y: float):
		pts = self._paths.get(role)
		s_profile = self._s_profiles.get(role)
		if not pts or not s_profile or len(pts) != len(s_profile) or len(pts) < 2:
			return None
		best_dist_sq = float("inf")
		best_idx = None
		best_t = 0.0
		best_proj = (x, y)
		for idx in range(len(pts) - 1):
			x1, y1 = pts[idx]
			x2, y2 = pts[idx + 1]
			dx = x2 - x1
			dy = y2 - y1
			seg_len_sq = dx * dx + dy * dy
			if seg_len_sq < 1e-6:
				continue
			t = ((x - x1) * dx + (y - y1) * dy) / seg_len_sq
			t = max(0.0, min(1.0, t))
			px = x1 + dx * t
			py = y1 + dy * t
			d2 = (px - x) * (px - x) + (py - y) * (py - y)
			if d2 < best_dist_sq:
				best_dist_sq = d2
				best_idx = idx
				best_t = t
				best_proj = (px, py)
		if best_idx is None:
			return None
		seg_len = math.hypot(pts[best_idx + 1][0] - pts[best_idx][0], pts[best_idx + 1][1] - pts[best_idx][1])
		if seg_len < 1e-6:
			s_now = s_profile[best_idx]
		else:
			s_now = s_profile[best_idx] + best_t * seg_len
		heading = math.atan2(pts[best_idx + 1][1] - pts[best_idx][1], pts[best_idx + 1][0] - pts[best_idx][0])
		return s_now, s_profile, pts, best_proj[0], best_proj[1], heading

	def _sample_at_s(self, pts: List[Tuple[float, float]], s_profile: List[float], s_target: float):
		if len(pts) < 2 or not s_profile or len(s_profile) != len(pts):
			return None
		if s_target <= s_profile[0]:
			return pts[0][0], pts[0][1], 0
		if s_target >= s_profile[-1]:
			return pts[-1][0], pts[-1][1], len(pts) - 1
		for i in range(len(s_profile) - 1):
			if s_profile[i + 1] >= s_target:
				ds = max(1e-6, s_profile[i + 1] - s_profile[i])
				t = (s_target - s_profile[i]) / ds
				x1, y1 = pts[i]
				x2, y2 = pts[i + 1]
				tx = x1 + (x2 - x1) * t
				ty = y1 + (y2 - y1) * t
				return tx, ty, i
		return None

	def _bev_cb(self, msg: BEVInfo) -> None:
		# 여러 대 지원: 메시지 길이와 role_names 수 중 작은 값만큼 처리
		n = min(len(self.role_names), len(msg.center_xs), len(msg.center_ys), len(msg.yaws))
		if n <= 0:
			return
		if len(msg.center_xs) < n or len(msg.center_ys) < n or len(msg.yaws) < n:
			rospy.logwarn_throttle(
				1.0,
				"BEV message missing fields: detCounts=%d, lens=(%d,%d,%d)",
				msg.detCounts,
				len(msg.center_xs),
				len(msg.center_ys),
				len(msg.yaws),
			)
			return

		stamp = msg.header.stamp
		seq = msg.header.seq
		if self._last_seq is not None and seq <= self._last_seq:
			return
		self._last_seq = seq
		now = rospy.Time.now()
		age = (now - stamp).to_sec()
		if age > 0.3:
			return
		if self._last_stamp is not None:
			if stamp == self._last_stamp:
				return
			dt = (stamp - self._last_stamp).to_sec()
			if dt <= 0:
				return
		self._last_stamp = stamp
		for i in range(n):
			role = self.role_names[i]
			actor = self._get_actor(role)
			if actor is None:
				rospy.logwarn_throttle(1.0, "teleport skip: actor %s not found", role)
				continue
			if self.disable_collisions and not self._collision_disabled.get(role, False):
				try:
					if hasattr(actor, "set_collision_enabled"):
						actor.set_collision_enabled(False)
					if hasattr(actor, "set_simulate_physics"):
						actor.set_simulate_physics(False)
					self._collision_disabled[role] = True
				except Exception:
					pass
			try:
				x = float(msg.center_xs[i])
				y = float(msg.center_ys[i])
				raw_yaw = float(msg.yaws[i])
				yaw_rad = math.radians(raw_yaw) if self.yaw_in_degrees else raw_yaw

				# 경로 기반 lookahead 보정: path 투영 후 s+lookahead
				projected = self._project_on_path(role, x, y)
				tx, ty, yaw_for_pose = x, y, yaw_rad
				# if projected is not None:
				# 	s_now, s_profile, pts, px, py, heading_proj = projected
				# 	# BEV -> path 위치 편차(횡방향)를 유지하기 위해 lateral 오프셋 계산
				# 	dx_off = x - px
				# 	dy_off = y - py
				# 	nx = -math.sin(heading_proj)
				# 	ny = math.cos(heading_proj)
				# 	lat_off = dx_off * nx + dy_off * ny
				# 	s_target = s_now + max(0.0, self.lookahead_m)
				# 	sample = self._sample_at_s(pts, s_profile, s_target)
				# 	if sample is not None:
				# 		bx, by, seg_idx = sample
				# 		# 목표 지점에서 동일한 lateral offset 적용
				# 		tx = bx + nx * lat_off
				# 		ty = by + ny * lat_off

				yaw_deg = math.degrees(yaw_for_pose)
				location = carla.Location(x=tx, y=ty, z=self.default_z)
				rotation = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)
				tf = carla.Transform(location=location, rotation=rotation)
				actor.set_transform(tf)
			except Exception as exc:
				rospy.logwarn_throttle(1.0, "SimpleBevTeleporter: teleport failed for %s: %s", role, exc)


if __name__ == "__main__":
	try:
		SimpleBevTeleporter()
		rospy.spin()
	except rospy.ROSInterruptException:
		pass


