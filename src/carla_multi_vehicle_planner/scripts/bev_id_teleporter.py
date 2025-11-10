#!/usr/bin/env python3

import math
import threading
from typing import Dict, List, Optional, Tuple

import rospy

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

		# Role name handling
		role_names_csv: str = str(rospy.get_param("~role_names", "")).strip()
		if role_names_csv:
			self.role_names: List[str] = [s.strip() for s in role_names_csv.split(",") if s.strip()]
		else:
			self.role_names = [f"ego_vehicle_{i}" for i in range(1, self.max_vehicle_count + 1)]

		# Yaw interpretation
		self.yaw_in_degrees: bool = bool(rospy.get_param("~yaw_in_degrees", False))

		# Height / waypoint snapping
		self.snap_to_waypoint_height: bool = bool(rospy.get_param("~snap_to_waypoint_height", True))
		self.default_z: float = float(rospy.get_param("~default_z", 0.5))
		self.z_offset: float = float(rospy.get_param("~z_offset", 0.0))

		# Connect to CARLA
		self.client = carla.Client(self.host, self.port)
		self.client.set_timeout(5.0)
		self.world = self.client.get_world()
		self.carla_map = self.world.get_map()

		# State
		self._lock = threading.Lock()
		self.role_to_actor: Dict[str, carla.Actor] = {}
		self.id_to_role: Dict[int, str] = {}

		self._refresh_role_to_actor()

		# ROS I/O
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

	def _resolve_actor_for_role(self, role: str) -> Optional[carla.Actor]:
		with self._lock:
			actor = self.role_to_actor.get(role)
		if actor is not None and actor.is_alive:
			return actor
		# Try to refresh cache once
		self._refresh_role_to_actor()
		with self._lock:
			return self.role_to_actor.get(role)

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

	def _bev_cb(self, msg: BEVInfo) -> None:
		# Validate array lengths
		n = int(msg.detCounts)
		ids = list(msg.ids[:n])
		cxs = list(msg.center_xs[:n])
		cys = list(msg.center_ys[:n])
		yaws = list(msg.yaws[:n]) if len(msg.yaws) >= n else [0.0] * n

		if not ids:
			return

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
			yaw_in = float(yaws[idx]) if idx < len(yaws) else 0.0
			yaw_deg = yaw_in if self.yaw_in_degrees else deg(yaw_in)

			actor = self._resolve_actor_for_role(role)
			if actor is None:
				continue

			# Compute transform
			z = self._pick_height(x, y)
			location = carla.Location(x=x, y=y, z=z)
			rotation = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)
			tf = carla.Transform(location=location, rotation=rotation)

			try:
				actor.set_transform(tf)
			except Exception as exc:
				rospy.logwarn("Teleport failed for %s (id=%s): %s", role, str(vid), exc)


if __name__ == "__main__":
	try:
		BevIdTeleporter()
		rospy.spin()
	except rospy.ROSInterruptException:
		pass


