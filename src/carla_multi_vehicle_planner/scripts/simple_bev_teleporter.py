#!/usr/bin/env python3

import math
from typing import Dict, List

import rospy
from capstone_msgs.msg import BEVInfo

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

		# minimal parameters
		self.topic = rospy.get_param("~topic", "/bev_info")
		self.host = rospy.get_param("~carla_host", "localhost")
		self.port = int(rospy.get_param("~carla_port", 2000))
		self.max_vehicle_count = max(1, min(6, int(rospy.get_param("~max_vehicle_count", 1))))
		self.default_z = float(rospy.get_param("~default_z", 0.5))
		self.yaw_in_degrees = bool(rospy.get_param("~yaw_in_degrees", False))

		self.client = carla.Client(self.host, self.port)
		self.client.set_timeout(5.0)
		self.world = self.client.get_world()

		self.role_names: List[str] = [f"ego_vehicle_{i}" for i in range(1, self.max_vehicle_count + 1)]
		self.role_to_actor: Dict[str, carla.Actor] = {}
		self._refresh_role_actors()

		rospy.Subscriber(self.topic, BEVInfo, self._bev_cb, queue_size=1)
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

	def _bev_cb(self, msg: BEVInfo) -> None:
		n = int(min(self.max_vehicle_count, msg.detCounts))
		if n <= 0:
			return
		if len(msg.center_xs) < n or len(msg.center_ys) < n or len(msg.yaws) < n:
			return
		for i in range(n):
			role = self.role_names[i]
			actor = self._get_actor(role)
			if actor is None:
				continue
			try:
				x = float(msg.center_xs[i])
				y = float(msg.center_ys[i])
				raw_yaw = float(msg.yaws[i])
				yaw_rad = math.radians(raw_yaw) if self.yaw_in_degrees else raw_yaw
				yaw_deg = math.degrees(yaw_rad)
				location = carla.Location(x=x, y=y, z=self.default_z)
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


