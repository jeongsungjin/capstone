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

		# minimal parameters (default to /bev_info_raw to match inference_receiver)
		self.topic = rospy.get_param("~topic", "/bev_info_raw")
		self.host = rospy.get_param("~carla_host", "localhost")
		self.port = int(rospy.get_param("~carla_port", 2000))
		self.max_vehicle_count = max(1, min(6, int(rospy.get_param("~max_vehicle_count", 1))))
		self.default_z = float(rospy.get_param("~default_z", 0.3))
		self.yaw_in_degrees = bool(rospy.get_param("~yaw_in_degrees", False))
		# ENU(CCW) → CARLA(CW) 변환이 필요하면 True로 설정
		self.yaw_flip_sign = bool(rospy.get_param("~yaw_flip_sign", False))
		self._last_stamp = None
		self._last_seq = None

		self.client = carla.Client(self.host, self.port)
		self.client.set_timeout(5.0)
		self.world = self.client.get_world()

		self.role_names: List[str] = [f"ego_vehicle_{i}" for i in range(1, self.max_vehicle_count + 1)]
		self.role_to_actor: Dict[str, carla.Actor] = {}
		self._refresh_role_actors()

		# 최신 메시지만 사용하도록 queue_size=1, tcp_nodelay
		rospy.Subscriber(self.topic, BEVInfo, self._bev_cb, queue_size=1, tcp_nodelay=True)
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
		# 강제로 0번 인덱스만 사용 (단일 차량 텔레포트)
		n = 1
		if n <= 0:
			return
		if len(msg.center_xs) < 1 or len(msg.center_ys) < 1 or len(msg.yaws) < 1:
			rospy.logwarn_throttle(1.0, "BEV message missing fields: detCounts=%d, lens=(%d,%d,%d)",
			                       msg.detCounts, len(msg.center_xs), len(msg.center_ys), len(msg.yaws))
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
			try:
				idx = 0  # 강제로 첫 번째 검출만 사용
				x = float(msg.center_xs[idx])
				y = float(msg.center_ys[idx])
				raw_yaw = float(msg.yaws[idx])
				yaw_rad = math.radians(raw_yaw) if self.yaw_in_degrees else raw_yaw
				yaw_deg = math.degrees(yaw_rad)
				if self.yaw_flip_sign:
					yaw_deg = -yaw_deg
				while yaw_deg > 180.0:
					yaw_deg -= 360.0
				while yaw_deg < -180.0:
					yaw_deg += 360.0
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


