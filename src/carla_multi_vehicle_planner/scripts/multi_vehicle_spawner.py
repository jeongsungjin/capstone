#!/usr/bin/env python3

import math
import random
import sys
import os
import time
from typing import List, Optional, Set

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

# Prefer centralized CARLA path setup if available
try:
    from setup_carla_path import CARLA_BUILD_PATH  # noqa: F401
except Exception:
    _env = os.environ.get("CARLA_PYTHON_PATH")
    _default = os.path.expanduser("~/carla/PythonAPI/carla/build/lib.linux-x86_64-cpython-38")
    CARLA_BUILD_PATH = _env if _env else _default
if CARLA_BUILD_PATH and CARLA_BUILD_PATH not in sys.path:
    sys.path.insert(0, CARLA_BUILD_PATH)

try:
    import carla
except ImportError as exc:
    rospy.logfatal(f"CARLA import failed: {exc}")
    carla = None


DEFAULT_MODEL_MAP: List[str] = [
    "vehicle.vehicle.yellowxycar",   # ego_vehicle_1
    "vehicle.vehicle.greenxycar",     # ego_vehicle_2
    "vehicle.vehicle.redxycar",  # ego_vehicle_3
    "vehicle.vehicle.whitexycar",  # ego_vehicle_4
    "vehicle.vehicle.pinkxycar",    # ego_vehicle_5
    "vehicle.vehicle.purplexycar",   # ego_vehicle_6
]
# 이슈 1 특정 차량 전역 경로 게획 제한
# 추가적으로 특정 차량은 y좌표가 +10 이상인 영역으로는 경로 계획 및 주행을 안했으면 좋겠음 가능할지?
# 이유는 맵에 언덕 구간이 있는데, 특정 차량들은 모터 기능 이슈로 언덕을 등판하지 못함.
# 그렇다면 ui에 언덕 등판 못하는 차량을 선택했을때는 언덕을 꼭 지나가야만하는 위치는 선택 못하게끔 표시하는 기능도 추가해야겠군 (이건 ui적 문제니 동의형에게 맡겨야겠다)

# 이슈 2 인지 및 차량 운용 다양성 
# 젤 문제가 서버 인지 정보가 색상이 안나옴. 수동으로 꼭 매칭 해야하는가?
# 색상을 좀 잡는다면? 초기 헝가리안 매칭 되고, 인지가 조금 나아지는다는 가정하에 매칭 풀릴 이슈는 없을거임
# 매칭이 안풀린다면 각 차량의 위치 바뀜 없을것이고, 차량별 ip매핑이 안풀릴것임. 논리적 모순이나 리스크 포인트가 있는지?
# 현재는 carla vehicle 번호랑 색상, 아이피가 고정 매핑 되어잇어서 유동적으로 노랑, 퍼플 2대만 굴린다거나 그린, 레드, 퍼플 3대만 굴리는 시나리오를 실행하려면 하려면 많은 라인의 코드 수정이 필요
# 런치할때마다 운용 차량 대수, 사용 차량 색상 문자열 순서대로 입력하면 그렇게 스폰 및 아이피 할당하게끔 만들어야할듯
# 아이피와 색상은 udp_akermann_senders.launch에서 매핑되는 것으로 이해함

# 

# 차량번호별 스폰 차량 모델, ip, 색상 할당.
# 번호랑 색상이 꼭 맞아야 즉, 
# DEFAULT_MODEL_MAP: List[str] = [
#     "vehicle.vehicle.redxycar",   # ego_vehicle_1
#     "vehicle.vehicle.purplexycar",     # ego_vehicle_2
#     "vehicle.vehicle.pinkxycar",  # ego_vehicle_3
#     "vehicle.vehicle.whitexycar",  # ego_vehicle_4
#     "vehicle.vehicle.greenxycar",    # ego_vehicle_5
#     "vehicle.vehicle.yellowxycar",   # ego_vehicle_6
# ]


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


def pick_spawn_transform(world, seed=None, retry_limit=20):
    """Yield randomized spawn transforms pulled from map spawn points."""
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return

    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed(time.time())

    indices = list(range(len(spawn_points)))
    rng.shuffle(indices)
    attempt_limit = len(indices) if retry_limit is None or retry_limit <= 0 else min(retry_limit, len(indices))

    for attempt_index in range(attempt_limit):
        idx = indices[attempt_index]
        yield spawn_points[idx], idx


class MultiVehicleSpawner:
    def __init__(self):
        rospy.init_node("multi_vehicle_spawner", anonymous=True)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.num_vehicles = rospy.get_param("~num_vehicles", 3)
        self.vehicle_model = rospy.get_param("~vehicle_model", "vehicle.vehicle.coloredxycar")
        self.enable_autopilot = rospy.get_param("~enable_autopilot", False)
        self.spawn_delay = rospy.get_param("~spawn_delay", 0.5)
        self.target_speed = rospy.get_param("~target_speed", 8.0)
        self.randomize_spawn = bool(rospy.get_param("~randomize_spawn", False))
        self.spawn_seed = rospy.get_param("~spawn_seed", None)
        self.spawn_retry_limit = int(rospy.get_param("~spawn_retry_limit", 20))
        self.spawn_min_separation_m = float(rospy.get_param("~spawn_min_separation_m", 8.0))
        self.spawn_z_offset_m = float(rospy.get_param("~spawn_z_offset_m", 0.5))
        self.use_waypoint_spawn_fallback = bool(rospy.get_param("~use_waypoint_spawn_fallback", True))
        self.waypoint_spawn_spacing_m = float(rospy.get_param("~waypoint_spawn_spacing_m", 8.0))
        self.vehicle_models_param = str(rospy.get_param("~vehicle_models", "")).strip()
        self.spawn_indices_param = str(rospy.get_param("~spawn_indices", "")).strip()
        self.align_spawn_heading = bool(rospy.get_param("~align_spawn_heading", True))
        self.spawn_heading_offset_deg = float(rospy.get_param("~spawn_heading_offset_deg", 0.0))

        # Optional platoon spawn alignment
        self.platoon_enable = bool(rospy.get_param("~platoon_enable", False))
        self.platoon_leader = str(rospy.get_param("~platoon_leader", "ego_vehicle_1"))
        followers_str = str(rospy.get_param("~platoon_followers", "")).strip()
        self.platoon_followers = [s.strip() for s in followers_str.split(",") if s.strip()] if followers_str else []
        self.platoon_gap_m = float(rospy.get_param("~platoon_gap_m", 9.0))

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        # Initialize Traffic Manager only if autopilot is enabled to avoid
        # triggering TM map build on custom maps that may crash.
        self.traffic_manager = None
        if self.enable_autopilot:
            try:
                self.traffic_manager = self.client.get_trafficmanager()
            except Exception as exc:
                rospy.logwarn("Traffic Manager init failed: %s; continuing without autopilot", exc)
                self.enable_autopilot = False
        self.carla_map = self.world.get_map()
        self.spawn_points = self.carla_map.get_spawn_points()
        # Fallback: if map provides too few spawn points (custom maps), synthesize from waypoints
        if (not self.spawn_points or len(self.spawn_points) < max(2, self.num_vehicles)) and self.use_waypoint_spawn_fallback:
            try:
                self._augment_spawn_points_with_waypoints()
                rospy.loginfo(
                    "multi_vehicle_spawner: waypoint fallback used, total spawn candidates=%d",
                    len(self.spawn_points),
                )
            except Exception as exc:
                rospy.logwarn("multi_vehicle_spawner: waypoint fallback failed: %s", exc)
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in CARLA map (and waypoint fallback failed)")

        seed_value = int(self.spawn_seed) if self.spawn_seed is not None else None
        initial_seed = seed_value if seed_value is not None else time.time()
        self._spawn_rng = random.Random(initial_seed)
        self.fixed_spawn_indices: List[int] = self._parse_spawn_indices(self.spawn_indices_param)
        self._used_spawn_indices: Set[int] = set()

        self.initial_pose_pub = rospy.Publisher("/initialpose", PoseStamped, queue_size=1, latch=True)

        self.vehicles = []
        self.odom_publishers = {}
        self.spawned_transforms = {}

        self._autopilot_pending = []
        self.autopilot_post_delay = float(rospy.get_param("~autopilot_enable_post_delay_sec", 0.8))
        self.spawn_vehicles()
        if self.platoon_enable:
            try:
                self._align_platoon_after_spawn()
            except Exception as exc:
                rospy.logwarn("platoon alignment after spawn failed: %s", exc)
        if self.enable_autopilot and self._autopilot_pending:
            rospy.Timer(rospy.Duration(self.autopilot_post_delay), self._enable_autopilot_for_pending, oneshot=True)
        rospy.on_shutdown(self.cleanup)
        rospy.Timer(rospy.Duration(0.1), self.publish_odometry)

    def _parse_spawn_indices(self, csv: str) -> List[int]:
        if not csv:
            return []
        result: List[int] = []
        for token in csv.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                idx = int(token)
                if idx >= 0:
                    result.append(idx)
            except ValueError:
                rospy.logwarn("multi_vehicle_spawner: invalid spawn index '%s' ignored", token)
        return result

    def _build_model_map(self) -> List[str]:
        if self.vehicle_models_param:
            models = [m.strip() for m in self.vehicle_models_param.split(",") if m.strip()]
            if models:
                return models
        return list(DEFAULT_MODEL_MAP)

    def _iter_candidate_transforms(self, seed_hint: Optional[float]):
        if self.randomize_spawn:
            yield from pick_spawn_transform(self.world, seed_hint, self.spawn_retry_limit)
        else:
            yield from self._iter_fixed_spawn_transforms()

    def _iter_fixed_spawn_transforms(self):
        if not self.spawn_points:
            return
        ordered = self.fixed_spawn_indices if self.fixed_spawn_indices else list(range(len(self.spawn_points)))
        for idx in ordered:
            if idx in self._used_spawn_indices:
                continue
            if idx < 0 or idx >= len(self.spawn_points):
                continue
            yield self.spawn_points[idx], idx

    def _consume_spawn_index(self, spawn_index: Optional[int]) -> None:
        if not self.randomize_spawn and spawn_index is not None:
            self._used_spawn_indices.add(spawn_index)

    def _align_heading_to_lane(self, transform: carla.Transform) -> carla.Transform:
        if not self.align_spawn_heading or self.carla_map is None:
            return transform
        try:
            waypoint = self.carla_map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "multi_vehicle_spawner: waypoint lookup failed: %s", exc)
            return transform
        if waypoint is None:
            return transform
        yaw = waypoint.transform.rotation.yaw + self.spawn_heading_offset_deg
        aligned_rotation = carla.Rotation(
            pitch=transform.rotation.pitch,
            roll=transform.rotation.roll,
            yaw=yaw,
        )
        return carla.Transform(transform.location, aligned_rotation)

    def _configure_tm_globals(self):
        tm = self.traffic_manager
        # Best-effort: guard for API availability across versions
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            tm.set_hybrid_physics_mode(self.tm_hybrid_physics)
        except Exception:
            pass
        try:
            tm.set_global_distance_to_leading_vehicle(self.tm_global_min_headway)
        except Exception:
            pass

    def spawn_vehicles(self):
        self._used_spawn_indices.clear()
        blueprint_library = self.world.get_blueprint_library()
        model_map = self._build_model_map()
        if self.enable_autopilot:
            self._configure_tm_globals()

        for index in range(self.num_vehicles):
            role_name = f"ego_vehicle_{index + 1}"
            # Pick model by index; fallback to configured default, then first available
            desired_model = model_map[index] if index < len(model_map) else "vehicle.vehicle.coloredxycar"
            blueprint = blueprint_library.find(desired_model)
            if blueprint is None:
                rospy.logwarn(
                    "%s: vehicle model %s not found; falling back to %s",
                    role_name,
                    desired_model,
                    "vehicle.vehicle.coloredxycar",
                )
                blueprint = blueprint_library.find("vehicle.vehicle.coloredxycar")
            if blueprint is None:
                bp_list = blueprint_library.filter("vehicle.*")
                if not bp_list:
                    rospy.logerr("%s: no vehicle blueprints available", role_name)
                    continue
                blueprint = bp_list[0]
            # Use a fresh blueprint instance per spawn (by id) and set role
            blueprint = blueprint_library.find(blueprint.id)
            if blueprint is not None and blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", role_name)

            vehicle = None
            chosen_transform = None
            chosen_index = None

            seed_hint = self._spawn_rng.random() if self.randomize_spawn else None

            for transform, spawn_index in self._iter_candidate_transforms(seed_hint):
                aligned_transform = self._align_heading_to_lane(transform)
                # Lift Z slightly to avoid immediate collisions with ground/props on custom maps
                try:
                    aligned_transform.location.z = float(aligned_transform.location.z) + float(self.spawn_z_offset_m)
                except Exception:
                    pass
                # Enforce minimum separation from previously spawned vehicles
                too_close = False
                for _role, (prev_tf, _idx) in self.spawned_transforms.items():
                    dx = aligned_transform.location.x - prev_tf.location.x
                    dy = aligned_transform.location.y - prev_tf.location.y
                    dz = aligned_transform.location.z - prev_tf.location.z
                    if math.sqrt(dx * dx + dy * dy + dz * dz) < max(0.0, self.spawn_min_separation_m):
                        too_close = True
                        break
                if too_close:
                    self._consume_spawn_index(spawn_index)
                    continue
                vehicle = self.world.try_spawn_actor(blueprint, aligned_transform)
                if vehicle is not None:
                    chosen_transform = aligned_transform
                    chosen_index = spawn_index
                    self._consume_spawn_index(spawn_index)
                    break
                self._consume_spawn_index(spawn_index)
                time.sleep(0.1)

            if vehicle is None:
                rospy.logerr(f"Failed to spawn vehicle {role_name} after {self.spawn_retry_limit} attempts")
                continue

            if self.enable_autopilot:
                # Defer enabling autopilot until after spawn loop finishes to avoid TM map-build races
                self._autopilot_pending.append(vehicle)

            self.vehicles.append(vehicle)
            topic = f"/carla/{role_name}/odometry"
            self.odom_publishers[role_name] = rospy.Publisher(topic, Odometry, queue_size=10)
            transform = vehicle.get_transform()
            if chosen_transform is None:
                chosen_transform = transform
            self.spawned_transforms[role_name] = (chosen_transform, chosen_index)
            spawn_idx_str = "unknown" if chosen_index is None else str(chosen_index)
            rospy.loginfo("%s: spawned at map spawn index %s", role_name, spawn_idx_str)
            self.publish_initial_pose(chosen_transform)
            time.sleep(self.spawn_delay)

    def _augment_spawn_points_with_waypoints(self) -> None:
        """
        Populate additional spawn candidates from map waypoints to support custom maps
        that provide few or no built-in spawn points.
        """
        if self.carla_map is None:
            return
        try:
            spacing = max(2.0, float(self.waypoint_spawn_spacing_m))
        except Exception:
            spacing = 8.0
        try:
            waypoints = self.carla_map.generate_waypoints(spacing)
        except Exception as exc:
            rospy.logwarn("generate_waypoints failed: %s", exc)
            return
        if not waypoints:
            return
        # Use a simple thinning by distance to avoid dense clusters
        candidates = list(self.spawn_points) if self.spawn_points else []
        for wp in waypoints:
            tf = wp.transform
            # Slight Z lift will be applied again during spawn, but keep here too so initial pose is sensible
            try:
                tf.location.z = float(tf.location.z) + float(self.spawn_z_offset_m)
            except Exception:
                pass
            if self._far_from_existing(tf, candidates, self.spawn_min_separation_m * 0.8):
                candidates.append(tf)
        self.spawn_points = candidates

    @staticmethod
    def _far_from_existing(tf: "carla.Transform", existing: List["carla.Transform"], min_dist: float) -> bool:
        if not existing:
            return True
        md = max(0.0, float(min_dist))
        x = float(tf.location.x)
        y = float(tf.location.y)
        z = float(tf.location.z)
        for etf in existing:
            dx = x - float(etf.location.x)
            dy = y - float(etf.location.y)
            dz = z - float(etf.location.z)
            if math.sqrt(dx * dx + dy * dy + dz * dz) < md:
                return False
        return True

    def _enable_autopilot_for_pending(self, _event):
        if not self._autopilot_pending:
            return
        tm_port = None
        if self.traffic_manager is not None:
            try:
                tm_port = self.traffic_manager.get_port()
            except Exception as exc:
                rospy.logwarn("Traffic Manager port read failed: %s", exc)
        for veh in list(self._autopilot_pending):
            try:
                if tm_port is not None:
                    veh.set_autopilot(True, tm_port)
                    try:
                        # Keep a conservative speed difference; CARLA expects percentage of limit
                        self.traffic_manager.vehicle_percentage_speed_difference(veh, 30)
                    except Exception as exc:
                        rospy.logwarn("TM speed config failed: %s", exc)
                else:
                    veh.set_autopilot(True)
            except Exception as exc:
                rospy.logwarn("set_autopilot failed: %s", exc)
        self._autopilot_pending.clear()

    def publish_odometry(self, _event):
        stamp = rospy.Time.now()
        for vehicle in self.vehicles:
            if vehicle is None or not vehicle.is_alive:
                continue
            role_name = vehicle.attributes.get("role_name", "")
            publisher = self.odom_publishers.get(role_name)
            if publisher is None:
                continue

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            angular_velocity = vehicle.get_angular_velocity()

            pose = Pose()
            pose.position.x = transform.location.x
            pose.position.y = transform.location.y
            pose.position.z = transform.location.z
            pose.orientation = euler_to_quaternion(
                math.radians(transform.rotation.roll),
                math.radians(transform.rotation.pitch),
                math.radians(transform.rotation.yaw),
            )

            twist = Twist()
            twist.linear = Vector3(velocity.x, velocity.y, velocity.z)
            twist.angular = Vector3(angular_velocity.x, angular_velocity.y, angular_velocity.z)

            odom = Odometry()
            odom.header = Header(stamp=stamp, frame_id="map")
            odom.child_frame_id = f"{role_name}/base_link"
            odom.pose.pose = pose
            odom.twist.twist = twist
            publisher.publish(odom)

    def cleanup(self):
        for vehicle in self.vehicles:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
        self.vehicles.clear()
        # Spectator control moved to cam_perspective.py

    def publish_initial_pose(self, transform):
        if self.initial_pose_pub is None:
            return
        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=rospy.Time.now(), frame_id="map")
        pose_msg.pose.position.x = transform.location.x
        pose_msg.pose.position.y = transform.location.y
        pose_msg.pose.position.z = transform.location.z
        pose_msg.pose.orientation = euler_to_quaternion(
            math.radians(transform.rotation.roll),
            math.radians(transform.rotation.pitch),
            math.radians(transform.rotation.yaw),
        )
        self.initial_pose_pub.publish(pose_msg)

    def _align_platoon_after_spawn(self):
        if not self.platoon_followers:
            return
        leader = None
        for veh in self.vehicles:
            try:
                role = veh.attributes.get("role_name", "")
            except Exception:
                continue
            if role == self.platoon_leader:
                leader = veh
                break
        if leader is None:
            rospy.logwarn("platoon: leader %s not found among spawned vehicles", self.platoon_leader)
            return
        base = leader.get_transform()
        yaw_rad = math.radians(base.rotation.yaw)
        fx = math.cos(yaw_rad)
        fy = math.sin(yaw_rad)
        for idx, follower_name in enumerate(self.platoon_followers, start=1):
            follower = None
            for veh in self.vehicles:
                try:
                    role = veh.attributes.get("role_name", "")
                except Exception:
                    continue
                if role == follower_name:
                    follower = veh
                    break
            if follower is None:
                rospy.logwarn("platoon: follower %s not found; skip", follower_name)
                continue
            offset = self.platoon_gap_m * float(idx)
            loc = carla.Location(
                x=base.location.x - fx * offset,
                y=base.location.y - fy * offset,
                z=base.location.z,
            )
            tf = carla.Transform(location=loc, rotation=base.rotation)
            try:
                follower.set_transform(tf)
                rospy.loginfo("platoon: aligned %s behind %s at %.1fm", follower_name, self.platoon_leader, offset)
            except Exception as exc:
                rospy.logwarn("platoon: set_transform failed for %s: %s", follower_name, exc)


if __name__ == "__main__":
    try:
        MultiVehicleSpawner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
