#!/usr/bin/env python3

import math
import random
import sys
import time

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

CARLA_BUILD_PATH = "/home/jamie/carla/PythonAPI/carla/build/lib.linux-x86_64-cpython-38"
if CARLA_BUILD_PATH not in sys.path:
    sys.path.insert(0, CARLA_BUILD_PATH)

try:
    import carla
except ImportError as exc:
    rospy.logfatal(f"CARLA import failed: {exc}")
    carla = None


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
        self.randomize_spawn = rospy.get_param("~randomize_spawn", True)
        self.spawn_seed = rospy.get_param("~spawn_seed", None)
        self.spawn_retry_limit = int(rospy.get_param("~spawn_retry_limit", 20))
        self.spawn_min_separation_m = float(rospy.get_param("~spawn_min_separation_m", 8.0))

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
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        seed_value = int(self.spawn_seed) if self.spawn_seed is not None else None
        initial_seed = seed_value if seed_value is not None else time.time()
        self._spawn_rng = random.Random(initial_seed)

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
        blueprint_library = self.world.get_blueprint_library()
        if self.enable_autopilot:
            self._configure_tm_globals()

        # Per-vehicle model mapping
        model_map = [
            "vehicle.vehicle.greenxycar",   # vehicle 1
            "vehicle.vehicle.purplexycar",    # vehicle 2
            "vehicle.vehicle.redxycar", # vehicle 3
            "vehicle.vehicle.yellowxycar", # vehicle 4
            "vehicle.vehicle.pinkxycar", # vehicle 5
            "vehicle.vehicle.whitexycar", # vehicle 6
        ]

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

            if self.randomize_spawn:
                seed_hint = self._spawn_rng.random()
            else:
                seed_hint_base = int(self.spawn_seed) if self.spawn_seed is not None else 0
                seed_hint = seed_hint_base + index

            for transform, spawn_index in pick_spawn_transform(
                self.world, seed_hint, self.spawn_retry_limit
            ):
                # Enforce minimum separation from previously spawned vehicles
                too_close = False
                for _role, (prev_tf, _idx) in self.spawned_transforms.items():
                    dx = transform.location.x - prev_tf.location.x
                    dy = transform.location.y - prev_tf.location.y
                    dz = transform.location.z - prev_tf.location.z
                    if math.sqrt(dx * dx + dy * dy + dz * dz) < max(0.0, self.spawn_min_separation_m):
                        too_close = True
                        break
                if too_close:
                    continue
                vehicle = self.world.try_spawn_actor(blueprint, transform)
                if vehicle is not None:
                    chosen_transform = transform
                    chosen_index = spawn_index
                    break
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
