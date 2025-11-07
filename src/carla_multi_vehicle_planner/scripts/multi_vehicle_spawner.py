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
        self.vehicle_model = rospy.get_param("~vehicle_model", "vehicle.vehicle.xycar")
        self.enable_autopilot = rospy.get_param("~enable_autopilot", True)
        self.spawn_delay = rospy.get_param("~spawn_delay", 0.5)
        self.target_speed = rospy.get_param("~target_speed", 8.0)
        self.randomize_spawn = rospy.get_param("~randomize_spawn", True)
        self.spawn_seed = rospy.get_param("~spawn_seed", None)
        self.spawn_retry_limit = int(rospy.get_param("~spawn_retry_limit", 20))
        # Autopilot/TM options
        self.autopilot_random_route = bool(rospy.get_param("~autopilot_random_route", True))
        self.autopilot_route_len = int(rospy.get_param("~autopilot_route_len", 300))
        self.autopilot_step_m = float(rospy.get_param("~autopilot_step_m", 2.0))
        self.tm_global_min_headway = float(rospy.get_param("~tm_global_min_headway", 2.5))
        self.tm_hybrid_physics = bool(rospy.get_param("~tm_hybrid_physics", True))
        self.tm_auto_lane_change = bool(rospy.get_param("~tm_auto_lane_change", True))
        self.tm_keep_right = bool(rospy.get_param("~tm_keep_right", False))
        self.tm_ignore_lights_pct = int(rospy.get_param("~tm_ignore_lights_pct", 0))
        self.tm_ignore_signs_pct = int(rospy.get_param("~tm_ignore_signs_pct", 0))

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager()
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

        self.spawn_vehicles()
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
            "vehicle.vehicle.bluexycar",    # vehicle 2
            "vehicle.vehicle.coloredxycar", # vehicle 3
            "vehicle.vehicle.yellowxycar", # vehicle 4
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
                vehicle = self.world.try_spawn_actor(blueprint, transform)
                if vehicle is not None:
                    chosen_transform = transform
                    chosen_index = spawn_index
                    break
                time.sleep(0.1)

            if vehicle is None:
                rospy.logerr(f"Failed to spawn vehicle {role_name} after {self.spawn_retry_limit} attempts")
                continue

            vehicle.set_autopilot(self.enable_autopilot, self.traffic_manager.get_port())
            if self.enable_autopilot:
                # Speed: negative means faster, positive slower (percentage)
                try:
                    self.traffic_manager.vehicle_percentage_speed_difference(
                        vehicle, max(0, 100 - self.target_speed * 2)
                    )
                except Exception:
                    pass
                # Per-vehicle TM preferences (best-effort, ignore if not supported)
                try:
                    self.traffic_manager.auto_lane_change(vehicle, self.tm_auto_lane_change)
                except Exception:
                    pass
                try:
                    self.traffic_manager.keep_right_rule_percentage(vehicle, 100 if self.tm_keep_right else 0)
                except Exception:
                    pass
                try:
                    self.traffic_manager.ignore_lights_percentage(vehicle, self.tm_ignore_lights_pct)
                except Exception:
                    pass
                try:
                    self.traffic_manager.ignore_signs_percentage(vehicle, self.tm_ignore_signs_pct)
                except Exception:
                    pass
                # Optionally assign a forward random route so vehicles actually start cruising
                if self.autopilot_random_route:
                    try:
                        self._assign_tm_forward_path(vehicle, chosen_transform.location if chosen_transform else transform.location)
                    except Exception as exc:
                        rospy.logwarn("%s: TM path assignment failed: %s", role_name, exc)

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

        # Spectator control moved to cam_perspective.py

    def _assign_tm_forward_path(self, vehicle, start_location):
        # Build a forward path along lane centerlines starting from nearest waypoint
        start_wp = self.carla_map.get_waypoint(
            start_location,
            project_to_road=True,
            lane_type=getattr(carla.LaneType, 'Driving', 1),
        )
        if start_wp is None:
            return
        route_wps = [start_wp]
        rng = self._spawn_rng
        current_wp = start_wp
        for _ in range(max(1, self.autopilot_route_len)):
            next_wps = current_wp.next(self.autopilot_step_m)
            if not next_wps:
                break
            # Choose one if junction/split; prefer keeping lane
            current_wp = rng.choice(next_wps)
            route_wps.append(current_wp)
        # Convert to locations for TM
        path_locs = [wp.transform.location for wp in route_wps]
        # Best-effort: set_path may be unavailable in some versions
        if hasattr(self.traffic_manager, 'set_path'):
            self.traffic_manager.set_path(vehicle, path_locs)
        elif hasattr(self.traffic_manager, 'set_route'):
            # Some versions expect waypoints instead of locations
            try:
                self.traffic_manager.set_route(vehicle, route_wps)
            except Exception:
                pass

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

    # Spectator lock callback removed (handled by cam_perspective.py)


if __name__ == "__main__":
    try:
        MultiVehicleSpawner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
