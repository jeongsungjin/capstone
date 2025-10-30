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
        self.enable_autopilot = rospy.get_param("~enable_autopilot", False)
        self.spawn_delay = rospy.get_param("~spawn_delay", 0.5)
        self.target_speed = rospy.get_param("~target_speed", 3.0)
        self.randomize_spawn = rospy.get_param("~randomize_spawn", True)
        self.spawn_seed = rospy.get_param("~spawn_seed", None)
        self.spawn_retry_limit = int(rospy.get_param("~spawn_retry_limit", 20))
        self.reset_world_on_shutdown = bool(rospy.get_param("~reset_world_on_shutdown", False))

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
        # Spectator lock to keep BEV view fixed against mouse/touch moves
        self._spectator_target = None  # (x, y, z, yaw_deg)
        self._spectator_lock_timer = None

        self.spawn_vehicles()
        rospy.on_shutdown(self.cleanup)
        rospy.Timer(rospy.Duration(0.1), self.publish_odometry)

    def spawn_vehicles(self):
        blueprint_library = self.world.get_blueprint_library()
        base_bp = blueprint_library.find(self.vehicle_model)
        if base_bp is None:
            rospy.logwarn(f"Vehicle model {self.vehicle_model} not found; using first available")
            base_bp = blueprint_library.filter("vehicle.*")[0]

        for index in range(self.num_vehicles):
            role_name = f"ego_vehicle_{index + 1}"
            blueprint = blueprint_library.find(base_bp.id)
            if blueprint.has_attribute("role_name"):
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
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, max(0, 100 - self.target_speed * 2))

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

        # Set spectator to a top-down BEV view over the map center, with optional yaw/auto-height
        try:
            if self.spawn_points:
                xs = [sp.location.x for sp in self.spawn_points]
                ys = [sp.location.y for sp in self.spawn_points]
                cx = sum(xs) / float(len(xs))
                cy = sum(ys) / float(len(ys))
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                span_x = max_x - min_x
                span_y = max_y - min_y

                auto_height = bool(rospy.get_param("~spectator_auto_height", False))
                min_height = float(rospy.get_param("~spectator_min_height", 65.0))
                height = float(rospy.get_param("~spectator_height", 65.0))
                if auto_height:
                    # Heuristic: use max span to choose a height that likely fits the whole map
                    height = max(min_height, max(span_x, span_y) * 1.1)

                yaw_deg = float(rospy.get_param("~spectator_yaw_deg", -90.0))
                # Optional offsets: map-axis offsets (meters)
                offset_x = float(rospy.get_param("~spectator_offset_x", -16.0))
                offset_y = float(rospy.get_param("~spectator_offset_y", -10.0))
                # Optional view-relative offsets (meters): right/up on screen after yaw applied
                view_right_m = float(rospy.get_param("~spectator_view_right_m", 0.0))
                view_up_m = float(rospy.get_param("~spectator_view_up_m", 0.0))

                # Map-axis center first
                target_x = cx + offset_x
                target_y = cy + offset_y

                # Apply view-relative offset by rotating right/up vectors by yaw
                if view_right_m != 0.0 or view_up_m != 0.0:
                    import math
                    yaw_rad = math.radians(yaw_deg)
                    # Image right vector (world) for given yaw
                    rx = math.cos(yaw_rad)
                    ry = math.sin(yaw_rad)
                    # Image up vector (world)
                    ux = -math.sin(yaw_rad)
                    uy = math.cos(yaw_rad)
                    target_x += view_right_m * rx + view_up_m * ux
                    target_y += view_right_m * ry + view_up_m * uy
                spectator = self.world.get_spectator()
                bev_loc = carla.Location(x=target_x, y=target_y, z=height)
                bev_rot = carla.Rotation(pitch=-90.0, yaw=yaw_deg, roll=0.0)
                spectator.set_transform(carla.Transform(bev_loc, bev_rot))
                # Lock spectator by periodically re-applying the BEV transform
                self._spectator_target = (target_x, target_y, height, yaw_deg)
                if self._spectator_lock_timer is None:
                    # 5 Hz refresh is usually enough to cancel user camera moves
                    self._spectator_lock_timer = rospy.Timer(
                        rospy.Duration(0.2), self._spectator_lock_cb
                    )
                rospy.loginfo(
                    "multi_vehicle_spawner: spectator BEV at(%.1f, %.1f, %.1f) yaw=%.1f span=(%.1f, %.1f)",
                    target_x,
                    target_y,
                    height,
                    yaw_deg,
                    span_x,
                    span_y,
                )
        except Exception as exc:
            rospy.logwarn("multi_vehicle_spawner: failed to set spectator BEV view: %s", exc)

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
        if self._spectator_lock_timer is not None:
            try:
                self._spectator_lock_timer.shutdown()
            except Exception:
                pass
            self._spectator_lock_timer = None
        # Optionally reset the CARLA world when this node is shutting down,
        # so that actor IDs start from 1 on the next run without impacting startup.
        if self.reset_world_on_shutdown:
            try:
                current_map = None
                try:
                    current_map = self.world.get_map().name if self.world is not None else None
                except RuntimeError:
                    current_map = None
                self.client.reload_world()
                if current_map:
                    try:
                        self.client.load_world(current_map)
                    except RuntimeError:
                        pass
                rospy.loginfo("multi_vehicle_spawner: CARLA world reloaded on shutdown")
            except RuntimeError as exc:
                rospy.logwarn(f"multi_vehicle_spawner: failed to reload CARLA world on shutdown: {exc}")

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

    def _spectator_lock_cb(self, _event):
        target = self._spectator_target
        if target is None:
            return
        try:
            x, y, z, yaw_deg = target
            spectator = self.world.get_spectator()
            bev_loc = carla.Location(x=x, y=y, z=z)
            bev_rot = carla.Rotation(pitch=-90.0, yaw=yaw_deg, roll=0.0)
            spectator.set_transform(carla.Transform(bev_loc, bev_rot))
        except Exception:
            pass


if __name__ == "__main__":
    try:
        MultiVehicleSpawner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
