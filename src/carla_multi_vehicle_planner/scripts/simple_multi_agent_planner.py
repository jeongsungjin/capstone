#!/usr/bin/env python3

import math
import random
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header

# Ensure CARLA Python API and Agents are on sys.path (side-effect import)
try:
    import setup_carla_path  # noqa: F401
except Exception:
    setup_carla_path = None  # type: ignore

try:
    import carla  # type: ignore
except Exception as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA package: {exc}")

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO  # type: ignore
except Exception:
    GlobalRoutePlanner = None  # type: ignore
    GlobalRoutePlannerDAO = None  # type: ignore


class SimpleMultiAgentPlanner:
    """
    최소 기능 다중 차량 글로벌 플래너.
    - 각 ego 차량의 현재 전방 위치를 시작점으로 사용
    - 스폰 포인트 중 거리 제약을 만족하는 목적지 임의 선택
    - CARLA GlobalRoutePlanner로 경로 생성 후 /global_path_{role} 퍼블리시
    - 충돌 회피, 수동 목표, 오프셋/시각화 등 부가 기능은 제거
    """

    def __init__(self) -> None:
        rospy.init_node("multi_agent_conflict_free_planner", anonymous=True)

        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.global_route_resolution = float(rospy.get_param("~global_route_resolution", 1.0))
        self.replan_interval = float(rospy.get_param("~replan_interval", 1.0))
        self.min_destination_distance = float(rospy.get_param("~min_destination_distance", 80.0))
        self.max_destination_distance = float(rospy.get_param("~max_destination_distance", 180.0))

        # CARLA world/map
        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        timeout = float(rospy.get_param("~carla_timeout", 10.0))
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        # Route planner (prefer CARLA agents; fallback to simple lane-follow)
        self.route_planner = None
        if GlobalRoutePlanner is not None:
            try:
                if GlobalRoutePlannerDAO is not None:
                    dao = GlobalRoutePlannerDAO(self.carla_map, self.global_route_resolution)
                    self.route_planner = GlobalRoutePlanner(dao)
                else:
                    self.route_planner = GlobalRoutePlanner(self.carla_map, self.global_route_resolution)
                if hasattr(self.route_planner, "setup"):
                    self.route_planner.setup()
                rospy.loginfo("SimpleMultiAgentPlanner: using CARLA GlobalRoutePlanner")
            except Exception as exc:
                self.route_planner = None
                rospy.logwarn(f"GlobalRoutePlanner unavailable, falling back to lane-follow routing: {exc}")
        else:
            rospy.logwarn("GlobalRoutePlanner module not found; using lane-follow routing fallback")

        self.spawn_points = self.carla_map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        # Publishers
        self.path_publishers: Dict[str, rospy.Publisher] = {}
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/global_path_{role}"
            self.path_publishers[role] = rospy.Publisher(topic, Path, queue_size=1, latch=True)

        rospy.sleep(0.5)
        self._plan_once()
        rospy.Timer(rospy.Duration(self.replan_interval), self._timer_cb)

    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _timer_cb(self, _evt) -> None:
        self._plan_once()

    def _get_ego_vehicles(self) -> List[carla.Actor]:
        actors = self.world.get_actors().filter("vehicle.*")
        vehicles: List[carla.Actor] = []
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role.startswith("ego_vehicle_"):
                vehicles.append(actor)
        vehicles.sort(key=lambda v: v.attributes.get("role_name", ""))
        return vehicles

    def _vehicle_front(self, vehicle: carla.Actor) -> carla.Location:
        tf = vehicle.get_transform()
        yaw_rad = math.radians(tf.rotation.yaw)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        offset = 2.0
        bb = getattr(vehicle, "bounding_box", None)
        if bb is not None and getattr(bb, "extent", None) is not None:
            offset = bb.extent.x + 0.3
        return carla.Location(
            x=tf.location.x + forward_x * offset,
            y=tf.location.y + forward_y * offset,
            z=tf.location.z,
        )

    def _choose_destination(self, start: carla.Location, max_trials: int = 80) -> Optional[carla.Location]:
        for _ in range(max_trials):
            cand = random.choice(self.spawn_points).location
            dist = math.hypot(cand.x - start.x, cand.y - start.y)
            if self.min_destination_distance <= dist <= self.max_destination_distance:
                return cand
        return None

    def _trace_route(self, start: carla.Location, dest: carla.Location):
        if self.route_planner is not None:
            try:
                return self.route_planner.trace_route(start, dest)
            except Exception as exc:
                rospy.logwarn(f"trace_route failed: {exc}")
                return None
        # Fallback: generate a simple forward route by lane-follow for a fixed length
        return self._lane_follow_route(start, length_m=150.0, step_m=2.0)

    def _lane_follow_route(self, start: carla.Location, length_m: float = 120.0, step_m: float = 2.0):
        try:
            wp = self.carla_map.get_waypoint(start, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                return None
            route = []
            traveled = 0.0
            current = wp
            # Seed with start waypoint
            route.append((current, None))
            while traveled < max(5.0, float(length_m)):
                nxt = current.next(float(step_m))
                if not nxt:
                    break
                # Choose the first option deterministically
                current = nxt[0]
                route.append((current, None))
                traveled += float(step_m)
            return route if len(route) >= 2 else None
        except Exception as exc:
            rospy.logwarn(f"lane-follow routing failed: {exc}")
            return None

    def _route_to_points(self, route) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        if not route:
            return points
        last_x, last_y = None, None
        for wp, _ in route:
            loc = wp.transform.location
            x, y = float(loc.x), float(loc.y)
            if last_x is None:
                points.append((x, y))
                last_x, last_y = x, y
                continue
            dx, dy = x - last_x, y - last_y
            if dx * dx + dy * dy >= 0.2 * 0.2:
                points.append((x, y))
                last_x, last_y = x, y
        if len(points) >= 2:
            return points
        return points

    def _publish_path(self, points: List[Tuple[float, float]], role: str) -> None:
        if role not in self.path_publishers:
            return
        msg = Path()
        msg.header = Header(frame_id="map", stamp=rospy.Time.now())
        for x, y in points:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = 0.0
            msg.poses.append(p)
        self.path_publishers[role].publish(msg)

    def _plan_once(self) -> None:
        vehicles = self._get_ego_vehicles()
        if not vehicles:
            rospy.loginfo_throttle(5.0, "No ego vehicles found")
            return
        for index, vehicle in enumerate(vehicles[: self.num_vehicles]):
            role = self._role_name(index)
            start_loc = self._vehicle_front(vehicle)
            dest_loc = self._choose_destination(start_loc)
            if dest_loc is None:
                rospy.logwarn_throttle(5.0, f"{role}: destination not found within distance bounds")
                continue
            route = self._trace_route(start_loc, dest_loc)
            if not route or len(route) < 2:
                rospy.logwarn_throttle(5.0, f"{role}: route trace failed")
                continue
            points = self._route_to_points(route)
            if len(points) < 2:
                rospy.logwarn_throttle(5.0, f"{role}: insufficient path points")
                continue
            self._publish_path(points, role)


if __name__ == "__main__":
    try:
        SimpleMultiAgentPlanner()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleMultiAgentPlanner crashed: {e}")
        raise


