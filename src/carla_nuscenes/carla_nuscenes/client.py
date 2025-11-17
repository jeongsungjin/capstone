import carla
from .sensor import *
from .vehicle import Vehicle
from .walker import Walker
import math
from .utils import generate_token,get_nuscenes_rt,get_intrinsic,transform_timestamp,clamp
import random
import os

class Client:
    def __init__(self,client_config):
        self.client = carla.Client(client_config["host"],client_config["port"])
        self.client.set_timeout(client_config["time_out"])

    def generate_world(self,world_config):
        print("generate world start!")
        use_existing = bool(world_config.get("use_existing_world", False))
        if not use_existing:
            # Load requested map fresh
            self.client.load_world(world_config["map_name"])
        # Attach to current world (either freshly loaded or already running)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        if not use_existing:
            # Only modify map layers when we control the map
            try:
                self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
            except Exception:
                pass
        self.ego_vehicle = None
        self.sensors = None
        self.vehicles = None
        self.walkers = None

        get_category = lambda bp: "vehicle.car" if bp.id.split(".")[0] == "vehicle" else "human.pedestrian.adult" if bp.id.split(".")[0] == "walker" else None
        self.category_dict = {bp.id: get_category(bp) for bp in self.world.get_blueprint_library()}
        get_attribute = lambda bp: ["vehicle.moving"] if bp.id.split(".")[0] == "vehicle" else ["pedestrian.moving"] if bp.id.split(".")[0] == "walker" else None
        self.attribute_dict = {bp.id: get_attribute(bp) for bp in self.world.get_blueprint_library()}

        self.trafficmanager = self.client.get_trafficmanager()
        self.trafficmanager.set_synchronous_mode(True)
        self.trafficmanager.set_respawn_dormant_vehicles(False)
        self.settings = carla.WorldSettings(**world_config["settings"]) if not use_existing else self.world.get_settings()
        # Enforce sync mode for recording stability
        self.settings.synchronous_mode = True
        # Allow config to request no_rendering_mode for stability
        try:
            cfg_no_render = bool(world_config.get("settings", {}).get("no_rendering_mode", False))
        except Exception:
            cfg_no_render = False
        self.settings.no_rendering_mode = cfg_no_render
        # If existing world has variable delta, set a safe default fixed delta
        try:
            if use_existing and (not hasattr(self.settings, 'fixed_delta_seconds') or self.settings.fixed_delta_seconds in [None, 0.0]):
                self.settings.fixed_delta_seconds = float(world_config.get("settings", {}).get("fixed_delta_seconds", 0.05))
        except Exception:
            pass
        try:
            self.world.apply_settings(self.settings)
        except Exception:
            # If simulator was started with incompatible options, proceed with current settings
            pass
        self.world.set_pedestrians_cross_factor(1)
        print("generate world success!")

    def generate_scene(self,scene_config):
        print("generate scene start!")
        if scene_config["custom"]:
            self.generate_custom_scene(scene_config)
        else:
            self.generate_random_scene(scene_config)
        print("generate scene success!")

    def generate_custom_scene(self,scene_config):
        
        if scene_config.get("weather_presets") and not bool(scene_config.get("use_existing_world", False)):
            try:
                preset = random.choice(scene_config["weather_presets"])  # e.g., "ClearNoon"
                self.weather = getattr(carla.WeatherParameters, preset)
            except Exception:
                self.weather = carla.WeatherParameters(**self.get_random_weather())
        elif scene_config.get("randomize_weather", False) and not bool(scene_config.get("use_existing_world", False)):
            self.weather = carla.WeatherParameters(**self.get_random_weather())
        else:
            if scene_config["weather_mode"] == "custom" and not bool(scene_config.get("use_existing_world", False)):
                self.weather = carla.WeatherParameters(**scene_config["weather"])
            else:
                self.weather = getattr(carla.WeatherParameters, scene_config["weather_mode"])
        # Only change weather if we control the world
        try:
            if not bool(scene_config.get("use_existing_world", False)):
                self.world.set_weather(self.weather)
        except Exception:
            pass
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        self.ego_vehicle = Vehicle(world=self.world,**scene_config["ego_vehicle"])
        center = carla.Location(0.0, 0.0, 0.0)
        if "spawn_box" in scene_config:
            center = carla.Location(scene_config["spawn_box"].get("center_x",0.0), scene_config["spawn_box"].get("center_y",0.0), 0.0)
        
        # Helpers to move laterally across driving lanes relative to center
        def prefer_outer_lane(wp: carla.Waypoint) -> carla.Waypoint:
            if wp is None:
                return wp
            best = wp
            best_dist = wp.transform.location.distance(center)
            changed = True
            while changed:
                changed = False
                for nxt in (best.get_left_lane(), best.get_right_lane()):
                    if nxt and nxt.lane_type == carla.LaneType.Driving:
                        d = nxt.transform.location.distance(center)
                        if d > best_dist + 0.05:
                            best = nxt
                            best_dist = d
                            changed = True
                            break
            return best
        def prefer_inner_lane(wp: carla.Waypoint) -> carla.Waypoint:
            if wp is None:
                return wp
            best = wp
            best_dist = wp.transform.location.distance(center)
            changed = True
            while changed:
                changed = False
                for nxt in (best.get_left_lane(), best.get_right_lane()):
                    if nxt and nxt.lane_type == carla.LaneType.Driving:
                        d = nxt.transform.location.distance(center)
                        if d < best_dist - 0.05:
                            best = nxt
                            best_dist = d
                            changed = True
                            break
            return best
        def radial_probe(loc: carla.Location, toward_outer: bool) -> carla.Waypoint:
            center_loc = carla.Location(0.0, 0.0, 0.0)
            dir_vec = carla.Vector3D(loc.x - center_loc.x, loc.y - center_loc.y, 0.0)
            mag = math.hypot(dir_vec.x, dir_vec.y)
            if mag < 1e-3:
                return self.world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            dir_vec.x /= mag; dir_vec.y /= mag
            base_wp = self.world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if base_wp is None:
                return None
            base_r = base_wp.transform.location.distance(center_loc)
            # Probe up to ±5 m radially to switch lanes if needed
            steps = [i*0.5 for i in range(1, 11)]
            if not toward_outer:
                steps = [-s for s in steps]
            for s in steps:
                probe = carla.Location(loc.x + dir_vec.x*s, loc.y + dir_vec.y*s, loc.z)
                wp = self.world.get_map().get_waypoint(probe, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is not None:
                    r = wp.transform.location.distance(center_loc)
                    if (toward_outer and r > base_r + 0.2) or ((not toward_outer) and r < base_r - 0.2):
                        return wp
            # Fallback to neighbor-lane walk
            return prefer_outer_lane(base_wp) if toward_outer else prefer_inner_lane(base_wp)
        
        # ROS-like random spawn using map spawn points
        ros_like = scene_config.get("ros_like_random_spawn", False)
        already_spawned_vehicles = False
        spawn_points = []
        if ros_like:
            spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(spawn_points)
            # Ego from first spawn point
            if spawn_points:
                self.ego_vehicle.transform = spawn_points[0]
        
        # Precompute spawn candidates from external path files if provided (for non-ros_like modes)
        file_candidates_in = []
        file_candidates_out = []
        def _load_path_file(path_file, bias_outer=False):
            transforms = []
            if not path_file:
                return transforms
            if not os.path.isabs(path_file):
                path_file = os.path.join(os.getcwd(), path_file)
            try:
                with open(path_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = [p.strip() for p in line.replace(',', ' ').split() if p.strip()]
                        if len(parts) >= 2:
                            x = float(parts[0]); y = float(parts[1])
                            loc = carla.Location(x=x, y=y, z=0.0)
                            # Radially probe to lock onto inner/outer lane robustly
                            wp = radial_probe(loc, toward_outer=bias_outer)
                            if wp is not None:
                                transforms.append(wp.transform)
            except Exception:
                pass
            return transforms
        if not ros_like:
            file_candidates_in = _load_path_file(scene_config.get("spawn_path_file_in", None), bias_outer=False)
            file_candidates_out = _load_path_file(scene_config.get("spawn_path_file_out", None), bias_outer=True)
        file_candidates = list(file_candidates_in) + list(file_candidates_out)
        random.shuffle(file_candidates)
        
        def _compute_yaw_deg_from_tr_list(tr_list):
            try:
                if len(tr_list) >= 2:
                    p1 = tr_list[0].location; p2 = tr_list[1].location
                    dx = p2.x - p1.x; dy = p2.y - p1.y
                    return math.degrees(math.atan2(dy, dx))
            except Exception:
                pass
            return None
        
        # Snap ego to nearest drivable waypoint (initial) if not ros-like
        if not ros_like:
            try:
                ego_loc = scene_config["ego_vehicle"]["location"]
                ego_rot = scene_config["ego_vehicle"]["rotation"]
                waypoint = self.world.get_map().get_waypoint(carla.Location(**ego_loc), project_to_road=True, lane_type=carla.LaneType.Driving)
                if waypoint is not None:
                    self.ego_vehicle.transform = carla.Transform(waypoint.transform.location, waypoint.transform.rotation)
            except Exception as e:
                pass
        # Optional: ego spawn from paths if not ros-like
        if not ros_like and scene_config.get("ego_random_from_paths", False) and file_candidates:
            try:
                spawn_mode = scene_config.get("ego_spawn_mode", "random")  # "first" or "random"
                path_pref = scene_config.get("ego_path_preference", "any")  # "in"|"out"|"any"
                yaw_from_path = bool(scene_config.get("yaw_from_path", False))
                yaw_offset_deg = float(scene_config.get("yaw_offset_deg", 0.0))
                selected_tr = None
                if spawn_mode == "first":
                    if path_pref == "out" and file_candidates_out:
                        selected_tr = file_candidates_out[0]
                        yaw_deg = _compute_yaw_deg_from_tr_list(file_candidates_out)
                    elif path_pref == "in" and file_candidates_in:
                        selected_tr = file_candidates_in[0]
                        yaw_deg = _compute_yaw_deg_from_tr_list(file_candidates_in)
                    else:
                        src = file_candidates_out if file_candidates_out else file_candidates_in
                        selected_tr = src[0]
                        yaw_deg = _compute_yaw_deg_from_tr_list(src)
                    if yaw_from_path and yaw_deg is not None:
                        rot = carla.Rotation(pitch=0.0, yaw=yaw_deg + yaw_offset_deg, roll=0.0)
                        self.ego_vehicle.transform = carla.Transform(selected_tr.location, rot)
                    else:
                        self.ego_vehicle.transform = selected_tr
                else:
                    if path_pref == "out" and file_candidates_out:
                        selected_tr = random.choice(file_candidates_out)
                    elif path_pref == "in" and file_candidates_in:
                        selected_tr = random.choice(file_candidates_in)
                    else:
                        selected_tr = random.choice(file_candidates)
                    if yaw_from_path:
                        src = file_candidates_out if (selected_tr in file_candidates_out) else file_candidates_in
                        yaw_deg = _compute_yaw_deg_from_tr_list(src)
                        if yaw_deg is not None:
                            rot = carla.Rotation(pitch=0.0, yaw=yaw_deg + yaw_offset_deg, roll=0.0)
                            self.ego_vehicle.transform = carla.Transform(selected_tr.location, rot)
                        else:
                            self.ego_vehicle.transform = selected_tr
                    else:
                        self.ego_vehicle.transform = selected_tr
            except Exception:
                pass
        
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')
        # Retry spawn robustly: prefer valid map spawn points, then jittered fallbacks
        spawned = False
        # 1) Try official map spawn points first
        try_points = []
        try:
            try_points = list(self.world.get_map().get_spawn_points())
        except Exception:
            try_points = []
        if try_points:
            random.shuffle(try_points)
            for sp in try_points:
                self.ego_vehicle.transform = sp
                if self.ego_vehicle.spawn_actor():
                    spawned = True
                    break
        # 2) Try path-based candidates if provided
        if (not spawned) and (not ros_like) and file_candidates:
            for tr in file_candidates:
                self.ego_vehicle.transform = tr
                if self.ego_vehicle.spawn_actor():
                    spawned = True
                    break
        # 3) Try small jitter around current transform
        if not spawned:
            base = self.ego_vehicle.transform.location
            for dx, dy in [(0.5,0),(-0.5,0),(0,0.5),(0,-0.5),(1.0,0),(0,1.0),(-1.0,0),(0,-1.0)]:
                jittered = carla.Location(base.x+dx, base.y+dy, base.z)
                self.ego_vehicle.transform.location = jittered
                if self.ego_vehicle.spawn_actor():
                    spawned = True
                    break
        # 4) As a last resort: snap to nearest drivable waypoint of current location
        if not spawned:
            try:
                waypoint = self.world.get_map().get_waypoint(self.ego_vehicle.transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                if waypoint is not None:
                    self.ego_vehicle.transform = carla.Transform(waypoint.transform.location, waypoint.transform.rotation)
                    spawned = self.ego_vehicle.spawn_actor()
            except Exception:
                pass
        if not spawned:
            raise RuntimeError("Ego spawn failed due to collisions")
        self.ego_vehicle.get_actor().set_autopilot()
        self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),0)
        self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)
        self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)
        
        # Build vehicles: list or randomized
        if not ros_like:
            randomize = "vehicle_count_range" in scene_config
            if randomize:
                count_min, count_max = scene_config["vehicle_count_range"][0], scene_config["vehicle_count_range"][1]
                target_total = max(1, min(3, int(random.randint(count_min, count_max))))
                vehicles_to_spawn = max(0, target_total - 1)
                bp_name = scene_config.get("traffic_bp_name", "vehicle.vehicle.kintax")
                half = None
                if "spawn_box" in scene_config:
                    center = carla.Location(scene_config["spawn_box"].get("center_x",0.0), scene_config["spawn_box"].get("center_y",0.0), 0.0)
                    half = float(scene_config["spawn_box"].get("half",10.0))
                def in_box_loc(loc: carla.Location):
                    if half is None:
                        return True
                    return abs(loc.x - center.x) <= half and abs(loc.y - center.y) <= half
                if file_candidates:
                    candidate_transforms = []
                    mixed = list(file_candidates_out) + list(file_candidates_in)
                    random.shuffle(mixed)
                    candidate_transforms = [tr for tr in mixed if in_box_loc(tr.location)]
                else:
                    base_wps = [wp for wp in self.world.get_map().generate_waypoints(5.0) if wp.lane_type == carla.LaneType.Driving]
                    candidate_transforms = [wp.transform for wp in base_wps if in_box_loc(wp.transform.location)]
                    random.shuffle(candidate_transforms)
                def _dist(a: carla.Location, b: carla.Location):
                    dx = a.x - b.x; dy = a.y - b.y; dz = a.z - b.z
                    return math.sqrt(dx*dx + dy*dy + dz*dz)
                reserved = [self.ego_vehicle.transform.location]
                chosen = []
                used_lanes = set()
                for tr in candidate_transforms:
                    wp = self.world.get_map().get_waypoint(tr.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if wp and wp.lane_id in used_lanes:
                        continue
                    if any(_dist(tr.location, r) < 6.0 for r in reserved):
                        continue
                    chosen.append(tr)
                    reserved.append(tr.location)
                    if wp:
                        used_lanes.add(wp.lane_id)
                    if len(chosen) >= vehicles_to_spawn:
                        break
                if len(chosen) < vehicles_to_spawn:
                    for tr in candidate_transforms:
                        if any(_dist(tr.location, r) < 6.0 for r in reserved):
                            continue
                        chosen.append(tr)
                        reserved.append(tr.location)
                        if len(chosen) >= vehicles_to_spawn:
                            break
                self.vehicles = []
                for tr in chosen:
                    location = {attr:getattr(tr.location,attr) for attr in ["x","y","z"]}
                    rotation = {attr:getattr(tr.rotation,attr) for attr in ["yaw","pitch","roll"]}
                    self.vehicles.append(Vehicle(world=self.world,bp_name=bp_name,location=location,rotation=rotation,path=[]))
            else:
                self.vehicles = [Vehicle(world=self.world,**vehicle_config) for vehicle_config in scene_config["vehicles"]]
        else:
            # ROS-like: spawn vehicles at random map spawn points
            self.vehicles = []
            count_min, count_max = scene_config.get("vehicle_count_range", [1, 1])
            target_total = max(1, min(3, int(random.randint(count_min, count_max))))
            vehicles_to_spawn = max(0, target_total - 1)
            bp_name = scene_config.get("traffic_bp_name", "vehicle.vehicle.kintax")
            ego_loc = self.ego_vehicle.transform.location
            def _dist(a: carla.Location, b: carla.Location):
                dx = a.x - b.x; dy = a.y - b.y; dz = a.z - b.z
                return math.sqrt(dx*dx + dy*dy + dz*dz)
            for sp in spawn_points[1:]:
                if len(self.vehicles) >= vehicles_to_spawn:
                    break
                if _dist(sp.location, ego_loc) < 6.0:
                    continue
                location = {attr:getattr(sp.location,attr) for attr in ["x","y","z"]}
                rotation = {attr:getattr(sp.rotation,attr) for attr in ["yaw","pitch","roll"]}
                v = Vehicle(world=self.world,bp_name=bp_name,location=location,rotation=rotation,path=[])
                if v.spawn_actor():
                    v.get_actor().set_autopilot(True, self.trafficmanager.get_port())
                    self.trafficmanager.distance_to_leading_vehicle(v.get_actor(), 2.0)
                    self.vehicles.append(v)
            already_spawned_vehicles = True
        
        # Snap vehicles to nearest drivable waypoints
        if not ros_like:
            for vehicle in self.vehicles:
                try:
                    waypoint = self.world.get_map().get_waypoint(vehicle.transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if waypoint is not None:
                        vehicle.transform = carla.Transform(waypoint.transform.location, waypoint.transform.rotation)
                except Exception:
                    pass
            # Ensure spacing along lane
            def _dist(a: carla.Location, b: carla.Location):
                dx = a.x - b.x; dy = a.y - b.y; dz = a.z - b.z
                return math.sqrt(dx*dx + dy*dy + dz*dz)
            reserved = [self.ego_vehicle.transform.location]
            for vehicle in self.vehicles:
                tries = 0
                while any(_dist(vehicle.transform.location, r) < 3.0 for r in reserved) and tries < 10:
                    current_wp = self.world.get_map().get_waypoint(vehicle.transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                    next_list = current_wp.next(4.0) if current_wp is not None else []
                    if next_list:
                        next_wp = next_list[0]
                        vehicle.transform = carla.Transform(next_wp.transform.location, next_wp.transform.rotation)
                    else:
                        loc = vehicle.transform.location
                        vehicle.transform.location = carla.Location(loc.x + 1.0, loc.y, loc.z)
                    tries += 1
                reserved.append(vehicle.transform.location)
        
        # Spawn vehicles with retry (skip if already spawned in ros-like mode)
        if not already_spawned_vehicles:
            spawned_vehicles = []
            for vehicle in self.vehicles:
                spawned = vehicle.spawn_actor()
                if not spawned:
                    base = vehicle.transform.location
                    for dx, dy in [(0.5,0),( -0.5,0),(0,0.5),(0,-0.5),(1,0),(0,1),(-1,0),(0,-1)]:
                        jittered = carla.Location(base.x+dx, base.y+dy, base.z)
                        vehicle.transform.location = jittered
                        if vehicle.spawn_actor():
                            spawned = True
                            break
                if not spawned and not ros_like:
                    # try candidate transforms
                    for sp in file_candidates:
                        vehicle.transform = sp
                        if vehicle.spawn_actor():
                            spawned = True
                            break
                if spawned:
                    vehicle.get_actor().set_autopilot(True, self.trafficmanager.get_port())
                    self.trafficmanager.distance_to_leading_vehicle(vehicle.get_actor(), 2.0)
                    self.trafficmanager.auto_lane_change(vehicle.get_actor(), True)
                    spawned_vehicles.append(vehicle)
            self.vehicles = spawned_vehicles
        
        # Assign paths to traffic manager if available
        for vehicle in self.vehicles:
            snapped_path = []
            for loc in vehicle.path:
                try:
                    waypoint = self.world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if waypoint is not None:
                        snapped_path.append(waypoint.transform.location)
                except Exception:
                    continue
            if snapped_path:
                self.trafficmanager.set_path(vehicle.get_actor(),snapped_path)

        self.walkers = [Walker(world=self.world,**walker_config) for walker_config in scene_config["walkers"]]
        walkers_batch = [SpawnActor(walker.blueprint,walker.transform) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_batch)):
            if not response.error:
                self.walkers[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.walkers = list(filter(lambda walker:walker.get_actor(),self.walkers))

        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        walkers_controller_batch = [SpawnActor(walker_controller_bp,carla.Transform(),walker.get_actor()) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_controller_batch)):
                    if not response.error:
                        self.walkers[i].set_controller(response.actor_id)
                    else:
                        print(response.error)
        self.world.tick()
        for walker in self.walkers:
            walker.start()

        # 센서 생성 로직 확장: attach_to가 'world' 이거나 생략되면 월드 고정 센서로 스폰
        import time
        self.sensors = []
        for sensor_config in scene_config["calibrated_sensors"]["sensors"]:
            # 원본 변형 방지
            cfg = dict(sensor_config)
            attach_to_cfg = cfg.pop("attach_to", "ego")
            attach_actor = None if attach_to_cfg == "world" else self.ego_vehicle.get_actor()
            sensor_obj = Sensor(world=self.world, attach_to=attach_actor, **cfg)
            # 개별 스폰 + 재시도 + 틱 대기
            actor_id = None
            for attempt in range(5):
                try:
                    actor = self.world.try_spawn_actor(sensor_obj.blueprint, sensor_obj.transform, sensor_obj.attach_to)
                    if actor is not None:
                        actor_id = actor.id
                        sensor_obj.set_actor(actor_id)
                        break
                except Exception as e:
                    print(f"[sensor-spawn] {sensor_obj.name} attempt {attempt+1} error: {e}")
                # 틱 및 소폭 대기로 엔진 안정화
                try:
                    self.world.tick()
                except Exception:
                    pass
                time.sleep(0.02)
            if actor_id is None:
                print(f"[sensor-spawn] failed: {sensor_obj.name}")
            else:
                print(f"[sensor-spawn] success: {sensor_obj.name} id={actor_id}")
            self.sensors.append(sensor_obj)
        # 유효 센서만 유지
        self.sensors = [s for s in self.sensors if s.get_actor() is not None]

    def tick(self):
        self.world.tick()

    def generate_random_scene(self,scene_config):
        print("generate random scene start!")
        self.weather = carla.WeatherParameters(**self.get_random_weather())
        self.world.set_weather(self.weather)


        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        
        ego_bp_name=scene_config["ego_bp_name"]
        ego_location={attr:getattr(spawn_points[0].location,attr) for attr in ["x","y","z"]}
        ego_rotation={attr:getattr(spawn_points[0].rotation,attr) for attr in ["yaw","pitch","roll"]}
        self.ego_vehicle = Vehicle(world=self.world,bp_name=ego_bp_name,location=ego_location,rotation=ego_rotation)
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')
        self.ego_vehicle.spawn_actor()
        self.ego_vehicle.get_actor().set_autopilot()
        self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),0)
        self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)
        self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)

        vehicle_bp_list = self.world.get_blueprint_library().filter("vehicle")
        self.vehicles = []
        for spawn_point in spawn_points[1:random.randint(1,len(spawn_points))]:
            location = {attr:getattr(spawn_point.location,attr) for attr in ["x","y","z"]}
            rotation = {attr:getattr(spawn_point.rotation,attr) for attr in ["yaw","pitch","roll"]}
            bp_name = random.choice(vehicle_bp_list).id
            self.vehicles.append(Vehicle(world=self.world,bp_name=bp_name,location=location,rotation=rotation))
        vehicles_batch = [SpawnActor(vehicle.blueprint,vehicle.transform)
                            .then(SetAutopilot(FutureActor, True, self.trafficmanager.get_port())) 
                            for vehicle in self.vehicles]

        for i,response in enumerate(self.client.apply_batch_sync(vehicles_batch)):
            if not response.error:
                self.vehicles[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.vehicles = list(filter(lambda vehicle:vehicle.get_actor(),self.vehicles))

        walker_bp_list = self.world.get_blueprint_library().filter("pedestrian")
        self.walkers = []
        for i in range(random.randint(len(spawn_points),len(spawn_points)*2)):
            spawn = self.world.get_random_location_from_navigation()
            if spawn != None:
                bp_name=random.choice(walker_bp_list).id
                spawn_location = {attr:getattr(spawn,attr) for attr in ["x","y","z"]}
                destination=self.world.get_random_location_from_navigation()
                destination_location={attr:getattr(destination,attr) for attr in ["x","y","z"]}
                rotation = {"yaw":random.random()*360,"pitch":random.random()*360,"roll":random.random()*360}
                self.walkers.append(Walker(world=self.world,location=spawn_location,rotation=rotation,destination=destination_location,bp_name=bp_name))
            else:
                print("walker generate fail")
        walkers_batch = [SpawnActor(walker.blueprint,walker.transform) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_batch)):
            if not response.error:
                self.walkers[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.walkers = list(filter(lambda walker:walker.get_actor(),self.walkers))

        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        walkers_controller_batch = [SpawnActor(walker_controller_bp,carla.Transform(),walker.get_actor()) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_controller_batch)):
                    if not response.error:
                        self.walkers[i].set_controller(response.actor_id)
                    else:
                        print(response.error)
        self.world.tick()
        for walker in self.walkers:
            walker.start()

        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
        for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                self.sensors[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))
        print("generate random scene success!")        

    def destroy_scene(self):
        if self.walkers is not None:
            for walker in self.walkers:
                walker.controller.stop()
                walker.destroy()
        if self.vehicles is not None:
            for vehicle in self.vehicles:
                vehicle.destroy()
        if self.sensors is not None:
            for sensor in self.sensors:
                sensor.destroy()
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()


    def destroy_world(self):
        try:
            if hasattr(self, 'trafficmanager') and self.trafficmanager is not None:
                self.trafficmanager.set_synchronous_mode(False)
        except Exception:
            pass
        self.ego_vehicle = None
        self.sensors = None
        self.vehicles = None
        self.walkers = None
        try:
            if hasattr(self, 'world') and hasattr(self, 'original_settings') and self.world is not None and self.original_settings is not None:
                self.world.apply_settings(self.original_settings)
        except Exception:
            pass

    def get_calibrated_sensor(self,sensor):
        sensor_token = generate_token("sensor",sensor.name)
        channel = sensor.name
        if sensor.bp_name == "sensor.camera.rgb":
            intrinsic = get_intrinsic(float(sensor.get_actor().attributes["fov"]),
                            float(sensor.get_actor().attributes["image_size_x"]),
                            float(sensor.get_actor().attributes["image_size_y"])).tolist()
            rotation,translation = get_nuscenes_rt(sensor.transform,"zxy")
        else:
            intrinsic = []
            rotation,translation = get_nuscenes_rt(sensor.transform)
        return sensor_token,channel,translation,rotation,intrinsic
        
    def get_ego_pose(self,sample_data):
        timestamp = transform_timestamp(sample_data[1].timestamp)
        rotation,translation = get_nuscenes_rt(sample_data[0])
        return timestamp,translation,rotation
    
    def get_sample_data(self,sample_data):
        height = 0
        width = 0
        if isinstance(sample_data[1],carla.Image):
            height = sample_data[1].height
            width = sample_data[1].width
        return sample_data,height,width

    def get_sample(self):
        return (transform_timestamp(self.world.get_snapshot().timestamp.elapsed_seconds),)

    def get_instance(self,scene_token,instance):
        category_token = generate_token("category",self.category_dict[instance.blueprint.id])
        id = hash((scene_token,instance.get_actor().id))
        return category_token,id

    def get_sample_annotation(self,scene_token,instance):
        instance_token = generate_token("instance",hash((scene_token,instance.get_actor().id)))
        visibility_token = str(self.get_visibility(instance))
        
        attribute_tokens = [generate_token("attribute",attribute) for attribute in self.get_attributes(instance)]
        rotation,translation = get_nuscenes_rt(instance.get_transform())
        size = [instance.get_size().y,instance.get_size().x,instance.get_size().z]#xyz to whl
        num_lidar_pts = 0
        num_radar_pts = 0
        for sensor in self.sensors:
            if sensor.bp_name == 'sensor.lidar.ray_cast':
                num_lidar_pts += self.get_num_lidar_pts(instance,sensor.get_last_data(),sensor.get_transform())
            elif sensor.bp_name == 'sensor.other.radar':
                num_radar_pts += self.get_num_radar_pts(instance,sensor.get_last_data(),sensor.get_transform())
        return instance_token,visibility_token,attribute_tokens,translation,rotation,size,num_lidar_pts,num_radar_pts

    def get_visibility(self,instance):
        max_visible_point_count = 0
        for sensor in self.sensors:
            if sensor.bp_name == 'sensor.lidar.ray_cast':
                ego_position = sensor.get_transform().location
                ego_position.z += self.ego_vehicle.get_size().z*0.5
                instance_position = instance.get_transform().location
                visible_point_count1 = 0
                visible_point_count2 = 0
                for i in range(5):
                    size = instance.get_size()
                    size.z = 0
                    check_point = instance_position-(i-2)*size*0.5
                    ray_points =  self.world.cast_ray(ego_position,check_point)
                    points = list(filter(lambda point:not self.ego_vehicle.get_actor().bounding_box.contains(point.location,self.ego_vehicle.get_actor().get_transform()) 
                                        and not instance.get_actor().bounding_box.contains(point.location,instance.get_actor().get_transform()) 
                                        and point.label is not carla.libcarla.CityObjectLabel.NONE,ray_points))
                    if not points:
                        visible_point_count1+=1
                    size.x = -size.x
                    check_point = instance_position-(i-2)*size*0.5
                    ray_points =  self.world.cast_ray(ego_position,check_point)
                    points = list(filter(lambda point:not self.ego_vehicle.get_actor().bounding_box.contains(point.location,self.ego_vehicle.get_actor().get_transform()) 
                                        and not instance.get_actor().bounding_box.contains(point.location,instance.get_actor().get_transform()) 
                                        and point.label is not carla.libcarla.CityObjectLabel.NONE,ray_points))
                    if not points:
                        visible_point_count2+=1
                if max(visible_point_count1,visible_point_count2)>max_visible_point_count:
                    max_visible_point_count = max(visible_point_count1,visible_point_count2)
        visibility_dict = {0:0,1:1,2:1,3:2,4:3,5:4}
        return visibility_dict[max_visible_point_count]

    def get_attributes(self,instance):
        return self.attribute_dict[instance.bp_name]

    def get_num_lidar_pts(self,instance,lidar_data,lidar_transform):
        num_lidar_pts = 0
        if lidar_data is not None:
            for data in lidar_data[1]:
                point = lidar_transform.transform(data.point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_lidar_pts+=1
        return num_lidar_pts

    def get_num_radar_pts(self,instance,radar_data,radar_transform):
        num_radar_pts = 0
        if radar_data is not None:
            for data in radar_data[1]:
                point = carla.Location(data.depth*math.cos(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.azimuth)
                        )
                point = radar_transform.transform(point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_radar_pts+=1
        return num_radar_pts

    def get_random_weather(self):
        weather_param = {
            "cloudiness":clamp(random.gauss(0,30)),
            "sun_azimuth_angle":random.random()*360,
            "sun_altitude_angle":random.random()*120-30,
            "precipitation":clamp(random.gauss(0,30)),
            "precipitation_deposits":clamp(random.gauss(0,30)),
            "wind_intensity":random.random()*100,
            "fog_density":clamp(random.gauss(0,30)),
            "fog_distance":random.random()*100,
            "wetness":clamp(random.gauss(0,30)),
            "fog_falloff":random.random()*5,
            "scattering_intensity":max(random.random()*2-1,0),
            "mie_scattering_scale":max(random.random()*2-1,0),
            "rayleigh_scattering_scale":max(random.random()*2-1,0),
            "dust_storm":clamp(random.gauss(0,30))
        }
        return weather_param

    