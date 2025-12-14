import psipp
import carla
import setup_carla_path
import math
from collections import defaultdict, deque

host = "localhost"
port = 2000
timeout = 10.0

client = carla.Client(host, port)
client.set_timeout(timeout)
world = client.get_world()
carla_map = world.get_map()

# ===== 로드맵 생성 =====
WAYPOINT_SPACING = 2.0
print(f"Building connected roadmap with {WAYPOINT_SPACING}m spacing...")

spawn_points = carla_map.get_spawn_points()
print(f"Starting from {len(spawn_points)} spawn points")

visited_wp_ids = set()
wp_queue = deque()
all_waypoints = []

for sp in spawn_points:
    wp = carla_map.get_waypoint(sp.location)
    if wp.id not in visited_wp_ids:
        visited_wp_ids.add(wp.id)
        wp_queue.append((wp, None))

while wp_queue:
    wp, parent_id = wp_queue.popleft()
    all_waypoints.append((wp, parent_id))
    
    next_wps = wp.next(WAYPOINT_SPACING)
    for next_wp in next_wps:
        if next_wp.id not in visited_wp_ids:
            visited_wp_ids.add(next_wp.id)
            wp_queue.append((next_wp, wp.id))

print(f"Found {len(all_waypoints)} connected waypoints")

coord_precision = 1
coord_to_id = {}
vertices = []
edges = set()
wp_id_to_vertex_id = {}

for wp, parent_id in all_waypoints:
    loc = wp.transform.location
    key = (round(loc.x, coord_precision), round(loc.y, coord_precision))
    
    if key not in coord_to_id:
        vertex_id = len(vertices)
        coord_to_id[key] = vertex_id
        vertices.append((loc.x, loc.y))
    
    vertex_id = coord_to_id[key]
    wp_id_to_vertex_id[wp.id] = vertex_id
    
    if parent_id is not None and parent_id in wp_id_to_vertex_id:
        parent_vertex_id = wp_id_to_vertex_id[parent_id]
        if parent_vertex_id != vertex_id:
            edges.add((parent_vertex_id, vertex_id))

edges = list(edges)
print(f"Vertices: {len(vertices)}, Edges: {len(edges)}")

planner = psipp.Planner()
planner.set_roadmap(vertices, edges, radius=2.0)
print(f"Roadmap ready!")

def find_closest_vertex(x, y):
    min_dist = float('inf')
    closest_id = None
    for vid, (vx, vy) in enumerate(vertices):
        dist = math.sqrt((x - vx) ** 2 + (y - vy) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_id = vid
    return closest_id

# ===== 3대 차량 동시 계획 =====
print("\n" + "="*60)
print("=== 3-VEHICLE COLLISION-FREE PATH PLANNING ===")
print("="*60)

spawn_vertices = []
for sp in spawn_points:
    vid = find_closest_vertex(sp.location.x, sp.location.y)
    spawn_vertices.append((sp, vid))

# 3대 차량 설정
# Vehicle 0: spawn[0] -> spawn[5]
# Vehicle 1: spawn[2] -> spawn[6]
# Vehicle 2: spawn[4] -> spawn[1]

tasks = [
    (spawn_vertices[0][1], spawn_vertices[5][1]),  # Vehicle 0
    (spawn_vertices[2][1], spawn_vertices[6][1]),  # Vehicle 1
    (spawn_vertices[4][1], spawn_vertices[1][1]),  # Vehicle 2
]

print("\n[3 Vehicles Planning]")
for i, (start, goal) in enumerate(tasks):
    print(f"Vehicle {i}: ({vertices[start][0]:.1f}, {vertices[start][1]:.1f}) -> ({vertices[goal][0]:.1f}, {vertices[goal][1]:.1f})")

print("\nPlanning collision-free paths...")
plans = planner.plan(tasks)

# 결과
print("\n" + "="*60)
print("=== RESULTS ===")
print("="*60)

colors = [
    carla.Color(255, 0, 0),    # Red - Vehicle 0
    carla.Color(0, 255, 0),    # Green - Vehicle 1
    carla.Color(0, 0, 255),    # Blue - Vehicle 2
]
color_names = ["RED", "GREEN", "BLUE"]

debug = world.debug

for plan in plans:
    color = colors[plan.agent_id]
    color_name = color_names[plan.agent_id]
    
    print(f"\nVehicle {plan.agent_id} ({color_name}): {'SUCCESS' if plan.success else 'FAILED'}")
    
    if plan.success:
        print(f"  Path length: {len(plan.vertex_path)} vertices")
        print(f"  Arrival time: {plan.arrival_time:.2f}s")
        
        total_dist = sum(
            math.sqrt((vertices[plan.vertex_path[i]][0] - vertices[plan.vertex_path[i+1]][0])**2 +
                     (vertices[plan.vertex_path[i]][1] - vertices[plan.vertex_path[i+1]][1])**2)
            for i in range(len(plan.vertex_path)-1)
        )
        print(f"  Total distance: {total_dist:.2f}m")
        
        path = plan.vertex_path
        z_offset = 0.5 + plan.agent_id * 0.3
        
        for i in range(len(path) - 1):
            v1 = vertices[path[i]]
            v2 = vertices[path[i + 1]]
            debug.draw_line(
                carla.Location(x=v1[0], y=v1[1], z=z_offset),
                carla.Location(x=v2[0], y=v2[1], z=z_offset),
                thickness=0.2, color=color, life_time=60.0
            )
        
        # 시작점
        debug.draw_point(
            carla.Location(x=vertices[path[0]][0], y=vertices[path[0]][1], z=1.5),
            size=0.4, color=carla.Color(255, 255, 255), life_time=60.0
        )
        # 목표점
        debug.draw_point(
            carla.Location(x=vertices[path[-1]][0], y=vertices[path[-1]][1], z=1.5),
            size=0.4, color=color, life_time=60.0
        )

print("\n" + "="*60)
print("VISUALIZATION:")
print("  RED path   = Vehicle 0")
print("  GREEN path = Vehicle 1")
print("  BLUE path  = Vehicle 2")
print("  WHITE points = Start positions")
print("  Colored points = Goal positions")
print("  Paths visible for 60 seconds")
print("="*60)

# 충돌 분석
if all(p.success for p in plans):
    print("\n[Collision Analysis]")
    
    all_paths = [set(p.vertex_path) for p in plans]
    
    for i in range(len(plans)):
        for j in range(i+1, len(plans)):
            shared = all_paths[i] & all_paths[j]
            if shared:
                print(f"  Vehicle {i} & Vehicle {j}: {len(shared)} shared vertices (time-separated)")
            else:
                print(f"  Vehicle {i} & Vehicle {j}: No spatial overlap")
