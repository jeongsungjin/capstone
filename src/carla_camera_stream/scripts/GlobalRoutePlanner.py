#!/usr/bin/env python3
import carla
import math

# CARLA 서버 연결
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()

# waypoint 생성 (도로 전체를 일정 간격으로 샘플링)
waypoints = carla_map.generate_waypoints(distance=1.0)  # 1~2m 간격 추천

print(f"총 웨이포인트 개수: {len(waypoints)}")

# 저장 함수
def save_waypoints_to_file(waypoints, file_path="carla_map_waypoints.txt"):
    with open(file_path, "w") as f:
        f.write("x,y,z,yaw,road_id,lane_id\n")
        for wp in waypoints:
            loc = wp.transform.location
            yaw = wp.transform.rotation.yaw
            f.write(f"{loc.x:.3f},{loc.y:.3f},{loc.z:.3f},{yaw:.3f},{wp.road_id},{wp.lane_id}\n")

    print(f"💾 저장 완료: {file_path}")


# 실행
save_waypoints_to_file(waypoints, "/home/ctrl/carla_map_waypoints.txt")
