#!/usr/bin/env python3

import sys
import os

# CARLA Python API 경로 설정
default_build_path = "/home/jamie/carla/PythonAPI/carla/build/lib.linux-x86_64-cpython-38"
CARLA_BUILD_PATH = os.environ.get("CARLA_BUILD_PATH", default_build_path)

if CARLA_BUILD_PATH and CARLA_BUILD_PATH not in sys.path:
    sys.path.insert(0, CARLA_BUILD_PATH)

try:
    import carla
except ImportError as exc:
    print(f"Failed to import CARLA: {exc}")
    sys.exit(1)

def cleanup_carla_actors():
    """CARLA에서 모든 차량과 액터를 제거"""
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        
        # 모든 차량 제거
        vehicles = world.get_actors().filter("vehicle.*")
        for vehicle in vehicles:
            vehicle.destroy()
        
        # 모든 보행자 제거
        walkers = world.get_actors().filter("walker.*")
        for walker in walkers:
            walker.destroy()
        
        # 모든 센서 제거
        sensors = world.get_actors().filter("sensor.*")
        for sensor in sensors:
            sensor.destroy()
        
        print(f"Cleaned up {len(vehicles)} vehicles, {len(walkers)} walkers, {len(sensors)} sensors")
        
    except Exception as e:
        print(f"Cleanup failed: {e}")

if __name__ == "__main__":
    cleanup_carla_actors()