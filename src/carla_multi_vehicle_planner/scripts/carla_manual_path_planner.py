#!/usr/bin/env python3
"""
CARLA Manual Path Planner Tool

This tool loads a CARLA map and allows users to manually select waypoints
to create a global path by clicking on the map. The path can be saved as a YAML file
and loaded for ego_vehicle_2 to follow.

Usage:
    rosrun carla_multi_vehicle_planner carla_manual_path_planner.py
    or
    python3 carla_manual_path_planner.py --carla_host localhost --carla_port 2000 --output_path path.yaml
"""

import argparse
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Ensure CARLA API is on sys.path
try:
    from setup_carla_path import *  # noqa: F401,F403
except Exception:
    pass

try:
    import carla  # type: ignore
except ImportError as exc:
    print(f"Failed to import CARLA: {exc}")
    print("Please ensure CARLA Python API is available.")
    sys.exit(1)


class CarlaManualPathPlanner:
    def __init__(self, host: str = "localhost", port: int = 2000, map_name: Optional[str] = None, output_file: Optional[str] = None):
        """Initialize CARLA client and load map."""
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        try:
            world = self.client.get_world()
            if map_name:
                self.client.load_world(map_name)
                world = self.client.get_world()
            
            self.map = world.get_map()
            self.world = world
        except Exception as e:
            print(f"Failed to connect to CARLA or load map: {e}")
            sys.exit(1)
        
        # Get map boundaries from waypoints
        waypoints = self.map.generate_waypoints(2.0)
        if not waypoints:
            print("No waypoints found in map. Cannot determine map boundaries.")
            sys.exit(1)
        
        x_coords = [wp.transform.location.x for wp in waypoints]
        y_coords = [wp.transform.location.y for wp in waypoints]
        
        self.min_x = min(x_coords)
        self.max_x = max(x_coords)
        self.min_y = min(y_coords)
        self.max_y = max(y_coords)
        
        self.map_width = self.max_x - self.min_x
        self.map_height = self.max_y - self.min_y
        
        # Image dimensions (adjust for better visualization)
        self.image_width = 1200
        self.image_height = int(self.image_width * (self.map_height / self.map_width))
        
        # Path points: list of (x, y, yaw) tuples
        self.path_points: List[Tuple[float, float, float]] = []
        
        # Nearest waypoint cache for finding closest waypoint to click
        waypoint_resolution = 0.5
        self._waypoint_cache = [
            wp for wp in self.map.generate_waypoints(waypoint_resolution)
            if wp.lane_type == carla.LaneType.Driving
        ]
        
        # Output file
        if output_file:
            self.output_file = output_file
        else:
            # Default: save in current directory with map name and timestamp
            map_name_safe = self.map.name.replace("/", "_") if self.map.name else "unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"manual_path_{map_name_safe}_{timestamp}.yaml"
        
        print(f"Map loaded: {self.map.name}")
        print(f"Map bounds: x=[{self.min_x:.1f}, {self.max_x:.1f}], y=[{self.min_y:.1f}, {self.max_y:.1f}]")
        print(f"Image size: {self.image_width}x{self.image_height}")
        print(f"Output file: {self.output_file}")
        print("\nInstructions:")
        print("  - Left Click: Add waypoint to path")
        print("  - Right Click: Remove last waypoint")
        print("  - 'c': Clear all waypoints")
        print("  - 's': Save path to YAML file")
        print("  - 'r': Toggle route smoothing (auto-connect between waypoints)")
        print("  - 'q' or ESC: Quit")
        print("  - Mouse wheel: Zoom in/out (future feature)")
        
        # Settings
        self.route_smoothing = True  # Auto-connect waypoints using CARLA route planner
        
        # Mouse state
        self.mouse_pos = None
        
    def world_to_image(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert CARLA world coordinates to image pixel coordinates."""
        # Flip x-axis as requested in previous tool
        norm_x = (world_x - self.min_x) / self.map_width
        norm_y = (world_y - self.min_y) / self.map_height
        
        # Flip x-axis
        img_x = int((1.0 - norm_x) * self.image_width)
        img_y = int(norm_y * self.image_height)
        
        return (img_x, img_y)
    
    def image_to_world(self, img_x: int, img_y: int) -> Tuple[float, float]:
        """Convert image pixel coordinates to CARLA world coordinates."""
        norm_x = 1.0 - (float(img_x) / float(self.image_width))
        norm_y = float(img_y) / float(self.image_height)
        
        world_x = self.min_x + norm_x * self.map_width
        world_y = self.min_y + norm_y * self.map_height
        
        return (world_x, world_y)
    
    def find_nearest_waypoint(self, world_x: float, world_y: float) -> Optional[carla.Waypoint]:
        """Find the nearest driving waypoint to the given world coordinates."""
        if not self._waypoint_cache:
            return None
        
        min_dist = float('inf')
        nearest_wp = None
        
        for wp in self._waypoint_cache:
            loc = wp.transform.location
            dist = math.hypot(loc.x - world_x, loc.y - world_y)
            if dist < min_dist:
                min_dist = dist
                nearest_wp = wp
        
        return nearest_wp
    
    def trace_route_between_waypoints(self, start_wp: carla.Waypoint, end_wp: carla.Waypoint) -> List[Tuple[float, float, float]]:
        """Trace route between two waypoints using CARLA route planner."""
        try:
            # Use CARLA's route planner
            route = self.map.get_waypoint(start_wp.transform.location).next_until_lane_end(2.0)
            
            # Simple approach: just connect start and end with intermediate waypoints
            # For better results, use GlobalRoutePlanner if available
            start_loc = start_wp.transform.location
            end_loc = end_wp.transform.location
            
            # If waypoints are on the same road, try to connect them
            if abs(start_wp.road_id - end_wp.road_id) < 10:  # Same or nearby road
                # Try to find intermediate waypoints
                current_wp = start_wp
                route_points = []
                
                # Add start point
                route_points.append((
                    start_loc.x, 
                    start_loc.y,
                    math.radians(start_wp.transform.rotation.yaw)
                ))
                
                max_iterations = 200
                iteration = 0
                target_dist = math.hypot(end_loc.x - start_loc.x, end_loc.y - start_loc.y)
                
                while iteration < max_iterations:
                    # Get next waypoints
                    next_wps = current_wp.next(2.0)
                    if not next_wps:
                        break
                    
                    # Choose waypoint closest to target
                    best_wp = None
                    best_score = float('inf')
                    
                    for next_wp in next_wps:
                        next_loc = next_wp.transform.location
                        dist_to_end = math.hypot(next_loc.x - end_loc.x, next_loc.y - end_loc.y)
                        dist_from_start = math.hypot(next_loc.x - start_loc.x, next_loc.y - start_loc.y)
                        
                        # Prefer waypoints that get closer to target without going too far
                        if dist_from_start > target_dist * 1.5:
                            continue
                        
                        score = dist_to_end
                        if score < best_score:
                            best_score = score
                            best_wp = next_wp
                    
                    if best_wp is None:
                        break
                    
                    best_loc = best_wp.transform.location
                    route_points.append((
                        best_loc.x,
                        best_loc.y,
                        math.radians(best_wp.transform.rotation.yaw)
                    ))
                    
                    # Check if we're close enough to target
                    if math.hypot(best_loc.x - end_loc.x, best_loc.y - end_loc.y) < 5.0:
                        break
                    
                    current_wp = best_wp
                    iteration += 1
                
                # Add end point
                route_points.append((
                    end_loc.x,
                    end_loc.y,
                    math.radians(end_wp.transform.rotation.yaw)
                ))
                
                return route_points
            else:
                # Different roads - just return start and end
                return [
                    (start_loc.x, start_loc.y, math.radians(start_wp.transform.rotation.yaw)),
                    (end_loc.x, end_loc.y, math.radians(end_wp.transform.rotation.yaw))
                ]
        except Exception as e:
            print(f"Warning: Failed to trace route between waypoints: {e}")
            # Fallback: just return start and end points
            start_loc = start_wp.transform.location
            end_loc = end_wp.transform.location
            return [
                (start_loc.x, start_loc.y, math.radians(start_wp.transform.rotation.yaw)),
                (end_loc.x, end_loc.y, math.radians(end_wp.transform.rotation.yaw))
            ]
    
    def draw_map(self) -> np.ndarray:
        """Draw the CARLA map with waypoints and current path."""
        # Create white background
        img = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8) * 255
        
        # Draw map waypoints as gray lines
        waypoint_resolution = 2.0
        waypoints = self.map.generate_waypoints(waypoint_resolution)
        
        drawn_segments = set()
        for wp in waypoints:
            if wp.lane_type != carla.LaneType.Driving:
                continue
            
            loc = wp.transform.location
            img_x, img_y = self.world_to_image(loc.x, loc.y)
            
            # Draw waypoint as small circle
            cv2.circle(img, (img_x, img_y), 2, (200, 200, 200), -1)
            
            # Draw connection to next waypoint
            next_wps = wp.next(waypoint_resolution)
            if next_wps:
                for next_wp in next_wps:
                    next_loc = next_wp.transform.location
                    next_img_x, next_img_y = self.world_to_image(next_loc.x, next_loc.y)
                    
                    # Avoid drawing duplicate segments
                    segment_key = tuple(sorted([(img_x, img_y), (next_img_x, next_img_y)]))
                    if segment_key in drawn_segments:
                        continue
                    drawn_segments.add(segment_key)
                    
                    cv2.line(img, (img_x, img_y), (next_img_x, next_img_y), (180, 180, 180), 1)
        
        # Draw current path
        if len(self.path_points) > 0:
            path_img_points = []
            for x, y, yaw in self.path_points:
                img_x, img_y = self.world_to_image(x, y)
                path_img_points.append((img_x, img_y))
            
            # Draw path as thick blue line
            if len(path_img_points) > 1:
                for i in range(len(path_img_points) - 1):
                    cv2.line(img, path_img_points[i], path_img_points[i + 1], (0, 0, 255), 3)
            
            # Draw waypoints as red circles
            for img_x, img_y in path_img_points:
                cv2.circle(img, (img_x, img_y), 5, (0, 0, 255), -1)
                cv2.circle(img, (img_x, img_y), 8, (0, 0, 255), 2)
        
        # Draw mouse position indicator
        if self.mouse_pos:
            mx, my = self.mouse_pos
            cv2.circle(img, (mx, my), 3, (0, 255, 0), -1)
            cv2.circle(img, (mx, my), 10, (0, 255, 0), 1)
        
        # Draw info text
        info_text = [
            f"Path points: {len(self.path_points)}",
            f"Route smoothing: {'ON' if self.route_smoothing else 'OFF'}",
            "Left Click: Add waypoint | Right Click: Remove | 'c': Clear | 's': Save | 'q': Quit"
        ]
        y_offset = 25
        for text in info_text:
            cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += 25
        
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: add waypoint
            world_x, world_y = self.image_to_world(x, y)
            
            # Find nearest waypoint
            nearest_wp = self.find_nearest_waypoint(world_x, world_y)
            if nearest_wp:
                loc = nearest_wp.transform.location
                yaw_rad = math.radians(nearest_wp.transform.rotation.yaw)
                
                # Check if route smoothing is enabled
                if self.route_smoothing and len(self.path_points) > 0:
                    # Get last waypoint
                    last_x, last_y, last_yaw = self.path_points[-1]
                    last_wp = self.find_nearest_waypoint(last_x, last_y)
                    
                    if last_wp:
                        # Trace route between last and current waypoint
                        intermediate_points = self.trace_route_between_waypoints(last_wp, nearest_wp)
                        # Remove first point (already in path_points)
                        if len(intermediate_points) > 0:
                            intermediate_points = intermediate_points[1:]
                        # Add intermediate points
                        self.path_points.extend(intermediate_points)
                    else:
                        # Fallback: just add the waypoint
                        self.path_points.append((loc.x, loc.y, yaw_rad))
                else:
                    # No smoothing: just add the waypoint
                    self.path_points.append((loc.x, loc.y, yaw_rad))
                
                print(f"Added waypoint {len(self.path_points)}: ({loc.x:.2f}, {loc.y:.2f}), yaw={math.degrees(yaw_rad):.1f}deg")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: remove last waypoint
            if len(self.path_points) > 0:
                removed = self.path_points.pop()
                print(f"Removed waypoint: ({removed[0]:.2f}, {removed[1]:.2f})")
    
    def save_path(self):
        """Save path to YAML file."""
        if len(self.path_points) < 2:
            print("Error: Need at least 2 waypoints to save a path.")
            return False
        
        path_data = {
            "map_name": self.map.name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_points": len(self.path_points),
            "loop_path": False,  # User can set this to True for closed loop paths
            "path": []
        }
        
        for x, y, yaw in self.path_points:
            path_data["path"].append({
                "x": float(x),
                "y": float(y),
                "yaw_rad": float(yaw),
                "yaw_deg": float(math.degrees(yaw))
            })
        
        # Calculate total path length
        total_length = 0.0
        for i in range(len(self.path_points) - 1):
            x1, y1, _ = self.path_points[i]
            x2, y2, _ = self.path_points[i + 1]
            total_length += math.hypot(x2 - x1, y2 - y1)
        path_data["total_length_m"] = float(total_length)
        
        try:
            with open(self.output_file, 'w') as f:
                yaml.dump(path_data, f, default_flow_style=False, sort_keys=False)
            print(f"\n✓ Path saved to: {self.output_file}")
            print(f"  - Points: {len(self.path_points)}")
            print(f"  - Total length: {total_length:.2f}m")
            return True
        except Exception as e:
            print(f"Error saving path: {e}")
            return False
    
    def run(self):
        """Main loop."""
        window_name = "CARLA Manual Path Planner"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\nStarting path planner...")
        
        while True:
            img = self.draw_map()
            cv2.imshow(window_name, img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('c'):
                self.path_points.clear()
                print("Path cleared.")
            elif key == ord('s'):
                self.save_path()
            elif key == ord('r'):
                self.route_smoothing = not self.route_smoothing
                print(f"Route smoothing: {'ON' if self.route_smoothing else 'OFF'}")
        
        cv2.destroyAllWindows()
        
        # Ask to save before quitting
        if len(self.path_points) >= 2:
            save = input("\nSave path before quitting? (y/n): ").strip().lower()
            if save == 'y':
                self.save_path()


def main():
    parser = argparse.ArgumentParser(description="CARLA Manual Path Planner")
    parser.add_argument("--carla_host", type=str, default="localhost", help="CARLA server host")
    parser.add_argument("--carla_port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--map_name", type=str, default=None, help="CARLA map name to load")
    parser.add_argument("--output_path", type=str, default=None, help="Output YAML file path")
    
    args = parser.parse_args()
    
    planner = CarlaManualPathPlanner(
        host=args.carla_host,
        port=args.carla_port,
        map_name=args.map_name,
        output_file=args.output_path
    )
    
    planner.run()


if __name__ == "__main__":
    main()
