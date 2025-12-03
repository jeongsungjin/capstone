#!/usr/bin/env python3
"""
CARLA Map Region Selector Tool

This tool loads a CARLA map and allows users to select rectangular regions
by dragging the mouse. It returns the 4 corner coordinates of the selected region.

Usage:
    rosrun carla_multi_vehicle_planner carla_map_region_selector.py
    or
    python3 carla_map_region_selector.py --carla_host localhost --carla_port 2000
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


class CarlaMapRegionSelector:
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
        
        # Get map boundaries
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
        
        # Mouse interaction state
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
        # Regions: list of dicts with corners, speed_gain, steering_gain
        self.regions: List[Dict] = []  # List of selected regions with metadata
        # Default gains for new regions
        self.default_speed_gain = 1.0
        self.default_steering_gain = 1.0
        
        # Output file
        if output_file:
            self.output_file = output_file
        else:
            # Default: save in current directory with map name and timestamp
            map_name_safe = self.map.name.replace("/", "_") if self.map.name else "unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"carla_regions_{map_name_safe}_{timestamp}.yaml"
        
        print(f"Map loaded: {self.map.name}")
        print(f"Map bounds: x=[{self.min_x:.1f}, {self.max_x:.1f}], y=[{self.min_y:.1f}, {self.max_y:.1f}]")
        print(f"Image size: {self.image_width}x{self.image_height}")
        print("\nInstructions:")
        print("  - Left click and drag to select a rectangular region")
        print("  - Press 's' to save current selection")
        print("  - Press 'c' to clear current selection")
        print("  - Press 'r' to reset all selections")
        print("  - Press 'q' or ESC to quit and print all regions")
        print("  - Press 'p' to print current region coordinates")
        print(f"  - Regions will be saved to: {self.output_file}")
    
    def world_to_image(self, x: float, y: float) -> Tuple[int, int]:
        """Convert CARLA world coordinates to image pixel coordinates."""
        img_x = int((x - self.min_x) / self.map_width * self.image_width)
        img_y = int((self.max_y - y) / self.map_height * self.image_height)  # Flip Y axis
        # Flip X axis
        img_x = self.image_width - 1 - img_x
        return img_x, img_y
    
    def image_to_world(self, img_x: int, img_y: int) -> Tuple[float, float]:
        """Convert image pixel coordinates to CARLA world coordinates."""
        # Flip X axis first
        img_x = self.image_width - 1 - img_x
        x = self.min_x + (img_x / self.image_width) * self.map_width
        y = self.max_y - (img_y / self.image_height) * self.map_height  # Flip Y axis
        return x, y
    
    def render_map(self) -> np.ndarray:
        """Render CARLA map as an image."""
        # Create blank image
        img = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Draw waypoints (roads)
        waypoints = self.map.generate_waypoints(2.0)
        for wp in waypoints:
            loc = wp.transform.location
            img_x, img_y = self.world_to_image(loc.x, loc.y)
            if 0 <= img_x < self.image_width and 0 <= img_y < self.image_height:
                cv2.circle(img, (img_x, img_y), 1, (100, 100, 100), -1)
        
        # Draw lane markings
        for wp in waypoints:
            try:
                left_markings = wp.left_lane_markings
                right_markings = wp.right_lane_markings
                
                for marking in left_markings + right_markings:
                    loc = marking.location
                    img_x, img_y = self.world_to_image(loc.x, loc.y)
                    if 0 <= img_x < self.image_width and 0 <= img_y < self.image_height:
                        cv2.circle(img, (img_x, img_y), 1, (200, 200, 200), -1)
            except Exception:
                pass  # Some waypoints may not have lane markings
        
        # Draw selected regions
        for region in self.regions:
            corners = region.get("corners", [])
            if len(corners) == 4:
                pts = np.array([self.world_to_image(x, y) for x, y in corners], dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                # Fill with semi-transparent green
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                # Display region info
                region_id = region.get("id", "unknown")
                speed_gain = region.get("speed_gain", 1.0)
                steering_gain = region.get("steering_gain", 1.0)
                center_x = int(sum([self.world_to_image(x, y)[0] for x, y in corners]) / 4)
                center_y = int(sum([self.world_to_image(x, y)[1] for x, y in corners]) / 4)
                cv2.putText(img, f"{region_id} (v:{speed_gain:.1f}, s:{steering_gain:.1f})", 
                           (center_x - 50, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw current selection rectangle
        if self.start_point and self.end_point:
            cv2.rectangle(img, self.start_point, self.end_point, (0, 0, 255), 2)
        
        # Add text overlay
        cv2.putText(img, f"Regions: {len(self.regions)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "Drag to select | 's'=save | 'c'=clear | 'r'=reset | 'q'=quit", 
                   (10, self.image_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for rectangle selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.end_point = (x, y)
                self.drawing = False
    
    def get_current_region_coords(self) -> Optional[List[Tuple[float, float]]]:
        """Get the 4 corner coordinates of the current selection."""
        if not self.start_point or not self.end_point:
            return None
        
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Ensure proper rectangle (min/max)
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        
        # Convert to world coordinates
        corners = [
            self.image_to_world(min_x, min_y),  # Top-left
            self.image_to_world(max_x, min_y),  # Top-right
            self.image_to_world(max_x, max_y),  # Bottom-right
            self.image_to_world(min_x, max_y),  # Bottom-left
        ]
        
        return corners
    
    def print_region(self, corners: List[Tuple[float, float]], index: int):
        """Print region coordinates in a formatted way."""
        if index >= 0:
            print(f"\nRegion {index + 1}:")
        else:
            print("\nCurrent selection:")
        print("  Corners (x, y):")
        labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        for i, (x, y) in enumerate(corners):
            print(f"    {labels[i]}: ({x:.3f}, {y:.3f})")
        print(f"  Center: ({(corners[0][0] + corners[2][0]) / 2:.3f}, {(corners[0][1] + corners[2][1]) / 2:.3f})")
        print(f"  Width: {abs(corners[1][0] - corners[0][0]):.3f}m, Height: {abs(corners[2][1] - corners[0][1]):.3f}m")
    
    def run(self):
        """Main loop for interactive region selection."""
        window_name = "CARLA Map Region Selector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            img = self.render_map()
            cv2.imshow(window_name, img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Save current selection
                region_coords = self.get_current_region_coords()
                if region_coords:
                    region_dict = {
                        "id": f"region_{len(self.regions) + 1}",
                        "corners": region_coords,
                        "speed_gain": self.default_speed_gain,
                        "steering_gain": self.default_steering_gain,
                    }
                    self.regions.append(region_dict)
                    print(f"\nRegion {len(self.regions)} saved!")
                    print(f"  Speed gain: {self.default_speed_gain:.2f}, Steering gain: {self.default_steering_gain:.2f}")
                    self.print_region(region_coords, len(self.regions) - 1)
                    self.start_point = None
                    self.end_point = None
                else:
                    print("No region selected!")
            elif key == ord('c'):  # Clear current selection
                self.start_point = None
                self.end_point = None
                print("Current selection cleared.")
            elif key == ord('r'):  # Reset all
                self.regions.clear()
                self.start_point = None
                self.end_point = None
                print("All regions cleared.")
            elif key == ord('p'):  # Print current region
                region_coords = self.get_current_region_coords()
                if region_coords:
                    print("\nCurrent selection:")
                    self.print_region(region_coords, -1)
                    print(f"  Speed gain: {self.default_speed_gain:.2f} (default), Steering gain: {self.default_steering_gain:.2f} (default)")
                else:
                    print("No region selected!")
        
        cv2.destroyAllWindows()
        
        # Print all regions in a format suitable for configuration
        print("\n" + "="*60)
        print("All Selected Regions:")
        print("="*60)
        for i, region in enumerate(self.regions):
            corners = region.get("corners", [])
            if len(corners) >= 4:
                self.print_region(corners, i)
                print(f"  Speed gain: {region.get('speed_gain', 1.0):.2f}, Steering gain: {region.get('steering_gain', 1.0):.2f}")
        
        # Print in YAML/JSON-like format for easy copy-paste
        print("\n" + "="*60)
        print("Regions in YAML format:")
        print("="*60)
        print("regions:")
        for i, region in enumerate(self.regions):
            corners = region.get("corners", [])
            print(f"  - id: {region.get('id', f'region_{i+1}')}")
            print(f"    speed_gain: {region.get('speed_gain', 1.0):.2f}")
            print(f"    steering_gain: {region.get('steering_gain', 1.0):.2f}")
            print("    corners:")
            labels = ["top_left", "top_right", "bottom_right", "bottom_left"]
            for label, (x, y) in zip(labels, corners):
                print(f"      {label}: [{x:.3f}, {y:.3f}]")
            if len(corners) >= 4:
                center_x = (corners[0][0] + corners[2][0]) / 2
                center_y = (corners[0][1] + corners[2][1]) / 2
                print(f"    center: [{center_x:.3f}, {center_y:.3f}]")
        
        # Save to YAML file
        self.save_to_yaml()
    
    def save_to_yaml(self):
        """Save all regions to a YAML file."""
        yaml_data = {
            "map_name": self.map.name,
            "map_bounds": {
                "min_x": float(self.min_x),
                "max_x": float(self.max_x),
                "min_y": float(self.min_y),
                "max_y": float(self.max_y),
            },
            "regions": []
        }
        
        for i, region in enumerate(self.regions):
            corners = region.get("corners", [])
            if len(corners) < 4:
                continue
            region_data = {
                "id": region.get("id", f"region_{i+1}"),
                "speed_gain": float(region.get("speed_gain", self.default_speed_gain)),
                "steering_gain": float(region.get("steering_gain", self.default_steering_gain)),
                "corners": {
                    "top_left": [float(corners[0][0]), float(corners[0][1])],
                    "top_right": [float(corners[1][0]), float(corners[1][1])],
                    "bottom_right": [float(corners[2][0]), float(corners[2][1])],
                    "bottom_left": [float(corners[3][0]), float(corners[3][1])],
                },
                "center": [
                    float((corners[0][0] + corners[2][0]) / 2),
                    float((corners[0][1] + corners[2][1]) / 2)
                ],
                "width": float(abs(corners[1][0] - corners[0][0])),
                "height": float(abs(corners[2][1] - corners[0][1])),
            }
            yaml_data["regions"].append(region_data)
        
        # Get absolute path
        output_path = os.path.abspath(self.output_file)
        output_dir = os.path.dirname(output_path)
        
        # Create directory if it doesn't exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to file
        try:
            with open(output_path, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print("\n" + "="*60)
            print(f"Regions saved to: {output_path}")
            print("="*60)
        except Exception as e:
            print(f"\nError saving YAML file: {e}")
            print(f"Failed to save to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CARLA Map Region Selector")
    parser.add_argument("--carla_host", default="localhost", help="CARLA server host")
    parser.add_argument("--carla_port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--map", default=None, help="CARLA map name (optional)")
    parser.add_argument("--output", "-o", default=None, help="Output YAML file path (default: carla_regions_<map>_<timestamp>.yaml in current directory)")
    
    args = parser.parse_args()
    
    selector = CarlaMapRegionSelector(
        host=args.carla_host,
        port=args.carla_port,
        map_name=args.map,
        output_file=args.output
    )
    
    selector.run()


if __name__ == "__main__":
    main()


