#!/usr/bin/env python3
import numpy as np

import math
import random
import json
from typing import Dict, List, Optional, Tuple

import rospy
from capstone_msgs.msg import PathMeta  # type: ignore
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from capstone_msgs.msg import PathMeta
from std_msgs.msg import Header, Float32, String

from frenet_path import FrenetPath

try:
    from capstone_msgs.msg import Uplink  # type: ignore
except Exception:
    Uplink = None  # type: ignore

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
    from agents.navigation.global_route_planner import GlobalRoutePlanner as CARLAGlobalRoutePlanner  # type: ignore
except Exception:
    CARLAGlobalRoutePlanner = None  # type: ignore

try:
    from global_planner import GlobalPlanner
except Exception:
    GlobalPlanner = None  # type: ignore

try:
    from obstacle_planner import ObstaclePlanner
except Exception:
    ObstaclePlanner = None  # type: ignore

# 차량 색상 맵핑 (ego_vehicle_1 ~ ego_vehicle_5)
VEHICLE_COLORS = ["red", "yellow", "green", "black", "white"]

def _get_vehicle_color(role: str) -> str:
    """ego_vehicle_N → 색상 이름 변환"""
    try:
        idx = int(role[-1]) - 1
        return VEHICLE_COLORS[idx] if 0 <= idx < len(VEHICLE_COLORS) else role
    except:
        return role


class SimpleMultiAgentPlanner:
    """
    최소 기능 다중 차량 글로벌 플래너.
    - 각 ego 차량의 현재 전방 위치를 시작점으로 사용
    - 스폰 포인트 중 거리 제약을 만족하는 목적지 임의 선택
    - CARLA GlobalRoutePlanner로 경로 생성 후 /global_path_{role} 퍼블리시
    - 충돌 회피, 수동 목표, 오프셋/시각화 등 부가 기능은 제거
    """
    # 안녕 강욱아 나는 성진이야 지금은 1227 0824이고 암튼 그래
    def __init__(self) -> None:
        rospy.init_node("multi_agent_conflict_free_planner", anonymous=True)

        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 5))
        self.platoon_enable = bool(rospy.get_param("~platoon_enable", False))
        self.leader_role = str(rospy.get_param("~leader_role", "ego_vehicle_1"))
        self.global_route_resolution = float(rospy.get_param("~global_route_resolution", 1.0))
        self.path_thin_min_m = float(rospy.get_param("~path_thin_min_m", 0.1))            # default denser than 0.2
        self.replan_soft_distance_m = float(rospy.get_param("~replan_soft_distance_m", 20.0))
        self.replan_check_interval = float(rospy.get_param("~replan_check_interval", 0.01))
        # Align first path segment with vehicle heading by looking slightly ahead when replanning
        self.heading_align_lookahead_m = float(rospy.get_param("~heading_align_lookahead_m", 2.5))
        
        # Start waypoint/path stitch parameters
        self.start_join_max_gap_m = float(rospy.get_param("~start_join_max_gap_m", 12.0))
        self.start_offset_m = float(rospy.get_param("~start_offset_m", 3.0))
        self.path_extension_overlap_m = float(rospy.get_param("~path_extension_overlap_m", 30.0))
        self.max_extend_attempts = int(rospy.get_param("~max_extend_attempts", 3))
        self.min_destination_distance = float(rospy.get_param("~min_destination_distance", 70.0))
        self.max_destination_distance = float(rospy.get_param("~max_destination_distance", 100.0))
        
        # Low-voltage handling (forced destination / parking)
        self.low_voltage_threshold = float(rospy.get_param("~low_voltage_threshold", 5.0))
        self.low_voltage_dest_x = float(rospy.get_param("~low_voltage_dest_x", -12.5))
        self.low_voltage_dest_y = float(rospy.get_param("~low_voltage_dest_y", -16.5))
        self.parking_dest_x = float(rospy.get_param("~parking_dest_x", -27.0))
        self.parking_dest_y = float(rospy.get_param("~parking_dest_y", -16.5))
        self.parking_trigger_distance_m = float(rospy.get_param("~parking_trigger_distance_m", 0.5))
        
         # Destination memory per role (for distance-gated replanning)
        self._current_dest: Dict[str, Optional[carla.Location]] = {self._role_name(i): None for i in range(self.num_vehicles)}
        self._active_paths: Dict[str, List[Tuple[float, float]]] = {}
        self._original_paths: Dict[str, List[Tuple[float, float]]] = {}  # 회피 적용 전 원본 경로
        self._active_path_s: Dict[str, List[float]] = {}
        self._active_path_len: Dict[str, float] = {}
        self._voltage: Dict[int, float] = {}

        # CARLA world/map
        host = rospy.get_param("~carla_host", "localhost")
        port = int(rospy.get_param("~carla_port", 2000))
        timeout = float(rospy.get_param("~carla_timeout", 10.0))
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        # Route planner (GlobalPlanner 우선, 없으면 CARLA GlobalRoutePlanner)
        self.route_planner = GlobalPlanner(self.carla_map, self.global_route_resolution)
        
        # ObstaclePlanner 초기화 (route_planner 설정 후)
        self._obstacle_planner = None
        if ObstaclePlanner is not None:
            self._obstacle_planner = ObstaclePlanner(
                self.route_planner, 
                self.num_vehicles,
                on_obstacle_change_callback=self._on_obstacle_change
            )
            rospy.loginfo("SimpleMultiAgentPlanner: ObstaclePlanner initialized")

        self._backup_blocked_path: Dict[str, List[Tuple[int, int]]] = {}
        self._should_stop_pos: Dict[str, carla.Location] = {}

        self.spawn_points = self.carla_map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        # Publishers
        self.path_publishers: Dict[str, rospy.Publisher] = {}
        self.path_meta_publishers: Dict[str, rospy.Publisher] = {}
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/global_path_{role}"
            self.path_publishers[role] = rospy.Publisher(topic, Path, queue_size=1, latch=True)

            meta_topic = f"/global_path_meta_{role}"
            self.path_meta_publishers[role] = rospy.Publisher(meta_topic, PathMeta, queue_size=1, latch=True)

        # Destination memory per role (for distance-gated replanning)
        self._current_dest: Dict[str, Optional[carla.Location]] = {self._role_name(i): None for i in range(self.num_vehicles)}
        self._active_paths: Dict[str, List[Tuple[float, float]]] = {}
        self._active_path_s: Dict[str, List[float]] = {}
        self._active_path_len: Dict[str, float] = {}
        self._voltage: Dict[int, float] = {}
        # Low-voltage 단계 관리: idle -> to_buffer(low_voltage_dest) -> to_parking(parking_dest) -> parked
        self._lv_stage: Dict[str, str] = {self._role_name(i): "idle" for i in range(self.num_vehicles)}
        # 수동 목적지 오버라이드 (PoseStamped에서 받음)
        self._override_goal: Dict[str, Optional[carla.Location]] = {self._role_name(i): None for i in range(self.num_vehicles)}
        self._override_active: Dict[str, bool] = {self._role_name(i): False for i in range(self.num_vehicles)}
        self.override_clear_radius = float(rospy.get_param("~override_clear_radius", 3.0))
        self.override_hold_sec = float(rospy.get_param("~override_hold_sec", 5.0))
        self._override_hold_until: Dict[str, float] = {self._role_name(i): 0.0 for i in range(self.num_vehicles)}
        # seq 에 epoch을 더해 경로 재퍼블리시 시 절대 증가하도록 유지
        self._seq_epoch: Dict[str, int] = {self._role_name(i): 0 for i in range(self.num_vehicles)}
        # 너무 큰 점프를 피하기 위해 기본 50만큼만 여유를 둔다 (경로 길이 + stride)
        self.seq_epoch_stride = int(rospy.get_param("~seq_epoch_stride", 50))

        # Uplink subscriber (voltage)
        self.uplink_topic = str(rospy.get_param("~uplink_topic", "/uplink"))
        if Uplink is not None and self.uplink_topic:
            rospy.Subscriber(self.uplink_topic, Uplink, self._uplink_cb, queue_size=10)

        # Override goal subscribers per vehicle
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/override_goal/{role}"
            rospy.Subscriber(topic, PoseStamped, self._override_goal_cb, callback_args=role, queue_size=1)

        # 신호등별 정지 차량 정보 (컨트롤러에서 발행)
        self._vehicles_at_tl: Dict[str, List[str]] = {}
        rospy.Subscriber("/vehicles_at_tl", String, self._vehicles_at_tl_cb, queue_size=1)

        rospy.sleep(0.5)
        self._plan_once()
        rospy.Timer(rospy.Duration(self.replan_check_interval), self._replan_check_cb)
    
    def _role_name(self, index: int) -> str:
        return f"ego_vehicle_{index + 1}"

    def _vehicles_at_tl_cb(self, msg: String) -> None:
        """컨트롤러에서 발행하는 신호등별 정지 차량 정보 수신"""
        try:
            self._vehicles_at_tl = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[Planner] Failed to parse vehicles_at_tl: {e}")

    def _is_vehicle_at_tl(self, role: str) -> Optional[str]:
        """해당 차량이 신호등에서 대기 중인지 확인
        
        Returns:
            대기 중인 신호등 이름, 없으면 None
        """
        for signal_name, roles in self._vehicles_at_tl.items():
            if role in roles:
                return signal_name
        return None

    def _on_obstacle_change(self) -> None:
        """장애물 변화 시 즉시 호출되는 콜백 - 모든 활성 경로에 회피 적용"""
        vehicles = self._get_ego_vehicles()
        
        if not vehicles: return
        if self._obstacle_planner is None: return
        
        for index, vehicle in enumerate(vehicles[:self.num_vehicles]):
            role = self._role_name(index)
            
            # 원본 경로 사용 (없으면 현재 경로 사용)
            path = self._backup_blocked_path.get(role)
            if not path or len(path) < 2:
                path = self._original_paths.get(role)
            
            if not path or len(path) < 2:
                path = self._active_paths.get(role)

            if not path or len(path) < 2:
                continue

            # 경로에 중복 제거 (원본 경로 구조 유지)
            path = self._unique_points(path)
            
            # 장애물 탐지
            obstacles_on_path = self._obstacle_planner._find_obstacle_on_path(path, is_frenet=False)
            
            if not obstacles_on_path:
                # 장애물 없음 -> 원본 경로로 복구
                if role in self._obstacle_planner._obstacle_blocked_roles:
                    self._obstacle_planner._obstacle_blocked_roles.pop(role)

                rospy.loginfo(f"[RECOVERY] {role}: recovered to original path (no obstacles)")
                self._publish_path(path, role, "normal", [], [])
                self._store_active_path(role, path)
                self._original_paths[role] = list(path)  # 원본 경로 갱신
                
                if role in self._backup_blocked_path:
                    self._backup_blocked_path.pop(role)

            else:
                # 장애물 있음 -> 회피 경로 생성
                rospy.loginfo(f"[RECOVERY] {role}: {len(obstacles_on_path)} obstacle(s) detected, generating avoidance")
                modified_path, s_starts, s_ends = self._obstacle_planner.apply_avoidance_to_path(
                    role, path, obstacles_on_path
                )
                
                if modified_path is not None and len(modified_path) >= 2:
                    rospy.loginfo(f"[RECOVERY] {role}: avoidance path generated with {len(modified_path)} points")
                    self._publish_path(modified_path, role, "obstacle", s_starts, s_ends)
                    self._store_active_path(role, modified_path)
                    # 원본은 유지 (다음 복구 시 사용)
                else:
                    rospy.logwarn(f"[RECOVERY] {role}: avoidance path generation failed, keeping original")
                    self._publish_path(path, role, "normal", [], [])
                    self._store_active_path(role, path)

    def has_conflict_opposite(self, role: str, stop_pos: carla.Location, d_offset: float, s_start: float, s_end: float) -> bool:
        """
        좌표 기반 반대 차선 충돌 감지 (Active Path & S-Profile 기반)
        
        Args:
            role: 현재 차량 role
            stop_pos: 회피 시작 위치 (정지 위치 - 장애물 근처로 간주)
            d_offset: 회피 방향 횡방향 오프셋 (단순 참조용, 실제 판단에는 안 씀)
            avoidance_length: 회피 경로 길이
        
        Returns:
            True if should wait, False if can proceed
        """
        # 내 경로 정보 가져오기
        my_path = self._active_paths.get(role)
        if not my_path or len(my_path) < 2:
            if role in self._backup_blocked_path:
                my_path = self._backup_blocked_path[role]

            else:
                return False

        # 2. 회피 시작 지점(stop_pos)의 s 좌표 계산
        safe_distance = (s_end - s_start) * 2.5
        my_color = _get_vehicle_color(role)
        
        vehicles = self._get_ego_vehicles()
        for vehicle in vehicles:
            other_role = vehicle.attributes.get("role_name", "")
            
            # 자기 자신 제외
            if other_role == role:
                continue

            # 타 차량의 Active Path 가져오기
            other_path = self._active_paths.get(other_role)
            if not other_path or len(other_path) < 2:
                continue

            # 1. 장애물(stop_pos)과 타 차량 경로 간의 거리 확인
            #    거리 > 1.0m 이면 충돌 가능성 있는 차량(반대 차선 등)으로 간주
            other_frenet_path = FrenetPath(other_path)
            other_loc = self._vehicle_rear(vehicle) # 전방 위치 기준

            # 회피 시작 지점의 s 및 d 좌표 계산 (타 차량 경로 기준)
            s_obs_on_other, d_obs_on_other = other_frenet_path.cartesian_to_frenet(stop_pos.x, stop_pos.y)
            
            # 타 차량 위치의 s 및 d 좌표 계산 (타 차량 경로 기준)
            s_other_vehicle_on_other, _ = other_frenet_path.cartesian_to_frenet(other_loc.x, other_loc.y)

            # 타 차량 경로 기준 장애물이 3 ~ 6m 수직 거리로 떨어져 있으면 == 반대 차선에서 오는 괘씸한 녀석이다.
            if 3 < abs(d_obs_on_other) < 6:
                # 회피 경로의 2배 이상 거리가 나지 않으면 정지
                remain_s_on_other = s_obs_on_other - s_other_vehicle_on_other
                if -1 < remain_s_on_other <= safe_distance:
                    # 최소 안전 거리 이내에 있고, 신호에 걸리지 않은 녀석이라면 -> 정지
                    return self._is_vehicle_at_tl(other_role) is None
        
        return False

    def _get_vehicle_by_role(self, role: str):
        """role 이름으로 차량 액터 찾기"""
        for vehicle in self._get_ego_vehicles():
            if vehicle.attributes.get("role_name", "") == role:
                return vehicle
        return None
        
    def _replan_check_cb(self, _evt) -> None:
        # Distance-gated replanning per vehicle
        vehicles = self._get_ego_vehicles()
        if not vehicles:
            return
    
        for index, vehicle in enumerate(vehicles[: self.num_vehicles]):
            role = self._role_name(index)
            
            dest_override = None
            front_loc = self._vehicle_front(vehicle)
            active_path = self._active_paths.get(role)
            
            # 우회 불가하지만, 정지해야 하는 녀석
            if self._obstacle_planner and role in self._obstacle_planner._obstacle_blocked_roles:
                # 정지 지점과 가까워 진 경우
                color = _get_vehicle_color(role)
                stop_pos, d_offset, s_start, s_end = self._obstacle_planner.get_stop_pos(role)
                
                chk_path = active_path if active_path and len(active_path) >= 2 else self._backup_blocked_path[role]
                s_on_mine, d_on_mine = FrenetPath(chk_path).cartesian_to_frenet(front_loc.x, front_loc.y)

                if -1.0 <= (s_start - s_on_mine) <= 1.0:
                    # 좌표 기반 충돌 검사 (회피 경로 길이의 2.5배 이내면 대기)
                    if d_offset is None or self.has_conflict_opposite(role, stop_pos, d_offset, s_start, s_end):
                        if role not in self._backup_blocked_path:
                            self._backup_blocked_path[role] = self._active_paths.get(role)

                        stop_pts = [(front_loc.x, front_loc.y)]
                        self._publish_path(stop_pts, role, "stop")
                        self._store_active_path(role, stop_pts)
                        continue
                    
                    # 처음부터 정지하지 않아도 됐거나 정지 상황이 해제됐거나 타 차량이 반대 차선을 점유하지 않는 경우
                    else:
                        self._obstacle_planner.stop_done(role)
                        
                        # 만약에 백업해둔 경로가 있다면 복구하기
                        if role in self._backup_blocked_path:
                            self._publish_path(self._backup_blocked_path.get(role), role, "stop")
                            self._store_active_path(role, self._backup_blocked_path.get(role))
                            self._backup_blocked_path.pop(role)

                        continue

            # 사용자가 임의 목적지를 설정한 경우
            elif self._override_active.get(role, False):
                category = "llm"

                # 임의 목적지로 향하고 있는 경우
                if self._override_goal.get(role) is not None:
                    if front_loc.distance(self._override_goal[role]) <= self.override_clear_radius:
                        self._override_goal[role] = None
                        hold_t = rospy.Time.now().to_sec() + max(0.0, float(self.override_hold_sec))
                        self._override_hold_until[role] = hold_t
                        
                        stop_pts = [(front_loc.x, front_loc.y)]
                        rospy.logfatal(f"[STOP] {role}: 임의 목적지 도착 대기 중")
                        self._publish_path(stop_pts, role, "stop")
                        self._store_active_path(role, stop_pts)
                        rospy.loginfo(f"{role}: override goal reached -> hold for {self.override_hold_sec:.1f}s")

                    else:
                        dest_override = self._override_goal[role]
                        current_path = self._active_paths.get(role)
                
                        if current_path and len(current_path) >= 2:
                            # 현재 경로가 사용자의 임의 목적지와 먼 경우에만 재계획하기
                            if math.hypot(current_path[-1][0] - dest_override.x, current_path[-1][1] - dest_override.y) > self.start_join_max_gap_m:
                                prefix = current_path[:-1] if len(current_path) > 1 else current_path
                                if not self._plan_for_role(vehicle, role, front_loc, category, prefix_points=prefix, dest_override=dest_override, force_direct=False):
                                    rospy.logwarn_throttle(5.0, f"{role}: failed to append override path")
                        
                        else:
                            # 경로가 없으면 현재 위치 기준으로 계획
                            if not self._plan_for_role(vehicle, role, front_loc, category, dest_override=dest_override, force_direct=False):
                                rospy.logwarn_throttle(5.0, f"{role}: failed to plan override path")

                # 도착해서 대기 중인 경우
                else:
                    if rospy.Time.now().to_sec() >= self._override_hold_until[role]:
                        self._override_hold_until[role] = 0.0
                        self._override_active[role] = False

                continue

            # override가 없을 때만 저전압/일반 로직 수행
            elif self._is_low_voltage(role):
                category = "battery"

                stage = self._lv_stage.get(role, "idle")
                
                # Distance to buffer and parking targets (front of vehicle)
                dist_buffer = math.hypot(front_loc.x - float(self.low_voltage_dest_x), front_loc.y - float(self.low_voltage_dest_y))
                dist_parking = math.hypot(front_loc.x - float(self.parking_dest_x), front_loc.y - float(self.parking_dest_y))
                
                # 1) idle -> buffer dest (low_voltage_dest)
                if stage == "idle":
                    dest_override = carla.Location(x=float(self.low_voltage_dest_x), y=float(self.low_voltage_dest_y), z=front_loc.z)
                    if self._plan_for_role(vehicle, role, front_loc, category, dest_override=dest_override, force_direct=False):
                        rospy.loginfo(f"{role}: LOW-V start -> buffer ({self.low_voltage_dest_x:.2f},{self.low_voltage_dest_y:.2f})")
                        self._lv_stage[role] = "to_buffer"
                    else:
                        rospy.logwarn_throttle(5.0, f"{role}: low-voltage buffer plan failed")
                
                # 2) heading to buffer: 모니터링하다가 여유거리 도달 시 최종 주차 목적지로 재계획
                elif stage == "to_buffer":
                    remaining = self._remaining_path_distance(role, front_loc)
                    # Use either remaining arc-length or actual distance-to-target to trigger parking leg
                    trigger_dist = max(0.0, float(self.parking_trigger_distance_m))
                    if remaining is None:
                        dest_override = carla.Location(x=float(self.low_voltage_dest_x), y=float(self.low_voltage_dest_y), z=front_loc.z)
                        self._plan_for_role(vehicle, role, front_loc, category, dest_override=dest_override, force_direct=False)
                
                    elif remaining <= trigger_dist or dist_buffer <= trigger_dist:
                        dest_override = carla.Location(x=float(self.parking_dest_x), y=float(self.parking_dest_y), z=front_loc.z)
                        if self._plan_for_role(vehicle, role, front_loc, category, dest_override=dest_override, force_direct=True):
                            rospy.loginfo(f"{role}: LOW-V buffer reached (remaining {remaining:.1f} m, dist {dist_buffer:.1f} m) -> parking ({self.parking_dest_x:.2f},{self.parking_dest_y:.2f})")
                            self._lv_stage[role] = "to_parking"
                        else:
                            rospy.logwarn_throttle(5.0, f"{role}: parking plan failed after buffer")
                
                # 3) heading to final parking: 도착 감지, 필요 시 복구
                elif stage == "to_parking":
                    remaining = self._remaining_path_distance(role, front_loc)
                    if remaining is None:
                        dest_override = carla.Location(x=float(self.parking_dest_x), y=float(self.parking_dest_y), z=front_loc.z)
                        self._plan_for_role(vehicle, role, front_loc, category, dest_override=dest_override, force_direct=True)
                    
                    elif remaining <= 1.0 or dist_parking <= 1.5:
                        rospy.loginfo(f"{role}: LOW-V parking reached (remaining {remaining:.1f} m, dist {dist_parking:.1f} m) -> parked")
                        self._lv_stage[role] = "parked"
                    
                # 4) parked: 더 이상 계획 안 함
                elif stage == "parked":
                    pass
                
                continue

            category = "normal"

            current_path = self._active_paths.get(role)
            fail_inital_path = not current_path or len(current_path) < 2

            remaining = self._remaining_path_distance(role, front_loc)
            fail_remaining_none = remaining is None

            # (현재 Path가 없거나 매우 짧음) 이거나 (남은 거리 계산 불가) -> 전체 재계획
            if fail_inital_path or fail_remaining_none:
                if not self._plan_for_role(vehicle, role, front_loc, category):
                    msg = "failed to plan initial path" if fail_inital_path else "failed to replan after progress loss"
                    rospy.logwarn_throttle(5.0, f"{role}: {msg}")
            
            # 남은 거리가 소프트 임계값 이하 -> 경로 연장 시도
            elif remaining <= max(0.0, float(self.replan_soft_distance_m)):
                rospy.loginfo(f"{role}: remaining {remaining:.1f} m <= soft {self.replan_soft_distance_m:.1f} m -> path extension")
                if not self._extend_path(vehicle, role, front_loc, category):
                    rospy.logwarn_throttle(5.0, f"{role}: path extension failed; forcing fresh plan")
                    self._plan_for_role(vehicle, role, front_loc, category)

    def _extend_path(self, vehicle, role: str, front_loc: carla.Location, category: str, dest_override: Optional[carla.Location] = None) -> bool:
        current = self._active_paths.get(role)
        if not current or len(current) < 2:
            rospy.logwarn_throttle(5.0, f"{role}: no active path to extend; full replan")
            return self._plan_for_role(vehicle, role, front_loc, category, dest_override=dest_override)
        
        s_profile = self._active_path_s.get(role)
        suffix = current
        if s_profile and len(s_profile) == len(current):
            progress = self._project_progress_on_path(current, s_profile, front_loc.x, front_loc.y)
            if progress is not None:
                overlap_target = max(0.0, progress - float(self.path_extension_overlap_m))
                start_idx = 0
                for i, s in enumerate(s_profile):
                    if s >= overlap_target:
                        start_idx = i
                        break
                suffix = current[start_idx:]
        attempts = 0
        
        while attempts < max(1, int(self.max_extend_attempts)):
            prefix_copy = list(suffix)
            if len(prefix_copy) < 2:
                rospy.logwarn_throttle(5.0, f"{role}: prefix too short during extension; replanning")
                return self._plan_for_role(vehicle, role, front_loc, category)
            
            remaining = self._remaining_path_distance(role, front_loc)
            if remaining is not None:
                rospy.loginfo(f"{role}: extending path (remaining {remaining:.1f} m, overlap {self.path_extension_overlap_m} m, attempt {attempts + 1})")
            
            if self._plan_for_role(vehicle, role, front_loc, category, prefix_points=prefix_copy):
                return True
            
            attempts += 1
            rospy.logwarn_throttle(5.0, f"{role}: path extension attempt {attempts} failed; retrying")
        
        rospy.logwarn_throttle(5.0, f"{role}: all extension attempts failed; falling back to fresh plan")
        return self._plan_for_role(vehicle, role, front_loc, category)

    def _get_ego_vehicles(self) -> List[carla.Actor]:
        actors = self.world.get_actors().filter("vehicle.*")
        vehicles: List[carla.Actor] = []
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role.startswith("ego_vehicle_"):
                vehicles.append(actor)
        vehicles.sort(key=lambda v: v.attributes.get("role_name", ""))
        return vehicles

    def _uplink_cb(self, msg: "Uplink") -> None:
        try:
            self._voltage[int(msg.vehicle_id)] = float(msg.voltage)
        except Exception:
            pass

    def _is_low_voltage(self, role: str) -> bool:
        try:
            vid = int(role.split("_")[-1])
            v = self._voltage.get(vid, float("inf"))
            return v <= float(self.low_voltage_threshold)
        except Exception:
            return False

    def _vehicle_front(self, vehicle: carla.Actor) -> carla.Location:
        tf = vehicle.get_transform()
        yaw_rad = math.radians(tf.rotation.yaw)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        offset = max(0.5, float(self.start_offset_m))
        bb = getattr(vehicle, "bounding_box", None)
        if bb is not None and getattr(bb, "extent", None) is not None:
            offset = bb.extent.x + 0.3
        return carla.Location(
            x=tf.location.x + forward_x * offset,
            y=tf.location.y + forward_y * offset,
            z=tf.location.z
        )

    def _vehicle_rear(self, vehicle: carla.Actor) -> carla.Location:
        """차량 후륜축 위치 반환"""
        tf = vehicle.get_transform()
        yaw_rad = math.radians(tf.rotation.yaw)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        # 후륜축은 차량 중심에서 뒤로 (bounding box 절반 - 약간)
        offset = 1.0  # 기본값
        bb = getattr(vehicle, "bounding_box", None)
        if bb is not None and getattr(bb, "extent", None) is not None:
            offset = bb.extent.x * 0.5  # 차량 길이의 1/4 정도 뒤
        return carla.Location(
            x=tf.location.x - forward_x * offset,
            y=tf.location.y - forward_y * offset,
            z=tf.location.z
        )

    def _choose_destination(self, start: carla.Location, max_trials: int=200) -> Optional[carla.Location]:
        for _ in range(max_trials):
            cand = random.choice(self.spawn_points).location
            dist = math.hypot(cand.x - start.x, cand.y - start.y)
            if self.min_destination_distance <= dist <= self.max_destination_distance:
                return cand
        return None

    def _trace_route(self, start: carla.Location, dest: carla.Location):
        """
        경로 탐색
        
        Returns:
            route 또는 None
        """
        try:
            route = self.route_planner.trace_route(start, dest)
            return route
        
        except Exception as exc:
            rospy.logwarn(f"trace_route failed: {exc}")
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
            # Thin only when spacing exceeds configured threshold
            if dx * dx + dy * dy >= float(self.path_thin_min_m) * float(self.path_thin_min_m):
                points.append((x, y))
                last_x, last_y = x, y
        
        return points

    def _straight_line_points(self, start_loc: carla.Location, dest_loc: carla.Location, spacing: float = 1.0) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        dx = float(dest_loc.x) - float(start_loc.x)
        dy = float(dest_loc.y) - float(start_loc.y)
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return [(float(start_loc.x), float(start_loc.y)), (float(dest_loc.x), float(dest_loc.y))]
        steps = max(1, int(dist / max(0.1, spacing)))
        for i in range(steps + 1):
            t = float(i) / float(steps)
            x = float(start_loc.x) + dx * t
            y = float(start_loc.y) + dy * t
            points.append((x, y))
        return points

    def _ensure_path_starts_at_vehicle(self, path_points: List[Tuple[float, float]], front_xy: Tuple[float, float]):
        if not path_points:
            return []
        gap = math.hypot(front_xy[0] - path_points[0][0], front_xy[1] - path_points[0][1])
        max_gap = max(0.0, float(self.start_join_max_gap_m))
        if gap < 0.1:
            path_points[0] = front_xy
            return path_points
        if gap > max_gap:
            # Insert an intermediate point along heading direction before original path
            first_x, first_y = path_points[0]
            vec_x = first_x - front_xy[0]
            vec_y = first_y - front_xy[1]
            dist = math.hypot(vec_x, vec_y)
            if dist > 1e-3:
                span = min(dist, float(self.heading_align_lookahead_m))
                yaw = math.atan2(vec_y, vec_x)
                mid = (front_xy[0] + math.cos(yaw) * span, front_xy[1] + math.sin(yaw) * span)
                return [front_xy, mid] + path_points
        return [front_xy] + path_points

    def _compute_path_profile(self, points: List[Tuple[float, float]]):
        if len(points) < 2:
            return [], 0.0
        
        cumulative = [0.0]
        total = 0.0
        for i in range(1, len(points)):
            step = math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            total += step
            cumulative.append(total)
        return cumulative, total

    def _project_progress_on_path(self, points: List[Tuple[float, float]], s_profile: List[float], px: float, py: float):
        if len(points) < 2: return None
        
        if not s_profile or len(s_profile) != len(points):
            s_profile, _ = self._compute_path_profile(points)
            if not s_profile or len(s_profile) != len(points):
                return None
        
        best_dist_sq = float("inf")
        best_index = None
        best_t = 0.0
        
        for idx in range(len(points) - 1):
            x1, y1 = points[idx]
            x2, y2 = points[idx + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
        
            if seg_len_sq < 1e-6:
                continue
        
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + dx * t
            proj_y = y1 + dy * t
            dist_sq = (proj_x - px) ** 2 + (proj_y - py) ** 2
        
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_index = idx
                best_t = t
        
        if best_index is None:
            rospy.logwarn_throttle(2.0, "SimpleMultiAgentPlanner: unable to project progress (disjoint path)")
            return None
        
        seg_length = math.hypot(points[best_index + 1][0] - points[best_index][0], points[best_index + 1][1] - points[best_index][1])
        s_now = s_profile[best_index] + best_t * seg_length * int(seg_length >= 1e-6)
        return s_now

    def _remaining_path_distance(self, role: str, front_loc: carla.Location):
        points = self._active_paths.get(role)
        if not points or len(points) < 2:
            return None
        s_profile = self._active_path_s.get(role) or []
        progress = self._project_progress_on_path(points, s_profile, front_loc.x, front_loc.y)
        if progress is None:
            rospy.logwarn_throttle(2.0, f"{role}: progress projection failed; forcing replan fallback")
            return None
        path_len = self._active_path_len.get(role, 0.0)
        remaining = path_len - progress
        if remaining < 0.0:
            remaining = 0.0
        return remaining

    def _store_active_path(self, role: str, points: List[Tuple[float, float]]):
        if len(points) < 2:
            self._active_paths.pop(role, None)
            self._active_path_s.pop(role, None)
            self._active_path_len.pop(role, None)
            return
        
        s_profile, total_len = self._compute_path_profile(points)
        self._active_paths[role] = points
        self._active_path_s[role] = s_profile
        self._active_path_len[role] = total_len

    def _publish_path(self, points: List[Tuple[float, float]], role: str, category: str, s_starts: List[float]=[], s_ends: List[float]=[]) -> None:
        # 플래툰 모드에서는 리더 외에는 경로를 내보내지 않아 follower가 덮어쓰지 않도록 함
        if self.platoon_enable and role != self.leader_role:
            return
        
        if role not in self.path_publishers:
            return

        msg = Path()
        msg.header = Header(frame_id="map", stamp=rospy.Time.now())
        epoch = self._seq_epoch.get(role, 0)
        for i, (x, y) in enumerate(points):
            p = PoseStamped()
            p.header = msg.header
            p.header.seq = epoch + i  # 웨이포인트 인덱스를 seq에 기록해 추종 시 사용 (절대 증가)
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = 0.0
            msg.poses.append(p)

        # 다음 퍼블리시를 위해 epoch 증가 (경로 길이 + stride)
        self._seq_epoch[role] = epoch + max(len(points), 1) + self.seq_epoch_stride
        self.path_publishers[role].publish(msg)
        
        if category != "stop":
            meta = PathMeta()
            meta.header = msg.header
            meta.resolution.data = 0.1
            meta.category.data = category
            meta.s_starts.data = s_starts
            meta.s_ends.data = s_ends

            self.path_meta_publishers[role].publish(meta)

    def _plan_once(self) -> None:
        vehicles = self._get_ego_vehicles()
        if not vehicles:
            rospy.loginfo_throttle(5.0, "No ego vehicles found")
            return
        for index, vehicle in enumerate(vehicles[: self.num_vehicles]):
            role = self._role_name(index)
            front_loc = self._vehicle_front(vehicle)
            self._plan_for_role(vehicle, role, front_loc, "normal")
    
    # 안녕 강욱아 나는 성진이야 지금은 1227 1022이고 암튼 그래 먼가 마니 만들엇네
    def _plan_for_role(self, 
        vehicle, role: str, front_loc: carla.Location, 
        category: str, prefix_points: Optional[List[Tuple[float, float]]] = None, 
        dest_override: Optional[carla.Location] = None, force_direct: bool = False) -> bool:
        # Sample (or resample) destination and publish a fresh path
        dest_loc = dest_override if dest_override is not None else self._choose_destination(front_loc)
        if dest_loc is None:
            rospy.logwarn_throttle(5.0, f"{role}: destination not found within distance bounds")
            return False
        
        # Detect off-road override target (e.g., parking off the lane)
        offroad_override = False
        if dest_override is not None:
            try:
                wp_test = self.carla_map.get_waypoint(dest_loc, project_to_road=False)
                if wp_test is None:
                    offroad_override = True
                    
            except Exception:
                offroad_override = True
        
        tf = vehicle.get_transform()
        start_loc, start_heading = None, math.radians(tf.rotation.yaw)
        if prefix_points is not None and len(prefix_points) >= 2:
            dx = prefix_points[-1][0] - prefix_points[-2][0]
            dy = prefix_points[-1][1] - prefix_points[-2][1]
            if dx * dx + dy * dy > 1e-3:
                start_heading = math.atan2(dy, dx)
        
        if prefix_points is not None and len(prefix_points) >= 1:
            last = prefix_points[-1]
            start_loc = carla.Location(x=last[0], y=last[1], z=front_loc.z)
        
        if start_loc is None:
            start_loc = carla.Location(x=front_loc.x + math.cos(start_heading) * float(self.heading_align_lookahead_m),
                                       y=front_loc.y + math.sin(start_heading) * float(self.heading_align_lookahead_m),
                                       z=front_loc.z)
        
        start_loc.z = front_loc.z
        route = None
        max_attempts = 5
        if not force_direct:
            for _ in range(max_attempts):
                route = self._trace_route(start_loc, dest_loc)
                if route and len(route) >= 2:
                    break

                dest_loc = self._choose_destination(front_loc)
                if dest_loc is None:
                    route = None
                    break
        
        # 1. 새 경로 생성
        new_points: List[Tuple[float, float]] = []
        if not force_direct and route and len(route) >= 2:
            new_points = self._route_to_points(route)

        else:
            new_points = self._straight_line_points(start_loc, dest_loc, spacing=max(0.5, float(self.path_thin_min_m)))
            rospy.logwarn_throttle(2.0, f"{role}: using direct line to dest ({dest_loc.x:.2f},{dest_loc.y:.2f}) (force_direct={force_direct}, route_ok={route is not None and len(route)>=2})")
        
        # 2. prefix 경로와 연결 
        if prefix_points is not None and len(prefix_points) >= 1:
            points = prefix_points[:-1] + new_points
        
        else:
            points = self._ensure_path_starts_at_vehicle(new_points, (front_loc.x, front_loc.y))

        if len(points) < 2:
            rospy.logwarn_throttle(5.0, f"{role}: insufficient path points")
            return False
        
        points = self._unique_points(points)
        points = self._densify_points(points, float(self.path_thin_min_m))

        obstacles_on_path = self._obstacle_planner._find_obstacle_on_path(points, is_frenet=False)
        rospy.logfatal(f"{_get_vehicle_color(role)}: obstacles_on_path={obstacles_on_path}")

        s_starts, s_ends = [], []
        original_points = list(points)
        if self._obstacle_planner is not None and not offroad_override and not force_direct:
            rospy.logwarn(f"{role}: obstacles on path: {obstacles_on_path}")
            if len(obstacles_on_path) > 0:
                points, s_starts, s_ends = self._obstacle_planner.apply_avoidance_to_path(role, points, obstacles_on_path)
                self._original_paths[role] = original_points

                rospy.loginfo(f"{role}: applied obstacle avoidance; new path has {s_starts, s_ends} points")

        rospy.logdebug(f"{role}: publishing path with {len(points)} points (prefix={'yes' if prefix_points else 'no'})")

        self._publish_path(points, role, category, s_starts, s_ends)
        self._store_active_path(role, points)        
        self._current_dest[role] = dest_loc
        return True

    def _unique_points(self, points):
        points = np.array(points)
        dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
        points = points[np.append([True], dist > 1e-6), :].tolist()
        
        return points

    def _densify_points(self, points: List[Tuple[float, float]], max_spacing: float) -> List[Tuple[float, float]]:
        """연속 점 사이 거리가 max_spacing을 초과하면 중간 점을 추가해 0.1m 수준으로 퍼블리시"""
        if len(points) < 2 or max_spacing <= 0.0:
            return points

        densified: List[Tuple[float, float]] = [points[0]]
        for i in range(1, len(points)):
            x1, y1 = densified[-1]
            x2, y2 = points[i]
            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy)
            if dist <= max_spacing:
                densified.append((x2, y2))
                continue

            steps = max(1, int(dist / max_spacing))
            for step in range(1, steps + 1):
                t = min(1.0, step / steps)
                densified.append((x1 + dx * t, y1 + dy * t))

        return densified

    def _override_goal_cb(self, msg: PoseStamped, role: str) -> None:
        try:
            loc = carla.Location(
                x=float(msg.pose.position.x),
                y=float(msg.pose.position.y),
                z=float(msg.pose.position.z),
            )
            self._override_goal[role] = loc
            self._override_active[role] = True
            rospy.loginfo(f"{role}: override goal set to ({loc.x:.2f},{loc.y:.2f})")

        except Exception as exc:
            rospy.logwarn_throttle(2.0, f"{role}: failed to set override goal: {exc}")

# 안녕 강욱아 나는 성진이야 지금은 1227 1230이고 암튼 그래 리뷰를 하진 못했어 그냥 여기까지왔어
# 안녕 강욱햄 나는 연지야 지금은 1227이고 암튼 그래 나는 박교수님 뵙고 올게   
# 안녕 강욱아 나는 문영이야 지금은 화성시 금성분 지구시야 암튼 그래 나는 잠을 자고 올게
# 안녕 강욱아 나는 수성이야 지금은 목성달 토성일 수성시 화성분이야. 암튼 그래 나는 우주에 다녀올게
# 안녕 강욱아 나는 동의야 지금은 동동시 동동분 동동초야. 암튼ㅁ 그래 나는 결혼하고 올게 

if __name__ == "__main__":
    try:
        SimpleMultiAgentPlanner()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"SimpleMultiAgentPlanner crashed: {e}")
        raise
