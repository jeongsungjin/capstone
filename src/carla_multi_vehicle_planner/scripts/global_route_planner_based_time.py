import carla
import networkx as nx
import math
from scipy.spatial import KDTree

class GlobalRoutePlannerBasedTime:
    def __init__(self, world_or_map):
        """
        초기화 함수: CARLA 맵 데이터를 받아 그래프와 검색 트리를 구축합니다.
        :param world_or_map: carla.World 객체 또는 carla.Map 객체
        """
        # 1. Map 객체 확보
        if isinstance(world_or_map, carla.World):
            self.map = world_or_map.get_map()
        else:
            self.map = world_or_map
            
        print("Creating Global Route Planner...")
        print(" - Loading map topology...")

        # 2. 토폴로지 추출 (CARLA API)
        # get_topology()는 도로의 각 세그먼트(시작점, 끝점) 리스트를 반환합니다.
        self.topology = self.map.get_topology()
        
        # 3. 그래프 및 데이터 구조 초기화
        self.graph = nx.DiGraph()  # 방향 그래프 (일방통행 고려)
        self.id_to_waypoint = {}   # Node ID -> Carla Waypoint 객체 (나중에 필요할 때 역참조용)
        self.node_ids = []         # KD-Tree와 매칭될 Node ID 리스트
        node_coords = []           # KD-Tree 구축용 (x, y) 좌표 리스트
        
        print(" - Building NetworkX graph...")

        # 4. 그래프 구축 루프 (Heavy Operation - 1회 수행)
        for segment in self.topology:
            w1 = segment[0] # 시작 웨이포인트
            w2 = segment[1] # 끝 웨이포인트
            
            # 노드 ID 생성 (간단하게 고유 ID 사용)
            n1_id = w1.id
            n2_id = w2.id
            
            # 좌표 추출 (x, y) - z는 경로계획에서 보통 무시
            n1_loc = (w1.transform.location.x, w1.transform.location.y)
            n2_loc = (w2.transform.location.x, w2.transform.location.y)

            # 그래프에 노드 추가 (좌표 정보 포함)
            if n1_id not in self.graph:
                self.graph.add_node(n1_id, vertex=n1_loc)
                self.id_to_waypoint[n1_id] = w1
                self.node_ids.append(n1_id)
                node_coords.append(n1_loc)
                
            if n2_id not in self.graph:
                self.graph.add_node(n2_id, vertex=n2_loc)
                self.id_to_waypoint[n2_id] = w2
                self.node_ids.append(n2_id)
                node_coords.append(n2_loc)

            # 엣지 가중치(거리) 계산
            dist = w1.transform.location.distance(w2.transform.location)
            
            # 그래프에 엣지 추가 (기본 가중치: 거리)
            self.graph.add_edge(n1_id, n2_id, weight=dist, length=dist)

        print(f" - Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

        # 5. 공간 인덱싱 (KD-Tree) 구축
        # 좌표 리스트를 트리 구조로 변환하여 검색 속도 O(log N) 확보
        self.tree = KDTree(node_coords)
        
        # 6. 예약 테이블 초기화 (Multi-Agent Time-based Planning용)
        # 구조: { (u, v): [(entry_unix_time, exit_unix_time, car_id), ...] }
        self.reservation_table = {} 
        
        print("Initialization Complete.")

    def get_closest_node_id(self, location):
        """
        주어진 CARLA Location과 가장 가까운 그래프 노드 ID를 반환
        """
        # KD-Tree query: (거리, 인덱스) 반환
        # location은 carla.Location 객체라고 가정
        query_loc = [location.x, location.y]
        _, index = self.tree.query(query_loc)
        
        return self.node_ids[index]

    def get_optimal_route(self, start_location, end_location):
        # Implement the logic here...
        pass