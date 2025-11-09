import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import argparse
import yaml

def get_projection_matrix(fx, fy, cx, cy, cam_pos, cam_rot):
    """
    fx, fy, cx, cy : intrinsic parameters
    cam_pos : (x, y, z) camera position in CARLA world
    cam_rot : (roll, pitch, yaw) in degrees (CARLA 기준)
    """

    # --- Intrinsic matrix ---
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # --- CARLA → OpenCV 좌표계 변환 ---
    R_carla2cam = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    # --- 카메라 회전행렬 (world→carla 기준) ---
    # CARLA는 (roll, pitch, yaw) = (X, Y, Z)
    R_world2carla = R.from_euler('xyz', cam_rot, degrees=True).as_matrix().T

    # --- world→camera 회전 ---
    R_world2cam = R_carla2cam @ R_world2carla

    # --- translation (world→camera) ---
    t_world2cam = -R_world2cam @ np.array(cam_pos).reshape(3, 1)

    # --- 최종 extrinsic [R|t] ---
    extrinsic = np.hstack((R_world2cam, t_world2cam))

    # --- 투영행렬 (3x4) ---
    P = K @ extrinsic

    return P

cam_infos = {
    1: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    2: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    3: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    4: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    5: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    6: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    7: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    8: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    9: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    10: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    11: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    12: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    13: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },

    14: {
        'ip': '192.168.0.20',
        'frame_id': 'ipcam_1',
        'intrin': [653.54682, 653.54682, 640.0, 360.0], # fx, fy, cx, cy
        'pos'   : [0.38, -40.4, 8.55],                  # x, y, z
        'rot'   : [0.0, 24.34, 68.85]                   # roll, pitch, yaw
    },
}

config_dir = '/home/ctrl/capstone/src/ip_camera/config'
for cam_id in range(1, 15):
    yaml_path = os.path.join(config_dir, f'ipcam_{cam_id}.yaml')
    if not os.path.exists(yaml_path):
        print(f'ipcam_{cam_id}.yaml 파일이 없음')
        break

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    H_world2img = get_projection_matrix(
        fx=cam_infos[cam_id]['intrin'][0],
        fy=cam_infos[cam_id]['intrin'][1],
        cx=cam_infos[cam_id]['intrin'][2],
        cy=cam_infos[cam_id]['intrin'][3],
        cam_pos=cam_infos[cam_id]['pos'], # x, y, z
        cam_rot=cam_infos[cam_id]['rot']  # roll, pitch, yaw
    )

    homograpy = H_world2img[:, [0,1,3]]
    H = np.linalg.inv(homograpy)
    H /= H[2,2]

    data['camera_id'] = cam_id
    data['frame_id'] = f'ipcam_{cam_id}'
    data['ip'] = cam_infos[cam_id]['ip']
    data['H'] = [[float(x) for x in row] for row in H]

    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

