import os
import json
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. 원본 path 로딩
# -----------------------------
custom_edge_path = (
    (8, 18), (10, 11), (13, 25), (16, 24), (20, 12),
    (21, 15), (21, 24), (23, 19), (23, 12), (26, 22)
)

pkg_path = '/home/jamie/capstone/src/perception/scripts'
edges = dict()

for u, v in custom_edge_path:
    edges[(u, v)] = []
    for i in range(1, 6):  # 각 edge별로 5개 파일
        file_path = os.path.join(
            pkg_path, 'data', f'{u}-{v}',
            f'recorded_path_id1_{u}to{v}_{i}.json'
        )
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        pts = np.array(data['points'])  # (N, 2) [x, y]
        edges[(u, v)].append(pts)
        print(f"{u}->{v}, run {i}, points: {len(pts)}")

# -----------------------------
# 2. 보간 + 평균 대표 경로 생성
# -----------------------------

N_SAMPLES = 200  # 각 경로를 이 개수로 재표본화
representative_paths = {}  # (u, v) -> (N_SAMPLES, 2)

def resample_by_arclength(path, n_samples=N_SAMPLES):
    """
    path: (N, 2) [x, y]
    n_samples: 보간 후 포인트 수
    """
    # 각 segment 길이
    deltas = np.diff(path, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))  # (N-1,)
    s = np.concatenate(([0.0], np.cumsum(seg_lengths)))  # (N,)

    total_length = s[-1]
    if total_length == 0:
        # 모든 점이 같을 때 대비
        return np.repeat(path[:1, :], n_samples, axis=0)

    # 0~1로 정규화
    s_norm = s / total_length

    # 공통 파라미터 (0~1)
    t = np.linspace(0.0, 1.0, n_samples)

    # x, y 각각 보간
    x_new = np.interp(t, s_norm, path[:, 0])
    y_new = np.interp(t, s_norm, path[:, 1])
    return np.stack([x_new, y_new], axis=1)  # (n_samples, 2)

# 경로별로 보간 후 평균
for (u, v), path_list in edges.items():
    resampled_list = []

    for run_idx, path in enumerate(path_list):
        resampled = resample_by_arclength(path, N_SAMPLES)
        resampled_list.append(resampled)

    resampled_array = np.stack(resampled_list, axis=0)  # (num_runs, N_SAMPLES, 2)

    # run dimension 에 대해 평균 → 대표 경로
    mean_path = resampled_array.mean(axis=0)  # (N_SAMPLES, 2)
    representative_paths[(u, v)] = mean_path

    # -----------------------------
    # 3. 디버깅용 플롯
    # -----------------------------
    plt.figure()
    # 각 주행 경로 (보간 후) 얇게
    for r in resampled_list:
        plt.plot(r[:, 0], r[:, 1], alpha=0.3, linewidth=1)

    # 평균 경로 굵게
    plt.plot(mean_path[:, 0], mean_path[:, 1], linewidth=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{u} -> {v} (resampled + mean)")
    plt.grid(True)
    plt.show()

# -----------------------------
# 4. 대표 경로 JSON으로 저장
# -----------------------------
out_dir = os.path.join(pkg_path, 'custom_paths')
os.makedirs(out_dir, exist_ok=True)

for (u, v), mean_path in representative_paths.items():
    out_file = os.path.join(out_dir, f'custom_path_{u}to{v}.json')
    obj = {
        "u": u,
        "v": v,
        "points": mean_path.tolist()
    }
    with open(out_file, 'w') as f:
        json.dump(obj, f, indent=2)
    print("Saved:", out_file)
