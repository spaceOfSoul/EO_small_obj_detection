"""2단계: RW(Random Walker) 기반 로컬 세그멘테이션.

후보 픽셀 중심의 window x window 영역을 121(=11x11 기본)개 노드로 하는 그래프를 구성한다.
- 중심 픽셀 -> 카테고리 0 (표적 시드)
- 최외곽 테두리 픽셀 -> 중심 기준 각도로 8등분해 카테고리 1~8 (배경 시드)
- 나머지 -> 미라벨

그래프 위상(노드/엣지/카테고리 배정)은 window 크기에만 의존하므로 1회 계산 후 캐싱하고,
호출마다 실제 픽셀 밝기로 정해지는 엣지 가중치만 새로 계산해 선형시스템을 푼다.
"""
from functools import lru_cache

import numpy as np

N_CATEGORIES = 9  # 0(표적) + 1~8(배경 8방향)


@lru_cache(maxsize=None)
def _build_topology(window: int) -> dict:
    n = window * window
    center = window // 2

    rows, cols = np.meshgrid(np.arange(window), np.arange(window), indexing="ij")
    rows = rows.ravel()
    cols = cols.ravel()

    is_center = (rows == center) & (cols == center)
    is_border = (rows == 0) | (rows == window - 1) | (cols == 0) | (cols == window - 1)

    category = np.full(n, -1, dtype=np.int64)
    category[is_center] = 0

    border_idx = np.nonzero(is_border & ~is_center)[0]
    dy = rows[border_idx] - center
    dx = cols[border_idx] - center
    angle = np.arctan2(dy, dx)  # (-pi, pi]
    sector = np.floor((angle + np.pi / 8) / (np.pi / 4)).astype(np.int64) % 8
    category[border_idx] = sector + 1

    labeled_idx = np.nonzero(category >= 0)[0]
    unlabeled_idx = np.nonzero(category < 0)[0]

    node_idx = np.arange(n).reshape(window, window)
    edge_i = np.concatenate([node_idx[:, :-1].ravel(), node_idx[:-1, :].ravel()])
    edge_j = np.concatenate([node_idx[:, 1:].ravel(), node_idx[1:, :].ravel()])

    p_m = np.zeros((labeled_idx.size, N_CATEGORIES))
    p_m[np.arange(labeled_idx.size), category[labeled_idx]] = 1.0

    return {
        "n": n,
        "labeled_idx": labeled_idx,
        "unlabeled_idx": unlabeled_idx,
        "edge_i": edge_i,
        "edge_j": edge_j,
        "p_m": p_m,
    }


def random_walker_probs(patch: np.ndarray, beta: float) -> np.ndarray:
    """patch: (window, window) grayscale 배열.

    returns: (window*window, 9) 확률 행렬. 행 순서는 patch.ravel()과 동일(행 우선).
    """
    window = patch.shape[0]
    topo = _build_topology(window)
    g = patch.astype(np.float64).ravel()
    g_norm = g / 255.0  # beta=15 권장값은 [0,1] 정규화 밝기 기준(RW 가중치 관례)이라 정규화 후 사용

    edge_i, edge_j = topo["edge_i"], topo["edge_j"]
    # 완전히 평평한 영역에서 가중치가 전부 0으로 언더플로우해 L_U가 특이행렬이 되는 것을 막기 위한 최소 가중치
    w = np.exp(-beta * (g_norm[edge_i] - g_norm[edge_j]) ** 2) + 1e-8

    n = topo["n"]
    laplacian = np.zeros((n, n))
    laplacian[edge_i, edge_j] -= w
    laplacian[edge_j, edge_i] -= w
    deg = np.zeros(n)
    np.add.at(deg, edge_i, w)
    np.add.at(deg, edge_j, w)
    laplacian[np.arange(n), np.arange(n)] = deg

    labeled_idx = topo["labeled_idx"]
    unlabeled_idx = topo["unlabeled_idx"]
    p_m = topo["p_m"]

    b = laplacian[np.ix_(labeled_idx, unlabeled_idx)]  # L_M,U
    l_u = laplacian[np.ix_(unlabeled_idx, unlabeled_idx)]

    p_u = np.linalg.solve(l_u, -(b.T @ p_m))

    prob = np.zeros((n, N_CATEGORIES))
    prob[labeled_idx] = p_m
    prob[unlabeled_idx] = p_u
    return prob
