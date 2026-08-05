"""3단계: PLLCM 계산 (확률가중 대비 + 크기 제약)."""
import numpy as np

from pllcm.params import PLLCMParams


def compute_pllcm(patch: np.ndarray, prob: np.ndarray, params: PLLCMParams) -> float:
    """patch: (window, window) grayscale 배열. prob: random_walker_probs()의 (window*window, 9) 결과."""
    g = patch.astype(np.float64).ravel()

    col_sums = prob.sum(axis=0)  # 카테고리별 확률 합 (길이 9)
    if (col_sums <= 0).any():
        return 0.0

    weighted_mean = (prob * g[:, None]).sum(axis=0) / col_sums
    m_t = weighted_mean[0]
    m_bk = weighted_mean[1:]

    diffs = m_t - m_bk
    if (diffs <= 0).any():
        return 0.0
    pllcm_pw = float(diffs.min())

    labels = prob.argmax(axis=1).reshape(patch.shape)
    rows, cols = np.nonzero(labels == 0)
    if rows.size == 0:
        return 0.0

    h_size = int(rows.max() - rows.min() + 1)
    w_size = int(cols.max() - cols.min() + 1)
    is_valid_size = h_size * w_size > params.size_min and h_size <= params.size_max and w_size <= params.size_max
    pllcm_sc = 1.0 if is_valid_size else 0.0

    return pllcm_pw * pllcm_sc
