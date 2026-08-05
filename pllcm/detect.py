"""4단계: 최종 탐지(적응형 임계값 + 연결요소 -> bbox) + 전체 파이프라인 오케스트레이션."""
import cv2
import numpy as np

from pllcm.candidate import extract_candidates
from pllcm.measure import compute_pllcm
from pllcm.params import PLLCMParams
from pllcm.segmentation import random_walker_probs

Detection = tuple[int, float, float, float, float, float]


def build_enhancement_map(img: np.ndarray, params: PLLCMParams) -> np.ndarray:
    """모든 후보 픽셀에 대해 2~3단계를 수행해 PLLCM 값을 담은 강화맵 E(희소)를 만든다."""
    h, w = img.shape[:2]
    e = np.zeros((h, w), dtype=np.float64)

    _, candidates, _ = extract_candidates(img, params)
    patch_margin = params.window // 2

    for r, c in candidates:
        if r - patch_margin < 0 or r + patch_margin >= h or c - patch_margin < 0 or c + patch_margin >= w:
            continue  # window/scales 파라미터를 따로 바꿔 margin이 어긋난 경우 방어
        patch = img[r - patch_margin : r + patch_margin + 1, c - patch_margin : c + patch_margin + 1]
        prob = random_walker_probs(patch, params.beta)
        e[r, c] = compute_pllcm(patch, prob, params)

    return e


def detect(array: np.ndarray, params: PLLCMParams | None = None) -> list[Detection]:
    """array: neg_array(.npy)에서 로드한 (H, W) grayscale numpy 배열.

    returns: (cls_id, x1, y1, x2, y2, conf) 픽셀 좌표 리스트.
    """
    params = params or PLLCMParams()
    e = build_enhancement_map(array, params)

    e_max = float(e.max())
    if e_max <= 0.0:
        return []

    th_e = params.lambda_th * e_max + (1 - params.lambda_th) * float(e.mean())
    mask = (e > th_e).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    detections: list[Detection] = []
    for label in range(1, num_labels):  # 라벨 0 = 배경
        x, y, bw, bh, _area = stats[label]
        region_e = e[y : y + bh, x : x + bw]
        conf = float(region_e.max() / e_max)
        detections.append((0, float(x), float(y), float(x + bw), float(y + bh), conf))

    return detections
