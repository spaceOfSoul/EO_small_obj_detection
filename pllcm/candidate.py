"""1단계: 후보 픽셀 추출 (Candidate Extraction).

멀티스케일 슬라이딩 윈도우 M_L(L=3,5,7,9,11)을 이미지에 적용해 필터링 이미지 F를 구하고,
F의 평균/표준편차 기반 적응형 임계값보다 밝은 픽셀만 후보로 채택한다.

M_L 커널은 중심(4L-4)과 최외곽 테두리(-1)만 값을 갖고 나머지는 0이므로, L×L 박스합에서
(L-2)×(L-2) 박스합을 뺀 "테두리합"과 동치다. 박스합은 적분영상(summed-area table)으로
스케일에 상관없이 O(1)/픽셀에 구해 scipy 없이도 빠르게 계산한다.
"""
import cv2
import numpy as np

from pllcm.params import PLLCMParams


def _box_sum(integral: np.ndarray, half: int, margin: int, h: int, w: int) -> np.ndarray:
    """(2*half+1)x(2*half+1) 박스합을, 이미지 내부 영역([margin, h-margin) x [margin, w-margin))의
    각 중심 픽셀에 대해 벡터화 계산. margin >= half이어야 박스가 이미지 범위를 벗어나지 않는다."""
    ra0, ra1 = margin - half, h - margin - half
    rb0, rb1 = margin + half + 1, h - margin + half + 1
    ca0, ca1 = margin - half, w - margin - half
    cb0, cb1 = margin + half + 1, w - margin + half + 1
    return (
        integral[rb0:rb1, cb0:cb1]
        - integral[ra0:ra1, cb0:cb1]
        - integral[rb0:rb1, ca0:ca1]
        + integral[ra0:ra1, ca0:ca1]
    )


def compute_f_interior(img: np.ndarray, scales: tuple[int, ...]) -> tuple[np.ndarray, int]:
    """이미지 내부 영역(가장자리 margin=max(scales)//2 제외)에 대해 F(x,y)를 계산.

    returns: (F_interior, margin) — F_interior[i, j]는 원본 이미지의 (i+margin, j+margin) 픽셀에 대응.
    """
    h, w = img.shape[:2]
    margin = max(scales) // 2
    integral = cv2.integral(img.astype(np.float64))

    interior_img = img[margin : h - margin, margin : w - margin].astype(np.float64)

    f_stack = []
    for scale in scales:
        half = scale // 2
        outer_sum = _box_sum(integral, half, margin, h, w)
        inner_sum = _box_sum(integral, half - 1, margin, h, w) if half > 0 else 0.0
        ring_sum = outer_sum - inner_sum
        norm = 4 * scale - 4
        f_stack.append((norm * interior_img - ring_sum) / norm)

    f_interior = np.maximum.reduce(f_stack)
    return f_interior, margin


def extract_candidates(img: np.ndarray, params: PLLCMParams) -> tuple[np.ndarray, np.ndarray, float]:
    """returns: (F_interior, candidate_coords(N,2) [row,col] 원본 이미지 좌표, Th_F)"""
    f_interior, margin = compute_f_interior(img, params.scales)

    th_f = float(f_interior.mean() + params.k_th * f_interior.std())
    rows, cols = np.nonzero(f_interior > th_f)
    candidate_coords = np.stack([rows + margin, cols + margin], axis=1)

    return f_interior, candidate_coords, th_f
