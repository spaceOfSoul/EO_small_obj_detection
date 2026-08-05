"""PLLCM 알고리즘 튜닝 파라미터. 값 출처: PLLCM_method_summary.pdf "7. 파라미터 요약"."""
from dataclasses import dataclass


@dataclass
class PLLCMParams:
    scales: tuple[int, ...] = (3, 5, 7, 9, 11)  # 1단계: 멀티스케일 필터 크기
    k_th: float = 5.2  # 1단계: 후보 추출 임계값 계수 (Th_F = mean + k_th*std)
    window: int = 11  # 2단계: RW 세그멘테이션 로컬 윈도우 크기 (홀수)
    beta: float = 15.0  # 2단계: RW 엣지 가중치 상수
    size_min: int = 1  # 3단계: PLLCM_SC 크기 제약 하한 (H*W > size_min)
    size_max: int = 9  # 3단계: PLLCM_SC 크기 제약 상한 (H, W <= size_max)
    lambda_th: float = 0.5  # 4단계: 최종 임계값 가중치 (Th_E = lambda*E_max + (1-lambda)*E_mean), 권장 0.4~0.6
