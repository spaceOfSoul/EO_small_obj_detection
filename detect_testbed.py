"""
알고리즘 시험/튜닝용 테스트베드.

test/dataset_classified/neg_array의 .npy(negative filter 적용된 grayscale numpy 배열)를
로드 -> PLLCM detect 알고리즘 실행 -> YOLO 포맷(txt) 결과 저장까지의 파이프라인.

- detect 알고리즘 본체는 pllcm 패키지(PLLCM_method_summary.pdf 기반 구현)에 있다.
  더미 파이프라인 검증용으로 돌아가려면 `from dummy_detector import detect`로 되돌리면 된다.
- 출력 포맷은 YOLO와 동일: `class x_center y_center width height confidence`
  (좌표/크기는 이미지 W, H로 정규화한 0~1 값).
- 실행마다 experiment_result/Experiment_YYMMDD_HH_MM_SS/ 폴더를 새로 만들어 결과(txt)와
  사용한 파라미터(params.json)를 저장한다. 시각화는 visualize_results.py로 확인한다.
"""
import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from pllcm import PLLCMParams, detect

SCRIPT_DIR = Path(__file__).resolve().parent

# 데이터셋/실험결과 경로 - 커맨드라인 인자(--dataset-dir, --output-dir)로 덮어쓸 수 있다.
DATASET_DIR = SCRIPT_DIR / "test" / "dataset_classified" / "neg_array"
EXPERIMENT_ROOT_DIR = SCRIPT_DIR / "experiment_result"


def load_neg_array(path: Path) -> np.ndarray:
    return np.load(path)


def save_yolo_txt(
    detections: list[tuple[int, float, float, float, float, float]],
    img_h: int,
    img_w: int,
    out_path: Path,
) -> None:
    """detections: (cls_id, x1, y1, x2, y2, conf) 픽셀 좌표 리스트 -> YOLO 정규화 txt로 저장"""
    lines = []
    for cls_id, x1, y1, x2, y2, conf in detections:
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.6f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(dataset_dir: Path, output_dir: Path, params: PLLCMParams) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "params.json").write_text(
        json.dumps({"dataset_dir": str(dataset_dir), **asdict(params)}, indent=2, default=list),
        encoding="utf-8",
    )

    npy_paths = sorted(dataset_dir.glob("*.npy"))
    if not npy_paths:
        print(f"'{dataset_dir}'에서 npy 파일을 찾지 못했습니다.")
        return

    total_start = time.perf_counter()
    for path in npy_paths:
        array = load_neg_array(path)
        img_h, img_w = array.shape[:2]

        start = time.perf_counter()
        detections = detect(array, params)
        elapsed = time.perf_counter() - start

        out_path = (output_dir / path.stem).with_suffix(".txt")
        save_yolo_txt(detections, img_h, img_w, out_path)
        print(f"{path.name}: {len(detections)}개 detect ({elapsed:.2f}s) -> {out_path.name}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n총 {len(npy_paths)}개 처리 완료 ({total_elapsed:.1f}s, 평균 {total_elapsed / len(npy_paths):.2f}s/장)")
    print(f"결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    default_params = PLLCMParams()
    default_output_dir = EXPERIMENT_ROOT_DIR / f"Experiment_{datetime.now():%y%m%d_%H_%M_%S}"

    parser = argparse.ArgumentParser(description="neg_array 데이터셋에 PLLCM detect 알고리즘을 실행하는 테스트베드")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="neg_array(.npy) 폴더 경로")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="YOLO 포맷 결과(txt) 저장 경로 (기본: experiment_result/Experiment_YYMMDD_HH_MM_SS)",
    )
    parser.add_argument("--k-th", type=float, default=default_params.k_th, help="1단계 후보 추출 임계값 계수")
    parser.add_argument("--beta", type=float, default=default_params.beta, help="2단계 RW 엣지 가중치 상수")
    parser.add_argument("--window", type=int, default=default_params.window, help="2단계 RW 로컬 윈도우 크기(홀수)")
    parser.add_argument("--size-min", type=int, default=default_params.size_min, help="3단계 크기 제약 하한(H*W >)")
    parser.add_argument("--size-max", type=int, default=default_params.size_max, help="3단계 크기 제약 상한(H,W <=)")
    parser.add_argument("--lambda-th", type=float, default=default_params.lambda_th, help="4단계 최종 임계값 가중치")
    args = parser.parse_args()

    cli_params = PLLCMParams(
        k_th=args.k_th,
        window=args.window,
        beta=args.beta,
        size_min=args.size_min,
        size_max=args.size_max,
        lambda_th=args.lambda_th,
    )

    run(args.dataset_dir, args.output_dir, cli_params)
