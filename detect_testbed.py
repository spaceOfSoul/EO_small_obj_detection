"""
알고리즘 시험/튜닝용 테스트베드.

test/dataset_classified/neg_array의 .npy(negative filter 적용된 grayscale numpy 배열)를
로드 -> PLLCM detect 알고리즘 실행 -> YOLO 포맷(txt) 결과 저장까지의 파이프라인.

- detect 알고리즘 본체는 pllcm 패키지(PLLCM_method_summary.pdf 기반 구현)에 있다.
  더미 파이프라인 검증용으로 돌아가려면 `from dummy_detector import detect`로 되돌리면 된다.
- 출력 포맷은 YOLO와 동일: `class x_center y_center width height confidence`
  (좌표/크기는 이미지 W, H로 정규화한 0~1 값).
- 실행마다 experiment_result/Experiment_YYMMDD_HH_MM_SS/ 폴더를 새로 만들어 결과(txt),
  파라미터(params.json), 실험 세팅 요약(experiment_summary.txt)을 저장한다.
  시각화는 visualize_results.py로 확인한다.
- PLLCM은 원래 저해상도(scidb_dataset 기준 256x256) IR 영상, 2x1~9x9px 소형 표적 기준으로
  튜닝된 알고리즘이다. 지금 쓰는 neg_array는 2800x2100인데, 실측해보면 비행기 같은 실제 물체의
  픽셀상 크기가 이미 수십x수십px로 9x9 가정을 훌쩍 넘는다 - RW window/size 제약을 건드리는 대신
  --scale로 입력 이미지 자체를 다운스케일해서 표적을 다시 설계 범위(약 9px 이하)로 되돌리는
  방식을 우선 시도한다 (window/size 제약을 올리면 RW 선형시스템 비용이 커져 비현실적으로 느려짐).
"""
import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pllcm import PLLCMParams, detect

SCRIPT_DIR = Path(__file__).resolve().parent

# 데이터셋/실험결과 경로 - 커맨드라인 인자(--dataset-dir, --output-dir)로 덮어쓸 수 있다.
DATASET_DIR = SCRIPT_DIR / "test" / "dataset_classified" / "neg_array"
EXPERIMENT_ROOT_DIR = SCRIPT_DIR / "experiment_result"


def load_neg_array(path: Path, scale: float = 1.0) -> np.ndarray:
    """scale < 1.0이면 다운스케일(cv2.INTER_AREA)해서 반환.

    PLLCM의 표적 크기 가정(2x1~9x9px)에 맞춰, 원본 해상도에서 수십px에 달하는 실제 물체를
    다시 그 범위로 줄여넣기 위한 전처리. YOLO 출력은 정규화(0~1) 좌표라 스케일을 바꿔도
    원본 이미지 위에 그대로 겹쳐 그릴 수 있다 (visualize_results.py / gui_viewer.py 수정 불필요).
    """
    array = np.load(path)
    if scale != 1.0:
        array = cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return array


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


def write_experiment_summary(
    output_dir: Path,
    dataset_dir: Path,
    params: PLLCMParams,
    scale: float,
    start_time: datetime,
    num_images: int,
    total_elapsed: float,
) -> None:
    lines = [
        "PLLCM 실험 요약",
        "=" * 40,
        f"실행 시각: {start_time:%Y-%m-%d %H:%M:%S}",
        f"데이터셋 경로: {dataset_dir}",
        f"처리 이미지 수: {num_images}",
        f"총 소요 시간: {total_elapsed:.1f}초 (평균 {total_elapsed / num_images:.2f}초/장)",
        "",
        "[파라미터]",
        f"scale: {scale}",
    ]
    lines += [f"{name}: {value}" for name, value in asdict(params).items()]

    (output_dir / "experiment_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run(dataset_dir: Path, output_dir: Path, params: PLLCMParams, scale: float = 1.0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    (output_dir / "params.json").write_text(
        json.dumps({"dataset_dir": str(dataset_dir), "scale": scale, **asdict(params)}, indent=2, default=list),
        encoding="utf-8",
    )

    npy_paths = sorted(dataset_dir.glob("*.npy"))
    if not npy_paths:
        print(f"'{dataset_dir}'에서 npy 파일을 찾지 못했습니다.")
        return

    total_start = time.perf_counter()
    for path in npy_paths:
        array = load_neg_array(path, scale)
        img_h, img_w = array.shape[:2]

        start = time.perf_counter()
        detections = detect(array, params)
        elapsed = time.perf_counter() - start

        out_path = (output_dir / path.stem).with_suffix(".txt")
        save_yolo_txt(detections, img_h, img_w, out_path)
        print(f"{path.name}: {len(detections)}개 detect ({elapsed:.2f}s) -> {out_path.name}")

    total_elapsed = time.perf_counter() - total_start
    write_experiment_summary(output_dir, dataset_dir, params, scale, start_time, len(npy_paths), total_elapsed)

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
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="이미지 다운스케일 배율(1.0=원본). PLLCM은 저해상도(scidb_dataset 기준 256x256) 기준 "
        "2x1~9x9px 소형 표적 가정으로 튜닝돼 있어, 고해상도 원본에서 표적이 그보다 훨씬 크게 찍히면 "
        "1보다 작은 값으로 낮춰 표적을 다시 그 크기 범위로 되돌리는 용도. 예: 0.1",
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

    run(args.dataset_dir, args.output_dir, cli_params, args.scale)
