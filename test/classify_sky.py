"""
test/original의 jpg 이미지를 Ultralytics YOLO26 semantic segmentation 모델(yolo26n-sem.pt)로
추론하여 "sky" 클래스 픽셀 비율이 임계값 이상이면 sky, 아니면 others로 분류해
test/dataset_classified/{sky,others}로 복사한다.

주의: yolo26n-sem.pt는 Cityscapes(19-class) 사전학습 가중치로, ADE20K 전용 사전학습
가중치는 Ultralytics 공식 배포본에 없다(ade20k.yaml은 학습용 데이터셋 설정만 제공).
다만 Cityscapes 클래스에도 "sky"가 포함되어 있어(class 10) sky/others 분류에는 그대로 사용 가능하다.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
SKY_CLASS_NAME = "sky"


def classify_sky(input_dir: Path, output_dir: Path, model_name: str, threshold: float) -> None:
    sky_dir = output_dir / "sky"
    others_dir = output_dir / "others"
    sky_dir.mkdir(parents=True, exist_ok=True)
    others_dir.mkdir(parents=True, exist_ok=True)

    jpg_paths = sorted(input_dir.glob("*.jpg"))
    if not jpg_paths:
        print(f"'{input_dir}'에서 jpg 파일을 찾지 못했습니다.")
        return

    model = YOLO(model_name)

    sky_count = 0
    for path in jpg_paths:
        result = model(path, verbose=False)[0]
        sky_ratio = next(
            (entry["pixel_ratio"] for entry in result.summary() if entry["name"] == SKY_CLASS_NAME),
            0.0,
        )

        dest_dir = sky_dir if sky_ratio >= threshold else others_dir
        shutil.copy2(path, dest_dir / path.name)
        sky_count += dest_dir is sky_dir
        print(f"{path.name}: sky_ratio={sky_ratio:.2f} -> {dest_dir.name}")

    print(f"\n총 {len(jpg_paths)}장 중 sky={sky_count}장, others={len(jpg_paths) - sky_count}장")
    print(f"결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="semantic segmentation의 sky 픽셀 비율로 하늘 사진을 분류")
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR / "original")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "dataset_classified")
    parser.add_argument("--model", type=str, default="yolo26n-sem.pt")
    parser.add_argument("--threshold", type=float, default=0.3, help="sky로 분류할 최소 sky 픽셀 비율(0~1)")
    args = parser.parse_args()

    classify_sky(args.input_dir, args.output_dir, args.model, args.threshold)
