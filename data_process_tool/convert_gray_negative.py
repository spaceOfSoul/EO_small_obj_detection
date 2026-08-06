"""
dataset_EO/dataset_classified/sky 폴더의 jpg 이미지에 대해
- gray_image : grayscale 이미지 파일(.png)
- neg_image  : grayscale에 negative 필터(반전)를 적용한 이미지 파일(.png)
- neg_array  : negative 필터가 적용된 numpy 배열(.npy)
세 폴더에 각각 저장한다.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset_EO"


def convert_gray_negative(
    input_dir: Path, gray_image_out_dir: Path, neg_image_out_dir: Path, neg_array_out_dir: Path
) -> None:
    gray_image_out_dir.mkdir(parents=True, exist_ok=True)
    neg_image_out_dir.mkdir(parents=True, exist_ok=True)
    neg_array_out_dir.mkdir(parents=True, exist_ok=True)

    jpg_paths = sorted(input_dir.glob("*.jpg"))
    if not jpg_paths:
        print(f"'{input_dir}'에서 jpg 파일을 찾지 못했습니다.")
        return

    for path in jpg_paths:
        gray_img = Image.open(path).convert("L")
        neg_img = ImageOps.invert(gray_img)

        gray_img.save((gray_image_out_dir / path.stem).with_suffix(".png"))
        neg_img.save((neg_image_out_dir / path.stem).with_suffix(".png"))
        np.save(neg_array_out_dir / f"{path.stem}.npy", np.array(neg_img))

    print(f"{len(jpg_paths)}개 이미지 변환 완료")
    print(f"  grayscale 이미지 저장 위치 : {gray_image_out_dir}")
    print(f"  negative 이미지 저장 위치  : {neg_image_out_dir}")
    print(f"  negative 배열 저장 위치    : {neg_array_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jpg 이미지를 grayscale/negative로 변환해 이미지·numpy 배열로 저장")
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "dataset_classified" / "sky")
    parser.add_argument("--gray-image-out-dir", type=Path, default=DATA_DIR / "dataset_classified" / "gray_image")
    parser.add_argument("--neg-image-out-dir", type=Path, default=DATA_DIR / "dataset_classified" / "neg_image")
    parser.add_argument("--neg-array-out-dir", type=Path, default=DATA_DIR / "dataset_classified" / "neg_array")
    args = parser.parse_args()

    convert_gray_negative(args.input_dir, args.gray_image_out_dir, args.neg_image_out_dir, args.neg_array_out_dir)
