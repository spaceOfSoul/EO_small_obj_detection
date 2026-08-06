"""
dataset_EO/original 폴더의 jpg 이미지를 grayscale로 변환하여
- dataset_EO/gray_image : grayscale 이미지 파일(.png)
- dataset_EO/gray_array : grayscale numpy 배열(.npy)
두 폴더에 각각 저장한다.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset_EO"


def convert_to_grayscale(input_dir: Path, image_out_dir: Path, array_out_dir: Path) -> None:
    image_out_dir.mkdir(parents=True, exist_ok=True)
    array_out_dir.mkdir(parents=True, exist_ok=True)

    jpg_paths = sorted(input_dir.glob("*.jpg"))
    if not jpg_paths:
        print(f"'{input_dir}'에서 jpg 파일을 찾지 못했습니다.")
        return

    for path in jpg_paths:
        gray_img = Image.open(path).convert("L")

        gray_img.save((image_out_dir / path.stem).with_suffix(".png"))
        np.save(array_out_dir / f"{path.stem}.npy", np.array(gray_img))

    print(f"{len(jpg_paths)}개 이미지 변환 완료")
    print(f"  이미지 저장 위치 : {image_out_dir}")
    print(f"  배열 저장 위치   : {array_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jpg 이미지를 grayscale로 변환하여 이미지 파일과 numpy 배열로 각각 저장")
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "original")
    parser.add_argument("--image-out-dir", type=Path, default=DATA_DIR / "gray_image")
    parser.add_argument("--array-out-dir", type=Path, default=DATA_DIR / "gray_array")
    args = parser.parse_args()

    convert_to_grayscale(args.input_dir, args.image_out_dir, args.array_out_dir)
