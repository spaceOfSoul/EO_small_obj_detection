"""
detect_testbed.py가 experiment_result/Experiment_YYMMDD_HH_MM_SS/에 남긴 YOLO 포맷 txt 결과를
원본 이미지 위에 박스로 그려 육안으로 확인할 수 있게 저장하는 뷰어.

기본적으로 experiment_result에서 가장 최근 실험 폴더를 찾아 사용하며, 결과 이미지는
<experiment_dir>/vis/ 에 저장한다.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent

EXPERIMENT_ROOT_DIR = SCRIPT_DIR / "experiment_result"
IMAGE_DIR = SCRIPT_DIR / "test" / "dataset_classified" / "neg_image"

BOX_COLOR = (255, 0, 0)
BOX_WIDTH = 3
BOX_PAD = 8  # 검출 박스가 몇 픽셀 수준으로 작아 육안으로 보기 쉽게 여유를 둔다


def find_latest_experiment_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"'{root}'에 실험 결과 폴더가 없습니다.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_yolo_txt(txt_path: Path) -> list[tuple[int, float, float, float, float, float]]:
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    detections = []
    for line in text.splitlines():
        cls_id_str, xc_str, yc_str, w_str, h_str, *rest = line.split()
        conf = float(rest[0]) if rest else 1.0
        detections.append((int(cls_id_str), float(xc_str), float(yc_str), float(w_str), float(h_str), conf))
    return detections


def render_detections(image_path: Path, detections: list) -> Image.Image:
    """박스를 그린 PIL 이미지를 반환 (저장은 하지 않음) - gui_viewer.py에서도 재사용."""
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    for _cls_id, xc, yc, w, h, conf in detections:
        x1 = (xc - w / 2) * img_w - BOX_PAD
        y1 = (yc - h / 2) * img_h - BOX_PAD
        x2 = (xc + w / 2) * img_w + BOX_PAD
        y2 = (yc + h / 2) * img_h + BOX_PAD
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
        draw.text((x2 + 2, y1), f"{conf:.2f}", fill=BOX_COLOR)

    return img


def draw_detections(image_path: Path, detections: list, out_path: Path) -> None:
    img = render_detections(image_path, detections)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def run(experiment_dir: Path, image_dir: Path, output_dir: Path) -> None:
    txt_paths = sorted(experiment_dir.glob("*.txt"))
    if not txt_paths:
        print(f"'{experiment_dir}'에서 결과 txt 파일을 찾지 못했습니다.")
        return

    total_boxes = 0
    skipped = 0
    for txt_path in txt_paths:
        image_path = image_dir / f"{txt_path.stem}.png"
        if not image_path.exists():
            print(f"{txt_path.stem}: 대응 이미지 '{image_path}'를 찾지 못해 건너뜀")
            skipped += 1
            continue

        detections = parse_yolo_txt(txt_path)
        total_boxes += len(detections)

        out_path = output_dir / f"{txt_path.stem}.png"
        draw_detections(image_path, detections, out_path)
        print(f"{txt_path.stem}: {len(detections)}개 박스 -> {out_path.name}")

    print(f"\n총 {len(txt_paths)}장 중 {len(txt_paths) - skipped}장 시각화 완료 (박스 {total_boxes}개, 누락 {skipped}장)")
    print(f"결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="detect_testbed.py의 YOLO 포맷 결과를 이미지 위에 그려 확인하는 뷰어")
    parser.add_argument(
        "experiment_dir",
        type=Path,
        nargs="?",
        default=None,
        help="결과 txt가 있는 experiment_result/Experiment_* 폴더 (생략 시 가장 최근 실험 사용)",
    )
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR, help="박스를 그릴 원본 이미지 폴더")
    parser.add_argument("--output-dir", type=Path, default=None, help="시각화 결과 저장 폴더 (기본: <experiment_dir>/vis)")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir or find_latest_experiment_dir(EXPERIMENT_ROOT_DIR)
    output_dir = args.output_dir or (experiment_dir / "vis")

    print(f"실험 폴더: {experiment_dir}")
    run(experiment_dir, args.image_dir, output_dir)
