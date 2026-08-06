"""
detect_testbed.py가 experiment_result/Experiment_YYMMDD_HH_MM_SS/에 남긴 YOLO 포맷 txt 결과를
원본 이미지 위에 박스로 그려 육안으로 확인할 수 있게 저장하는 뷰어.

기본적으로 experiment_result에서 가장 최근 실험 폴더를 찾아 사용하며, 결과 이미지는
<experiment_dir>/vis/ 에 저장한다.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent

EXPERIMENT_ROOT_DIR = SCRIPT_DIR / "experiment_result"

BOX_COLOR = (255, 0, 0)
BOX_WIDTH = 3
BOX_PAD = 8  # 검출 박스가 몇 픽셀 수준으로 작아 육안으로 보기 쉽게 여유를 둔다
FONT_SIZE = 40

try:
    _FONT = ImageFont.truetype("arial.ttf", FONT_SIZE)
except OSError:
    _FONT = ImageFont.load_default(size=FONT_SIZE)


def find_latest_experiment_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"'{root}'에 실험 결과 폴더가 없습니다.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _dataset_dir_from_params(experiment_dir: Path) -> Path | None:
    """실험 폴더의 params.json에 detect_testbed.py가 남긴 dataset_dir(.npy 폴더)을 읽는다."""
    params_path = experiment_dir / "params.json"
    if not params_path.exists():
        return None
    try:
        raw = json.loads(params_path.read_text(encoding="utf-8")).get("dataset_dir")
    except (OSError, ValueError):
        return None
    if not raw:
        return None
    dataset_dir = Path(raw)
    if not dataset_dir.is_absolute():
        dataset_dir = SCRIPT_DIR / dataset_dir
    return dataset_dir


def infer_image_dir(dataset_dir: Path) -> Path | None:
    """dataset_dir(.npy 폴더)과 같은 부모 아래에서 파일명이 대응되는 이미지 폴더를 찾는다.

    데이터셋마다 이미지 폴더명 규칙이 달라(neg_array/neg_image, np_array/image 등) 고정 경로를
    쓸 수 없어, npy 파일명과 겹치는 png가 가장 많은 형제 폴더를 찾는 방식으로 추정한다.
    """
    if not dataset_dir.exists():
        return None
    npy_stems = {p.stem for p in dataset_dir.glob("*.npy")}
    if not npy_stems:
        return None

    best_dir, best_score = None, 0
    for candidate in dataset_dir.parent.iterdir():
        if not candidate.is_dir() or candidate == dataset_dir:
            continue
        score = len(npy_stems & {p.stem for p in candidate.glob("*.png")})
        if score > best_score:
            best_dir, best_score = candidate, score
    return best_dir


def resolve_image_dir(experiment_dir: Path) -> Path | None:
    """실험 폴더의 params.json으로부터 대응 이미지 폴더를 추정한다. 실패 시 None."""
    dataset_dir = _dataset_dir_from_params(experiment_dir)
    if dataset_dir is None:
        return None
    return infer_image_dir(dataset_dir)


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
        draw.text((x2 + 2, y1), f"{conf:.2f}", fill=BOX_COLOR, font=_FONT)

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
    parser.add_argument("--experiment-dir", dest="experiment_dir_opt", type=Path, default=None, help="위와 동일 (플래그 형태)")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="박스를 그릴 원본 이미지 폴더 (생략 시 실험 폴더의 params.json에 기록된 dataset_dir로부터 자동 추정)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="시각화 결과 저장 폴더 (기본: <experiment_dir>/vis)")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir_opt or args.experiment_dir or find_latest_experiment_dir(EXPERIMENT_ROOT_DIR)
    output_dir = args.output_dir or (experiment_dir / "vis")

    image_dir = args.image_dir or resolve_image_dir(experiment_dir)
    if image_dir is None:
        parser.error(
            f"'{experiment_dir}'의 params.json에서 이미지 폴더를 추정하지 못했습니다. --image-dir로 직접 지정하세요."
        )

    print(f"실험 폴더: {experiment_dir}")
    print(f"이미지 폴더: {image_dir}")
    run(experiment_dir, image_dir, output_dir)
