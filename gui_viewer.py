"""
detect_testbed.py 결과(experiment_result/Experiment_*/*.txt)를 넘겨가며 보는 tkinter GUI 뷰어.

박스 그리기는 visualize_results.py의 render_detections()를 그대로 재사용한다(파일로 저장하는
기능은 visualize_results.py에 남겨두고, 이 프로그램은 화면에 즉시 렌더링해서 보여주기만 한다).

조작: <-/-> 방향키 또는 화면의 이전/다음 버튼.
"""
import argparse
import tkinter as tk
from pathlib import Path

from PIL import ImageTk

from visualize_results import EXPERIMENT_ROOT_DIR, IMAGE_DIR, find_latest_experiment_dir, parse_yolo_txt, render_detections

MAX_DISPLAY_SIZE = (1240, 820)


class Viewer(tk.Tk):
    def __init__(self, experiment_dir: Path, image_dir: Path):
        super().__init__()
        self.title(f"PLLCM 결과 뷰어 - {experiment_dir.name}")

        self.txt_paths = sorted(experiment_dir.glob("*.txt"))
        if not self.txt_paths:
            raise FileNotFoundError(f"'{experiment_dir}'에서 결과 txt 파일을 찾지 못했습니다.")
        self.image_dir = image_dir
        self.index = 0
        self._photo = None  # PhotoImage 참조 유지(안 하면 GC로 이미지가 사라짐)

        self.image_label = tk.Label(self, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar()
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X)

        nav = tk.Frame(self)
        nav.pack(fill=tk.X)
        tk.Button(nav, text="<< 이전 (←)", command=self.show_prev).pack(side=tk.LEFT)
        tk.Button(nav, text="다음 (→) >>", command=self.show_next).pack(side=tk.RIGHT)

        self.bind("<Left>", lambda _event: self.show_prev())
        self.bind("<Right>", lambda _event: self.show_next())

        self.geometry("1280x900")
        self.show_current()

    def show_current(self) -> None:
        txt_path = self.txt_paths[self.index]
        image_path = self.image_dir / f"{txt_path.stem}.png"
        detections = parse_yolo_txt(txt_path)

        if image_path.exists():
            img = render_detections(image_path, detections)
            img.thumbnail(MAX_DISPLAY_SIZE)
            self._photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self._photo, text="")
        else:
            self._photo = None
            self.image_label.configure(image="", text=f"이미지 없음: {image_path}")

        self.status_var.set(f"[{self.index + 1}/{len(self.txt_paths)}] {txt_path.stem} - {len(detections)}개 detect")

    def show_prev(self) -> None:
        self.index = (self.index - 1) % len(self.txt_paths)
        self.show_current()

    def show_next(self) -> None:
        self.index = (self.index + 1) % len(self.txt_paths)
        self.show_current()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="detect 결과를 넘겨가며 보는 tkinter GUI 뷰어")
    parser.add_argument(
        "experiment_dir",
        type=Path,
        nargs="?",
        default=None,
        help="결과 txt가 있는 experiment_result/Experiment_* 폴더 (생략 시 가장 최근 실험 사용)",
    )
    parser.add_argument("--experiment-dir", dest="experiment_dir_opt", type=Path, default=None, help="위와 동일 (플래그 형태)")
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR, help="박스를 그릴 원본 이미지 폴더")
    args = parser.parse_args()

    target_dir = args.experiment_dir_opt or args.experiment_dir or find_latest_experiment_dir(EXPERIMENT_ROOT_DIR)
    print(f"실험 폴더: {target_dir}")

    Viewer(target_dir, args.image_dir).mainloop()
