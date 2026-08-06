"""
grayscale / negative / autocontrast(cutoff) / scale(해상도) / blur 값을
슬라이더로 조절하면서 원본과 결과를 나란히 미리보기하는 tkinter GUI 툴.

실행:
    python data_process_tool/image_tune_tool.py [이미지 경로]
이미지 경로를 생략하면 dataset_EO/dataset_classified/sky 또는 dataset_EO/original에서
첫 번째 jpg를 기본으로 연다.
"""
import json
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageTk

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset_EO"
DEFAULT_SEARCH_DIRS = [
    DATA_DIR / "dataset_classified" / "sky",
    DATA_DIR / "original",
]
PREVIEW_MAX = 480


def find_default_image() -> Path | None:
    for d in DEFAULT_SEARCH_DIRS:
        if d.is_dir():
            files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
            if files:
                return files[0]
    return None


class ImageTuneTool:
    def __init__(self, root: tk.Tk, image_path: Path | None):
        self.root = root
        self.root.title("Image Tune Tool")

        self.image_path: Path | None = image_path
        self.original: Image.Image | None = Image.open(image_path).convert("RGB") if image_path else None
        self.processed: Image.Image | None = None
        self._photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self.value_labels: dict[str, ttk.Label] = {}
        self.batch_input_dir: Path | None = None

        self._build_ui()
        if self.original is not None:
            self._refresh()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="이미지 열기...", command=self._open_image).pack(side="left")
        ttk.Button(top, text="결과 저장...", command=self._save_image).pack(side="left", padx=6)
        ttk.Button(top, text="초기화", command=self._reset).pack(side="left")
        self.path_label = ttk.Label(top, text=str(self.image_path) if self.image_path else "이미지를 선택하세요")
        self.path_label.pack(side="left", padx=10)

        preview = ttk.Frame(self.root, padding=8)
        preview.pack(fill="both", expand=True)

        before_frame = ttk.LabelFrame(preview, text="원본")
        before_frame.pack(side="left", padx=8, pady=4)
        self.before_label = ttk.Label(before_frame)
        self.before_label.pack()

        after_frame = ttk.LabelFrame(preview, text="결과 (튜닝 적용)")
        after_frame.pack(side="left", padx=8, pady=4)
        self.after_label = ttk.Label(after_frame)
        self.after_label.pack()

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        controls.columnconfigure(2, weight=1)

        self.gray_var = tk.BooleanVar(value=True)
        self.neg_var = tk.BooleanVar(value=True)
        self.contrast_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(controls, text="Grayscale", variable=self.gray_var, command=self._refresh).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Checkbutton(controls, text="Negative", variable=self.neg_var, command=self._refresh).grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Checkbutton(controls, text="Autocontrast", variable=self.contrast_var, command=self._refresh).grid(
            row=2, column=0, sticky="w", pady=2
        )

        self.cutoff_var = tk.DoubleVar(value=0)
        self.scale_var = tk.DoubleVar(value=100)
        self.blur_var = tk.DoubleVar(value=0)

        self._add_slider(controls, "cutoff", "Cutoff (%)", self.cutoff_var, 0, 49, row=0)
        self._add_slider(controls, "scale", "Scale / 해상도 (%)", self.scale_var, 10, 100, row=1)
        self._add_slider(controls, "blur", "Blur radius", self.blur_var, 0, 10, row=2)

        batch = ttk.LabelFrame(self.root, text="폴더 일괄 처리", padding=8)
        batch.pack(fill="x", padx=8, pady=(0, 8))
        batch.columnconfigure(1, weight=1)

        ttk.Button(batch, text="대상 폴더 선택...", command=self._select_batch_folder).grid(row=0, column=0, sticky="w")
        self.batch_dir_label = ttk.Label(batch, text="폴더를 선택하세요")
        self.batch_dir_label.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(batch, text="출력 폴더 이름 접두어:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.folder_name_var = tk.StringVar(value="data")
        ttk.Entry(batch, textvariable=self.folder_name_var, width=20).grid(row=1, column=1, sticky="w", padx=8, pady=(6, 0))

        btn_row = ttk.Frame(batch)
        btn_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(btn_row, text="현재 파라미터 저장(JSON)...", command=self._save_params).pack(side="left")
        ttk.Button(btn_row, text="선택 폴더 일괄 처리 실행", command=self._run_batch).pack(side="left", padx=6)

        self.batch_status_label = ttk.Label(batch, text="")
        self.batch_status_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _add_slider(self, parent, key: str, label: str, var: tk.DoubleVar, frm: float, to: float, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=1, sticky="w", padx=(24, 6))
        scale = ttk.Scale(parent, from_=frm, to=to, variable=var, orient="horizontal", command=lambda _=None: self._refresh())
        scale.grid(row=row, column=2, sticky="we", padx=4)
        value_label = ttk.Label(parent, width=6, anchor="w")
        value_label.grid(row=row, column=3, sticky="w")
        self.value_labels[key] = value_label

    # ---------- Image processing ----------
    def _process(self) -> Image.Image:
        img = self.original
        scale = self.scale_var.get() / 100.0
        if scale < 1.0:
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.LANCZOS)

        if self.gray_var.get():
            img = img.convert("L")

        if self.neg_var.get():
            img = ImageOps.invert(img)

        if self.contrast_var.get():
            img = ImageOps.autocontrast(img, cutoff=self.cutoff_var.get())

        blur = self.blur_var.get()
        if blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))

        return img

    # ---------- UI refresh ----------
    def _refresh(self, *_args) -> None:
        if self.original is None:
            return
        self.processed = self._process()

        self._set_preview(self.before_label, "before", self.original)
        self._set_preview(self.after_label, "after", self.processed)

        self.value_labels["cutoff"].configure(text=f"{self.cutoff_var.get():.0f}")
        self.value_labels["scale"].configure(text=f"{self.scale_var.get():.0f}")
        self.value_labels["blur"].configure(text=f"{self.blur_var.get():.1f}")

    def _set_preview(self, label_widget: ttk.Label, key: str, img: Image.Image) -> None:
        disp = img.copy()
        disp.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
        if disp.mode not in ("RGB", "L"):
            disp = disp.convert("RGB")
        photo = ImageTk.PhotoImage(disp)
        label_widget.configure(image=photo)
        self._photo_refs[key] = photo  # keep reference alive

    # ---------- Actions ----------
    def _open_image(self) -> None:
        initial_dir = self.image_path.parent if self.image_path else DATA_DIR
        path_str = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if not path_str:
            return
        self.image_path = Path(path_str)
        self.original = Image.open(self.image_path).convert("RGB")
        self.path_label.configure(text=str(self.image_path))
        self._refresh()

    def _save_image(self) -> None:
        if self.processed is None:
            messagebox.showwarning("저장 불가", "먼저 이미지를 여세요.")
            return
        path_str = filedialog.asksaveasfilename(
            initialdir=str(DATA_DIR),
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("NumPy array", "*.npy")],
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() == ".npy":
            np.save(path, np.array(self.processed))
        else:
            self.processed.save(path)
        messagebox.showinfo("저장 완료", f"저장됨: {path}")

    # ---------- Params / batch processing ----------
    def _current_params(self) -> dict:
        return {
            "grayscale": self.gray_var.get(),
            "negative": self.neg_var.get(),
            "autocontrast": self.contrast_var.get(),
            "cutoff": self.cutoff_var.get(),
            "scale_percent": self.scale_var.get(),
            "blur_radius": self.blur_var.get(),
        }

    def _save_params(self) -> None:
        path_str = filedialog.asksaveasfilename(
            initialdir=str(DATA_DIR),
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path_str:
            return
        with open(path_str, "w", encoding="utf-8") as f:
            json.dump(self._current_params(), f, ensure_ascii=False, indent=2)
        messagebox.showinfo("저장 완료", f"파라미터 저장됨: {path_str}")

    def _select_batch_folder(self) -> None:
        dir_str = filedialog.askdirectory(initialdir=str(DATA_DIR))
        if not dir_str:
            return
        self.batch_input_dir = Path(dir_str)
        self.batch_dir_label.configure(text=str(self.batch_input_dir))

    def _run_batch(self) -> None:
        if self.batch_input_dir is None:
            messagebox.showwarning("폴더 없음", "먼저 대상 폴더를 선택하세요.")
            return

        image_paths = sorted(
            p
            for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in self.batch_input_dir.glob(ext)
        )
        if not image_paths:
            messagebox.showwarning("이미지 없음", f"'{self.batch_input_dir}'에서 이미지를 찾지 못했습니다.")
            return

        prefix = self.folder_name_var.get().strip() or "data"
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        out_dir = self.batch_input_dir.parent / f"{prefix}_{timestamp}"
        image_out_dir = out_dir / "image"
        array_out_dir = out_dir / "np_array"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        array_out_dir.mkdir(parents=True, exist_ok=True)

        params = self._current_params()
        params["source_dir"] = str(self.batch_input_dir)
        params["created_at"] = datetime.now().isoformat(timespec="seconds")
        params["num_images"] = len(image_paths)
        with open(out_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        saved_original = self.original
        for i, path in enumerate(image_paths, start=1):
            self.original = Image.open(path).convert("RGB")
            result = self._process()
            result.save((image_out_dir / path.stem).with_suffix(".png"))
            np.save(array_out_dir / f"{path.stem}.npy", np.array(result))

            self.batch_status_label.configure(text=f"처리 중... {i}/{len(image_paths)}")
            self.root.update_idletasks()
        self.original = saved_original
        self._refresh()

        self.batch_status_label.configure(text=f"완료: {len(image_paths)}장 -> {out_dir}")
        messagebox.showinfo("일괄 처리 완료", f"{len(image_paths)}장 처리 완료\n저장 위치: {out_dir}")

    def _reset(self) -> None:
        self.gray_var.set(True)
        self.neg_var.set(True)
        self.contrast_var.set(False)
        self.cutoff_var.set(0)
        self.scale_var.set(100)
        self.blur_var.set(0)
        self._refresh()


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_default_image()

    root = tk.Tk()
    if image_path is None:
        messagebox.showwarning("이미지 없음", "기본 이미지를 찾지 못했습니다. '이미지 열기'로 직접 선택하세요.")
    ImageTuneTool(root, image_path)
    root.mainloop()


if __name__ == "__main__":
    main()
