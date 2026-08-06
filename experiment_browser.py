"""
experiment_result 안의 실험 폴더들을 목록으로 보여주고, 선택한 실험을 gui_viewer.py로
열어주는 tkinter 런처 GUI.

기능:
1. experiment_result 내 실험 폴더 목록 표시
2. 목록 항목 더블클릭(또는 "뷰어 열기" 버튼)시 해당 폴더로 gui_viewer.py 실행
3. 우측에 선택한 실험의 experiment_summary.txt 내용 표시
4. "새로고침" 버튼으로 experiment_result를 다시 스캔 (신규 실험 반영)
"""
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from visualize_results import EXPERIMENT_ROOT_DIR

SCRIPT_DIR = Path(__file__).resolve().parent
GUI_VIEWER_PATH = SCRIPT_DIR / "gui_viewer.py"

LIST_FONT = ("", 13)
SUMMARY_FONT = ("Consolas", 12)


class ExperimentBrowser(tk.Tk):
    def __init__(self, root_dir: Path):
        super().__init__()
        self.title("PLLCM 실험 결과 브라우저")
        self.geometry("1000x640")

        self.root_dir = root_dir
        self.experiment_dirs: list[Path] = []

        self._build_widgets()
        self.refresh()

    def _build_widgets(self) -> None:
        toolbar = tk.Frame(self)
        toolbar.pack(fill=tk.X, padx=8, pady=6)
        tk.Button(toolbar, text="새로고침", command=self.refresh).pack(side=tk.LEFT)
        tk.Button(toolbar, text="뷰어 열기 (더블클릭)", command=self.open_selected).pack(side=tk.LEFT, padx=(6, 0))

        body = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=6)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        list_frame = tk.Frame(body)
        self.listbox = tk.Listbox(list_frame, font=LIST_FONT, activestyle="dotbox")
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.listbox.bind("<Double-Button-1>", lambda _e: self.open_selected())
        self.listbox.bind("<Return>", lambda _e: self.open_selected())
        body.add(list_frame, minsize=280)

        summary_frame = tk.Frame(body)
        tk.Label(summary_frame, text="experiment_summary.txt", anchor="w", font=("", 12, "bold")).pack(fill=tk.X)
        self.summary_text = tk.Text(summary_frame, font=SUMMARY_FONT, wrap="word", state="disabled")
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        body.add(summary_frame, minsize=400)

        self.status_var = tk.StringVar()
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=8, pady=(0, 6))

    def refresh(self) -> None:
        selected_name = self._selected_dir_name()

        if self.root_dir.exists():
            self.experiment_dirs = sorted(
                (p for p in self.root_dir.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        else:
            self.experiment_dirs = []

        self.listbox.delete(0, tk.END)
        for exp_dir in self.experiment_dirs:
            count = len(list(exp_dir.glob("*.txt")))
            self.listbox.insert(tk.END, f"{exp_dir.name}  ({count}장)")

        self.status_var.set(f"실험 {len(self.experiment_dirs)}개 - {self.root_dir}")

        if selected_name:
            for i, exp_dir in enumerate(self.experiment_dirs):
                if exp_dir.name == selected_name:
                    self.listbox.selection_set(i)
                    self.listbox.see(i)
                    self._show_summary(exp_dir)
                    return
        self._show_summary(None)

    def _selected_dir_name(self) -> str | None:
        idx = self._selected_index()
        if idx is None:
            return None
        return self.experiment_dirs[idx].name

    def _selected_index(self) -> int | None:
        sel = self.listbox.curselection()
        if not sel:
            return None
        return sel[0]

    def _on_select(self, _event: object) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        self._show_summary(self.experiment_dirs[idx])

    def _show_summary(self, exp_dir: Path | None) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        if exp_dir is not None:
            summary_path = exp_dir / "experiment_summary.txt"
            if summary_path.exists():
                self.summary_text.insert(tk.END, summary_path.read_text(encoding="utf-8"))
            else:
                self.summary_text.insert(tk.END, f"'{summary_path.name}' 파일이 없습니다.")
        self.summary_text.configure(state="disabled")

    def open_selected(self) -> None:
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("안내", "먼저 목록에서 실험을 선택하세요.")
            return
        exp_dir = self.experiment_dirs[idx]
        try:
            subprocess.Popen([sys.executable, str(GUI_VIEWER_PATH), str(exp_dir)])
        except OSError as exc:
            messagebox.showerror("뷰어 실행 실패", str(exc))


if __name__ == "__main__":
    ExperimentBrowser(EXPERIMENT_ROOT_DIR).mainloop()
