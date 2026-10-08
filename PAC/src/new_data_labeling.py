"""--label-new-data: register NEW symbol / marking images (separate from --label-scenes).

Put new images in data/new_symbols/incoming/ (png / jpg / bmp). For each image the person enters the
transcription, source_group (all crops / rotations / brightness variants / copies of one original marking
MUST share one source_group), acquisition_source, capture_session, split (empty = automatic group-level
split) and readable / uncertain flags. No model output is shown and nothing is pre-filled.
Saved after every change to data/new_symbol_dataset_manifest.csv; reopening resumes.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import DATA_DIR, SUPPORTED_EXTENSIONS, AppConfig
from .dataset import read_csv_rows, rel, write_csv_rows
from .symbol_expansion import MANIFEST_COLUMNS


def incoming_images(cfg: AppConfig) -> List[str]:
    d = DATA_DIR / cfg.symbols.incoming_dir
    d.mkdir(parents=True, exist_ok=True)
    return sorted(rel(p) for p in d.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def run_new_data_gui(cfg: AppConfig) -> int:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                                 QMainWindow, QPushButton, QVBoxLayout, QWidget)

    manifest_path = DATA_DIR / cfg.symbols.manifest_file
    paths = incoming_images(cfg)
    if not paths:
        print(f"[NEW DATA] no images in {DATA_DIR / cfg.symbols.incoming_dir} - copy new images there first.")
        return 2
    rows: Dict[str, Dict[str, str]] = {r["image_path"]: r for r in read_csv_rows(manifest_path)}
    app = QApplication.instance() or QApplication(sys.argv)

    class Win(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Register new marking / symbol images (manual; no model output shown)")
            self.resize(1100, 800)
            self.idx = next((k for k, p in enumerate(paths) if p not in rows), 0)
            self.img, self.info = QLabel(), QLabel()
            self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.f = {k: QLineEdit() for k in ("transcription", "source_group", "acquisition_source", "capture_session", "note")}
            self.f["transcription"].setPlaceholderText("exact text, e.g. F8 →  (arrows: ← → ↑ ↓)")
            self.f["source_group"].setPlaceholderText("same value for every crop / variant of one original marking")
            self.split = QComboBox()
            self.split.addItems(["", "train", "dev", "test"])
            self.readable, self.uncertain = QCheckBox("Readable"), QCheckBox("Uncertain")
            self.readable.setChecked(True)
            form = QFormLayout()
            for k, w in self.f.items():
                form.addRow(k, w)
            form.addRow("split (empty = auto by group)", self.split)
            flags = QHBoxLayout()
            flags.addWidget(self.readable)
            flags.addWidget(self.uncertain)
            form.addRow("", QWidget())
            btns = QHBoxLayout()
            for text, fn in (("◀ Prev", lambda: self.go(-1)), ("Save", self.store),
                             ("Save + Next ▶", lambda: (self.store(), self.go(1)))):
                b = QPushButton(text)
                b.clicked.connect(fn)
                btns.addWidget(b)
            lay = QVBoxLayout()
            lay.addWidget(self.info)
            lay.addWidget(self.img, 1)
            lay.addLayout(form)
            lay.addLayout(flags)
            lay.addLayout(btns)
            c = QWidget()
            c.setLayout(lay)
            self.setCentralWidget(c)
            self.show_current()

        def show_current(self) -> None:
            p = paths[self.idx]
            from .config import PROJECT_ROOT
            pm = QPixmap(str(PROJECT_ROOT / p))
            self.img.setPixmap(pm.scaled(1000, 500, Qt.AspectRatioMode.KeepAspectRatio))
            r = rows.get(p, {})
            for k, w in self.f.items():
                w.setText(r.get(k, ""))
            self.split.setCurrentText(r.get("split", ""))
            self.readable.setChecked(r.get("readable", "True") != "False")
            self.uncertain.setChecked(r.get("uncertain", "") == "True")
            self.info.setText(f"{self.idx + 1}/{len(paths)}  {Path(p).name}   registered {len(rows)}/{len(paths)}")

        def store(self) -> None:
            p = paths[self.idx]
            rows[p] = {"image_path": p, **{k: w.text().strip() for k, w in self.f.items()},
                       "split": self.split.currentText(), "readable": str(self.readable.isChecked()),
                       "uncertain": str(self.uncertain.isChecked()),
                       "registered_at": datetime.now().isoformat(timespec="seconds")}
            write_csv_rows(manifest_path, MANIFEST_COLUMNS, [rows[k] for k in paths if k in rows])

        def go(self, d: int) -> None:
            self.idx = max(0, min(len(paths) - 1, self.idx + d))
            self.show_current()

    w = Win()
    w.show()
    return app.exec()
