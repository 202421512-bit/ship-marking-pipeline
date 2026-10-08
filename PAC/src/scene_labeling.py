"""Manual scene transcription tool (--label-scenes).

Shows each full-scene image with the detector's regions outlined (so the labeller can find the
marking) but NEVER shows any model reading - the transcription must come from the person.
Saves to data/scene_labels.csv after every change; reopening resumes at the first unlabelled scene.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .config import DATA_DIR, PROJECT_ROOT, SYMBOLS_DIR, AppConfig
from .dataset import read_csv_rows, rel, write_csv_rows

COLUMNS = ["image_path", "transcription", "unreadable", "uncertain", "labeled_at", "note"]


def scene_paths(cfg: AppConfig) -> List[str]:
    import re

    folder = SYMBOLS_DIR / cfg.dataset.informal_dir
    return sorted(rel(p) for p in folder.glob("*.png") if re.match(cfg.dataset.scene_pattern, p.stem))


def load_labels(cfg: AppConfig) -> Dict[str, Dict[str, str]]:
    return {r["image_path"]: r for r in read_csv_rows(DATA_DIR / cfg.scene.labels_file)}


def save_labels(cfg: AppConfig, labels: Dict[str, Dict[str, str]], order: List[str]) -> None:
    rows = [labels.get(p, {"image_path": p, "transcription": "", "unreadable": "", "uncertain": "",
                           "labeled_at": "", "note": ""}) for p in order]
    write_csv_rows(DATA_DIR / cfg.scene.labels_file, COLUMNS, rows)


def run_labeling_gui(cfg: AppConfig) -> int:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (QApplication, QCheckBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
                                 QVBoxLayout, QWidget)

    from .image_io import load_image
    from .text_detection_integration import Detector

    paths = scene_paths(cfg)
    if not paths:
        print("[LABEL] no scene_*.png found")
        return 2
    labels = load_labels(cfg)
    try:
        det = Detector(cfg)
    except FileNotFoundError:
        det = None

    app = QApplication.instance() or QApplication(sys.argv)

    class Win(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Scene transcription (manual) - type exactly what is written; no model output shown")
            self.resize(1200, 860)
            self.idx = next((k for k, p in enumerate(paths) if not (labels.get(p, {}).get("transcription")
                                                                   or labels.get(p, {}).get("unreadable") in ("True", "1", "true"))), 0)
            self.img = QLabel()
            self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.info = QLabel()
            self.text = QLineEdit()
            self.text.setPlaceholderText("Transcription (all marking text, left-to-right; separate lines with ' / ')")
            self.unreadable = QCheckBox("Unreadable")
            self.uncertain = QCheckBox("Uncertain")
            self.note = QLineEdit()
            self.note.setPlaceholderText("optional note")
            prev_b, next_b, save_b = QPushButton("◀ Prev"), QPushButton("Save + Next ▶"), QPushButton("Save")
            prev_b.clicked.connect(lambda: self.go(-1))
            next_b.clicked.connect(lambda: (self.store(), self.go(1)))
            save_b.clicked.connect(self.store)
            self.text.returnPressed.connect(lambda: (self.store(), self.go(1)))
            row = QHBoxLayout()
            for w in (self.unreadable, self.uncertain, self.note, prev_b, save_b, next_b):
                row.addWidget(w)
            lay = QVBoxLayout()
            lay.addWidget(self.info)
            lay.addWidget(self.img, 1)
            lay.addWidget(self.text)
            lay.addLayout(row)
            c = QWidget()
            c.setLayout(lay)
            self.setCentralWidget(c)
            self.show_current()

        def show_current(self) -> None:
            p = paths[self.idx]
            img = load_image(PROJECT_ROOT / p).image
            img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vis = img.copy()
            if det is not None:
                polys, _ = det.detect(img)
                for q in polys:
                    cv2.polylines(vis, [q.astype(np.int32)], True, (0, 200, 255), 2)
            rgb = np.ascontiguousarray(vis[..., ::-1])
            q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format.Format_RGB888).copy()
            self.img.setPixmap(QPixmap.fromImage(q).scaled(1150, 700, Qt.AspectRatioMode.KeepAspectRatio))
            lab = labels.get(p, {})
            self.text.setText(lab.get("transcription", ""))
            self.unreadable.setChecked(lab.get("unreadable", "") in ("True", "1", "true"))
            self.uncertain.setChecked(lab.get("uncertain", "") in ("True", "1", "true"))
            self.note.setText(lab.get("note", ""))
            done = sum(1 for x in paths if labels.get(x, {}).get("transcription")
                       or labels.get(x, {}).get("unreadable") in ("True", "1", "true"))
            self.info.setText(f"{self.idx + 1}/{len(paths)}  {Path(p).name}   labelled {done}/{len(paths)}   "
                              "(orange boxes = detector regions; reading is NOT shown)")
            self.text.setFocus()

        def store(self) -> None:
            p = paths[self.idx]
            labels[p] = {"image_path": p, "transcription": self.text.text().strip(),
                         "unreadable": str(self.unreadable.isChecked()), "uncertain": str(self.uncertain.isChecked()),
                         "labeled_at": datetime.now().isoformat(timespec="seconds"), "note": self.note.text().strip()}
            save_labels(cfg, labels, paths)

        def go(self, d: int) -> None:
            self.idx = max(0, min(len(paths) - 1, self.idx + d))
            self.show_current()

    w = Win()
    w.show()
    return app.exec()
