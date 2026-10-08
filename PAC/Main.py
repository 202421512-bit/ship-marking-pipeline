"""PAC Mission 3 - Ship block marking analysis (research prototype).

Usage:
    python Main.py --help
    python Main.py --demo
    python Main.py --input <single-character crop>.png [--no-vlm]
    python Main.py --real-marking-recognition   (full marking strings, real data)
    python Main.py --gui
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    PROJECT_SUBTITLE,
    PROJECT_TITLE,
    PROTOTYPE_LABEL,
    RESULTS_DIR,
    SUPPORTED_EXTENSIONS,
    AppConfig,
    ensure_directories,
)
from src.image_io import ImageLoadError, write_demo_files
from src.pipeline import PipelineResult, run_pipeline

LINE = "=" * 50


# ====================================================================== CLI
def format_report(r: PipelineResult) -> str:
    """Build the final terminal report."""
    obs, res, g = r.observed.topology, r.restored.topology, r.restored.geometry
    best, b, v, f, d = r.restoration.best, r.bayesian, r.vlm, r.fusion, r.decision

    def arrow(a: int, c: int) -> str:
        return f"{c}" if a == c else f"{c}   (observed {a} -> restored {c})"

    lines: List[str] = [
        LINE, PROJECT_TITLE, PROJECT_SUBTITLE, LINE, f"({PROTOTYPE_LABEL})", "",
        "[INPUT]",
        f"Path = {r.result_json.get('raw_image', r.loaded.path)}",
        f"Size = {r.loaded.width} x {r.loaded.height} px, channels = {r.loaded.channels}",
        f"SHA-256 = {r.loaded.sha256[:32]}...",
        "",
        "[TOPOLOGY]",
        f"β0 = {arrow(obs.beta_0, res.beta_0)}",
        f"β1 = {arrow(obs.beta_1, res.beta_1)}",
        f"Euler = {arrow(obs.euler_characteristic, res.euler_characteristic)}",
        f"Skeleton Length = {arrow(obs.skeleton_length, res.skeleton_length)}",
        f"Endpoints = {arrow(obs.endpoints, res.endpoints)}",
        f"Branch Points = {arrow(obs.branch_points, res.branch_points)}",
        f"Persistent Homology = {obs.persistence.get('status')}"
        + (f" (near-holes in observed = {obs.persistence.get('near_hole_count')})"
           if obs.persistence.get("status") == "OK" else ""),
        "",
        "[RESTORATION]",
        "Preliminary (Bayesian before restoration) = "
        + ", ".join(f"{c}:{p:.3f}" for c, p in r.bayesian_before.top3),
        f"Best Operation = {best.operation}  (reference prototype '{best.reference_class}')",
        f"J = {best.J:.4f}",
        f"L_data = {best.L_data:.4f}",
        f"L_topology = {best.L_topology:.4f}",
        f"L_geometry = {best.L_geometry:.4f}",
        f"L_change = {best.L_change:.4f}",
        "",
        "[GEOMETRY]",
        f"Aspect Ratio = {g.aspect_ratio:.4f}",
        f"Area Ratio = {g.area_ratio:.4f}",
        f"Circularity = {g.circularity:.4f}",
        f"Solidity = {g.solidity:.4f}",
        f"Eccentricity = {g.eccentricity:.4f}",
        "",
        "[BAYESIAN]  (Posterior Probability w.r.t. prototype reference model, not calibrated)",
        f"Before restoration: Top-1 = {r.bayesian_before.top1}, Posterior = {r.bayesian_before.top1_posterior:.4f}, "
        f"Entropy = {r.bayesian_before.entropy:.4f}",
        "After restoration:",
        f"Top-1 = {b.top1}",
        f"Posterior = {b.top1_posterior:.4f}",
        f"Entropy = {b.entropy:.4f} nats (normalized {b.normalized_entropy:.4f})",
        "Top-5 = " + ", ".join(f"{c}:{p:.3f}" for c, p in b.top5),
        "",
        "[VLM]  (Model-Reported Confidence, not calibrated)",
        f"Top-1 = {v.top_candidate if v.top_candidate else '-'}",
        f"Reported Confidence = {v.confidence:.3f}" if v.confidence is not None else "Reported Confidence = -",
        f"Status = {v.status}" + (f"  ({v.message})" if v.message else ""),
        "",
        "[FUSION]",
        f"Final Candidate = {f.final_candidate}",
        f"Fusion Score = {f.fusion_score:.4f}  ({f.method})",
        f"Agreement = {f.agreement}",
        "",
        "[DECISION]  (demo decision thresholds)",
        f"Level = {d.level}",
        f"Final Decision = {d.final_decision}",
        f"Topology Consistency = {d.topology_consistency['consistent']}",
        f"Geometry Consistency = {d.geometry_consistency['consistent']} "
        f"(mean |z| = {d.geometry_consistency['mean_abs_z']})",
    ]
    lines += [f"  - {reason}" for reason in d.reasons]
    lines += ["", "[TIMING]"]
    lines += [f"{k} = {val:.2f} ms" for k, val in r.timing_ms.items()]
    lines += ["", "[OUTPUT]"] + [f"{k} = {val}" for k, val in r.files.items()]
    lines += [LINE]
    return "\n".join(lines)


def run_cli(input_path: Path, cfg: AppConfig, use_vlm: bool, output_dir: Path,
            rebuild: bool, extra: Optional[Dict[str, object]] = None) -> int:
    """Run the pipeline once and print the report. Returns a process exit code."""
    try:
        result = run_pipeline(input_path, cfg, use_vlm=use_vlm, output_dir=output_dir,
                              rebuild_prototypes=rebuild, extra_input_info=extra,
                              progress=lambda s: print(f"  ... {s}"))
    except ImageLoadError as exc:
        print(f"[ERROR] {exc}")
        return 2
    except Exception as exc:
        traceback.print_exc()
        print(f"[ERROR] Pipeline failed: {type(exc).__name__}: {exc}")
        return 1
    print(format_report(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command line interface definition."""
    parser = argparse.ArgumentParser(
        prog="Main.py",
        description=("PAC Mission 3 research prototype: topology + geometry + Bayesian + VLM analysis "
                     "of a single cropped ship-block marking, with human-review routing."),
    )
    parser.add_argument("--demo", action="store_true",
                        help="generate a degraded synthetic '8' (seed 42) and analyze it")
    parser.add_argument("--input", type=Path, help=f"cropped marking image {SUPPORTED_EXTENSIONS}")
    parser.add_argument("--no-vlm", action="store_true", help="skip VLM verification (Bayesian only)")
    parser.add_argument("--gui", action="store_true", help="launch the PyQt6 GUI")
    parser.add_argument("--output", type=Path, default=None,
                        help="results directory (default: results/demo_synthetic/ for --demo, "
                             "results/single_image/ for --input / --gui)")
    parser.add_argument("--rebuild-prototypes", action="store_true",
                        help="rebuild the prototype reference model cache")
    parser.add_argument("--dataset-audit", action="store_true",
                        help="scan Symbols/, segment characters, validate transcriptions, write manifest")
    parser.add_argument("--train-reference", action="store_true",
                        help="re-run the audit, then build the DATA-FITTED reference model from validated "
                             "transcriptions (run again after editing data/transcriptions.csv)")
    parser.add_argument("--real-symbols", action="store_true",
                        help="REAL DATA EXPERIMENT on data/input/Symbols (all images; welding CV, "
                             "formal/informal analysis, OOD probe, scenes) -> results/real_symbols/")
    parser.add_argument("--welding-evaluate", action="store_true",
                        help="REAL DATA: welding symbol cross-validation only")
    parser.add_argument("--character-experiment", action="store_true",
                        help="REAL DATA: geometry vs topology vs combined on transcribed characters "
                             "(formal, informal, formal->informal, formal+informal->informal) "
                             "-> results/character_experiment/")
    parser.add_argument("--degradation-experiment", action="store_true",
                        help="REAL DATA: clean-train / degraded-test robustness of geometry vs topology vs combined "
                             "-> results/degradation_experiment/")
    parser.add_argument("--uncertainty-validation", action="store_true",
                        help="REAL DATA: calibration / error detection / selective classification of the existing "
                             "Bayesian posterior -> results/uncertainty_validation/")
    parser.add_argument("--real-marking-recognition", action="store_true",
                        help="REAL DATA: whole-marking recognition check (segmentation -> topology/geometry -> "
                             "Bayesian -> string), leave-one-group-out -> results/real_marking_recognition/")
    parser.add_argument("--direction-aware-experiment", action="store_true",
                        help="REAL DATA (exploratory): existing Topo+Geo vs +direction vs +line-relative features, "
                             "fixed segmentation -> results/direction_aware_experiment/")
    parser.add_argument("--text-detection-experiment", action="store_true",
                        help="REAL DATA: PP-OCRv3 DB text detection (OpenCV DNN) -> existing segmentation -> "
                             "topology/geometry/direction recognition; CRNN comparison -> "
                             "results/text_detection_integration/")
    parser.add_argument("--full-scene-stabilization", action="store_true",
                        help="REAL DATA: detection -> padded crop -> segmentation -> topology recognition; tight vs "
                             "padded, scene evaluation from manual labels -> results/full_scene_stabilization/")
    parser.add_argument("--label-scenes", action="store_true",
                        help="open the manual scene transcription tool (saves data/scene_labels.csv)")
    parser.add_argument("--scene-error-audit", action="store_true",
                        help="REAL DATA: character-level root causes of failed scene readings (frozen system) "
                             "-> results/scene_error_audit/")
    parser.add_argument("--symbol-expansion", action="store_true",
                        help="segmentation repair regression + symbol registry + new-data split audit / experimental "
                             "training -> results/symbol_expansion/")
    parser.add_argument("--label-new-data", action="store_true",
                        help="register NEW symbol / marking images from data/new_symbols/incoming/ "
                             "(separate from --label-scenes)")
    parser.add_argument("--show-results", action="store_true",
                        help="open results/real_symbols/final_dashboard.png in the default viewer")
    parser.add_argument("--reference", choices=["auto", "synthetic", "data"], default="auto",
                        help="reference model for --input/--gui (default auto: data-fitted if built). "
                             "--demo always uses synthetic")
    return parser


def run_real_symbols(cfg: AppConfig, welding_only: bool = False) -> int:
    """REAL SYMBOLS DATA EXPERIMENT (never uses synthetic prototypes, demo images or test.png)."""
    import time

    from src.config import REAL_RESULTS_DIR, SYMBOLS_DIR
    from src.real_experiment import run_real_experiment
    from src.real_report import save_reports, terminal_summary
    from src.real_visualization import render_all

    if not SYMBOLS_DIR.exists():
        print(f"[ERROR] Real data folder not found: {SYMBOLS_DIR}\n        Put the Symbols folder in data/input/.")
        return 2
    t0 = time.perf_counter()
    try:
        result = run_real_experiment(cfg, SYMBOLS_DIR, REAL_RESULTS_DIR, welding_only=welding_only)
        print("[REAL] rendering figures")
        render_all(result, cfg)
        save_reports(result)
    except AssertionError as exc:
        print(f"[ERROR] integrity check failed (run aborted): {exc}")
        return 3
    except Exception as exc:
        traceback.print_exc()
        print(f"[ERROR] real experiment failed: {type(exc).__name__}: {exc}")
        return 1
    print(terminal_summary(result))
    print(f"[REAL] done in {time.perf_counter() - t0:.1f} s")
    return 0


def run_character_cli(cfg: AppConfig) -> int:
    """CHARACTER EXPERIMENT; any leakage assertion aborts with ERROR."""
    from src.character_experiment import LeakageError, run_character_experiment
    from src.character_report import render_and_save, terminal_summary

    try:
        result = run_character_experiment(cfg)
        render_and_save(result, cfg)
    except LeakageError as exc:
        print(f"[ERROR] DATA LEAKAGE ASSERTION FAILED - experiment aborted: {exc}")
        return 3
    except Exception as exc:
        traceback.print_exc()
        print(f"[ERROR] character experiment failed: {type(exc).__name__}: {exc}")
        return 1
    print(terminal_summary(result))
    return 0


def run_degradation_cli(cfg: AppConfig) -> int:
    """DEGRADATION EXPERIMENT; leakage assertion failures abort with ERROR."""
    from src.degradation_experiment import LeakageError, run_degradation_experiment
    from src.degradation_report import render_and_save, terminal_summary

    try:
        result = run_degradation_experiment(cfg)
        render_and_save(result, cfg)
    except LeakageError as exc:
        print(f"[ERROR] DATA LEAKAGE / FITTING ASSERTION FAILED - experiment aborted: {exc}")
        return 3
    except Exception as exc:
        traceback.print_exc()
        print(f"[ERROR] degradation experiment failed: {type(exc).__name__}: {exc}")
        return 1
    print(terminal_summary(result))
    return 0


def run_uncertainty_cli(cfg: AppConfig) -> int:
    """UNCERTAINTY VALIDATION; partition-overlap assertion failures abort with ERROR."""
    from src.uncertainty_report import render_and_save, terminal_summary
    from src.uncertainty_validation import LeakageError, run_uncertainty_validation

    try:
        result = run_uncertainty_validation(cfg)
        render_and_save(result, cfg)
    except LeakageError as exc:
        print(f"[ERROR] PARTITION LEAKAGE ASSERTION FAILED - experiment aborted: {exc}")
        return 3
    except Exception as exc:
        traceback.print_exc()
        print(f"[ERROR] uncertainty validation failed: {type(exc).__name__}: {exc}")
        return 1
    print(terminal_summary(result, cfg))
    return 0


def run_marking_cli(cfg: AppConfig) -> int:
    """Whole-marking recognition check on real Formal / Informal images."""
    from src.marking_report import render_and_save, terminal_summary
    from src.real_marking_recognition import run_marking_recognition

    try:
        result = run_marking_recognition(cfg)
        render_and_save(result, cfg)
    except Exception as exc:
        traceback.print_exc()
        print(f"[ERROR] marking recognition check failed: {type(exc).__name__}: {exc}")
        return 1
    print(terminal_summary(result, cfg))
    return 0


def show_results() -> int:
    """Open the real-data dashboard with the OS default image viewer."""
    from src.config import REAL_RESULTS_DIR

    path = REAL_RESULTS_DIR / "final_dashboard.png"
    if not path.exists() or path.stat().st_size == 0:
        print(f"[ERROR] {path} not found.\n        Run first:  .venv\\Scripts\\python.exe Main.py --real-symbols")
        return 2
    try:
        if os.name == "nt":
            os.startfile(str(path))  # noqa: S606 - default image viewer
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"[ERROR] could not open viewer ({exc}). File: {path}")
        return 1
    print(f"[SHOW] opened {path}")
    return 0


# ====================================================================== GUI
def launch_gui(cfg: AppConfig, use_vlm: bool, output_dir: Path) -> int:
    """Start the PyQt6 GUI (imported lazily so the CLI does not need a display)."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(cfg, use_vlm=use_vlm, output_dir=output_dir)
    window.show()
    return app.exec()


try:  # GUI classes are defined only when PyQt6 is importable
    from PyQt6.QtCore import QObject, QRectF, Qt, QThread, QUrl, pyqtSignal
    from PyQt6.QtGui import QDesktopServices, QImage, QImageReader, QPainter, QPixmap, QWheelEvent
    from PyQt6.QtWidgets import (
        QCheckBox, QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
        QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter, QStatusBar, QTextBrowser,
        QVBoxLayout, QWidget,
    )

    UI_TEXT: Dict[str, Dict[str, str]] = {
        "ko": {
            "open": "도면/이미지 열기", "demo": "Demo", "run": "분석 시작", "folder": "결과 폴더 열기",
            "vlm": "VLM 검증 사용", "view_orig": "원본 이미지", "view_rest": "복원 이미지",
            "view_skel": "Skeleton overlay", "tooltip": "휠=확대/축소, 드래그=패닝, 더블클릭=맞춤",
            "no_result": "결과 없음", "start_hint": "이미지를 열거나 Demo를 누르세요.",
            "read_fail": "이미지를 읽을 수 없습니다", "input": "입력", "press_run": "'분석 시작'을 누르세요.",
            "waiting": "분석 대기", "open_fail": "열기 실패", "demo_fail": "Demo 실패", "running": "분석 중…",
            "error": "오류", "analysis_error": "분석 오류", "done": "완료 - 결과", "folder_fail": "폴더 열기 실패",
            "real": "REAL DATA RESULTS",
            "real_missing": "결과가 없습니다. 먼저 'python Main.py --real-symbols'를 실행하세요.",
        },
        "en": {
            "open": "Open drawing/image", "demo": "Demo", "run": "Start analysis", "folder": "Open results folder",
            "vlm": "Use VLM verification", "view_orig": "Original image", "view_rest": "Restored image",
            "view_skel": "Skeleton overlay", "tooltip": "wheel=zoom, drag=pan, double-click=fit",
            "no_result": "No result", "start_hint": "Open an image or press Demo.",
            "read_fail": "Cannot read image", "input": "Input", "press_run": "Press 'Start analysis'.",
            "waiting": "Ready to analyze", "open_fail": "Open failed", "demo_fail": "Demo failed",
            "running": "Analyzing…", "error": "Error", "analysis_error": "Analysis error", "done": "Done - results",
            "folder_fail": "Could not open folder",
            "real": "REAL DATA RESULTS",
            "real_missing": "No results yet. Run 'python Main.py --real-symbols' first.",
        },
    }

    def select_ui_language() -> str:
        """Use Korean labels only if a Korean-capable font is installed (else English)."""
        from PyQt6.QtGui import QFont, QFontDatabase
        from PyQt6.QtWidgets import QApplication

        from src.config import KOREAN_FONT_CANDIDATES

        families = set(QFontDatabase.families(QFontDatabase.WritingSystem.Korean))
        preferred = next((f for f in KOREAN_FONT_CANDIDATES if f in families), None)
        if preferred is None and not families:
            return "en"
        app = QApplication.instance()
        if app is not None and preferred is not None:
            font = QFont(preferred)
            font.setPointSize(app.font().pointSize())
            app.setFont(font)
        return "ko"

    def numpy_to_qimage(array: np.ndarray) -> QImage:
        """Convert gray / RGB / bool numpy arrays to a deep-copied QImage."""
        if array.dtype == bool:
            array = np.where(array, 40, 250).astype(np.uint8)
        if array.dtype != np.uint8:
            array = np.clip(array * (255.0 if array.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        array = np.ascontiguousarray(array)
        h, w = array.shape[:2]
        if array.ndim == 2:
            return QImage(array.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        return QImage(array.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    class ZoomableImageView(QGraphicsView):
        """Image view with mouse-wheel zoom, drag panning and double-click fit."""

        def __init__(self, title: str, cfg: AppConfig, tooltip: str = "") -> None:
            """Create an empty view."""
            super().__init__()
            self.cfg = cfg
            self.title = title
            self.setScene(QGraphicsScene(self))
            self.item = QGraphicsPixmapItem()
            self.item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            self.scene().addItem(self.item)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            self.setBackgroundBrush(Qt.GlobalColor.white)
            self.setToolTip(f"{title}: {tooltip}")
            self._zoom = 1.0

        def set_image(self, image: QImage) -> None:
            """Display a QImage and fit it to the view."""
            self.item.setPixmap(QPixmap.fromImage(image))
            self.scene().setSceneRect(QRectF(self.item.pixmap().rect()))
            self.fit()

        def fit(self) -> None:
            """Fit the whole image into the viewport."""
            if self.item.pixmap().isNull():
                return
            self.resetTransform()
            self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()

        def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
            """Zoom in/out around the cursor."""
            step = self.cfg.gui.zoom_step if event.angleDelta().y() > 0 else 1.0 / self.cfg.gui.zoom_step
            new = self._zoom * step
            if self.cfg.gui.zoom_min <= new <= self.cfg.gui.zoom_max:
                self.scale(step, step)
                self._zoom = new

        def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
            """Double click resets the zoom to fit."""
            self.fit()

    class PipelineWorker(QObject):
        """Runs the analysis pipeline in a background thread."""

        progress = pyqtSignal(str)
        finished = pyqtSignal(object)
        failed = pyqtSignal(str)

        def __init__(self, path: Path, cfg: AppConfig, use_vlm: bool, output_dir: Path,
                     extra: Optional[Dict[str, object]]) -> None:
            """Store job parameters."""
            super().__init__()
            self.path, self.cfg, self.use_vlm, self.output_dir, self.extra = path, cfg, use_vlm, output_dir, extra

        def run(self) -> None:
            """Execute the pipeline and emit the result or an error message."""
            try:
                result = run_pipeline(self.path, self.cfg, use_vlm=self.use_vlm, output_dir=self.output_dir,
                                      progress=self.progress.emit, extra_input_info=self.extra)
                self.finished.emit(result)
            except Exception as exc:  # reported in the GUI, never crashes the app
                self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")

    class MainWindow(QMainWindow):
        """Main GUI window."""

        def __init__(self, cfg: AppConfig, use_vlm: bool = True, output_dir: Path = RESULTS_DIR) -> None:
            """Build widgets and connect actions."""
            super().__init__()
            self.cfg = cfg
            self.output_dir = output_dir
            self.current_path: Optional[Path] = None
            self.extra_info: Optional[Dict[str, object]] = None
            self.thread: Optional[QThread] = None
            self.worker: Optional[PipelineWorker] = None
            self.last_result: Optional[PipelineResult] = None
            self.setWindowTitle(f"{PROJECT_TITLE} - {PROJECT_SUBTITLE}  [{PROTOTYPE_LABEL}]")
            self.resize(*cfg.gui.window_size)
            self.lang = select_ui_language()
            self.t = UI_TEXT[self.lang]
            t = self.t

            self.btn_open = QPushButton(t["open"])
            self.btn_demo = QPushButton(t["demo"])
            self.btn_run = QPushButton(t["run"])
            self.btn_folder = QPushButton(t["folder"])
            self.btn_real = QPushButton(t["real"])
            self.btn_real.clicked.connect(self.show_real_results)
            self.chk_vlm = QCheckBox(t["vlm"])
            self.chk_vlm.setChecked(use_vlm)
            self.btn_run.setEnabled(False)
            for btn in (self.btn_open, self.btn_demo, self.btn_run, self.btn_folder):
                btn.setMinimumHeight(36)
            self.btn_open.clicked.connect(self.open_image)
            self.btn_demo.clicked.connect(self.load_demo)
            self.btn_run.clicked.connect(self.start_analysis)
            self.btn_folder.clicked.connect(self.open_results_folder)

            top = QHBoxLayout()
            self.btn_real.setMinimumHeight(36)
            for w in (self.btn_open, self.btn_demo, self.btn_run, self.btn_folder, self.btn_real, self.chk_vlm):
                top.addWidget(w)
            top.addStretch(1)

            self.views = [ZoomableImageView(t[k], cfg, t["tooltip"]) for k in ("view_orig", "view_rest", "view_skel")]
            images = QHBoxLayout()
            for view in self.views:
                box = QVBoxLayout()
                label = QLabel(view.title)
                label.setStyleSheet("font-weight: bold; font-size: 13px;")
                box.addWidget(label)
                box.addWidget(view, 1)
                images.addLayout(box, 1)
            image_panel = QWidget()
            image_panel.setLayout(images)

            self.decision_label = QLabel(t["no_result"])
            self.decision_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.decision_label.setMinimumHeight(64)
            self._style_decision(None)
            self.report = QTextBrowser()
            self.report.setOpenExternalLinks(False)
            right = QVBoxLayout()
            right.addWidget(self.decision_label)
            right.addWidget(self.report, 1)
            right_panel = QWidget()
            right_panel.setLayout(right)

            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(image_panel)
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1)

            root = QVBoxLayout()
            root.addLayout(top)
            root.addWidget(splitter, 1)
            central = QWidget()
            central.setLayout(root)
            self.setCentralWidget(central)
            self.setStatusBar(QStatusBar())
            self.statusBar().showMessage(t["start_hint"])

        # ------------------------------------------------------------ actions
        def _style_decision(self, level: Optional[str]) -> None:
            colors = {"HIGH_CONFIDENCE": "#0ca30c", "REVIEW_REQUIRED": "#fab219", "AMBIGUOUS": "#d03b3b"}
            color = colors.get(level or "", "#c3c2b7")
            self.decision_label.setStyleSheet(
                f"font-size: 20px; font-weight: bold; color: #0b0b0b; border: 3px solid {color};"
                f"border-radius: 8px; background: #fcfcfb; padding: 6px;")

        def load_preview(self, path: Path) -> QImage:
            """Read a downscaled preview so huge drawings do not exhaust memory."""
            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            size = reader.size()
            limit = self.cfg.gui.preview_max_side
            if size.isValid() and max(size.width(), size.height()) > limit:
                size.scale(limit, limit, Qt.AspectRatioMode.KeepAspectRatio)
                reader.setScaledSize(size)
            image = reader.read()
            if image.isNull():
                raise ImageLoadError(f"{self.t['read_fail']}: {path} ({reader.errorString()})")
            return image

        def set_input(self, path: Path, extra: Optional[Dict[str, object]] = None) -> None:
            """Register and display a new input image."""
            image = self.load_preview(path)
            self.current_path, self.extra_info = path, extra
            self.views[0].set_image(image)
            for view in self.views[1:]:
                view.item.setPixmap(QPixmap())
            self.report.setHtml(f"<b>{self.t['input']}:</b> {path.name}<br>{self.t['press_run']}")
            self.decision_label.setText(self.t["waiting"])
            self._style_decision(None)
            self.btn_run.setEnabled(True)
            self.statusBar().showMessage(f"Loaded {path} (preview {image.width()}x{image.height()})")

        def open_image(self) -> None:
            """File dialog -> display image."""
            patterns = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
            name, _ = QFileDialog.getOpenFileName(self, self.t["open"], str(PROJECT_ROOT / "data" / "input"),
                                                  f"Images ({patterns})")
            if not name:
                return
            try:
                self.set_input(Path(name))
            except Exception as exc:
                QMessageBox.critical(self, self.t["open_fail"], str(exc))

        def load_demo(self) -> None:
            """Generate the demo image and load it."""
            try:
                path, info = write_demo_files(self.cfg.demo)
                self.set_input(path, {"demo": info})
            except Exception as exc:
                QMessageBox.critical(self, self.t["demo_fail"], str(exc))

        def start_analysis(self) -> None:
            """Run the pipeline in a worker thread."""
            if self.current_path is None or (self.thread is not None and self.thread.isRunning()):
                return
            self.btn_run.setEnabled(False)
            self.decision_label.setText(self.t["running"])
            self.thread = QThread(self)
            from src.config import DEMO_RESULTS_DIR

            is_demo = bool((self.extra_info or {}).get("demo"))
            run_cfg = replace(self.cfg, reference_model="synthetic") if is_demo else self.cfg
            self.worker = PipelineWorker(self.current_path, run_cfg, self.chk_vlm.isChecked(),
                                         DEMO_RESULTS_DIR if is_demo else self.output_dir, self.extra_info)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.progress.connect(lambda s: self.statusBar().showMessage(s))
            self.worker.finished.connect(self.show_result)
            self.worker.failed.connect(self.show_error)
            self.worker.finished.connect(self.thread.quit)
            self.worker.failed.connect(self.thread.quit)
            self.thread.start()

        def show_error(self, message: str) -> None:
            """Report a pipeline failure without closing the app."""
            self.btn_run.setEnabled(True)
            self.decision_label.setText(self.t["error"])
            QMessageBox.critical(self, self.t["analysis_error"], message[:2000])

        def show_result(self, r: PipelineResult) -> None:
            """Show restored image, skeleton overlay and the evidence report."""
            self.last_result = r
            self.btn_run.setEnabled(True)
            mask = r.restoration.best.mask
            self.views[1].set_image(numpy_to_qimage(mask))
            overlay = np.where(mask[..., None], np.array([150, 150, 145], np.uint8),
                               np.array([252, 252, 251], np.uint8)).astype(np.uint8)
            overlay[r.restored.topology.skeleton] = (235, 104, 52)
            for y, x in r.restored.topology.endpoint_coords:
                overlay[y, x] = (42, 120, 214)
            for y, x in r.restored.topology.branch_coords:
                overlay[y, x] = (74, 58, 167)
            self.views[2].set_image(numpy_to_qimage(overlay))
            d = r.decision
            icon = {"HIGH_CONFIDENCE": "✔", "REVIEW_REQUIRED": "⚠", "AMBIGUOUS": "✖"}.get(d.level, "")
            self.decision_label.setText(f"{icon} {d.level.replace('_', ' ')}\n{d.final_decision.replace('_', ' ')}")
            self._style_decision(d.level)
            self.report.setHtml(self.report_html(r))
            self.statusBar().showMessage(f"{self.t['done']}: {r.output_dir}")

        @staticmethod
        def report_html(r: PipelineResult) -> str:
            """HTML summary for the right-hand panel."""
            o, t, g = r.observed.topology, r.restored.topology, r.restored.geometry
            b, v, f, d, best = r.bayesian, r.vlm, r.fusion, r.decision, r.restoration.best

            def sec(title: str, rows: List[tuple]) -> str:
                body = "".join(f"<tr><td>{k}</td><td align='right'><b>{val}</b></td></tr>" for k, val in rows)
                return f"<h3 style='margin-bottom:2px'>{title}</h3><table width='100%'>{body}</table>"

            html = [
                sec("Topology (observed → restored)", [
                    ("β0", f"{o.beta_0} → {t.beta_0}"), ("β1", f"{o.beta_1} → {t.beta_1}"),
                    ("Euler χ", f"{o.euler_characteristic} → {t.euler_characteristic}"),
                    ("Skeleton length", f"{o.skeleton_length} → {t.skeleton_length}"),
                    ("Endpoints", f"{o.endpoints} → {t.endpoints}"),
                    ("Branch points", f"{o.branch_points} → {t.branch_points}"),
                    ("Persistent H1 near-holes", o.persistence.get("near_hole_count", "n/a"))]),
                sec("Restoration", [
                    ("Best operation", best.operation), ("J", f"{best.J:.4f}"),
                    ("L_data / L_topology", f"{best.L_data:.3f} / {best.L_topology:.3f}"),
                    ("L_geometry / L_change", f"{best.L_geometry:.3f} / {best.L_change:.3f}")]),
                sec("Geometry", [
                    ("Aspect ratio", f"{g.aspect_ratio:.3f}"), ("Area ratio", f"{g.area_ratio:.3f}"),
                    ("Circularity", f"{g.circularity:.3f}"), ("Solidity", f"{g.solidity:.3f}"),
                    ("Eccentricity", f"{g.eccentricity:.3f}")]),
                sec("Bayesian (Posterior Probability)", [
                    ("Before restoration", f"{r.bayesian_before.top1} {r.bayesian_before.top1_posterior:.3f} "
                                           f"(H {r.bayesian_before.entropy:.3f})"),
                    ("Top-1 (after)", b.top1), ("Posterior", f"{b.top1_posterior:.4f}"), ("Entropy", f"{b.entropy:.4f}"),
                    ("Top-5", ", ".join(f"{c}:{p:.3f}" for c, p in b.top5))]),
                sec("VLM (Model-Reported Confidence)", [
                    ("Status", v.status), ("Top-1", v.top_candidate or "-"),
                    ("Reported confidence", "-" if v.confidence is None else f"{v.confidence:.3f}"),
                    ("Alternatives", ", ".join(f"{a['label']}:{a['confidence']:.2f}" for a in v.alternatives) or "-")]),
                sec("Fusion", [
                    ("Final candidate", f.final_candidate), ("Fusion score", f"{f.fusion_score:.4f}"),
                    ("Agreement", f.agreement)]),
                sec("Decision (demo thresholds)", [
                    ("Level", d.level), ("Final", d.final_decision),
                    ("Topology consistency", d.topology_consistency["consistent"]),
                    ("Geometry consistency", d.geometry_consistency["consistent"])]),
                "<ul>" + "".join(f"<li>{x}</li>" for x in d.reasons) + "</ul>",
                "<p style='color:#898781'>Research prototype. Posterior is relative to the prototype reference "
                "model; VLM confidence is model-reported. Welding conditions are not determined.</p>",
            ]
            if v.message:
                html.insert(5, f"<p style='color:#52514e'>VLM: {v.message}</p>")
            return "".join(html)

        def show_real_results(self) -> None:
            """Tabbed viewer of results/real_symbols/*.png (REAL DATA EXPERIMENT)."""
            from PyQt6.QtWidgets import QDialog, QTabWidget

            from src.config import REAL_RESULTS_DIR

            pages = [("Final Dashboard", "final_dashboard.png"), ("Confusion Matrix", "welding_confusion_matrix.png"),
                     ("Model Comparison", "welding_model_comparison.png"),
                     ("Topology Contribution", "topology_contribution.png"),
                     ("Restoration Effect", "restoration_effect.png"),
                     ("Formal vs Informal", "formal_vs_informal.png"),
                     ("Scene Analysis", "scene_analysis_summary.png"),
                     ("Welding ROI", "welding_roi_preview.png")]
            available = [(name, REAL_RESULTS_DIR / f) for name, f in pages if (REAL_RESULTS_DIR / f).exists()]
            if not available:
                QMessageBox.information(self, self.t["real"], self.t["real_missing"])
                return
            dialog = QDialog(self)
            dialog.setWindowTitle("REAL DATA RESULTS - Source: data/input/Symbols")
            dialog.resize(1400, 950)
            tabs = QTabWidget(dialog)
            for name, path in available:
                view = ZoomableImageView(name, self.cfg, self.t["tooltip"])
                view.set_image(self.load_preview(path))
                tabs.addTab(view, name)
            layout = QVBoxLayout(dialog)
            layout.addWidget(tabs)
            dialog.show()
            self._real_dialog = dialog  # keep a reference (non-modal)

        def open_results_folder(self) -> None:
            """Open the results folder in Windows Explorer (or the OS file manager)."""
            self.output_dir.mkdir(parents=True, exist_ok=True)
            try:
                if os.name == "nt":
                    os.startfile(str(self.output_dir))  # noqa: S606 - opens Explorer
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(self.output_dir)])
                else:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))
            except Exception as exc:
                QMessageBox.warning(self, self.t["folder_fail"], str(exc))

except ImportError:  # pragma: no cover - PyQt6 is in requirements.txt
    MainWindow = None  # type: ignore[assignment]


# ====================================================================== main
def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_directories()
    cfg = replace(DEFAULT_CONFIG, reference_model=args.reference)
    use_vlm = not args.no_vlm

    if args.show_results:
        return show_results()
    if args.character_experiment:
        return run_character_cli(cfg)
    if args.degradation_experiment:
        return run_degradation_cli(cfg)
    if args.uncertainty_validation:
        return run_uncertainty_cli(cfg)
    if args.real_marking_recognition:
        return run_marking_cli(cfg)
    if args.label_new_data:
        from src.new_data_labeling import run_new_data_gui

        return run_new_data_gui(cfg)
    if args.symbol_expansion:
        from src.symbol_expansion import SplitLeakageError
        from src.symbol_expansion_report import run_all

        try:
            print(run_all(cfg))
        except SplitLeakageError as exc:
            print(f"[ERROR] SPLIT LEAKAGE - run aborted: {exc}")
            return 3
        except Exception as exc:
            traceback.print_exc()
            print(f"[ERROR] symbol expansion failed: {type(exc).__name__}: {exc}")
            return 1
        return 0
    if args.scene_error_audit:
        from src.scene_error_audit import run_audit
        from src.scene_error_report import build_and_write, summary

        try:
            res = run_audit(cfg)
            build_and_write(res)
        except Exception as exc:
            traceback.print_exc()
            print(f"[ERROR] scene error audit failed: {type(exc).__name__}: {exc}")
            return 1
        print(summary(res))
        return 0
    if args.label_scenes:
        from src.scene_labeling import run_labeling_gui

        return run_labeling_gui(cfg)
    if args.full_scene_stabilization:
        from src.config import SCENE_RESULTS_DIR
        from src.full_scene_report import render_and_save as fs_render, terminal_summary as fs_summary
        from src.full_scene_stabilization import evaluate as fs_evaluate, run_stabilization

        try:
            res = run_stabilization(cfg)
            ev = fs_evaluate(res, cfg)
            report = fs_render(res, ev, cfg, SCENE_RESULTS_DIR)
        except Exception as exc:
            traceback.print_exc()
            print(f"[ERROR] full-scene stabilization failed: {type(exc).__name__}: {exc}")
            return 1
        print(fs_summary(ev, report))
        return 0
    if args.text_detection_experiment:
        from src.text_detection_integration import run_text_detection
        from src.text_detection_report import render_and_save as td_render, terminal_summary as td_summary

        try:
            res = run_text_detection(cfg)
            td_render(res, cfg)
        except Exception as exc:
            traceback.print_exc()
            print(f"[ERROR] text detection experiment failed: {type(exc).__name__}: {exc}")
            return 1
        print(td_summary(res))
        return 0
    if args.direction_aware_experiment:
        from src.direction_aware_experiment import run_direction_experiment
        from src.direction_aware_report import render_and_save as dir_render, terminal_summary as dir_summary

        try:
            res = run_direction_experiment(cfg)
            dir_render(res, cfg)
        except Exception as exc:
            traceback.print_exc()
            print(f"[ERROR] direction-aware experiment failed: {type(exc).__name__}: {exc}")
            return 1
        print(dir_summary(res, cfg))
        return 0
    if args.real_symbols or args.welding_evaluate:
        return run_real_symbols(cfg, welding_only=args.welding_evaluate and not args.real_symbols)
    if args.dataset_audit or args.train_reference:
        from src.dataset import run_dataset_audit
        from src.prototypes import build_data_fitted_model

        try:
            if args.train_reference:
                build_data_fitted_model(cfg)   # includes a fresh audit
            else:
                run_dataset_audit(cfg)
        except Exception as exc:
            traceback.print_exc()
            print(f"[ERROR] dataset step failed: {type(exc).__name__}: {exc}")
            return 1
        return 0
    from src.config import DEMO_RESULTS_DIR, SINGLE_IMAGE_RESULTS_DIR

    if args.gui:
        if MainWindow is None:
            print("[ERROR] PyQt6 is not available.")
            return 1
        return launch_gui(cfg, use_vlm, args.output or SINGLE_IMAGE_RESULTS_DIR)
    if args.demo:
        path, info = write_demo_files(cfg.demo)
        print(f"[DEMO] generated degraded '{cfg.demo.target}' (seed {cfg.demo.seed}): {path}")
        print("[DEMO] SYNTHETIC DEMO - uses the synthetic font-prototype model, not a real-dataset evaluation")
        return run_cli(path, replace(cfg, reference_model="synthetic"), use_vlm, args.output or DEMO_RESULTS_DIR,
                       args.rebuild_prototypes, {"demo": info})
    if args.input:
        path = args.input if args.input.is_absolute() else (Path.cwd() / args.input)
        if not path.exists() and (PROJECT_ROOT / args.input).exists():
            path = PROJECT_ROOT / args.input
        return run_cli(path, cfg, use_vlm, args.output or SINGLE_IMAGE_RESULTS_DIR, args.rebuild_prototypes)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
