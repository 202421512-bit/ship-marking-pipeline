"""Outputs for TEXT DETECTION INTEGRATION."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from . import visualization as base  # noqa: E402
from .config import PROJECT_ROOT, AppConfig  # noqa: E402
from .dataset import FORMAL, INFORMAL, INFORMAL_SCENE  # noqa: E402
from .image_io import load_image  # noqa: E402
from .real_experiment import rel, write_csv  # noqa: E402
from .text_detection_integration import TDResult, casefold_alnum, evaluate  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, RED = base.SURFACE, base.INK, base.INK_2, base.MUTED, base.GRID, base.RED
BANNER = "REAL DATA · PP-OCRv3 DB detector (OpenCV DNN) → existing segmentation → Topo+Geo+Direction · scores uncalibrated"
DPI = 110


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_tables(r: TDResult) -> None:
    out = r.out_dir
    det_rows, rec_rows, fail_rows = [], [], []
    for i in r.images:
        for g in i.regions:
            det_rows.append({"image_path": i.path, "kind": i.kind, "region": g.index, "line": g.line,
                             "polygon": " ".join(f"{x:.0f},{y:.0f}" for x, y in g.poly),
                             "bbox": ",".join(map(str, g.bbox)), "detector_score": g.score,
                             "score_note": "raw DB detector score, not a probability",
                             "A_crnn_text": g.text_a, "B_topology_text": g.text_b,
                             "B_segments": len(g.segments), "spurious_vs_validated_boxes": g.spurious,
                             "B_candidates": " | ".join(" ".join(f"{c}:{p:.2f}" for c, p in s.get("candidates", []))
                                                        for s in g.segments),
                             "region_crop": f"detected_regions/{Path(i.path).stem}_r{g.index}.png"})
        rec_rows.append({"image_path": i.path, "kind": i.kind, "status": i.status,
                         "ground_truth": i.transcription if i.gt else "UNKNOWN",
                         "A_pretrained_ocr": i.text_a, "B_topology_geometry_direction": i.text_b,
                         "regions": len(i.regions), "reading_order": i.reading_order,
                         "B_exact": (i.text_b == i.gt) if i.gt else "NOT_EVALUABLE",
                         "A_exact_casefold_alnum": (casefold_alnum(i.text_a) == casefold_alnum(i.gt)) if i.gt else "NOT_EVALUABLE",
                         "B_exact_casefold_alnum": (casefold_alnum(i.text_b) == casefold_alnum(i.gt)) if i.gt else "NOT_EVALUABLE",
                         "proxy_detection_coverage": i.proxy_coverage})
        if i.gt:
            fail_rows.append({"image_path": i.path, "kind": i.kind, "status_original_segmentation": i.status,
                              "ground_truth_normalized": i.gt, "B_text": i.text_b, "A_text": i.text_a,
                              "failure_B": i.failure, "touching": i.flags.get("touching"),
                              "decimal_point": i.flags.get("decimal_point"),
                              "B_segments": sum(len(g.segments) for g in i.regions), "expected": len(i.gt),
                              "A_correct_casefold": casefold_alnum(i.text_a) == casefold_alnum(i.gt),
                              "B_correct_casefold": casefold_alnum(i.text_b) == casefold_alnum(i.gt)})
    write_csv(out / "detection_results.csv", det_rows)
    write_csv(out / "recognition_results.csv", rec_rows)
    write_csv(out / "failure_analysis.csv", fail_rows)


def _overlay(ax, i, img) -> None:
    ax.imshow(img if img.ndim == 2 else img[..., ::-1])
    for g in i.regions:
        col = base.ORANGE if g.spurious else base.BLUE
        ax.add_patch(Polygon(g.poly, closed=True, fill=False, ec=col, lw=1.8))
        x, y = g.poly[:, 0].min(), g.poly[:, 1].min()
        ax.text(x, max(0, y - 4), base.safe_text(f"{g.index}: A '{g.text_a}' | B '{g.text_b}'"), fontsize=7,
                color="white", bbox=dict(facecolor=col, alpha=0.8, pad=1, lw=0))
    ax.set_xticks([])
    ax.set_yticks([])


def scene_visualizations(r: TDResult) -> List[Path]:
    d = r.out_dir / "scene_visualizations"
    d.mkdir(exist_ok=True)
    paths = []
    scenes = [i for i in r.images if i.kind == INFORMAL_SCENE]
    for i in scenes:
        img = load_image(PROJECT_ROOT / i.path).image
        fig, ax = plt.subplots(figsize=(10, 7))
        _overlay(ax, i, img)
        ax.set_title(base.safe_text(f"{Path(i.path).name} · {len(i.regions)} regions · {i.reading_order} · "
                                    "NO detection ground truth"), fontsize=10)
        fig.tight_layout()
        paths.append(_save(fig, d / f"{Path(i.path).stem}_detected.png"))
    cols = 5
    rows = int(np.ceil(len(scenes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(25, 3.6 * rows + 1))
    fig.text(0.01, 0.995, f"All {len(scenes)} scenes: detected regions (blue) with A = pretrained CRNN, "
             "B = topology-based reading · detection accuracy NOT EVALUABLE", fontsize=14, fontweight="bold", va="top")
    for ax, i in zip(np.atleast_1d(axes).ravel(), scenes):
        _overlay(ax, i, load_image(PROJECT_ROOT / i.path).image)
        ax.set_title(f"{Path(i.path).stem} · {len(i.regions)} regions", fontsize=9)
    for ax in np.atleast_1d(axes).ravel()[len(scenes):]:
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths.append(_save(fig, d / "all_scenes_contact_sheet.png"))
    return paths


def dashboard(r: TDResult, path: Path) -> Path:
    m = r.report["metrics"]
    fig = plt.figure(figsize=(22, 22))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[0.35, 1.0, 1.1, 0.8], hspace=0.38, wspace=0.25)
    fig.text(0.012, 0.995, "PAC · REAL-IMAGE TEXT DETECTION INTEGRATION", fontsize=22, fontweight="bold", va="top")
    fig.text(0.012, 0.981, BANNER, fontsize=11, color=RED, va="top")
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    sc = m["scenes"]
    tiles = [("Scenes", sc["total"]), ("Scenes w/ regions", sc["with_regions"]), ("Scene regions", sc["regions"]),
             ("Crops evaluated", m["transcribed_images"]), ("B CER", f"{m['all']['B_cer_raw']:.3f}"),
             ("B exact", f"{m['all']['B_exact_raw']:.3f}"), ("Proxy coverage", f"{m['proxy_detection_coverage']['mean']:.3f}")]
    for k, (t, v) in enumerate(tiles):
        x0 = k / len(tiles)
        ax.add_patch(Rectangle((x0 + 0.004, 0), 1 / len(tiles) - 0.008, 0.9, transform=ax.transAxes,
                               facecolor=SURFACE, edgecolor=GRID))
        ax.text(x0 + 0.5 / len(tiles), 0.52, str(v), transform=ax.transAxes, ha="center", fontsize=20, fontweight="bold")
        ax.text(x0 + 0.5 / len(tiles), 0.14, t, transform=ax.transAxes, ha="center", fontsize=10, color=INK_2)
    ax = fig.add_subplot(gs[1, 0])
    labels = ["CER raw", "CER case/alnum", "exact raw", "exact case/alnum"]
    a = [m["all"]["A_cer_raw"], m["all"]["A_cer_casefold_alnum"], m["all"]["A_exact_raw"], m["all"]["A_exact_casefold_alnum"]]
    b = [m["all"]["B_cer_raw"], m["all"]["B_cer_casefold_alnum"], m["all"]["B_exact_raw"], m["all"]["B_exact_casefold_alnum"]]
    x = np.arange(4)
    ax.bar(x - 0.2, a, 0.4, color=base.ORANGE, label="A pretrained CRNN (0-9a-z)")
    ax.bar(x + 0.2, b, 0.4, color=base.BLUE, label="B Topo+Geo+Direction")
    for xi, va, vb in zip(x, a, b):
        ax.text(xi - 0.2, min(va, 1.4) + 0.02, f"{va:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, min(vb, 1.4) + 0.02, f"{vb:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("Crops with ground truth (273): A vs B (scores never combined)", fontsize=11, loc="left")
    ax = fig.add_subplot(gs[1, 1])
    fc = m["failure_counts"]
    keys = sorted(fc, key=lambda k: -fc[k])
    ax.barh(keys[::-1], [fc[k] for k in keys[::-1]], color=[base.STATUS["HIGH_CONFIDENCE"] if k == "CORRECT" else
                                                            base.STATUS["AMBIGUOUS"] for k in keys[::-1]])
    for yv, k in enumerate(keys[::-1]):
        ax.text(fc[k] + 1, yv, str(fc[k]), va="center")
    ax.set_title("B failure attribution (transcribed crops)", fontsize=11, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    ab = m["A_vs_B_casefold"]
    lines = ["A vs B on case-folded alphanumerics:", f"both correct  {ab['both_correct']}",
             f"only A correct {ab['only_A_correct']}", f"only B correct {ab['only_B_correct']}",
             f"both wrong    {ab['both_wrong']}", "",
             f"touching crops: {m['category_touching']['images']} imgs, B exact {m['category_touching']['B_exact']:.3f}",
             f"decimal-point crops: {m['category_decimal_point']['images']} imgs, B exact "
             f"{m['category_decimal_point']['B_exact']:.3f}",
             f"scene regions w/o B segments: {sc['regions_without_B_segments']}",
             f"multi-line scenes (order uncertain): {sc['multi_line_scenes']}"]
    for k, t in enumerate(lines):
        ax.text(0, 0.95 - k * 0.09, t, fontsize=11, family="monospace", transform=ax.transAxes)
    scenes = [i for i in r.images if i.kind == INFORMAL_SCENE][:6]
    sub = gs[2, :].subgridspec(2, 3, hspace=0.2, wspace=0.05)
    for k, i in enumerate(scenes):
        ax = fig.add_subplot(sub[k // 3, k % 3])
        _overlay(ax, i, load_image(PROJECT_ROOT / i.path).image)
        ax.set_title(f"{Path(i.path).stem} · {len(i.regions)} regions", fontsize=9)
    ax = fig.add_subplot(gs[3, :])
    ax.set_axis_off()
    ax.set_title("LIMITATIONS", loc="left", fontsize=12, fontweight="bold")
    for k, t in enumerate(r.report["limitations"]):
        ax.text(0, 0.9 - k * 0.14, "• " + base.safe_text(t)[:210], fontsize=10.5, color=INK_2, transform=ax.transAxes)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.955, bottom=0.03)
    return _save(fig, path)


def build_report(r: TDResult, cfg: AppConfig) -> Dict[str, object]:
    m = evaluate(r)
    t = cfg.text_detection
    return {
        "experiment": "REAL-IMAGE TEXT DETECTION INTEGRATION",
        "detector": {"selected": "PaddleOCR PP-OCRv3 DBNet (English), ONNX from the OpenCV model zoo",
                     "runtime": "cv2.dnn_TextDetectionModel_DB (opencv-python 4.10.0.84, already pinned)",
                     "dependencies": "no new Python packages; weight files only in data/models/",
                     "cpu_inference": "yes (OpenCV DNN default CPU backend)",
                     "parameters": {"binary_threshold": t.det_binary_threshold, "polygon_threshold": t.det_polygon_threshold,
                                    "unclip_ratio": t.det_unclip_ratio, "max_side": t.det_max_side},
                     "rejected": "PaddleOCR+PaddlePaddle: pip dry-run would add 53 packages incl. paddlepaddle 3.3.1, "
                                 "paddlex and opencv-contrib-python next to the pinned opencv-python (cv2 module clash)"},
        "ocr_comparison_model": {"selected": "CRNN-EN 2021sep (OpenCV zoo), CTC greedy, charset 0-9a-z",
                                 "note": "cannot output upper case or symbols; compared on case-folded alphanumerics"},
        "recognition_B": "existing segmentation + topology + geometry + 13 fixed direction features, 1/3 per group; "
                         "leave-one-group-out for crops, all 989 validated characters for scenes",
        "metrics": m, "info": r.info,
        "leakage": {"train_test_group_overlap": 0, "ground_truth_used_in_inference": False,
                    "scenes_in_training": False},
        "limitations": [
            "No detection ground truth: detection recall / precision are NOT EVALUABLE; coverage is a proxy against "
            "validated segmentation boxes (COUNT_MATCH crops only).",
            "Scene recognition is NOT EVALUABLE (scenes are not transcribed); scene results are qualitative.",
            "Synthetic-looking dataset crops and scenes are not real shipyard photographs; no field validation.",
            "B classifier figures from the direction experiment are exploratory (features designed on this data).",
            "Posteriors and detector scores are uncalibrated and not used for automatic acceptance.",
            "CRNN-EN covers only 0-9a-z; its raw CER is penalised for upper case and symbols by design of the model.",
        ],
    }


def terminal_summary(r: TDResult) -> str:
    rep = r.report
    m = rep["metrics"]
    sc = m["scenes"]
    d = rep["detector"]
    a = m["all"]
    o = ["=" * 60, "PAC TEXT DETECTION INTEGRATION (REAL DATA)", "=" * 60, "", "[TEXT DETECTOR]",
         f"Selected model = {d['selected']}", f"Dependencies = {d['dependencies']} ({d['runtime']})",
         f"CPU inference = {d['cpu_inference']}", f"Rejected = {d['rejected']}", "", "[SCENE IMAGES]",
         f"Total scenes = {sc['total']}", f"Scenes with detected text = {sc['with_regions']}",
         f"Total detected regions = {sc['regions']}",
         f"False positives observed = NOT EVALUABLE without ground truth; proxy: {sc['regions_without_B_segments']} "
         f"regions gave no B segments, {sc['regions_with_empty_A_text']} gave empty A text",
         f"Multi-line scenes (reading order uncertain) = {sc['multi_line_scenes']}", "",
         f"[RECOGNITION]  (crops with ground truth: {m['transcribed_images']})",
         f"Topology-based recognition (B) = CER {a['B_cer_raw']:.4f}, exact {a['B_exact_raw']:.4f} "
         f"(case/alnum: CER {a['B_cer_casefold_alnum']:.4f}, exact {a['B_exact_casefold_alnum']:.4f})",
         f"OCR recognition (A, CRNN-EN) = raw CER {a['A_cer_raw']:.4f}, exact {a['A_exact_raw']:.4f} "
         f"(case/alnum: CER {a['A_cer_casefold_alnum']:.4f}, exact {a['A_exact_casefold_alnum']:.4f})",
         f"Formal B CER {m[FORMAL]['B_cer_raw']:.4f} / Informal B CER {m[INFORMAL]['B_cer_raw']:.4f}",
         f"A vs B (case/alnum) = {m['A_vs_B_casefold']}",
         f"Proxy detection coverage = {m['proxy_detection_coverage']['mean']:.4f} "
         f"({m['proxy_detection_coverage']['images_fully_covered']}/{m['proxy_detection_coverage']['images']} "
         "COUNT_MATCH crops fully covered; proxy, not recall)", "", "[FAILURES]  (B, transcribed crops)"]
    fc = m["failure_counts"]
    o += [f"Detection = {fc.get('DETECTION_MISS', 0)} miss + {fc.get('DETECTION_FALSE_POSITIVE', 0)} false-positive",
          f"Segmentation = {fc.get('SEGMENTATION', 0)}", f"Classification = {fc.get('CLASSIFICATION', 0)}",
          f"Correct = {fc.get('CORRECT', 0)}",
          f"Touching crops = {m['category_touching']}", f"Decimal-point crops = {m['category_decimal_point']}",
          "", "[LEAKAGE]", "Training/test group overlap = 0 (leave-one-group-out; scenes never in training)",
          "Ground truth used in inference = False", "", "[LIMITATIONS]"]
    o += [f"- {t}" for t in rep["limitations"]] + ["=" * 60]
    return "\n".join(o)


def render_and_save(r: TDResult, cfg: AppConfig) -> None:
    r.report = build_report(r, cfg)
    write_tables(r)
    for name, fn in (("scene_visualizations", lambda: scene_visualizations(r)),
                     ("final_dashboard", lambda: dashboard(r, r.out_dir / "final_dashboard.png"))):
        try:
            fn()
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] {name}: {type(exc).__name__}: {exc}")
    (r.out_dir / "final_report.json").write_text(json.dumps(r.report, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    (r.out_dir / "final_report.txt").write_text(terminal_summary(r), encoding="utf-8")
