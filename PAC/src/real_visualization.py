"""Figures for the REAL SYMBOLS DATA EXPERIMENT (results/real_symbols/*.png).

Only values computed by src/real_experiment.py are drawn. Every figure carries the
real-data banner so it cannot be confused with the synthetic demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from . import visualization as base  # noqa: E402  (palette, fonts, safe_text)
from .dataset import FORMAL, INFORMAL, INFORMAL_SCENE, segment_characters  # noqa: E402
from .image_io import load_image  # noqa: E402
from .real_experiment import (MODEL_D, MODELS, POSTERIOR_LABEL, WELDING, ExperimentResult,  # noqa: E402
                              confusion, most_confused, rel, welding_roi)

SURFACE, PAGE, INK, INK_2, MUTED, GRID = base.SURFACE, base.PAGE, base.INK, base.INK_2, base.MUTED, base.GRID
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = base.BLUE, base.ORANGE, base.AQUA, base.YELLOW, base.VIOLET, base.RED
MODEL_LABELS = {"A_geometry_only": "A  Geometry only", "B_topology_only": "B  Topology only",
                "C_topology_geometry": "C  Topology + Geometry", MODEL_D: "D  Topo + Geo + Restoration"}
MODEL_COLORS = {"A_geometry_only": AQUA, "B_topology_only": ORANGE, "C_topology_geometry": BLUE, MODEL_D: VIOLET}
BANNER = "REAL DATA EXPERIMENT  ·  Source: data/input/Symbols"
DPI = 120


def _banner(fig: plt.Figure, title: str) -> None:
    fig.text(0.01, 0.985, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.99, 0.985, BANNER, fontsize=10, color=RED, fontweight="bold", va="top", ha="right")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _rgb(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else img[..., ::-1]


def _img(ax: plt.Axes, img: np.ndarray, title: str = "", cmap: Optional[str] = "gray") -> None:
    ax.imshow(img, cmap=cmap if img.ndim == 2 else None, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(base.safe_text(title), fontsize=9)


def _grid_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, lw=0.6, color=GRID)
    ax.set_axisbelow(True)


# ====================================================================== figures
def dataset_overview(r: ExperimentResult, path: Path) -> Path:
    a = r.audit
    fig = plt.figure(figsize=(16, 8.5))
    _banner(fig, "Dataset overview (filesystem audit)")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(gs[0, :2])
    names = ["Formal", "Informal crops", "Informal scenes", "Welding"]
    vals = [a["formal_images"], a["informal_crop_images"], a["informal_scene_images"], a["welding_images"]]
    ax.barh(names[::-1], vals[::-1], color=BLUE, edgecolor=SURFACE, linewidth=2, height=0.6)
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 2, y, str(v), va="center", fontsize=11)
    ax.set_title(f"Images by type (total {a['total_images']}, corrupted {a['invalid_or_corrupted_images']})", fontsize=11)
    _grid_axis(ax, "x")
    ax2 = fig.add_subplot(gs[0, 2:])
    per = a["images_per_welding_class"]
    ax2.bar(range(len(per)), list(per.values()), color=ORANGE, edgecolor=SURFACE, linewidth=2)
    ax2.set_xticks(range(len(per)))
    ax2.set_xticklabels(list(per.keys()), rotation=60, ha="right", fontsize=8)
    ax2.set_title(f"Welding images per class ({a['welding_classes']} classes, label = filename)", fontsize=11)
    _grid_axis(ax2)
    picks = []
    for t in (FORMAL, INFORMAL, INFORMAL_SCENE, WELDING):
        rec = next((x for x in r.records if x.dataset_type == t and x.status != "FAILED"), None)
        if rec:
            picks.append((t, rec))
    for i, (t, rec) in enumerate(picks):
        axi = fig.add_subplot(gs[1, i])
        _img(axi, _rgb(load_image(rec.path).image), f"{t}\n{rec.path.name}")
    return _save(fig, path)


def welding_roi_preview(r: ExperimentResult, path: Path, cfg) -> Path:
    samples = sorted(r.welding, key=lambda s: rel(s.record.path))
    chosen = samples[::max(1, len(samples) // 6)][:6]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 2.4 * len(chosen) + 1))
    _banner(fig, "Welding ROI: one fixed rule for all images (dark ink only + caption band removed)")
    for row, s in zip(np.atleast_2d(axes), chosen):
        img = load_image(s.record.path).image
        mask, info = welding_roi(img, cfg)
        _img(row[0], _rgb(img), f"Original: {s.record.path.name}")
        band = info["caption_band_px"]
        row[0].add_patch(Rectangle((0, 0), img.shape[1], band, fill=True, color=RED, alpha=0.15))
        roi = np.where(mask, 0, 255).astype(np.uint8)
        _img(row[1], roi, f"Analysis ROI (caption band removed: {info['removed_components']} comps, "
                          f"band ink after = {info['roi_ink_pixels_in_caption_band']})")
        _img(row[2], np.where(s.mask_n, 0, 255).astype(np.uint8), "Binary mask (normalized canvas)")
    lk = r.report.get("label_leakage_status", {})
    fig.text(0.01, 0.005, f"LABEL_LEAKAGE_RISK = {str(lk.get('LABEL_LEAKAGE_RISK')).upper()}  ·  caption: "
             f"{lk.get('caption_text')}  ·  NEAR_DUPLICATE_RISK = TRUE (variants are warped copies)",
             fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, path)


def confusion_figure(r: ExperimentResult, path: Path, model: str = "C_topology_geometry") -> Path:
    m = confusion(r.cv["predictions"], r.classes, model)
    fig, ax = plt.subplots(figsize=(13, 11.5))
    _banner(fig, f"Welding confusion matrix — {MODEL_LABELS[model]} (54 out-of-fold predictions)")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
    ax.imshow(m, cmap=cmap, vmin=0, vmax=max(3, m.max()))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if m[i, j] >= 2 else INK, fontweight="bold")
    ax.set_xticks(range(len(r.classes)))
    ax.set_yticks(range(len(r.classes)))
    ax.set_xticklabels(r.classes, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(r.classes, fontsize=9)
    ax.set_xlabel("Predicted (Top-1)")
    ax.set_ylabel("Ground truth (filename label)")
    pairs = most_confused(m, r.classes, 3)
    ax.set_title("Most confused: " + ("; ".join(f"{p['pair']} ({p['count']})" for p in pairs) or "none"), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def model_comparison(r: ExperimentResult, path: Path) -> Path:
    met = r.metrics
    names = list(MODELS) + [MODEL_D]
    keys = [("top1_accuracy", "Top-1"), ("top3_accuracy", "Top-3"), ("macro_accuracy", "Macro")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.25})
    _banner(fig, "Welding Symbol supervised evaluation (3-fold leave-one-sample-per-class, n=54)")
    x = np.arange(len(keys))
    w = 0.2
    for i, n in enumerate(names):
        vals = [met[n][k] for k, _ in keys]
        ax.bar(x + (i - 1.5) * w, vals, w, color=MODEL_COLORS[n], edgecolor=SURFACE, linewidth=2, label=MODEL_LABELS[n])
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1.5) * w, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([k for _, k in keys], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("accuracy")
    ax.axhline(1 / max(1, len(r.classes)), color=MUTED, ls="--", lw=1)
    ax.text(2.45, 1 / max(1, len(r.classes)) + 0.01, "chance (1/18)", fontsize=8, color=MUTED, ha="right")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    _grid_axis(ax)
    ent = [met[n]["mean_entropy"] for n in names]
    ax2.barh([MODEL_LABELS[n] for n in names][::-1], ent[::-1], color=[MODEL_COLORS[n] for n in names][::-1],
             edgecolor=SURFACE, linewidth=2, height=0.6)
    for y, v in enumerate(ent[::-1]):
        ax2.text(v + 0.02, y, f"{v:.3f}", va="center", fontsize=9)
    ax2.set_title(f"Mean entropy (nats, over 18 classes)\n{POSTERIOR_LABEL}", fontsize=10)
    _grid_axis(ax2, "x")
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.05, right=0.97)
    return _save(fig, path)


def topology_contribution(r: ExperimentResult, path: Path) -> Path:
    met = r.metrics
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.8], "wspace": 0.3})
    _banner(fig, "Topology contribution (real welding data)")
    names = ["A_geometry_only", "B_topology_only", "C_topology_geometry"]
    vals = [met[n]["top1_accuracy"] for n in names]
    ax.bar([MODEL_LABELS[n].replace("  ", "\n", 1) for n in names], vals, color=[MODEL_COLORS[n] for n in names],
           edgecolor=SURFACE, linewidth=2, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    tc = met["topology_contribution"]
    ax.set_title(f"Topology Contribution = Acc(C) - Acc(A) = {tc:+.3f}", fontsize=11,
                 color=INK if tc >= 0 else RED)
    ax.set_ylim(0, 1.05)
    _grid_axis(ax)
    pa, pc = met["A_geometry_only"]["per_class_accuracy"], met["C_topology_geometry"]["per_class_accuracy"]
    y = np.arange(len(r.classes))
    for c_i, c in enumerate(r.classes):
        ax2.plot([pa[c], pc[c]], [c_i, c_i], color=GRID, lw=2, zorder=1)
    ax2.scatter([pa[c] for c in r.classes], y, s=70, color=AQUA, edgecolors=SURFACE, linewidths=1.5, zorder=2,
                label="A Geometry only")
    ax2.scatter([pc[c] for c in r.classes], y, s=70, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                label="C Topology + Geometry")
    ax2.set_yticks(y)
    ax2.set_yticklabels(r.classes, fontsize=9)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_xlabel("per-class accuracy (3 test images per class)")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    _grid_axis(ax2, "x")
    return _save(fig, path)


def restoration_effect(r: ExperimentResult, path: Path) -> Path:
    met = r.metrics
    eff = met["restoration_effect_counts"]
    ops = met["restoration_operations"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"wspace": 0.35})
    _banner(fig, "Restoration effect (Model C before -> Model D after, no ground truth used in restoration)")
    before, after = met["C_topology_geometry"]["top1_accuracy"], met[MODEL_D]["top1_accuracy"]
    axes[0].bar(["Before (C)", "After (D)"], [before, after], color=[BLUE, VIOLET], edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate([before, after]):
        axes[0].text(i, v + 0.015, f"{v:.3f}", ha="center")
    rc = met["restoration_contribution"]
    axes[0].set_title(f"Top-1 accuracy · Restoration Contribution = {rc:+.3f}", fontsize=10,
                      color=INK if rc >= 0 else RED)
    axes[0].set_ylim(0, 1.05)
    _grid_axis(axes[0])
    labels = ["corrected", "degraded", "unchanged"]
    axes[1].bar(labels, [eff[k] for k in labels], color=[base.STATUS["HIGH_CONFIDENCE"], base.STATUS["AMBIGUOUS"], MUTED],
                edgecolor=SURFACE, linewidth=2)
    for i, k in enumerate(labels):
        axes[1].text(i, eff[k] + 0.6, str(eff[k]), ha="center")
    axes[1].set_title("Per-image outcome (54 test images)", fontsize=10)
    _grid_axis(axes[1])
    axes[2].barh(list(ops.keys()), list(ops.values()), color=ORANGE, edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(ops.values()):
        axes[2].text(v + 0.3, y, str(v), va="center")
    axes[2].set_title("Selected restoration operation (min J)", fontsize=10)
    _grid_axis(axes[2], "x")
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.05, right=0.98)
    return _save(fig, path)


FVI_FEATURES = [("beta_0", "β0"), ("beta_1", "β1"), ("euler_characteristic", "Euler"),
                ("skeleton_length_norm", "Skeleton length (norm)"), ("endpoints", "Endpoints"),
                ("branch_points", "Branch points"), ("aspect_ratio", "Aspect ratio"), ("area_ratio", "Area ratio"),
                ("circularity", "Circularity"), ("solidity", "Solidity")]


def formal_vs_informal(r: ExperimentResult, path: Path) -> Tuple[Path, Dict[str, Dict[str, float]]]:
    segs = r.text["segments"]
    fs = [s for s in segs if s["style"] == FORMAL]
    ins = [s for s in segs if s["style"] == INFORMAL]
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.5))
    _banner(fig, "Formal vs Informal — STYLE / STRUCTURAL FEATURE ANALYSIS (not recognition accuracy)")
    stats: Dict[str, Dict[str, float]] = {}
    for ax, (key, label) in zip(axes.ravel(), FVI_FEATURES):
        a = [float(s[key]) for s in fs]
        b = [float(s[key]) for s in ins]
        if not a or not b:
            ax.set_axis_off()
            continue
        bp = ax.boxplot([a, b], widths=0.55, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], (BLUE, ORANGE)):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            patch.set_edgecolor(col)
        for med in bp["medians"]:
            med.set_color(INK)
        try:
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        stats[key] = {"formal_median": float(np.median(a)), "informal_median": float(np.median(b)),
                      "formal_mean": float(np.mean(a)), "informal_mean": float(np.mean(b)), "mannwhitney_p": p}
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Formal\nn={len(a)}", f"Informal\nn={len(b)}"], fontsize=8)
        ax.set_title(f"{label}\nmedian {np.median(a):.2f} vs {np.median(b):.2f} · p={p:.2g}", fontsize=9)
        _grid_axis(ax)
    fig.text(0.01, 0.01, "Character segments (64 px canvas). Mann-Whitney U p-values are descriptive: segments from the "
             "same image/group are not independent.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(fig, path), stats


def _welding_case(r: ExperimentResult, pred: Optional[Dict[str, object]], title: str, path: Path) -> Path:
    fig = plt.figure(figsize=(15, 5.2))
    _banner(fig, title)
    if pred is None:
        fig.text(0.5, 0.5, "No such case in the 54 out-of-fold predictions", ha="center", fontsize=14)
        return _save(fig, path)
    s = next(x for x in r.welding if rel(x.record.path) == pred["image_path"])
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.6, 0.8, 1.3], wspace=0.25)
    img = load_image(s.record.path).image
    ax0 = fig.add_subplot(gs[0])
    _img(ax0, _rgb(img), s.record.path.name)
    band = int(0.22 * img.shape[0]) if s.roi_info.get("caption_band_px") is None else int(s.roi_info["caption_band_px"])
    ax0.add_patch(Rectangle((0, 0), img.shape[1], band, color=RED, alpha=0.15))
    ax0.text(img.shape[1] - 5, band - 6, "caption band: not used in analysis", ha="right", fontsize=8, color=RED)
    _img(fig.add_subplot(gs[1]), np.where(s.mask_n, 0, 255).astype(np.uint8), "ROI mask (analysis input)")
    ax = fig.add_subplot(gs[2])
    top3 = str(pred["C_topology_geometry_top3"]).split("|")
    ax.set_axis_off()
    lines = [
        f"Ground truth: {pred['ground_truth']}   (fold {pred['fold']})",
        f"Model C Top-1: {pred['C_topology_geometry_top1']}  "
        f"({'correct' if pred['C_topology_geometry_correct'] else 'WRONG'})",
        f"Top-1 posterior: {float(pred['C_topology_geometry_top1_posterior']):.3f}",
        f"Top-3: {', '.join(top3)}",
        f"Entropy: {float(pred['C_topology_geometry_entropy']):.3f} nats",
        f"Model D (restoration {pred['restoration_operation']}): {pred[MODEL_D + '_top1']}",
        f"J={float(pred['J']):.3f}  ({pred['restoration_effect']})",
        f"{POSTERIOR_LABEL}",
    ]
    for i, t in enumerate(lines):
        ax.text(0.0, 0.92 - i * 0.12, base.safe_text(t), fontsize=11, transform=ax.transAxes,
                fontweight="bold" if i < 2 else "normal", color=MUTED if i == len(lines) - 1 else INK)
    return _save(fig, path)


def _text_case(r: ExperimentResult, row: Optional[Dict[str, object]], title: str, path: Path, cfg) -> Path:
    fig = plt.figure(figsize=(15, 6))
    _banner(fig, title)
    if row is None:
        fig.text(0.5, 0.5, "No image available", ha="center", fontsize=14)
        return _save(fig, path)
    p = Path(cfg_root(cfg)) / row["image_path"]
    img = load_image(p).image
    seg = segment_characters(img, cfg.preprocessing, cfg.dataset)
    gs = GridSpec(2, max(6, seg.count), figure=fig, height_ratios=[2.2, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    _img(ax, _rgb(img), f"{Path(row['image_path']).name} · segments={row['segment_count']} · "
                        f"ground_truth={row['ground_truth']} · segmentation_valid={row['segmentation_valid']}")
    for k, (x0, y0, x1, y1) in enumerate(seg.boxes):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=base.STATUS["HIGH_CONFIDENCE"], lw=2))
        ax.text(x0, y0 - 3, str(k + 1), color=base.STATUS["HIGH_CONFIDENCE"], fontsize=9)
    for k, cm in enumerate(seg.char_masks[: max(6, seg.count)]):
        _img(fig.add_subplot(gs[1, k]), np.where(cm, 0, 255).astype(np.uint8), f"seg {k + 1}")
    reason = row.get("failure_reason") or "-"
    fig.text(0.01, 0.01, base.safe_text(f"possible touching = {row.get('possible_touching_segments')}, "
                                         f"punctuation-like = {row.get('small_segments_punctuation_like')}, "
                                         f"note: {reason}"), fontsize=9, color=INK_2)
    return _save(fig, path)


def cfg_root(cfg) -> Path:
    from .config import PROJECT_ROOT

    return PROJECT_ROOT


def scene_summary(r: ExperimentResult, path: Path) -> Path:
    items = sorted(r.scenes["overlays"].items())
    cols = 6
    rows = max(1, int(np.ceil(len(items) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2.9 * rows + 1))
    _banner(fig, f"Scene analysis: {len(items)} scenes, {len(r.scenes['candidates'])} candidate regions · "
                 "DETECTION ACCURACY = NOT EVALUABLE (no scene ground truth)")
    counts = {}
    for c in r.scenes["candidates"]:
        counts[c["image_path"]] = counts.get(c["image_path"], 0) + 1
    for ax, (k, vis) in zip(np.atleast_1d(axes).ravel(), items):
        _img(ax, _rgb(vis), f"{Path(k).stem} · {counts.get(k, 0)} candidates")
    for ax in np.atleast_1d(axes).ravel()[len(items):]:
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


# ====================================================================== case selection
def select_cases(r: ExperimentResult) -> Dict[str, Optional[Dict[str, object]]]:
    """Automatic representative cases (deterministic)."""
    P = r.cv["predictions"]
    key = "C_topology_geometry"
    correct = [p for p in P if p[f"{key}_correct"]]
    wrong = [p for p in P if not p[f"{key}_correct"]]
    imgs = [i for i in r.text["images"] if "segment_count" in i]
    def clean(i):
        return i["possible_touching_segments"] == 0 and i["small_segments_punctuation_like"] == 0 and i["segment_count"] > 0
    def difficulty(i):
        return i["possible_touching_segments"] + i["small_segments_punctuation_like"] + (10 if i["segment_count"] == 0 else 0)
    formal = sorted([i for i in imgs if i["style"] == FORMAL and clean(i)],
                    key=lambda i: (-i["segment_count"], i["image_path"]))
    inf_ok = sorted([i for i in imgs if i["style"] == INFORMAL and clean(i)],
                    key=lambda i: (-i["segment_count"], i["image_path"]))
    inf_hard = sorted([i for i in imgs if i["style"] == INFORMAL], key=lambda i: (-difficulty(i), i["image_path"]))
    counts: Dict[str, int] = {}
    for c in r.scenes["candidates"]:
        counts[c["image_path"]] = counts.get(c["image_path"], 0) + 1
    scenes = sorted(r.scenes["overlays"], key=lambda k: (counts.get(k, 0), k))
    return {
        "welding_correct": max(correct, key=lambda p: p[f"{key}_top1_posterior"]) if correct else None,
        "welding_failure": max(wrong, key=lambda p: p[f"{key}_top1_posterior"]) if wrong else None,
        "welding_ambiguous": max(P, key=lambda p: p[f"{key}_entropy"]) if P else None,
        "formal": formal[0] if formal else (next((i for i in imgs if i["style"] == FORMAL), None)),
        "informal_success": inf_ok[0] if inf_ok else None,
        "informal_difficult": inf_hard[0] if inf_hard else None,
        "scene": scenes[len(scenes) // 2] if scenes else None,
    }


# ====================================================================== dashboard
def _section(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}. {title}", loc="left", fontsize=11, fontweight="bold")


def final_dashboard(r: ExperimentResult, path: Path, cases: Dict, fvi: Dict[str, Dict[str, float]],
                    cfg_caption_ratio: float = 0.22) -> Path:
    rep = r.report
    met = r.metrics
    fig = plt.figure(figsize=(24, 30))
    gs = GridSpec(7, 6, figure=fig, height_ratios=[0.42, 1.0, 1.25, 1.0, 1.0, 1.05, 0.8], hspace=0.55, wspace=0.45)
    fig.text(0.012, 0.995, "PAC MISSION 3  ·  REAL SYMBOLS DATA EXPERIMENT", fontsize=26, fontweight="bold", va="top")
    fig.text(0.012, 0.982, "Source: data/input/Symbols   ·   no synthetic data, demo image or test.png used   ·   "
             f"{POSTERIOR_LABEL}s", fontsize=13, color=RED, fontweight="bold", va="top")

    # A. DATASET (tiles)
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    _section(ax, "A", "DATASET (filesystem audit)")
    tiles = [("Total Images", rep["total_images"]), ("Formal Images", rep["formal_images"]),
             ("Informal Crops", rep["informal_images"]), ("Informal Scenes", rep["scene_images"]),
             ("Welding Images", rep["welding_images"]), ("Welding Classes", rep["welding_classes"]),
             ("Processed OK", rep["processed_images"]), ("Failed", rep["failed_images"])]
    for i, (k, v) in enumerate(tiles):
        x = i / len(tiles)
        ax.add_patch(Rectangle((x + 0.005, 0.0), 1 / len(tiles) - 0.01, 0.9, transform=ax.transAxes,
                               facecolor=SURFACE, edgecolor=GRID, lw=1.2))
        ax.text(x + 0.5 / len(tiles), 0.55, str(v), transform=ax.transAxes, ha="center", fontsize=24, fontweight="bold")
        ax.text(x + 0.5 / len(tiles), 0.15, k, transform=ax.transAxes, ha="center", fontsize=11, color=INK_2)

    # B. WELDING SUPERVISED EVALUATION
    ax = fig.add_subplot(gs[1, :3])
    _section(ax, "B", "WELDING SYMBOL SUPERVISED EVALUATION (3-fold CV, n=54)")
    names = list(MODELS) + [MODEL_D]
    vals = [met[n]["top1_accuracy"] for n in names]
    short = {"A_geometry_only": "A Geometry", "B_topology_only": "B Topology", "C_topology_geometry": "C Topo+Geo",
             MODEL_D: "D Topo+Geo+Rest."}
    ax.barh([short[n] for n in names][::-1], vals[::-1], color=[MODEL_COLORS[n] for n in names][::-1],
            edgecolor=SURFACE, linewidth=2, height=0.6)
    ax.tick_params(axis="y", labelsize=11)
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=12)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(f"Top-1 accuracy · Model C: Top-3 {rep['top3_accuracy']:.3f}, Macro {rep['macro_accuracy']:.3f}, "
                  f"mean entropy {rep['mean_entropy']:.3f}")
    _grid_axis(ax, "x")

    # C. TOPOLOGY CONTRIBUTION
    ax = fig.add_subplot(gs[1, 3:])
    ax.set_axis_off()
    _section(ax, "C", "TOPOLOGY CONTRIBUTION (actual accuracy differences)")
    tc, rc = rep["topology_contribution"], rep["restoration_contribution"]
    for i, (k, v, sub) in enumerate([
            ("Topology Contribution", tc, "Acc(C Topo+Geo) − Acc(A Geo only)"),
            ("Restoration Contribution", rc, "Acc(D +Restoration) − Acc(C Topo+Geo)"),
            ("Topology only vs Geometry only", met["B_topology_only"]["top1_accuracy"] - met["A_geometry_only"]["top1_accuracy"],
             "Acc(B) − Acc(A)")]):
        y = 0.78 - i * 0.32
        ax.text(0.02, y, k, fontsize=14, transform=ax.transAxes, color=INK_2)
        ax.text(0.98, y, f"{v:+.3f}", fontsize=26, fontweight="bold", transform=ax.transAxes, ha="right",
                color=INK if v >= 0 else RED)
        ax.text(0.02, y - 0.1, sub, fontsize=10, transform=ax.transAxes, color=MUTED)

    # D. CONFUSION MATRIX
    ax = fig.add_subplot(gs[2, :3])
    m = confusion(r.cv["predictions"], r.classes, "C_topology_geometry")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
    ax.imshow(m, cmap=cmap, vmin=0, vmax=max(3, m.max()))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] >= 2 else INK)
    ax.set_xticks(range(len(r.classes)))
    ax.set_yticks(range(len(r.classes)))
    ax.set_xticklabels(r.classes, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(r.classes, fontsize=7)
    _section(ax, "D", "CONFUSION MATRIX (Model C, rows = ground truth)")

    # E. RESTORATION EFFECT
    ax = fig.add_subplot(gs[2, 3:])
    eff = rep["restoration_effect"]
    labels = ["corrected", "degraded", "unchanged"]
    ax.bar(labels, [eff[k] for k in labels], color=[base.STATUS["HIGH_CONFIDENCE"], base.STATUS["AMBIGUOUS"], MUTED],
           edgecolor=SURFACE, linewidth=2, width=0.6)
    for i, k in enumerate(labels):
        ax.text(i, eff[k] + 0.5, str(eff[k]), ha="center", fontsize=13, fontweight="bold")
    _section(ax, "E", f"RESTORATION EFFECT · before {met['C_topology_geometry']['top1_accuracy']:.3f} → "
                      f"after {met[MODEL_D]['top1_accuracy']:.3f}")
    ax.set_ylabel("test images")
    _grid_axis(ax)

    # F. FORMAL VS INFORMAL
    keys = [k for k, _ in FVI_FEATURES if k in fvi]
    ax = fig.add_subplot(gs[3, :4])
    x = np.arange(len(keys))
    f_med = [fvi[k]["formal_mean"] for k in keys]
    i_med = [fvi[k]["informal_mean"] for k in keys]
    scale = [max(abs(a), abs(b), 1e-9) for a, b in zip(f_med, i_med)]
    ax.bar(x - 0.2, [a / s for a, s in zip(f_med, scale)], 0.4, color=BLUE, edgecolor=SURFACE, linewidth=2,
           label="Formal (mean)")
    ax.bar(x + 0.2, [b / s for b, s in zip(i_med, scale)], 0.4, color=ORANGE, edgecolor=SURFACE, linewidth=2,
           label="Informal (mean)")
    for xi, k in zip(x, keys):
        ax.text(xi, 1.06, f"{fvi[k]['formal_mean']:.2f}/{fvi[k]['informal_mean']:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([dict(FVI_FEATURES)[k] for k in keys], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("mean / max(|formal|,|informal|)")
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1.06), ncol=2)
    _section(ax, "F", "FORMAL VS INFORMAL — structural feature analysis (segments; not recognition accuracy)")
    _grid_axis(ax)

    # G. OOD PROBE
    ax = fig.add_subplot(gs[3, 4:])
    ax.set_axis_off()
    _section(ax, "G", "WELDING-TRAINED OOD PROBE (not character classification)")
    ood = r.text["ood"]
    rows = []
    for style in (FORMAL, INFORMAL):
        sub = [o for o in ood if o["style"] == style]
        n_ood = sum(o["classification"] == "UNKNOWN_OOD" for o in sub)
        rows.append((style, len(sub), len(sub) - n_ood, n_ood,
                     float(np.mean([o["entropy"] for o in sub])) if sub else float("nan")))
    ax.text(0.0, 0.9, f"threshold = {r.ood_threshold:.3f} ({rep['ood_threshold_rule'][:46]}…)", fontsize=9,
            transform=ax.transAxes, color=INK_2)
    hdr = ["style", "segments", "in-dist-like", "UNKNOWN_OOD", "mean entropy"]
    for j, h in enumerate(hdr):
        ax.text(0.0 + j * 0.2, 0.72, h, fontsize=10, transform=ax.transAxes, color=INK_2, fontweight="bold")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            txt = f"{v:.3f}" if isinstance(v, float) else str(v)
            ax.text(0.0 + j * 0.2, 0.55 - i * 0.17, txt, fontsize=13, transform=ax.transAxes,
                    fontweight="bold" if j == 3 else "normal")
    od = rep.get("ood_distance_summary", {})
    if od.get("FORMAL") and od.get("INFORMAL"):
        ax.text(0.0, 0.2, f"min text distance {min(od['FORMAL']['combined']['min'], od['INFORMAL']['combined']['min']):.2f}"
                f" vs welding out-of-fold max {od['welding_out_of_fold']['max']:.2f}", fontsize=10,
                transform=ax.transAxes)
    ax.text(0.0, 0.08, "Near-zero entropy on OOD text: the closed-set posterior is overconfident outside the\n"
            "training distribution — distance, not posterior, flags UNKNOWN. Distance = 0.5·mean|z| topo + 0.5·mean|z| geo.",
            fontsize=8.5, transform=ax.transAxes, color=MUTED)

    # H. REPRESENTATIVE RESULTS
    case_specs = [("welding_correct", "Correct welding"), ("welding_failure", "Failed welding"),
                  ("welding_ambiguous", "Ambiguous welding (max entropy)")]
    for j, (k, title) in enumerate(case_specs):
        ax = fig.add_subplot(gs[4, 2 * j:2 * j + 2])
        p = cases.get(k)
        if p is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{title}: none", ha="center", transform=ax.transAxes)
            continue
        img = load_image(cfg_root(None) / p["image_path"]).image
        _img(ax, _rgb(img), f"{title}\nGT {p['ground_truth']} → pred {p['C_topology_geometry_top1']} "
                            f"(p={float(p['C_topology_geometry_top1_posterior']):.2f}, "
                            f"H={float(p['C_topology_geometry_entropy']):.2f})")
        band = int(cfg_caption_ratio * img.shape[0])
        ax.add_patch(Rectangle((0, 0), img.shape[1], band, color=RED, alpha=0.15))
        ax.text(img.shape[1] - 5, band - 6, "caption band: not used in analysis", ha="right", fontsize=8, color=RED)
        if j == 0:
            ax.text(0.0, 1.32, "H. REPRESENTATIVE RESULTS", transform=ax.transAxes, fontsize=11, fontweight="bold")
    for j, (k, title) in enumerate([("formal", "Formal segmentation"), ("informal_success", "Informal (successful)"),
                                    ("scene", "Scene candidates")]):
        ax = fig.add_subplot(gs[5, 2 * j:2 * j + 2])
        c = cases.get(k)
        if c is None:
            ax.set_axis_off()
            continue
        if k == "scene":
            _img(ax, _rgb(r.scenes["overlays"][c]), f"{title}: {Path(c).name}")
            continue
        img = load_image(cfg_root(None) / c["image_path"]).image
        _img(ax, _rgb(img), f"{title}: {Path(c['image_path']).name} · {c['segment_count']} segments · "
                            f"GT {c['ground_truth']}")
    # I. LIMITATIONS
    ax = fig.add_subplot(gs[6, :])
    ax.set_axis_off()
    _section(ax, "I", "LIMITATIONS")
    for i, t in enumerate(rep["limitations"]):
        ax.text(0.0, 0.88 - i * 0.12, "• " + base.safe_text(t), fontsize=10.5, transform=ax.transAxes, color=INK_2)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.955, bottom=0.02)
    return _save(fig, path)


def render_all(r: ExperimentResult, cfg) -> Dict[str, Path]:
    """Render every REAL-DATA figure; one failing figure does not stop the others."""
    out = r.out_dir
    files: Dict[str, Path] = {}
    cases = select_cases(r)
    fvi: Dict[str, Dict[str, float]] = {}
    jobs = [
        ("dataset_overview", lambda: dataset_overview(r, out / "dataset_overview.png")),
        ("welding_roi_preview", lambda: welding_roi_preview(r, out / "welding_roi_preview.png", cfg)),
        ("welding_confusion_matrix", lambda: confusion_figure(r, out / "welding_confusion_matrix.png")),
        ("welding_model_comparison", lambda: model_comparison(r, out / "welding_model_comparison.png")),
        ("topology_contribution", lambda: topology_contribution(r, out / "topology_contribution.png")),
        ("restoration_effect", lambda: restoration_effect(r, out / "restoration_effect.png")),
        ("welding_correct_case", lambda: _welding_case(r, cases["welding_correct"],
                                                       "Welding: best correct case (Model C)", out / "welding_correct_case.png")),
        ("welding_failure_case", lambda: _welding_case(r, cases["welding_failure"],
                                                       "Welding: misclassified case (most confident error)",
                                                       out / "welding_failure_case.png")),
        ("welding_ambiguous_case", lambda: _welding_case(r, cases["welding_ambiguous"],
                                                         "Welding: highest-entropy case", out / "welding_ambiguous_case.png")),
    ]
    for name, job in jobs:
        try:
            files[name] = job()
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] figure {name} failed: {type(exc).__name__}: {exc}")
    if r.text["segments"]:
        try:
            files["formal_vs_informal"], fvi = formal_vs_informal(r, out / "formal_vs_informal.png")
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] figure formal_vs_informal failed: {exc}")
        for name, key, title in (("formal_example", "formal", "Formal Text: representative segmentation"),
                                 ("informal_example", "informal_success",
                                  "Informal Text: successful segmentation")):
            try:
                files[name] = _text_case(r, cases[key], title, out / f"{name}.png", cfg)
            except Exception as exc:
                plt.close("all")
                print(f"[WARN] figure {name} failed: {exc}")
        try:
            files["informal_difficult_example"] = _text_case(r, cases["informal_difficult"],
                                                             "Informal Text: difficult segmentation",
                                                             out / "informal_difficult_example.png", cfg)
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] figure informal_difficult_example failed: {exc}")
    if r.scenes["overlays"]:
        try:
            files["scene_analysis_summary"] = scene_summary(r, out / "scene_analysis_summary.png")
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] figure scene_analysis_summary failed: {exc}")
    try:
        files["final_dashboard"] = final_dashboard(r, out / "final_dashboard.png", cases, fvi,
                                                   cfg.real.caption_band_ratio)
    except Exception as exc:
        plt.close("all")
        print(f"[WARN] final dashboard failed: {type(exc).__name__}: {exc}")
    r.report["representative_cases"] = {k: (v if isinstance(v, str) else (v or {}).get("image_path"))
                                        for k, v in cases.items()}
    r.report["formal_vs_informal_feature_stats"] = fvi
    r.report["figures"] = {k: rel(v) for k, v in files.items()}
    return files
