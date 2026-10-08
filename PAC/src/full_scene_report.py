"""Outputs for FULL-SCENE RECOGNITION STABILIZATION."""

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
from .full_scene_stabilization import SceneImage, casefold_alnum, characteristics  # noqa: E402
from .image_io import load_image  # noqa: E402
from .real_experiment import write_csv  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, RED = base.SURFACE, base.INK, base.INK_2, base.MUTED, base.GRID, base.RED
GOOD, BAD = base.STATUS["HIGH_CONFIDENCE"], base.STATUS["AMBIGUOUS"]
DPI = 110


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_tables(images: List[SceneImage], ev: Dict, out: Path, cfg: AppConfig) -> None:
    labels_rows, rec_rows, seg_rows, cmp_rows = [], [], [], []
    for i in images:
        if i.kind == INFORMAL_SCENE:
            labels_rows.append({"image_path": i.path, "detected_regions": len(i.regions),
                                "region_polygons": " | ".join(" ".join(f"{x:.0f},{y:.0f}" for x, y in g["poly"])
                                                              for g in i.regions),
                                "labeled": i.label_flags.get("labeled"), "unreadable": i.label_flags.get("unreadable"),
                                "uncertain": i.label_flags.get("uncertain"),
                                "transcription": i.gt_raw if i.label_flags.get("labeled") else "",
                                "label_source": "data/scene_labels.csv (manual, --label-scenes)"})
        rec_rows.append({"image_path": i.path, "kind": i.kind, "ground_truth": i.gt_raw if i.gt else "UNKNOWN",
                         "regions": len(i.regions), "reading_order": i.reading_order,
                         "A_crnn": i.text_a, "B_tight": i.text_b["tight"], "B_padded": i.text_b["padded"],
                         "B_exact_tight": (i.text_b["tight"] == i.gt) if i.gt else "NOT_EVALUABLE",
                         "B_exact_padded": (i.text_b["padded"] == i.gt) if i.gt else "NOT_EVALUABLE",
                         "failure_tight": i.failure["tight"], "failure_padded": i.failure["padded"],
                         "B_char_polygons_original_coords_padded": " | ".join(
                             " ".join(f"{x:.0f},{y:.0f}" for x, y in p) for r in i.runs["padded"] for p in r.char_polys),
                         "B_candidates_padded": " || ".join(c for r in i.runs["padded"] for c in r.candidates)})
        if i.gt:
            for c in ("tight", "padded"):
                rs = i.runs[c]
                seg_rows.append({"image_path": i.path, "kind": i.kind, "condition": c, "gt": i.gt,
                                 "segments": sum(r.n_segments for r in rs), "expected": len(i.gt),
                                 "failure": i.failure[c],
                                 "border_touching_segments": sum(r.border_touching_segments for r in rs),
                                 "small_segments": sum(r.small_segments for r in rs),
                                 "excluded_line_components": sum(r.excluded_lines for r in rs),
                                 "other_regions_inside_crop": sum(r.other_regions_in_crop for r in rs),
                                 "crop_clipped_by_image_border": any(r.clipped for r in rs),
                                 "rotated_region": any(r.rotated for r in rs),
                                 "touching_flag": bool(i.marking_flags.get("possible_touching")),
                                 "decimal_point": "." in i.gt})
    for k, v in ev["_cats"].items():
        for i in v:
            cmp_rows.append({"category": k, "image_path": i.path, "kind": i.kind, "gt": i.gt_raw,
                             "A_crnn": i.text_a, "B_padded": i.text_b["padded"],
                             **characteristics(i, cfg.scene.slant_handwriting_deg)})
    write_csv(out / "scene_labeling_manifest.csv", labels_rows)
    write_csv(out / "scene_recognition_results.csv", rec_rows)
    write_csv(out / "segmentation_failure_analysis.csv", seg_rows)
    write_csv(out / "ocr_topology_comparison.csv", cmp_rows)
    write_csv(out / "topology_only_traits.csv", ev["category_traits"])


def _show_crop(ax, path: str, title: str, col=INK) -> None:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    ax.imshow(img[..., ::-1])
    ax.set_title(base.safe_text(title), fontsize=8.5, color=col)
    ax.set_xticks([])
    ax.set_yticks([])


def crop_comparison(images: List[SceneImage], out: Path, cfg: AppConfig) -> Path:
    """Deterministic examples (lowest path first): fixed by padding / still failing / scenes."""
    k = cfg.scene.examples_per_kind
    crops = sorted([i for i in images if i.gt and i.kind in (FORMAL, INFORMAL)], key=lambda i: i.path)
    fixed = [i for i in crops if i.failure["tight"] != "CORRECT" and i.failure["padded"] == "CORRECT"][:k]
    still = [i for i in crops if i.failure["padded"] != "CORRECT"][:k]
    scenes = [i for i in sorted(images, key=lambda i: i.path) if i.kind == INFORMAL_SCENE and i.regions][:2]
    rows = [("fixed by padding", i) for i in fixed] + [("still failing", i) for i in still] + \
           [("scene (no GT unless labelled)", i) for i in scenes]
    fig = plt.figure(figsize=(18, 2.9 * len(rows) + 1))
    fig.text(0.01, 0.995, "Crop stabilization: tight (0.5 x height) vs padded (2.0 x height) crops · right: padded "
             "character boxes mapped back to ORIGINAL coordinates (padding value is post-hoc)", fontsize=13,
             fontweight="bold", va="top")
    gs = GridSpec(len(rows), 4, figure=fig, width_ratios=[1, 1.3, 1.6, 1.4], hspace=0.6, wspace=0.15)
    for r, (kind, i) in enumerate(rows):
        t, p = i.runs["tight"][0], i.runs["padded"][0]
        _show_crop(fig.add_subplot(gs[r, 0]), t.crop_path, f"tight: '{i.text_b['tight']}' [{i.failure['tight'].split(' ')[0]}]",
                   GOOD if i.failure["tight"] == "CORRECT" else (BAD if i.gt else INK))
        _show_crop(fig.add_subplot(gs[r, 1]), p.crop_path, f"padded: '{i.text_b['padded']}' [{i.failure['padded'].split(' ')[0]}]",
                   GOOD if i.failure["padded"] == "CORRECT" else (BAD if i.gt else INK))
        ax = fig.add_subplot(gs[r, 2])
        img = load_image(PROJECT_ROOT / i.path).image
        ax.imshow(img if img.ndim == 2 else img[..., ::-1])
        for g in i.regions:
            ax.add_patch(Polygon(g["poly"], closed=True, fill=False, ec=base.BLUE, lw=1.2))
        for rr in i.runs["padded"]:
            for poly in rr.char_polys:
                ax.add_patch(Polygon(np.array(poly), closed=True, fill=False, ec=base.ORANGE, lw=1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{Path(i.path).name} · blue = detector polygon, orange = characters", fontsize=8)
        info = fig.add_subplot(gs[r, 3])
        info.set_axis_off()
        lines = [kind.upper(), f"GT: '{i.gt_raw}'" if i.gt else "GT: UNKNOWN", f"A (CRNN): '{i.text_a}'",
                 f"segments tight/padded: {sum(x.n_segments for x in i.runs['tight'])}/"
                 f"{sum(x.n_segments for x in i.runs['padded'])} (expected {len(i.gt) if i.gt else '?'})",
                 f"neighbour regions in padded crop: {sum(x.other_regions_in_crop for x in i.runs['padded'])}"]
        for kk, l in enumerate(lines):
            info.text(0, 0.95 - kk * 0.2, base.safe_text(l), fontsize=9, fontweight="bold" if kk == 0 else "normal",
                      transform=info.transAxes, va="top")
    return _save(fig, out / "crop_comparison.png")


def dashboard(ev: Dict, out: Path, report: Dict) -> Path:
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.4, 1.0, 0.9], hspace=0.4, wspace=0.3)
    fig.text(0.012, 0.995, "PAC · FULL-SCENE RECOGNITION STABILIZATION", fontsize=20, fontweight="bold", va="top")
    fig.text(0.012, 0.975, "Padding 2.0 x text height (post-hoc choice) · scene GT only from manual labels · "
             "OCR and topology compared, never fused", fontsize=11, color=RED, va="top")
    s = ev["scenes"]
    c = ev["crops"]
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    tiles = [("Scenes", s["total"]), ("Detected", s["detected"]), ("Missed", len(s["missed"])),
             ("Labelled scenes", s["labeled"]), ("Crop CER tight", f"{c['tight']['cer']:.3f}"),
             ("Crop CER padded", f"{c['padded']['cer']:.3f}"), ("Exact padded", f"{c['padded']['exact']:.3f}")]
    for k, (t, v) in enumerate(tiles):
        x0 = k / len(tiles)
        ax.add_patch(Rectangle((x0 + 0.004, 0), 1 / len(tiles) - 0.008, 0.9, transform=ax.transAxes,
                               facecolor=SURFACE, edgecolor=GRID))
        ax.text(x0 + 0.5 / len(tiles), 0.52, str(v), transform=ax.transAxes, ha="center", fontsize=20, fontweight="bold")
        ax.text(x0 + 0.5 / len(tiles), 0.14, t, transform=ax.transAxes, ha="center", fontsize=10, color=INK_2)
    ax = fig.add_subplot(gs[1, 0])
    keys = sorted(set(c["tight"]["failures"]) | set(c["padded"]["failures"]))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [c["tight"]["failures"].get(k, 0) for k in keys], 0.4, color=MUTED, label="tight")
    ax.bar(x + 0.2, [c["padded"]["failures"].get(k, 0) for k in keys], 0.4, color=base.BLUE, label="padded")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=9)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("Failure attribution (273 transcribed crops)", fontsize=11, loc="left")
    ax = fig.add_subplot(gs[1, 1])
    comp = ev["segmentation_compatibility"]
    items = sorted({k.split("|")[1] for k in comp})
    y = np.arange(len(items))
    ax.barh(y - 0.2, [comp.get(f"tight|{k}", 0) for k in items], 0.4, color=MUTED, label="tight")
    ax.barh(y + 0.2, [comp.get(f"padded|{k}", 0) for k in items], 0.4, color=base.BLUE, label="padded")
    ax.set_yticks(y)
    ax.set_yticklabels(items, fontsize=8.5)
    ax.legend(frameon=False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_title("Segmentation compatibility checks (images)", fontsize=11, loc="left")
    ax = fig.add_subplot(gs[1, 2])
    ov = ev["ocr_vs_topology"]
    order = ["both_correct", "ocr_only_correct", "topology_only_correct", "both_wrong"]
    ax.bar(range(4), [ov.get(k, 0) for k in order], color=[GOOD, base.ORANGE, base.BLUE, BAD])
    for k, kk in enumerate(order):
        ax.text(k, ov.get(kk, 0) + 1, str(ov.get(kk, 0)), ha="center")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["both", "OCR only", "Topology only", "neither"])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("A (CRNN) vs B (padded), case-folded alnum", fontsize=11, loc="left")
    ax = fig.add_subplot(gs[2, :])
    ax.set_axis_off()
    ax.set_title("LIMITATIONS", loc="left", fontsize=12, fontweight="bold")
    lim = report["limitations"]
    for k, (kk, t) in enumerate(lim.items()):
        ax.text(0, 0.9 - k * 0.16, base.safe_text(f"{kk}: {t}")[:230], fontsize=10.5, color=INK_2, transform=ax.transAxes)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.03)
    return _save(fig, out / "final_dashboard.png")


def scene_label_status(s: Dict) -> str:
    """Label-coverage sentence that reflects the ACTUAL labelling state (fixes the stale NOT EVALUABLE text)."""
    labelled, total = s["labeled"], s["total"]
    ev = s.get("evaluation")
    if labelled == 0:
        return (f"0/{total}; scene end-to-end metrics are NOT EVALUABLE until scenes are labelled "
                "(--label-scenes).")
    pad = ev["padded"] if isinstance(ev, dict) else {}
    head = (f"{labelled}/{total}; Scene exact = {pad.get('exact', float('nan')):.4f}; "
            f"Scene CER = {pad.get('cer', float('nan')):.4f}")
    rest = []
    if s.get("unreadable"):
        rest.append(f"{s['unreadable']} marked unreadable")
    if s.get("unlabeled"):
        rest.append(f"{s['unlabeled']} still unlabelled (excluded from scene metrics)")
    return head + ("; " + ", ".join(rest) if rest else "")


def build_report(ev: Dict, res: Dict, cfg: AppConfig) -> Dict:
    s = ev["scenes"]
    return {
        "experiment": "FULL-SCENE RECOGNITION STABILIZATION",
        "padding_rule": {"pad_per_side": f"{cfg.scene.pad_ratio} x text height (short side of the region's "
                                         "minimum-area rectangle)", "before": f"{cfg.scene.tight_ratio} x text height",
                         "rotation": f"regions with |angle| >= {cfg.scene.rotation_threshold_deg} deg are deskewed "
                                     "around the region centre before cropping", "border": "crop clipped to the image",
                         "coordinates": "every character box is mapped back to original-image coordinates",
                         "provenance": "value suggested by the earlier post-hoc margin diagnostic on these crops"},
        "metrics": {k: v for k, v in ev.items() if not k.startswith("_")},
        "leakage": {"ground_truth_used_during_inference": False, "train_test_group_overlap": 0,
                    "scene_labels_source": "manual only (data/scene_labels.csv); none generated automatically"},
        "limitations": {
            "Post-hoc padding": "pad 2.0 was chosen after the previous run on the same 273 crops; the crop "
                                "improvement is not independent validation.",
            "Scene labels": scene_label_status(s),
            "Unsupported OCR characters": (f"CRNN-EN reads only 0-9a-z: {ev['unsupported_by_crnn_images']} evaluated "
                                           "images contain symbols it cannot output; case is lost."),
            "Other": "No detection ground truth; dataset images are not real shipyard photographs; topology features "
                     "were designed on this data (exploratory); posteriors and detector scores are uncalibrated.",
        },
    }


def terminal_summary(ev: Dict, report: Dict) -> str:
    s, c = ev["scenes"], ev["crops"]
    ov = ev["ocr_vs_topology"]
    sup = ev["ocr_vs_topology_supported_only"]
    o = ["=" * 60, "PAC FULL-SCENE RECOGNITION STABILIZATION", "=" * 60, "", "[SCENE DETECTION]",
         f"Total = {s['total']}", f"Detected = {s['detected']}", f"Missed = {len(s['missed'])} {s['missed']}", "",
         "[CROP STABILIZATION]  (273 transcribed crops; padding value post-hoc)",
         f"Original CER = {c['tight']['cer']:.4f} (exact {c['tight']['exact']:.4f})",
         f"Padded CER = {c['padded']['cer']:.4f} (exact {c['padded']['exact']:.4f})",
         f"Segmentation failures before = {c['tight']['failures'].get('SEGMENTATION', 0)} "
         f"(+ cropping {c['tight']['failures'].get('CROPPING', 0)})",
         f"Segmentation failures after = {c['padded']['failures'].get('SEGMENTATION', 0)} "
         f"(+ cropping {c['padded']['failures'].get('CROPPING', 0)})",
         f"By style padded: Formal CER {ev['crops_by_style'][FORMAL]['padded']['cer']:.4f}, Informal CER "
         f"{ev['crops_by_style'][INFORMAL]['padded']['cer']:.4f}",
         f"Failure attribution tight = {c['tight']['failures']}", f"Failure attribution padded = {c['padded']['failures']}",
         f"Compatibility checks = {ev['segmentation_compatibility']}", "", "[SCENE LABELING]",
         f"Labeled = {s['labeled']}", f"Unlabeled = {s['unlabeled']}", f"Unreadable = {s['unreadable']}", "",
         "[END-TO-END RECOGNITION]  (scenes with manual labels only)"]
    if isinstance(s["evaluation"], dict):
        e = s["evaluation"]["padded"]
        o += [f"Evaluable scenes = {e['images']}", f"Exact match = {e['exact']:.4f}", f"CER = {e['cer']:.4f}",
              f"Failures = {e['failures']}"]
    else:
        o += ["Evaluable scenes = 0", f"Exact match = {s['evaluation']}", f"CER = {s['evaluation']}"]
    o += ["", "[OCR COMPLEMENTARITY]  (padded B vs CRNN A, case-folded alphanumerics; crops + labelled scenes)",
          f"Both correct = {ov.get('both_correct', 0)}", f"OCR only correct = {ov.get('ocr_only_correct', 0)}",
          f"Topology only correct = {ov.get('topology_only_correct', 0)}", f"Both wrong = {ov.get('both_wrong', 0)}",
          f"Images with characters outside CRNN charset = {ev['unsupported_by_crnn_images']}; on CRNN-supported "
          f"images ({sup['images']}): A exact {sup['A_exact_casefold']:.4f} vs B exact {sup['B_exact_casefold']:.4f}",
          "Category traits (fraction of images):"]
    for t in ev["category_traits"]:
        o.append("  " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
    o += ["", "[LEAKAGE]", "Ground truth used during inference = False",
          "Training/test group overlap = 0 (leave-one-group-out; scenes never in training)", "", "[LIMITATIONS]"]
    o += [f"{k} = {v}" for k, v in report["limitations"].items()] + ["=" * 60]
    return "\n".join(o)


def render_and_save(res: Dict, ev: Dict, cfg: AppConfig, out: Path) -> Dict:
    report = build_report(ev, res, cfg)
    write_tables(res["images"], ev, out, cfg)
    for name, fn in (("crop_comparison", lambda: crop_comparison(res["images"], out, cfg)),
                     ("final_dashboard", lambda: dashboard(ev, out, report))):
        try:
            fn()
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] {name}: {type(exc).__name__}: {exc}")
    (out / "final_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "final_report.txt").write_text(terminal_summary(ev, report), encoding="utf-8")
    return report
