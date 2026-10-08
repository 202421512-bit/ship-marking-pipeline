"""Figures and reports for REAL MARKING RECOGNITION."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import visualization as base  # noqa: E402
from .config import PROJECT_ROOT, AppConfig  # noqa: E402
from .dataset import FORMAL, INFORMAL  # noqa: E402
from .image_io import load_image  # noqa: E402
from .real_experiment import rel  # noqa: E402
from .real_marking_recognition import (AUDIT_CATEGORIES, MODELS, MarkingImage, MarkingResult,  # noqa: E402
                                       audit_membership, edit_ops)

SURFACE, INK, INK_2, MUTED, GRID, RED = base.SURFACE, base.INK, base.INK_2, base.MUTED, base.GRID, base.RED
GOOD, BAD = base.STATUS["HIGH_CONFIDENCE"], base.STATUS["AMBIGUOUS"]
BANNER = "REAL DATA · whole-marking recognition · leave-one-group-out · no digit-8 demo"
DPI = 110

PIPELINE = {
    "Text detection": ("PARTIAL", "No detector feeds recognition. Inputs are pre-cropped marking images; the "
                                  "scene heuristic (real_experiment.scene_candidates) is qualitative only."),
    "Character segmentation": ("IMPLEMENTED", "dataset.segment_characters (connected components + contrast / "
                                             "text-band / line filters + x-overlap merge); failures measured below."),
    "Topology extraction": ("IMPLEMENTED", "topology.analyze_topology (+ GUDHI persistence counts)."),
    "Geometry extraction": ("IMPLEMENTED", "geometry.analyze_geometry."),
    "Bayesian recognition": ("IMPLEMENTED", "per segment; data-fitted Gaussian model (Combined 0.5/0.5), unchanged."),
    "String reconstruction": ("PARTIAL", "implemented now: single-line left-to-right concatenation; spaces and "
                                         "multi-line order are FLAGGED, not decided."),
    "Scene -> text end-to-end": ("NOT IMPLEMENTED", "full block photo -> detection -> crop -> recognition is not "
                                                   "connected."),
}

BIAS_AUDIT = [
    {"item": "Digit '8' as default input", "where": "config.DemoConfig.target='8'; image_io.generate_demo_image; "
     "Main --demo / GUI Demo", "impact": "demo only (isolated: data/demo/, results/demo_synthetic/)",
     "action": "kept, isolated"},
    {"item": "Digit '8' assumed as answer / beta1=2 fixed", "where": "none found (restoration targets come from "
     "Top-K reference statistics)", "impact": "none", "action": "none needed"},
    {"item": "8-specific restoration rule", "where": "none found", "impact": "none", "action": "none needed"},
    {"item": "Demo '8' copied into real input folder", "where": "image_io.write_demo_files wrote data/input/test.png",
     "impact": "REAL: README/Main examples ran --input on the synthetic 8; results/ root showed a synthetic-8 "
     "result labelled 'SINGLE IMAGE'", "action": "FIXED: no write into data/input; legacy outputs moved to "
     "results/demo_synthetic/legacy_root_outputs/"},
    {"item": "Demo / single-image outputs in results/ root (GUI 'open results folder')",
     "where": "Main --output default = results/", "impact": "REAL: demo-8 dashboard looked like the main result",
     "action": "FIXED: --demo -> results/demo_synthetic/, --input/--gui -> results/single_image/"},
    {"item": "Synthetic font reference preferred over real data", "where": "prototypes.load_reference_model "
     "(auto)", "impact": "data-fitted model (46 classes) is used when present; silent synthetic fallback otherwise",
     "action": "FIXED: fallback now prints an explicit warning"},
    {"item": "Demo images inside real evaluations", "where": "real_experiment / character / degradation / "
     "uncertainty / this check read only data/input/Symbols", "impact": "none", "action": "verified"},
]


def _banner(fig, title: str) -> None:
    fig.text(0.01, 0.995, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.99, 0.995, BANNER, fontsize=9, color=RED, fontweight="bold", va="top", ha="right")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _rgb(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else img[..., ::-1]


def _boxes(ax, it: MarkingImage, model: str) -> None:
    for s in it.segments:
        x0, y0, x1, y1 = s.box
        ok = None
        if it.status == "COUNT_MATCH" and s.index < len(it.gt) and s.preds.get(model):
            ok = s.preds[model][0][0] == it.gt[s.index]
        col = GOOD if ok else BAD if ok is False else base.YELLOW
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.6))
        ax.text(x0, y0 - 3, str(s.index + 1), color=col, fontsize=7)


def select_examples(r: MarkingResult, style: str, cfg: AppConfig) -> List[MarkingImage]:
    """Pre-declared rule: lowest image path first; correct / incorrect / segmentation-failure kinds."""
    k = cfg.marking.examples_per_kind
    pm = cfg.marking.primary_model
    tr = sorted([i for i in r.items if i.style == style and i.transcribed], key=lambda i: i.path)
    correct = [i for i in tr if i.status == "COUNT_MATCH" and i.text[pm] == i.gt][:k]
    wrong = [i for i in tr if i.status == "COUNT_MATCH" and i.text[pm] != i.gt][:k]
    segfail = [i for i in tr if i.status != "COUNT_MATCH"][:max(1, k - 1)]
    return correct + wrong + segfail


def examples_figure(r: MarkingResult, style: str, path: Path, cfg: AppConfig) -> Path:
    ex = select_examples(r, style, cfg)
    pm = cfg.marking.primary_model
    fig = plt.figure(figsize=(18, 3.9 * max(1, len(ex)) + 0.8))
    _banner(fig, f"{style} markings: original → segmentation → per-character candidates → reconstructed text "
                 "(rule: lowest path first; correct / incorrect / segmentation failure)")
    gs = GridSpec(max(1, len(ex)), 1, figure=fig, hspace=0.55)
    for row, it in enumerate(ex):
        ncol = max(8, len(it.segments))
        sub = GridSpecFromSubplotSpec(2, ncol, subplot_spec=gs[row], height_ratios=[1.4, 1], hspace=0.6, wspace=0.25)
        ax = fig.add_subplot(sub[0, :ncol // 2])
        img = load_image(PROJECT_ROOT / it.path).image
        ax.imshow(_rgb(img))
        _boxes(ax, it, pm)
        ax.set_xticks([])
        ax.set_yticks([])
        ok = it.text[pm] == it.gt
        d = edit_ops(it.text[pm], it.gt)[0]
        info = fig.add_subplot(sub[0, ncol // 2:])
        info.set_axis_off()
        lines = [(f"{Path(it.path).name}   [{it.status}]", INK, "bold"),
                 (f"Ground truth : '{it.transcription}'  (compared as '{it.gt}')", INK, "normal"),
                 (f"Recognized   : '{it.text[pm]}'   (Topo+Geo)", INK, "bold"),
                 (f"{'CORRECT' if ok else 'INCORRECT'}   edit distance {d}, CER {d / max(1, len(it.gt)):.2f}   "
                  f"segments {len(it.segments)} / expected {len(it.gt)}", GOOD if ok else BAD, "bold"),
                 (f"Geometry only: '{it.text['geometry']}'   Topology only: '{it.text['topology']}'", INK_2, "normal"),
                 (f"possible spaces after segment: {it.flags.get('possible_space_after') or '-'}  ·  "
                  f"{it.flags.get('reading_order', '')}", MUTED, "normal")]
        for k, (t, col, w) in enumerate(lines):
            info.text(0, 0.95 - k * 0.17, base.safe_text(t), fontsize=9.5, color=col, fontweight=w,
                      transform=info.transAxes, va="top")
        for s in it.segments[:ncol]:
            a = fig.add_subplot(sub[1, s.index])
            a.imshow(np.where(s.mask, 0, 255), cmap="gray", vmin=0, vmax=255)
            a.set_xticks([])
            a.set_yticks([])
            cand = s.preds.get(pm, [])
            gt_ch = it.gt[s.index] if (it.status == "COUNT_MATCH" and s.index < len(it.gt)) else None
            col = INK if gt_ch is None else (GOOD if cand and cand[0][0] == gt_ch else BAD)
            txt = "\n".join(f"{c} {p:.2f}" for c, p in cand) or "no features"
            topo = (f"β0={s.topo.get('beta_0', 0):.0f} β1={s.topo.get('beta_1', 0):.0f} "
                    f"e={s.topo.get('endpoints', 0):.0f}") if s.topo else ""
            a.set_title(base.safe_text(f"#{s.index + 1}" + (f" GT '{gt_ch}'" if gt_ch else "")), fontsize=7.5, color=col)
            a.set_xlabel(base.safe_text(txt + "\n" + topo), fontsize=7, color=col)
    return _save(fig, path)


def audit_sheet(r: MarkingResult, cat: str, path: Path, cfg: AppConfig) -> Path:
    pm = cfg.marking.primary_model
    mem = sorted([i for i in r.items if cat in audit_membership(i, cfg)], key=lambda i: i.path)
    shown = mem[:cfg.marking.audit_images_per_category]
    cols = 3
    rows = max(1, int(np.ceil(len(shown) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 2.9 * rows + 1))
    _banner(fig, f"Segmentation audit: {cat} — {AUDIT_CATEGORIES[cat]} ({len(mem)} images; first "
                 f"{len(shown)} by path). Green = correct, red = wrong, yellow = unaligned")
    for ax in np.atleast_1d(axes).ravel():
        ax.set_axis_off()
    for ax, it in zip(np.atleast_1d(axes).ravel(), shown):
        ax.set_axis_on()
        ax.imshow(_rgb(load_image(PROJECT_ROOT / it.path).image))
        _boxes(ax, it, pm)
        ax.set_xticks([])
        ax.set_yticks([])
        gt = f"GT '{it.transcription}'" if it.transcribed else "GT UNKNOWN"
        ax.set_title(base.safe_text(f"{Path(it.path).stem} [{it.status}] seg {len(it.segments)}/"
                                    f"{len(it.gt) if it.transcribed else '?'}\n{gt} → '{it.text[pm]}'"), fontsize=8,
                     color=GOOD if (it.transcribed and it.text[pm] == it.gt) else (BAD if it.transcribed else INK_2))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def dashboard(r: MarkingResult, path: Path, cfg: AppConfig) -> Path:
    rep = r.report
    m = r.metrics
    pm = cfg.marking.primary_model
    fig = plt.figure(figsize=(22, 25))
    gs = GridSpec(5, 2, figure=fig, height_ratios=[0.35, 0.95, 1.1, 1.1, 1.0], hspace=0.4, wspace=0.18)
    fig.text(0.012, 0.995, "PAC MISSION 3 · REAL MARKING RECOGNITION CHECK (not a digit-8 demo)", fontsize=22,
             fontweight="bold", va="top")
    fig.text(0.012, 0.981, "Source: data/input/Symbols Formal / Informal marking images · segmentation → topology / "
             "geometry → Bayesian (Topo+Geo) → string · leave-one-group-out · GT used only for scoring",
             fontsize=11, color=RED, va="top")
    cov = rep["coverage"]
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    tiles = [("Formal images", cov["formal_images"]), ("Informal images", cov["informal_images"]),
             ("Welding (separate model)", cov["welding_images"]), ("Character classes", cov["character_classes"]),
             ("Trainable classes", cov["actual_training_classes"]),
             ("String exact match", f"{m[pm]['all']['string_exact_match']:.3f}"),
             ("CER", f"{m[pm]['all']['cer']:.3f}"), ("Segmentation OK", f"{m['segmentation_success_rate_count_match']:.3f}")]
    for i, (k, v) in enumerate(tiles):
        x0 = i / len(tiles)
        ax.add_patch(Rectangle((x0 + 0.004, 0), 1 / len(tiles) - 0.008, 0.9, transform=ax.transAxes,
                               facecolor=SURFACE, edgecolor=GRID))
        ax.text(x0 + 0.5 / len(tiles), 0.52, str(v), transform=ax.transAxes, ha="center", fontsize=20, fontweight="bold")
        ax.text(x0 + 0.5 / len(tiles), 0.14, k, transform=ax.transAxes, ha="center", fontsize=10, color=INK_2)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    ax.set_title("PIPELINE STATUS", loc="left", fontsize=12, fontweight="bold")
    for k, (name, (st, note)) in enumerate(PIPELINE.items()):
        col = GOOD if st == "IMPLEMENTED" else base.YELLOW if st == "PARTIAL" else BAD
        ax.text(0, 0.9 - k * 0.13, f"{name}", fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(0.33, 0.9 - k * 0.13, st, fontsize=11, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(0.33, 0.9 - k * 0.13 - 0.05, note[:95], fontsize=8, color=INK_2, transform=ax.transAxes)
    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    ax.set_title("RECOGNITION (transcribed images; leave-one-group-out)", loc="left", fontsize=12, fontweight="bold")
    hdr = ["model / scope", "images", "exact", "CER", "seg.char acc", "missed", "extra"]
    xs = [0, 0.3, 0.42, 0.53, 0.64, 0.8, 0.9]
    for x0, h in zip(xs, hdr):
        ax.text(x0, 0.95, h, fontsize=10, fontweight="bold", transform=ax.transAxes)
    k = 0
    for mm in MODELS:
        for scope in ("all", FORMAL, INFORMAL):
            b = m[mm][scope]
            vals = [f"{mm} / {scope}", str(b["images"]), f"{b['string_exact_match']:.3f}", f"{b['cer']:.3f}",
                    f"{b['segmented_char_accuracy_count_match']:.3f}", str(b["deletions_missed_chars"]),
                    str(b["insertions_extra_chars"])]
            for x0, v in zip(xs, vals):
                ax.text(x0, 0.87 - k * 0.09, v, fontsize=9.5, transform=ax.transAxes,
                        fontweight="bold" if mm == pm else "normal")
            k += 1
    ax = fig.add_subplot(gs[2, :])
    pc = sorted(r.per_class, key=lambda x: x["accuracy"])
    xpos = np.arange(len(pc))
    ax.bar(xpos, [x["accuracy"] for x in pc], color=[GOOD if x["accuracy"] >= 0.8 else base.YELLOW
                                                   if x["accuracy"] >= 0.5 else BAD for x in pc], edgecolor=SURFACE)
    ax.set_xticks(xpos)
    ax.set_xticklabels([base.safe_text(f"{x['class']}\n{x['n']}") for x in pc], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("Per-class accuracy of segmented characters (Topo+Geo, COUNT_MATCH images; label = class / n)",
                 loc="left", fontsize=12, fontweight="bold")
    ax = fig.add_subplot(gs[3, 0])
    ax.set_axis_off()
    ax.set_title("SEGMENTATION AUDIT (COUNT_MATCH ≠ proof of correct segmentation)", loc="left", fontsize=12,
                 fontweight="bold")
    for k, a in enumerate(r.audit_rows):
        cm = f"{a['count_match_rate']:.2f}" if a["count_match_rate"] != "" else "-"
        em = f"{a['string_exact_match_rate']:.2f}" if a["string_exact_match_rate"] != "" else "-"
        ax.text(0, 0.9 - k * 0.12, f"{a['category']:<16} images {a['images']:>3}  transcribed {a['transcribed']:>3}  "
                f"COUNT_MATCH {cm}  exact {em}", fontsize=10, family="monospace", transform=ax.transAxes)
    ax = fig.add_subplot(gs[3, 1])
    ax.set_axis_off()
    ax.set_title("DEMO BIAS AUDIT", loc="left", fontsize=12, fontweight="bold")
    for k, b in enumerate(BIAS_AUDIT):
        ax.text(0, 0.92 - k * 0.135, base.safe_text(f"{b['item']}"), fontsize=9.5, fontweight="bold", transform=ax.transAxes)
        ax.text(0, 0.92 - k * 0.135 - 0.05, base.safe_text(f"→ {b['action']}"[:110]), fontsize=8.5,
                color=BAD if b["action"].startswith("FIXED") else INK_2, transform=ax.transAxes)
    ax = fig.add_subplot(gs[4, :])
    ax.set_axis_off()
    ax.set_title("REMAINING BLOCKERS (measured)", loc="left", fontsize=12, fontweight="bold")
    for k, t in enumerate(rep["blockers"]):
        ax.text(0, 0.88 - k * 0.2, base.safe_text(f"{k + 1}. {t}")[:230], fontsize=11, transform=ax.transAxes)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.955, bottom=0.02)
    return _save(fig, path)


def build_report(r: MarkingResult, cfg: AppConfig) -> Dict[str, object]:
    pm = cfg.marking.primary_model
    m = r.metrics
    items = r.items
    tr = [i for i in items if i.transcribed]
    # error decomposition (primary model): where does the edit distance come from?
    seg_err = sum(edit_ops(i.text[pm], i.gt)[0] for i in tr if i.status != "COUNT_MATCH")
    cls_err = sum(edit_ops(i.text[pm], i.gt)[0] for i in tr if i.status == "COUNT_MATCH")
    unseen = sum(1 for i in tr if i.status == "COUNT_MATCH" for ch in i.gt
                 if ch in i.flags.get("gt_classes_unseen_in_training", []))
    total = seg_err + cls_err
    cats = Counter(c["category"].split(" ")[0] for c in r.coverage)
    blockers = [
        "No text detection on full ship-block photos: recognition only runs on pre-cropped marking images "
        f"(30 scene images cannot be read end-to-end).",
        f"Character classification errors on correctly counted segments: {cls_err} of {total} edit operations "
        f"({cls_err / max(1, total):.0%}); segmented-character accuracy {m[pm]['all']['segmented_char_accuracy_count_match']:.3f} "
        f"(Formal {m[pm][FORMAL]['segmented_char_accuracy_count_match']:.3f} / Informal "
        f"{m[pm][INFORMAL]['segmented_char_accuracy_count_match']:.3f}); {unseen} test characters belong to classes "
        "absent from training.",
        f"Segmentation: {sum(v for v in m['segmentation_failures'].values())} of {len(tr)} transcribed images fail the "
        f"count check ({seg_err} of {total} edit operations, {seg_err / max(1, total):.0%}); touching characters, "
        "dots and thin strokes are the audited failure modes.",
    ]
    # order blockers 2/3 by measured share (blocker 1 is structural)
    if seg_err > cls_err:
        blockers[1], blockers[2] = blockers[2], blockers[1]
    return {
        "experiment": "REAL MARKING RECOGNITION CHECK",
        "source": "data/input/Symbols (Formal / Informal crops); welding symbols use a separate classifier",
        "coverage": {
            "formal_images": sum(1 for i in items if i.style == FORMAL),
            "informal_images": sum(1 for i in items if i.style == INFORMAL),
            "welding_images": 54, "informal_scenes_not_recognized": 30,
            "character_classes": len(r.coverage),
            "actual_training_classes": sum(1 for c in r.coverage if c["training_samples"] > 0),
            "category_counts": dict(cats),
            "untranscribed_images": m["not_transcribed_images"],
        },
        "pipeline": {k: {"status": v[0], "note": v[1]} for k, v in PIPELINE.items()},
        "fit": r.fit_info,
        "metrics": m,
        "error_decomposition_primary": {"edit_ops_total": total, "from_segmentation_failure_images": seg_err,
                                        "from_count_match_images": cls_err,
                                        "count_match_chars_of_unseen_classes": unseen},
        "protocol": {"model": "existing Combined (0.5/0.5) + Geometry / Topology for comparison, unchanged",
                     "training": "COUNT_MATCH characters of all OTHER groups (Formal + Informal), leave-one-group-out",
                     "string": "left-to-right single line, no spaces inserted; possible spaces / off-line segments "
                               "flagged", "comparison": "normalized transcription (spaces removed, space_policy)",
                     "cer": "Levenshtein(prediction, GT) / len(GT); missing and extra characters count as errors",
                     "examples": "deterministic: lowest image path first per kind (correct / incorrect / "
                                 "segmentation failure)"},
        "demo_bias_audit": BIAS_AUDIT,
        "blockers": blockers,
        "limitations": [
            "COUNT_MATCH is necessary, not sufficient: two compensating segmentation errors are not detectable "
            "without box-level ground truth (see audit sheets).",
            "Posteriors are uncalibrated (see results/uncertainty_validation).",
            "Spaces / multi-line layout are not reconstructed.",
            "No data-fitted model file with the current 20-feature set exists for single-image string inference; "
            "this check fits per-fold models in memory.",
        ],
    }


def terminal_summary(r: MarkingResult, cfg: AppConfig) -> str:
    rep = r.report
    m = r.metrics
    pm = cfg.marking.primary_model
    cov = rep["coverage"]
    o = ["=" * 60, "PAC REAL MARKING RECOGNITION CHECK", "=" * 60, "", "[REAL DATA COVERAGE]",
         f"Formal images = {cov['formal_images']}", f"Informal images = {cov['informal_images']}",
         f"Welding symbol images = {cov['welding_images']} (separate 18-class welding classifier; not mixed)",
         f"Character classes = {cov['character_classes']} (transcribed, incl. classes only in failed segmentations)",
         f"Actual training classes = {cov['actual_training_classes']}  {cov['category_counts']}", "", "[PIPELINE]"]
    o += [f"{k} = {v['status']}  ({v['note']})" for k, v in rep["pipeline"].items()]
    a = m[pm]["all"]
    o += ["", f"[ACTUAL RECOGNITION]  (Topo+Geo; leave-one-group-out; GT only for scoring)",
          f"Evaluable images = {m['transcribed_images']} transcribed ({m['status_counts']}); "
          f"{m['not_transcribed_images']} untranscribed images recognized but NOT EVALUABLE",
          f"Character accuracy (segmented, COUNT_MATCH) = {a['segmented_char_accuracy_count_match']:.4f} "
          f"({a['segmented_chars_evaluated']} chars)",
          f"End-to-end character accuracy (1 - CER) = {a['end_to_end_char_accuracy_1_minus_cer']:.4f}",
          f"String exact-match accuracy = {a['string_exact_match']:.4f} (COUNT_MATCH only: "
          f"{a['string_exact_match_count_match_only']:.4f})",
          f"CER = {a['cer']:.4f}  (substitutions {a['substitutions']}, missed {a['deletions_missed_chars']}, "
          f"extra {a['insertions_extra_chars']})",
          f"Segmentation failures = {m['segmentation_failures']} -> success rate "
          f"{m['segmentation_success_rate_count_match']:.4f} (COUNT_MATCH; not proof of correct boxes)"]
    for scope in (FORMAL, INFORMAL):
        b = m[pm][scope]
        o.append(f"  {scope}: images {b['images']}  char acc {b['segmented_char_accuracy_count_match']:.4f}  "
                 f"exact {b['string_exact_match']:.4f}  CER {b['cer']:.4f}")
    for mm in ("geometry", "topology"):
        b = m[mm]["all"]
        o.append(f"  ({mm} only: char acc {b['segmented_char_accuracy_count_match']:.4f}, exact "
                 f"{b['string_exact_match']:.4f}, CER {b['cer']:.4f})")
    o += ["", "[DEMO BIAS AUDIT]"]
    o += [f"- {b['item']}: {b['where']} | impact: {b['impact']} | {b['action']}" for b in rep["demo_bias_audit"]]
    o += ["", "[REMAINING BLOCKERS]"] + [f"{k + 1}. {t}" for k, t in enumerate(rep["blockers"])] + ["=" * 60]
    return "\n".join(o)


def render_and_save(r: MarkingResult, cfg: AppConfig) -> Dict[str, Path]:
    r.report = build_report(r, cfg)
    out = r.out_dir
    files: Dict[str, Path] = {}
    jobs = [("examples_formal", lambda: examples_figure(r, FORMAL, out / "examples_formal.png", cfg)),
            ("examples_informal", lambda: examples_figure(r, INFORMAL, out / "examples_informal.png", cfg))]
    for cat in AUDIT_CATEGORIES:
        jobs.append((f"segmentation_audit_{cat}",
                     lambda c=cat: audit_sheet(r, c, out / f"segmentation_audit_{c}.png", cfg)))
    jobs.append(("final_dashboard", lambda: dashboard(r, out / "final_dashboard.png", cfg)))
    for name, job in jobs:
        try:
            files[name] = job()
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] {name}: {type(exc).__name__}: {exc}")
    r.report["figures"] = {k: rel(v) for k, v in files.items()}
    (out / "final_report.json").write_text(json.dumps(r.report, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    (out / "final_report.txt").write_text(terminal_summary(r, cfg), encoding="utf-8")
    return files
