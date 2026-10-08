"""Outputs for SYMBOL EXPANSION & SEGMENTATION REPAIR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import visualization as base  # noqa: E402
from .config import DATA_DIR, SYMBOL_RESULTS_DIR, AppConfig  # noqa: E402
from .dataset import binarize_for_segmentation  # noqa: E402
from .real_experiment import write_csv  # noqa: E402
from .symbol_expansion import (SplitLeakageError, class_inventory, dot_check, experimental_evaluation,  # noqa: E402
                               regression_metrics, run_regression, split_audit, symbol_feature_analysis,
                               touching_metrics, unicode_checks, verify_baseline)


def before_after_figure(runs, cfg: AppConfig, out: Path) -> Path:
    """Debug cases + first changed non-debug image + first decimal-point image (deterministic)."""
    pick = [r for r in runs if Path(r.path).stem in ("scene_011", "scene_015")]
    others = [r for r in runs if r not in pick and r.regions and (r.text["baseline"] != r.text["repaired"]
                                                                  or r.nseg["baseline"] != r.nseg["repaired"])]
    pick += others[:1]
    pick += [r for r in runs if r.regions and r.gt and "." in r.gt][:1]
    fig, axes = plt.subplots(len(pick), 5, figsize=(22, 3.0 * len(pick) + 1))
    fig.text(0.01, 0.995, "Component filtering before / after (debug cases are NOT independent evidence)",
             fontsize=14, fontweight="bold", va="top")
    for row, r in zip(np.atleast_2d(axes), pick):
        crop = r.crops[0].image
        mask, _, _ = binarize_for_segmentation(crop, cfg.preprocessing, cfg.dataset)
        n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        rng = np.random.default_rng(0)
        colors = np.vstack([[250, 250, 250], rng.integers(40, 220, size=(max(n - 1, 1), 3))]).astype(np.uint8)
        panels = [(crop[..., ::-1], f"{Path(r.path).name}: original crop"), (mask, "binary"),
                  (colors[lab], f"connected components ({n - 1})")]
        for ax, (im, t) in zip(row[:3], panels):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax.set_title(t, fontsize=9)
        for ax, cond in zip(row[3:], ("baseline", "repaired")):
            ax.imshow(crop[..., ::-1])
            for (x0, y0, x1, y1) in r.segs[cond][0].boxes:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=base.ORANGE, lw=1.5))
            ok = r.text[cond] == r.gt
            ax.set_title(base.safe_text(f"{cond}: '{r.text[cond]}' ({r.nseg[cond]} seg) GT '{r.gt_raw}'"), fontsize=9,
                         color=base.STATUS["HIGH_CONFIDENCE"] if ok else base.STATUS["AMBIGUOUS"])
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "component_filter_before_after.png"
    fig.savefig(p, dpi=105)
    plt.close(fig)
    return p


def run_all(cfg: AppConfig, say=print) -> str:
    out = SYMBOL_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    runs, models, pool, items = run_regression(cfg, say)
    ver = verify_baseline(runs)
    say(f"[SYM] baseline arm reproduces stored strings for {ver['images_checked']} images")
    reg = regression_metrics(runs)
    tch = touching_metrics(runs, items)
    dots = dot_check(runs, cfg)
    reg_rows = [{"image_path": r.path, "kind": r.kind, "gt": r.gt_raw or "UNKNOWN", "baseline": r.text["baseline"],
                 "repaired": r.text["repaired"], "segments_baseline": r.nseg["baseline"],
                 "segments_repaired": r.nseg["repaired"], "expected": len(r.gt) if r.gt else "",
                 "exact_baseline": (r.text["baseline"] == r.gt) if r.gt else "", "exact_repaired": (r.text["repaired"] == r.gt) if r.gt else "",
                 "state_baseline": r.status["baseline"], "state_repaired": r.status["repaired"],
                 "changed": r.text["baseline"] != r.text["repaired"] or r.nseg["baseline"] != r.nseg["repaired"]}
                for r in runs]
    write_csv(out / "segmentation_regression.csv", reg_rows)
    before_after_figure(runs, cfg, out)
    # new data
    try:
        new_rows, split_rows = split_audit(cfg)
    except SplitLeakageError as exc:
        say(f"[ERROR] SPLIT LEAKAGE - stopping: {exc}")
        raise
    write_csv(out / "independent_split_audit.csv", split_rows or [{"status": "NO NEW DATA REGISTERED",
                                                                    "manifest": str(DATA_DIR / cfg.symbols.manifest_file),
                                                                    "incoming_folder": str(DATA_DIR / cfg.symbols.incoming_dir)}])
    write_csv(out / "new_symbol_dataset_manifest.csv", new_rows or [{c: "" for c in
                                                                     ["image_path", "transcription", "source_group",
                                                                      "acquisition_source", "capture_session", "split",
                                                                      "readable", "uncertain"]}])
    exp = experimental_evaluation(new_rows, pool, cfg)
    inv = class_inventory(pool, cfg, exp.get("new_counts", {}))
    write_csv(out / "class_inventory.csv", inv)
    uni = unicode_checks(cfg, out)
    feat_rows = symbol_feature_analysis(runs, new_rows, cfg, models["__all__"])
    write_csv(out / "symbol_feature_analysis.csv", feat_rows or [{"status": "no symbol instances"}])
    text = summary(reg, tch, dots, inv, exp, new_rows, split_rows, feat_rows, uni, ver, cfg)
    (out / "final_report.txt").write_text(text, encoding="utf-8")
    (out / "final_report.json").write_text(json.dumps({"regression": reg, "touching": tch, "dots": dots,
                                                       "experimental": exp, "unicode": uni, "verification": ver},
                                                      indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return text


def summary(reg, tch, dots, inv, exp, new_rows, split_rows, feats, uni, ver, cfg) -> str:
    c, s, d = reg["crops"], reg["scenes"], reg["decimal_point_images"]
    new_inv = [x for x in inv if x["origin"].startswith("registered")]
    with_data = [x["class"] for x in new_inv if x["samples"] > 0]
    o = ["=" * 64, "PAC SYMBOL EXPANSION & SEGMENTATION REPAIR", "=" * 64,
         f"Baseline arm reproduces stored full_scene_stabilization strings: {ver['images_checked']} images, 0 mismatches",
         "", "[SEGMENTATION REPAIR]  (EXPLORATORY - existing data; scene_011/015 are the debugging cases)",
         "i dot preserved = " + "; ".join(f"{x['scene']} {x['condition']}: beta0={x['i_segment_components(beta0)']} "
                                           f"'{x['prediction']}' exact={x['exact']}" for x in dots),
         f"Decimal point regression = {d['baseline']['images']} images with '.': exact {d['baseline']['exact']:.3f} -> "
         f"{d['repaired']['exact']:.3f}, segment-count OK {d['baseline']['segment_count_ok']} -> {d['repaired']['segment_count_ok']}",
         f"Touching-character regression = {tch['baseline']['images']} flagged crops: exact {tch['baseline']['exact']:.3f} -> "
         f"{tch['repaired']['exact']:.3f}, CER {tch['baseline']['cer']:.4f} -> {tch['repaired']['cer']:.4f}, "
         f"segment-count OK {tch['baseline']['segment_count_ok']} -> {tch['repaired']['segment_count_ok']}",
         f"CER before/after = crops {c['baseline']['cer']:.4f} -> {c['repaired']['cer']:.4f} (exact {c['baseline']['exact']:.4f} "
         f"-> {c['repaired']['exact']:.4f}); scenes {s['baseline']['cer']:.4f} -> {s['repaired']['cer']:.4f} "
         f"(exact {s['baseline']['exact']:.4f} -> {s['repaired']['exact']:.4f})",
         f"Images whose output changed = {reg['changed_images']}"]
    o += [f"  {x['image']}: GT '{x['gt']}'  '{x['baseline']}' -> '{x['repaired']}'  segments {x['segments']}"
          for x in reg["changed"]]
    o += ["", "[SYMBOL CLASSES]", f"Existing classes = {sum(1 for x in inv if x['origin'].startswith('existing'))}",
          f"Registered new classes = {len(new_inv)}: {' '.join(x['class'] for x in new_inv)}",
          f"New symbols with actual training data = {with_data or 'none'}",
          f"New symbols without data = {[x['class'] for x in new_inv if x['samples'] == 0]}",
          f"Unicode path checks = {uni}", "", "[TRAINING]"]
    nc = exp.get("new_counts", {})
    o += [f"Training samples per new class = { {k: nc.get(k, {}).get('train_samples', 0) for k in cfg.symbols.new_classes} }",
          f"Source groups per class = { {k: nc.get(k, {}).get('train_groups', 0) for k in cfg.symbols.new_classes} }",
          f"Model trained = {exp.get('model_trained')} ({exp.get('status')})", "",
          "[INDEPENDENT EVALUATION]",
          f"New independent images = {sum(1 for r in new_rows if r.get('eligible_for_final_holdout'))} final-holdout eligible "
          f"of {len(new_rows)} registered",
          f"Independent source groups = {len(split_rows)}",
          "Leakage = " + ("0 (no source group in more than one split)" if new_rows else "n/a (no new data)")]
    if exp.get("model_trained"):
        fh = exp["final_holdout"]
        o += [f"Exact match = {fh['exact']}", f"CER = {fh['cer']}", f"Symbol confusion = {fh['symbol_confusion']}",
              f"Per-class generalization = {exp['per_class_generalization']}"]
    else:
        o += ["Exact match = NOT EVALUABLE", "CER = NOT EVALUABLE",
              "Detection recall = NOT EVALUABLE (no detection ground truth)"]
    o += ["", "[SYMBOL FEATURE ANALYSIS]  (available instances; debug instances are not training data)"]
    for f in feats:
        if "beta_0" in f:
            o.append(f"  '{f['symbol']}' {f['source']}: beta0 {f['beta_0']:.0f} beta1 {f['beta_1']:.0f} endpoints "
                     f"{f['endpoints']:.0f} branch {f['branch_points']:.0f} | LR asym {f['left_right_asymmetry']:+.2f} "
                     f"TB asym {f['top_bottom_asymmetry']:+.2f} | nearest {f['nearest_existing_classes(mean|z|)']} | "
                     f"baseline output '{f['baseline_classifier_output']}'")
        else:
            o.append(f"  {f}")
    return "\n".join(o)
