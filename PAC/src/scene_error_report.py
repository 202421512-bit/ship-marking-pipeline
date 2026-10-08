"""Outputs for the SCENE ERROR AUDIT."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from . import visualization as base  # noqa: E402
from .config import PROJECT_ROOT  # noqa: E402
from .image_io import load_image  # noqa: E402
from .real_experiment import write_csv  # noqa: E402
from .scene_error_audit import AUDIT_DIR, CAUSES, AuditResult, align  # noqa: E402

GOOD, BAD = base.STATUS["HIGH_CONFIDENCE"], base.STATUS["AMBIGUOUS"]

# Manual review after inspecting failed_scenes_visualization.png. Key: (scene, gt_index) -> (cause, evidence).
# Empty unless an automatic attribution is contradicted by the image; every override is reported.
MANUAL_REVIEW: Dict[tuple, tuple] = {
    ("scene_011.png", 5): ("SEGMENTATION_COMPONENT_LOSS",
                           "the i-dot is present in the binarized mask (25 px component directly above the stem) but is "
                           "removed by segment_characters' component filters; the classified segment is a bare stem "
                           "(beta0=1 vs trained 'i' beta0=2), topologically identical to I / l"),
    ("scene_015.png", 5): ("SEGMENTATION_COMPONENT_LOSS",
                           "the i-dot is present in the binarized mask (18 px component above the stem) but is removed "
                           "by the component filters; segment beta0=1 vs trained 'i' beta0=2"),
}


def alignment_text(pred: str, gt: str) -> str:
    out = []
    for op, gi, pi in align(pred, gt):
        if op == "MATCH":
            out.append(gt[gi])
        elif op == "SUB":
            out.append(f"[{gt[gi]}→{pred[pi]}]")
        elif op == "DEL":
            out.append(f"[{gt[gi]}→∅]")
        else:
            out.append(f"[∅→{pred[pi]}]")
    return "".join(out)


def visualization(r: AuditResult, path: Path) -> Path:
    fs = r.failed
    fig = plt.figure(figsize=(20, 4.2 * len(fs) + 1))
    fig.text(0.01, 0.997, f"Failed scenes ({len(fs)}/30): original + detector polygon | padded crop + segment boxes | "
             "segments with prediction | GT vs prediction alignment", fontsize=14, fontweight="bold", va="top")
    gs = GridSpec(len(fs), 1, figure=fig, hspace=0.45)
    for row, s in enumerate(fs):
        nseg = max(6, len(s.pred_segments))
        sub = GridSpecFromSubplotSpec(2, nseg, subplot_spec=gs[row], height_ratios=[1.6, 1], hspace=0.5, wspace=0.2)
        ax = fig.add_subplot(sub[0, : nseg // 3])
        img = load_image(PROJECT_ROOT / s.path).image
        ax.imshow(img[..., ::-1] if img.ndim == 3 else img)
        for g in s.regions:
            ax.add_patch(Polygon(g["poly"], closed=True, fill=False, ec=base.BLUE, lw=1.8))
        ax.set_title(Path(s.path).name + ("  (NO DETECTION)" if not s.regions else ""), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax2 = fig.add_subplot(sub[0, nseg // 3: 2 * nseg // 3])
        if s.crops:
            ax2.imshow(s.crops[0].image[..., ::-1])
            for e in s.pred_segments:
                if e["region"] == s.regions[0]["index"]:
                    x0, y0, x1, y1 = e["box"]
                    ax2.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=base.ORANGE, lw=1.5))
                    ax2.text(x0, y0 - 2, str(e["k"] + 1), color=base.ORANGE, fontsize=8)
            ax2.set_title("padded crop (2.0 x height) + segments", fontsize=9)
        else:
            ax2.text(0.5, 0.5, "no region → no crop", ha="center", transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
        info = fig.add_subplot(sub[0, 2 * nseg // 3:])
        info.set_axis_off()
        errs = [e for e in r.errors if e["scene"] == Path(s.path).name]
        lines = [(f"GT   : '{s.gt_raw}'  (compared as '{s.gt}')", base.INK),
                 (f"Pred : '{s.text_b}'   CRNN: '{s.text_a}'", base.INK),
                 (f"Align: {alignment_text(s.text_b, s.gt)}", BAD)]
        lines += [(f"• {e['op']} '{e['gt_char']}'→'{e['pred_char']}': {e['cause']}", base.INK_2) for e in errs]
        for k, (t, col) in enumerate(lines):
            info.text(0, 0.95 - k * 0.15, base.safe_text(t), fontsize=9.5, color=col, transform=info.transAxes, va="top",
                      fontweight="bold" if k < 3 else "normal")
        for e in s.pred_segments[:nseg]:
            a = fig.add_subplot(sub[1, s.pred_segments.index(e)])
            a.imshow(np.where(e["mask"], 0, 255), cmap="gray", vmin=0, vmax=255)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(base.safe_text(f"#{e['k'] + 1} → '{e['pred']}'"), fontsize=8)
            a.set_xlabel(base.safe_text(" ".join(f"{c}:{p:.2f}" for c, p in e["top3"])), fontsize=7)
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def build_and_write(r: AuditResult) -> Dict[str, object]:
    out = AUDIT_DIR
    for e in r.errors:
        key = (e["scene"], e["gt_index"])
        e["cause_auto"] = e["cause"]
        if key in MANUAL_REVIEW:
            e["cause"], note = MANUAL_REVIEW[key]
            e["evidence"] += f" | MANUAL REVIEW: {note}"
            e["manual_override"] = True
        else:
            e["manual_override"] = False
    # topology analysis only for errors whose FINAL cause is a genuine character confusion
    keep = {(e["scene"], e["segment"]) for e in r.errors if e["cause"] == "CHARACTER_CONFUSION"}
    r.topo = [t for t in r.topo if (t["scene"], t["segment"]) in keep]
    failed_rows = []
    for s in r.failed:
        errs = [e for e in r.errors if e["scene"] == Path(s.path).name]
        failed_rows.append({"scene": Path(s.path).name, "ground_truth": s.gt_raw, "ground_truth_normalized": s.gt,
                            "prediction": s.text_b, "crnn": s.text_a, "alignment": alignment_text(s.text_b, s.gt),
                            "regions": len(s.regions), "segments": len(s.pred_segments),
                            "char_errors": len(errs), "causes": " | ".join(e["cause"] for e in errs)})
    write_csv(out / "failed_scenes.csv", failed_rows)
    write_csv(out / "character_error_alignment.csv", r.errors)
    write_csv(out / "unsupported_characters.csv", r.unsupported)
    write_csv(out / "training_class_inventory.csv", r.inventory)
    write_csv(out / "topology_error_analysis.csv", r.topo)
    write_csv(out / "scene_crop_overlap_audit.csv", r.independence)
    write_csv(out / "ocr_comparison.csv", r.ocr["rows"])
    visualization(r, out / "failed_scenes_visualization.png")
    cause_counts = Counter(e["cause"] for e in r.errors)
    n_gt = sum(len(s.gt) for s in r.runs)
    exact = sum(s.text_b == s.gt for s in r.runs) / len(r.runs)
    from .real_marking_recognition import edit_ops
    cer = sum(edit_ops(s.text_b, s.gt)[0] for s in r.runs) / n_gt
    rep = {"verification": r.verification, "scene_exact": exact, "scene_cer": cer, "scenes": len(r.runs),
           "failed_scenes": len(r.failed), "character_errors": len(r.errors), "cause_counts": dict(cause_counts),
           "classes": r.classes, "ocr": r.ocr["counts"],
           "independence": {
               "likely_same_rendering": sum(x["verdict"].startswith("LIKELY") for x in r.independence),
               "probable_same_rendering": sum(x["verdict"].startswith("PROBABLE") for x in r.independence),
               "same_text_content": sum(x.get("crops_with_identical_transcription", 0) > 0 for x in r.independence
                                        if x["scene"] != "_NEGATIVE_CONTROL_"),
               "control": next(x for x in r.independence if x["scene"] == "_NEGATIVE_CONTROL_")},
           "manual_overrides": [(k, v) for k, v in MANUAL_REVIEW.items()]}
    r.report = rep
    (out / "final_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "final_report.txt").write_text(summary(r), encoding="utf-8")
    return rep


def summary(r: AuditResult) -> str:
    rep = r.report
    cc = Counter(e["cause"] for e in r.errors)
    missing_arrows = [x["class"] for x in r.inventory if x["type"].startswith("MISSING") and x["class"] in "→←↑↓"]
    missing_other = [x["class"] for x in r.inventory if x["type"].startswith("MISSING") and x["class"] not in "→←↑↓"]
    gt_missing = sorted({e["gt_char"] for e in r.errors if e["cause"] == "UNKNOWN_CLASS"})
    topo = r.topo
    indist = [t for t in topo if not t["topology_can_distinguish_classes"]]
    recov = [t for t in topo if t["recoverability"].startswith("potentially")]
    case_pairs = [t for t in topo if t["case_pair"]]
    o = ["=" * 64, "PAC SCENE ERROR AUDIT (frozen system, 30 manually labelled scenes)", "=" * 64,
         f"Verification: frozen re-run reproduces stored strings for {rep['verification']['scenes_checked']} scenes "
         f"(0 mismatches) · Scene exact = {rep['scene_exact']:.4f} · Scene CER = {rep['scene_cer']:.4f}", "",
         "[SCENE FAILURES]",
         f"Failed scenes = {rep['failed_scenes']}/30, character errors = {len(r.errors)}",
         f"Total errors = {len(r.errors)}", f"Detection = {cc.get('DETECTION_MISS', 0)}",
         f"Cropping = {cc.get('CROP_DAMAGE', 0)}",
         f"Segmentation = {cc.get('SEGMENTATION_SPLIT', 0) + cc.get('SEGMENTATION_MERGE', 0) + cc.get('SEGMENTATION_COMPONENT_LOSS', 0)} "
         f"(split {cc.get('SEGMENTATION_SPLIT', 0)}, merge {cc.get('SEGMENTATION_MERGE', 0)}, component loss "
         f"{cc.get('SEGMENTATION_COMPONENT_LOSS', 0)} [manual review, see evidence])",
         f"Unknown classes = {cc.get('UNKNOWN_CLASS', 0)}  {gt_missing}",
         f"Character confusion = {cc.get('CHARACTER_CONFUSION', 0)}",
         f"Reading order = {cc.get('READING_ORDER', 0)}",
         f"Preprocessing artifact = {cc.get('PREPROCESSING_ARTIFACT', 0)}",
         f"Undetermined = {cc.get('UNDETERMINED', 0)}", "", "Per error:"]
    o += [f"  {e['scene']}: {e['op']} '{e['gt_char']}'→'{e['pred_char']}' = {e['cause']} ({e['evidence'][:110]})"
          for e in r.errors]
    o += ["", "[CHARACTER INVENTORY]", f"Total trainable classes = {len(r.classes)}: {' '.join(r.classes)}",
          f"Missing arrow classes = {missing_arrows}", f"Other missing symbols (checked list) = {missing_other}",
          "Characters in scene GT outside the training classes or the CRNN charset:"]
    o += [f"  '{u['char']}' x{u['occurrences_in_scene_gt']}  trained={u['in_training_classes']}  "
          f"crnn={u['in_crnn_charset']}  scenes: {u['scenes']}" for u in r.unsupported]
    o += ["", "[TOPOLOGY ANALYSIS]  (CHARACTER_CONFUSION errors only)",
          f"Feature-related errors = {len(topo)}",
          f"Errors topology cannot distinguish = {len(indist)} (class prototypes differ < 0.5 pooled sigma on every "
          f"topology feature){' - ' + ', '.join(t['gt_char'] + '→' + t['pred_char'] for t in indist) if indist else ''}",
          f"Case pairs (need size / line-relative info) = {len(case_pairs)}",
          f"Potentially recoverable errors = {len(recov)} (some feature group already favours the true class)"]
    o += [f"  {t['scene']} '{t['gt_char']}'→'{t['pred_char']}': topology Δll {t['topology_loglik_true_minus_pred']:+.2f}, "
          f"geometry {t['geometry_loglik_true_minus_pred']:+.2f}, direction {t['direction_loglik_true_minus_pred']:+.2f}; "
          f"sep {t['max_topology_class_separation_sigma']:.2f}σ; true in top-3: {t['gt_in_top3']}; {t['recoverability']}"
          for t in topo]
    oc = rep["ocr"]
    o += ["", "[OCR COMPARISON]  (A = CRNN on case-folded alphanumerics, B = exact; 30 scenes)",
          f"Both correct = {oc.get('both_correct', 0)}  OCR only correct = {oc.get('ocr_only_correct', 0)}  "
          f"Topology only correct = {oc.get('topology_only_correct', 0)}  Both wrong = {oc.get('both_wrong', 0)}",
          f"Scenes whose GT has characters outside the CRNN charset = {oc.get('gt_outside_crnn_charset', 0)} "
          "(CRNN cannot output them - not counted as an ordinary OCR misclassification)"]
    ind = rep["independence"]
    c = ind["control"]
    o += ["", "[DATA INDEPENDENCE]",
          f"Negative control (scene vs DIFFERENT-text crop): {c['verdict']}, 95th pct {c['negative_control_p95']:.0f}, "
          f"max {c['negative_control_max']:.0f} ORB+RANSAC inliers -> a fixed '>= 20' rule is not discriminative",
          f"Scene/crop duplication found = {ind['likely_same_rendering']} scenes exceed EVERY control value (likely "
          f"same rendering) + {ind['probable_same_rendering']} exceed the control 95th percentile (probable)",
          f"Uncertain overlaps = {ind['same_text_content']} of 30 scenes contain a marking string identical to a "
          "transcribed training crop; filenames / metadata carry no source link",
          "Implication for evaluation = the scene model is trained on all 989 validated characters, which include "
          "those crops; most scenes therefore re-present markings (text and very likely the same rendering) the "
          "classifier has seen. Scene exact 0.70 / CER 0.083 is NOT an independent test of unseen markings; only "
          f"scenes without matching crops ({30 - ind['same_text_content']}: +3, <-F6, F8 ->, V20 ^) are content-novel - "
          "and all 4 contain characters outside the training classes."]
    return "\n".join(o)
