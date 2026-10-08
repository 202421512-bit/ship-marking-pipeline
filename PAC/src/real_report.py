"""Terminal summary, final_report.txt and final_report.json for the REAL SYMBOLS experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .real_experiment import MODEL_D, POSTERIOR_LABEL, ExperimentResult

LINE = "=" * 50


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "N/A" if v is None else str(v)


def terminal_summary(r: ExperimentResult) -> str:
    """The [DATASET] ... [LIMITATIONS] block with actual numbers."""
    rep, m = r.report, r.metrics
    cc = rep["character_recognition"]
    char_acc = (f"{cc['character_accuracy']:.4f} (leave-one-group-out, {cc['test_characters']} chars)"
                if cc.get("status") == "EVALUATED" else f"NOT EVALUABLE ({cc.get('reason', cc.get('status'))})")
    ood = rep["ood_segments"]
    fig = rep.get("figures", {})
    lines: List[str] = [
        LINE, "PAC REAL SYMBOLS EXPERIMENT", LINE, "",
        "[DATASET]", "",
        f"Source = {rep['dataset_source']}",
        f"Total Images = {rep['total_images']}",
        f"Formal = {rep['formal_images']}",
        f"Informal Crops = {rep['informal_images']}",
        f"Informal Scenes = {rep['scene_images']}",
        f"Welding = {rep['welding_images']}",
        f"Welding Classes = {rep['welding_classes']}", "",
        "[PROCESSING]", "",
        f"Successfully Processed = {rep['processed_images']}",
        f"Failed = {rep['failed_images']}",
    ]
    if rep.get("not_run_images"):
        lines.append(f"Not run (mode) = {rep['not_run_images']}")
    lines += [f"  FAILED: {f['image']} -> {f['reason']}" for f in rep["failed_list"]]
    lines += [
        "", f"[WELDING SUPERVISED EVALUATION]  ({POSTERIOR_LABEL}; 3-fold leave-one-sample-per-class, n=54)", "",
        f"Geometry Only Accuracy = {_fmt(rep['geometry_accuracy'])}",
        f"Topology Only Accuracy = {_fmt(rep['topology_accuracy'])}",
        f"Topology + Geometry Accuracy = {_fmt(rep['combined_accuracy'])}",
        f"Topology + Geometry + Restoration Accuracy = {_fmt(rep['restoration_accuracy'])}", "",
        f"Top-3 Accuracy = {_fmt(rep['top3_accuracy'])}   (Model C)",
        f"Macro Accuracy = {_fmt(rep['macro_accuracy'])}   (Model C)",
        f"Mean Entropy = {_fmt(rep['mean_entropy'])} nats   (Model C, 18 classes)", "",
        "[CONTRIBUTION]", "",
        f"Topology Contribution = {rep['topology_contribution']:+.4f}",
        f"Restoration Contribution = {rep['restoration_contribution']:+.4f}",
        f"Restoration effect = {rep['restoration_effect']}", "",
        "[FORMAL / INFORMAL]", "",
        f"Formal Images Analyzed = {sum(1 for i in r.text['images'] if i['style'] == 'FORMAL' and 'segment_count' in i)}",
        f"Informal Images Analyzed = {sum(1 for i in r.text['images'] if i['style'] == 'INFORMAL' and 'segment_count' in i)}",
        f"Transcribed Formal = {rep['transcribed_images']['FORMAL']}",
        f"Transcribed Informal = {rep['transcribed_images']['INFORMAL']}",
        f"Character Accuracy = {char_acc}", "",
        "[WELDING-TRAINED OOD PROBE]  (not character recognition)", "",
        f"OOD threshold = {rep['welding_ood_threshold']:.4f}",
        f"Formal UNKNOWN/OOD = {_rate(r, 'FORMAL')}  of {ood['FORMAL']} segments",
        f"Informal UNKNOWN/OOD = {_rate(r, 'INFORMAL')}  of {ood['INFORMAL']} segments", "",
        "[SCENE ANALYSIS]", "",
        f"Scenes Processed = {rep['scenes_processed']}",
        f"Candidate Regions = {rep['scene_candidate_regions']}",
        "Detection Accuracy = NOT EVALUABLE (no scene ground truth)", "",
        "[LABEL LEAKAGE]", "",
        f"Status = LABEL_LEAKAGE_RISK={rep['label_leakage_status']['LABEL_LEAKAGE_RISK']}  "
        f"(caption: {rep['label_leakage_status']['caption_text']}); NEAR_DUPLICATE_RISK=TRUE", "",
        "[OUTPUT]", "",
        f"Dashboard = {fig.get('final_dashboard', 'NOT GENERATED')}",
        f"Confusion Matrix = {fig.get('welding_confusion_matrix', 'NOT GENERATED')}",
        f"Formal vs Informal = {fig.get('formal_vs_informal', 'NOT GENERATED')}",
        "Predictions CSV = results/real_symbols/welding_predictions.csv",
        "Final Report = results/real_symbols/final_report.json / final_report.txt", "",
        "[LIMITATIONS]", "",
    ]
    lines += [f"- {t}" for t in rep["limitations"]]
    lines += [LINE]
    return "\n".join(lines)


def _rate(r: ExperimentResult, style: str) -> str:
    sub = [o for o in r.text["ood"] if o["style"] == style]
    if not sub:
        return "N/A"
    n = sum(o["classification"] == "UNKNOWN_OOD" for o in sub)
    return f"{n} ({n / len(sub):.1%})"


def text_report(r: ExperimentResult) -> str:
    """final_report.txt (human readable)."""
    rep, m = r.report, r.metrics
    fvi = rep.get("formal_vs_informal_feature_stats", {})
    tc, rc = rep["topology_contribution"], rep["restoration_contribution"]
    fvi_lines = [f"  {k}: formal mean {v['formal_mean']:.3f} / informal mean {v['informal_mean']:.3f} "
                 f"(medians {v['formal_median']:.3f} / {v['informal_median']:.3f}, Mann-Whitney p={v['mannwhitney_p']:.3g})"
                 for k, v in fvi.items()]
    sec = [
        "PAC MISSION 3 - REAL SYMBOLS DATA EXPERIMENT", "Source: data/input/Symbols (no synthetic data used)", "",
        "DATASET",
        f"  total {rep['total_images']} | formal {rep['formal_images']} | informal crops {rep['informal_images']} | "
        f"scenes {rep['scene_images']} | welding {rep['welding_images']} ({rep['welding_classes']} classes)",
        f"  processed {rep['processed_images']} | failed {rep['failed_images']}", "",
        "METHOD",
        "  Welding: fixed ROI rule (dark ink < threshold, caption band removed), topology (Betti, Euler, skeleton,",
        "  endpoints, branch points, GUDHI persistence counts) + geometry features on a 128 px canvas.",
        "  Gaussian class-conditional model per fold: mean from reference subset, std shrunk toward pooled",
        f"  within-class variance (lambda={rep['fixed_hyperparameters']['variance_shrinkage_lambda']}), sigma floors "
        f"{rep['fixed_hyperparameters']['sigma_floor_relative']} x reference std. Combined = 0.5 mean topology",
        "  log-lik + 0.5 mean geometry log-lik (fixed a priori). Restoration: Model C preliminary -> Top-3 ->",
        "  fold reference statistics -> min J -> re-inference (no ground truth).", "",
        f"WELDING RESULTS ({POSTERIOR_LABEL})",
    ]
    for k in ("A_geometry_only", "B_topology_only", "C_topology_geometry", MODEL_D):
        x = m[k]
        sec.append(f"  {k:<34} top1 {x['top1_accuracy']:.4f} | top3 {x['top3_accuracy']:.4f} | "
                   f"macro {x['macro_accuracy']:.4f} | mean entropy {x['mean_entropy']:.4f}")
    sec.append("  most confused (Model C): " + "; ".join(f"{p['pair']} x{p['count']}"
                                                        for p in rep["most_confused_pairs_model_C"]))
    sec += ["", "TOPOLOGY CONTRIBUTION", f"  Acc(C) - Acc(A) = {tc:+.4f}", "",
            "RESTORATION EFFECT", f"  Acc(D) - Acc(C) = {rc:+.4f}; {rep['restoration_effect']}; "
                                  f"operations {rep['restoration_operations']}", "",
            "FORMAL/INFORMAL ANALYSIS (structural feature analysis, not recognition accuracy)"] + fvi_lines
    cc = rep["character_recognition"]
    sec += [f"  character recognition: {cc.get('status')} {cc.get('reason', '')}", "",
            "OOD ANALYSIS (welding-trained probe; not character recognition)",
            f"  threshold {rep['welding_ood_threshold']:.4f} ({rep['ood_threshold_rule']})",
            f"  formal OOD rate {_fmt(rep['formal_ood_rate'])} | informal OOD rate {_fmt(rep['informal_ood_rate'])}", "",
            "SCENE ANALYSIS",
            f"  {rep['scenes_processed']} scenes, {rep['scene_candidate_regions']} candidate regions; "
            "detection accuracy NOT EVALUABLE", "",
            "LIMITATIONS"] + [f"  - {t}" for t in rep["limitations"]]
    sec += ["", "CONCLUSION", "  " + conclusion(r)]
    return "\n".join(sec)


def conclusion(r: ExperimentResult) -> str:
    """Data-driven conclusion paragraph (wording follows the computed signs)."""
    rep = r.report
    tc, rc = rep["topology_contribution"], rep["restoration_contribution"]
    f_ood, i_ood = rep["formal_ood_rate"], rep["informal_ood_rate"]
    parts = [
        f"On the real welding symbols, adding topology to geometry changed Top-1 accuracy by {tc:+.3f} "
        f"({rep['geometry_accuracy']:.3f} -> {rep['combined_accuracy']:.3f}), "
        + ("so topology helped in this baseline." if tc > 0 else "so topology did not help in this baseline."),
        f"Restoration changed accuracy by {rc:+.3f} ({rep['restoration_effect']['corrected']} corrected, "
        f"{rep['restoration_effect']['degraded']} degraded), "
        + ("i.e. it helped." if rc > 0 else "i.e. it did not help on these clean line drawings."),
    ]
    if f_ood is not None and i_ood is not None:
        parts.append(f"In the welding-trained feature space {f_ood:.1%} of formal and {i_ood:.1%} of informal "
                     "character segments exceed the 95th-percentile welding distance (UNKNOWN_OOD).")
    parts.append("Results are a small-sample baseline with near-duplicate welding variants and uncalibrated "
                 "posteriors; character recognition needs user transcriptions.")
    return " ".join(parts)


def save_reports(r: ExperimentResult) -> None:
    """Write final_report.json (with figure list) and final_report.txt."""
    r.report["conclusion"] = conclusion(r)
    (r.out_dir / "final_report.json").write_text(json.dumps(r.report, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    (r.out_dir / "final_report.txt").write_text(text_report(r), encoding="utf-8")
