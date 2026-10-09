"""Final presentation figures P-S1..P-S5 + per-image stage figures, from SAVED files only (no inference, no new masks).

Source of every Phase 4 Recovery stage: reports/phase4_recovery/inference/<image>/ (run of 2026-10-10 03:46, = frozen
stable_baseline, = R1-R5). Baseline (one-polarity, Phase 4 refined M5) only where explicitly labelled 'baseline'.
Pipeline order verified in scripts/phase4_recovery.py + src/pac2/recovery.py:
  channels (bright / dark: local-median contrast AND stroke-scale top/black-hat; low-contrast hysteresis)
  -> per-polarity candidate -> structural segments removed -> component gating (large-structure edges, blob, speck,
  no ridge) = accepted groups -> grouping + 7 shape features + RF probability (AUXILIARY reject only) -> final mask -> overlay.
The recovery pipeline has NO M5 stroke-refinement stage (that belongs to Phase 4 refined, the baseline).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2.visualization import BASE, ERR, FINAL, MID, plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INF = ROOT / "reports" / "phase4_recovery" / "inference"
BASEL = ROOT / "reports" / "phase4_refined" / "inference"
OUT = ROOT / "results" / "final_presentation_6h"
FS = (13.333, 7.5)
rd = lambda p, f=cv2.IMREAD_UNCHANGED: cv2.imdecode(np.fromfile(str(p), np.uint8), f) if Path(p).exists() else None  # noqa: E731
LOG = []


def src(path):
    LOG.append(str(Path(path).relative_to(ROOT)).replace("\\", "/"))
    return path


def mask(n, f, base=False):
    p = (BASEL if base else INF) / n / f
    m = rd(src(p), 0) if p.exists() else None
    return None if m is None else m > 0


def orig(n):
    return rd(src(INF / n / "original.png"), cv2.IMREAD_COLOR)


def show(ax, im, title, binary=False, missing=False):
    if missing or im is None:
        ax.text(0.5, 0.5, "NO SAVED DATA\nfor this stage", ha="center", va="center", fontsize=10, color=ERR)
        ax.set_facecolor("#f2f2f2")
    else:
        ax.imshow(im if im.ndim == 3 else im, cmap=None if im.ndim == 3 else "gray", interpolation="nearest", vmin=0, vmax=1 if im.dtype == bool else 255)
    ax.set_title(title + ("\n[binary mask]" if binary else ""), fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])


def ov(bgr, keep, rem=None):
    v = bgr.copy()
    if rem is not None:
        v[rem] = (0.5 * v[rem] + 0.5 * np.array([40, 40, 214])).astype(np.uint8)
    v[keep] = (0, 200, 0)
    return v[..., ::-1]


def save(fig, name, data=None):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    if data is not None:
        data.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)


def removed_union(n):
    m = None
    for k in ("structure_segment", "structure_large", "structure_edge", "blob", "speck", "no_ridge"):
        x = mask(n, f"removed_{k}.png")
        if x is not None:
            m = x if m is None else m | x
    return m


def stages_figure(n, label, fname, title_prefix):
    o = orig(n)
    g = pd.read_csv(src(INF / n / "groups.csv")) if (INF / n / "groups.csv").stat().st_size > 5 else pd.DataFrame()
    fig, axes = plt.subplots(2, 4, figsize=FS)
    a = axes.ravel()
    show(a[0], o[..., ::-1], "1 original steel plate (RGB)")
    show(a[1], mask(n, "bright_candidate.png"), "2a bright candidate (chalk)", True)
    show(a[2], mask(n, "dark_candidate.png"), "2b dark candidate (marker / shadow)", True)
    show(a[3], mask(n, "lowcontrast_candidate.png"), "2c low-contrast candidate (faint)", True)
    show(a[4], mask(n, "combined_candidate.png"), "3 combined candidate", True)
    rem = removed_union(n)
    rv = np.zeros((*o.shape[:2], 3), np.uint8)
    if rem is not None:
        rv[rem] = (230, 40, 40)
    acc = mask(n, "accepted_groups.png")
    if acc is not None:
        rv[acc] = (230, 230, 230)
    show(a[5], rv, "4 pixel gating: white kept, red removed\n(structure / blob / speck / no ridge)")
    fin = mask(n, "final_mask.png")
    show(a[6], fin, f"6 final mask ({int(fin.sum())} px)", True)
    show(a[7], ov(o, fin, mask(n, "combined_candidate.png") & ~fin), "7 final overlay (RGB): green kept, red removed")
    rf = "; ".join(f"g{int(r.group_id)}: p_RF {r.p_RF_random_forest:.2f} -> {'keep' if r.rf_accept else 'REJECT'}" for r in g.itertuples()) if len(g) else "no groups"
    fig.suptitle(f"{title_prefix}  {label} - Phase 4 Recovery (saved masks). Stage 5 (7 features + RF auxiliary decision) logged per group: {rf[:160]}"
                 + ("..." if len(rf) > 160 else "") + "\nThe recovery pipeline has no separate M5 stroke-refinement step; pixel-level cleaning = stage 4 gating.", fontsize=8.5)
    save(fig, fname, g)
    return g


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stages").mkdir(exist_ok=True)

    # ---------- P-S1 overall pipeline (thumbnails = saved example1 stages)
    n = "field__예시1"; o = orig(n)
    steps = [("1 Original\nsteel plate", o[..., ::-1]),
             ("2 Candidate generation\nbright + dark + low-contrast\n(local-median contrast AND\nstroke-scale top/black-hat)", mask(n, "combined_candidate.png")),
             ("3 Pixel gating\nstructure lines / large-object\nedges / blobs / specks /\nno two-sided ridge removed", mask(n, "accepted_groups.png")),
             ("4 Grouping + 7 shape features\nCV_w CV_h CV_g S_theta B CV_A R\n-> RF P(handwritten)\n(auxiliary: can only reject)", None),
             ("5 Final binary mask\n(stroke pixels only)", mask(n, "final_mask.png")),
             ("6 Overlay (RGB)\ngreen = extracted strokes", ov(o, mask(n, "final_mask.png")))]
    fig, ax = plt.subplots(1, 6, figsize=(13.333, 4.6))
    for k, (a, (t, im)) in enumerate(zip(ax, steps)):
        if im is None:
            g = pd.read_csv(INF / n / "groups.csv")
            txt = "\n".join(f"group {int(r.group_id)}: {int(r.pixels)} px\nCV_h {r.CV_h:.2f}  S_theta {r.S_theta:.1f}\nR {r.R:.3f}  CV_A {r.CV_A:.2f}\np_RF = {r.p_RF_random_forest:.2f} -> {'keep' if r.rf_accept else 'reject'}\nreliability {r.reliability}\n"
                            for r in g.itertuples())
            a.text(0.5, 0.5, txt.replace("nan", "missing"), ha="center", va="center", fontsize=7, family="monospace", bbox=dict(fc="#eef4fb", ec=MID))
            a.axis("off")
            a.text(0.5, 0.80, t, ha="center", va="bottom", fontsize=8.5, color=MID, transform=a.transAxes)
            t = ""
        else:
            show(a, im, "")
        a.set_title(t, fontsize=8.5, color=MID)
        if k < 5:
            a.annotate("", xy=(1.12, 0.5), xytext=(1.0, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2, color=BASE))
    fig.suptitle("P-S1  PAC2 Phase 4 Recovery pipeline (order verified in scripts/phase4_recovery.py). Thumbnails = saved example1 outputs. "
                 "Models are reused from Phase 3 (no retraining).", fontsize=10)
    save(fig, "P-S1_overall_pipeline")

    # ---------- P-S2 example1 evolution (incl. R1 bright candidate)
    o = orig(n)
    bf = mask(n, "final_mask.png", base=True)
    fig, ax = plt.subplots(2, 4, figsize=FS)
    a = ax.ravel()
    show(a[0], o[..., ::-1], "original (RGB)")
    show(a[1], bf, "BEFORE: baseline final\n(Phase 4 refined M5, one polarity 'dark')", True)
    show(a[2], ov(o, bf), "BEFORE overlay: dark plate between\nchalk strokes + seam extracted")
    show(a[3], mask(n, "dark_candidate.png"), "recovery: dark candidate\n(what the old one-polarity rule saw)", True)
    show(a[4], mask(n, "bright_candidate.png"), "R1 / P01: BRIGHT candidate (chalk)\n= intermediate stage, not final accuracy", True)
    rem = removed_union(n); rv = np.zeros((*o.shape[:2], 3), np.uint8); rv[rem] = (230, 40, 40); rv[mask(n, "accepted_groups.png")] = (230, 230, 230)
    show(a[5], rv, "gating: seam removed as structure (red)")
    fin = mask(n, "final_mask.png")
    show(a[6], fin, f"AFTER: recovery final mask ({int(fin.sum())} px)", True)
    show(a[7], ov(o, fin), "AFTER overlay (RGB): chalk strokes kept")
    fig.suptitle("P-S2  example1 processing evolution: one-polarity baseline -> dual-polarity recovery. Saved masks only "
                 "(baseline: reports/phase4_refined, recovery: reports/phase4_recovery).", fontsize=10)
    save(fig, "P-S2_example1_evolution")

    # ---------- P-S3 faint chalk (field 2, R2 crop) and dark marker (field 1, R4)
    fig, ax = plt.subplots(2, 4, figsize=FS)
    y0, y1, x0, x1 = 90, 240, 40, 240
    o2 = orig("field__2")
    c = lambda m: m[y0:y1, x0:x1]  # noqa: E731
    show(ax[0, 0], c(o2)[..., ::-1], "field 2 original (crop, RGB)")
    show(ax[0, 1], c(mask("field__2", "candidate_mask.png", base=True)), "BEFORE: baseline candidates\n(chalk lost at candidate stage)", True)
    show(ax[0, 2], c(mask("field__2", "lowcontrast_candidate.png") | mask("field__2", "bright_candidate.png")), "recovery: bright + low-contrast\ncandidates", True)
    show(ax[0, 3], c(mask("field__2", "final_mask.png")), "AFTER = R2 / P02: recovery final mask", True)
    o1 = orig("field__1")
    fin1 = mask("field__1", "final_mask.png")
    dk = np.zeros((*o1.shape[:2], 3), np.uint8)
    dk[fin1 & mask("field__1", "bright_candidate.png")] = (255, 255, 255); dk[fin1 & mask("field__1", "dark_candidate.png")] = (255, 140, 0)
    show(ax[1, 0], o1[..., ::-1], "field 1 original (RGB)")
    show(ax[1, 1], mask("field__1", "final_mask.png", base=True), "BEFORE: baseline final\n(one polarity 'bright' -> marker missed)", True)
    show(ax[1, 2], dk, "recovery final split by polarity:\nwhite = bright ink, orange = dark marker")
    show(ax[1, 3], ov(o1, fin1), "AFTER = R4 / P05: recovery overlay")
    fig.suptitle("P-S3  Faint chalk (field 2, crop x40-240 y90-240) and dark marker (field 1). Same size per row. Remaining errors: see P-S5.", fontsize=10)
    save(fig, "P-S3_faint_chalk_dark_marker")

    # ---------- stage figures for the 5 priority images
    for n_, lab in (("field__예시1", "example1"), ("field__1", "field 1"), ("field__2", "field 2"), ("field__weld_marking_W79", "W79"), ("dev__handwritten_013_V8", "V8")):
        stages_figure(n_, lab, f"stages/{lab.replace(' ', '_')}_stages", "Stage view")

    # ---------- P-S4 mathematical learning -> extraction
    g1 = pd.read_csv(src(ROOT / "results" / "phase3" / "tables" / "G1_loss_convergence.csv"))
    cs = pd.read_csv(src(ROOT / "results" / "phase3" / "tables" / "coefficient_stability.csv"))
    cs = cs[cs.model == "C_shape_domain_matched|train_all|torch_logistic"].set_index("feature")
    h3 = pd.read_csv(src(ROOT / "results" / "progress_evolution_refined" / "H3_optimization_progress_summary.csv"))
    mc = pd.read_csv(src(ROOT / "results" / "phase3" / "tables" / "model_comparison.csv")).set_index("model")
    rf_ba = float(mc.loc["C_shape_domain_matched|train_all|random_forest", "oof_balanced_accuracy"])
    fig = plt.figure(figsize=FS)
    gs = fig.add_gridspec(2, 3)
    a0 = fig.add_subplot(gs[0, 0]); a0.axis("off")
    a0.text(0, 1, "7 shape features (PDF numbering)\n1 CV_w  stroke-width variation\n2 CV_h  character-height variation\n4 CV_g  gap variation\n"
            "5 S_theta slant variation (deg)\n6 B     baseline wobble\n10 CV_A component-area variation\n11 R    contour irregularity\n\n"
            "p = sigmoid(b + sum w_j z_j)\nL = class-weighted BCE + lambda ||w||^2", va="top", fontsize=8.3, family="monospace")
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(g1.epoch, g1.train_loss_mean, color=MID, lw=2, label="train"); a1.plot(g1.epoch, g1.val_loss_mean, color=FINAL, lw=2, ls="--", label="validation")
    a1.set(xlabel="epoch", ylabel="loss (5-fold mean)", title=f"Loss {g1.train_loss_mean.iloc[0]:.3f} -> {g1.train_loss_mean.iloc[-1]:.3f} (train)"); a1.legend(fontsize=8)
    a2 = fig.add_subplot(gs[0, 2])
    order = ["CV_w", "CV_h", "CV_g", "S_theta", "B", "CV_A", "R"]
    a2.barh(order, cs.loc[order, "mean"], xerr=cs.loc[order, "std"], color=[FINAL if v > 0 else ERR for v in cs.loc[order, "mean"]], edgecolor="black")
    a2.axvline(0, color="black", lw=0.8); a2.set(xlabel="learned weight (z-space, 5-fold mean ± SD)", title="Feature weights (S3)")
    a3 = fig.add_subplot(gs[1, 0])
    a3.plot(h3.stage, h3.BA, marker="o", color=FINAL, lw=2.5)
    for xx, v in zip(h3.stage, h3.BA):
        a3.annotate(f"{v:.3f}", (xx, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    a3.set(ylim=(0.6, 1.05), ylabel="balanced accuracy (OOF)", title=f"Font vs handwriting classification (RF {rf_ba:.3f})")
    a4 = fig.add_subplot(gs[1, 1:]); a4.axis("off")
    g = pd.read_csv(INF / "field__예시1" / "groups.csv")
    a4.text(0, 1, "How the classifier is used on steel (Phase 4 Recovery):\n"
            "  candidate strokes -> groups -> the same 7 features -> RF P(handwritten)\n"
            "  RF can only REJECT a group (p < recall-priority threshold); it never re-admits removed pixels.\n"
            f"  example1 logged: " + ", ".join(f"group {int(r.group_id)} p_RF={r.p_RF_random_forest:.2f}" for r in g.itertuples()) + "\n"
            "  In the 3 GT images the RF rejected 0 % of handwriting pixels (IV2): extraction quality is decided by\n"
            "  candidate generation and pixel gating, not by the classifier.\n\n"
            "Not the same quantity: BA 0.93-0.99 = rendered font vs photographed handwriting (development, 90 crops).\n"
            "Steel pixel Dice 0.14-0.60 = stroke-pixel overlap on 3 field images (see P-S5).", va="top", fontsize=9.5)
    fig.suptitle("P-S4  From the mathematical model (7 features, weighted BCE + L2) to extraction. Values read from G1 / coefficient_stability / H3 CSVs.", fontsize=10)
    save(fig, "P-S4_math_to_extraction")

    # ---------- P-S5 results and limitations (GT metrics verified)
    m = pd.read_csv(src(ROOT / "results" / "industrial_validation" / "gt_metrics.csv")); m["image"] = m.image.astype(str)
    reported = {"예시1": (0.491, 0.756, 0.595), "2": (0.078, 0.518, 0.136), "1": (0.249, 0.549, 0.343)}
    ver = []
    for s, (p, r, d) in reported.items():
        row = m[m.image == s].iloc[0]
        ver.append({"image": {"예시1": "example1", "2": "field 2", "1": "field 1"}[s], "precision": row.stable_final_precision, "recall": row.stable_final_recall,
                    "dice": row.stable_final_dice, "iou": row.stable_final_iou, "fp": row.stable_final_fp, "fn": row.stable_final_fn,
                    "matches_reported": all(abs(a_ - b_) < 5e-4 for a_, b_ in ((row.stable_final_precision, p), (row.stable_final_recall, r), (row.stable_final_dice, d)))})
    vt = pd.DataFrame(ver)
    fig = plt.figure(figsize=FS)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])
    GT = ROOT / "data" / "field_test" / "masks"
    for k, (s, lab) in enumerate((("예시1", "example1 - SUCCESS (chalk)"), ("2", "field 2 - structure FP"), ("1", "field 1 - marker ok, chalk missed"))):
        nn = f"field__{s}"; o = orig(nn); fin = mask(nn, "final_mask.png"); gt = rd(src(GT / f"{s}_gt.png"), 0) > 0
        e = (o // 2).copy(); e[fin & gt] = (0, 200, 0); e[fin & ~gt] = (40, 40, 230); e[~fin & gt] = (0, 170, 255)
        show(fig.add_subplot(gs[0, k]), e[..., ::-1], f"{lab}\ngreen TP, red FP, orange FN")
    a = fig.add_subplot(gs[0, 3]); a.axis("off")
    a.text(0, 0.95, "Exact-pixel metrics vs hand-drawn GT\n(3 DEVELOPMENT images, stable)\n\n" +
           "\n".join(f"{r.image:9s} P {r.precision:.3f} R {r.recall:.3f}\n          Dice {r.dice:.3f} IoU {r.iou:.3f}" for r in vt.itertuples()) +
           f"\n\nall match reported values: {bool(vt.matches_reported.all())}", va="top", fontsize=9, family="monospace")
    b = fig.add_subplot(gs[1, :2])
    x = np.arange(3)
    for j, (k_, col) in enumerate((("precision", BASE), ("recall", MID), ("dice", FINAL), ("iou", "#17becf"))):
        bb = b.bar(x + (j - 1.5) * 0.2, vt[k_], 0.2, color=col, edgecolor="black", label=k_)
        for q, v in zip(bb, vt[k_]):
            b.text(q.get_x() + q.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    b.set_xticks(x, vt.image); b.set_ylim(0, 1); b.legend(fontsize=8, ncol=4); b.set_title("pixel metrics (stable, development images)", fontsize=9)
    c_ = fig.add_subplot(gs[1, 2:]); c_.axis("off")
    c_.text(0, 1, "Limitations\n- handwriting lost mainly at CANDIDATE GENERATION (21-35 % of GT pixels);\n  example1 misses are mostly 1-px boundary width (recall 0.986 at 1 px tolerance)\n"
            "- precision limited by structures: machinery, plate edges, seams, floor\n- faint / coloured chalk ('Angle face', '350/490', yellow 'do5') partly missed\n"
            "- thick marker strokes (V8) only partly captured\n- 2 post-processing trials (Fix2, structure_v2) rejected: they erased handwriting\n"
            "- no independent field images: field generalization UNVERIFIED", va="top", fontsize=9)
    fig.suptitle("P-S5  Results and limitations. GT = 3 hand-labelled development images (used during tuning) - not independent field performance.", fontsize=10)
    save(fig, "P-S5_results_limitations", vt)
    (OUT / "sources_used.txt").write_text("\n".join(sorted(set(LOG))), encoding="utf-8")
    print(vt.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
