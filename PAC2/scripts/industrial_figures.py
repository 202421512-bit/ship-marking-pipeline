"""IV1-IV5 from saved masks / CSVs only (no inference). All GT images are DEVELOPMENT images (n = 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2.visualization import BASE, ERR, FINAL, MID, plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "industrial_validation"
ST = ROOT / "versions" / "stable_baseline" / "outputs"
FX = ROOT / "reports" / "industrial_fixes"
GT = ROOT / "data" / "field_test" / "masks"
TAG = "GT images: 3, all DEVELOPMENT (used for tuning) - not independent field performance"
rd = lambda p, f=0: cv2.imdecode(np.fromfile(str(p), np.uint8), f)  # noqa: E731
IMGS = [("예시1", "example1"), ("2", "field 2"), ("1", "field 1")]


def save(fig, name, data):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    data.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)


def err(o, pred, gt):
    v = (o // 2).copy()
    v[pred & gt] = (0, 200, 0); v[pred & ~gt] = (40, 40, 230); v[~pred & gt] = (0, 170, 255)
    return v[..., ::-1]


def main() -> int:
    m = pd.read_csv(OUT / "gt_metrics.csv"); fx = pd.read_csv(OUT / "fix_metrics.csv")
    m["image"] = m.image.astype(str)
    # IV1
    fig, axes = plt.subplots(3, 5, figsize=(13.333, 8))
    rows = []
    for row, (stem, lab) in zip(axes, IMGS):
        n = f"field__{stem}"
        o = rd(ST / n / "original.png", 1); gt = rd(GT / f"{stem}_gt.png") > 0
        st = rd(ST / n / "final_mask.png") > 0; f1 = rd(FX / n / "fix1_final.png") > 0
        g = o.copy(); g[gt] = (0, 255, 255)
        r = fx[fx.image == n].iloc[0]
        for a, im, t in zip(row, (o[..., ::-1], g[..., ::-1], err(o, st, gt), err(o, f1, gt), err(o, rd(FX / n / "fix2_final.png") > 0, gt)),
                            ("original", "ground truth (yellow)", f"STABLE  Dice {r.stable_dice:.3f}", f"Fix1 (E1)  Dice {r.fix1_dice:.3f}",
                             f"Fix2 (rejected)  Dice {r.fix2_dice:.3f}")):
            a.imshow(im, interpolation="nearest"); a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
        row[0].set_ylabel(lab, fontsize=10)
        rows.append(r.to_dict())
    fig.suptitle(f"IV1  Error overlay: green = correct stroke (TP), red = false positive, orange = missed handwriting (FN).  {TAG}", fontsize=10)
    save(fig, "IV1_error_overlay", pd.DataFrame(rows))
    # IV2 candidate vs final recall (stage losses)
    fig, ax = plt.subplots(figsize=(13.333, 6))
    x = np.arange(3); labs = [l for _, l in IMGS]
    mm = m.set_index("image").loc[[s for s, _ in IMGS]]
    for k, (c, col, name, h) in enumerate((("candidate_recall", BASE, "candidate recall", None), ("gated_recall", MID, "after structure / stroke gating", "//"),
                                           ("stable_final_recall", FINAL, "final (after RF reject)", "\\\\"))):
        b = ax.bar(x + (k - 1) * 0.27, mm[c], 0.27, color=col, hatch=h, edgecolor="black", label=name)
        for bb, v in zip(b, mm[c]):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, labs); ax.set_ylim(0, 1.05); ax.set(ylabel="pixel recall of GT handwriting (exact pixels)", title="IV2  Where handwriting is lost (stable)")
    ax.legend()
    fig.suptitle(f"{TAG}. Lost at generation: " + ", ".join(f"{l} {v:.0%}" for l, v in zip(labs, mm.lost_at_generation)) +
                 "; RF reject removed 0 %. With 1 px tolerance example1 recall = 0.986 (GT brush 4 px vs predicted 2.8 px).", fontsize=9)
    save(fig, "IV2_candidate_vs_final_recall", mm.reset_index()[["image", "candidate_recall", "gated_recall", "stable_final_recall", "lost_at_generation", "lost_at_gating", "lost_at_rf_reject"]])
    # IV3 metrics
    fig, axes = plt.subplots(1, 4, figsize=(13.333, 5), sharey=True)
    data = []
    for a, k in zip(axes, ("precision", "recall", "dice", "iou")):
        for j, (tag, col, h) in enumerate((("stable", BASE, None), ("fix1", FINAL, "//"), ("fix2", ERR, "xx"))):
            v = [float(fx[fx.image == f"field__{s}"][f"{tag}_{k}"].iloc[0]) for s, _ in IMGS]
            b = a.bar(x + (j - 1) * 0.27, v, 0.27, color=col, hatch=h, edgecolor="black", label={"stable": "stable", "fix1": "Fix1 (E1)", "fix2": "Fix2 (rejected)"}[tag])
            for bb, vv in zip(b, v):
                a.text(bb.get_x() + bb.get_width() / 2, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=7, rotation=90)
            data += [{"metric": k, "version": tag, "image": l, "value": vv} for l, vv in zip(labs, v)]
        a.set_xticks(x, labs, fontsize=8); a.set_title(k, fontsize=11); a.set_ylim(0, 1.1)
    axes[0].legend(fontsize=8); axes[0].set_ylabel("pixel metric")
    fig.suptitle(f"IV3  Pixel precision / recall / Dice / IoU per image.  {TAG}", fontsize=10)
    save(fig, "IV3_pixel_metrics", pd.DataFrame(data))
    # IV4 FP suppression vs stroke preservation
    fig, ax = plt.subplots(figsize=(13.333, 6))
    data = []
    for tag, col, mk in (("stable", BASE, "o"), ("fix1", FINAL, "s"), ("fix2", ERR, "^")):
        for (s, l) in IMGS:
            r = fx[fx.image == f"field__{s}"].iloc[0]
            ax.scatter(r[f"{tag}_fp"], r[f"{tag}_recall"], c=col, marker=mk, s=120, edgecolors="black")
            ax.annotate(f"{l} ({tag})", (r[f"{tag}_fp"], r[f"{tag}_recall"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
            data.append({"image": l, "version": tag, "fp_px": r[f"{tag}_fp"], "recall": r[f"{tag}_recall"]})
    ax.set(xlabel="false-positive pixels (lower = fewer structures / rust kept)", ylabel="handwriting pixel recall", ylim=(0, 1),
           title="IV4  False-positive suppression vs true-stroke preservation (GT images)")
    nongt = fx[~fx.has_gt]
    fig.suptitle(f"{TAG}. Fix2 keeps recall on GT images but removes handwriting in W79 (no GT; '9' and right-edge marks, {int(nongt[nongt.image == 'field__weld_marking_W79'].fix2_removed_vs_fix1_px.iloc[0])} px) -> rejected.", fontsize=9)
    save(fig, "IV4_fp_vs_preservation", pd.DataFrame(data))
    # IV5 best cases and remaining failures (crops)
    cases = [("예시1", (20, 40, 150, 140), "BEST: example1 chalk 'F55' kept, seam removed (Fix1/stable)"),
             ("2", (40, 90, 240, 240), "PARTIAL: field 2 faint chalk - 'Angle face' strokes partly missed (orange)"),
             ("2", (0, 0, 343, 110), "FAILURE: field 2 machinery kept as handwriting (red), even after Fix1"),
             ("1", (100, 140, 300, 260), "PARTIAL: field 1 black marker kept; yellow-green chalk 'do5' / '835' missed (orange)"),
             ("1", (0, 60, 80, 220), "FAILURE: field 1 chalk '350 / 490' missed, plate edge kept (red)")]
    fig, axes = plt.subplots(len(cases), 3, figsize=(13.333, 14))
    for row, (stem, (x0, y0, x1, y1), txt) in zip(axes, cases):
        n = f"field__{stem}"
        o = rd(ST / n / "original.png", 1); gt = rd(GT / f"{stem}_gt.png") > 0; f1 = rd(FX / n / "fix1_final.png") > 0
        g = o.copy(); g[gt] = (0, 255, 255)
        for a, im, t in zip(row, (o[y0:y1, x0:x1, ::-1], g[y0:y1, x0:x1, ::-1], err(o, f1, gt)[y0:y1, x0:x1]), ("original crop", "ground truth", "Fix1 = stable+E1 errors")):
            a.imshow(im, interpolation="nearest"); a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([])
        row[0].set_ylabel(txt, fontsize=8, rotation=0, ha="right", va="center", color=ERR if "FAILURE" in txt else "black")
    fig.suptitle(f"IV5  Best recovery cases and remaining failures (green TP, red FP, orange FN).  {TAG}", fontsize=10)
    save(fig, "IV5_best_and_failures", pd.DataFrame([{"image": c[0], "crop": c[1], "note": c[2]} for c in cases]))
    print("IV1-IV5 written to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
