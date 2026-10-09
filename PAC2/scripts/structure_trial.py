"""Run structure_suppression_v2 once on all stable outputs; evaluate with GT where available; regression figures.
Stable / Fix1 / Fix2 / P01-P08 untouched. Output: results/structure_suppression_trial/, masks in reports/structure_suppression_trial/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2.structure_suppression_v2 import CFG, suppress  # noqa: E402
from pac2.visualization import plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ST = ROOT / "versions" / "stable_baseline" / "outputs"
GT = ROOT / "data" / "field_test" / "masks"
RAW = ROOT / "reports" / "structure_suppression_trial"
OUT = ROOT / "results" / "structure_suppression_trial"
rd = lambda p, f=0: cv2.imdecode(np.fromfile(str(p), np.uint8), f)  # noqa: E731
u8 = lambda m: (m > 0).astype(np.uint8) * 255  # noqa: E731
# known weak spots to inspect (image coordinates, x0, y0, x1, y1) - fixed before running
WEAK = {"field__weld_marking_W79": [("'W#79' box and digits", (40, 160, 200, 260)), ("'9' bottom right", (440, 690, 509, 774))],
        "dev__handwritten_013_V8": [("V8 junctions", (0, 0, 9999, 9999))],
        "field__2": [("faint chalk", (40, 90, 240, 240))]}


def met(p, g):
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "dice": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn), "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, weak_rows = [], []
    for d in sorted(ST.iterdir()):
        n = d.name
        o = rd(d / "original.png", 1); fin = rd(d / "final_mask.png") > 0; se = rd(d / "structure_edges.png") > 0
        r = suppress(fin, se)
        o_dir = RAW / n; o_dir.mkdir(parents=True, exist_ok=True)
        for k in ("final", "removed", "uncertain", "protect"):
            cv2.imencode(".png", u8(r[k]))[1].tofile(str(o_dir / f"v2_{k}.png"))
        pd.DataFrame([{**{k: v for k, v in s.items() if k != "seg"}, "seg": list(s["seg"])} for s in r["segments"]]).to_csv(o_dir / "segments.csv", index=False, encoding="utf-8-sig")
        stem = n.split("__")[1]
        row = {"image": n, "stable_px": int(fin.sum()), "v2_px": int(r["final"].sum()), "removed_px": int(r["removed"].sum()),
               "uncertain_px": int(r["uncertain"].sum()), "segments": len(r["segments"]),
               "structural_removed_segments": sum(s["status"] == "removed" for s in r["segments"]),
               "uncertain_segments": sum(s["status"].startswith("UNCERTAIN") for s in r["segments"])}
        g = (GT / f"{stem}_gt.png")
        if g.exists():
            gt = rd(g) > 0
            ms, mv = met(fin, gt), met(r["final"], gt)
            row.update({f"stable_{k}": v for k, v in ms.items()}); row.update({f"v2_{k}": v for k, v in mv.items()})
            row.update(fp_reduction=ms["fp"] - mv["fp"], tp_loss=ms["tp"] - mv["tp"], recall_change=mv["recall"] - ms["recall"], dice_change=mv["dice"] - ms["dice"])
        for lab, (x0, y0, x1, y1) in WEAK.get(n, []):
            reg = np.zeros_like(fin); reg[y0:y1, x0:x1] = True
            weak_rows.append({"image": n, "weak_spot": lab, "stable_px_in_region": int((fin & reg).sum()), "removed_px_in_region": int((r["removed"] & reg).sum())})
        rows.append(row)
        diff = np.zeros((*fin.shape, 3), np.uint8); diff[r["final"]] = (200, 200, 200); diff[r["removed"]] = (230, 40, 40); diff[r["uncertain"]] = (255, 200, 0)
        panels = [(o[..., ::-1], "original"), (u8(fin), f"stable {int(fin.sum())} px"), (u8(r["final"]), f"v2 {int(r['final'].sum())} px"),
                  (diff, f"removed (red) {int(r['removed'].sum())} px, uncertain kept (yellow) {int(r['uncertain'].sum())} px")]
        if g.exists():
            ev = (o // 2).copy(); ev[r["final"] & gt] = (0, 200, 0); ev[r["final"] & ~gt] = (40, 40, 230); ev[~r["final"] & gt] = (0, 170, 255)
            panels.append((ev[..., ::-1], f"v2 vs GT: Dice {row['v2_dice']:.3f} (stable {row['stable_dice']:.3f})"))
        fig, ax = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6))
        for a, (im, t) in zip(ax, panels):
            a.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest"); a.set_title(t, fontsize=7.5); a.axis("off")
        fig.suptitle(f"{n.replace('예시', 'example')} - structure_suppression_v2 (single proposal, GT not used in inference)", fontsize=9)
        nm = n.replace("예시", "example")
        fig.savefig(OUT / f"{nm}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / f"{nm}.svg", bbox_inches="tight"); plt.close(fig)
    t = pd.DataFrame(rows); w = pd.DataFrame(weak_rows)
    t.to_csv(OUT / "trial_metrics.csv", index=False, encoding="utf-8-sig"); w.to_csv(OUT / "weak_spot_check.csv", index=False, encoding="utf-8-sig")
    print(t.round(3).to_string(index=False)); print(w.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
