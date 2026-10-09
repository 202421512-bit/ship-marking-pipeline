"""Two limited fixes on top of the frozen stable final masks (stable code / outputs untouched), evaluated with GT.

FIX1 = E1-only (already defined in src/pac2/experimental_recovery.py; read from reports/versions/experimental/*/e1_only.png)
FIX2 = FIX1 + remove components that touch the image border (<= 4 px) AND have bounding-box fill >= 0.25 (blob-like).
       Rule derived from the 3 GT development images (no GT handwriting component touched the border) -> dev-fitted.
Writes reports/industrial_fixes/<image>/fix{1,2}_final.png for all 8 images and results/industrial_validation/fix_metrics.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STABLE = ROOT / "versions" / "stable_baseline" / "outputs"
EXP = ROOT / "reports" / "versions" / "experimental"
OUT = ROOT / "reports" / "industrial_fixes"
MASKS = ROOT / "data" / "field_test" / "masks"
FIX2 = {"border_px": 4, "fill_min": 0.25}
rd = lambda p: cv2.imdecode(np.fromfile(str(p), np.uint8), 0) > 0  # noqa: E731


def fix2(m: np.ndarray) -> np.ndarray:
    H, W = m.shape
    b = FIX2["border_px"]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    out = m.copy()
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if (x <= b or y <= b or x + w >= W - b or y + h >= H - b) and a / (w * h) >= FIX2["fill_min"]:
            out[lab == i] = False
    return out


def met(p, gt):
    tp, fp, fn = int((p & gt).sum()), int((p & ~gt).sum()), int((~p & gt).sum())
    return {"recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp), "dice": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn),
            "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    rows = []
    for d in sorted(STABLE.iterdir()):
        st = rd(d / "final_mask.png")
        f1 = rd(EXP / d.name / "e1_only.png")
        f2 = fix2(f1)
        o = OUT / d.name
        o.mkdir(parents=True, exist_ok=True)
        for k, m in (("fix1_final", f1), ("fix2_final", f2)):
            cv2.imencode(".png", (m * 255).astype(np.uint8))[1].tofile(str(o / f"{k}.png"))
        stem = d.name.split("__")[1]
        r = {"image": d.name, "stable_px": int(st.sum()), "fix1_px": int(f1.sum()), "fix2_px": int(f2.sum()),
             "fix2_removed_vs_fix1_px": int((f1 & ~f2).sum()), "has_gt": (MASKS / f"{stem}_gt.png").exists()}
        if r["has_gt"]:
            gt = rd(MASKS / f"{stem}_gt.png")
            for tag, m in (("stable", st), ("fix1", f1), ("fix2", f2)):
                r.update({f"{tag}_{k}": v for k, v in met(m, gt).items()})
        rows.append(r)
    t = pd.DataFrame(rows)
    (ROOT / "results" / "industrial_validation").mkdir(parents=True, exist_ok=True)
    t.to_csv(ROOT / "results" / "industrial_validation" / "fix_metrics.csv", index=False, encoding="utf-8-sig")
    print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
