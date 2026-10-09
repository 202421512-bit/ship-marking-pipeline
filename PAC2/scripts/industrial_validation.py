"""Industrial validation: ground-truth pixel metrics for the frozen stable pipeline (and an experimental variant if given).

Refuses to compute any accuracy without ground truth: data/field_test/masks/<stem>_gt.png (255 = handwritten stroke),
optional <stem>_ignore.png, optional split.csv (stem,role with role = dev | eval; default dev).
Stage-separated metrics on the frozen stable outputs (versions/stable_baseline/outputs/<image>/):
  candidate recall  : combined_candidate (both polarities + low contrast)
  gated recall      : accepted_groups (after structure / blob / speck / ridge gating)
  final             : final_mask (after the RF auxiliary reject)
=> where handwriting is lost: generation (1 - candidate recall), gating (candidate - gated), RF reject (gated - final).
Outputs: results/industrial_validation/ IV1-IV5 (PNG 300 dpi + SVG + CSV) once GT exists.
Run: .venv\\Scripts\\python.exe scripts\\industrial_validation.py [--experimental-dir reports/versions/experimental]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASKS = ROOT / "data" / "field_test" / "masks"
STABLE = ROOT / "versions" / "stable_baseline" / "outputs"
OUT = ROOT / "results" / "industrial_validation"
rd = lambda p: (cv2.imdecode(np.fromfile(str(p), np.uint8), 0) > 0) if Path(p).exists() else None  # noqa: E731


def metrics(pred, gt, ign):
    v = ~ign
    g, p = gt & v, pred & v
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    return {"recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp), "dice": 2 * tp / max(1, 2 * tp + fp + fn),
            "iou": tp / max(1, tp + fp + fn), "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experimental-dir", default=None)
    a = ap.parse_args()
    gts = sorted(MASKS.glob("*_gt.png")) if MASKS.exists() else []
    if not gts:
        print("NO GROUND TRUTH: data/field_test/masks/<stem>_gt.png not found. Accuracy evaluation stopped.\n"
              "Label first:  label_gt.bat   (or .venv\\Scripts\\python.exe scripts\\gt_label_tool.py 2.png)")
        return 2
    roles = {}
    if (MASKS / "split.csv").exists():
        with open(MASKS / "split.csv", encoding="utf-8-sig") as f:
            roles = {r["stem"]: r["role"] for r in csv.DictReader(f)}
    rows = []
    for gp in gts:
        stem = gp.name[:-len("_gt.png")]
        name = f"field__{stem}"
        d = STABLE / name
        gt = rd(gp)
        ign = rd(MASKS / f"{stem}_ignore.png")
        ign = np.zeros_like(gt) if ign is None else ign
        if not d.exists() or gt.sum() == 0:
            print("skip", stem, "(no stable output or empty GT)")
            continue
        r = {"image": stem, "role": roles.get(stem, "dev"), "gt_px": int((gt & ~ign).sum()), "ignore_px": int(ign.sum())}
        for stage, f in (("candidate", "combined_candidate.png"), ("gated", "accepted_groups.png"), ("stable_final", "final_mask.png")):
            m = metrics(rd(d / f), gt, ign)
            r.update({f"{stage}_{k}": v for k, v in m.items()})
        if a.experimental_dir:
            for tag, fn in (("experimental", "final_mask.png"), ("e1only", "e1_only.png")):
                e = rd(ROOT / a.experimental_dir / name / fn)
                if e is not None:
                    r.update({f"{tag}_{k}": v for k, v in metrics(e, gt, ign).items()})
        r["lost_at_generation"] = 1 - r["candidate_recall"]
        r["lost_at_gating"] = r["candidate_recall"] - r["gated_recall"]
        r["lost_at_rf_reject"] = r["gated_recall"] - r["stable_final_recall"]
        rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "gt_metrics.csv", index=False, encoding="utf-8-sig")
    print(t.round(3).to_string(index=False))
    print(f"\nGT images: {len(t)} (dev {int((t.role == 'dev').sum())}, eval {int((t.role == 'eval').sum())}). "
          "dev images were used for tuning -> development performance only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
