"""Phase 2: candidate generation + 7 features on the 90 DEVELOPMENT images only (test images are not opened).

Writes:
  data/splits/split_manifest.csv              frozen 90/30 split + 5 dev folds (seed 42)
  reports/phase2/features_dev.csv             7 features x candidate methods, missing flags + reasons
  reports/phase2/missing_summary.csv          missing rate per feature / method / class
  reports/phase2/dev_feature_summary.csv      descriptive medians per class (dev only, no fitting)
  reports/phase2/candidates_*.png             candidate masks per method (display only)
  reports/phase2/summary.txt
Run: .venv\\Scripts\\python.exe scripts\\phase2_features.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2 import ROOT, load_config  # noqa: E402
from pac2.candidates import candidates, ink_polarity, read_bgr  # noqa: E402
from pac2.features import FEATURES, NUMBER, extract  # noqa: E402
from pac2.split import build_manifest  # noqa: E402

OUT = ROOT / "reports" / "phase2"


def main() -> int:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(cfg, ROOT / "data" / "splits" / "split_manifest.csv")
    dev = [r for r in manifest if r["set"] == "dev"]
    print(f"split: dev {len(dev)} (" + ", ".join(f"{c}={sum(r['class'] == c for r in dev)}" for c in cfg["data"]["classes"]) +
          f"), test {sum(r['set'] == 'test' for r in manifest)} locked")
    rows, panels = [], {m: [] for m in cfg["candidates"]["methods"]}
    for r in sorted(dev, key=lambda r: r["image_id"]):
        bgr = read_bgr(ROOT / r["path"])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cands = candidates(bgr, cfg["candidates"])
        for m, mask in cands.items():
            f = extract(mask, cfg["features"])
            rows.append({"image_id": r["image_id"], "class": r["class"], "label": r["label"], "fold": r["fold"], "method": m,
                         "polarity": ink_polarity(gray), "width": bgr.shape[1], "height": bgr.shape[0], **f})
            if len(panels[m]) < 24 and (r["class"] == "formal" or len(panels[m]) % 2 == 0 or len(panels[m]) < 24):
                panels[m].append((r["image_id"], bgr, mask))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "features_dev.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    miss, desc, lines = [], [], []
    for m in cfg["candidates"]["methods"]:
        for cls in cfg["data"]["classes"]:
            rs = [r for r in rows if r["method"] == m and r["class"] == cls]
            mr = {"method": m, "class": cls, "n": len(rs), "median_components": float(np.median([r["n_components"] for r in rs]))}
            dr = {"method": m, "class": cls, "n": len(rs)}
            for k in FEATURES:
                v = np.array([r[k] for r in rs], float)
                mr[f"{NUMBER[k]}_{k}_missing_rate"] = round(float(np.isnan(v).mean()), 3)
                dr[f"{NUMBER[k]}_{k}_median"] = round(float(np.nanmedian(v)), 4) if np.isfinite(v).any() else ""
                dr[f"{NUMBER[k]}_{k}_iqr"] = round(float(np.nanpercentile(v, 75) - np.nanpercentile(v, 25)), 4) if np.isfinite(v).any() else ""
            miss.append(mr)
            desc.append(dr)
    for name, data in (("missing_summary.csv", miss), ("dev_feature_summary.csv", desc)):
        with open(OUT / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    # candidate panels: 12 formal + 12 handwritten per method (original | mask)
    for m in cfg["candidates"]["methods"]:
        sel = [p for p in panels[m] if p[0].startswith("formal")][:8] + [p for p in panels[m] if p[0].startswith("handwritten")][:16]
        tiles = []
        for iid, bgr, mask in sel:
            s = 140 / bgr.shape[0]
            a = cv2.resize(bgr, (int(bgr.shape[1] * s), 140))
            b = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (int(bgr.shape[1] * s), 140), interpolation=cv2.INTER_NEAREST)
            t = np.hstack([a, np.full((140, 4, 3), 255, np.uint8), b])
            t = cv2.copyMakeBorder(t, 18, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, iid.split("/")[1][:-4], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
            tiles.append(cv2.resize(t, (520, 162)))
        while len(tiles) % 4:
            tiles.append(np.full((162, 520, 3), 255, np.uint8))
        sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
        cv2.imencode(".png", sheet)[1].tofile(str(OUT / f"candidates_{m}.png"))

    prim = cfg["candidates"]["primary"]
    lines.append(f"DEV ONLY ({len(dev)} images); test images not opened. Primary candidate method (configs/default.yaml, change reason recorded there): {prim}")
    for mr in miss:
        lines.append(f"{mr['method']:16s} {mr['class']:11s} median CC {mr['median_components']:5.1f} | missing: " +
                     ", ".join(f"{k}={mr[f'{NUMBER[k]}_{k}_missing_rate']:.2f}" for k in FEATURES))
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
