"""Frozen 90 / 30 split (seed from config). Test images are locked until the final model is fixed.

No scene/origin metadata exists: each image is its own group (reported as a limitation). If group ids are added
later, splitting must be redone at group level before any training.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from . import ROOT


def build_manifest(cfg: dict, out_path: Path) -> list:
    if out_path.exists():                       # frozen: never regenerated silently
        with open(out_path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for cls, label in cfg["data"]["classes"].items():
        files = sorted((ROOT / cfg["data"]["source_dir"] / cls).glob("*.png"))
        idx = rng.permutation(len(files))
        n_test = cfg["split"]["test_per_class"][cls]
        for rank, i in enumerate(idx):
            rows.append({"image_id": f"{cls}/{files[i].name}", "path": str(files[i].relative_to(ROOT)), "class": cls,
                         "label": label, "group_id": f"{cls}/{files[i].stem}", "set": "test" if rank < n_test else "dev"})
    dev = [r for r in rows if r["set"] == "dev"]
    skf = StratifiedKFold(cfg["split"]["dev_folds"], shuffle=True, random_state=cfg["seed"])
    for k, (_, va) in enumerate(skf.split(np.zeros(len(dev)), [r["label"] for r in dev])):
        for i in va:
            dev[i]["fold"] = k
    for r in rows:
        r.setdefault("fold", "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "path", "class", "label", "group_id", "set", "fold"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image_id"]))
    return rows
