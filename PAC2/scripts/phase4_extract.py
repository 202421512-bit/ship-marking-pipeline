"""Phase 4 inference: real field images (data/field_test/images/*.png) with the full-dev checkpoints, plus development
demo crops with the checkpoint of the fold that EXCLUDED them (out-of-fold). No training, no Test 30 access.

Outputs: reports/phase4/inference/<field|dev>__<stem>/ (original, preprocessed, candidate/accepted/rejected/final masks,
overlay, handwritten_only, candidate_scores.csv, run_info.json) and reports/phase4/extraction_summary.csv.
Run: .venv\\Scripts\\python.exe scripts\\phase4_extract.py
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2 import ROOT, load_config  # noqa: E402
from pac2.extraction import CKPT_DIR, MODELS, extract_image, load_checkpoint  # noqa: E402

FIELD = ROOT / "data" / "field_test" / "images"
OUT = ROOT / "reports" / "phase4"
DEV_DEMO = {"handwritten/handwritten_002.png": "S6 (Phase 3 failure)", "handwritten/handwritten_009.png": "S16 (Phase 3 failure)",
            "handwritten/handwritten_013.png": "V8 (Phase 3 failure)", "handwritten/handwritten_020.png": "S8 120 (Phase 3 correct)",
            "formal/formal_002_F8.png": "F8 font (Phase 3 correct)"}


def modes_for(ck: dict) -> dict:
    a, b = ck["thresholds"]["recall_priority"], ck["thresholds"]["precision_priority"]
    return {"recall_priority": min(a, b), "precision_priority": max(a, b)}


def main() -> int:
    cfg = load_config()
    sel = json.loads((CKPT_DIR / "selection.json").read_text(encoding="utf-8"))
    primary, dev_ranges = sel["primary_model"], sel["dev_ranges"]
    hw = {"cpu": platform.processor() or platform.machine(), "os": platform.platform(), "python": platform.python_version(),
          "logical_cpus": os.cpu_count(), "note": "single process, CPU only, includes candidate generation + features + 4 models"}
    full = {m: load_checkpoint(m) for m in MODELS}
    with open(ROOT / "data" / "splits" / "split_manifest.csv", encoding="utf-8-sig") as f:
        man = {r["image_id"]: r for r in csv.DictReader(f)}
    oof = pd.read_csv(ROOT / "reports" / "phase3" / "oof_predictions.csv")
    jobs = [("field", p, full, None) for p in sorted(FIELD.glob("*.png"))]
    for iid, desc in DEV_DEMO.items():
        r = man[iid]
        assert r["set"] == "dev"
        k = int(r["fold"])
        jobs.append(("dev", ROOT / r["path"], {m: load_checkpoint(m, CKPT_DIR / f"fold{k}") for m in MODELS}, (iid, k, desc)))
    summary = []
    for kind, path, cks, dev_info in jobs:
        stem = path.stem
        out_dir = OUT / "inference" / f"{kind}__{stem}"
        modes = modes_for(cks[primary])
        res = extract_image(path, cks, cfg, dev_ranges, primary, modes, out_dir)
        rows = res["rows"]
        cand_px = int((res["candidate_mask"] > 0).sum())
        kept = res["masks"]["recall_priority"]
        rec = {"kind": kind, "image": stem, "out_dir": str(out_dir.relative_to(ROOT)), "width": res["bgr"].shape[1],
               "height": res["bgr"].shape[0], "primary_model": primary, "thr_recall": modes["recall_priority"],
               "thr_precision": modes["precision_priority"], "candidate_groups": len(rows),
               "accepted_recall": sum(r["keep_recall_priority"] for r in rows), "accepted_precision": sum(r["keep_precision_priority"] for r in rows),
               "candidate_pixels": cand_px, "retained_pixels_recall": int(kept.sum()),
               "retained_pixels_precision": int(res["masks"]["precision_priority"].sum()),
               "removed_candidate_pixel_ratio_recall": round(1 - kept.sum() / max(1, cand_px), 4),
               "mean_score_primary": round(float(np.mean([r[f"p_{primary}"] for r in rows])), 4) if rows else "",
               "groups_ood": sum(bool(r["ood_flags"]) for r in rows), "groups_low_reliability": sum(r["reliability"] == "LOW" for r in rows),
               "seconds": round(res["seconds"], 3), "checkpoint": "full-dev" if kind == "field" else f"fold{dev_info[1]} (image excluded)",
               "hardware": json.dumps(hw)}
        for m in MODELS:
            rec[f"accepted_{m}@recall"] = sum(r[f"keep_{m}@recall"] for r in rows)
            rec[f"retained_px_{m}@recall"] = int(sum(g["pixels"] for g, r in zip(res["groups"], rows) if r[f"keep_{m}@recall"]))
        if dev_info:
            iid, k, desc = dev_info
            o = oof[(oof.image_id == iid)]
            rec["description"] = desc
            rec["phase3_image_level_oof"] = json.dumps({m: round(float(o[o.model == n].score.iloc[0]), 3) for m, n in (
                ("S3", "C_shape_domain_matched|train_all|torch_logistic"), ("S4", "S4_hard_example_reweighted"),
                ("RF", "C_shape_domain_matched|train_all|random_forest"), ("S1", "S1_basic_unweighted"))})
        summary.append(rec)
        print(f"{kind:5s} {stem:22s} groups {len(rows):4d} kept(recall) {rec['accepted_recall']:4d} kept(prec) {rec['accepted_precision']:4d} "
              f"px kept {rec['retained_pixels_recall']:6d}/{cand_px:6d} OOD {rec['groups_ood']:4d} {rec['seconds']:.2f}s")
    cols = list(dict.fromkeys(k for r in summary for k in r))
    with open(OUT / "extraction_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    # versioned snapshot: candidate / grouping / threshold configuration + summary, keyed by a config hash (never overwritten)
    import hashlib
    import inspect
    import shutil
    from pac2 import extraction as ex
    conf = {"stroke_cfg": ex.STROKE_CFG, "group_components_defaults": str(inspect.signature(ex.group_components)),
            "primary_model": primary, "thresholds": {m: full[m]["thresholds"] for m in MODELS},
            "extraction_source_sha256": hashlib.sha256(Path(ex.__file__).read_bytes()).hexdigest()}
    ver = hashlib.sha256(json.dumps(conf, sort_keys=True, default=str).encode()).hexdigest()[:12]
    snap = OUT / "snapshots" / ver
    if not snap.exists():
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(conf, indent=1, default=str), encoding="utf-8")
        shutil.copy2(OUT / "extraction_summary.csv", snap / "extraction_summary.csv")
        shutil.copytree(OUT / "inference", snap / "inference")
    print("config snapshot:", snap.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
