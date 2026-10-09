"""Phase 4 checkpoints: Phase 3 saved no model files, so S1 / S3 / S4 / RF are re-fitted on the 90 DEVELOPMENT images only
(full-dev models for new images; per-fold models so development demos stay out-of-fold). Test 30 is never read.

Operating thresholds come from the Phase 3 out-of-fold predictions (reports/phase3/oof_predictions.csv):
  recall_priority    = highest threshold with OOF handwritten recall >= 0.97
  precision_priority = lowest threshold with OOF formal false-positive rate 0 (else the BA-optimal threshold)
Applying OOF-derived thresholds to the full-dev refit is an approximation (recorded in each checkpoint).
Run: .venv\\Scripts\\python.exe scripts\\phase4_build_checkpoints.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pac2 import ROOT, load_config  # noqa: E402
from pac2.candidates import candidates, read_bgr  # noqa: E402
from pac2.extraction import CKPT_DIR, save_checkpoint  # noqa: E402
from pac2.features import FEATURES, extract  # noqa: E402
from pac2.logistic import TorchLogistic, fit_hard_example_reweighted  # noqa: E402
from pac2.normalize import normalize_mask  # noqa: E402
from pac2.prep import FoldPrep  # noqa: E402
from phase3_validation import HEIGHTS, inner_select  # noqa: E402

OOF_NAME = {"S1_basic_unweighted": "S1_basic_unweighted", "S3_inner_cv_logistic": "C_shape_domain_matched|train_all|torch_logistic",
            "S4_hard_example_reweighted": "S4_hard_example_reweighted", "RF_random_forest": "C_shape_domain_matched|train_all|random_forest"}
GRID = np.round(np.arange(0.01, 0.991, 0.01), 2)


def op_metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {"threshold": float(t), "handwritten_recall": float((pred[y == 1] == 1).mean()), "formal_fpr": float((pred[y == 0] == 1).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "f1": float(f1_score(y, pred))}


def thresholds_from_oof(oof: pd.DataFrame, model: str) -> dict:
    o = oof[oof.model == OOF_NAME[model]]
    y, p = o.label.to_numpy(), o.score.to_numpy()
    rec_ok = [t for t in GRID if op_metrics(y, p, t)["handwritten_recall"] >= 0.97]
    t_rec = max(rec_ok) if rec_ok else float(GRID[0])
    fpr0 = [t for t in GRID if op_metrics(y, p, t)["formal_fpr"] == 0]
    t_prec = min(fpr0) if fpr0 else float(max(GRID, key=lambda t: op_metrics(y, p, t)["balanced_accuracy"]))
    t_ba = float(max(GRID, key=lambda t: (op_metrics(y, p, t)["balanced_accuracy"], -abs(t - 0.5))))
    return {"recall_priority": float(t_rec), "precision_priority": float(t_prec), "ba_optimal": t_ba,
            "oof_at_recall_priority": op_metrics(y, p, t_rec), "oof_at_precision_priority": op_metrics(y, p, t_prec),
            "oof_at_ba_optimal": op_metrics(y, p, t_ba), "recall_target_0.97_met": bool(rec_ok),
            "source": "Phase 3 out-of-fold predictions of the per-fold models (approximation for the refit model)"}


def fit_all(C: dict, y: np.ndarray, seed: int, oof: pd.DataFrame, n_test: int) -> dict:
    key, lam, thr_inner = inner_select({h: C[h] for h in HEIGHTS}, y, "train_all", seed)
    X = C[key]
    prep = FoldPrep("train_all").fit(X, y)
    Z = prep.transform(X)
    base = {"features": FEATURES, "candidate_method": "gray_otsu (training crops); consensus >=2/4 methods on new images",
            "normalization": "crop ink bbox + uniform rescale to H_t + re-binarize (Phase 3 experiment C)", "H_t": int(key),
            "standardization": "train_all z-score", "impute_median": prep.median.tolist(), "mu": prep.mu.tolist(),
            "sigma": prep.sigma.tolist(), "trained_on": {"dev_images": int(len(y)), "test_images": n_test,
                                                          "handwritten": int(y.sum()), "formal": int((y == 0).sum())}}
    out = {}
    m1 = TorchLogistic(0.01, "lbfgs", class_weighted=False).fit(Z, y)
    out["S1_basic_unweighted"] = ({**base, "kind": "logistic", "lambda": 0.01, "class_weighted": False, "w": m1.w.tolist(), "b": m1.b}, None)
    m3 = TorchLogistic(lam, "lbfgs").fit(Z, y)
    out["S3_inner_cv_logistic"] = ({**base, "kind": "logistic", "lambda": lam, "class_weighted": True, "inner_cv_threshold": thr_inner,
                                     "w": m3.w.tolist(), "b": m3.b}, None)
    m4, _ = fit_hard_example_reweighted(Z, y, lam, seed=seed)
    out["S4_hard_example_reweighted"] = ({**base, "kind": "logistic", "lambda": lam, "class_weighted": True,
                                           "reweighting": {"rounds": 10, "eta": 1.0, "ema": 0.5, "clip": [0.5, 3.0]}, "w": m4.w.tolist(), "b": m4.b}, None)
    rf = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=seed).fit(Z, y)
    out["RF_random_forest"] = ({**base, "kind": "random_forest", "n_estimators": 300}, rf)
    for name, (ck, obj) in out.items():
        ck["model"] = name
        ck["thresholds"] = thresholds_from_oof(oof, name)
    return out


def main() -> int:
    cfg = load_config()
    seed = cfg["seed"]
    with open(ROOT / "data" / "splits" / "split_manifest.csv", encoding="utf-8-sig") as f:
        man = list(csv.DictReader(f))
    dev = sorted([r for r in man if r["set"] == "dev"], key=lambda r: r["image_id"])
    y = np.array([int(r["label"]) for r in dev])
    fold = np.array([int(r["fold"]) for r in dev])
    ccfg = dict(cfg["candidates"], methods=["gray_otsu"])
    C = {h: [] for h in HEIGHTS}
    ncc, aspect = [], []
    for r in dev:
        mask = candidates(read_bgr(ROOT / r["path"]), ccfg)["gray_otsu"]
        ys, xs = np.nonzero(mask)
        aspect.append((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
        ncc.append(extract(mask, cfg["features"])["n_components"])
        for h in HEIGHTS:
            fx = extract(normalize_mask(mask, h), cfg["features"])
            C[h].append([fx[k] for k in FEATURES])
    C = {h: np.array(v, float) for h, v in C.items()}
    oof = pd.read_csv(ROOT / "reports" / "phase3" / "oof_predictions.csv")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    dev_ranges = {"n_cc": [int(min(ncc)), int(max(ncc))], "aspect": [float(min(aspect)), float(max(aspect))]}
    (CKPT_DIR / "dev_ranges.json").write_text(json.dumps(dev_ranges, indent=1), encoding="utf-8")
    full = fit_all(C, y, seed, oof, 0)
    for name, (ck, obj) in full.items():
        save_checkpoint(name, ck, obj)
    for k in range(5):
        tr = fold != k
        fk = fit_all({h: v[tr] for h, v in C.items()}, y[tr], seed, oof, 0)
        for name, (ck, obj) in fk.items():
            ck["fold_excluded"] = k
            ck["excluded_images"] = [r["image_id"] for r, f_ in zip(dev, fold) if f_ == k]
            save_checkpoint(name, ck, obj, CKPT_DIR / f"fold{k}")
    # operating-point table and primary-model rule: at the recall-priority threshold, lowest formal FPR, then highest BA
    rows = []
    for name, (ck, _) in full.items():
        t = ck["thresholds"]
        for mode in ("recall_priority", "precision_priority", "ba_optimal"):
            rows.append({"model": name, "mode": mode, **t[f"oof_at_{mode}"], "H_t": ck["H_t"], "lambda": ck.get("lambda", "")})
    tab = pd.DataFrame(rows)
    rp = tab[tab["mode"] == "recall_priority"].sort_values(["formal_fpr", "balanced_accuracy"], ascending=[True, False])
    primary = rp.iloc[0]["model"]
    tab.to_csv(CKPT_DIR / "operating_points.csv", index=False, encoding="utf-8-sig")
    (CKPT_DIR / "selection.json").write_text(json.dumps({"primary_model": primary, "rule": "recall-priority OOF threshold (handwritten recall >= 0.97); "
                                                         "lowest formal FPR, tie -> highest BA", "dev_ranges": dev_ranges}, indent=1), encoding="utf-8")
    print(tab.round(3).to_string(index=False))
    print("primary model:", primary, "| dev ranges:", dev_ranges)
    return 0


if __name__ == "__main__":
    sys.exit(main())
