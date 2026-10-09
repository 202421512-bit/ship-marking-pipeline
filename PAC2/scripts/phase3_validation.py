"""Phase 3: shape-feature classification under domain-bias controls (DEV 90 only; test 30 stays locked).

Experiments (same 5 outer folds from data/splits/split_manifest.csv):
  A  7 shape features on the original gray_otsu masks          -> PyTorch logistic
  B  non-shape imaging-domain features only (control, never deployed) -> PyTorch logistic
  C  7 shape features on domain-matched masks (crop + uniform rescale to H_t + re-binarize) -> PyTorch logistic
Everything fitted inside the training fold: imputation, standardization, lambda / H_t (inner 3-fold CV), threshold
(inner out-of-fold predictions). Comparison models (RF, AdaBoost, GP classifier, OneClassSVM) on A and C features.
Run: .venv\\Scripts\\python.exe scripts\\phase3_validation.py
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2 import ROOT, load_config  # noqa: E402
from pac2.candidates import candidates, read_bgr  # noqa: E402
from pac2.domain import DOMAIN_FEATURES, domain_features  # noqa: E402
from pac2.features import FEATURES, NUMBER, extract  # noqa: E402
from pac2.experiment_logger import ExperimentLogger  # noqa: E402
from pac2.logistic import TorchLogistic, class_weights, fit_hard_example_reweighted  # noqa: E402
from pac2.metrics import binary_metrics  # noqa: E402
from pac2.normalize import normalize_mask  # noqa: E402
from pac2.prep import FoldPrep  # noqa: E402

OUT = ROOT / "reports" / "phase3"
LAMBDAS = (1e-3, 1e-2, 1e-1)
HEIGHTS = (48, 64, 96)
THRESH = np.round(np.arange(0.05, 0.951, 0.01), 2)


def best_threshold(y, p) -> float:
    from sklearn.metrics import balanced_accuracy_score
    sc = [(balanced_accuracy_score(y, (p >= t).astype(int)), -abs(t - 0.5), t) for t in THRESH]
    return float(max(sc)[2])


def inner_select(Xs: dict, y: np.ndarray, std: str, seed: int):
    """Inner 3-fold CV on the training fold over (H_t key, lambda); returns best key, lambda, threshold."""
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    best = None
    from sklearn.metrics import balanced_accuracy_score, log_loss
    for key, X in Xs.items():
        for lam in LAMBDAS:
            oof = np.zeros(len(y))
            for tr, va in skf.split(X, y):
                prep = FoldPrep(std).fit(X[tr], y[tr])
                m = TorchLogistic(lam).fit(prep.transform(X[tr]), y[tr])
                oof[va] = m.predict_proba(prep.transform(X[va]))
            ba = balanced_accuracy_score(y, (oof >= 0.5).astype(int))
            ll = log_loss(y, np.clip(oof, 1e-9, 1 - 1e-9), labels=[0, 1])
            cand = (ba, -ll, key, lam, oof)
            if best is None or cand[:2] > best[:2]:
                best = cand
    return best[2], best[3], best_threshold(y, best[4])


def main() -> int:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    logger = ExperimentLogger(OUT, seed, {"config": cfg, "lambdas": LAMBDAS, "heights": HEIGHTS, "inner_cv": 3,
                                          "adam": {"lr": 0.05, "epochs": 600, "patience": 100}})
    with open(ROOT / "data" / "splits" / "split_manifest.csv", encoding="utf-8-sig") as f:
        dev = sorted([r for r in csv.DictReader(f) if r["set"] == "dev"], key=lambda r: r["image_id"])
    ccfg = dict(cfg["candidates"], methods=["gray_otsu"])
    y = np.array([int(r["label"]) for r in dev])
    fold = np.array([int(r["fold"]) for r in dev])
    ids = [r["image_id"] for r in dev]
    A, B, C = [], [], {h: [] for h in HEIGHTS}
    missA, norm_examples = [], []
    for r in dev:
        bgr = read_bgr(ROOT / r["path"])
        mask = candidates(bgr, ccfg)["gray_otsu"]
        fa = extract(mask, cfg["features"])
        A.append([fa[k] for k in FEATURES])
        missA.append([fa[f"{k}_missing"] for k in FEATURES])
        B.append([domain_features(bgr)[k] for k in DOMAIN_FEATURES])
        for h in HEIGHTS:
            nm = normalize_mask(mask, h)
            fc = extract(nm, cfg["features"])
            C[h].append([fc[k] for k in FEATURES])
            if h == 64 and len(norm_examples) < 12 and (r["class"] == "formal" or len(norm_examples) >= 6):
                norm_examples.append((r["image_id"], bgr, mask, nm))
    A, B = np.array(A, float), np.array(B, float)
    C = {h: np.array(v, float) for h, v in C.items()}
    missA = np.array(missA)

    fold_rows, oof_rows, coef_rows, hist_rows, failures, prep_warn = [], [], [], [], [], []
    oof = {}
    configs = [("A_shape_original", "train_all"), ("A_shape_original", "formal_ref"), ("B_domain_only", "train_all"),
               ("C_shape_domain_matched", "train_all"), ("C_shape_domain_matched", "formal_ref")]
    chosen_h, chosen_c = {}, {}
    for exp, std in configs:
        name = f"{exp}|{std}|torch_logistic"
        oof[name] = {"p": np.zeros(len(y)), "pred": np.zeros(len(y), int), "pred05": np.zeros(len(y), int)}
        for k in range(5):
            tr, va = fold != k, fold == k
            Xs = {"orig": A} if exp.startswith("A") else {"dom": B} if exp.startswith("B") else {h: C[h] for h in HEIGHTS}
            key, lam, thr = inner_select({kk: v[tr] for kk, v in Xs.items()}, y[tr], std, seed)
            X = Xs[key]
            if exp.startswith("C") and std == "train_all":
                chosen_h[k] = key
                chosen_c[k] = (key, lam, thr)
            prep = FoldPrep(std).fit(X[tr], y[tr])
            prep_warn += [f"{name} fold{k}: {w}" for w in prep.warnings]
            Ztr, Zva = prep.transform(X[tr]), prep.transform(X[va])
            m = TorchLogistic(lam, "lbfgs").fit(Ztr, y[tr], Zva, y[va])
            madam = TorchLogistic(lam, "adam").fit(Ztr, y[tr], Zva, y[va])
            names_ = FEATURES if not exp.startswith("B") else DOMAIN_FEATURES
            logger.log("feature_weights", name, k, {"standardization": std, "lambda": lam, "H_t": key},
                       [{"stage": st, "optimizer": op, "feature": n_, "weight": float(v)}
                        for st, op, vec in (("initial", "zero_init", madam.initial_w), ("final", "lbfgs", m.w), ("final", "adam", madam.w))
                        for n_, v in zip(names_, vec)])
            settings = {"experiment": exp, "standardization": std, "H_t": key, "lambda": lam, "threshold": thr,
                        "class_weight": m.class_weight, "features": FEATURES if not exp.startswith("B") else DOMAIN_FEATURES}
            for mm, opt in ((m, "lbfgs"), (madam, "adam")):
                if mm.failure:
                    failures.append(f"{name} fold{k} {opt}: {mm.failure}")
                logger.log("training_history", name, k, settings,
                           [{"optimizer": opt, "early_stop_epoch": mm.early_stop_epoch, "best_val_epoch_monitor_only": mm.best_val_epoch, **h}
                            for h in mm.history])
            p = m.predict_proba(Zva)
            oof[name]["p"][va] = p
            oof[name]["pred"][va] = (p >= thr).astype(int)
            oof[name]["pred05"][va] = (p >= 0.5).astype(int)
            met = binary_metrics(y[va], (p >= thr).astype(int), p)
            fold_rows.append({"model": name, "fold": k, "selected": f"H={key} lambda={lam} thr={thr}",
                              "class_weight": json.dumps({str(a): round(b, 3) for a, b in m.class_weight.items()}), **met,
                              "ba_at_0.5": binary_metrics(y[va], (p >= 0.5).astype(int))["balanced_accuracy"],
                              "adam_vs_lbfgs_max_coef_diff": round(float(np.abs(m.w - madam.w).max()), 4)})
            names = FEATURES if not exp.startswith("B") else DOMAIN_FEATURES
            coef_rows.append({"model": name, "fold": k, "lambda": lam, "H_t": key, "bias": round(m.b, 4),
                              **{f"w_{n}": round(float(v), 4) for n, v in zip(names, m.w)}})

    # comparison models on A / C shape features (train_all standardization, C uses the fold's selected H_t)
    cmp_models = {
        "random_forest": lambda: RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=seed),
        "adaboost": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=seed),
        # length scale bounded to (0.1, 10) on z-standardized inputs: unbounded optimization drove it to ~6e4 (constant kernel,
        # every probability 0.823 -> BA 0.5). Changed after that diagnosis; recorded in the report.
        "gp_classifier": lambda: GaussianProcessClassifier(ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (0.1, 10.0)),
                                                           n_restarts_optimizer=3, random_state=seed),
    }
    for exp in ("A_shape_original", "C_shape_domain_matched"):
        for mname in list(cmp_models) + ["one_class_svm"]:
            name = f"{exp}|train_all|{mname}"
            oof[name] = {"p": np.zeros(len(y)), "pred": np.zeros(len(y), int)}
            for k in range(5):
                tr, va = fold != k, fold == k
                X = A if exp.startswith("A") else C[chosen_h[k]]
                if mname == "one_class_svm":
                    trh = tr & (y == 1)                                      # handwritten only = normal class
                    prep = FoldPrep("train_all").fit(X[trh], y[trh])
                    oc = OneClassSVM(nu=0.1, gamma="scale").fit(prep.transform(X[trh]))
                    s = oc.decision_function(prep.transform(X[va]))
                    oof[name]["p"][va], oof[name]["pred"][va] = s, (s >= 0).astype(int)
                    met = binary_metrics(y[va], (s >= 0).astype(int), s, prob=False)
                else:
                    def fit_predict(Xa, ya, Xb):
                        prep = FoldPrep("train_all").fit(Xa, ya)
                        mdl = cmp_models[mname]()
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if mname == "adaboost":
                                cw = class_weights(ya)
                                mdl.fit(prep.transform(Xa), ya, sample_weight=np.array([cw[v] for v in ya]))
                            else:
                                mdl.fit(prep.transform(Xa), ya)
                        return mdl.predict_proba(prep.transform(Xb))[:, 1]
                    # threshold from inner 3-fold out-of-fold predictions on the training fold (same rule as the logistic)
                    Xtr, ytr = X[tr], y[tr]
                    inner = np.zeros(len(ytr))
                    for itr, iva in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr):
                        inner[iva] = fit_predict(Xtr[itr], ytr[itr], Xtr[iva])
                    thr = best_threshold(ytr, inner)
                    p = fit_predict(Xtr, ytr, X[va])
                    oof[name]["p"][va], oof[name]["pred"][va] = p, (p >= thr).astype(int)
                    met = binary_metrics(y[va], (p >= thr).astype(int), p)
                    met["ba_at_0.5"] = binary_metrics(y[va], (p >= 0.5).astype(int))["balanced_accuracy"]
                fold_rows.append({"model": name, "fold": k, "selected": f"H={chosen_h[k] if exp.startswith('C') else 'orig'} "
                                  + ("nu=0.1 (a priori)" if mname == "one_class_svm" else f"thr={thr} (inner CV)"), **met})

    # ---------------- optimization stages on identical C features / folds (Graph 3)
    #  S1 basic: unweighted BCE, lambda 0.01, threshold 0.5 | S2 + class weights | S3 = C|train_all (inner-CV lambda, H_t,
    #  threshold) | S4 = S3 + hard-example reweighting (rounds 10, eta 1, EMA 0.5, clip 0.5-3 fixed a priori)
    stage_names = {"S1_basic_unweighted": None, "S2_class_weighted": None, "S4_hard_example_reweighted": None}
    for sname in stage_names:
        oof[sname] = {"p": np.zeros(len(y)), "pred": np.zeros(len(y), int)}
    for k in range(5):
        tr, va = fold != k, fold == k
        key, lam, thr = chosen_c[k]
        X = C[key]
        prep = FoldPrep("train_all").fit(X[tr], y[tr])
        Ztr, Zva = prep.transform(X[tr]), prep.transform(X[va])
        for sname, cw_on in (("S1_basic_unweighted", False), ("S2_class_weighted", True)):
            m = TorchLogistic(0.01, "lbfgs", class_weighted=cw_on).fit(Ztr, y[tr])
            ma = TorchLogistic(0.01, "adam", class_weighted=cw_on).fit(Ztr, y[tr], Zva, y[va])
            p = m.predict_proba(Zva)
            oof[sname]["p"][va], oof[sname]["pred"][va] = p, (p >= 0.5).astype(int)
            st = {"H_t": key, "lambda": 0.01, "threshold": 0.5, "class_weighted": cw_on}
            fold_rows.append({"model": sname, "fold": k, "selected": json.dumps(st), **binary_metrics(y[va], (p >= 0.5).astype(int), p)})
            logger.log("training_history", sname, k, st, [{"optimizer": "adam", "early_stop_epoch": ma.early_stop_epoch,
                                                            "best_val_epoch_monitor_only": ma.best_val_epoch, **h} for h in ma.history])
            if m.failure or ma.failure:
                failures.append(f"{sname} fold{k}: {m.failure or ma.failure}")
            logger.log("feature_weights", sname, k, {"standardization": "train_all", **st},
                       [{"stage": "final", "optimizer": "lbfgs", "feature": n_, "weight": float(v)} for n_, v in zip(FEATURES, m.w)])
        # S4: threshold from inner OOF of the reweighted procedure (training fold only)
        inner = np.zeros(int(tr.sum()))
        for itr, iva in StratifiedKFold(3, shuffle=True, random_state=seed).split(Ztr, y[tr]):
            p_ = FoldPrep("train_all").fit(X[tr][itr], y[tr][itr])
            mi, _ = fit_hard_example_reweighted(p_.transform(X[tr][itr]), y[tr][itr], lam, seed=seed)
            inner[iva] = mi.predict_proba(p_.transform(X[tr][iva]))
        thr4 = best_threshold(y[tr], inner)
        m4, rh = fit_hard_example_reweighted(Ztr, y[tr], lam, X_val=Zva, y_val=y[va], seed=seed)
        p = m4.predict_proba(Zva)
        oof["S4_hard_example_reweighted"]["p"][va] = p
        oof["S4_hard_example_reweighted"]["pred"][va] = (p >= thr4).astype(int)
        st = {"H_t": key, "lambda": lam, "threshold": thr4, "rounds": 10, "eta": 1.0, "ema": 0.5, "clip": [0.5, 3.0]}
        fold_rows.append({"model": "S4_hard_example_reweighted", "fold": k, "selected": json.dumps(st),
                          **binary_metrics(y[va], (p >= thr4).astype(int), p)})
        logger.log("reweighting_history", "S4_hard_example_reweighted", k, st, rh)
        logger.log("feature_weights", "S4_hard_example_reweighted", k, {"standardization": "train_all", **st},
                   [{"stage": "final", "optimizer": "lbfgs_reweighted", "feature": n_, "weight": float(v)} for n_, v in zip(FEATURES, m4.w)])
        # AdaBoost per-iteration history (C features, class-weighted start), SAMME weights reconstructed and checked
        cw = class_weights(y[tr])
        sw0 = np.array([cw[v] for v in y[tr]])
        ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=seed).fit(Ztr, y[tr], sample_weight=sw0)
        swr = sw0 / sw0.sum()
        staged_tr = list(ada.staged_predict_proba(Ztr))
        staged_va = list(ada.staged_predict_proba(Zva))
        rows = []
        for it, (est, aw, err) in enumerate(zip(ada.estimators_, ada.estimator_weights_, ada.estimator_errors_)):
            inc = est.predict(Ztr) != y[tr]
            recon_err = float((swr * inc).sum() / swr.sum())
            ptr, pva = staged_tr[it][:, 1], staged_va[it][:, 1]
            rows.append({"iteration": it + 1, "estimator_weight": float(aw), "estimator_error_sklearn": float(err),
                         "estimator_error_reconstructed": recon_err, "weight_min": swr.min(), "weight_median": float(np.median(swr)),
                         "weight_max": swr.max(), "weight_share_formal": float(swr[y[tr] == 0].sum() / swr.sum()),
                         "ensemble_train_misclassified": int(((ptr >= 0.5).astype(int) != y[tr]).sum()),
                         "ensemble_train_logloss": float(np.mean(-(y[tr] * np.log(ptr + 1e-12) + (1 - y[tr]) * np.log(1 - ptr + 1e-12)))),
                         "train_recall_formal": float(((ptr < 0.5) & (y[tr] == 0)).sum() / max(1, (y[tr] == 0).sum())),
                         "train_recall_handwritten": float(((ptr >= 0.5) & (y[tr] == 1)).sum() / max(1, (y[tr] == 1).sum())),
                         "val_balanced_accuracy": binary_metrics(y[va], (pva >= 0.5).astype(int))["balanced_accuracy"]})
            swr = swr * np.exp(aw * inc * (swr > 0))
            swr = swr / swr.sum()
        logger.log("adaboost_history", "C_shape_domain_matched|adaboost", k, {"n_estimators": 50, "base": "stump", "init": "class_weight"}, rows)

    # missing-indicator ablation (A, train_all, lambda 1e-2 fixed a priori)
    name = "A_shape_original|train_all|torch_logistic+missing_indicators(lambda=0.01)"
    oof[name] = {"p": np.zeros(len(y)), "pred": np.zeros(len(y), int)}
    for k in range(5):
        tr, va = fold != k, fold == k
        prep = FoldPrep("train_all").fit(A[tr], y[tr])
        Z = np.hstack([prep.transform(A), missA])
        m = TorchLogistic(1e-2).fit(Z[tr], y[tr])
        p = m.predict_proba(Z[va])
        oof[name]["p"][va], oof[name]["pred"][va] = p, (p >= 0.5).astype(int)
        fold_rows.append({"model": name, "fold": k, "selected": "fixed", **binary_metrics(y[va], (p >= 0.5).astype(int), p)})

    # pooled out-of-fold summary
    summary = []
    for name, o in oof.items():
        prob = "one_class_svm" not in name
        met = binary_metrics(y, o["pred"], o["p"], prob=prob)
        fr = [r for r in fold_rows if r["model"] == name]
        summary.append({"model": name, **met, "fold_ba_mean": round(float(np.mean([r["balanced_accuracy"] for r in fr])), 4),
                        "fold_ba_std": round(float(np.std([r["balanced_accuracy"] for r in fr])), 4)})
        anym = missA.any(axis=1)
        summary[-1]["ba_rows_with_missing"] = round(float(binary_metrics(y[anym], o["pred"][anym])["balanced_accuracy"]), 4) if len(set(y[anym])) == 2 else ""
        summary[-1]["ba_rows_complete"] = round(float(binary_metrics(y[~anym], o["pred"][~anym])["balanced_accuracy"]), 4)
        for i in range(len(y)):
            oof_rows.append({"model": name, "image_id": ids[i], "label": y[i], "fold": fold[i], "score": round(float(o["p"][i]), 5),
                             "pred": int(o["pred"][i]), "correct": int(o["pred"][i] == y[i]), "any_missing_A": int(missA[i].any())})

    # coefficient stability
    stab = []
    for name in sorted({r["model"] for r in coef_rows}):
        rs = [r for r in coef_rows if r["model"] == name]
        for c in [k for k in rs[0] if k.startswith("w_")]:
            v = np.array([r[c] for r in rs])
            stab.append({"model": name, "feature": c[2:], "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                         "sign_consistent": bool((v > 0).all() or (v < 0).all()), "folds": list(v)})

    # within-handwritten correlation of shape features with imaging-domain variables (domain sensitivity)
    corr = []
    hw = y == 1
    for setname, X in (("A_original", A), ("C_H64", C[64])):
        for j, f in enumerate(FEATURES):
            for dn in ("height", "log_sharpness", "saturation_mean"):
                d = B[:, DOMAIN_FEATURES.index(dn)]
                ok = hw & np.isfinite(X[:, j])
                rho, pv = spearmanr(X[ok, j], d[ok])
                corr.append({"features": setname, "feature": f"{NUMBER[f]}_{f}", "domain_var": dn, "spearman_rho": round(float(rho), 3),
                             "p_value": round(float(pv), 4), "n": int(ok.sum())})

    def wcsv(fn, rows):
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(OUT / fn, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    wcsv("fold_metrics.csv", fold_rows)
    wcsv("oof_summary.csv", summary)
    wcsv("oof_predictions.csv", oof_rows)
    wcsv("coefficients_per_fold.csv", coef_rows)
    wcsv("coefficient_stability.csv", stab)
    # model_comparison.csv: pooled OOF + fold mean/sd per metric (all models incl. optimization stages)
    mc = []
    for s in summary:
        fr = [r for r in fold_rows if r["model"] == s["model"]]
        row = {"model": s["model"], "n_images": len(y), "n_folds": 5}
        for mname in ("balanced_accuracy", "precision", "recall", "f1", "formal_recall(specificity)", "roc_auc", "log_loss", "brier"):
            row[f"oof_{mname}"] = s.get(mname, "")
            vals = [r[mname] for r in fr if mname in r and r[mname] != ""]
            row[f"fold_{mname}_mean"] = round(float(np.mean(vals)), 4) if vals else ""
            row[f"fold_{mname}_sd"] = round(float(np.std(vals)), 4) if vals else ""
        row["score_type"] = "decision_function (not a probability)" if "one_class_svm" in s["model"] else "probability"
        mc.append(row)
    wcsv("model_comparison.csv", mc)
    # extraction material for Graph 8: candidate masks of dev images (display only; decisions come from OOF predictions)
    ext = OUT / "extraction_masks"
    ext.mkdir(exist_ok=True)
    for r in dev:
        mask = candidates(read_bgr(ROOT / r["path"]), ccfg)["gray_otsu"]
        cv2.imencode(".png", mask)[1].tofile(str(ext / (r["image_id"].replace("/", "__"))))
    logger.save()
    wcsv("domain_correlation_within_handwritten.csv", corr)
    (OUT / "failures_and_warnings.txt").write_text("\n".join(["[TRAINING FAILURES]"] + (failures or ["none"]) +
                                                             ["", "[STANDARDIZATION WARNINGS]"] + (prep_warn or ["none"])), encoding="utf-8")
    _panels(dev, ids, y, oof, norm_examples)
    print(json.dumps({"n_failures": len(failures), "n_prep_warnings": len(prep_warn)}, indent=0))
    for s in summary:
        print(f"{s['model']:78s} BA {s['balanced_accuracy']:.3f} (fold {s['fold_ba_mean']:.3f}+-{s['fold_ba_std']:.3f}) "
              f"F1 {s['f1']:.3f} formal_rec {s['formal_recall(specificity)']:.3f} AUC {s.get('roc_auc', float('nan')):.3f} "
              f"fp {s['fp']} fn {s['fn']}")
    return 0


def _panels(dev, ids, y, oof, norm_examples):
    # misclassified (pooled OOF) for the main shape models
    for name in ("A_shape_original|train_all|torch_logistic", "C_shape_domain_matched|train_all|torch_logistic"):
        o = oof[name]
        wrong = [i for i in range(len(y)) if o["pred"][i] != y[i]]
        tiles = []
        for i in wrong[:30]:
            r = next(d for d in dev if d["image_id"] == ids[i])
            img = read_bgr(ROOT / r["path"])
            s = 120 / img.shape[0]
            t = cv2.resize(img, (max(1, int(img.shape[1] * s)), 120))[:, :300]
            t = cv2.copyMakeBorder(t, 22, 4, 4, max(4, 304 - t.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, f"{ids[i].split('/')[1][:-4]} y={y[i]} p={o['p'][i]:.2f}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
            tiles.append(cv2.resize(t, (308, 146)))
        if tiles:
            while len(tiles) % 4:
                tiles.append(np.full((146, 308, 3), 255, np.uint8))
            sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
            cv2.imencode(".png", sheet)[1].tofile(str(OUT / f"misclassified_{name.split('|')[0]}.png"))
    # domain-matched normalization examples (H_t = 64): original | gray_otsu mask | normalized mask
    tiles = []
    for iid, bgr, mask, nm in norm_examples:
        row = []
        for im in (bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(nm, cv2.COLOR_GRAY2BGR)):
            s = 90 / im.shape[0]
            row.append(cv2.copyMakeBorder(cv2.resize(im, (max(1, int(im.shape[1] * s)), 90), interpolation=cv2.INTER_NEAREST),
                                          0, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        t = np.hstack(row)
        t = cv2.copyMakeBorder(t, 20, 4, 4, max(0, 900 - t.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))[:, :904]
        cv2.putText(t, iid, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
        tiles.append(t)
    cv2.imencode(".png", np.vstack(tiles))[1].tofile(str(OUT / "domain_matched_masks_H64.png"))


if __name__ == "__main__":
    sys.exit(main())
