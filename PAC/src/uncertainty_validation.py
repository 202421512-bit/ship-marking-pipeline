"""UNCERTAINTY VALIDATION & CALIBRATION of the existing Bayesian (pseudo-)posterior.

Nothing in the classifier is changed (features, weights, variance handling as in the character
experiment). Protocol fixed a priori (config.UncertaintyValidationConfig):

* Nested group-level splits: outer K folds define FINAL TEST groups (each group tested once);
  inside the outer-train groups a group-level split gives TRAIN / CALIBRATION.
  A  Formal -> Formal, B  Informal -> Informal,
  C  Formal -> Informal (shared classes; calibration on held-out FORMAL groups - no target labels),
  C2 Formal -> Informal (same train / test as C; calibration on INFORMAL groups disjoint from test).
* Temperature T: minimise calibration NLL. Applied as softmax(log p / T) (= softmax(score / T)).
* Selective threshold: largest calibration coverage whose calibration risk <= target; frozen for test.
* Classes absent from the training partition are never dropped: counted as errors for accuracy, ECE,
  Brier, error detection and selective risk; NLL (undefined) is reported on the evaluable subset.
* Error-detection AUROC / AUPRC: POSITIVE CLASS = ERROR (wrong top-1).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import rankdata

from .character_experiment import Sample, class_audit, load_samples
from .config import UNCERTAINTY_RESULTS_DIR, AppConfig
from .dataset import FORMAL, INFORMAL
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, fit_reference, write_csv

MODELS = ("geometry", "topology", "combined")
EXPERIMENTS = {"A": "Formal -> Formal", "B": "Informal -> Informal",
               "C": "Formal -> Informal (calibrated on Formal)", "C2": "Formal -> Informal (calibrated on Informal)"}


class LeakageError(RuntimeError):
    """Partition overlap / fitting violation (aborts)."""


# ====================================================================== exact log posterior
def log_posterior(clf, x: Dict[str, float]) -> np.ndarray:
    """Exact normalised log posterior of BayesianShapeClassifier.predict (same arithmetic, no underflow)."""
    cfg = clf.cfg
    z = np.empty(len(clf.classes))
    for i, c in enumerate(clf.classes):
        lt = clf._group_loglik(x, c, clf.topology_features)
        lg = clf._group_loglik(x, c, clf.geometry_features)
        if cfg.aggregation == "mean":
            lt /= max(1, len(clf.topology_features))
            lg /= max(1, len(clf.geometry_features))
        z[i] = (cfg.topology_weight * lt + cfg.geometry_weight * lg) / cfg.likelihood_temperature
    z = z + clf.log_priors()
    return z - logsumexp(z)


@dataclass
class Prediction:
    """One scored sample (label kept separately from the scores; used only in evaluation)."""

    sample_id: str
    group_id: str
    style: str
    label: str
    classes: Tuple[str, ...] = field(repr=False)
    logp: np.ndarray = field(repr=False)

    def probs(self, T: float = 1.0) -> np.ndarray:
        z = self.logp / T
        return np.exp(z - logsumexp(z))


# ====================================================================== splits
def _shuffle_groups(groups: Sequence[str], seed: int) -> List[str]:
    g = sorted(set(groups))
    np.random.default_rng(seed).shuffle(g)
    return g


def _folds(groups: Sequence[str], k: int, seed: int) -> List[List[str]]:
    g = _shuffle_groups(groups, seed)
    return [g[i::k] for i in range(k)]


@dataclass
class Partition:
    """Train / calibration / test sample lists of one outer fold."""

    name: str
    train: List[Sample]
    cal: List[Sample]
    test: List[Sample]


def check_partition(p: Partition) -> Dict[str, int]:
    """Group- and sample-level disjointness of train / cal / test (raises on any overlap)."""
    tg, cg, sg = ({s.group_id for s in part} for part in (p.train, p.cal, p.test))
    ov = {"train_cal": len(tg & cg), "train_test": len(tg & sg), "cal_test": len(cg & sg)}
    ids = [{s.sample_id for s in part} for part in (p.train, p.cal, p.test)]
    ov["sample_overlap"] = len(ids[0] & ids[1]) + len(ids[0] & ids[2]) + len(ids[1] & ids[2])
    if any(ov.values()):
        raise LeakageError(f"{p.name}: partition overlap {ov}")
    return ov


def make_partitions(exp: str, samples: Sequence[Sample], shared: set, cfg: AppConfig) -> List[Partition]:
    """Nested group-level partitions for one experiment (fixed seed, no label information used)."""
    u = cfg.uncertainty
    formal = [s for s in samples if s.style == FORMAL]
    informal = [s for s in samples if s.style == INFORMAL]
    parts: List[Partition] = []
    if exp in ("A", "B"):
        pool = formal if exp == "A" else informal
        for k, test_groups in enumerate(_folds([s.group_id for s in pool], u.outer_folds, u.seed)):
            rest = _shuffle_groups([s.group_id for s in pool if s.group_id not in test_groups], u.seed + 1 + k)
            n_cal = max(1, int(round(u.calibration_fraction * len(rest))))
            cal_groups = set(rest[:n_cal])
            tg = set(test_groups)
            parts.append(Partition(f"{exp}:fold{k}",
                                   [s for s in pool if s.group_id not in tg and s.group_id not in cal_groups],
                                   [s for s in pool if s.group_id in cal_groups],
                                   [s for s in pool if s.group_id in tg]))
    else:
        f_sh = [s for s in formal if s.character in shared]
        i_sh = [s for s in informal if s.character in shared]
        f_parts = _folds([s.group_id for s in f_sh], u.domain_shift_formal_parts, u.seed)
        for k, test_groups in enumerate(_folds([s.group_id for s in i_sh], u.outer_folds, u.seed)):
            cal_f = set(f_parts[k % len(f_parts)])
            train = [s for s in f_sh if s.group_id not in cal_f]
            tg = set(test_groups)
            test = [s for s in i_sh if s.group_id in tg]
            cal = ([s for s in f_sh if s.group_id in cal_f] if exp == "C"
                   else [s for s in i_sh if s.group_id not in tg])
            parts.append(Partition(f"{exp}:fold{k}", train, cal, test))
    return parts


# ====================================================================== scoring
def score(models: Dict[str, object], samples: Sequence[Sample]) -> Dict[str, List[Prediction]]:
    """Log posteriors of every sample for each model (labels are not passed to the model)."""
    out = {m: [] for m in models}
    for i, s in enumerate(samples):
        x = dict(s.x)
        if any(k in x for k in ("class", "character", "label")):
            raise LeakageError("label inside inference input")
        for m, ref in models.items():
            lp = log_posterior(ref.clf, x)
            if i < 3:  # audit: the exact log posterior must reproduce the classifier's own posterior
                post = ref.clf.predict(x).posteriors
                ref_p = np.array([post[c] for c in ref.clf.classes])
                if not np.allclose(np.exp(lp), ref_p, rtol=1e-9, atol=1e-12):
                    raise RuntimeError("log_posterior does not reproduce BayesianShapeClassifier.predict")
            out[m].append(Prediction(s.sample_id, s.group_id, s.style, s.character, tuple(ref.clf.classes), lp))
    return out


def per_sample(preds: Sequence[Prediction], T: float = 1.0) -> Dict[str, np.ndarray]:
    """Arrays of confidence / correctness / scores for a list of predictions."""
    conf, corr, ent, marg, brier, nll, evaluable = [], [], [], [], [], [], []
    for p in preds:
        pr = p.probs(T)
        order = np.argsort(-pr)
        top = p.classes[order[0]]
        conf.append(pr[order[0]])
        corr.append(top == p.label)
        nz = pr[pr > 0]
        ent.append(float(-(nz * np.log(nz)).sum()))
        marg.append(pr[order[0]] - (pr[order[1]] if len(pr) > 1 else 0.0))
        ev = p.label in p.classes
        evaluable.append(ev)
        if ev:
            j = p.classes.index(p.label)
            y = np.zeros_like(pr)
            y[j] = 1.0
            brier.append(float(((pr - y) ** 2).sum()))
            nll.append(float(-(p.logp[j] / T - logsumexp(p.logp / T))))
        else:  # true class outside the closed set: probability 0 for the true class
            brier.append(float((pr ** 2).sum() + 1.0))
            nll.append(float("nan"))
    return {k: np.asarray(v, dtype=float) for k, v in (("conf", conf), ("correct", corr), ("entropy", ent),
                                                        ("margin", marg), ("brier", brier), ("nll", nll),
                                                        ("evaluable", evaluable))}


def ece(conf: np.ndarray, correct: np.ndarray, bins: int, equal_mass: bool = False) -> float:
    """Expected calibration error: sum_b |B|/n * |acc(B) - mean conf(B)| on the top-1 confidence."""
    if conf.size == 0:
        return float("nan")
    if equal_mass:
        order = np.argsort(conf)
        chunks = np.array_split(order, bins)
    else:
        idx = np.minimum((conf * bins).astype(int), bins - 1)
        chunks = [np.nonzero(idx == b)[0] for b in range(bins)]
    return float(sum(len(c) / conf.size * abs(correct[c].mean() - conf[c].mean()) for c in chunks if len(c)))


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int) -> List[Dict[str, float]]:
    """Per-bin (equal-width) mean confidence, accuracy and count."""
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin_low": b / bins, "bin_high": (b + 1) / bins, "n": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else float("nan"),
                     "accuracy": float(correct[m].mean()) if m.any() else float("nan")})
    return rows


def auroc(score_: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUROC (higher score -> more likely positive)."""
    pos = positive.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score_)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(score_: np.ndarray, positive: np.ndarray) -> float:
    """Average precision (step-wise area under the precision-recall curve)."""
    pos = positive.astype(bool)
    if pos.sum() == 0:
        return float("nan")
    order = np.argsort(-score_, kind="mergesort")
    tp = np.cumsum(pos[order])
    precision = tp / np.arange(1, len(order) + 1)
    return float((precision * pos[order]).sum() / pos.sum())


def fit_temperature(preds: Sequence[Prediction], bounds: Tuple[float, float]) -> float:
    """T minimising mean NLL on CALIBRATION predictions whose true class is in the model's class set."""
    ev = [p for p in preds if p.label in p.classes]
    if not ev:
        return 1.0
    idx = [p.classes.index(p.label) for p in ev]

    def nll(logT: float) -> float:
        T = math.exp(logT)
        return float(np.mean([-(p.logp[j] / T - logsumexp(p.logp / T)) for p, j in zip(ev, idx)]))

    res = minimize_scalar(nll, bounds=(math.log(bounds[0]), math.log(bounds[1])), method="bounded")
    return float(math.exp(res.x))


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> Tuple[float, float, float]:
    """Largest-coverage confidence threshold with CALIBRATION selective risk <= target.

    Returns (threshold, calibration coverage, calibration risk). Threshold = inf -> nothing accepted.
    """
    order = np.argsort(-conf, kind="mergesort")
    c, k = conf[order], correct[order]
    risk = np.cumsum(1 - k) / np.arange(1, len(k) + 1)
    best = None
    for n in range(len(c), 0, -1):
        if n < len(c) and c[n - 1] == c[n]:   # threshold must not split tied confidences
            continue
        if risk[n - 1] <= target:
            best = n
            break
    if best is None:
        return float("inf"), 0.0, float("nan")
    return float(c[best - 1]), best / len(c), float(risk[best - 1])


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Risk-coverage curve and its area (AURC; lower is better)."""
    order = np.argsort(-conf, kind="mergesort")
    k = correct[order]
    n = np.arange(1, len(k) + 1)
    risk = np.cumsum(1 - k) / n
    cov = n / len(k)
    return cov, risk, float(risk.mean())


# ====================================================================== experiment runner
@dataclass
class ExperimentOutput:
    """Collected test / calibration predictions and per-fold artefacts of one experiment."""

    test: Dict[str, List[Prediction]]
    cal: Dict[str, List[Prediction]]
    T: Dict[str, List[float]]
    thresholds: Dict[str, List[Dict[str, float]]]
    test_fold: Dict[str, List[int]]
    overlaps: List[Dict[str, int]]
    fold_info: List[Dict[str, object]]


def run_experiment(exp: str, samples: Sequence[Sample], shared: set, cfg: AppConfig) -> ExperimentOutput:
    """Fit on TRAIN, calibrate on CALIBRATION, freeze, then score FINAL TEST."""
    c = cfg.character
    weights = {"geometry": (0.0, 1.0), "topology": (1.0, 0.0), "combined": (c.w_topology, c.w_geometry)}
    out = ExperimentOutput({m: [] for m in MODELS}, {m: [] for m in MODELS}, {m: [] for m in MODELS},
                           {m: [] for m in MODELS}, {m: [] for m in MODELS}, [], [])
    tested = Counter()
    for k, part in enumerate(make_partitions(exp, samples, shared, cfg)):
        out.overlaps.append(check_partition(part))
        labels = [s.character for s in part.train]
        if len(set(labels)) < 2:
            raise LeakageError(f"{part.name}: fewer than 2 training classes")
        models = {m: fit_reference([s.x for s in part.train], labels, TOPO_FEATURES, GEO_FEATURES, w, cfg)
                  for m, w in weights.items()}
        cal = score(models, part.cal)
        test = score(models, part.test)
        for m in MODELS:
            # calibration artefacts are computed from CALIBRATION predictions only, before test is touched
            T = fit_temperature(cal[m], cfg.uncertainty.temperature_bounds)
            cs = per_sample(cal[m], T)
            thr, cov, rsk = choose_threshold(cs["conf"], cs["correct"], cfg.uncertainty.target_risk)
            out.T[m].append(T)
            out.thresholds[m].append({"fold": k, "threshold": thr, "cal_coverage": cov, "cal_risk": rsk,
                                      "temperature": T, "n_cal": len(cal[m])})
            out.cal[m] += cal[m]
            out.test[m] += test[m]
            out.test_fold[m] += [k] * len(test[m])
        for s in part.test:
            tested[s.sample_id] += 1
        out.fold_info.append({"fold": k, "train_groups": len({s.group_id for s in part.train}),
                              "cal_groups": len({s.group_id for s in part.cal}),
                              "test_groups": len({s.group_id for s in part.test}),
                              "train_samples": len(part.train), "cal_samples": len(part.cal),
                              "test_samples": len(part.test), "train_classes": len(set(labels)),
                              "test_unseen_class_samples": sum(s.character not in set(labels) for s in part.test),
                              "cal_unseen_class_samples": sum(s.character not in set(labels) for s in part.cal)})
    if any(v != 1 for v in tested.values()):
        raise LeakageError(f"{exp}: a sample was tested more than once")
    return out


def metric_block(preds: Sequence[Prediction], T_per_pred: np.ndarray, bins: int,
                 sens: Sequence[int]) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Accuracy / ECE / NLL / Brier on a list of predictions with per-prediction temperatures."""
    rows = [per_sample([p], float(t)) for p, t in zip(preds, T_per_pred)]
    if not rows:
        return {}, {}
    arr = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    ev = arr["evaluable"].astype(bool)
    out = {"n": int(len(preds)), "n_evaluable": int(ev.sum()), "accuracy": float(arr["correct"].mean()),
           "accuracy_evaluable": float(arr["correct"][ev].mean()) if ev.any() else float("nan"),
           "mean_confidence": float(arr["conf"].mean()),
           f"ece_{bins}": ece(arr["conf"], arr["correct"], bins),
           f"ece_{bins}_equal_mass": ece(arr["conf"], arr["correct"], bins, equal_mass=True),
           "ece_evaluable": ece(arr["conf"][ev], arr["correct"][ev], bins),
           "nll_evaluable": float(np.nanmean(arr["nll"][ev])) if ev.any() else float("nan"),
           "brier": float(arr["brier"].mean()),
           "brier_evaluable": float(arr["brier"][ev].mean()) if ev.any() else float("nan")}
    for b in sens:
        out[f"ece_{b}"] = ece(arr["conf"], arr["correct"], b)
    out["overconfidence"] = out["mean_confidence"] - out["accuracy"]
    return out, arr


@dataclass
class UVResult:
    """All results for reports and figures."""

    samples: List[Sample]
    outputs: Dict[str, ExperimentOutput]
    calibration: List[Dict[str, object]]
    error_detection: List[Dict[str, object]]
    selective: List[Dict[str, object]]
    curves: Dict[Tuple[str, str], Dict[str, object]]
    reliability: Dict[Tuple[str, str, str], List[Dict[str, float]]]
    distributions: Dict[Tuple[str, str, str], Dict[str, np.ndarray]]
    novelty_proxy: List[Dict[str, object]]
    data_info: Dict[str, object]
    report: Dict[str, object] = field(default_factory=dict)
    out_dir: Path = UNCERTAINTY_RESULTS_DIR


def _group_boot(groups: np.ndarray, fn: Callable[[np.ndarray], float], n_boot: int, seed: int) -> List[float]:
    """Group-level bootstrap CI of a statistic computed on a boolean sample selection."""
    ug = np.unique(groups)
    idx_by_g = {g: np.nonzero(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        sel = np.concatenate([idx_by_g[g] for g in pick])
        vals.append(fn(sel))
    v = np.asarray(vals, dtype=float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def run_uncertainty_validation(cfg: AppConfig, out_dir: Path = UNCERTAINTY_RESULTS_DIR,
                               say: Callable[[str], None] = print) -> UVResult:
    """Run A, B, C, C2 and compute every metric (CSV outputs written here)."""
    u = cfg.uncertainty
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, info = load_samples(cfg, say)
    dist, _, groups = class_audit(samples)
    shared = set(groups["shared"])
    outputs = {}
    for exp in EXPERIMENTS:
        say(f"[UV] experiment {exp}: {EXPERIMENTS[exp]}")
        outputs[exp] = run_experiment(exp, samples, shared, cfg)
    cal_rows, ed_rows, sel_rows = [], [], []
    curves, rel, dists, novelty = {}, {}, {}, []
    for exp, o in outputs.items():
        for m in MODELS:
            preds = o.test[m]
            folds = np.asarray(o.test_fold[m])
            T_test = np.asarray([o.T[m][f] for f in folds])
            g = np.asarray([p.group_id for p in preds])
            for stage, Tv in (("raw", np.ones(len(preds))), ("temperature_scaled", T_test)):
                block, arr = metric_block(preds, Tv, u.ece_bins, u.ece_bins_sensitivity)
                ci = {
                    "accuracy_ci95": _group_boot(g, lambda s: arr["correct"][s].mean(), u.bootstrap_iterations, u.seed),
                    f"ece_{u.ece_bins}_ci95": _group_boot(g, lambda s: ece(arr["conf"][s], arr["correct"][s], u.ece_bins),
                                                         u.bootstrap_iterations, u.seed),
                }
                cal_rows.append({"experiment": exp, "description": EXPERIMENTS[exp], "model": m, "stage": stage,
                                 "temperature_mean": float(np.mean(o.T[m])) if stage != "raw" else 1.0,
                                 "temperature_per_fold": "|".join(f"{t:.3f}" for t in o.T[m]) if stage != "raw" else "1",
                                 **block, **{k: f"[{v[0]:.4f}, {v[1]:.4f}]" for k, v in ci.items()}})
                rel[(exp, m, stage)] = reliability(arr["conf"], arr["correct"], u.ece_bins)
                dists[(exp, m, stage)] = arr
                err = 1 - arr["correct"]
                for sname, sval in (("max_posterior", -arr["conf"]), ("entropy", arr["entropy"]),
                                    ("margin", -arr["margin"])):
                    ed_rows.append({"experiment": exp, "model": m, "stage": stage, "score": sname,
                                    "positive_class": "ERROR (wrong top-1, incl. classes unseen in training)",
                                    "score_direction": "higher = more likely error",
                                    "auroc": auroc(sval, err), "auprc": auprc(sval, err),
                                    "error_prevalence": float(err.mean()), "n": int(len(err)),
                                    "auroc_ci95": "[{:.4f}, {:.4f}]".format(*_group_boot(
                                        g, lambda s, sv=sval: auroc(sv[s], err[s]), u.bootstrap_iterations, u.seed))})
                # class-novelty proxy (NOT an OOD validation): unseen-class samples vs seen-class samples
                unseen = 1 - arr["evaluable"]
                if stage == "raw" and 0 < unseen.sum() < len(unseen):
                    novelty.append({"experiment": exp, "model": m, "n_unseen_class": int(unseen.sum()),
                                    "auroc_max_posterior_unseen_vs_seen": auroc(-arr["conf"], unseen),
                                    "mean_conf_unseen": float(arr["conf"][unseen == 1].mean()),
                                    "mean_conf_seen": float(arr["conf"][unseen == 0].mean())})
            # selective classification: calibration-chosen thresholds, frozen, applied per test fold
            _, arr_c = metric_block(preds, T_test, u.ece_bins, u.ece_bins_sensitivity)
            thr = np.asarray([o.thresholds[m][f]["threshold"] for f in folds])
            accept = arr_c["conf"] >= thr
            cov = float(accept.mean())
            sel_acc = float(arr_c["correct"][accept].mean()) if accept.any() else float("nan")
            covs, risks, aurc = risk_coverage(arr_c["conf"], arr_c["correct"])
            curves[(exp, m)] = {"coverage": covs, "risk": risks, "aurc": aurc,
                                "operating_point": (cov, 1 - sel_acc if accept.any() else float("nan"))}
            sel_rows.append({"experiment": exp, "model": m, "target_risk": u.target_risk,
                             "threshold_rule": "largest calibration coverage with calibration risk <= target; frozen",
                             "thresholds_per_fold": "|".join(f"{t['threshold']:.4f}" for t in o.thresholds[m]),
                             "cal_coverage_per_fold": "|".join(f"{t['cal_coverage']:.3f}" for t in o.thresholds[m]),
                             "test_coverage": cov, "test_selective_accuracy": sel_acc,
                             "test_selective_risk": 1 - sel_acc if accept.any() else float("nan"),
                             "target_met_on_test": bool(accept.any() and (1 - sel_acc) <= u.target_risk),
                             "human_review_fraction": 1 - cov, "test_aurc": aurc,
                             "full_coverage_risk": float(1 - arr_c["correct"].mean()),
                             "n_test": int(len(preds)), "n_accepted": int(accept.sum())})
    write_csv(out_dir / "calibration_metrics.csv", cal_rows)
    write_csv(out_dir / "error_detection_metrics.csv", ed_rows)
    write_csv(out_dir / "selective_classification.csv", sel_rows)
    write_csv(out_dir / "class_novelty_proxy.csv", novelty)
    write_csv(out_dir / "fold_partitions.csv", [dict(fi, experiment=e) for e, o in outputs.items() for fi in o.fold_info])
    rel_rows = [dict(r, experiment=e, model=m, stage=s) for (e, m, s), rows in rel.items() for r in rows]
    write_csv(out_dir / "reliability_bins.csv", rel_rows)
    info["class_distribution"] = dist
    info["shared_classes"] = sorted(shared)
    return UVResult(list(samples), outputs, cal_rows, ed_rows, sel_rows, curves, rel, dists, novelty, info,
                    out_dir=out_dir)
