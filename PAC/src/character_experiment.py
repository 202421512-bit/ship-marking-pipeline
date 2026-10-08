"""CHARACTER EXPERIMENT: does topology add information over geometry (formal / informal / domain shift)?

Data: COUNT_MATCH images of the real Symbols dataset (manual transcriptions); one sample per
character segment. Protocol fixed a priori (config.CharacterExperimentConfig):

* Models  G = geometry only, T = topology only, TG = 0.5 * mean topology log-lik + 0.5 * mean
  geometry log-lik. Feature lists are the SAME as the welding experiment (real_experiment).
* Class statistics, shrinkage variances and sigma floors are fitted on the training fold only.
* Every split is by group_id (all variants _1/_2/_3 of one string stay together).
  A  Formal -> Formal              leave-one-group-out
  B  Informal -> Informal          leave-one-group-out
  C  Formal -> Informal            single split, SHARED classes only
  D  Formal + Informal -> Informal leave-one-informal-group-out (test group never in training)
* A test sample is evaluable only if its class occurs in the training fold (pre-declared).
* No restoration. Ground truth is used only for scoring.
* 95% CI: paired bootstrap over TEST GROUPS (never over characters).
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import CHAR_RESULTS_DIR, AppConfig
from .dataset import FORMAL, INFORMAL, run_dataset_audit
from .prototypes import extract_features
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, char_canvas, feature_dict, fit_reference, rel, write_csv

MODEL_NAMES = ("geometry", "topology", "combined")
EXPERIMENTS = {
    "A": ("formal", "FORMAL -> FORMAL"),
    "B": ("informal", "INFORMAL -> INFORMAL"),
    "C": ("domainshift", "FORMAL -> INFORMAL"),
    "D": ("mixed", "FORMAL + INFORMAL -> INFORMAL"),
}


class LeakageError(RuntimeError):
    """Raised when any data-leakage assertion fails (experiment aborted)."""


@dataclass
class Sample:
    """One character segment of a validated (COUNT_MATCH) image."""

    sample_id: str
    image_id: str
    group_id: str
    style: str
    character: str
    x: Dict[str, float] = field(repr=False)      # features ONLY (no label inside)


# ====================================================================== data
def load_samples(cfg: AppConfig, say: Callable[[str], None] = print) -> Tuple[List[Sample], Dict[str, object]]:
    """Fresh audit (current transcriptions) -> features of every COUNT_MATCH character segment."""
    audit = run_dataset_audit(cfg, write=True, previews=False, verbose=False)
    samples: List[Sample] = []
    status = Counter(str(m["validation_status"]) for m in audit.manifest)
    for m in audit.manifest:
        if m["validation_status"] != "COUNT_MATCH" or m["valid_for_training"] is not True:
            continue
        seg = audit.segmentations.get(str(m["image_path"]))
        text = str(m["normalized_transcription"])
        if seg is None or seg.count != len(text):
            raise LeakageError(f"alignment invariant broken for {m['image_path']}")
        image_id = Path(str(m["image_path"])).stem
        for idx, (cm, ch) in enumerate(zip(seg.char_masks, text)):
            bundle = extract_features(char_canvas(cm, cfg), cfg, with_persistence=True)
            f = feature_dict(bundle, with_ph=True)
            x = {k: float(f[k]) for k in TOPO_FEATURES + GEO_FEATURES}
            samples.append(Sample(f"{image_id}#{idx}", image_id, str(m["group_id"]), str(m["style"]), ch, x))
    info = {"validation_status": dict(status),
            "excluded_images": {k: v for k, v in status.items() if k not in ("COUNT_MATCH", "EXCLUDED_SCENE")},
            "scene_images_not_used": status.get("EXCLUDED_SCENE", 0)}
    say(f"[CHAR] {len(samples)} character samples from "
        f"{len({s.image_id for s in samples})} COUNT_MATCH images")
    return samples, info


def class_audit(samples: Sequence[Sample]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, List[str]]]:
    """Per-class counts and the formal / informal class overlap."""
    by_class: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.character].append(s)
    dist, overlap = [], []
    groups = {"shared": [], "formal_only": [], "informal_only": []}
    for c in sorted(by_class):
        ss = by_class[c]
        nf = sum(s.style == FORMAL for s in ss)
        ni = sum(s.style == INFORMAL for s in ss)
        kind = "shared" if nf and ni else "formal_only" if nf else "informal_only"
        groups[kind].append(c)
        dist.append({"class": c, "total_count": len(ss), "formal_count": nf, "informal_count": ni,
                     "number_of_source_groups": len({s.group_id for s in ss}),
                     "formal_groups": len({s.group_id for s in ss if s.style == FORMAL}),
                     "informal_groups": len({s.group_id for s in ss if s.style == INFORMAL})})
        overlap.append({"class": c, "domain": {"shared": "A_SHARED", "formal_only": "B_FORMAL_ONLY",
                                               "informal_only": "C_INFORMAL_ONLY"}[kind],
                        "formal_count": nf, "informal_count": ni})
    return dist, overlap, groups


# ====================================================================== folds + leakage checks
@dataclass
class Fold:
    """One train / test split."""

    name: str
    train: List[Sample]
    test: List[Sample]


def check_fold(fold: Fold, fitted_ids: Iterable[str]) -> None:
    """All leakage assertions for one fold; raises LeakageError on any violation."""
    train_groups = {s.group_id for s in fold.train}
    test_groups = {s.group_id for s in fold.test}
    if train_groups & test_groups:
        raise LeakageError(f"{fold.name}: train/test group overlap {sorted(train_groups & test_groups)[:5]}")
    test_ids = {s.sample_id for s in fold.test}
    fitted = set(fitted_ids)
    if test_ids & fitted:
        raise LeakageError(f"{fold.name}: test sample used in prototype/normalization fitting")
    train_images = {s.image_id for s in fold.train}
    test_images = {s.image_id for s in fold.test}
    # variants of one group share the group id; check through image ids too
    variant_group = {s.image_id: s.group_id for s in list(fold.train) + list(fold.test)}
    if {variant_group[i] for i in train_images} & {variant_group[i] for i in test_images}:
        raise LeakageError(f"{fold.name}: variants of one group in train and test")


def logo_folds(name: str, test_pool: Sequence[Sample], extra_train: Sequence[Sample] = ()) -> List[Fold]:
    """Leave-one-group-out over the test pool's groups; extra_train never contains test-pool groups."""
    groups = sorted({s.group_id for s in test_pool})
    pool_groups = set(groups)
    if any(s.group_id in pool_groups for s in extra_train):
        raise LeakageError(f"{name}: extra training data contains test-pool groups")
    folds = []
    for g in groups:
        test = [s for s in test_pool if s.group_id == g]
        train = [s for s in test_pool if s.group_id != g] + list(extra_train)
        folds.append(Fold(f"{name}:{g}", train, test))
    return folds


# ====================================================================== evaluation
def run_folds(folds: Sequence[Fold], cfg: AppConfig) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Fit G / T / TG on each training fold and predict its test samples (label never passed)."""
    c = cfg.character
    weights = {"geometry": (0.0, 1.0), "topology": (1.0, 0.0), "combined": (c.w_topology, c.w_geometry)}
    preds: List[Dict[str, object]] = []
    checks = Counter()
    for fold in folds:
        counts = Counter(s.character for s in fold.train)
        fit_set = [s for s in fold.train if counts[s.character] >= c.min_train_samples_per_class]
        if len({s.character for s in fit_set}) < 2:
            for s in fold.test:
                preds.append(_unevaluable(fold, s, "fewer than 2 classes in training fold"))
            continue
        models = {m: fit_reference([s.x for s in fit_set], [s.character for s in fit_set], TOPO_FEATURES,
                                   GEO_FEATURES, w, cfg) for m, w in weights.items()}
        check_fold(fold, [s.sample_id for s in fit_set])   # prototype + normalization statistics source
        checks["folds_checked"] += 1
        classes = set(models["combined"].classes)
        for s in fold.test:
            if s.character not in classes:
                preds.append(_unevaluable(fold, s, "class not in training fold"))
                continue
            x = dict(s.x)
            if any(k in x for k in ("class", "character", "label", "ground_truth")):
                raise LeakageError("ground truth present in inference input")
            row: Dict[str, object] = {"fold": fold.name, "sample_id": s.sample_id, "group_id": s.group_id,
                                      "style": s.style, "ground_truth": s.character, "evaluable": True,
                                      "n_classes_in_fold": len(classes)}
            for m, model in models.items():
                res = model.clf.predict(x)
                ranked = list(res.posteriors)
                row[f"{m}_top1"] = res.top1
                row[f"{m}_posterior"] = res.top1_posterior
                row[f"{m}_top3"] = "|".join(ranked[:3])
                row[f"{m}_entropy"] = res.entropy
                row[f"{m}_correct"] = res.top1 == s.character
                row[f"{m}_top3_correct"] = s.character in ranked[:3]
            preds.append(row)
            checks["predictions"] += 1
    return preds, dict(checks)


def _unevaluable(fold: Fold, s: Sample, reason: str) -> Dict[str, object]:
    return {"fold": fold.name, "sample_id": s.sample_id, "group_id": s.group_id, "style": s.style,
            "ground_truth": s.character, "evaluable": False, "not_evaluable_reason": reason}


def metrics(preds: Sequence[Dict[str, object]], cfg: AppConfig) -> Dict[str, object]:
    """Accuracy metrics + paired group bootstrap CIs (incl. gains over geometry)."""
    ev = [p for p in preds if p["evaluable"]]
    out: Dict[str, object] = {"n_samples_total": len(preds), "n_samples": len(ev),
                              "n_not_evaluable": len(preds) - len(ev),
                              "n_classes": len({p["ground_truth"] for p in ev}),
                              "n_test_groups": len({p["group_id"] for p in ev})}
    if not ev:
        return out
    for m in MODEL_NAMES:
        per_class = defaultdict(list)
        for p in ev:
            per_class[p["ground_truth"]].append(p[f"{m}_correct"])
        out[m] = {"top1_accuracy": float(np.mean([p[f"{m}_correct"] for p in ev])),
                  "top3_accuracy": float(np.mean([p[f"{m}_top3_correct"] for p in ev])),
                  "macro_accuracy": float(np.mean([np.mean(v) for v in per_class.values()])),
                  "mean_entropy": float(np.mean([p[f"{m}_entropy"] for p in ev]))}
    out["combined_gain_over_geometry_pp"] = 100 * (out["combined"]["top1_accuracy"] - out["geometry"]["top1_accuracy"])
    out["topology_gain_over_geometry_pp"] = 100 * (out["topology"]["top1_accuracy"] - out["geometry"]["top1_accuracy"])
    # paired bootstrap over groups
    groups = sorted({p["group_id"] for p in ev})
    idx = {g: i for i, g in enumerate(groups)}
    n = np.zeros(len(groups))
    corr = {m: np.zeros(len(groups)) for m in MODEL_NAMES}
    for p in ev:
        n[idx[p["group_id"]]] += 1
        for m in MODEL_NAMES:
            corr[m][idx[p["group_id"]]] += p[f"{m}_correct"]
    rng = np.random.default_rng(cfg.character.bootstrap_seed)
    draws = rng.integers(0, len(groups), size=(cfg.character.bootstrap_iterations, len(groups)))
    nn = n[draws].sum(axis=1)
    boot = {m: corr[m][draws].sum(axis=1) / nn for m in MODEL_NAMES}
    for m in MODEL_NAMES:
        out[m]["ci95"] = [float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))]
    for m in ("combined", "topology"):
        d = 100 * (boot[m] - boot["geometry"])
        out[f"{m}_gain_over_geometry_ci95_pp"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    out["ci_method"] = (f"paired bootstrap over {len(groups)} test groups, "
                        f"{cfg.character.bootstrap_iterations} draws, seed {cfg.character.bootstrap_seed}")
    out["ci_unstable"] = len(groups) < cfg.character.ci_unstable_min_groups
    return out


def confused_pairs(preds: Sequence[Dict[str, object]], model: str, top: int = 10) -> List[Dict[str, object]]:
    """Most frequent symmetric confusions actually present in the predictions."""
    cnt = Counter()
    for p in preds:
        if p["evaluable"] and not p[f"{model}_correct"]:
            a, b = sorted([str(p["ground_truth"]), str(p[f"{model}_top1"])])
            cnt[(a, b)] += 1
    return [{"pair": f"{a} <-> {b}", "count": n} for (a, b), n in cnt.most_common(top)]


def per_class(exp: str, preds: Sequence[Dict[str, object]], cfg: AppConfig) -> List[Dict[str, object]]:
    """Per-class accuracies and pre-declared difference categories."""
    c = cfg.character
    rows = []
    by = defaultdict(list)
    for p in preds:
        if p["evaluable"]:
            by[p["ground_truth"]].append(p)
    for cls in sorted(by):
        ps = by[cls]
        acc = {m: float(np.mean([p[f"{m}_correct"] for p in ps])) for m in MODEL_NAMES}
        n = len(ps)
        big = n >= c.large_difference_min_n
        rows.append({"experiment": exp, "class": cls, "n_test": n,
                     "geometry_accuracy": acc["geometry"], "topology_accuracy": acc["topology"],
                     "combined_accuracy": acc["combined"],
                     "topology_minus_geometry": acc["topology"] - acc["geometry"],
                     "combined_minus_geometry": acc["combined"] - acc["geometry"],
                     "topology_much_better": big and acc["topology"] - acc["geometry"] >= c.large_difference,
                     "geometry_better_than_topology": acc["geometry"] > acc["topology"],
                     "combined_improves_over_geometry": acc["combined"] > acc["geometry"]})
    return rows


# ====================================================================== orchestration
@dataclass
class CharacterResult:
    """Everything needed for figures and reports."""

    samples: List[Sample]
    data_info: Dict[str, object]
    distribution: List[Dict[str, object]]
    overlap_groups: Dict[str, List[str]]
    predictions: Dict[str, List[Dict[str, object]]]
    metrics: Dict[str, Dict[str, object]]
    per_class: List[Dict[str, object]]
    leakage: Dict[str, object]
    bd_common: Dict[str, object] = field(default_factory=dict)
    report: Dict[str, object] = field(default_factory=dict)
    out_dir: Path = CHAR_RESULTS_DIR


def run_character_experiment(cfg: AppConfig, out_dir: Path = CHAR_RESULTS_DIR,
                             say: Callable[[str], None] = print) -> CharacterResult:
    """Experiments A-D, assertions, CSV outputs (figures/reports are rendered separately)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    samples, info = load_samples(cfg, say)
    dist, overlap, groups = class_audit(samples)
    write_csv(out_dir / "class_distribution.csv", dist)
    write_csv(out_dir / "domain_class_overlap.csv", overlap)
    formal = [s for s in samples if s.style == FORMAL]
    informal = [s for s in samples if s.style == INFORMAL]
    shared = set(groups["shared"])

    say("[CHAR] A formal->formal | B informal->informal | C formal->informal | D formal+informal->informal")
    folds = {
        "A": logo_folds("A", formal),
        "B": logo_folds("B", informal),
        "C": [Fold("C:formal->informal", [s for s in formal if s.character in shared],
                   [s for s in informal if s.character in shared])],
        "D": logo_folds("D", informal, extra_train=formal),
    }
    preds, checks = {}, {}
    for exp, fs in folds.items():
        preds[exp], checks[exp] = run_folds(fs, cfg)
    mets = {exp: metrics(p, cfg) for exp, p in preds.items()}
    for exp in mets:
        mets[exp]["top10_confusions"] = {m: confused_pairs(preds[exp], m) for m in MODEL_NAMES}
    # B vs D on the identical evaluable subset (fair comparison of adding formal data)
    b_ok = {p["sample_id"] for p in preds["B"] if p["evaluable"]}
    d_ok = {p["sample_id"] for p in preds["D"] if p["evaluable"]}
    common = b_ok & d_ok
    bd = {"n_common_samples": len(common),
          "B_on_common": metrics([p for p in preds["B"] if p["sample_id"] in common], cfg),
          "D_on_common": metrics([p for p in preds["D"] if p["sample_id"] in common], cfg)}
    pc = [row for exp in preds for row in per_class(exp, preds[exp], cfg)]
    write_csv(out_dir / "per_class_results.csv", pc)
    write_csv(out_dir / "predictions.csv", [dict(p, experiment=e) for e, ps in preds.items() for p in ps])
    _write_split_manifest(out_dir, samples, folds, shared)
    _write_experiment_results(out_dir, mets)
    leakage = {"group_overlap": 0, "test_used_for_fitting": False, "test_used_for_normalization": False,
               "ground_truth_used_in_inference": False, "variants_split_across_train_test": False,
               "folds_checked": {e: c.get("folds_checked", 0) for e, c in checks.items()},
               "status": "PASS (all assertions executed per fold; any violation aborts the run)"}
    say(f"[CHAR] experiments done in {time.perf_counter() - t0:.1f} s")
    return CharacterResult(samples, info, dist, groups, preds, mets, pc, leakage, bd, out_dir=out_dir)


def _write_split_manifest(out_dir: Path, samples: Sequence[Sample], folds: Dict[str, List[Fold]],
                          shared: set) -> None:
    role: Dict[str, Dict[str, str]] = defaultdict(dict)
    for exp, fs in folds.items():
        for f in fs:
            for s in f.test:
                role[s.sample_id][exp] = "test" if exp == "C" else f"test@fold={s.group_id}"
            if exp == "C":
                for s in f.train:
                    role[s.sample_id][exp] = "train"
    rows = []
    for s in samples:
        r = role[s.sample_id]
        rows.append({"sample_id": s.sample_id, "image_id": s.image_id, "group_id": s.group_id, "style": s.style,
                     "character": s.character,
                     "split_A_formal_logo": r.get("A", "not_used"),
                     "split_B_informal_logo": r.get("B", "not_used"),
                     "split_C_domainshift": r.get("C", "excluded_non_shared_class"),
                     "split_D_mixed_logo": r.get("D", "train_only(formal, never test)")
                     if s.style == FORMAL else r.get("D", "not_used"),
                     "split": "group-level (all variants of a group_id share the same split in every experiment)"})
    write_csv(out_dir / "split_manifest.csv", rows)


def _write_experiment_results(out_dir: Path, mets: Dict[str, Dict[str, object]]) -> None:
    rows = []
    for exp, m in mets.items():
        for model in MODEL_NAMES:
            if model not in m:
                continue
            x = m[model]
            rows.append({"experiment": exp, "description": EXPERIMENTS[exp][1], "model": model,
                         "top1_accuracy": x["top1_accuracy"], "top1_ci95_low": x["ci95"][0],
                         "top1_ci95_high": x["ci95"][1], "top3_accuracy": x["top3_accuracy"],
                         "macro_accuracy": x["macro_accuracy"], "mean_entropy": x["mean_entropy"],
                         "n_samples": m["n_samples"], "n_classes": m["n_classes"],
                         "n_test_groups": m["n_test_groups"], "n_not_evaluable": m["n_not_evaluable"],
                         "gain_over_geometry_pp": 0.0 if model == "geometry" else
                         m[f"{model}_gain_over_geometry_pp"],
                         "ci_unstable": m["ci_unstable"]})
    write_csv(out_dir / "experiment_results.csv", rows)
