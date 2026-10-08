"""Figures, final_report.json / .txt and terminal summary for UNCERTAINTY VALIDATION."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import visualization as base  # noqa: E402
from .config import AppConfig  # noqa: E402
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, rel  # noqa: E402
from .uncertainty_validation import EXPERIMENTS, MODELS, UVResult  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, RED = base.SURFACE, base.INK, base.INK_2, base.MUTED, base.GRID, base.RED
COLORS = {"geometry": base.AQUA, "topology": base.ORANGE, "combined": base.BLUE}
LABELS = {"geometry": "Geometry", "topology": "Topology", "combined": "Topology + Geometry"}
STAGE_COLORS = {"raw": MUTED, "temperature_scaled": base.VIOLET}
BANNER = "REAL DATA · uncertainty validation · group-level train / calibration / test · no classifier changes"
DPI = 120


def _banner(fig, title: str) -> None:
    fig.text(0.01, 0.988, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.99, 0.988, BANNER, fontsize=9, color=RED, fontweight="bold", va="top", ha="right")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _row(r: UVResult, exp: str, m: str, stage: str) -> Dict:
    return next(x for x in r.calibration if x["experiment"] == exp and x["model"] == m and x["stage"] == stage)


def _ed(r: UVResult, exp: str, m: str, stage: str, score: str) -> Dict:
    return next(x for x in r.error_detection if x["experiment"] == exp and x["model"] == m
                and x["stage"] == stage and x["score"] == score)


def _sel(r: UVResult, exp: str, m: str) -> Dict:
    return next(x for x in r.selective if x["experiment"] == exp and x["model"] == m)


def reliability_figure(r: UVResult, path: Path, cfg: AppConfig) -> Path:
    exps = list(EXPERIMENTS)
    fig, axes = plt.subplots(len(exps), 3, figsize=(15, 4.3 * len(exps)))
    _banner(fig, f"Reliability diagrams (final test, {cfg.uncertainty.ece_bins} equal-width bins on top-1 confidence)")
    for i, exp in enumerate(exps):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color=GRID, lw=1.2)
            for stage in ("raw", "temperature_scaled"):
                bins = [b for b in r.reliability[(exp, m, stage)] if b["n"] > 0]
                ax.plot([b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", ms=4,
                        color=STAGE_COLORS[stage], lw=1.8,
                        label=f"{'raw' if stage == 'raw' else 'temp.-scaled'}  ECE {_row(r, exp, m, stage)[f'ece_{cfg.uncertainty.ece_bins}']:.3f}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{exp}. {EXPERIMENTS[exp]} · {LABELS[m]}", fontsize=9.5)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            ax.grid(color=GRID, lw=0.5)
            if j == 0:
                ax.set_ylabel("accuracy in bin")
            if i == len(exps) - 1:
                ax.set_xlabel("mean confidence in bin")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def confidence_histogram(r: UVResult, path: Path) -> Path:
    exps = list(EXPERIMENTS)
    fig, axes = plt.subplots(len(exps), 3, figsize=(15, 3.6 * len(exps)), sharex=True)
    _banner(fig, "Raw top-1 confidence: correct vs error predictions (final test)")
    edges = np.linspace(0, 1, 21)
    for i, exp in enumerate(exps):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            arr = r.distributions[(exp, m, "raw")]
            c = arr["correct"].astype(bool)
            ax.hist(arr["conf"][c], bins=edges, color=base.BLUE, alpha=0.75, label=f"correct (n={c.sum()})")
            ax.hist(arr["conf"][~c], bins=edges, color=base.ORANGE, alpha=0.75, label=f"error (n={(~c).sum()})")
            ax.set_title(f"{exp}. {LABELS[m]}", fontsize=9.5)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            ax.grid(axis="y", color=GRID, lw=0.5)
            if i == len(exps) - 1:
                ax.set_xlabel("top-1 posterior (raw)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def risk_coverage_figure(r: UVResult, path: Path, cfg: AppConfig) -> Path:
    exps = list(EXPERIMENTS)
    fig, axes = plt.subplots(1, len(exps), figsize=(5 * len(exps), 5.2), sharey=True)
    _banner(fig, "Risk-coverage curves (final test, temperature-scaled confidence); ● = frozen calibration threshold")
    for ax, exp in zip(axes, exps):
        for m in MODELS:
            c = r.curves[(exp, m)]
            ax.plot(c["coverage"], c["risk"], color=COLORS[m], lw=2, label=f"{LABELS[m]} (AURC {c['aurc']:.3f})")
            cov, risk = c["operating_point"]
            if cov > 0:
                ax.scatter([cov], [risk], color=COLORS[m], s=60, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.axhline(cfg.uncertainty.target_risk, color=RED, lw=1, ls="--")
        ax.text(0.02, cfg.uncertainty.target_risk + 0.01, f"target risk {cfg.uncertainty.target_risk:.0%}",
                color=RED, fontsize=8)
        ax.set_title(f"{exp}. {EXPERIMENTS[exp]}", fontsize=9.5)
        ax.set_xlabel("coverage (fraction auto-accepted)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.5)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("selective risk (error rate among accepted)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def domain_shift_figure(r: UVResult, path: Path, cfg: AppConfig) -> Path:
    b = cfg.uncertainty.ece_bins
    metrics = [("accuracy", "Accuracy ↑"), (f"ece_{b}", f"ECE ({b} bins) ↓"), ("nll_evaluable", "NLL (evaluable) ↓"),
               ("brier", "Brier ↓"), ("auroc", "Error-detection AUROC ↑ (max posterior)")]
    exps = ["B", "C", "C2"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5.6))
    _banner(fig, "Informal test: in-domain (B) vs domain shift (C: calibrated on Formal, C2: calibrated on Informal)")
    x = np.arange(len(exps))
    for ax, (key, title) in zip(axes, metrics):
        for i, m in enumerate(MODELS):
            for k, stage in enumerate(("raw", "temperature_scaled")):
                if key == "accuracy" and stage == "temperature_scaled":
                    continue
                vals = [(_ed(r, e, m, stage, "max_posterior")["auroc"] if key == "auroc" else _row(r, e, m, stage)[key])
                        for e in exps]
                pos = x + (i - 1) * 0.27 + (0.07 if stage == "temperature_scaled" else -0.07 if key != "accuracy" else 0)
                ax.bar(pos, vals, 0.12 if key != "accuracy" else 0.25, color=COLORS[m],
                       alpha=1.0 if stage == "raw" else 0.45, edgecolor=SURFACE,
                       label=f"{LABELS[m]} {'raw' if stage == 'raw' else 'temp.'}" if key == f"ece_{b}" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(exps)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.text(0.01, 0.01, "solid = raw posterior, light = temperature-scaled (T from calibration partition only). "
             "Accuracy is unchanged by temperature scaling.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    return _save(fig, path)


def dashboard(r: UVResult, path: Path, cfg: AppConfig) -> Path:
    rep = r.report
    b = cfg.uncertainty.ece_bins
    fig = plt.figure(figsize=(22, 24))
    gs = GridSpec(5, 3, figure=fig, height_ratios=[0.5, 1.3, 1.1, 1.0, 0.85], hspace=0.45, wspace=0.3)
    fig.text(0.012, 0.995, "PAC MISSION 3 · BAYESIAN UNCERTAINTY VALIDATION & CALIBRATION", fontsize=22,
             fontweight="bold", va="top")
    fig.text(0.012, 0.981, "Does the (uncalibrated) posterior predict real errors? group-level train / calibration / "
             "final test · temperature & thresholds from calibration only · classifier unchanged", fontsize=11,
             color=RED, va="top")
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.set_title("MODEL AUDIT (as implemented)", loc="left", fontsize=12, fontweight="bold")
    for i, (k, v) in enumerate(rep["model_audit"].items()):
        if i >= 5:
            break
        ax.text(0, 0.8 - i * 0.19, base.safe_text(f"{k}: {v}")[:210], fontsize=9.5, transform=ax.transAxes)
    # calibration table
    ax = fig.add_subplot(gs[1, :])
    ax.set_axis_off()
    ax.set_title("CALIBRATION on final test (raw → temperature-scaled)", loc="left", fontsize=12, fontweight="bold")
    hdr = ["Experiment", "Model", "Accuracy", f"ECE{b} raw→T", "NLL raw→T", "Brier raw→T", "T (mean)", "conf−acc raw"]
    xs = [0, 0.27, 0.41, 0.5, 0.63, 0.76, 0.87, 0.94]
    for x0, h in zip(xs, hdr):
        ax.text(x0, 0.97, h, fontsize=10, fontweight="bold", transform=ax.transAxes)
    k = 0
    for exp in EXPERIMENTS:
        for m in MODELS:
            ra, te = _row(r, exp, m, "raw"), _row(r, exp, m, "temperature_scaled")
            y = 0.9 - k * 0.073
            vals = [f"{exp}. {EXPERIMENTS[exp]}"[:34], LABELS[m], f"{ra['accuracy']:.3f}",
                    f"{ra[f'ece_{b}']:.3f}→{te[f'ece_{b}']:.3f}", f"{ra['nll_evaluable']:.2f}→{te['nll_evaluable']:.2f}",
                    f"{ra['brier']:.3f}→{te['brier']:.3f}", f"{te['temperature_mean']:.2f}", f"{ra['overconfidence']:+.3f}"]
            for x0, v in zip(xs, vals):
                ax.text(x0, y, v, fontsize=9.5, transform=ax.transAxes,
                        color=RED if (v.startswith("+") and x0 == xs[-1] and float(v) > 0.05) else INK)
            k += 1
    # error detection
    ax = fig.add_subplot(gs[2, 0:2])
    exps = list(EXPERIMENTS)
    x = np.arange(len(exps))
    for i, m in enumerate(MODELS):
        for j, sc in enumerate(("max_posterior", "entropy", "margin")):
            vals = [_ed(r, e, m, "raw", sc)["auroc"] for e in exps]
            ax.bar(x + (i - 1) * 0.27 + (j - 1) * 0.08, vals, 0.08, color=COLORS[m], alpha=[1, 0.65, 0.35][j],
                   edgecolor=SURFACE, label=f"{LABELS[m]} · {sc}" if True else None)
    ax.axhline(0.5, color=INK_2, lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}" for e in exps])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC (positive = ERROR)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("ERROR DETECTION (raw scores; dashed = chance)", loc="left", fontsize=12, fontweight="bold")
    ax = fig.add_subplot(gs[2, 2])
    ax.set_axis_off()
    ax.set_title(f"SELECTIVE (target risk {cfg.uncertainty.target_risk:.0%}, frozen)", loc="left", fontsize=12,
                 fontweight="bold")
    k = 0
    short = {"geometry": "Geo", "topology": "Topo", "combined": "Topo+Geo"}
    for exp in EXPERIMENTS:
        for m in MODELS:
            s = _sel(r, exp, m)
            met = "met" if s["target_met_on_test"] else "NOT met"
            txt = (f"{exp:<2} {short[m]:<8} cov {s['test_coverage']:.2f}  risk "
                   f"{s['test_selective_risk']:.3f} ({met})" if s["test_coverage"] > 0 else
                   f"{exp:<2} {short[m]:<8} cov 0.00 (all to HUMAN_REVIEW)")
            ax.text(0, 0.95 - k * 0.08, txt, fontsize=9, family="monospace", transform=ax.transAxes,
                    color=INK if s["target_met_on_test"] else RED)
            k += 1
    # risk coverage C
    ax = fig.add_subplot(gs[3, 0])
    for m in MODELS:
        c = r.curves[("B", m)]
        ax.plot(c["coverage"], c["risk"], color=COLORS[m], lw=2, label=LABELS[m])
    ax.axhline(cfg.uncertainty.target_risk, color=RED, ls="--", lw=1)
    ax.set_title("Risk-coverage: B Informal → Informal", fontsize=11)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=0.5)
    ax = fig.add_subplot(gs[3, 1])
    for m in MODELS:
        c = r.curves[("C", m)]
        ax.plot(c["coverage"], c["risk"], color=COLORS[m], lw=2, label=LABELS[m])
    ax.axhline(cfg.uncertainty.target_risk, color=RED, ls="--", lw=1)
    ax.set_title("Risk-coverage: C Formal → Informal", fontsize=11)
    ax.set_xlabel("coverage")
    ax.grid(color=GRID, lw=0.5)
    ax = fig.add_subplot(gs[3, 2])
    for m in MODELS:
        bins = [x_ for x_ in r.reliability[("C", m, "raw")] if x_["n"] > 0]
        ax.plot([x_["mean_confidence"] for x_ in bins], [x_["accuracy"] for x_ in bins], marker="o", color=COLORS[m],
                lw=2, label=LABELS[m])
    ax.plot([0, 1], [0, 1], color=GRID)
    ax.set_title("Reliability (raw): C Formal → Informal", fontsize=11)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=0.5)
    ax = fig.add_subplot(gs[4, :])
    ax.set_axis_off()
    ax.set_title("LIMITATIONS", loc="left", fontsize=12, fontweight="bold")
    for i, t in enumerate(rep["limitations"]["other"][:5] + [rep["limitations"]["ood"]]):
        ax.text(0, 0.92 - i * 0.15, "• " + base.safe_text(t)[:230], fontsize=10, transform=ax.transAxes, color=INK_2)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.02)
    return _save(fig, path)


# ====================================================================== report
def build_report(r: UVResult, cfg: AppConfig) -> Dict[str, object]:
    u = cfg.uncertainty
    c = cfg.character
    b = u.ece_bins
    audit = {
        "Likelihood": ("independent per-feature Gaussian: log N(x_f | mu_cf, sigma_cf^2) without the constant "
                       "-0.5 log 2pi (cancels after normalisation); group score = MEAN over the group's features "
                       f"({len(TOPO_FEATURES)} topology, {len(GEO_FEATURES)} geometry)"),
        "Prior": "uniform over the classes present in the training partition (log 1/K)",
        "Posterior normalization": ("log p(c|x) = s_c + log(1/K) - logsumexp_k(s_k + log(1/K)); closed set = "
                                    "training classes only (classes unseen in training receive probability 0)"),
        "Entropy": "H = -sum_c p_c ln p_c (nats) over the training classes",
        "Combined weighting": (f"s_c = {c.w_topology} * meanTopoLogLik + {c.w_geometry} * meanGeoLogLik "
                               f"(= sum_topo/{len(TOPO_FEATURES) / c.w_topology:.0f} + sum_geo/"
                               f"{len(GEO_FEATURES) / c.w_geometry:.0f}); Geometry-only / Topology-only use weight 1 "
                               "on one group"),
        "Variance": (f"sigma_cf = max(sqrt((n_c s_cf^2 + lambda pooled_f) / (n_c + lambda)), floor_f); lambda = "
                     f"{cfg.real.variance_shrinkage_lambda}, floor_f = {cfg.real.sigma_floor_relative} x training "
                     f"std of f (absolute {cfg.real.sigma_floor_absolute} for zero-variance features)"),
        "Interpretation": ("because log-likelihoods are AVERAGED (not summed) the result is a tempered "
                           "pseudo-posterior p ∝ prior * prod_f N(.)^(w/n_group), not a proper naive-Bayes posterior; "
                           "it is not calibrated by construction"),
        "Temperature scaling validity": ("log p is a normalised log posterior; softmax(log p / T) = softmax(s / T), "
                                         "i.e. T rescales the effective likelihood exponent. Monotone per sample: "
                                         "top-1 (accuracy) unchanged"),
        "Verification": "exact log posterior re-computed and asserted equal to BayesianShapeClassifier.predict",
    }
    fold_info = {e: o.fold_info for e, o in r.outputs.items()}
    overlaps = {e: o.overlaps for e, o in r.outputs.items()}
    total_overlap = {k: sum(ov[k] for o in r.outputs.values() for ov in o.overlaps)
                     for k in ("train_cal", "train_test", "cal_test", "sample_overlap")}
    single = [d["class"] for d in r.data_info["class_distribution"] if d["number_of_source_groups"] < 2]
    unseen = {e: sum(f["test_unseen_class_samples"] for f in o.fold_info) for e, o in r.outputs.items()}
    small_cal = {e: min(f["cal_groups"] for f in o.fold_info) for e, o in r.outputs.items()}
    lims_other = [
        f"Classes unseen in a fold's training partition are counted as errors (test samples affected: {unseen}); "
        "NLL is reported on the evaluable subset only.",
        f"Calibration partitions are small (minimum calibration groups per fold: {small_cal}); temperatures and "
        "thresholds vary across folds (see calibration_metrics.csv temperature_per_fold).",
        "Training partitions are smaller than in the leave-one-group-out character experiment, so accuracies are "
        "lower and not directly comparable to it.",
        "Variants _1/_2/_3 of a group are near-duplicates; effective sample size = number of groups (CIs use group "
        "bootstrap).",
        "Temperature scaling cannot change accuracy or the ranking used for error detection within a sample; it only "
        "rescales confidence.",
        "The 5% target risk is checked, not guaranteed: thresholds are point estimates on small calibration sets.",
    ]
    return {
        "experiment": "BAYESIAN UNCERTAINTY VALIDATION & CALIBRATION (REAL DATA)",
        "model_audit": audit,
        "protocol": {"splits": {"A/B": f"outer {u.outer_folds}-fold group split = final test; outer-train split "
                                       f"{1 - u.calibration_fraction:.0%}/{u.calibration_fraction:.0%} train/calibration "
                                       "(group level)",
                                "C": f"train/cal on Formal ({u.domain_shift_formal_parts} group parts, one part = "
                                     "calibration), final test = Informal fold (shared classes)",
                                "C2": "same train and test as C; calibration on Informal groups disjoint from test"},
                     "ece": f"equal-width {b} bins on top-1 confidence (primary); sensitivity "
                            f"{list(u.ece_bins_sensitivity)} bins and {b} equal-mass bins",
                     "error_detection_positive_class": "ERROR (wrong top-1)",
                     "temperature": "argmin calibration NLL over log T in "
                                    f"[{u.temperature_bounds[0]}, {u.temperature_bounds[1]}]",
                     "selective_rule": f"largest calibration coverage with calibration risk <= {u.target_risk}; "
                                       "threshold frozen per fold",
                     "seed": u.seed, "bootstrap": f"{u.bootstrap_iterations} group draws",
                     "classifier": "unchanged (features, weights 0.5/0.5, shrinkage, floors)"},
        "fold_partitions": fold_info,
        "leakage": {"train_calibration_group_overlap": total_overlap["train_cal"],
                    "train_test_group_overlap": total_overlap["train_test"],
                    "calibration_test_group_overlap": total_overlap["cal_test"],
                    "sample_overlap": total_overlap["sample_overlap"],
                    "temperature_and_threshold_source": "calibration partition only (test untouched until frozen)",
                    "status": "PASS" if not any(total_overlap.values()) else "FAIL", "per_fold": overlaps},
        "calibration": r.calibration, "error_detection": r.error_detection, "selective": r.selective,
        "class_novelty_proxy": r.novelty_proxy,
        "limitations": {
            "unevaluable_classes": {"single_source_group_classes": single,
                                    "note": "never present in both training and test under group splits"},
            "insufficient_group_counts": {e: {"min_cal_groups": small_cal[e],
                                              "test_groups_total": sum(f["test_groups"] for f in o.fold_info)}
                                          for e, o in r.outputs.items()},
            "ood": ("OOD detection NOT EVALUABLE: no out-of-distribution character/symbol set with labels exists in "
                    "this data. class_novelty_proxy.csv only measures confidence on characters whose class was absent "
                    "from training - a weak proxy, not an OOD validation. A closed-set posterior can be high on "
                    "unknown inputs (seen earlier: welding model on text)."),
            "other": lims_other,
        },
    }


def terminal_summary(r: UVResult, cfg: AppConfig) -> str:
    rep = r.report
    b = cfg.uncertainty.ece_bins
    a = rep["model_audit"]
    o = ["=" * 60, "PAC BAYESIAN UNCERTAINTY VALIDATION (REAL DATA)", "=" * 60, "", "[MODEL AUDIT]",
         f"Likelihood = {a['Likelihood']}", f"Prior = {a['Prior']}",
         f"Posterior normalization = {a['Posterior normalization']}", f"Entropy = {a['Entropy']}",
         f"Combined weighting = {a['Combined weighting']}", f"Interpretation = {a['Interpretation']}", "",
         f"[CALIBRATION]  (final test; ECE = {b} equal-width bins; NLL on evaluable subset)",
         f"{'Model':<10}{'Dataset':<44}{'Acc':>7}{'ECE':>16}{'NLL':>16}{'Brier':>16}{'T':>7}"]
    for exp in EXPERIMENTS:
        for m in MODELS:
            ra, te = _row(r, exp, m, "raw"), _row(r, exp, m, "temperature_scaled")
            o.append(f"{m:<10}{exp + '. ' + EXPERIMENTS[exp]:<44}{ra['accuracy']:>7.3f}"
                     f"{ra[f'ece_{b}']:>8.3f}->{te[f'ece_{b}']:<6.3f}{ra['nll_evaluable']:>8.2f}->{te['nll_evaluable']:<6.2f}"
                     f"{ra['brier']:>8.3f}->{te['brier']:<6.3f}{te['temperature_mean']:>7.2f}")
    o += ["(Before calibration -> After calibration; accuracy is unchanged by temperature scaling)", "",
          "[ERROR DETECTION]  (AUROC, positive class = ERROR, raw scores; 95% CI group bootstrap)"]
    for exp in EXPERIMENTS:
        for m in MODELS:
            e1, e2, e3 = (_ed(r, exp, m, "raw", s) for s in ("max_posterior", "entropy", "margin"))
            o.append(f"{exp:<3}{m:<10} Maximum posterior AUROC = {e1['auroc']:.3f} {e1['auroc_ci95']}  "
                     f"Entropy AUROC = {e2['auroc']:.3f}  Margin AUROC = {e3['auroc']:.3f}  "
                     f"(error rate {e1['error_prevalence']:.3f})")
    o += ["", f"[SELECTIVE CLASSIFICATION]  (target risk {cfg.uncertainty.target_risk:.0%}; threshold from calibration, frozen)"]
    for exp in EXPERIMENTS:
        for m in MODELS:
            s = _sel(r, exp, m)
            if s["test_coverage"] > 0:
                o.append(f"{exp:<3}{m:<10} Coverage = {s['test_coverage']:.3f}  Selective Accuracy = "
                         f"{s['test_selective_accuracy']:.3f}  Selective Risk = {s['test_selective_risk']:.3f}  "
                         f"target {'met' if s['target_met_on_test'] else 'NOT met'}  (AURC {s['test_aurc']:.3f})")
            else:
                o.append(f"{exp:<3}{m:<10} Coverage = 0.000 (no calibration threshold reached the target -> all "
                         f"HUMAN_REVIEW)  (AURC {s['test_aurc']:.3f})")
    o += ["", "[DOMAIN SHIFT]  (Informal test: B in-domain vs C Formal-trained & Formal-calibrated vs C2 Informal-calibrated)"]
    for m in MODELS:
        rb, rc, rc2 = (_row(r, e, m, "raw") for e in ("B", "C", "C2"))
        tc, tc2 = _row(r, "C", m, "temperature_scaled"), _row(r, "C2", m, "temperature_scaled")
        o.append(f"{m:<10} calibration: B acc {rb['accuracy']:.3f} ECE {rb[f'ece_{b}']:.3f} | C acc {rc['accuracy']:.3f} "
                 f"ECE raw {rc[f'ece_{b}']:.3f} -> T(formal) {tc[f'ece_{b}']:.3f} | C2 T(informal) {tc2[f'ece_{b}']:.3f}; "
                 f"conf-acc gap C raw {rc['overconfidence']:+.3f}")
        o.append(f"{'':<10} error detection AUROC (max posterior): B {_ed(r, 'B', m, 'raw', 'max_posterior')['auroc']:.3f}"
                 f" | C {_ed(r, 'C', m, 'raw', 'max_posterior')['auroc']:.3f}; selective risk C "
                 f"{_sel(r, 'C', m)['test_selective_risk']:.3f} @ coverage {_sel(r, 'C', m)['test_coverage']:.3f}")
    lk = rep["leakage"]
    o += ["", "[LEAKAGE CHECK]", f"Training/calibration overlap = {lk['train_calibration_group_overlap']}",
          f"Training/test overlap = {lk['train_test_group_overlap']}",
          f"Calibration/test overlap = {lk['calibration_test_group_overlap']}",
          f"Sample overlap = {lk['sample_overlap']}", f"Status = {lk['status']}", "", "[LIMITATIONS]",
          f"Unevaluable classes = {rep['limitations']['unevaluable_classes']['single_source_group_classes']} "
          "(single source group)",
          f"Insufficient group counts = {rep['limitations']['insufficient_group_counts']}",
          f"OOD limitations = {rep['limitations']['ood']}", "Other limitations ="]
    o += [f"  - {t}" for t in rep["limitations"]["other"]] + ["=" * 60]
    return "\n".join(o)


def render_and_save(r: UVResult, cfg: AppConfig) -> Dict[str, Path]:
    r.report = build_report(r, cfg)
    out = r.out_dir
    files = {}
    for name, fn in (("reliability_diagram", lambda: reliability_figure(r, out / "reliability_diagram.png", cfg)),
                     ("confidence_histogram", lambda: confidence_histogram(r, out / "confidence_histogram.png")),
                     ("risk_coverage_curve", lambda: risk_coverage_figure(r, out / "risk_coverage_curve.png", cfg)),
                     ("domain_shift_calibration", lambda: domain_shift_figure(r, out / "domain_shift_calibration.png", cfg)),
                     ("final_dashboard", lambda: dashboard(r, out / "final_dashboard.png", cfg))):
        try:
            files[name] = fn()
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] {name}: {type(exc).__name__}: {exc}")
    r.report["figures"] = {k: rel(v) for k, v in files.items()}
    (out / "final_report.json").write_text(json.dumps(r.report, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    (out / "final_report.txt").write_text(terminal_summary(r, cfg), encoding="utf-8")
    return files
