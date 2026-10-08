"""Figures, final_report.json / .txt and terminal summary for the CHARACTER EXPERIMENT."""

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
from .character_experiment import EXPERIMENTS, MODEL_NAMES, CharacterResult  # noqa: E402
from .config import AppConfig  # noqa: E402
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, rel  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, RED = base.SURFACE, base.INK, base.INK_2, base.MUTED, base.GRID, base.RED
COLORS = {"geometry": base.AQUA, "topology": base.ORANGE, "combined": base.BLUE}
LABELS = {"geometry": "G  Geometry only", "topology": "T  Topology only", "combined": "TG Topology + Geometry"}
BANNER = "REAL DATA · CHARACTER EXPERIMENT · group-level splits · UNCALIBRATED MODEL POSTERIORS"
DPI = 120


def _banner(fig: plt.Figure, title: str) -> None:
    fig.text(0.01, 0.988, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.99, 0.988, BANNER, fontsize=9, color=RED, fontweight="bold", va="top", ha="right")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _pp(v: float) -> str:
    return f"{v:+.1f} pp"


# ====================================================================== figures
def confusion_png(r: CharacterResult, exp: str, model: str, path: Path) -> Path:
    preds = [p for p in r.predictions[exp] if p["evaluable"]]
    classes = sorted({p["ground_truth"] for p in preds} | {p[f"{model}_top1"] for p in preds})
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), int)
    for p in preds:
        m[idx[p["ground_truth"]], idx[p[f"{model}_top1"]]] += 1
    size = max(8, 0.32 * len(classes) + 3)
    fig, ax = plt.subplots(figsize=(size, size))
    met = r.metrics[exp][model]
    _banner(fig, f"{exp}. {EXPERIMENTS[exp][1]} · {LABELS[model]}")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
    ax.imshow(m, cmap=cmap, vmin=0, vmax=max(3, m.max()))
    if len(classes) <= 50:
        for i in range(len(classes)):
            for j in range(len(classes)):
                if m[i, j]:
                    ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=6.5,
                            color="white" if m[i, j] > m.max() / 2 else INK)
    labels = [base.safe_text(c) for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted Top-1")
    ax.set_ylabel("ground truth (manual transcription)")
    ax.set_title(f"Top-1 {met['top1_accuracy']:.3f}  95% CI [{met['ci95'][0]:.3f}, {met['ci95'][1]:.3f}]  "
                 f"· n={r.metrics[exp]['n_samples']} samples, {r.metrics[exp]['n_classes']} classes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def _bars_with_ci(ax: plt.Axes, r: CharacterResult, exps: List[str]) -> None:
    x = np.arange(len(exps))
    w = 0.26
    for i, m in enumerate(MODEL_NAMES):
        vals, lo, hi = [], [], []
        for e in exps:
            met = r.metrics[e].get(m)
            v = met["top1_accuracy"] if met else 0.0
            vals.append(v)
            lo.append(v - met["ci95"][0] if met else 0)
            hi.append(met["ci95"][1] - v if met else 0)
        pos = x + (i - 1) * w
        ax.bar(pos, vals, w, color=COLORS[m], edgecolor=SURFACE, linewidth=2, label=LABELS[m])
        ax.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor=INK_2, capsize=3, lw=1)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}. {EXPERIMENTS[e][1]}\n(n={r.metrics[e]['n_samples']}, "
                        f"{r.metrics[e]['n_classes']} cls)" for e in exps], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Top-1 accuracy (error bar: 95% CI, group bootstrap)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")


def accuracy_comparison(r: CharacterResult, path: Path) -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.25})
    _banner(fig, "Geometry vs Topology vs Topology+Geometry (Top-1, same splits per experiment)")
    exps = [e for e in EXPERIMENTS if "combined" in r.metrics[e]]
    _bars_with_ci(ax, r, exps)
    gains = [r.metrics[e]["combined_gain_over_geometry_pp"] for e in exps]
    cis = [r.metrics[e]["combined_gain_over_geometry_ci95_pp"] for e in exps]
    tg = [r.metrics[e]["topology_gain_over_geometry_pp"] for e in exps]
    y = np.arange(len(exps))
    ax2.barh(y + 0.18, gains, 0.34, color=COLORS["combined"], edgecolor=SURFACE, label="TG − G")
    ax2.errorbar(gains, y + 0.18, xerr=[[g - c[0] for g, c in zip(gains, cis)], [c[1] - g for g, c in zip(gains, cis)]],
                 fmt="none", ecolor=INK_2, capsize=3)
    ax2.barh(y - 0.18, tg, 0.34, color=COLORS["topology"], edgecolor=SURFACE, label="T − G")
    ax2.axvline(0, color=INK_2, lw=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{e}. {EXPERIMENTS[e][1]}" for e in exps], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("gain over geometry (percentage points)")
    for yi, g in zip(y, gains):
        ax2.text(g, yi + 0.18, f" {g:+.1f}", va="center", fontsize=9)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(axis="x", color=GRID, lw=0.6)
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.05, right=0.98)
    return _save(fig, path)


def domain_shift_comparison(r: CharacterResult, path: Path) -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.25})
    _banner(fig, "Informal (handwritten) test data: in-domain vs domain shift vs mixed training")
    _bars_with_ci(ax, r, [e for e in ("B", "C", "D") if "combined" in r.metrics[e]])
    bd = r.bd_common
    b, d = bd["B_on_common"], bd["D_on_common"]
    if "combined" in b and "combined" in d:
        x = np.arange(3)
        ax2.bar(x - 0.2, [b[m]["top1_accuracy"] for m in MODEL_NAMES], 0.4, color=MUTED, edgecolor=SURFACE,
                label="B Informal only (train)")
        ax2.bar(x + 0.2, [d[m]["top1_accuracy"] for m in MODEL_NAMES], 0.4, color=base.VIOLET, edgecolor=SURFACE,
                label="D Formal + Informal (train)")
        for xi, m in zip(x, MODEL_NAMES):
            ax2.text(xi - 0.2, b[m]["top1_accuracy"] + 0.01, f"{b[m]['top1_accuracy']:.3f}", ha="center", fontsize=8)
            ax2.text(xi + 0.2, d[m]["top1_accuracy"] + 0.01, f"{d[m]['top1_accuracy']:.3f}", ha="center", fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(["G", "T", "TG"])
        ax2.set_ylim(0, 1.1)
        ax2.set_title(f"B vs D on the identical evaluable subset (n={bd['n_common_samples']})", fontsize=10)
        ax2.legend(frameon=False, fontsize=9)
        ax2.grid(axis="y", color=GRID, lw=0.6)
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.05, right=0.98)
    return _save(fig, path)


def dashboard(r: CharacterResult, path: Path) -> Path:
    rep = r.report
    fig = plt.figure(figsize=(22, 24))
    gs = GridSpec(5, 6, figure=fig, height_ratios=[0.38, 1.15, 1.0, 0.95, 0.75], hspace=0.5, wspace=0.5)
    fig.text(0.012, 0.995, "PAC MISSION 3 · CHARACTER EXPERIMENT (REAL DATA)", fontsize=24, fontweight="bold", va="top")
    fig.text(0.012, 0.981, "Does topology add information over geometry — formal, handwritten and domain shift? "
             "Source: data/input/Symbols (COUNT_MATCH characters) · group-level splits · fixed 0.5/0.5 weights · "
             "no restoration", fontsize=11, color=RED, va="top")
    ds = rep["dataset"]
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    tiles = [("Characters", ds["characters"]), ("Classes", ds["classes"]), ("Formal chars", ds["formal_characters"]),
             ("Informal chars", ds["informal_characters"]), ("Shared classes", ds["shared_classes"]),
             ("Formal-only", ds["formal_only_classes"]), ("Informal-only", ds["informal_only_classes"]),
             ("Leakage check", rep["leakage"]["status"].split()[0])]
    for i, (k, v) in enumerate(tiles):
        x0 = i / len(tiles)
        ax.add_patch(Rectangle((x0 + 0.004, 0), 1 / len(tiles) - 0.008, 0.9, transform=ax.transAxes,
                               facecolor=SURFACE, edgecolor=GRID))
        ax.text(x0 + 0.5 / len(tiles), 0.52, str(v), transform=ax.transAxes, ha="center", fontsize=22, fontweight="bold")
        ax.text(x0 + 0.5 / len(tiles), 0.14, k, transform=ax.transAxes, ha="center", fontsize=11, color=INK_2)
    ax = fig.add_subplot(gs[1, :4])
    _bars_with_ci(ax, r, [e for e in EXPERIMENTS if "combined" in r.metrics[e]])
    ax.set_title("Top-1 accuracy by experiment and feature group", loc="left", fontsize=12, fontweight="bold")
    ax = fig.add_subplot(gs[1, 4:])
    ax.set_axis_off()
    ax.set_title("Gain over geometry (percentage points, 95% CI)", loc="left", fontsize=12, fontweight="bold")
    for i, e in enumerate([e for e in EXPERIMENTS if "combined" in r.metrics[e]]):
        m = r.metrics[e]
        y = 0.9 - i * 0.23
        ax.text(0.0, y, f"{e}. {EXPERIMENTS[e][1]}", fontsize=11, transform=ax.transAxes, fontweight="bold")
        tg, ci = m["combined_gain_over_geometry_pp"], m["combined_gain_over_geometry_ci95_pp"]
        t, cit = m["topology_gain_over_geometry_pp"], m["topology_gain_over_geometry_ci95_pp"]
        ax.text(0.02, y - 0.07, f"TG − G = {tg:+.1f} pp  [{ci[0]:+.1f}, {ci[1]:+.1f}]", fontsize=11,
                transform=ax.transAxes, color=INK if tg >= 0 else RED)
        ax.text(0.02, y - 0.13, f"T − G  = {t:+.1f} pp  [{cit[0]:+.1f}, {cit[1]:+.1f}]", fontsize=11,
                transform=ax.transAxes, color=INK if t >= 0 else RED)
    ax = fig.add_subplot(gs[2, :3])
    if "combined" in r.bd_common["B_on_common"]:
        b, d = r.bd_common["B_on_common"], r.bd_common["D_on_common"]
        x = np.arange(3)
        ax.bar(x - 0.2, [b[m]["top1_accuracy"] for m in MODEL_NAMES], 0.4, color=MUTED, edgecolor=SURFACE,
               label="B: train Informal only")
        ax.bar(x + 0.2, [d[m]["top1_accuracy"] for m in MODEL_NAMES], 0.4, color=base.VIOLET, edgecolor=SURFACE,
               label="D: train Formal + Informal")
        for xi, m in zip(x, MODEL_NAMES):
            ax.text(xi - 0.2, b[m]["top1_accuracy"] + 0.01, f"{b[m]['top1_accuracy']:.3f}", ha="center")
            ax.text(xi + 0.2, d[m]["top1_accuracy"] + 0.01, f"{d[m]['top1_accuracy']:.3f}", ha="center")
        ax.set_xticks(x)
        ax.set_xticklabels(["Geometry", "Topology", "Topo+Geo"])
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=False)
        ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title(f"Does adding Formal data help Informal? (identical subset, n={r.bd_common['n_common_samples']})",
                 loc="left", fontsize=12, fontweight="bold")
    ax = fig.add_subplot(gs[2, 3:])
    ax.set_axis_off()
    ax.set_title("Top confusions (Topo+Geo model, actual results)", loc="left", fontsize=12, fontweight="bold")
    for i, e in enumerate([e for e in EXPERIMENTS if "combined" in r.metrics[e]]):
        pairs = r.metrics[e]["top10_confusions"]["combined"][:6]
        ax.text(0.0, 0.92 - i * 0.24, f"{e}. {EXPERIMENTS[e][1]}", fontsize=10.5, fontweight="bold", transform=ax.transAxes)
        ax.text(0.02, 0.84 - i * 0.24, base.safe_text("  ".join(f"{p['pair']}×{p['count']}" for p in pairs) or "none"),
                fontsize=10, transform=ax.transAxes)
    ax = fig.add_subplot(gs[3, :])
    pcs = [p for p in r.per_class if p["experiment"] == "B" and p["n_test"] >= r.report["protocol"]["large_difference_min_n"]]
    pcs.sort(key=lambda p: p["topology_minus_geometry"])
    if pcs:
        y = np.arange(len(pcs))
        ax.barh(y, [p["topology_minus_geometry"] * 100 for p in pcs], color=[COLORS["topology"] if p["topology_minus_geometry"] >= 0
                                                                           else COLORS["geometry"] for p in pcs],
                edgecolor=SURFACE)
        ax.set_yticks(y)
        ax.set_yticklabels([base.safe_text(f"'{p['class']}' (n={p['n_test']})") for p in pcs], fontsize=8)
        ax.axvline(0, color=INK_2)
        ax.set_xlabel("Topology − Geometry per-class accuracy (pp), Experiment B (handwritten), classes with n ≥ "
                      f"{r.report['protocol']['large_difference_min_n']}")
        ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title("Per-class: where topology beats geometry (orange) and vice versa (aqua) — handwritten",
                 loc="left", fontsize=12, fontweight="bold")
    ax = fig.add_subplot(gs[4, :])
    ax.set_axis_off()
    ax.set_title("LIMITATIONS", loc="left", fontsize=12, fontweight="bold")
    for i, t in enumerate(rep["limitations"][:9]):
        ax.text(0.0, 0.92 - i * 0.11, "• " + base.safe_text(t), fontsize=10.5, transform=ax.transAxes, color=INK_2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.955, bottom=0.02)
    return _save(fig, path)


# ====================================================================== report
def build_report(r: CharacterResult, cfg: AppConfig) -> Dict[str, object]:
    g = r.overlap_groups
    s = r.samples
    exps = [e for e in EXPERIMENTS if "combined" in r.metrics[e]]
    best = max(exps, key=lambda e: r.metrics[e]["combined_gain_over_geometry_pp"]) if exps else None
    few = sum(1 for d in r.distribution if d["number_of_source_groups"] < 2)
    lim = [
        f"{few} of {len(r.distribution)} classes come from a single source group: they can never be evaluated "
        "under group-level splits (counted as not evaluable, never forced).",
        f"Informal-only classes ({len(g['informal_only'])}) and formal-only classes ({len(g['formal_only'])}) are "
        "excluded from the domain-shift experiment C by design.",
        "Variants _1/_2/_3 of a group are near-duplicate renderings; the effective sample size is the number of "
        "groups, not characters (CIs use group bootstrap).",
        "Labels are manual transcriptions aligned by segment order; COUNT_MATCH cannot detect two compensating "
        "segmentation errors in one image.",
        f"Excluded images (not repaired): {r.data_info['excluded_images']}.",
        "Posteriors are UNCALIBRATED MODEL POSTERIORS; entropy is over the classes present in each training fold.",
        "Gaussian class-conditional model with small per-class samples (variance shrinkage + floors); "
        "no deep learning, no restoration.",
        "Features are character-level only (no context / language model); upper/lower case are separate classes.",
    ]
    unstable = [e for e in exps if r.metrics[e]["ci_unstable"]]
    if unstable:
        lim.append(f"CI unstable (fewer than {cfg.character.ci_unstable_min_groups} test groups): {unstable}")
    return {
        "experiment": "CHARACTER EXPERIMENT (REAL DATA)",
        "dataset_source": "data/input/Symbols (COUNT_MATCH images, manual transcriptions)",
        "research_question": "Do topology-based structural features add classification information over "
                             "geometry-only features, especially for handwritten/informal markings and domain shift?",
        "dataset": {"characters": len(s), "images": len({x.image_id for x in s}), "groups": len({x.group_id for x in s}),
                    "classes": len(r.distribution),
                    "formal_characters": sum(x.style == "FORMAL" for x in s),
                    "informal_characters": sum(x.style == "INFORMAL" for x in s),
                    "shared_classes": len(g["shared"]), "formal_only_classes": len(g["formal_only"]),
                    "informal_only_classes": len(g["informal_only"]),
                    "shared_class_list": g["shared"], "formal_only_list": g["formal_only"],
                    "informal_only_list": g["informal_only"], **r.data_info},
        "protocol": {
            "models": {"geometry": GEO_FEATURES, "topology": TOPO_FEATURES,
                       "combined": f"{cfg.character.w_topology} x mean topology log-lik + "
                                   f"{cfg.character.w_geometry} x mean geometry log-lik"},
            "feature_choice": "identical to the welding experiment; fixed before this run",
            "variance": f"per-class variance shrunk toward pooled within-class variance "
                        f"(lambda={cfg.real.variance_shrinkage_lambda}); sigma floor = "
                        f"{cfg.real.sigma_floor_relative} x training-fold feature std "
                        f"(absolute {cfg.real.sigma_floor_absolute} for zero-variance features)",
            "splits": {"A": "leave-one-formal-group-out", "B": "leave-one-informal-group-out",
                       "C": "train all formal, test all informal, shared classes only",
                       "D": "leave-one-informal-group-out, training = all formal + other informal groups"},
            "evaluable_rule": "test sample class must occur in the training fold",
            "restoration": "not used",
            "ci": f"paired group bootstrap, {cfg.character.bootstrap_iterations} draws, seed {cfg.character.bootstrap_seed}",
            "large_difference": cfg.character.large_difference,
            "large_difference_min_n": cfg.character.large_difference_min_n,
            "posterior_type": "UNCALIBRATED MODEL POSTERIOR",
            "tuning": "none (baseline; first run preserved)",
        },
        "experiments": {e: dict(r.metrics[e], description=EXPERIMENTS[e][1]) for e in EXPERIMENTS},
        "B_vs_D_identical_subset": r.bd_common,
        "per_class_summary": {
            e: {"topology_much_better": [p["class"] for p in r.per_class if p["experiment"] == e and p["topology_much_better"]],
                "geometry_better_than_topology": [p["class"] for p in r.per_class if p["experiment"] == e
                                                  and p["geometry_better_than_topology"]],
                "combined_improves_over_geometry": [p["class"] for p in r.per_class if p["experiment"] == e
                                                    and p["combined_improves_over_geometry"]]} for e in EXPERIMENTS},
        "top_contribution": None if best is None else {
            "experiment": f"{best}. {EXPERIMENTS[best][1]}",
            "geometry": r.metrics[best]["geometry"]["top1_accuracy"],
            "topology": r.metrics[best]["topology"]["top1_accuracy"],
            "combined": r.metrics[best]["combined"]["top1_accuracy"],
            "gain_pp": r.metrics[best]["combined_gain_over_geometry_pp"],
            "gain_ci95_pp": r.metrics[best]["combined_gain_over_geometry_ci95_pp"]},
        "leakage": r.leakage,
        "limitations": lim,
    }


def terminal_summary(r: CharacterResult) -> str:
    rep = r.report
    ds = rep["dataset"]
    out = ["=" * 50, "PAC CHARACTER EXPERIMENT (REAL DATA)", "=" * 50, "", "[DATASET]", "",
           f"Characters = {ds['characters']}", f"Classes = {ds['classes']}",
           f"Formal characters = {ds['formal_characters']}", f"Informal characters = {ds['informal_characters']}",
           f"Shared classes = {ds['shared_classes']}", f"Formal-only classes = {ds['formal_only_classes']}",
           f"Informal-only classes = {ds['informal_only_classes']}"]
    for e, (_, title) in EXPERIMENTS.items():
        m = r.metrics[e]
        out += ["", f"[{title}]", f"(n = {m['n_samples']} evaluated samples, {m['n_classes']} classes, "
                f"{m['n_test_groups']} test groups; not evaluable = {m['n_not_evaluable']})"]
        if "combined" not in m:
            out.append("NOT EVALUABLE")
            continue
        for model, name in (("geometry", "Geometry"), ("topology", "Topology"), ("combined", "Combined")):
            x = m[model]
            out += ["", f"{name}:", f"Top1 = {x['top1_accuracy']:.4f}", f"Top3 = {x['top3_accuracy']:.4f}",
                    f"Macro = {x['macro_accuracy']:.4f}", f"95% CI = [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]"
                    + ("  (UNSTABLE: few test groups)" if m["ci_unstable"] else "")]
        ci = m["combined_gain_over_geometry_ci95_pp"]
        out += ["", f"Combined gain over Geometry = {_pp(m['combined_gain_over_geometry_pp'])} "
                f"(95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}] pp)",
                f"Topology gain over Geometry = {_pp(m['topology_gain_over_geometry_pp'])}",
                "Top-10 confusions (Combined): " + (", ".join(f"{p['pair']}×{p['count']}"
                                                             for p in m["top10_confusions"]["combined"]) or "none")]
    bd = r.bd_common
    if "combined" in bd["B_on_common"]:
        out += ["", f"[B vs D on identical subset, n={bd['n_common_samples']}]",
                "  " + "  ".join(f"{m}: B {bd['B_on_common'][m]['top1_accuracy']:.4f} / D "
                                 f"{bd['D_on_common'][m]['top1_accuracy']:.4f}" for m in MODEL_NAMES)]
    lk = rep["leakage"]
    out += ["", "[LEAKAGE CHECK]", "", f"Group overlap = {lk['group_overlap']}",
            f"Test used for fitting = {lk['test_used_for_fitting']}", f"Status = {lk['status']}"]
    tc = rep["top_contribution"]
    if tc:
        out += ["", "[TOP CONTRIBUTION]", "", f"Largest combined-over-geometry gain experiment = {tc['experiment']}",
                f"Geometry = {tc['geometry']:.4f}", f"Topology = {tc['topology']:.4f}", f"Combined = {tc['combined']:.4f}",
                f"Gain = {_pp(tc['gain_pp'])} (95% CI [{tc['gain_ci95_pp'][0]:+.1f}, {tc['gain_ci95_pp'][1]:+.1f}] pp)"]
    out += ["", "[LIMITATIONS]", ""] + [f"- {t}" for t in rep["limitations"]] + ["=" * 50]
    return "\n".join(out)


def render_and_save(r: CharacterResult, cfg: AppConfig) -> Dict[str, Path]:
    """All figures + final_report.json / .txt."""
    r.report = build_report(r, cfg)
    out = r.out_dir
    files: Dict[str, Path] = {}
    for e in ("A", "B", "C", "D"):
        for m, short in (("geometry", "geometry"), ("topology", "topology"), ("combined", "combined")):
            if "combined" in r.metrics[e]:
                name = f"confusion_{EXPERIMENTS[e][0]}_{short}"
                try:
                    files[name] = confusion_png(r, e, m, out / f"{name}.png")
                except Exception as exc:
                    plt.close("all")
                    print(f"[WARN] {name}: {exc}")
    for name, fn in (("accuracy_comparison", accuracy_comparison), ("domain_shift_comparison", domain_shift_comparison),
                     ("final_dashboard", dashboard)):
        try:
            files[name] = fn(r, out / f"{name}.png")
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] {name}: {type(exc).__name__}: {exc}")
    r.report["figures"] = {k: rel(v) for k, v in files.items()}
    (out / "final_report.json").write_text(json.dumps(r.report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "final_report.txt").write_text(terminal_summary(r) + "\n\nPROTOCOL\n"
                                          + json.dumps(r.report["protocol"], indent=2, ensure_ascii=False), encoding="utf-8")
    return files
