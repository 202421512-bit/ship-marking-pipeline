"""Figures, reports and terminal summary for the DEGRADATION EXPERIMENT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import visualization as base  # noqa: E402
from .config import AppConfig  # noqa: E402
from .dataset import FORMAL, INFORMAL  # noqa: E402
from .degradation_experiment import (DEG_LABELS, DEGRADATIONS, FIELD_SURROGATE, MODELS, SEVERITIES,  # noqa: E402
                                     DegradationResult, binarize_char, degrade, variant_seed)
from .real_experiment import rel  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, RED = base.SURFACE, base.INK, base.INK_2, base.MUTED, base.GRID, base.RED
COLORS = {"geometry": base.AQUA, "topology": base.ORANGE, "combined": base.BLUE}
LABELS = {"geometry": "Geometry", "topology": "Topology", "combined": "Topology + Geometry"}
BANNER = "REAL DATA · SYNTHETIC DEGRADATION SURROGATES · CLEAN TRAIN → DEGRADED TEST · no restoration"
DPI = 120


def _banner(fig, title: str) -> None:
    fig.text(0.01, 0.988, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.99, 0.988, BANNER, fontsize=9, color=RED, fontweight="bold", va="top", ha="right")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _cell(r: DegradationResult, kind: str, s: int, domain: str = INFORMAL) -> Dict:
    return r.stats[domain]["cells"][f"{kind}|{s}"]


def _grid_curves(r: DegradationResult, path: Path, metric: str, title: str, domain: str = INFORMAL) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6), sharey=True)
    _banner(fig, title)
    st = r.stats[domain]
    for ax, kind in zip(axes.ravel(), DEGRADATIONS):
        for m in MODELS:
            if metric == "accuracy":
                y = [_cell(r, kind, s, domain)[m]["top1_accuracy"] for s in SEVERITIES]
                lo = [_cell(r, kind, s, domain)[m]["ci95"][0] for s in SEVERITIES]
                hi = [_cell(r, kind, s, domain)[m]["ci95"][1] for s in SEVERITIES]
            else:
                d = st["derived"][f"{kind}|{m}"]
                y = d["retention"]
                lo = [c[0] for c in d["retention_ci95"]]
                hi = [c[1] for c in d["retention_ci95"]]
            ax.fill_between(SEVERITIES, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(SEVERITIES, y, color=COLORS[m], lw=2, marker="o", ms=5, label=LABELS[m])
        ax.set_title(DEG_LABELS[kind], fontsize=11)
        ax.set_xticks(SEVERITIES)
        ax.set_ylim(0, 1.05 if metric == "accuracy" else 1.15)
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel("severity (0 = clean)")
    axes[0, 0].set_ylabel("Top-1 accuracy" if metric == "accuracy" else "retention  A(s) / A(0)")
    axes[1, 0].set_ylabel(axes[0, 0].get_ylabel())
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.text(0.01, 0.01, f"{domain} test characters, leave-one-group-out; shaded = 95% CI (group bootstrap).",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, path)


def robustness_auc(r: DegradationResult, path: Path, domain: str = INFORMAL) -> Path:
    st = r.stats[domain]["derived"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.8))
    _banner(fig, f"Severity-curve robustness AUC ({domain}; not a ROC-AUC)")
    x = np.arange(len(DEGRADATIONS))
    for ax, key, ttl in ((axes[0], "robustness_auc", "Accuracy-curve AUC (normalized, severity 0..4)"),
                         (axes[1], "retention_auc", "Retention-curve AUC (accuracy relative to own clean)")):
        for i, m in enumerate(MODELS):
            v = [st[f"{k}|{m}"][key] for k in DEGRADATIONS]
            ci = [st[f"{k}|{m}"][key + "_ci95"] for k in DEGRADATIONS]
            pos = x + (i - 1) * 0.27
            ax.bar(pos, v, 0.27, color=COLORS[m], edgecolor=SURFACE, linewidth=1.5, label=LABELS[m])
            ax.errorbar(pos, v, yerr=[[a - c[0] for a, c in zip(v, ci)], [c[1] - a for a, c in zip(v, ci)]],
                        fmt="none", ecolor=INK_2, capsize=2, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([DEG_LABELS[k] for k in DEGRADATIONS], rotation=25, ha="right", fontsize=9)
        ax.set_title(ttl, fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_ylim(0, 1.15)
    axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def stability_figure(r: DegradationResult, path: Path, domain: str = INFORMAL) -> Path:
    rows = [x for x in r.geo_stability if x["domain"] == domain]
    trows = [x for x in r.topo_stability if x["domain"] == domain]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6), sharey=False)
    _banner(fig, "Feature stability: clean vs degraded (same character), training-normalized distance")
    for ax, kind in zip(axes.ravel(), DEGRADATIONS):
        g = {x["severity"]: x for x in rows if x["degradation"] == kind}
        t = {x["severity"]: x for x in trows if x["degradation"] == kind}
        sev = [s for s in SEVERITIES if s in g]
        ax.plot([0] + sev, [0] + [g[s]["geometry_feature_distance_norm"] for s in sev], color=COLORS["geometry"],
                lw=2, marker="o", label="geometry distance (mean |Δz|)")
        ax.plot([0] + sev, [0] + [g[s]["topology_feature_distance_norm"] for s in sev], color=COLORS["topology"],
                lw=2, marker="s", label="topology distance (mean |Δz|)")
        ax.set_title(f"{DEG_LABELS[kind]}  ·  Betti change rate s4 = {t.get(4, {}).get('betti_change_rate', float('nan')):.2f}",
                     fontsize=10)
        ax.set_xticks(SEVERITIES)
        ax.grid(color=GRID, lw=0.6)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Δz = |x_degraded − x_clean| / std of the training fold (test data excluded). "
             "Betti change rate = fraction of variants whose (β0, β1) differ from clean.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, path)


def _variant_images(r: DegradationResult, item, kind: str, s: int, rep: int, cfg: AppConfig):
    seed = variant_seed(cfg.degradation.seed, kind, s, rep, item.index) if kind != "clean" else \
        variant_seed(cfg.degradation.seed, "clean", 0, 0, item.index)
    img, _ = degrade(kind if kind != "clean" else "blur", s, item.canonical, item.clean_mask, item.ink_level,
                     item.bg_level, seed, cfg)
    return img, binarize_char(img, item.dark_ink, cfg)


def broken_stroke_figure(r: DegradationResult, path: Path, cfg: AppConfig) -> Path:
    fig = plt.figure(figsize=(18, 9))
    _banner(fig, "Broken stroke (topology-destroying surrogate): accuracy, Betti / endpoint changes, example")
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.2, 0.8], hspace=0.4, wspace=0.35)
    ax = fig.add_subplot(gs[0, :2])
    for m in MODELS:
        y = [_cell(r, "broken_stroke", s)[m]["top1_accuracy"] for s in SEVERITIES]
        lo = [_cell(r, "broken_stroke", s)[m]["ci95"][0] for s in SEVERITIES]
        hi = [_cell(r, "broken_stroke", s)[m]["ci95"][1] for s in SEVERITIES]
        ax.fill_between(SEVERITIES, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
        ax.plot(SEVERITIES, y, color=COLORS[m], lw=2, marker="o", label=LABELS[m])
    ax.set_ylim(0, 1.05)
    ax.set_title("Top-1 accuracy (Informal)")
    ax.set_xlabel("severity = number of stroke gaps")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax = fig.add_subplot(gs[0, 2:4])
    t = {x["severity"]: x for x in r.topo_stability if x["domain"] == INFORMAL and x["degradation"] == "broken_stroke"}
    sev = [0] + sorted(t)
    for key, col, lab in (("mean_delta_beta0", base.VIOLET, "Δβ0 (components)"),
                          ("mean_delta_beta1", base.MAGENTA, "Δβ1 (holes)"),
                          ("mean_delta_endpoints", base.YELLOW, "Δ endpoints")):
        ax.plot(sev, [0] + [t[s][key] for s in sev[1:]], color=col, lw=2, marker="o", label=lab)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_title("Mean signed topology change vs clean")
    ax.set_xticks(SEVERITIES)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax = fig.add_subplot(gs[0, 4])
    ax.set_axis_off()
    lines = ["s  β0↑   β1↓   Betti Δ"]
    for s in sev[1:]:
        lines.append(f"{s}  {t[s]['rate_beta0_increased']:.2f}  {t[s]['rate_beta1_decreased']:.2f}  {t[s]['betti_change_rate']:.2f}")
    ax.text(0, 0.95, "\n".join(lines), family="monospace", fontsize=11, va="top", transform=ax.transAxes)
    ax.text(0, 0.35, "rates = fraction of variants\nβ0↑: components increased\nβ1↓: holes decreased", fontsize=9,
            color=INK_2, transform=ax.transAxes)
    item = sorted([i for i in r.items if i.style == INFORMAL], key=lambda i: i.sample_id)[0]
    for s in SEVERITIES:
        ax = fig.add_subplot(gs[1, s])
        img, m = _variant_images(r, item, "broken_stroke" if s else "clean", s, 0, cfg)
        ax.imshow(np.where(m, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"s{s}  ({item.sample_id})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    return _save(fig, path)


def examples_figure(r: DegradationResult, path: Path, cfg: AppConfig) -> Path:
    item = sorted([i for i in r.items if i.style == INFORMAL], key=lambda i: i.sample_id)[0]
    fig, axes = plt.subplots(len(DEGRADATIONS), 10, figsize=(20, 2.1 * len(DEGRADATIONS) + 1))
    _banner(fig, f"Degradation examples: sample {item.sample_id} (lowest sample_id), replicate 0 · grey | binarized")
    for row, kind in zip(axes, DEGRADATIONS):
        for s in SEVERITIES:
            img, m = _variant_images(r, item, kind if s else "clean", s, 0, cfg)
            row[2 * s].imshow(img, cmap="gray", vmin=0, vmax=255)
            row[2 * s + 1].imshow(np.where(m, 0, 255), cmap="gray", vmin=0, vmax=255)
            row[2 * s].set_title(f"{DEG_LABELS[kind] if s == 0 else ''} s{s}", fontsize=8, loc="left")
            for a in (row[2 * s], row[2 * s + 1]):
                a.set_xticks([])
                a.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


CATEGORIES = [("G_fail_T_success", "Geometry fails, Topology succeeds"),
              ("T_fail_G_success", "Topology fails, Geometry succeeds"),
              ("both_fail", "Both fail"),
              ("combined_recovers", "Combined correct, Geometry wrong")]


def select_failure_examples(r: DegradationResult, cfg: AppConfig) -> Dict[Tuple[str, str], Optional[Dict]]:
    """Pre-declared rule: Informal, severity = example_severity, replicate 0, lowest sample_id per category."""
    s = cfg.degradation.example_severity
    out = {}
    rows = sorted([p for p in r.predictions[INFORMAL] if p["evaluable"] and p["severity"] == s and p["replicate"] == 0],
                  key=lambda p: p["sample_id"])
    for kind in DEGRADATIONS:
        rk = [p for p in rows if p["degradation"] == kind]
        tests = {
            "G_fail_T_success": lambda p: not p["geometry_correct"] and p["topology_correct"],
            "T_fail_G_success": lambda p: not p["topology_correct"] and p["geometry_correct"],
            "both_fail": lambda p: not p["topology_correct"] and not p["geometry_correct"],
            "combined_recovers": lambda p: p["combined_correct"] and not p["geometry_correct"],
        }
        for cat, fn in tests.items():
            out[(kind, cat)] = next((p for p in rk if fn(p)), None)
    return out


def failure_examples_figure(r: DegradationResult, path: Path, cfg: AppConfig) -> Path:
    sel = select_failure_examples(r, cfg)
    by_id = {i.sample_id: i for i in r.items}
    fig, axes = plt.subplots(len(DEGRADATIONS), 8, figsize=(18, 2.2 * len(DEGRADATIONS) + 1.2))
    _banner(fig, f"Qualitative cases (rule: Informal, severity {cfg.degradation.example_severity}, replicate 0, "
                 "lowest sample_id per category) · clean | degraded")
    for ri, kind in enumerate(DEGRADATIONS):
        for ci, (cat, title) in enumerate(CATEGORIES):
            a0, a1 = axes[ri, 2 * ci], axes[ri, 2 * ci + 1]
            for a in (a0, a1):
                a.set_xticks([])
                a.set_yticks([])
            if ri == 0:
                a0.set_title(title, fontsize=9, loc="left")
            p = sel[(kind, cat)]
            if p is None:
                a0.text(0.5, 0.5, "none", ha="center", va="center", transform=a0.transAxes, color=MUTED)
                a1.set_axis_off()
                a0.set_ylabel(DEG_LABELS[kind] if ci == 0 else "", fontsize=9)
                continue
            it = by_id[p["sample_id"]]
            a0.imshow(np.where(it.clean_mask, 0, 255), cmap="gray", vmin=0, vmax=255)
            _, m = _variant_images(r, it, kind, cfg.degradation.example_severity, 0, cfg)
            a1.imshow(np.where(m, 0, 255), cmap="gray", vmin=0, vmax=255)
            a1.set_xlabel(base.safe_text(f"GT '{p['character']}' G:{p['geometry_top1']} T:{p['topology_top1']} "
                                         f"TG:{p['combined_top1']}"), fontsize=7)
            if ci == 0:
                a0.set_ylabel(DEG_LABELS[kind], fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def dashboard(r: DegradationResult, path: Path) -> Path:
    rep = r.report
    st = r.stats[INFORMAL]
    fig = plt.figure(figsize=(22, 25))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[0.35, 1.4, 1.0, 0.9, 0.75], hspace=0.45, wspace=0.35)
    fig.text(0.012, 0.995, "PAC MISSION 3 · DEGRADATION ROBUSTNESS EXPERIMENT (REAL CHARACTERS)", fontsize=22,
             fontweight="bold", va="top")
    fig.text(0.012, 0.981, "How do Geometry / Topology / Combined degrade with severity? Clean train → degraded test · "
             "synthetic degradation surrogates · group-level splits · no restoration · no tuning", fontsize=11,
             color=RED, va="top")
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ds = rep["dataset"]
    tiles = [("Test characters (Informal)", ds["test_characters"]), ("Test groups", ds["test_groups"]),
             ("Replicates", ds["replicates"]), ("Seed", ds["seed"]), ("Clean Geometry", f"{rep['clean']['geometry']:.3f}"),
             ("Clean Topology", f"{rep['clean']['topology']:.3f}"), ("Clean Combined", f"{rep['clean']['combined']:.3f}"),
             ("Leakage", rep["leakage"]["status"].split()[0])]
    for i, (k, v) in enumerate(tiles):
        x0 = i / len(tiles)
        ax.add_patch(Rectangle((x0 + 0.004, 0), 1 / len(tiles) - 0.008, 0.9, transform=ax.transAxes,
                               facecolor=SURFACE, edgecolor=GRID))
        ax.text(x0 + 0.5 / len(tiles), 0.52, str(v), transform=ax.transAxes, ha="center", fontsize=20, fontweight="bold")
        ax.text(x0 + 0.5 / len(tiles), 0.14, k, transform=ax.transAxes, ha="center", fontsize=10, color=INK_2)
    sub = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1, :], hspace=0.45, wspace=0.25)
    for i, kind in enumerate(DEGRADATIONS):
        ax = fig.add_subplot(sub[i // 4, i % 4])
        for m in MODELS:
            y = [_cell(r, kind, s)[m]["top1_accuracy"] for s in SEVERITIES]
            lo = [_cell(r, kind, s)[m]["ci95"][0] for s in SEVERITIES]
            hi = [_cell(r, kind, s)[m]["ci95"][1] for s in SEVERITIES]
            ax.fill_between(SEVERITIES, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(SEVERITIES, y, color=COLORS[m], lw=2, marker="o", ms=4, label=LABELS[m])
        ax.set_ylim(0, 1.05)
        ax.set_xticks(SEVERITIES)
        ax.set_title(DEG_LABELS[kind], fontsize=11)
        ax.grid(color=GRID, lw=0.6)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower left")
            ax.set_ylabel("Top-1 accuracy")
    ax = fig.add_subplot(gs[2, :2])
    x = np.arange(len(DEGRADATIONS))
    for i, m in enumerate(MODELS):
        v = [st["derived"][f"{k}|{m}"]["retention_auc"] for k in DEGRADATIONS]
        ci = [st["derived"][f"{k}|{m}"]["retention_auc_ci95"] for k in DEGRADATIONS]
        pos = x + (i - 1) * 0.27
        ax.bar(pos, v, 0.27, color=COLORS[m], edgecolor=SURFACE, label=LABELS[m])
        ax.errorbar(pos, v, yerr=[[a - c[0] for a, c in zip(v, ci)], [c[1] - a for a, c in zip(v, ci)]], fmt="none",
                    ecolor=INK_2, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([DEG_LABELS[k] for k in DEGRADATIONS], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("Retention AUC (severity-curve, relative to own clean accuracy)", loc="left", fontsize=12,
                 fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=9, ncol=3)
    ax = fig.add_subplot(gs[2, 2:])
    ax.set_axis_off()
    ax.set_title("Topology vs Geometry retention (paired 95% CI)", loc="left", fontsize=12, fontweight="bold")
    for i, row in enumerate(rep["statistical_summary"]):
        col = INK if row["verdict"].startswith("topology retention higher") else RED if "lower" in row["verdict"] else INK_2
        ax.text(0.0, 0.93 - i * 0.115, f"{DEG_LABELS[row['degradation']]:<20}", fontsize=11, transform=ax.transAxes,
                fontweight="bold")
        ax.text(0.32, 0.93 - i * 0.115, f"Δ retention-AUC {row['difference']:+.3f} [{row['ci95'][0]:+.3f}, "
                f"{row['ci95'][1]:+.3f}]  {row['verdict']}", fontsize=10, transform=ax.transAxes, color=col)
    ax = fig.add_subplot(gs[3, :2])
    t = {x["severity"]: x for x in r.topo_stability if x["domain"] == INFORMAL and x["degradation"] == "broken_stroke"}
    sev = [0] + sorted(t)
    for key, col, lab in (("mean_delta_beta0", base.VIOLET, "Δβ0"), ("mean_delta_beta1", base.MAGENTA, "Δβ1"),
                          ("mean_delta_endpoints", base.YELLOW, "Δ endpoints")):
        ax.plot(sev, [0] + [t[s][key] for s in sev[1:]], color=col, lw=2, marker="o", label=lab)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks(SEVERITIES)
    ax.legend(frameon=False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_title("Broken stroke: mean signed topology change (Informal)", loc="left", fontsize=12, fontweight="bold")
    ax = fig.add_subplot(gs[3, 2:])
    ax.set_axis_off()
    ax.set_title("Key results (computed)", loc="left", fontsize=12, fontweight="bold")
    sa, fp = rep["strongest_topology_advantage"], rep["topology_failure_point"]
    lines = [f"Strongest topology advantage: {DEG_LABELS[sa['degradation']]} s{sa['severity']}: "
             f"T {sa['topology_accuracy']:.3f} vs G {sa['geometry_accuracy']:.3f} ({sa['difference_pp']:+.1f} pp)"]
    if fp.get("degradation"):
        lines.append(f"Earliest topology failure (drop ≥ {rep['protocol']['failure_drop']:.0%}): "
                     f"{DEG_LABELS[fp['degradation']]} s{fp['severity']}, drop {fp['topology_accuracy_drop_pp']:.1f} pp")
    else:
        lines.append("No topology failure point (drop ≥ 20 pp) within tested severities")
    lines.append(f"Severity-0 identity with clean mask: {rep['dataset']['s0_identity_rate']:.3f}")
    lines.append(f"Extraction failures counted as errors: {rep['dataset']['extraction_failures']} variants")
    for i, t_ in enumerate(lines):
        ax.text(0, 0.85 - i * 0.17, base.safe_text(t_), fontsize=11, transform=ax.transAxes)
    ax = fig.add_subplot(gs[4, :])
    ax.set_axis_off()
    ax.set_title("LIMITATIONS", loc="left", fontsize=12, fontweight="bold")
    for i, t_ in enumerate(rep["limitations"][:7]):
        ax.text(0, 0.9 - i * 0.135, "• " + base.safe_text(t_), fontsize=10.5, transform=ax.transAxes, color=INK_2)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.02)
    return _save(fig, path)


# ====================================================================== report
def build_report(r: DegradationResult, cfg: AppConfig) -> Dict[str, object]:
    d = cfg.degradation
    st = r.stats[INFORMAL]
    clean = {m: _cell(r, "blur", 0)[m]["top1_accuracy"] for m in MODELS}
    best = None
    for kind in DEGRADATIONS:
        for s in SEVERITIES[1:]:
            c = _cell(r, kind, s)
            if best is None or c["topology_advantage_pp"] > best[2]:
                best = (kind, s, c["topology_advantage_pp"])
    kind, s, _ = best
    c = _cell(r, kind, s)
    strongest = {"degradation": kind, "severity": s, "geometry_accuracy": c["geometry"]["top1_accuracy"],
                 "topology_accuracy": c["topology"]["top1_accuracy"], "difference_pp": c["topology_advantage_pp"],
                 "difference_ci95_pp": c["topology_advantage_pp_ci95"]}
    failure = {}
    for kind in DEGRADATIONS:
        drops = st["derived"][f"{kind}|topology"]["drop_pp"]
        for s in SEVERITIES[1:]:
            if drops[s] >= 100 * d.failure_drop:
                cand = (s, -drops[s], kind)
                if not failure or cand < failure["_key"]:
                    t = next(x for x in r.topo_stability if x["domain"] == INFORMAL and x["degradation"] == kind
                             and x["severity"] == s)
                    failure = {"_key": cand, "degradation": kind, "severity": s, "topology_accuracy_drop_pp": drops[s],
                               "mean_delta_beta0": t["mean_delta_beta0"], "mean_delta_beta1": t["mean_delta_beta1"],
                               "mean_delta_endpoints": t["mean_delta_endpoints"],
                               "betti_change_rate": t["betti_change_rate"]}
                break
    failure.pop("_key", None)
    summary = []
    for kind in DEGRADATIONS:
        dd = st["derived"][f"{kind}|retention_auc_topology_minus_geometry"]
        lo, hi = dd["ci95"]
        verdict = ("topology retention higher (CI > 0)" if lo > 0 else
                   "topology retention lower - topology degrades faster (CI < 0)" if hi < 0 else
                   "no significant difference (CI includes 0)")
        summary.append({"degradation": kind, "difference": dd["value"], "ci95": [lo, hi], "verdict": verdict})
    broken = []
    for s in SEVERITIES:
        cc = _cell(r, "broken_stroke", s)
        t = next((x for x in r.topo_stability if x["domain"] == INFORMAL and x["degradation"] == "broken_stroke"
                  and x["severity"] == s), None)
        broken.append({"severity": s, **{m: cc[m]["top1_accuracy"] for m in MODELS},
                       "mean_delta_beta0": t["mean_delta_beta0"] if t else 0.0,
                       "mean_delta_beta1": t["mean_delta_beta1"] if t else 0.0,
                       "mean_delta_endpoints": t["mean_delta_endpoints"] if t else 0.0})
    inf_items = [i for i in r.items if i.style == INFORMAL]
    ev = {p["sample_id"] for p in r.predictions[INFORMAL] if p["evaluable"]}
    lims = [
        "Degradations are SYNTHETIC SURROGATES applied to a canonical (two-level) rendering of each validated "
        "character; they do not reproduce real rust, contamination, illumination or camera physics.",
        "The canonical clean image removes the original photo texture/noise; severity 0 is the clean segmentation mask.",
        "Test-time binarization is a fixed Otsu threshold without denoising (isolates representation response); "
        "the full pipeline's denoising could change the curves.",
        "Characters (not strings) are degraded; segmentation failures under degradation are not modelled.",
        f"Effective sample size is the number of groups ({r.stats[INFORMAL]['n_groups']} Informal test groups); "
        "replicates and characters are not independent (group bootstrap).",
        "Posteriors are UNCALIBRATED MODEL POSTERIORS; feature extraction failures count as errors.",
        "Severity levels are ordinal and their parameters were fixed a priori; they are not matched across "
        "degradation types (cross-type comparison of absolute AUC is only indicative).",
    ]
    return {
        "experiment": "DEGRADATION ROBUSTNESS EXPERIMENT (REAL CHARACTERS)",
        "research_question": "How does recognition performance of geometry-based vs topology-based representations "
                             "decrease as image degradation severity increases?",
        "dataset": {"source": "data/input/Symbols COUNT_MATCH characters (manual transcriptions)",
                    "primary_domain": INFORMAL,
                    "training_groups_per_fold": len({i.group_id for i in inf_items}) - 1,
                    "test_groups": r.stats[INFORMAL]["n_groups"],
                    "test_characters": len(ev), "replicates": d.replicates, "blur_replicates": d.blur_replicates,
                    "seed": d.seed, **r.info},
        "clean": clean,
        "clean_note": "severity-0 accuracy of this pipeline (canonical clean image, same folds as character "
                      "experiment B)",
        "protocol": {"split": "leave-one-group-out (Informal primary, Formal secondary); clean training only",
                     "models": "geometry / topology / combined (0.5/0.5 group-mean log-likelihood), unchanged",
                     "binarization": "Otsu + source polarity + small-component removal (fixed)",
                     "failure_drop": d.failure_drop, "bootstrap": f"{d.bootstrap_iterations} group draws, seed {d.seed}",
                     "auc": "severity-curve robustness AUC = normalized trapezoid over severities 0..4 "
                            "(NOT a ROC-AUC)",
                     "restoration": "not used", "tuning": "none (first run preserved)"},
        "parameters": {k: {"definition": FIELD_SURROGATE[k]} for k in DEGRADATIONS},
        "robustness_auc": {dom: {k: {m: st_["derived"][f"{k}|{m}"]["robustness_auc"] for m in MODELS}
                                 for k in DEGRADATIONS} for dom, st_ in r.stats.items()},
        "retention_auc": {dom: {k: {m: st_["derived"][f"{k}|{m}"]["retention_auc"] for m in MODELS}
                                for k in DEGRADATIONS} for dom, st_ in r.stats.items()},
        "strongest_topology_advantage": strongest,
        "topology_failure_point": failure,
        "broken_stroke": broken,
        "statistical_summary": summary,
        "formal_secondary": {"clean": {m: r.stats[FORMAL]["cells"]["blur|0"][m]["top1_accuracy"] for m in MODELS},
                             "n_groups": r.stats[FORMAL]["n_groups"]},
        "leakage": r.leakage,
        "limitations": lims,
    }


def terminal_summary(r: DegradationResult) -> str:
    rep = r.report
    ds = rep["dataset"]
    o = ["=" * 50, "PAC DEGRADATION EXPERIMENT (REAL CHARACTERS, SYNTHETIC DEGRADATIONS)", "=" * 50, "",
         "[DATASET]", "", f"Training groups = {ds['training_groups_per_fold']} per fold (leave-one-group-out, clean only)",
         f"Test groups = {ds['test_groups']}", f"Test characters = {ds['test_characters']}",
         f"Replicates = {ds['replicates']} (blur: {ds['blur_replicates']}, deterministic)", f"Seed = {ds['seed']}",
         f"Severity-0 identity with clean mask = {ds['s0_identity_rate']:.4f}",
         "", "[CLEAN BASELINE]  (Informal, this pipeline at severity 0)", "",
         f"Geometry = {rep['clean']['geometry']:.4f}", f"Topology = {rep['clean']['topology']:.4f}",
         f"Combined = {rep['clean']['combined']:.4f}", "",
         "[ROBUSTNESS AUC]  (severity-curve robustness AUC, Informal; not ROC-AUC)", "",
         f"{'':16}{'Geometry':>10}{'Topology':>10}{'Combined':>10}"]
    auc = rep["robustness_auc"][INFORMAL]
    for k in DEGRADATIONS:
        o.append(f"{DEG_LABELS[k]:<16}" + "".join(f"{auc[k][m]:>10.3f}" for m in MODELS))
    o += ["", "Retention AUC (relative to own clean accuracy):", f"{'':16}{'Geometry':>10}{'Topology':>10}{'Combined':>10}"]
    rauc = rep["retention_auc"][INFORMAL]
    for k in DEGRADATIONS:
        o.append(f"{DEG_LABELS[k]:<16}" + "".join(f"{rauc[k][m]:>10.3f}" for m in MODELS))
    sa = rep["strongest_topology_advantage"]
    o += ["", "[STRONGEST TOPOLOGY ADVANTAGE]", "", f"Degradation = {DEG_LABELS[sa['degradation']]}",
          f"Severity = {sa['severity']}", f"Geometry accuracy = {sa['geometry_accuracy']:.4f}",
          f"Topology accuracy = {sa['topology_accuracy']:.4f}",
          f"Difference = {sa['difference_pp']:+.1f} pp (95% CI [{sa['difference_ci95_pp'][0]:+.1f}, "
          f"{sa['difference_ci95_pp'][1]:+.1f}])"]
    fp = rep["topology_failure_point"]
    o += ["", f"[TOPOLOGY FAILURE POINT]  (first severity with topology drop >= {rep['protocol']['failure_drop']:.0%})", ""]
    if fp:
        o += [f"Degradation = {DEG_LABELS[fp['degradation']]}", f"Severity = {fp['severity']}",
              f"beta0 change = {fp['mean_delta_beta0']:+.3f} (mean signed)",
              f"beta1 change = {fp['mean_delta_beta1']:+.3f} (mean signed)",
              f"endpoint change = {fp['mean_delta_endpoints']:+.3f} (mean signed)",
              f"Topology accuracy drop = {fp['topology_accuracy_drop_pp']:.1f} pp"]
    else:
        o.append("No severity reached the failure criterion")
    o += ["", "[BROKEN STROKE]  (Informal Top-1: Geometry / Topology / Combined; mean Δβ0, Δβ1, Δendpoints)", ""]
    for b in rep["broken_stroke"]:
        o.append(f"Severity {b['severity']} = G {b['geometry']:.3f} / T {b['topology']:.3f} / TG {b['combined']:.3f}"
                 f"   Δβ0 {b['mean_delta_beta0']:+.2f}  Δβ1 {b['mean_delta_beta1']:+.2f}  "
                 f"Δend {b['mean_delta_endpoints']:+.2f}")
    o += ["", "[STATISTICAL SUMMARY]  (topology − geometry retention-AUC, paired group bootstrap 95% CI)", ""]
    for row in rep["statistical_summary"]:
        o.append(f"{DEG_LABELS[row['degradation']]:<20}{row['difference']:+.3f} "
                 f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]  {row['verdict']}")
    o += ["", "[LEAKAGE CHECK]", "", f"Status = {rep['leakage']['status']}", "", "[LIMITATIONS]", ""]
    o += [f"- {t}" for t in rep["limitations"]] + ["=" * 50]
    return "\n".join(o)


def render_and_save(r: DegradationResult, cfg: AppConfig) -> Dict[str, Path]:
    r.report = build_report(r, cfg)
    out = r.out_dir
    files = {}
    jobs = [
        ("accuracy_vs_severity", lambda: _grid_curves(r, out / "accuracy_vs_severity.png", "accuracy",
                                                      "Top-1 accuracy vs degradation severity (Informal test)")),
        ("retention_vs_severity", lambda: _grid_curves(r, out / "retention_vs_severity.png", "retention",
                                                       "Retention A(s)/A(0) vs severity (Informal test)")),
        ("accuracy_vs_severity_formal", lambda: _grid_curves(r, out / "accuracy_vs_severity_formal.png", "accuracy",
                                                             "Top-1 accuracy vs severity (Formal test, secondary)",
                                                             FORMAL)),
        ("robustness_auc", lambda: robustness_auc(r, out / "robustness_auc.png")),
        ("topology_vs_geometry_stability", lambda: stability_figure(r, out / "topology_vs_geometry_stability.png")),
        ("broken_stroke_analysis", lambda: broken_stroke_figure(r, out / "broken_stroke_analysis.png", cfg)),
        ("degradation_examples", lambda: examples_figure(r, out / "degradation_examples.png", cfg)),
        ("failure_examples", lambda: failure_examples_figure(r, out / "failure_examples.png", cfg)),
        ("final_dashboard", lambda: dashboard(r, out / "final_dashboard.png")),
    ]
    for name, job in jobs:
        try:
            files[name] = job()
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] {name}: {type(exc).__name__}: {exc}")
    r.report["figures"] = {k: rel(v) for k, v in files.items()}
    (out / "final_report.json").write_text(json.dumps(r.report, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    (out / "final_report.txt").write_text(terminal_summary(r), encoding="utf-8")
    return files
