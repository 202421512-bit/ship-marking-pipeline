"""Outputs for the EXPLORATORY direction-aware feature experiment."""

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
from .dataset import FORMAL, INFORMAL  # noqa: E402
from .direction_aware_experiment import MODEL_GROUPS, PAIRS, DirResult  # noqa: E402
from .real_experiment import char_canvas, rel, write_csv  # noqa: E402
from .topology import analyze_topology  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, RED = base.SURFACE, base.INK, base.INK_2, base.MUTED, base.GRID, base.RED
COLORS = {"A_baseline": MUTED, "B_direction": base.BLUE, "C_direction_line": base.VIOLET}
LABELS = {"A_baseline": "A  Topo+Geo (existing)", "B_direction": "B  + Direction",
          "C_direction_line": "C  + Direction + Line-relative"}
BANNER = "REAL DATA · EXPLORATORY (features designed after seeing errors) · fixed segmentation · LOGO split"
DPI = 115


def _banner(fig, title: str) -> None:
    fig.text(0.01, 0.99, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.99, 0.99, BANNER, fontsize=9, color=RED, fontweight="bold", va="top", ha="right")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_tables(r: DirResult) -> None:
    out = r.out_dir
    rows = []
    for m, met in r.metrics.items():
        for scope in ("all", FORMAL, INFORMAL):
            b = met[scope]
            rows.append({"model": m, "feature_groups": "+".join(MODEL_GROUPS[m]), "scope": scope, "images": b["images"],
                         "characters": b["chars"], "char_accuracy": b["char_accuracy"],
                         "char_accuracy_ci95": met.get("char_accuracy_ci95") if scope == "all" else "",
                         "gain_over_A_pp": 100 * (b["char_accuracy"] - r.metrics["A_baseline"][scope]["char_accuracy"]),
                         "gain_over_A_pp_ci95": met.get("gain_over_A_pp_ci95", "") if scope == "all" else "",
                         "string_exact_match": b["string_exact_match"], "cer": b["cer"]})
    write_csv(out / "feature_comparison.csv", rows)
    write_csv(out / "string_accuracy_comparison.csv",
              [{"image_path": it.path, "style": it.style, "status": it.status, "ground_truth": it.gt,
                **{f"text_{m}": r.texts[m][it.path] for m in MODEL_GROUPS},
                **{f"exact_{m}": r.texts[m][it.path] == it.gt for m in MODEL_GROUPS}}
               for it in r.items if it.transcribed])
    classes = sorted(set().union(*[set(r.metrics[m]["per_class"]) for m in MODEL_GROUPS]))
    write_csv(out / "per_class_comparison.csv",
              [{"class": c, "n": r.metrics["A_baseline"]["per_class"].get(c, (0, 0))[1],
                **{f"acc_{m}": r.metrics[m]["per_class"].get(c, (float("nan"), 0))[0] for m in MODEL_GROUPS},
                "delta_B_minus_A": r.metrics["B_direction"]["per_class"].get(c, (np.nan,))[0]
                - r.metrics["A_baseline"]["per_class"].get(c, (np.nan,))[0],
                "delta_C_minus_A": r.metrics["C_direction_line"]["per_class"].get(c, (np.nan,))[0]
                - r.metrics["A_baseline"]["per_class"].get(c, (np.nan,))[0]} for c in classes])
    write_csv(out / "confusion_comparison.csv",
              [{"pair": f"{a}/{b}", "model": m, **{k.replace("->", "_to_"): v for k, v in r.metrics[m]["pairs"][f"{a}/{b}"].items()}}
               for a, b in PAIRS for m in MODEL_GROUPS])
    write_csv(out / "feature_diagnostics.csv", r.diagnostics)


def visualization(r: DirResult, path: Path, cfg: AppConfig) -> Path:
    """Example 6 / 9 / 2 / 5 with hole centroid, endpoints, branch points + feature distributions."""
    fig = plt.figure(figsize=(18, 10))
    _banner(fig, "Direction-aware features: what 6 vs 9 and 2 vs 5 look like to the new features")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.1], hspace=0.35, wspace=0.3)
    # deterministic examples: first COUNT_MATCH occurrence (lowest image path) of each class
    for j, cls in enumerate(["6", "9", "2", "5"]):
        ex = next(((it, s) for it in sorted(r.items, key=lambda i: i.path) if it.status == "COUNT_MATCH"
                   for s in it.segments if s.index < len(it.gt) and it.gt[s.index] == cls), None)
        ax = fig.add_subplot(gs[0, j])
        ax.set_xticks([])
        ax.set_yticks([])
        if ex is None:
            ax.text(0.5, 0.5, f"no '{cls}'", ha="center", transform=ax.transAxes)
            continue
        it, s = ex
        m = char_canvas(s.mask, cfg)
        t = analyze_topology(m, cfg.topology, with_persistence=False)
        ax.imshow(np.where(m, 0.25, 0.97), cmap="gray", vmin=0, vmax=1)
        if t.hole_labels.max() > 0:
            hy, hx = np.nonzero(t.hole_labels > 0)
            ax.scatter([hx.mean()], [hy.mean()], marker="*", s=220, color=base.ORANGE, edgecolors=SURFACE, label="hole centroid")
        if t.endpoint_coords:
            ax.scatter([p[1] for p in t.endpoint_coords], [p[0] for p in t.endpoint_coords], s=60, color=base.BLUE,
                       edgecolors=SURFACE, label="endpoints")
        if t.branch_coords:
            ax.scatter([p[1] for p in t.branch_coords], [p[0] for p in t.branch_coords], marker="D", s=50,
                       color=base.VIOLET, edgecolors=SURFACE, label="branch points")
        f = r.feats[(it.path, s.index)]
        ax.set_title(f"'{cls}'  {it.image_id}\nhole_y {f['hole_y']:.2f} · endpoint_y {f['endpoint_y_mean']:.2f}",
                     fontsize=9.5)
        if j == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower left")
    for j, (feat, a, b) in enumerate([("hole_y", "6", "9"), ("endpoint_y_mean", "6", "9"),
                                      ("endpoint_y_mean", "2", "5"), ("orient_d45", "2", "5")]):
        ax = fig.add_subplot(gs[1, j])
        vals = {}
        for cls in (a, b):
            vals[cls] = [r.feats[(it.path, s.index)][feat] for it in r.items if it.status == "COUNT_MATCH"
                         for s in it.segments if s.index < len(it.gt) and it.gt[s.index] == cls
                         and (it.path, s.index) in r.feats]
        bp = ax.boxplot([vals[a], vals[b]], widths=0.5, patch_artist=True, showfliers=True)
        for patch, col in zip(bp["boxes"], (base.ORANGE, base.BLUE)):
            patch.set_facecolor(col)
            patch.set_alpha(0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"'{a}' n={len(vals[a])}", f"'{b}' n={len(vals[b])}"])
        ax.set_title(feat, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.5)
    fig.text(0.01, 0.01, "Descriptive only (all COUNT_MATCH characters, no model). y = 0 top, 1 bottom of the "
             "character box. Undefined positions (no hole / endpoint) are set to 0.5.", fontsize=9, color=INK_2)
    return _save(fig, path)


def dashboard(r: DirResult, path: Path) -> Path:
    met = r.metrics
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 0.9], hspace=0.38, wspace=0.22)
    fig.text(0.012, 0.995, "PAC · DIRECTION-AWARE TOPOLOGY RECOGNITION (EXPLORATORY)", fontsize=20, fontweight="bold",
             va="top")
    fig.text(0.012, 0.978, "Same real marking images, same fixed segmentation, same leave-one-group-out split · "
             "features designed after observing errors → exploratory, not a generalisation claim", fontsize=11,
             color=RED, va="top")
    ax = fig.add_subplot(gs[0, 0])
    names = list(MODEL_GROUPS)
    metrics = [("char_accuracy", "Char accuracy"), ("string_exact_match", "String exact"), ("cer", "CER (↓)")]
    x = np.arange(len(metrics))
    for i, m in enumerate(names):
        v = [met[m]["all"][k] for k, _ in metrics]
        ax.bar(x + (i - 1) * 0.27, v, 0.27, color=COLORS[m], edgecolor=SURFACE, label=LABELS[m])
        for xi, vv in zip(x, v):
            ax.text(xi + (i - 1) * 0.27, vv + 0.01, f"{vv:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("All transcribed images (char acc: COUNT_MATCH chars; string/CER: all 273 images)", fontsize=11, loc="left")
    ax = fig.add_subplot(gs[0, 1])
    for i, m in enumerate(names):
        v = [met[m][s]["char_accuracy"] for s in (FORMAL, INFORMAL)]
        ax.bar(np.arange(2) + (i - 1) * 0.27, v, 0.27, color=COLORS[m], edgecolor=SURFACE, label=LABELS[m])
        for xi, vv in zip(np.arange(2), v):
            ax.text(xi + (i - 1) * 0.27, vv + 0.01, f"{vv:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["Formal", "Informal"])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("Character accuracy by style", fontsize=11, loc="left")
    ax = fig.add_subplot(gs[1, 0])
    labels = []
    for k, (a, b) in enumerate(PAIRS):
        for i, m in enumerate(names):
            p = met[m]["pairs"][f"{a}/{b}"]
            tot = p[f"{a}->{b}"] + p[f"{b}->{a}"]
            ax.bar(k + (i - 1) * 0.27, tot, 0.27, color=COLORS[m], edgecolor=SURFACE)
            ax.text(k + (i - 1) * 0.27, tot + 0.2, str(tot), ha="center", fontsize=9)
        labels.append(f"{a}↔{b}\n(n {met['A_baseline']['pairs'][f'{a}/{b}'][f'{a}_n']}/"
                      f"{met['A_baseline']['pairs'][f'{a}/{b}'][f'{b}_n']})")
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("Target confusion pairs (both directions, count; lower is better)", fontsize=11, loc="left")
    ax = fig.add_subplot(gs[1, 1])
    a_pc, c_pc = met["A_baseline"]["per_class"], met["C_direction_line"]["per_class"]
    b_pc = met["B_direction"]["per_class"]
    rows = sorted(a_pc, key=lambda c: (b_pc.get(c, (0,))[0] - a_pc[c][0]))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [100 * (b_pc[c][0] - a_pc[c][0]) for c in rows], 0.4, color=COLORS["B_direction"], label="B − A")
    ax.barh(y + 0.2, [100 * (c_pc[c][0] - a_pc[c][0]) for c in rows], 0.4, color=COLORS["C_direction_line"], label="C − A")
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([base.safe_text(f"'{c}' n={a_pc[c][1]}") for c in rows], fontsize=7)
    ax.set_xlabel("per-class accuracy change (pp)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_title("Per-class change vs existing model", fontsize=11, loc="left")
    ax = fig.add_subplot(gs[2, :])
    ax.set_axis_off()
    ax.set_title("GENERALIZATION / LIMITATIONS", loc="left", fontsize=12, fontweight="bold")
    for k, t in enumerate(r.report["limitations"]):
        ax.text(0, 0.9 - k * 0.14, "• " + base.safe_text(t)[:200], fontsize=10.5, color=INK_2, transform=ax.transAxes)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.03)
    return _save(fig, path)


def build_report(r: DirResult, cfg: AppConfig) -> Dict[str, object]:
    d = cfg.direction
    return {
        "experiment": "DIRECTION-AWARE TOPOLOGY RECOGNITION (EXPLORATORY)",
        "models": {m: {"groups": list(g), "weights": f"1/{len(g)} per group (group-mean log-likelihood)"}
                   for m, g in MODEL_GROUPS.items()},
        "direction_features": list(d.direction_features), "line_features": list(d.line_features),
        "metrics": r.metrics, "identity_check": r.identity_check, "leakage": r.leakage,
        "diagnostics": r.diagnostics,
        "limitations": [
            "EXPLORATORY: the new features were designed after inspecting the 6/9, 2/5, s/S, l/I errors on this "
            "same data; gains are optimistic. A generalisation claim needs new, independent source groups.",
            "Segmentation is fixed: touching-character and decimal-point failures are unchanged and remain "
            "separate blockers (string metrics include them).",
            "Undefined positions (no hole / endpoint / branch point) are imputed with 0.5 (box centre).",
            "Few samples for several target classes (e.g. s, l: one source group) - pair counts are small.",
            "Posteriors remain uncalibrated; no hardcoded per-character rule is used.",
        ],
    }


def terminal_summary(r: DirResult, cfg: AppConfig) -> str:
    met = r.metrics
    A = met["A_baseline"]["all"]
    o = ["=" * 60, "PAC DIRECTION-AWARE EXPERIMENT (EXPLORATORY)", "=" * 60, "", "[BASELINE]  (A = existing Topo+Geo, "
         "reproduced: " + r.identity_check["status"] + ")",
         f"Character accuracy = {A['char_accuracy']:.4f}", f"String exact match = {A['string_exact_match']:.4f}",
         f"CER = {A['cer']:.4f}", "", "[NEW FEATURES]",
         f"Direction features = {', '.join(cfg.direction.direction_features)}",
         f"Line-relative features = {', '.join(cfg.direction.line_features)}", "", "[RESULTS]"]
    for m, name in (("A_baseline", "Baseline"), ("B_direction", "Direction-aware"),
                    ("C_direction_line", "Direction + Line-relative")):
        b = met[m]["all"]
        ci = met[m].get("gain_over_A_pp_ci95")
        o.append(f"{name} = char acc {b['char_accuracy']:.4f} {met[m]['char_accuracy_ci95']}"
                 + (f" (gain {100 * (b['char_accuracy'] - A['char_accuracy']):+.1f} pp, 95% CI "
                    f"[{ci[0]:+.1f}, {ci[1]:+.1f}])" if ci else "")
                 + f" | string exact {b['string_exact_match']:.4f} | CER {b['cer']:.4f} | "
                   f"Formal {met[m][FORMAL]['char_accuracy']:.4f} / Informal {met[m][INFORMAL]['char_accuracy']:.4f}")
    o += ["", "[CONFUSION ANALYSIS]  (A -> B -> C, counts both directions)"]
    for a, b in PAIRS:
        parts = []
        for m in MODEL_GROUPS:
            p = met[m]["pairs"][f"{a}/{b}"]
            parts.append(f"{a}->{b} {p[f'{a}->{b}']}, {b}->{a} {p[f'{b}->{a}']}")
        n = met["A_baseline"]["pairs"][f"{a}/{b}"]
        o.append(f"{a}/{b} (n {n[f'{a}_n']}/{n[f'{b}_n']}) = " + "  ->  ".join(parts))
    o += ["", "Top confusions C: " + ", ".join(met["C_direction_line"]["top_confusions"]), "", "[GENERALIZATION]",
          f"Group leakage = {r.leakage['group_overlap']} (leave-one-group-out, {r.leakage['folds']} folds)",
          f"Training-only normalization = {r.leakage['normalization']}",
          "Overfitting limitations ="] + [f"  - {t}" for t in r.report["limitations"]] + ["=" * 60]
    return "\n".join(o)


def render_and_save(r: DirResult, cfg: AppConfig) -> None:
    r.report = build_report(r, cfg)
    write_tables(r)
    for name, fn in (("direction_feature_visualization", lambda: visualization(r, r.out_dir / "direction_feature_visualization.png", cfg)),
                     ("final_dashboard", lambda: dashboard(r, r.out_dir / "final_dashboard.png"))):
        try:
            fn()
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] {name}: {type(exc).__name__}: {exc}")
    (r.out_dir / "final_report.json").write_text(json.dumps(r.report, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    (r.out_dir / "final_report.txt").write_text(terminal_summary(r, cfg), encoding="utf-8")
