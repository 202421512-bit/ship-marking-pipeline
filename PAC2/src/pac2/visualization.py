"""Presentation figures (16:9, PNG 300 dpi + SVG + CSV of the plotted numbers). Consistent colours:
baseline = grey, intermediate / calibration = blue, final = green, error / degradation = red.
Every function only reads logged results; nothing is retrained here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE, MID, FINAL, ERR = "#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"
FEATURE_LABELS = ["1 CV_w", "2 CV_h", "4 CV_g", "5 S_theta", "6 B", "10 CV_A", "11 R"]
FIGSIZE = (13.333, 7.5)          # 16:9
FEATURE_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
FEATURE_STYLES = ["-", "--", "-.", ":", "-", "--", "-."]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out_dir: Path, name: str, data: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    data.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)


def mean_sd_by_epoch(h: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean / SD over folds per epoch. Folds that stopped early carry their final (frozen) value forward, otherwise the
    mean would jump when a fold drops out. n_folds_active counts folds still training at that epoch."""
    piv = h.pivot_table(index="epoch", columns="fold", values=col)
    active = piv.notna().sum(axis=1).values
    piv = piv.ffill()
    return pd.DataFrame({"epoch": piv.index, f"{col}_mean": piv.mean(axis=1).values, f"{col}_sd": piv.std(axis=1, ddof=0).values,
                         f"{col}_n_folds_active": active})


def bar_with_err(ax, x, means, sds, color, label, hatch=None, width=0.8, rotate=False):
    bars = ax.bar(x, means, width, yerr=sds, color=color, edgecolor="black", linewidth=0.6, hatch=hatch, capsize=4, label=label)
    for b, m, s in zip(bars, means, sds):
        top = m + (s if np.isfinite(s) else 0)
        ax.annotate(f"{m:.3f}", (b.get_x() + b.get_width() / 2, top), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8 if rotate else 9, rotation=90 if rotate else 0)
    return bars


def confusion_panel(ax, tn, fp, fn, tp, title, color):
    M = np.array([[tp, fn], [fp, tn]], float)            # rows: actual handwritten, actual formal
    rows = M.sum(axis=1, keepdims=True)
    P = M / np.where(rows == 0, 1, rows)
    ax.imshow(P, cmap="Greens" if color == FINAL else "Greys", vmin=0, vmax=1)
    labels = [["handwritten -> handwritten", "handwritten -> formal"], ["formal -> handwritten", "formal -> formal"]]
    err = [[False, True], [True, False]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(M[i, j])}\n({P[i, j]:.1%} of row)\n{labels[i][j]}", ha="center", va="center", fontsize=11,
                    color=ERR if err[i][j] and M[i, j] > 0 else ("white" if P[i, j] > 0.6 else "black"), fontweight="bold")
    ax.set_xticks([0, 1], ["predicted handwritten", "predicted formal"])
    ax.set_yticks([0, 1], [f"actual handwritten (n={int(rows[0, 0])})", f"actual formal (n={int(rows[1, 0])})"])
    ax.set_title(title)
