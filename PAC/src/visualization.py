"""Presentation figures for the research prototype (matplotlib, Agg backend).

Figures: preprocessing, topology_analysis, restoration_comparison,
geometry_analysis, bayesian_posterior and final_dashboard. All values shown
are exactly the values computed by the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from matplotlib import font_manager  # noqa: E402

from .config import KOREAN_FONT_CANDIDATES, PROJECT_SUBTITLE, PROJECT_TITLE, PROTOTYPE_LABEL  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import PipelineResult

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
STATUS = {"HIGH_CONFIDENCE": "#0ca30c", "REVIEW_REQUIRED": "#fab219", "AMBIGUOUS": "#d03b3b"}
STATUS_ICON = {"HIGH_CONFIDENCE": "✔", "REVIEW_REQUIRED": "⚠", "AMBIGUOUS": "✖"}
DPI = 130

BASE_FONT = "DejaVu Sans"


def find_korean_font() -> Optional[str]:
    """Return the first installed Korean-capable font family known to matplotlib."""
    try:
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return None
    return next((name for name in KOREAN_FONT_CANDIDATES if name in installed), None)


KOREAN_FONT: Optional[str] = find_korean_font()

try:
    from matplotlib import ft2font

    _BASE_FT = ft2font.FT2Font(font_manager.findfont(BASE_FONT))
except Exception:  # pragma: no cover
    _BASE_FT = None


def safe_text(text: object) -> str:
    """Make text renderable without tofu (□).

    With a Korean font installed, matplotlib's per-glyph family fallback
    (DejaVu Sans -> Korean font) renders Hangul. Without one, characters
    missing from DejaVu Sans are replaced by '?'.
    """
    s = str(text)
    if KOREAN_FONT is not None or _BASE_FT is None:
        return s
    return "".join(c if c in "\n\t" or _BASE_FT.get_char_index(ord(c)) else "?" for c in s)


plt.rcParams.update({
    # DejaVu first (Greek β/χ, symbols), Korean family as glyph fallback.
    "font.family": [BASE_FONT] + ([KOREAN_FONT] if KOREAN_FONT else []),
    "axes.unicode_minus": False,
    "figure.facecolor": PAGE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "grid.color": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------- helpers
def _gray_bgr(image: np.ndarray) -> np.ndarray:
    """Return an RGB view of a BGR/gray image for imshow."""
    if image.ndim == 2:
        return image
    return image[..., ::-1]


def _ink_rgb(mask: np.ndarray) -> np.ndarray:
    """Dark ink on light surface RGB image from a bool mask."""
    rgb = np.full(mask.shape + (3,), 0.985, dtype=np.float32)
    rgb[mask] = (0.16, 0.16, 0.15)
    return rgb


def _show(ax: plt.Axes, img: np.ndarray, title: str, cmap: Optional[str] = "gray") -> None:
    """imshow without ticks."""
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.set_title(safe_text(title), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS)


def _overlay_skeleton(ax: plt.Axes, mask: np.ndarray, skel: np.ndarray, title: str,
                      ends: Sequence[Tuple[int, int]] = (), branches: Sequence[Tuple[int, int]] = ()) -> None:
    """Skeleton (orange) over the ink mask, with optional keypoints."""
    rgb = _ink_rgb(mask) * 0.55 + 0.45
    rgb[skel] = matplotlib.colors.to_rgb(ORANGE)
    _show(ax, rgb, title, cmap=None)
    if ends:
        ys, xs = zip(*ends)
        ax.scatter(xs, ys, s=46, c=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3, label="endpoint")
    if branches:
        ys, xs = zip(*branches)
        ax.scatter(xs, ys, s=58, marker="D", c=VIOLET, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label="branch point")


def _label_image(labels: np.ndarray) -> np.ndarray:
    """Color each label with categorical slots (background = surface)."""
    palette = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    rgb = np.full(labels.shape + (3,), 0.985, dtype=np.float32)
    for lab in range(1, int(labels.max()) + 1):
        rgb[labels == lab] = matplotlib.colors.to_rgb(palette[(lab - 1) % len(palette)])
    return rgb


def _panel(ax: plt.Axes, title: str, rows: List[Tuple[str, str]], accent: str = BLUE,
           note: str = "", value_size: int = 12, max_step: float = 0.11) -> None:
    """Card-style metric panel: title with accent bar, key/value rows."""
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=0.02",
                                transform=ax.transAxes, facecolor=SURFACE, edgecolor=GRID, linewidth=1.2))
    ax.add_patch(Rectangle((0.0, 0.93), 1.0, 0.07, transform=ax.transAxes, facecolor=accent,
                           edgecolor="none"))
    title, note = safe_text(title), safe_text(note)
    rows = [(safe_text(k), safe_text(v)) for k, v in rows]
    ax.text(0.04, 0.83, title, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    n = max(len(rows), 1)
    top, bottom = 0.70, 0.12 if note else 0.05
    step = min(max_step, (top - bottom) / n)
    for i, (key, value) in enumerate(rows):
        y = top - i * step
        ax.text(0.05, y, key, transform=ax.transAxes, fontsize=value_size - 1, color=INK_2, va="center")
        ax.text(0.95, y, value, transform=ax.transAxes, fontsize=value_size, color=INK,
                fontweight="bold", va="center", ha="right")
    if note:
        ax.text(0.05, 0.04, note, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom",
                style="italic", wrap=True)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save and close a figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


# ---------------------------------------------------------------- figures
def plot_preprocessing(r: "PipelineResult", path: Path) -> Path:
    """Original and every preprocessing stage side by side."""
    stages = r.preprocessing.stages
    n = len(stages) + 1
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.3 * rows + 0.8))
    axes = np.atleast_1d(axes).ravel()
    _show(axes[0], _gray_bgr(r.loaded.image), "0. original (unmodified)")
    for i, (ax, (name, img)) in enumerate(zip(axes[1:], stages.items()), start=1):
        _show(ax, img, f"{i}. " + name.split("_", 1)[-1].replace("_", " "))
    for ax in axes[n:]:
        ax.set_axis_off()
    info = r.preprocessing.info
    fig.suptitle("Preprocessing  |  polarity={}  threshold={} (Otsu)  NL-means h={}  CLAHE={}  "
                 "removed noise comps={}".format(info["polarity"], info["otsu_threshold"], info["nlmeans_h"],
                                                  "applied" if info["clahe_applied"] else "skipped",
                                                  info["removed_noise_components"]),
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    return _save(fig, path)


def _persistence_plot(ax: plt.Axes, persistence: Dict[str, object]) -> None:
    """Persistence diagram (H0, H1) or a fallback note."""
    ax.set_title(f"Persistence diagram (observed)\nrobust H1={persistence.get('robust_H1_count', '-')}, "
                 f"near holes={persistence.get('near_hole_count', '-')}", fontsize=10)
    if persistence.get("status") != "OK":
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"Persistent homology: {persistence.get('status')}\n(fallback: β0, β1, χ, skeleton)",
                ha="center", va="center", transform=ax.transAxes, color=INK_2, fontsize=9)
        return
    pts = {"H0": persistence.get("H0", []), "H1": persistence.get("H1", [])}
    finite = [p for k in pts for p in pts[k] if p["death"] is not None]
    hi = max([p["death"] for p in finite] + [1.0]) * 1.1
    lo = min([p["birth"] for k in pts for p in pts[k]] + [0.0]) * 1.1
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    for key, color, marker in (("H0", BLUE, "o"), ("H1", ORANGE, "s")):
        bs = [p["birth"] for p in pts[key] if p["death"] is not None]
        ds = [p["death"] for p in pts[key] if p["death"] is not None]
        ax.scatter(bs, ds, c=color, marker=marker, s=60, edgecolors=SURFACE, linewidths=1.5, label=key, zorder=3)
    ax.set_xlabel("birth (signed distance, px)")
    ax.set_ylabel("death")
    ax.legend(loc="lower right", frameon=False, fontsize=8)


def plot_topology(r: "PipelineResult", path: Path) -> Path:
    """Topology analysis of the observed mask and the restored mask."""
    obs, res = r.observed.topology, r.restored.topology
    m_obs, m_res = r.preprocessing.mask, r.restoration.best.mask
    fig = plt.figure(figsize=(17, 9.6))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 0.42], hspace=0.35, wspace=0.18)
    ax = [fig.add_subplot(gs[i // 5, i % 5]) for i in range(10)]
    _show(ax[0], _gray_bgr(r.loaded.image), "Original")
    _show(ax[1], _ink_rgb(m_obs), "Binary (observed)", cmap=None)
    _overlay_skeleton(ax[2], m_obs, obs.skeleton, "Skeleton overlay (observed)")
    _show(ax[3], _label_image(obs.component_labels), f"Connected components β0={obs.beta_0}", cmap=None)
    holes = _ink_rgb(m_obs) * 0.6 + 0.4
    holes[obs.hole_labels > 0] = matplotlib.colors.to_rgb(AQUA)
    _show(ax[4], holes, f"Holes β1={obs.beta_1} (observed)", cmap=None)

    _show(ax[5], _ink_rgb(m_res), f"Restored ({r.restoration.best.operation})", cmap=None)
    holes_r = _ink_rgb(m_res) * 0.6 + 0.4
    holes_r[res.hole_labels > 0] = matplotlib.colors.to_rgb(AQUA)
    _show(ax[6], holes_r, f"Holes β1={res.beta_1} (restored)", cmap=None)
    _overlay_skeleton(ax[7], m_res, res.skeleton, f"Endpoints = {res.endpoints} (restored)", ends=res.endpoint_coords)
    _overlay_skeleton(ax[8], m_res, res.skeleton, f"Branch points = {res.branch_points} (restored)",
                      branches=res.branch_coords)
    _persistence_plot(ax[9], obs.persistence)

    tbl = fig.add_subplot(gs[2, :])
    tbl.set_axis_off()
    keys = [("β0 (components)", "beta_0"), ("β1 (holes)", "beta_1"), ("Euler χ = β0-β1", "euler_characteristic"),
            ("Skeleton length (px)", "skeleton_length"), ("Endpoints", "endpoints"), ("Branch points", "branch_points")]
    xs = np.linspace(0.02, 0.86, len(keys))
    for x, (label, attr) in zip(xs, keys):
        a, b = getattr(obs, attr), getattr(res, attr)
        tbl.text(x, 0.78, label, fontsize=10, color=INK_2, transform=tbl.transAxes)
        tbl.text(x, 0.30, f"{a} → {b}", fontsize=17, fontweight="bold", transform=tbl.transAxes,
                 color=INK if a == b else ORANGE)
    tbl.text(0.02, -0.05, "observed → restored. Topology (connectivity, holes, skeleton) is used as a constraint in "
             "restoration and as evidence in Bayesian inference.", fontsize=9, color=MUTED, transform=tbl.transAxes)
    fig.suptitle("Topology Analysis  —  Topology-Aware Character Recognition", fontsize=14,
                 fontweight="bold", x=0.01, ha="left")
    return _save(fig, path)


def plot_restoration(r: "PipelineResult", path: Path) -> Path:
    """Restoration candidates and their weighted objective terms."""
    from .config import DEFAULT_CONFIG

    lam = (r.config or DEFAULT_CONFIG).restoration
    cands = r.restoration.candidates
    best_id = r.restoration.best.candidate_id
    fig = plt.figure(figsize=(17, 10.5))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 1.5], hspace=0.45, wspace=0.15)
    for i, c in enumerate(cands[:10]):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        _show(ax, _ink_rgb(c.mask), f"{c.operation}\nJ={c.J:.3f}  β0={c.beta_0} β1={c.beta_1}", cmap=None)
        ax.title.set_fontsize(9)
        if c.candidate_id == best_id:
            for spine in ax.spines.values():
                spine.set_color(GREEN)
                spine.set_linewidth(3.5)
            ax.set_title(f"★ BEST: {c.operation}\nJ={c.J:.3f}  β0={c.beta_0} β1={c.beta_1}",
                         fontsize=9, color=GREEN)
    ax = fig.add_subplot(gs[2, :])
    valid = [c for c in cands if c.valid]
    names = [c.operation for c in valid]
    terms = [("λ_data·L_data", BLUE, [lam.lambda_data * c.L_data for c in valid]),
             ("λ_topology·L_topology", ORANGE, [lam.lambda_topology * c.L_topology for c in valid]),
             ("λ_geometry·L_geometry", AQUA, [lam.lambda_geometry * c.L_geometry for c in valid]),
             ("λ_change·L_change", YELLOW, [lam.lambda_change * c.L_change for c in valid])]
    left = np.zeros(len(valid))
    y = np.arange(len(valid))
    for label, color, vals in terms:
        ax.barh(y, vals, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.62, label=label)
        left += np.array(vals)
    for yi, c in zip(y, valid):
        ax.text(c.J + 0.004, yi, f"J={c.J:.3f}  (ref: {c.reference_class})", va="center", fontsize=8.5,
                color=INK if c.candidate_id != best_id else GREEN,
                fontweight="bold" if c.candidate_id == best_id else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(c.J for c in valid) * 1.28)
    ax.grid(axis="x", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("objective J (lower is better)")
    ax.legend(loc="lower right", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.0))
    refs = ", ".join(f"{c}:{p:.2f}" for c, p in r.restoration.reference_classes)
    fig.suptitle(f"Topology-Constrained Restoration  |  Top-K reference prototypes (preliminary posterior): {refs}",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    return _save(fig, path)


def plot_geometry(r: "PipelineResult", path: Path) -> Path:
    """Geometric descriptors of the restored mask vs the final candidate prototype."""
    g = r.restored.geometry
    mask = r.restoration.best.mask
    cand = r.fusion.final_candidate
    fig = plt.figure(figsize=(17, 7.2))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1.6, 1.1], wspace=0.5)
    ax0 = fig.add_subplot(gs[0])
    _show(ax0, _ink_rgb(mask), "Bounding box, hull, centroid, axis", cmap=None)
    ys, xs = np.nonzero(mask)
    ax0.add_patch(Rectangle((xs.min() - 0.5, ys.min() - 0.5), g.width, g.height, fill=False, ec=BLUE, lw=2))
    if g.hull_points:
        hp = np.array(g.hull_points + [g.hull_points[0]])
        ax0.plot(hp[:, 0], hp[:, 1], color=AQUA, lw=2)
    ax0.scatter([g.centroid_x], [g.centroid_y], c=ORANGE, s=70, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    theta = np.radians(g.orientation_deg)
    length = 0.45 * max(g.width, g.height)
    ax0.plot([g.centroid_x - length * np.cos(theta), g.centroid_x + length * np.cos(theta)],
             [g.centroid_y - length * np.sin(theta), g.centroid_y + length * np.sin(theta)], color=ORANGE, lw=2)
    ax0.text(0.02, -0.08, "■ bbox  ■ convex hull  ● centroid / major axis", transform=ax0.transAxes,
             fontsize=8, color=INK_2)

    ax1 = fig.add_subplot(gs[1])
    bins = np.arange(len(g.radial_histogram))
    ax1.bar(bins, g.radial_histogram, color=BLUE, edgecolor=SURFACE, linewidth=2, width=0.85)
    ax1.set_title("Radial distance histogram", fontsize=10)
    ax1.set_xlabel("normalized distance from centroid (bin)")
    ax1.set_ylabel("ink share")
    ax1.grid(axis="y", lw=0.6)
    ax1.set_axisbelow(True)

    ax2 = fig.add_subplot(gs[2])
    clf_z = r.bayesian.feature_z.get(cand) or next(iter(r.bayesian.feature_z.values()))
    feats = [f for f in clf_z if not f.startswith(("beta", "euler", "skeleton", "endpoints", "branch"))]
    vals = [clf_z[f] for f in feats]
    yy = np.arange(len(feats))
    ax2.barh(yy, vals, color=BLUE, edgecolor=SURFACE, linewidth=1.5, height=0.65)
    ax2.axvline(0, color=AXIS, lw=1)
    for lim in (-1.5, 1.5):
        ax2.axvline(lim, color=MUTED, lw=1, ls="--")
    ax2.set_yticks(yy)
    ax2.set_yticklabels(feats, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel(f"z-score vs prototype '{cand}'  ( (x-μ)/σ )")
    ax2.set_title(f"Normalized geometry vs prototype '{cand}' (dashed: ±1.5 demo band)", fontsize=10)
    ax2.grid(axis="x", lw=0.6)
    ax2.set_axisbelow(True)

    ax3 = fig.add_subplot(gs[3])
    _panel(ax3, "GEOMETRY", [
        ("Width × Height (px)", f"{g.width} × {g.height}"),
        ("Aspect ratio", _fmt(g.aspect_ratio)), ("Area ratio", _fmt(g.area_ratio)),
        ("Perimeter (px)", _fmt(g.perimeter, 1)), ("Circularity", _fmt(g.circularity)),
        ("Compactness", _fmt(g.compactness, 2)), ("Convex hull area", _fmt(g.convex_hull_area, 1)),
        ("Solidity", _fmt(g.solidity)), ("Eccentricity", _fmt(g.eccentricity)),
        ("Orientation (deg)", _fmt(g.orientation_deg, 1)),
        ("Hu1 / Hu2 (-log10)", f"{g.hu_moments[0]:.2f} / {g.hu_moments[1]:.2f}"),
    ], accent=AQUA, value_size=10)
    fig.suptitle("Geometry Analysis (restored mask, normalized canvas)", fontsize=14, fontweight="bold",
                 x=0.01, ha="left")
    return _save(fig, path)


def plot_bayesian(r: "PipelineResult", path: Path) -> Path:
    """Top-5 posterior (preliminary vs final) and topology/geometry log-likelihood contributions."""
    b, pre = r.bayesian, r.bayesian_before
    labels = [c for c, _ in b.top5]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"wspace": 0.3})
    x = np.arange(len(labels))
    w = 0.38
    p_pre = [pre.posteriors.get(c, 0.0) for c in labels]
    p_fin = [b.posteriors.get(c, 0.0) for c in labels]
    ax0.bar(x - w / 2, p_pre, w, color=MUTED, edgecolor=SURFACE, linewidth=2,
            label=f"Before restoration (observed)  H={pre.entropy:.3f}")
    ax0.bar(x + w / 2, p_fin, w, color=BLUE, edgecolor=SURFACE, linewidth=2,
            label=f"After restoration (restored)  H={b.entropy:.3f}")
    for xi, v in zip(x, p_pre):
        ax0.text(xi - w / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=INK_2)
    for xi, v in zip(x, p_fin):
        ax0.text(xi + w / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=13, fontweight="bold")
    ax0.set_ylim(0, 1.1)
    ax0.set_ylabel("Posterior Probability (prototype reference model)")
    ax0.set_title(f"Top-5 shown; posterior & entropy over all {len(b.posteriors)} classes (nats)",
                  fontsize=10)
    ax0.legend(frameon=False, fontsize=9, loc="upper right")
    ax0.grid(axis="y", lw=0.6)
    ax0.set_axisbelow(True)

    lt = [b.contributions[c]["topology"] for c in labels]
    lg = [b.contributions[c]["geometry"] for c in labels]
    ax1.bar(x - w / 2, lt, w, color=ORANGE, edgecolor=SURFACE, linewidth=2, label="Topology log-likelihood")
    ax1.bar(x + w / 2, lg, w, color=AQUA, edgecolor=SURFACE, linewidth=2, label="Geometry log-likelihood")
    ax1.axhline(0, color=AXIS, lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=13, fontweight="bold")
    ax1.set_ylabel("log P(x|c) per feature group (higher = better fit)")
    ax1.set_title("Feature-group contribution (topology vs geometry)", fontsize=10)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax1.grid(axis="y", lw=0.6)
    ax1.set_axisbelow(True)
    fig.suptitle("Bayesian Shape Inference  P(c|x) ∝ P(x|c)P(c)  — posterior relative to prototype reference model,"
                 " not a calibrated field probability", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.06, right=0.98)
    return _save(fig, path)


def _flow_strip(ax: plt.Axes, steps: Sequence[str]) -> None:
    """Horizontal pipeline flow: boxes connected by arrows."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(steps)
    width = 0.92 / n
    for i, s in enumerate(steps):
        x0 = 0.01 + i * (width + 0.08 / n)
        ax.add_patch(FancyBboxPatch((x0, 0.18), width, 0.64, boxstyle="round,pad=0,rounding_size=0.006",
                                    facecolor=SURFACE, edgecolor=AXIS, linewidth=1.2))
        ax.text(x0 + width / 2, 0.5, s, ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x0 + width + 0.002, 0.5), (x0 + width + 0.08 / n - 0.002, 0.5),
                                         arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.4))


def plot_dashboard(r: "PipelineResult", path: Path) -> Path:
    """One large presentation figure summarizing the whole evidence chain."""
    obs, res, g = r.observed.topology, r.restored.topology, r.restored.geometry
    b, v, f, d = r.bayesian, r.vlm, r.fusion, r.decision
    fig = plt.figure(figsize=(21, 14.5))
    gs = GridSpec(4, 12, figure=fig, height_ratios=[0.34, 1.25, 1.05, 1.35], hspace=0.32, wspace=0.6)

    head = fig.add_subplot(gs[0, :])
    _flow_strip(head, ["Damaged field\nmarking", "Preprocessing", "Topology", "Geometry", "Restoration\n(min J)",
                       "Bayesian\nevidence", "VLM\nevidence", "Evidence\nfusion", "Confidence /\nambiguity",
                       "Human review?"])
    fig.text(0.012, 0.985, f"{PROJECT_TITLE}  ·  {PROJECT_SUBTITLE}", fontsize=19, fontweight="bold", va="top")
    fig.text(0.988, 0.985, f"{PROTOTYPE_LABEL}   run {r.run_id}", fontsize=10, color=MUTED, va="top", ha="right")
    source = r.model.metadata.get("source", "SYNTHETIC_FONT_PROTOTYPE")
    kind = "SYNTHETIC DEMO" if r.extra_input_info.get("demo") else "SINGLE IMAGE"
    fig.text(0.012, 0.958, safe_text(f"Input: {r.loaded.path.name}   sha256 {r.loaded.sha256[:16]}…   |   "
                                     f"{kind}   |   reference model: {source}"), fontsize=9.5,
             color=INK_2, va="top")

    ax = [fig.add_subplot(gs[1, 3 * i:3 * i + 3]) for i in range(4)]
    _show(ax[0], _gray_bgr(r.loaded.image), "① Original marking (damaged)")
    _show(ax[1], _ink_rgb(r.restoration.best.mask),
          f"② Restored marking — {r.restoration.best.operation} (J={r.restoration.best.J:.3f})", cmap=None)
    _overlay_skeleton(ax[2], r.restoration.best.mask, res.skeleton, "③ Skeleton overlay + keypoints",
                      ends=res.endpoint_coords, branches=res.branch_coords)
    ax[2].legend(loc="lower right", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    tmpl = r.model.templates.get(f.final_candidate)
    if tmpl is not None:
        _show(ax[3], 1.0 - tmpl, f"Reference prototype '{f.final_candidate}' (mean template, not input)")
    else:
        ax[3].set_axis_off()

    def arrow(a: int, bb: int) -> str:
        return f"{a} → {bb}" if a != bb else f"{bb}"

    _panel(fig.add_subplot(gs[2, 0:6]), "TOPOLOGY   (observed → restored)", [
        ("β0  connected components", arrow(obs.beta_0, res.beta_0)),
        ("β1  holes", arrow(obs.beta_1, res.beta_1)),
        ("Euler characteristic χ", arrow(obs.euler_characteristic, res.euler_characteristic)),
        ("Skeleton length (px)", arrow(obs.skeleton_length, res.skeleton_length)),
        ("Endpoints / Branch points", f"{arrow(obs.endpoints, res.endpoints)}  /  {arrow(obs.branch_points, res.branch_points)}"),
        ("Persistent H1 near-holes (observed)", str(obs.persistence.get("near_hole_count", "n/a"))),
    ], accent=ORANGE, note="Topology constrains restoration and is Bayesian evidence (not just a picture).")
    _panel(fig.add_subplot(gs[2, 6:12]), "GEOMETRY   (restored)", [
        ("Aspect ratio", _fmt(g.aspect_ratio)), ("Area ratio", _fmt(g.area_ratio)),
        ("Circularity", _fmt(g.circularity)), ("Solidity", _fmt(g.solidity)),
        ("Eccentricity", _fmt(g.eccentricity)),
        (f"Mean |z| vs prototype '{f.final_candidate}'", _fmt(d.geometry_consistency["mean_abs_z"], 2)),
    ], accent=AQUA, note="Normalized features compared to prototype statistics (mean, std).")

    axb = fig.add_subplot(gs[3, 0:4])
    labels = [c for c, _ in b.top5][::-1]
    vals = [p for _, p in b.top5][::-1]
    yy = np.arange(len(labels))
    axb.barh(yy, vals, color=[BLUE if c == b.top1 else "#86b6ef" for c in labels], edgecolor=SURFACE,
             linewidth=2, height=0.62)
    for yi, val in zip(yy, vals):
        axb.text(val + 0.02, yi, f"{val:.3f}", va="center", fontsize=10)
    axb.set_yticks(yy)
    axb.set_yticklabels(labels, fontsize=13, fontweight="bold")
    axb.set_xlim(0, 1.18)
    axb.grid(axis="x", lw=0.6)
    axb.set_axisbelow(True)
    bb = r.bayesian_before
    axb.set_title(f"BAYESIAN  ·  Top-1 {b.top1}   Posterior {b.top1_posterior:.3f}   Entropy {b.entropy:.3f}\n"
                  f"before restoration: Top-1 {bb.top1} {bb.top1_posterior:.3f}, Entropy {bb.entropy:.3f}",
                  fontsize=11, loc="left")
    axb.set_xlabel("Posterior Probability (prototype reference model, not calibrated)")

    if v.status == "OK":
        alts = ", ".join(f"{a['label']}:{a['confidence']:.2f}" for a in v.alternatives[:3]) or "—"
        vrows = [("Status", v.status), ("Top-1", str(v.top_candidate)),
                 ("Model-Reported Confidence", _fmt(v.confidence or 0.0, 2)), ("Alternatives", alts),
                 ("Model", v.model)]
        vnote = (v.visual_reason[:110] + "…") if len(v.visual_reason) > 110 else v.visual_reason
    else:
        vrows = [("Status", v.status), ("Top-1", "—"), ("Model-Reported Confidence", "—"), ("Alternatives", "—")]
        vnote = "VLM not used: final result = Bayesian evidence only."
    _panel(fig.add_subplot(gs[3, 4:8]), "VLM  (independent visual verification)", vrows, accent=ORANGE,
           note=vnote, value_size=11)

    axd = fig.add_subplot(gs[3, 8:12])
    _panel(axd, "FUSION / DECISION", [
        ("Final candidate", f.final_candidate), ("Fusion score", _fmt(f.fusion_score)),
        ("Agreement", f.agreement),
        ("Topology consistency", "consistent" if d.topology_consistency["consistent"] else "INCONSISTENT"),
        ("Geometry consistency", "consistent" if d.geometry_consistency["consistent"] else "INCONSISTENT"),
    ], accent=STATUS.get(d.level, MUTED), value_size=11, max_step=0.088)
    color = STATUS.get(d.level, MUTED)
    axd.add_patch(FancyBboxPatch((0.04, 0.03), 0.92, 0.19, boxstyle="round,pad=0,rounding_size=0.03",
                                 transform=axd.transAxes, facecolor=color, edgecolor="none", alpha=0.18))
    axd.text(0.5, 0.16, f"{STATUS_ICON.get(d.level, '')} {d.level.replace('_', ' ')}", transform=axd.transAxes,
             ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
    axd.text(0.5, 0.07, d.final_decision.replace("_", " "), transform=axd.transAxes, ha="center",
             va="center", fontsize=11, fontweight="bold", color=INK_2)

    timing = "  ".join(f"{k}: {val:.0f} ms" for k, val in r.timing_ms.items()
                       if k in ("Preprocessing", "Topology", "Geometry", "Restoration", "Bayesian", "VLM", "Fusion"))
    fig.text(0.012, 0.012, f"Timing — {timing}", fontsize=9, color=INK_2)
    fig.text(0.988, 0.012, "Demo decision thresholds (not industrial safety criteria) · welding conditions NOT "
             "determined · semantic interpretation NOT_IMPLEMENTED", fontsize=9, color=MUTED, ha="right")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.05)
    return _save(fig, path)


def render_all(r: "PipelineResult") -> Dict[str, Path]:
    """Render every figure; a failing figure is reported but does not stop the others."""
    out = r.output_dir
    jobs = {
        "preprocessing_png": (plot_preprocessing, out / "preprocessing.png"),
        "topology_png": (plot_topology, out / "topology_analysis.png"),
        "restoration_png": (plot_restoration, out / "restoration_comparison.png"),
        "geometry_png": (plot_geometry, out / "geometry_analysis.png"),
        "bayesian_png": (plot_bayesian, out / "bayesian_posterior.png"),
        "dashboard_png": (plot_dashboard, out / "final_dashboard.png"),
    }
    files: Dict[str, Path] = {}
    for key, (fn, path) in jobs.items():
        try:
            files[key] = fn(r, path)
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] figure {path.name} failed: {type(exc).__name__}: {exc}")
    return files
