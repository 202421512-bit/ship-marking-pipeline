"""EXPLORATORY: direction-aware + line-relative features for real marking recognition.

Existing model, features and results are untouched. Segmentation is FIXED (same segments as the
real-marking-recognition check); only the character representation changes:

  A  existing topology + geometry                 (0.5 / 0.5, must reproduce the existing classifier)
  B  A + direction group                          (1/3 per group)
  C  B + line-relative group                      (1/4 per group)

Same leave-one-group-out split for all models (training = COUNT_MATCH characters of other groups);
class statistics, variance shrinkage and sigma floors from the training partition only.
These features were designed after looking at the current errors (6/9, 2/5, s/S, l/I), so every
improvement here is EXPLORATORY; generalisation needs new, independent source groups.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.special import logsumexp

from .config import DIRECTION_RESULTS_DIR, AppConfig
from .dataset import FORMAL, INFORMAL
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, char_canvas, fit_reference, write_csv
from .real_marking_recognition import MarkingImage, edit_ops, load_markings
from .topology import analyze_topology

MODEL_GROUPS = {
    "A_baseline": ("topology", "geometry"),
    "B_direction": ("topology", "geometry", "direction"),
    "C_direction_line": ("topology", "geometry", "direction", "line"),
}
PAIRS = [("6", "9"), ("2", "5"), ("s", "S"), ("l", "I")]


# ====================================================================== features (label-free)
def direction_features(mask_canvas: np.ndarray, cfg: AppConfig) -> Dict[str, float]:
    """Orientation-sensitive descriptors on the normalized character canvas (no 180°/mirror invariance)."""
    u = cfg.direction.undefined_position
    ys, xs = np.nonzero(mask_canvas)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = max(1, y1 - y0), max(1, x1 - x0)
    ny = lambda y: (y - y0) / h          # noqa: E731
    nx = lambda x: (x - x0) / w          # noqa: E731
    topo = analyze_topology(mask_canvas, cfg.topology, with_persistence=False)
    hl = topo.hole_labels
    if hl.max() > 0:
        hy, hx = np.nonzero(hl > 0)
        hole_y, hole_x = float(ny(hy.mean())), float(nx(hx.mean()))
    else:
        hole_y = hole_x = u
    ep = topo.endpoint_coords
    bp = topo.branch_coords
    crop = mask_canvas[y0:y1, x0:x1].astype(float)
    total = crop.sum()
    rows = np.array_split(crop.sum(axis=1), 3)
    cols = np.array_split(crop.sum(axis=0), 3)
    # stroke orientation histogram: gradient orientation (mod 180) of the ink boundary, 4 bins
    g = cv2.GaussianBlur(mask_canvas.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    sel = mag > 0.1 * mag.max() if mag.max() > 0 else np.zeros_like(mag, bool)
    # edge direction = gradient direction + 90°, folded to [0, 180)
    ang = (np.degrees(np.arctan2(gy[sel], gx[sel])) + 90.0) % 180.0
    wts = mag[sel]
    bins = ((ang + 22.5) // 45).astype(int) % 4          # 0: ~0° (horizontal), 1: ~45°, 2: ~90° (vertical), 3: ~135°
    hist = np.bincount(bins, weights=wts, minlength=4)
    hist = hist / hist.sum() if hist.sum() > 0 else np.full(4, 0.25)
    return {
        "hole_y": hole_y, "hole_x": hole_x,
        "endpoint_y_mean": float(np.mean([ny(p[0]) for p in ep])) if ep else u,
        "endpoint_x_mean": float(np.mean([nx(p[1]) for p in ep])) if ep else u,
        "branch_y_mean": float(np.mean([ny(p[0]) for p in bp])) if bp else u,
        "proj_row_top": float(rows[0].sum() / total), "proj_row_bottom": float(rows[2].sum() / total),
        "proj_col_left": float(cols[0].sum() / total), "proj_col_right": float(cols[2].sum() / total),
        "orient_h": float(hist[0]), "orient_d45": float(hist[1]), "orient_v": float(hist[2]),
        "orient_d135": float(hist[3]),
        "_defined_hole": float(hl.max() > 0), "_defined_endpoint": float(bool(ep)), "_defined_branch": float(bool(bp)),
    }


def line_features(it: MarkingImage) -> List[Dict[str, float]]:
    """Line-relative descriptors from the image's own segment boxes (no ground truth used)."""
    segs = it.segments
    if not segs:
        return []
    hs = np.array([s.box[3] - s.box[1] for s in segs], float)
    ws = np.array([s.box[2] - s.box[0] for s in segs], float)
    med_h, med_w = float(np.median(hs)), float(np.median(ws))
    base = float(np.median([s.box[3] for s in segs]))
    center = (float(np.median([s.box[1] for s in segs])) + base) / 2.0
    gaps = [segs[k + 1].box[0] - segs[k].box[2] for k in range(len(segs) - 1)]
    fill = float(np.mean(gaps)) / med_h if gaps else 0.0
    out = []
    for k, s in enumerate(segs):
        out.append({"rel_height": hs[k] / med_h, "rel_width": ws[k] / max(med_w, 1.0),
                    "baseline_offset": (s.box[3] - base) / med_h,
                    "vcenter_offset": ((s.box[1] + s.box[3]) / 2.0 - center) / med_h,
                    "gap_left": gaps[k - 1] / med_h if k > 0 else fill,
                    "gap_right": gaps[k] / med_h if k < len(gaps) else fill})
    return out


# ====================================================================== N-group Gaussian model
@dataclass
class GroupModel:
    """Same statistics as real_experiment.fit_reference, generalised to N feature groups."""

    classes: List[str]
    groups: Dict[str, List[str]]
    weights: Dict[str, float]
    mean: np.ndarray            # C x F
    sigma: np.ndarray           # C x F
    feats: List[str]

    def log_posterior(self, X: np.ndarray) -> np.ndarray:
        """N x C normalised log posterior (uniform prior)."""
        score = np.zeros((X.shape[0], len(self.classes)))
        for g, fl in self.groups.items():
            idx = [self.feats.index(f) for f in fl]
            d = (X[:, None, idx] - self.mean[None, :, idx]) / self.sigma[None, :, idx]
            ll = -0.5 * (d ** 2 + np.log(self.sigma[None, :, idx] ** 2)).sum(axis=2) / len(idx)
            score += self.weights[g] * ll
        score += np.log(1.0 / len(self.classes))
        return score - logsumexp(score, axis=1, keepdims=True)


def fit_group_model(X: np.ndarray, y: Sequence[str], feats: List[str], groups: Dict[str, List[str]],
                    weights: Dict[str, float], cfg: AppConfig) -> GroupModel:
    """Per-class mean, shrunk std (lambda toward pooled within-class variance), floors from TRAIN std."""
    r = cfg.real
    y = np.asarray(y)
    classes = sorted(set(y))
    means = np.array([X[y == c].mean(axis=0) for c in classes])
    var = np.array([X[y == c].var(axis=0) for c in classes])
    n = np.array([(y == c).sum() for c in classes])[:, None]
    pooled = var.mean(axis=0)
    std = np.sqrt((n * var + r.variance_shrinkage_lambda * pooled) / (n + r.variance_shrinkage_lambda))
    ref_std = X.std(axis=0)
    floor = np.where(ref_std > 0, r.sigma_floor_relative * ref_std, r.sigma_floor_absolute)
    sigma = np.maximum(np.maximum(std, floor[None, :]), 1e-6)
    return GroupModel(classes, groups, weights, means, sigma, feats)


# ====================================================================== experiment
@dataclass
class DirResult:
    """Everything for reports."""

    items: List[MarkingImage]
    feats: Dict[Tuple[str, int], Dict[str, float]]
    preds: Dict[str, Dict[Tuple[str, int], List[Tuple[str, float]]]]
    texts: Dict[str, Dict[str, str]]
    metrics: Dict[str, Dict[str, object]]
    diagnostics: List[Dict[str, object]]
    identity_check: Dict[str, object]
    leakage: Dict[str, object]
    report: Dict[str, object] = field(default_factory=dict)
    out_dir: Path = DIRECTION_RESULTS_DIR


def run_direction_experiment(cfg: AppConfig, out_dir: Path = DIRECTION_RESULTS_DIR,
                             say: Callable[[str], None] = print) -> DirResult:
    """Compute features, run A / B / C on the same LOGO split, evaluate."""
    d = cfg.direction
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_markings(cfg, say)
    feats: Dict[Tuple[str, int], Dict[str, float]] = {}
    for it in items:
        lf = line_features(it)
        for s, l in zip(it.segments, lf):
            if s.x is None:
                continue
            f = dict(s.x)
            f.update(direction_features(char_canvas(s.mask, cfg), cfg))
            f.update(l)
            feats[(it.path, s.index)] = f
    groups_def = {"topology": list(TOPO_FEATURES), "geometry": list(GEO_FEATURES),
                  "direction": list(d.direction_features), "line": list(d.line_features)}
    all_feats = groups_def["topology"] + groups_def["geometry"] + groups_def["direction"] + groups_def["line"]
    pool = [(it.group_id, it.gt[s.index], (it.path, s.index)) for it in items if it.status == "COUNT_MATCH"
            and len(it.segments) == len(it.gt) for s in it.segments if (it.path, s.index) in feats]
    say(f"[DIR] {len(feats)} segments with features; training pool {len(pool)} characters "
        f"({len({p[0] for p in pool})} groups)")
    preds = {m: {} for m in MODEL_GROUPS}
    leak = Counter()
    identity_mismatch = 0
    identity_checked = 0
    by_group = defaultdict(list)
    for it in items:
        by_group[it.group_id].append(it)
    for g, imgs in sorted(by_group.items()):
        train = [p for p in pool if p[0] != g]
        if any(p[0] == g for p in train):
            raise RuntimeError("leakage: test group in training partition")
        leak["folds"] += 1
        Xtr = np.array([[feats[k][f] for f in all_feats] for _, _, k in train])
        ytr = [lab for _, lab, _ in train]
        keys = [(it.path, s.index) for it in imgs for s in it.segments if (it.path, s.index) in feats]
        if not keys:
            continue
        Xte = np.array([[feats[k][f] for f in all_feats] for k in keys])
        for m, gnames in MODEL_GROUPS.items():
            gsel = {gn: groups_def[gn] for gn in gnames}
            wts = {gn: 1.0 / len(gnames) for gn in gnames}
            fsel = [f for gn in gnames for f in groups_def[gn]]
            cols = [all_feats.index(f) for f in fsel]
            model = fit_group_model(Xtr[:, cols], ytr, fsel, gsel, wts, cfg)
            lp = model.log_posterior(Xte[:, cols])
            for k, row in zip(keys, lp):
                order = np.argsort(-row)[:3]
                preds[m][k] = [(model.classes[j], float(np.exp(row[j]))) for j in order]
        # identity check: model A must reproduce the existing classifier (first image of the group)
        ref = fit_reference([dict(zip(all_feats, x)) for x in Xtr], ytr, TOPO_FEATURES, GEO_FEATURES,
                            (cfg.character.w_topology, cfg.character.w_geometry), cfg)
        for k, x in list(zip(keys, Xte))[:2]:
            res = ref.clf.predict(dict(zip(all_feats, x)))
            identity_checked += 1
            identity_mismatch += int(res.top1 != preds["A_baseline"][k][0][0]
                                     or abs(res.top1_posterior - preds["A_baseline"][k][0][1]) > 1e-6)
    if identity_mismatch:
        raise RuntimeError(f"model A does not reproduce the existing classifier ({identity_mismatch} mismatches)")
    texts = {m: {it.path: "".join(preds[m][(it.path, s.index)][0][0] if (it.path, s.index) in preds[m] else "?"
                                  for s in it.segments) for it in items} for m in MODEL_GROUPS}
    metrics = {m: evaluate_model(items, preds[m], texts[m], cfg) for m in MODEL_GROUPS}
    paired_ci(items, preds, metrics, cfg)
    diag = diagnostics(items, feats, list(d.direction_features) + list(d.line_features),
                       list(TOPO_FEATURES) + list(GEO_FEATURES))
    return DirResult(items, feats, preds, texts, metrics, diag,
                     {"checked_predictions": identity_checked, "mismatches": identity_mismatch,
                      "status": "PASS: model A reproduces the existing classifier"},
                     {"group_overlap": 0, "folds": leak["folds"],
                      "normalization": "class mean / shrunk std / sigma floors from training partition only",
                      "status": "PASS"}, out_dir=out_dir)


def evaluate_model(items: Sequence[MarkingImage], pred: Dict, text: Dict[str, str], cfg: AppConfig) -> Dict[str, object]:
    """Character / string metrics, confusion pairs, per-class accuracy."""
    tr = [i for i in items if i.transcribed]
    out: Dict[str, object] = {}
    for scope, sel in (("all", tr), (FORMAL, [i for i in tr if i.style == FORMAL]),
                       (INFORMAL, [i for i in tr if i.style == INFORMAL])):
        cm = [i for i in sel if i.status == "COUNT_MATCH"]
        chars = [(pred[(i.path, s.index)][0][0], i.gt[s.index]) for i in cm for s in i.segments
                 if (i.path, s.index) in pred]
        ed = [edit_ops(text[i.path], i.gt) for i in sel]
        n_gt = sum(len(i.gt) for i in sel)
        out[scope] = {"images": len(sel), "char_accuracy": float(np.mean([p == g for p, g in chars])) if chars else float("nan"),
                      "chars": len(chars),
                      "string_exact_match": float(np.mean([text[i.path] == i.gt for i in sel])) if sel else float("nan"),
                      "cer": sum(e[0] for e in ed) / n_gt if n_gt else float("nan")}
    conf = Counter()
    per = defaultdict(list)
    for i in tr:
        if i.status != "COUNT_MATCH":
            continue
        for s in i.segments:
            k = (i.path, s.index)
            if k in pred:
                p, g = pred[k][0][0], i.gt[s.index]
                per[g].append(p == g)
                if p != g:
                    conf[(g, p)] += 1
    out["pairs"] = {f"{a}/{b}": {f"{a}->{b}": conf[(a, b)], f"{b}->{a}": conf[(b, a)],
                                 f"{a}_n": len(per[a]), f"{b}_n": len(per[b])} for a, b in PAIRS}
    out["per_class"] = {c: (float(np.mean(v)), len(v)) for c, v in per.items()}
    out["top_confusions"] = [f"{g}->{p} x{n}" for (g, p), n in conf.most_common(10)]
    return out


def paired_ci(items: Sequence[MarkingImage], preds: Dict, metrics: Dict, cfg: AppConfig) -> None:
    """Paired group bootstrap CI of char-accuracy differences B-A and C-A (COUNT_MATCH characters)."""
    rows = []
    for i in items:
        if i.transcribed and i.status == "COUNT_MATCH":
            for s in i.segments:
                k = (i.path, s.index)
                if all(k in preds[m] for m in MODEL_GROUPS):
                    rows.append((i.group_id, *[preds[m][k][0][0] == i.gt[s.index] for m in MODEL_GROUPS]))
    groups = sorted({r[0] for r in rows})
    gi = {g: j for j, g in enumerate(groups)}
    n = np.zeros(len(groups))
    c = np.zeros((len(MODEL_GROUPS), len(groups)))
    for r in rows:
        n[gi[r[0]]] += 1
        for j in range(len(MODEL_GROUPS)):
            c[j, gi[r[0]]] += r[1 + j]
    rng = np.random.default_rng(cfg.direction.seed)
    draws = rng.integers(0, len(groups), size=(cfg.direction.bootstrap_iterations, len(groups)))
    acc = c[:, draws].sum(axis=2) / n[draws].sum(axis=1)
    names = list(MODEL_GROUPS)
    for j, m in enumerate(names):
        metrics[m]["char_accuracy_ci95"] = [float(np.percentile(acc[j], 2.5)), float(np.percentile(acc[j], 97.5))]
        if j:
            dlt = 100 * (acc[j] - acc[0])
            metrics[m]["gain_over_A_pp_ci95"] = [float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))]


def diagnostics(items: Sequence[MarkingImage], feats: Dict, new: List[str], old: List[str]) -> List[Dict[str, object]]:
    """Computability, within-group stability and redundancy of each new feature (label-free, no selection)."""
    keys = list(feats)
    X = {f: np.array([feats[k][f] for k in keys]) for f in new + old + ["_defined_hole", "_defined_endpoint",
                                                                       "_defined_branch"]}
    group_of = {it.path: it.group_id for it in items}
    by_pos = defaultdict(list)          # same group + same segment index across variants
    for j, k in enumerate(keys):
        by_pos[(group_of[k[0]], k[1])].append(j)
    rows = []
    for f in new:
        v = X[f]
        tot = v.var()
        within = np.mean([v[idx].var() for idx in by_pos.values() if len(idx) > 1]) if by_pos else float("nan")
        corr_old = max((abs(np.corrcoef(v, X[o])[0, 1]) if X[o].std() > 0 and v.std() > 0 else 0.0, o) for o in old)
        corr_new = max(((abs(np.corrcoef(v, X[o])[0, 1]) if X[o].std() > 0 and v.std() > 0 else 0.0, o)
                        for o in new if o != f), default=(0.0, ""))
        defined = {"hole_y": "_defined_hole", "hole_x": "_defined_hole", "endpoint_y_mean": "_defined_endpoint",
                   "endpoint_x_mean": "_defined_endpoint", "branch_y_mean": "_defined_branch"}.get(f)
        rows.append({"feature": f, "computable_fraction": float(np.isfinite(v).mean()),
                     "defined_without_imputation": float(X[defined].mean()) if defined else 1.0,
                     "within_group_variance_ratio": float(within / tot) if tot > 0 else float("nan"),
                     "max_abs_corr_existing": float(corr_old[0]), "most_correlated_existing": corr_old[1],
                     "max_abs_corr_new": float(corr_new[0]), "most_correlated_new": corr_new[1],
                     "note": "diagnostic only; no data-driven feature selection was performed"})
    return rows
