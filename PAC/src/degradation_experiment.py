"""DEGRADATION EXPERIMENT: how do geometry / topology / combined degrade with image degradation severity?

Protocol (fixed before the first run, config.DegradationExperimentConfig):
* Characters: COUNT_MATCH segments of the real Symbols data (manual transcriptions).
* Canonical clean character image: the validated clean mask painted with the source image's own
  measured ink / background grey levels on a padded canvas. Degradations act on this image;
  test-time binarization is fixed (Otsu, source polarity, existing small-component removal).
  Training uses the same pipeline at severity 0 (clean) -> train / test pipelines are identical.
* CLEAN TRAIN -> DEGRADED TEST, leave-one-group-out (same group folds as character experiment B / A).
  Models are fitted once per fold on clean training groups BEFORE any degraded sample is scored.
* 8 synthetic degradation surrogates x severities 1..4 (+ shared clean severity 0), seeded replicates.
* Failed feature extraction (e.g. empty mask) counts as an error for every model (never excluded).
* Group-level paired bootstrap for all CIs. No restoration.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .config import DEGRADATION_RESULTS_DIR, AppConfig
from .dataset import FORMAL, INFORMAL, run_dataset_audit
from .image_io import load_image
from .preprocessing import remove_small_components, to_grayscale
from .prototypes import extract_features
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, char_canvas, feature_dict, fit_reference, write_csv

DEGRADATIONS = ["blur", "low_light", "noise", "contrast", "broken_stroke", "occlusion", "rotation", "perspective"]
DEG_LABELS = {"blur": "Gaussian Blur", "low_light": "Low Light", "noise": "Additive Noise",
              "contrast": "Contrast Reduction", "broken_stroke": "Broken Stroke", "occlusion": "Partial Occlusion",
              "rotation": "Rotation", "perspective": "Perspective"}
FIELD_SURROGATE = {
    "blur": "defocus / motion blur in field photography",
    "low_light": "low-light capture (exposure reduction + camera read noise)",
    "noise": "sensor noise / speckle contamination (simple surrogate)",
    "contrast": "faded paint / haze / dirt lowering visibility (simple surrogate)",
    "broken_stroke": "stroke breaks from rust, peeling or worn marking paint (surrogate)",
    "occlusion": "partial covering by dirt, weld spatter, tape or objects (surrogate)",
    "rotation": "camera roll / marking written at an angle",
    "perspective": "oblique (non-frontal) camera viewpoint",
}
MODELS = ("geometry", "topology", "combined")
SEVERITIES = (0, 1, 2, 3, 4)


class LeakageError(RuntimeError):
    """Any leakage / fitting assertion failure aborts the experiment."""


# ====================================================================== data
@dataclass
class CharItem:
    """One clean character and its canonical image (no label is ever passed to degradation)."""

    index: int
    sample_id: str
    image_id: str
    image_path: str
    group_id: str
    style: str
    character: str
    canonical: np.ndarray = field(repr=False)   # uint8 grey canvas
    clean_mask: np.ndarray = field(repr=False)  # bool canvas (foreground of the canonical image)
    dark_ink: bool = True
    ink_level: float = 0.0
    bg_level: float = 255.0


def build_items(cfg: AppConfig, say: Callable[[str], None] = print) -> List[CharItem]:
    """Canonical clean character images from COUNT_MATCH images (fresh audit, nothing written)."""
    d = cfg.degradation
    audit = run_dataset_audit(cfg, write=False, previews=False, verbose=False)
    items: List[CharItem] = []
    for m in sorted(audit.manifest, key=lambda r: str(r["image_path"])):
        if m["validation_status"] != "COUNT_MATCH" or m["valid_for_training"] is not True:
            continue
        if m["style"] not in (FORMAL, INFORMAL):
            continue
        seg = audit.segmentations[str(m["image_path"])]
        text = str(m["normalized_transcription"])
        if seg.count != len(text):
            raise LeakageError(f"alignment invariant broken: {m['image_path']}")
        gray = to_grayscale(load_image(Path(str(m["image_path"])) if Path(str(m["image_path"])).is_absolute()
                                       else _root() / str(m["image_path"])).image)
        all_ink = seg.mask
        image_id = Path(str(m["image_path"])).stem
        for k, ((x0, y0, x1, y1), cm, ch) in enumerate(zip(seg.boxes, seg.char_masks, text)):
            full = np.zeros_like(all_ink)
            full[y0:y1, x0:x1] = cm
            ring = (cv2.dilate(full.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~all_ink
            ink = float(np.median(gray[full]))
            bg = float(np.median(gray[ring])) if ring.any() else float(np.median(gray[~all_ink]))
            h, w = cm.shape
            p = int(math.ceil(d.pad_ratio * math.hypot(h, w))) + 4
            canvas = np.full((h + 2 * p, w + 2 * p), bg, np.float32)
            mask = np.zeros(canvas.shape, bool)
            mask[p:p + h, p:p + w] = cm
            canvas[mask] = ink
            items.append(CharItem(len(items), f"{image_id}#{k}", image_id, str(m["image_path"]), str(m["group_id"]),
                                  str(m["style"]), ch, np.clip(np.round(canvas), 0, 255).astype(np.uint8), mask,
                                  ink < bg, ink, bg))
    say(f"[DEG] {len(items)} clean characters (canonical images) from "
        f"{len({i.image_id for i in items})} COUNT_MATCH images")
    return items


def _root() -> Path:
    from .config import PROJECT_ROOT

    return PROJECT_ROOT


# ====================================================================== degradations (label-free)
def stroke_width(mask: np.ndarray) -> float:
    """2 x median distance-to-background on the skeleton (foreground information only)."""
    skel = skeletonize(mask)
    dt = ndimage.distance_transform_edt(mask)
    return max(1.0, 2.0 * float(np.median(dt[skel]))) if skel.any() else 1.0


def degrade(kind: str, s: int, img: np.ndarray, mask: np.ndarray, ink: float, bg: float,
            seed: int, cfg: AppConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply one degradation at severity s. Inputs: image + its own foreground only (never the class)."""
    d = cfg.degradation
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    meta: Dict[str, object] = {}
    if s == 0:
        return out, meta
    ys, xs = np.nonzero(mask)
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = bx1 - bx0, by1 - by0
    H, W = out.shape
    if kind == "blur":
        sigma = d.blur_sigma_x_stroke[s] * stroke_width(mask)
        out = cv2.GaussianBlur(out, (0, 0), sigma)
        meta["sigma_px"] = sigma
    elif kind == "low_light":
        out = out * d.low_light_factor[s] + rng.normal(0, d.read_noise_sigma, out.shape)
        out = np.round(out)
    elif kind == "noise":
        sigma = d.noise_sigma_x_contrast[s] * abs(ink - bg)
        out = out + rng.normal(0, sigma, out.shape)
        meta["sigma_grey"] = sigma
    elif kind == "contrast":
        mean = float(out.mean())
        out = mean + d.contrast_factor[s] * (out - mean) + rng.normal(0, d.read_noise_sigma, out.shape)
        out = np.round(out)
    elif kind == "broken_stroke":
        sw = stroke_width(mask)
        r = max(1, int(round(d.stroke_cut_radius_x_stroke[s] * sw)))
        sk = np.argwhere(skeletonize(mask))
        for _ in range(d.stroke_cuts[s]):
            cy, cx = sk[rng.integers(0, len(sk))]
            cv2.circle(out, (int(cx), int(cy)), r, float(bg), -1)
        meta["cut_radius_px"] = r
    elif kind == "occlusion":
        area = d.occlusion_area_fraction[s] * bw * bh
        aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        rw = max(1, int(round(math.sqrt(area * aspect))))
        rh = max(1, int(round(area / rw)))
        cx = rng.uniform(bx0, bx1)
        cy = rng.uniform(by0, by1)
        x0, y0 = int(round(cx - rw / 2)), int(round(cy - rh / 2))
        rect = np.zeros_like(mask)
        rect[max(0, y0):max(0, y0 + rh), max(0, x0):max(0, x0 + rw)] = True
        out[rect] = bg
        meta["occluded_fraction"] = float((rect & mask).sum() / mask.sum())
    elif kind in ("rotation", "perspective"):
        if kind == "rotation":
            angle = d.rotation_deg[s] * (1 if rng.random() < 0.5 else -1)
            M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
            warp = lambda a, fill, interp: cv2.warpAffine(a, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                                                          borderValue=fill)
            meta["angle_deg"] = angle
        else:
            shift = d.perspective_corner_shift[s] * max(bw, bh)
            src = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
            dst = src + rng.uniform(-shift, shift, src.shape).astype(np.float32)
            P = cv2.getPerspectiveTransform(src, dst)
            warp = lambda a, fill, interp: cv2.warpPerspective(a, P, (W, H), flags=interp,
                                                               borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
            meta["corner_shift_px"] = shift
        out = warp(out, float(bg), cv2.INTER_LINEAR)
        moved = warp(mask.astype(np.uint8), 0, cv2.INTER_NEAREST) > 0
        meta["clipped"] = bool(moved[0].any() or moved[-1].any() or moved[:, 0].any() or moved[:, -1].any()
                               or moved.sum() < 0.98 * mask.sum())
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 255), meta


def binarize_char(img: np.ndarray, dark_ink: bool, cfg: AppConfig) -> np.ndarray:
    """Fixed test-time binarization: Otsu + source polarity + existing small-component removal."""
    g = np.clip(np.round(img), 0, 255).astype(np.uint8)
    if int(g.max()) == int(g.min()):
        return np.zeros(g.shape, bool)
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = g <= thr if dark_ink else g > thr
    m, _ = remove_small_components(m, cfg.preprocessing.min_component_ratio, cfg.preprocessing.min_component_pixels)
    return m


def variant_seed(base: int, kind: str, s: int, r: int, index: int) -> int:
    """Deterministic, recorded seed for one degraded variant."""
    t = DEGRADATIONS.index(kind) if kind in DEGRADATIONS else 99
    return int(np.random.SeedSequence([base, t, s, r, index]).generate_state(1)[0])


def _features(mask: np.ndarray, cfg: AppConfig) -> Optional[Dict[str, float]]:
    if mask.sum() == 0:
        return None
    f = feature_dict(extract_features(char_canvas(mask, cfg), cfg, with_persistence=True), with_ph=True)
    return {k: float(f[k]) for k in TOPO_FEATURES + GEO_FEATURES}


def process_item(payload: Tuple[CharItem, AppConfig]) -> List[Dict[str, object]]:
    """All variants of one character (runs in a worker process). No label is used."""
    item, cfg = payload
    d = cfg.degradation
    rows: List[Dict[str, object]] = []

    def run(kind: str, s: int, r: int) -> Dict[str, object]:
        seed = variant_seed(d.seed, kind, s, r, item.index)
        img, meta = degrade(kind, s, item.canonical, item.clean_mask, item.ink_level, item.bg_level, seed, cfg)
        m = binarize_char(img, item.dark_ink, cfg)
        try:
            x = _features(m, cfg)
            fail = "" if x is not None else "empty mask after binarization"
        except Exception as exc:  # noqa: BLE001 - recorded as a failure, counted as an error
            x, fail = None, f"{type(exc).__name__}: {exc}"
        return {"sample_id": item.sample_id, "degradation": kind, "severity": s, "replicate": r, "seed": seed,
                "x": x, "failure": fail, "identity": bool(np.array_equal(m, item.clean_mask)) if s == 0 else None,
                "occluded_fraction": meta.get("occluded_fraction"), "clipped": meta.get("clipped"),
                **{k: v for k, v in meta.items() if k not in ("occluded_fraction", "clipped")}}

    rows.append(run("clean", 0, 0))
    for kind in DEGRADATIONS:
        reps = d.blur_replicates if kind == "blur" else d.replicates
        for s in SEVERITIES[1:]:
            for r in range(reps):
                rows.append(run(kind, s, r))
    return rows


def extract_all(items: Sequence[CharItem], cfg: AppConfig, say: Callable[[str], None]) -> Dict[str, List[Dict]]:
    """Parallel variant generation + feature extraction."""
    workers = cfg.degradation.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    out: Dict[str, List[Dict]] = {}
    t0 = time.perf_counter()
    payloads = [(it, cfg) for it in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(ex.map(process_item, payloads, chunksize=4)):
            out[items[i].sample_id] = rows
            if (i + 1) % max(1, len(items) // 10) == 0:
                say(f"[DEG] features {i + 1}/{len(items)} characters ({time.perf_counter() - t0:.0f} s)")
    return out


# ====================================================================== evaluation
def _model_fingerprint(models: Dict[str, object]) -> Tuple:
    m = models["combined"].clf.model
    return tuple(round(m.mean[c][f], 12) for c in m.classes for f in m.feature_names)


def evaluate_domain(domain: str, items: Sequence[CharItem], variants: Dict[str, List[Dict]],
                    cfg: AppConfig) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, np.ndarray]]:
    """Leave-one-group-out; clean training only; scores every variant of the held-out group."""
    c = cfg.character
    weights = {"geometry": (0.0, 1.0), "topology": (1.0, 0.0), "combined": (c.w_topology, c.w_geometry)}
    pool = [it for it in items if it.style == domain]
    groups = sorted({it.group_id for it in pool})
    preds: List[Dict[str, object]] = []
    checks = Counter()
    train_std: Dict[str, np.ndarray] = {}
    feats = TOPO_FEATURES + GEO_FEATURES
    for g in groups:
        test = [it for it in pool if it.group_id == g]
        train = [it for it in pool if it.group_id != g]
        if {it.group_id for it in train} & {g}:
            raise LeakageError("test group overlaps training group")
        fit_rows, fit_labels, fit_ids = [], [], []
        for it in train:
            v0 = variants[it.sample_id][0]
            if v0["degradation"] != "clean" or v0["severity"] != 0:
                raise LeakageError("degraded variant reached training")
            if v0["x"] is not None:
                fit_rows.append(v0["x"])
                fit_labels.append(it.character)
                fit_ids.append(it.sample_id)
        if {it.sample_id for it in test} & set(fit_ids):
            raise LeakageError("test clean image used in fitting")
        X = np.array([[r[f] for f in feats] for r in fit_rows])
        std = X.std(axis=0)
        std[std == 0] = 1.0
        for it in test:
            train_std[it.sample_id] = std
        if len(set(fit_labels)) < 2:
            continue
        models = {m: fit_reference(fit_rows, fit_labels, TOPO_FEATURES, GEO_FEATURES, w, cfg) for m, w in weights.items()}
        fp = _model_fingerprint(models)
        classes = set(models["combined"].classes)
        k = len(classes)
        for it in test:
            evaluable = it.character in classes
            for v in variants[it.sample_id]:
                row = {"domain": domain, "sample_id": it.sample_id, "group_id": g, "character": it.character,
                       "degradation": v["degradation"], "severity": v["severity"], "replicate": v["replicate"],
                       "evaluable": evaluable, "extraction_failed": v["x"] is None}
                if evaluable:
                    for name, model in models.items():
                        if v["x"] is None:
                            row.update({f"{name}_top1": "", f"{name}_correct": False, f"{name}_top3_correct": False,
                                        f"{name}_entropy": math.log(k)})
                            continue
                        x = dict(v["x"])
                        res = model.clf.predict(x)
                        ranked = list(res.posteriors)
                        row.update({f"{name}_top1": res.top1, f"{name}_correct": res.top1 == it.character,
                                    f"{name}_top3_correct": it.character in ranked[:3], f"{name}_entropy": res.entropy})
                preds.append(row)
        if _model_fingerprint(models) != fp:
            raise LeakageError("classifier parameters changed during degraded evaluation")
        checks["folds"] += 1
    return preds, dict(checks), train_std


# ====================================================================== statistics
def curve_stats(preds: Sequence[Dict[str, object]], cfg: AppConfig) -> Dict[str, object]:
    """Accuracy / retention / gains / AUC per degradation x severity x model, with paired group bootstrap."""
    ev = [p for p in preds if p["evaluable"]]
    groups = sorted({p["group_id"] for p in ev})
    gi = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(cfg.degradation.seed)
    draws = rng.integers(0, len(groups), size=(cfg.degradation.bootstrap_iterations, len(groups)))
    cells: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for p in ev:
        if p["degradation"] == "clean":
            for kind in DEGRADATIONS:
                cells[(kind, 0)].append(p)
        else:
            cells[(p["degradation"], p["severity"])].append(p)
    out: Dict[str, object] = {"n_groups": len(groups), "cells": {}, "boot": {}}
    boot_acc: Dict[Tuple[str, int, str], np.ndarray] = {}
    for (kind, s), ps in cells.items():
        n = np.zeros(len(groups))
        for p in ps:
            n[gi[p["group_id"]]] += 1
        cell = {"n_characters": len({p["sample_id"] for p in ps}), "n_variants": len(ps),
                "n_groups": len({p["group_id"] for p in ps}),
                "n_extraction_failed": sum(p["extraction_failed"] for p in ps)}
        for m in MODELS:
            corr = np.zeros(len(groups))
            for p in ps:
                corr[gi[p["group_id"]]] += p[f"{m}_correct"]
            per_class = defaultdict(list)
            for p in ps:
                per_class[p["character"]].append(p[f"{m}_correct"])
            b = corr[draws].sum(axis=1) / np.maximum(n[draws].sum(axis=1), 1)
            boot_acc[(kind, s, m)] = b
            cell[m] = {"top1_accuracy": float(corr.sum() / n.sum()),
                       "top3_accuracy": float(np.mean([p[f"{m}_top3_correct"] for p in ps])),
                       "macro_accuracy": float(np.mean([np.mean(v) for v in per_class.values()])),
                       "mean_entropy": float(np.mean([p[f"{m}_entropy"] for p in ps])),
                       "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        out["cells"][f"{kind}|{s}"] = cell
    # derived: drop, retention, gains, AUC (all with paired bootstrap)
    derived = {}
    for kind in DEGRADATIONS:
        for m in MODELS:
            b0 = boot_acc[(kind, 0, m)]
            a0 = out["cells"][f"{kind}|0"][m]["top1_accuracy"]
            accs = [out["cells"][f"{kind}|{s}"][m]["top1_accuracy"] for s in SEVERITIES]
            rets = [a / a0 if a0 > 0 else float("nan") for a in accs]
            bret = [boot_acc[(kind, s, m)] / np.where(b0 > 0, b0, np.nan) for s in SEVERITIES]
            auc = _auc(accs)
            bauc = _auc_arr([boot_acc[(kind, s, m)] for s in SEVERITIES])
            rauc = _auc(rets)
            brauc = _auc_arr(bret)
            derived[f"{kind}|{m}"] = {
                "accuracy": accs, "drop_pp": [100 * (a0 - a) for a in accs], "retention": rets,
                "retention_ci95": [[float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))] for x in bret],
                "robustness_auc": auc, "robustness_auc_ci95": _ci(bauc),
                "retention_auc": rauc, "retention_auc_ci95": _ci(brauc),
            }
            out["boot"][f"{kind}|{m}|auc"] = bauc
            out["boot"][f"{kind}|{m}|rauc"] = brauc
        for s in SEVERITIES:
            g = boot_acc[(kind, s, "geometry")]
            for m in ("topology", "combined"):
                d = 100 * (boot_acc[(kind, s, m)] - g)
                key = "topology_advantage_pp" if m == "topology" else "combined_gain_pp"
                cell = out["cells"][f"{kind}|{s}"]
                cell[key] = 100 * (cell[m]["top1_accuracy"] - cell["geometry"]["top1_accuracy"])
                cell[key + "_ci95"] = _ci(d)
        diff = out["boot"][f"{kind}|topology|rauc"] - out["boot"][f"{kind}|geometry|rauc"]
        derived[f"{kind}|retention_auc_topology_minus_geometry"] = {
            "value": derived[f"{kind}|topology"]["retention_auc"] - derived[f"{kind}|geometry"]["retention_auc"],
            "ci95": _ci(diff)}
    out["derived"] = derived
    out.pop("boot")
    return out


def _auc(vals: Sequence[float]) -> float:
    v = np.asarray(vals, dtype=float)
    return float((0.5 * v[0] + v[1:-1].sum() + 0.5 * v[-1]) / (len(v) - 1))


def _auc_arr(arrs: Sequence[np.ndarray]) -> np.ndarray:
    a = np.vstack(arrs)
    return (0.5 * a[0] + a[1:-1].sum(axis=0) + 0.5 * a[-1]) / (len(arrs) - 1)


def _ci(x: np.ndarray) -> List[float]:
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def stability(items: Sequence[CharItem], variants: Dict[str, List[Dict]], train_std: Dict[str, np.ndarray],
              domain: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Clean-vs-degraded feature changes (topology counts; training-normalized distances)."""
    feats = TOPO_FEATURES + GEO_FEATURES
    ti = [feats.index(f) for f in TOPO_FEATURES]
    gi = [feats.index(f) for f in GEO_FEATURES]
    acc = defaultdict(lambda: defaultdict(list))
    for it in items:
        if it.style != domain or it.sample_id not in train_std:
            continue
        vs = variants[it.sample_id]
        x0 = vs[0]["x"]
        if x0 is None:
            continue
        v0 = np.array([x0[f] for f in feats])
        std = train_std[it.sample_id]
        for v in vs[1:]:
            key = (v["degradation"], v["severity"])
            if v["x"] is None:
                acc[key]["failed"].append(1)
                continue
            acc[key]["failed"].append(0)
            x = v["x"]
            vv = np.array([x[f] for f in feats])
            z = np.abs(vv - v0) / std
            acc[key]["topo_dist"].append(float(z[ti].mean()))
            acc[key]["geo_dist"].append(float(z[gi].mean()))
            for f, name in (("beta_0", "d_beta0"), ("beta_1", "d_beta1"), ("euler_characteristic", "d_euler"),
                            ("endpoints", "d_endpoints"), ("branch_points", "d_branch")):
                acc[key][name].append(x[f] - x0[f])
            acc[key]["d_skel_rel"].append(abs(x["skeleton_length_norm"] - x0["skeleton_length_norm"])
                                          / max(x0["skeleton_length_norm"], 1e-6))
            acc[key]["betti_changed"].append(int(x["beta_0"] != x0["beta_0"] or x["beta_1"] != x0["beta_1"]))
            if v.get("occluded_fraction") is not None:
                acc[key]["occluded_fraction"].append(v["occluded_fraction"])
            if v.get("clipped") is not None:
                acc[key]["clipped"].append(int(v["clipped"]))
    topo_rows, geo_rows = [], []
    for (kind, s), a in sorted(acc.items(), key=lambda kv: (DEGRADATIONS.index(kv[0][0]), kv[0][1])):
        n = len(a["topo_dist"])
        base = {"domain": domain, "degradation": kind, "severity": s, "n_variants": n,
                "extraction_failed_rate": float(np.mean(a["failed"])) if a["failed"] else 0.0}
        topo_rows.append({**base,
                          "mean_delta_beta0": _m(a["d_beta0"]), "mean_abs_delta_beta0": _m(np.abs(a["d_beta0"])),
                          "rate_beta0_increased": _m(np.array(a["d_beta0"]) > 0),
                          "mean_delta_beta1": _m(a["d_beta1"]), "mean_abs_delta_beta1": _m(np.abs(a["d_beta1"])),
                          "rate_beta1_decreased": _m(np.array(a["d_beta1"]) < 0),
                          "mean_delta_euler": _m(a["d_euler"]), "mean_delta_endpoints": _m(a["d_endpoints"]),
                          "mean_abs_delta_endpoints": _m(np.abs(a["d_endpoints"])),
                          "mean_delta_branch_points": _m(a["d_branch"]),
                          "mean_rel_change_skeleton_length": _m(a["d_skel_rel"]),
                          "betti_change_rate": _m(a["betti_changed"]),
                          "topology_feature_distance_norm": _m(a["topo_dist"]),
                          "mean_occluded_fraction": _m(a["occluded_fraction"]) if a["occluded_fraction"] else "",
                          "clipping_rate": _m(a["clipped"]) if a["clipped"] else ""})
        geo_rows.append({**base, "geometry_feature_distance_norm": _m(a["geo_dist"]),
                         "geometry_distance_median": float(np.median(a["geo_dist"])) if n else float("nan"),
                         "topology_feature_distance_norm": _m(a["topo_dist"]),
                         "normalization": "|x_degraded - x_clean| / training-fold std (test excluded)"})
    return topo_rows, geo_rows


def _m(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(v.mean()) if v.size else float("nan")


# ====================================================================== orchestration
@dataclass
class DegradationResult:
    """In-memory results for figures / reports."""

    items: List[CharItem]
    variants: Dict[str, List[Dict]]
    predictions: Dict[str, List[Dict[str, object]]]
    stats: Dict[str, Dict[str, object]]
    topo_stability: List[Dict[str, object]]
    geo_stability: List[Dict[str, object]]
    leakage: Dict[str, object]
    info: Dict[str, object]
    report: Dict[str, object] = field(default_factory=dict)
    out_dir: Path = DEGRADATION_RESULTS_DIR


def parameter_rows(cfg: AppConfig) -> List[Dict[str, object]]:
    """degradation_parameters.csv (exact numeric parameters per severity)."""
    d = cfg.degradation
    spec = {
        "blur": ("Gaussian sigma = k x stroke width (px)", d.blur_sigma_x_stroke),
        "low_light": (f"intensity x factor + read noise sigma {d.read_noise_sigma} + 8-bit rounding", d.low_light_factor),
        "noise": ("additive Gaussian sigma = k x |ink - background| (grey levels)", d.noise_sigma_x_contrast),
        "contrast": (f"mean + factor x (I - mean) + read noise sigma {d.read_noise_sigma} + rounding", d.contrast_factor),
        "broken_stroke": ("number of background-coloured gaps on random skeleton points",
                          d.stroke_cuts),
        "occlusion": ("background-coloured rectangle area = k x foreground bbox area, aspect in [0.5, 2]",
                      d.occlusion_area_fraction),
        "rotation": ("rotation angle (deg), sign from seed", d.rotation_deg),
        "perspective": ("max corner shift = k x max(bbox w, h), uniform per corner", d.perspective_corner_shift),
    }
    rows = []
    for kind, (desc, vals) in spec.items():
        for s in SEVERITIES:
            extra = {}
            if kind == "broken_stroke":
                extra["gap_radius_x_stroke_width"] = d.stroke_cut_radius_x_stroke[s]
            rows.append({"degradation": kind, "label": DEG_LABELS[kind], "severity": s,
                         "severity_name": ["clean", "mild", "moderate", "strong", "severe"][s],
                         "parameter": vals[s], "definition": desc, **extra,
                         "field_surrogate_for": FIELD_SURROGATE[kind],
                         "replicates": 1 if s == 0 or kind == "blur" else d.replicates,
                         "note": "synthetic degradation surrogate; does not reproduce real rust/illumination physics"})
    return rows


def run_degradation_experiment(cfg: AppConfig, out_dir: Path = DEGRADATION_RESULTS_DIR,
                               say: Callable[[str], None] = print) -> DegradationResult:
    """Full experiment (Informal primary, Formal secondary); writes CSVs into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    items = build_items(cfg, say)
    variants = extract_all(items, cfg, say)
    say(f"[DEG] feature extraction done ({time.perf_counter() - t0:.0f} s)")
    preds, stats, checks, topo, geo, stds = {}, {}, {}, [], [], {}
    for domain in (INFORMAL, FORMAL):
        preds[domain], checks[domain], stds[domain] = evaluate_domain(domain, items, variants, cfg)
        stats[domain] = curve_stats(preds[domain], cfg)
        t_rows, g_rows = stability(items, variants, stds[domain], domain)
        topo += t_rows
        geo += g_rows
    s0_identity = [v[0]["identity"] for v in variants.values()]
    info = {"s0_identity_rate": float(np.mean(s0_identity)), "s0_identity_failures": int(len(s0_identity) - sum(s0_identity)),
            "characters": len(items), "variants": sum(len(v) for v in variants.values()),
            "extraction_failures": sum(1 for v in variants.values() for x in v if x["x"] is None),
            "runtime_s": round(time.perf_counter() - t0, 1)}
    leakage = {"degraded_test_image_used_in_training": False, "test_clean_image_used_in_fitting": False,
               "test_group_overlaps_training_group": False, "severity_changed_classifier_parameters": False,
               "ground_truth_used_to_construct_degradation": False,
               "folds_checked": {k: v.get("folds", 0) for k, v in checks.items()},
               "status": "PASS (assertions executed per fold; any violation aborts)"}
    # outputs
    write_csv(out_dir / "degradation_parameters.csv", parameter_rows(cfg))
    manifest = []
    by_id = {it.sample_id: it for it in items}
    for sid, vs in variants.items():
        it = by_id[sid]
        for v in vs:
            manifest.append({"variant_id": f"{sid}|{v['degradation']}|s{v['severity']}|r{v['replicate']}",
                             "sample_id": sid, "source_image": it.image_path, "group_id": it.group_id,
                             "style": it.style, "character": it.character, "degradation": v["degradation"],
                             "severity": v["severity"], "replicate": v["replicate"], "seed": v["seed"],
                             "occluded_fraction": v.get("occluded_fraction"), "clipped": v.get("clipped"),
                             "s0_identity": v.get("identity"), "extraction_failure": v["failure"]})
    write_csv(out_dir / "degradation_sample_manifest.csv", manifest)
    write_csv(out_dir / "topology_stability.csv", topo)
    write_csv(out_dir / "geometry_stability.csv", geo)
    write_csv(out_dir / "predictions_detail.csv", [p for d in preds.values() for p in d])
    res_rows, rob_rows = [], []
    for domain, st in stats.items():
        for key, cell in st["cells"].items():
            kind, s = key.split("|")
            for m in MODELS:
                res_rows.append({"domain": domain, "degradation": kind, "severity": int(s), "model": m,
                                 "top1_accuracy": cell[m]["top1_accuracy"], "top1_ci95_low": cell[m]["ci95"][0],
                                 "top1_ci95_high": cell[m]["ci95"][1], "top3_accuracy": cell[m]["top3_accuracy"],
                                 "macro_accuracy": cell[m]["macro_accuracy"], "mean_entropy": cell[m]["mean_entropy"],
                                 "n_characters": cell["n_characters"], "n_variants": cell["n_variants"],
                                 "n_groups": cell["n_groups"], "n_extraction_failed": cell["n_extraction_failed"],
                                 "drop_pp": st["derived"][f"{kind}|{m}"]["drop_pp"][int(s)],
                                 "retention": st["derived"][f"{kind}|{m}"]["retention"][int(s)],
                                 "topology_advantage_pp": cell.get("topology_advantage_pp"),
                                 "combined_gain_pp": cell.get("combined_gain_pp")})
        for kind in DEGRADATIONS:
            for m in MODELS:
                d = st["derived"][f"{kind}|{m}"]
                rob_rows.append({"domain": domain, "degradation": kind, "model": m,
                                 "severity_curve_robustness_auc": d["robustness_auc"],
                                 "auc_ci95_low": d["robustness_auc_ci95"][0], "auc_ci95_high": d["robustness_auc_ci95"][1],
                                 "retention_auc": d["retention_auc"], "retention_auc_ci95_low": d["retention_auc_ci95"][0],
                                 "retention_auc_ci95_high": d["retention_auc_ci95"][1],
                                 "retention_s4": d["retention"][4]})
    res_rows.sort(key=lambda r: (r["domain"], DEGRADATIONS.index(r["degradation"]), r["severity"], r["model"]))
    write_csv(out_dir / "degradation_results.csv", res_rows)
    write_csv(out_dir / "robustness_summary.csv", rob_rows)
    say(f"[DEG] evaluation done ({time.perf_counter() - t0:.0f} s)")
    return DegradationResult(items, variants, preds, stats, topo, geo, leakage, info, out_dir=out_dir)
