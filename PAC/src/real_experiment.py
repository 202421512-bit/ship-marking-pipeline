"""REAL SYMBOLS DATA EXPERIMENT (source: data/input/Symbols).

No synthetic prototype, demo image or data/input/test.png is used anywhere here.

* Welding Symbol (REAL LABELED DATASET, label = filename): deterministic ROI (caption
  removal by a fixed rule), topology + geometry (+ GUDHI persistence) features,
  3-fold leave-one-sample-per-class CV with per-fold reference statistics only,
  models A geometry / B topology / C topology+geometry / D C + restoration.
* Formal / Informal crops: segmentation + topology + geometry for every image;
  ground truth only from user transcriptions (else UNKNOWN / NOT_EVALUABLE);
  welding-trained OUT-OF-DISTRIBUTION PROBE on every character segment.
* Informal scenes: qualitative marking-candidate extraction (no detection ground truth).
Every discovered image ends in all_images_status.csv as SUCCESS or FAILED.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .bayesian import BayesianResult, BayesianShapeClassifier
from .config import PROJECT_ROOT, REAL_RESULTS_DIR, SUPPORTED_EXTENSIONS, SYMBOLS_DIR, AppConfig
from .dataset import (FORMAL, INFORMAL, INFORMAL_SCENE, UNKNOWN, Segmentation, normalize_transcription,
                      run_dataset_audit, scan_text_images)
from .image_io import ImageLoadError, load_image
from .preprocessing import normalize_size, to_grayscale
from .prototypes import extract_features
from .prototypes import PrototypeModel
from .restoration import restore

WELDING = "WELDING"
TOPO_FEATURES = ["beta_0", "beta_1", "euler_characteristic", "skeleton_length_norm", "endpoints",
                 "branch_points", "ph_robust_h1", "ph_near_holes"]
CHAR_TOPO_FEATURES = TOPO_FEATURES[:6]
GEO_FEATURES = ["aspect_ratio", "area_ratio", "perimeter_norm", "circularity", "solidity", "eccentricity",
                "orient_cos2", "orient_sin2", "hu_1", "hu_2", "hu_3", "hu_4"]
MODELS = {"A_geometry_only": (0.0, 1.0), "B_topology_only": (1.0, 0.0), "C_topology_geometry": (0.5, 0.5)}
MODEL_D = "D_topology_geometry_restoration"
POSTERIOR_LABEL = "UNCALIBRATED MODEL POSTERIOR"


# ====================================================================== helpers
def rel(path: Path) -> str:
    """Project-relative POSIX path."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Optional[Sequence[str]] = None) -> Path:
    """UTF-8-BOM CSV (Excel friendly); columns default to the union of row keys in order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for r in rows:
            columns += [k for k in r if k not in columns]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    return path


def feature_dict(bundle, with_ph: bool) -> Dict[str, float]:
    """Flat feature dict incl. orientation (doubled-angle) and optional persistence features."""
    f = dict(bundle.features)
    theta = math.radians(bundle.geometry.orientation_deg)
    f["orient_cos2"], f["orient_sin2"] = math.cos(2 * theta), math.sin(2 * theta)
    if with_ph:
        ph = bundle.topology.persistence
        ok = ph.get("status") == "OK"
        f["ph_robust_h1"] = float(ph.get("robust_H1_count", 0)) if ok else float(bundle.topology.beta_1)
        f["ph_near_holes"] = float(ph.get("near_hole_count", 0)) if ok else 0.0
        f["ph_status"] = ph.get("status", "")
    return f


def normalize_drawing(mask: np.ndarray, canvas: int, box: int) -> np.ndarray:
    """Fit a (thin-line) mask into a square canvas; dilate before downscaling so 1-2 px lines survive."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("empty mask")
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = box / float(max(crop.shape))
    k = max(1, int(round(1.0 / scale))) if scale < 1 else 1
    if k > 1:
        crop = cv2.dilate(crop.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    nh, nw = max(1, int(round(crop.shape[0] * scale))), max(1, int(round(crop.shape[1] * scale)))
    small = cv2.resize(crop.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) >= 0.5
    out = np.zeros((canvas, canvas), bool)
    oy, ox = (canvas - nh) // 2, (canvas - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = small
    return out


def char_canvas(mask: np.ndarray, cfg: AppConfig) -> np.ndarray:
    """Character-scale normalization (same as the single-character pipeline, 64 px canvas)."""
    gray = np.where(mask, 0, 255).astype(np.float32)
    m, _, _, _ = normalize_size(mask, mask.astype(np.float32), gray, cfg.preprocessing)
    return m


# ====================================================================== inventory
@dataclass
class ImageRecord:
    """One discovered image and how it was used."""

    path: Path
    dataset_type: str
    group_id: str = ""
    variant: str = ""
    label: str = ""
    width: int = 0
    height: int = 0
    status: str = "PENDING"
    failure_reason: str = ""
    used_for: List[str] = field(default_factory=list)

    def row(self) -> Dict[str, object]:
        """all_images_status.csv row."""
        return {"image_path": rel(self.path), "dataset_type": self.dataset_type, "group_id": self.group_id,
                "variant": self.variant, "label": self.label, "width": self.width, "height": self.height,
                "processing_status": self.status, "analysis_used_for": ";".join(self.used_for),
                "failure_reason": self.failure_reason}


def discover(symbols_dir: Path, cfg: AppConfig) -> List[ImageRecord]:
    """Every image file under Symbols/ (any supported extension), typed by its real folder/filename."""
    import re

    d = cfg.dataset
    text = {e.path.resolve(): e for e in scan_text_images(symbols_dir, d)}
    records: List[ImageRecord] = []
    for p in sorted(x for x in symbols_dir.rglob("*") if x.is_file() and x.suffix.lower() in SUPPORTED_EXTENSIONS):
        rp = p.resolve()
        if p.parent.name == d.welding_dir:
            m = re.match(d.welding_pattern, p.stem)
            records.append(ImageRecord(p, WELDING, m.group(1) if m else "", m.group(2) if m else "",
                                       m.group(1) if m else ""))
        elif rp in text:
            e = text[rp]
            records.append(ImageRecord(p, e.style, e.group_id, e.variant))
        else:
            records.append(ImageRecord(p, UNKNOWN))
    for r in records:
        try:
            img = load_image(r.path).image
            r.height, r.width = img.shape[:2]
        except (ImageLoadError, Exception) as exc:
            r.status, r.failure_reason = "FAILED", f"unreadable/corrupted: {exc}"
        if r.dataset_type == UNKNOWN and r.status != "FAILED":
            r.status, r.failure_reason = "FAILED", "unrecognized folder/filename pattern"
        if r.dataset_type == WELDING and not r.label and r.status != "FAILED":
            r.status, r.failure_reason = "FAILED", "welding filename does not match <class>_<index>"
    return records


def audit_summary(records: List[ImageRecord], symbols_dir: Path) -> Dict[str, object]:
    """dataset_audit.json content."""
    by = Counter(r.dataset_type for r in records)
    groups = defaultdict(set)
    for r in records:
        groups[r.dataset_type].add(r.group_id)
    weld = Counter(r.label for r in records if r.dataset_type == WELDING and r.label)
    return {
        "dataset_source": rel(symbols_dir),
        "total_images": len(records),
        "formal_images": by.get(FORMAL, 0), "formal_groups": len(groups.get(FORMAL, ())),
        "informal_crop_images": by.get(INFORMAL, 0), "informal_groups": len(groups.get(INFORMAL, ())),
        "informal_scene_images": by.get(INFORMAL_SCENE, 0),
        "welding_images": by.get(WELDING, 0), "welding_classes": len(weld),
        "images_per_welding_class": dict(sorted(weld.items())),
        "unknown_images": by.get(UNKNOWN, 0),
        "invalid_or_corrupted_images": sum(1 for r in records if r.failure_reason.startswith("unreadable")),
        "extensions": dict(Counter(r.path.suffix.lower() for r in records)),
    }


# ====================================================================== welding ROI + features
def welding_roi(image: np.ndarray, cfg: AppConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    """Drawing-only mask with one fixed rule for all images (never uses the label).

    1. ink = grey < welding_ink_threshold (drawing ink is near-black; caption text and
       warp border fill are mid-grey).
    2. components lying entirely inside the top caption band are removed (class-name caption).
    """
    r = cfg.real
    gray = to_grayscale(image)
    ink = gray < r.welding_ink_threshold
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    band = int(r.caption_band_ratio * gray.shape[0])
    keep = [i for i in range(1, n) if stats[i, 1] + stats[i, 3] > band]
    removed = (n - 1) - len(keep)
    mask = np.isin(labels, keep)
    # Caption check on a permissive threshold: is there any grey text left in the band after ROI?
    caption_like = (gray < 200)[:band]
    roi_band_ink = int(mask[:band].sum())
    return mask, {"caption_band_px": band, "removed_components": removed,
                  "caption_band_pixels_in_original_lt200": int(caption_like.sum()),
                  "roi_ink_pixels_in_caption_band": roi_band_ink,
                  "caption_removed_verified": roi_band_ink == 0}


def welding_cfg(cfg: AppConfig) -> AppConfig:
    """Config variant with the larger drawing canvas (feature normalization by canvas size)."""
    return replace(cfg, preprocessing=replace(cfg.preprocessing, normalized_size=cfg.real.welding_canvas,
                                              glyph_box=cfg.real.welding_box))


def drawing_features(mask_n: np.ndarray, cfg_w: AppConfig):
    """Topology + geometry + persistence features of a normalized drawing mask."""
    bundle = extract_features(mask_n, cfg_w, with_persistence=True)
    return feature_dict(bundle, with_ph=True), bundle


# ====================================================================== reference model
@dataclass
class ReferenceModel:
    """Class statistics fitted on a reference subset ONLY."""

    clf: BayesianShapeClassifier
    classes: List[str]
    n_reference: int
    floors: Dict[str, float]
    zero_variance_features: List[str]


def fit_reference(rows: Sequence[Dict[str, float]], labels: Sequence[str], topo: Sequence[str],
                  geo: Sequence[str], weights: Tuple[float, float], cfg: AppConfig) -> ReferenceModel:
    """Per-class mean + shrunk std (toward pooled within-class variance) from reference rows only."""
    r = cfg.real
    feats = list(topo) + list(geo)
    X = np.array([[row[f] for f in feats] for row in rows], dtype=np.float64)
    y = np.array(labels)
    classes = sorted(set(labels))
    means, varis, counts = {}, {}, {}
    for c in classes:
        Xc = X[y == c]
        means[c], varis[c], counts[c] = Xc.mean(axis=0), Xc.var(axis=0), len(Xc)
    pooled = np.mean([varis[c] for c in classes], axis=0)
    ref_std = X.std(axis=0)
    floors = {f: (r.sigma_floor_relative * s if s > 0 else r.sigma_floor_absolute) for f, s in zip(feats, ref_std)}
    zero_var = [f for f, s in zip(feats, ref_std) if s == 0]
    lam = r.variance_shrinkage_lambda
    std = {c: np.sqrt((counts[c] * varis[c] + lam * pooled) / (counts[c] + lam)) for c in classes}
    model = PrototypeModel(classes=classes, feature_names=feats,
                           mean={c: dict(zip(feats, map(float, means[c]))) for c in classes},
                           std={c: dict(zip(feats, map(float, std[c]))) for c in classes},
                           n_samples=counts, fonts_used=[], metadata={"source": "REAL_REFERENCE_SUBSET"})
    bcfg = replace(cfg.bayesian, topology_features=tuple(topo), geometry_features=tuple(geo),
                   topology_weight=weights[0], geometry_weight=weights[1], aggregation="mean",
                   likelihood_temperature=1.0, sigma_floor=floors, sigma_floor_default=r.sigma_floor_absolute,
                   variance_epsilon=1e-12, priors={}, top_n=5)
    return ReferenceModel(BayesianShapeClassifier(model, bcfg), classes, len(rows), floors, zero_var)


def distances(x: Dict[str, float], ref: ReferenceModel, topo: Sequence[str],
              geo: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    """Per class (topology, geometry, combined) mean |z| distance using the reference sigmas."""
    out = {}
    clf = ref.clf
    for c in ref.classes:
        zt = [abs(x[f] - clf.model.mean[c][f]) / clf.sigma(c, f) for f in topo]
        zg = [abs(x[f] - clf.model.mean[c][f]) / clf.sigma(c, f) for f in geo]
        dt, dg = float(np.mean(zt)), float(np.mean(zg))
        out[c] = (dt, dg, 0.5 * dt + 0.5 * dg)
    return out


# ====================================================================== welding CV
@dataclass
class WeldingSample:
    """One welding image with its ROI and features."""

    record: ImageRecord
    sample_index: int
    roi_info: Dict[str, object]
    mask_n: np.ndarray = field(repr=False)
    features: Dict[str, float] = field(default_factory=dict)
    bundle: object = field(default=None, repr=False)


def prepare_welding(records: List[ImageRecord], cfg: AppConfig) -> List[WeldingSample]:
    """ROI + normalized mask + features for every readable welding image."""
    cfg_w = welding_cfg(cfg)
    out = []
    for r in records:
        if r.dataset_type != WELDING or r.status == "FAILED":
            continue
        try:
            img = load_image(r.path).image
            mask, info = welding_roi(img, cfg)
            mask_n = normalize_drawing(mask, cfg.real.welding_canvas, cfg.real.welding_box)
            feats, bundle = drawing_features(mask_n, cfg_w)
            out.append(WeldingSample(r, int(r.variant), info, mask_n, feats, bundle))
            r.used_for.append("welding_roi_features")
        except Exception as exc:
            r.status, r.failure_reason = "FAILED", f"welding feature extraction: {type(exc).__name__}: {exc}"
    return out


def _pred_row(prefix: str, res: BayesianResult, gt: str) -> Dict[str, object]:
    ranked = list(res.posteriors.items())
    return {f"{prefix}_top1": res.top1, f"{prefix}_top1_posterior": res.top1_posterior,
            f"{prefix}_top3": "|".join(c for c, _ in ranked[:3]),
            f"{prefix}_entropy": res.entropy, f"{prefix}_correct": res.top1 == gt,
            f"{prefix}_top3_correct": gt in [c for c, _ in ranked[:3]]}


def welding_cross_validation(samples: List[WeldingSample], cfg: AppConfig) -> Dict[str, object]:
    """3-fold leave-one-sample-per-class CV; reference statistics from the 2 other samples only."""
    cfg_w = welding_cfg(cfg)
    by_class = defaultdict(dict)
    for s in samples:
        by_class[s.record.label][s.sample_index] = s
    folds = sorted({s.sample_index for s in samples})
    complete = {c: v for c, v in by_class.items() if set(v) == set(folds)}
    incomplete = sorted(set(by_class) - set(complete))
    predictions: List[Dict[str, object]] = []
    tested: Counter = Counter()
    fold_info = []
    oof_distance: Dict[str, float] = {}
    for k in folds:
        test = [v[k] for v in complete.values()]
        ref = [s for v in complete.values() for i, s in v.items() if i != k]
        test_ids = {id(s) for s in test}
        assert not test_ids & {id(s) for s in ref}, "LEAKAGE: test image inside reference subset"
        assert all(s.sample_index != k for s in ref), "LEAKAGE: fold index in reference"
        ref_rows, ref_labels = [s.features for s in ref], [s.record.label for s in ref]
        models = {name: fit_reference(ref_rows, ref_labels, TOPO_FEATURES, GEO_FEATURES, w, cfg)
                  for name, w in MODELS.items()}
        fold_info.append({"fold": k, "test_images": len(test), "reference_images": len(ref),
                          "zero_variance_features": models["C_topology_geometry"].zero_variance_features})
        for s in test:
            gt = s.record.label
            tested[rel(s.record.path)] += 1
            row: Dict[str, object] = {"image_path": rel(s.record.path), "fold": k, "ground_truth": gt,
                                      "posterior_type": POSTERIOR_LABEL}
            results = {name: m.clf.predict(s.features) for name, m in models.items()}
            for name, res in results.items():
                row.update(_pred_row(name, res, gt))
            # MODEL D: preliminary (model C, no ground truth) -> Top-K -> restoration -> re-inference
            mc = models["C_topology_geometry"]
            prelim = results["C_topology_geometry"]
            rest = restore(s.mask_n, s.mask_n.astype(np.float32), mc.clf, cfg_w, s.bundle, preliminary=prelim)
            r_feats, _ = drawing_features(rest.best.mask, cfg_w)
            after = mc.clf.predict(r_feats)
            row.update(_pred_row(MODEL_D, after, gt))
            b = rest.best
            row.update({"restoration_operation": b.operation, "J": b.J, "L_data": b.L_data,
                        "L_topology": b.L_topology, "L_geometry": b.L_geometry, "L_change": b.L_change,
                        "restoration_reference_class": b.reference_class,
                        "restoration_effect": ("corrected" if (not prelim.top1 == gt and after.top1 == gt) else
                                               "degraded" if (prelim.top1 == gt and after.top1 != gt) else
                                               "unchanged")})
            d = distances(s.features, mc, TOPO_FEATURES, GEO_FEATURES)
            nearest = min(d, key=lambda c: d[c][2])
            oof_distance[rel(s.record.path)] = d[nearest][2]
            row.update({"oof_nearest_class_by_distance": nearest, "oof_combined_distance": d[nearest][2],
                        "oof_distance_to_true_class": d[gt][2]})
            predictions.append(row)
            s.record.used_for.append(f"welding_cv_test_fold{k}")
        for s in ref:
            s.record.used_for.append(f"welding_reference_fold{k}")
    # every complete-class image must be tested exactly once
    expected = {rel(s.record.path) for v in complete.values() for s in v.values()}
    assert set(tested) == expected and all(n == 1 for n in tested.values()), \
        "CV integrity failed: an image was not tested exactly once"
    for r in samples:
        if r.record.label in incomplete:
            r.record.used_for.append("welding_not_in_cv(incomplete_class)")
    return {"predictions": predictions, "folds": fold_info, "incomplete_classes": incomplete,
            "oof_distance": oof_distance, "classes": sorted(complete)}


def welding_metrics(predictions: List[Dict[str, object]], classes: List[str]) -> Dict[str, object]:
    """Top-1 / Top-3 / macro / per-class accuracy and mean entropy for every model."""
    out: Dict[str, object] = {}
    for name in list(MODELS) + [MODEL_D]:
        per_class = {}
        for c in classes:
            rows = [p for p in predictions if p["ground_truth"] == c]
            per_class[c] = float(np.mean([p[f"{name}_correct"] for p in rows])) if rows else float("nan")
        out[name] = {
            "n_test": len(predictions),
            "top1_accuracy": float(np.mean([p[f"{name}_correct"] for p in predictions])),
            "top3_accuracy": float(np.mean([p[f"{name}_top3_correct"] for p in predictions])),
            "macro_accuracy": float(np.nanmean(list(per_class.values()))),
            "mean_entropy": float(np.mean([p[f"{name}_entropy"] for p in predictions])),
            "per_class_accuracy": per_class,
        }
    eff = Counter(p["restoration_effect"] for p in predictions)
    out["restoration_effect_counts"] = {k: eff.get(k, 0) for k in ("corrected", "degraded", "unchanged")}
    out["restoration_operations"] = dict(Counter(p["restoration_operation"] for p in predictions))
    a = out["A_geometry_only"]["top1_accuracy"]
    c = out["C_topology_geometry"]["top1_accuracy"]
    d = out[MODEL_D]["top1_accuracy"]
    out["topology_contribution"] = c - a
    out["restoration_contribution"] = d - c
    return out


def confusion(predictions: List[Dict[str, object]], classes: List[str], model: str) -> np.ndarray:
    """Rows = ground truth, columns = prediction."""
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), int)
    for p in predictions:
        m[idx[p["ground_truth"]], idx[p[f"{model}_top1"]]] += 1
    return m


def most_confused(m: np.ndarray, classes: List[str], top: int = 5) -> List[Dict[str, object]]:
    """Largest off-diagonal confusions (symmetric pair counts)."""
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            n = int(m[i, j] + m[j, i])
            if n:
                pairs.append({"pair": f"{classes[i]} <-> {classes[j]}", "count": n,
                              f"{classes[i]}->{classes[j]}": int(m[i, j]), f"{classes[j]}->{classes[i]}": int(m[j, i])})
    return sorted(pairs, key=lambda p: -p["count"])[:top]


# ====================================================================== text images
def text_analysis(records: List[ImageRecord], cfg: AppConfig,
                  probe: ReferenceModel, ood_threshold: float) -> Dict[str, object]:
    """Segmentation + topology + geometry for every Formal / Informal crop + OOD probe per segment."""
    r = cfg.real
    cfg_w = welding_cfg(cfg)
    audit = run_dataset_audit(cfg, write=True, previews=True, verbose=False)
    manifest = {m["image_path"]: m for m in audit.manifest}
    image_rows, seg_rows, ood_rows = [], [], []
    for rec in records:
        if rec.dataset_type not in (FORMAL, INFORMAL) or rec.status == "FAILED":
            continue
        key = rel(rec.path)
        seg: Optional[Segmentation] = audit.segmentations.get(key)
        m = manifest.get(key, {})
        try:
            if seg is None:
                raise RuntimeError("segmentation unavailable: " + str(m.get("notes", "")))
            transcription = str(m.get("transcription", "") or "")
            norm = str(m.get("normalized_transcription", "") or "")
            heights = [b[3] - b[1] for b in seg.boxes]
            med_h = float(np.median(heights)) if heights else 0.0
            touching = sum(1 for b in seg.boxes if med_h and (b[2] - b[0]) > r.touching_width_ratio * med_h)
            small = sum(1 for b in seg.boxes if med_h and (b[3] - b[1]) < r.small_height_ratio * med_h)
            if not transcription.strip():
                valid, reason = "UNVERIFIED", "NO_TRANSCRIPTION (ground_truth=UNKNOWN)"
            elif seg.count == len(norm):
                valid, reason = True, ""
            else:
                valid, reason = False, f"COUNT_MISMATCH segments={seg.count} chars={len(norm)}"
            if seg.count == 0:
                valid, reason = False, "NO_SEGMENTS"
            img_feats = {}
            if seg.mask.any():
                b = extract_features(char_canvas(seg.mask, cfg), cfg)
                img_feats = feature_dict(b, with_ph=False)
            image_rows.append({
                "image_path": key, "style": rec.dataset_type, "group_id": rec.group_id, "variant": rec.variant,
                "ground_truth": transcription if transcription.strip() else "UNKNOWN",
                "normalized_transcription": norm, "segment_count": seg.count,
                "normalized_transcription_length": len(norm) if transcription.strip() else "",
                "segmentation_valid": valid, "failure_reason": reason,
                "possible_touching_segments": touching, "small_segments_punctuation_like": small,
                "excluded_line_components": len(seg.excluded_lines),
                "recognition_accuracy": "NOT_EVALUABLE" if not transcription.strip() else "SEE_CHARACTER_CV",
                **{f"img_{k}": v for k, v in img_feats.items() if not isinstance(v, str)},
            })
            for idx, (box, cm) in enumerate(zip(seg.boxes, seg.char_masks)):
                label = norm[idx] if (valid is True and idx < len(norm)) else "UNKNOWN"
                cb = extract_features(char_canvas(cm, cfg), cfg)
                cf = feature_dict(cb, with_ph=False)
                seg_rows.append({"image_path": key, "style": rec.dataset_type, "group_id": rec.group_id,
                                 "segment_index": idx, "ground_truth": label,
                                 "box_x0": box[0], "box_y0": box[1], "box_x1": box[2], "box_y1": box[3],
                                 **{k: v for k, v in cf.items() if not isinstance(v, str)}})
                # OUT-OF-DISTRIBUTION PROBE in the welding-trained feature space
                wf, _ = drawing_features(normalize_drawing(cm, cfg.real.welding_canvas, cfg.real.welding_box), cfg_w)
                res = probe.clf.predict(wf)
                d = distances(wf, probe, TOPO_FEATURES, GEO_FEATURES)
                nearest = min(d, key=lambda c: d[c][2])
                ood_rows.append({
                    "image_path": key, "style": rec.dataset_type, "segment_index": idx,
                    "probe_type": "OUT-OF-DISTRIBUTION PROBE (welding-trained model; not character recognition)",
                    "nearest_welding_class": nearest, "max_posterior_class": res.top1,
                    "max_welding_posterior": res.top1_posterior, "entropy": res.entropy,
                    "topology_distance": d[nearest][0], "geometry_distance": d[nearest][1],
                    "combined_distance": d[nearest][2], "ood_threshold": ood_threshold,
                    "classification": "UNKNOWN_OOD" if d[nearest][2] > ood_threshold else "IN_DISTRIBUTION_LIKE",
                })
            rec.used_for += ["segmentation", "topology_geometry", "formal_vs_informal", "welding_ood_probe"]
            if transcription.strip():
                rec.used_for.append("transcription_validation")
        except Exception as exc:
            rec.status, rec.failure_reason = "FAILED", f"text analysis: {type(exc).__name__}: {exc}"
            image_rows.append({"image_path": key, "style": rec.dataset_type, "group_id": rec.group_id,
                               "variant": rec.variant, "segmentation_valid": False,
                               "failure_reason": rec.failure_reason})
    char_cv = character_cv(seg_rows, cfg)
    return {"images": image_rows, "segments": seg_rows, "ood": ood_rows, "character_cv": char_cv}


def character_cv(seg_rows: List[Dict[str, object]], cfg: AppConfig) -> Dict[str, object]:
    """Leave-one-group-out character recognition on USER-TRANSCRIBED segments only."""
    labeled = [s for s in seg_rows if s["ground_truth"] != "UNKNOWN"]
    if not labeled:
        return {"status": "NOT_EVALUABLE", "reason": "no validated user transcriptions (ground truth absent)"}
    groups = sorted({s["group_id"] for s in labeled})
    correct = total = unseen = 0
    by_style: Dict[str, List[bool]] = defaultdict(list)
    for g in groups:
        test = [s for s in labeled if s["group_id"] == g]
        ref = [s for s in labeled if s["group_id"] != g]
        counts = Counter(s["ground_truth"] for s in ref)
        ref = [s for s in ref if counts[s["ground_truth"]] >= 2]
        if len({s["ground_truth"] for s in ref}) < 2:
            unseen += len(test)
            total += len(test)
            continue
        model = fit_reference(ref, [s["ground_truth"] for s in ref], CHAR_TOPO_FEATURES, GEO_FEATURES,
                              (cfg.real.w_topology, cfg.real.w_geometry), cfg)
        for s in test:
            total += 1
            if s["ground_truth"] not in model.classes:
                unseen += 1
                by_style[s["style"]].append(False)
                continue
            ok = model.clf.predict(s).top1 == s["ground_truth"]
            correct += ok
            by_style[s["style"]].append(ok)
    return {"status": "EVALUATED", "protocol": "leave-one-group-out, transcribed segments only",
            "test_characters": total, "character_accuracy": correct / total if total else float("nan"),
            "class_unseen_in_reference_counted_wrong": unseen,
            "accuracy_by_style": {k: float(np.mean(v)) for k, v in by_style.items()}}


# ====================================================================== scenes
def scene_candidates(image: np.ndarray, cfg: AppConfig) -> Tuple[List[Dict[str, object]], np.ndarray]:
    """Qualitative marking-candidate regions in a full scene (OpenCV only; no ground truth)."""
    r = cfg.real
    gray = to_grayscale(image)
    lo, hi = np.percentile(gray, [5, 95])
    if hi - lo < cfg.preprocessing.clahe_low_contrast_range:
        gray = cv2.createCLAHE(cfg.preprocessing.clahe_clip_limit, (8, 8)).apply(gray)
    den = cv2.medianBlur(gray, 3)
    bg = cv2.medianBlur(den, r.scene_background_kernel).astype(np.float32)
    res = den.astype(np.float32) - bg
    pl, pd = np.percentile(res, 99.7), np.percentile(-res, 99.7)
    resp = res if pl > pd else -res
    mask = (resp > r.scene_threshold_ratio * max(pl, pd)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        fill = a / float(w * h)
        longest = max(w, h)
        line_like = (longest / max(1, min(w, h)) >= r.scene_line_aspect) or (longest > 60 and fill < 0.15)
        keep[i] = a >= 4 and not line_like
    text = keep[lab]
    merged = cv2.dilate(text.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, r.scene_merge_kernel))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    cands = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area_ratio = (w * h) / float(H * W)
        region = text[y:y + h, x:x + w]
        if not (r.scene_min_area_ratio <= area_ratio <= r.scene_max_area_ratio) or region.sum() < 15:
            continue
        b = extract_features(char_canvas(region, cfg), cfg)
        f = feature_dict(b, with_ph=False)
        cands.append({"x": x, "y": y, "w": w, "h": h, "foreground_density": float(region.mean()),
                      **{k: v for k, v in f.items() if k in CHAR_TOPO_FEATURES + GEO_FEATURES}})
    cands.sort(key=lambda c: (c["y"], c["x"]))
    return cands, text


def analyze_scenes(records: List[ImageRecord], cfg: AppConfig, out_dir: Path) -> Dict[str, object]:
    """Run candidate extraction on every scene and save an overlay per scene."""
    rows, overlays = [], {}
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    for rec in records:
        if rec.dataset_type != INFORMAL_SCENE or rec.status == "FAILED":
            continue
        try:
            img = load_image(rec.path).image
            cands, text = scene_candidates(img, cfg)
            vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for k, c in enumerate(cands):
                cv2.rectangle(vis, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (0, 220, 255), 2)
                cv2.putText(vis, str(k + 1), (c["x"], max(12, c["y"] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 220, 255), 1)
            out = out_dir / "scenes" / f"{rec.path.stem}_candidates.png"
            ok, enc = cv2.imencode(".png", vis)
            enc.tofile(str(out))
            overlays[rel(rec.path)] = vis
            for k, c in enumerate(cands):
                rows.append({"image_path": rel(rec.path), "candidate_index": k, **c,
                             "detection_accuracy": "NOT_EVALUABLE (no scene ground truth)"})
            rec.used_for += ["scene_candidate_extraction", "qualitative_visualization"]
            rec.label = f"{len(cands)} candidates"
        except Exception as exc:
            rec.status, rec.failure_reason = "FAILED", f"scene analysis: {type(exc).__name__}: {exc}"
    return {"candidates": rows, "overlays": overlays}


# ====================================================================== orchestration
@dataclass
class ExperimentResult:
    """All in-memory results needed for figures and reports."""

    records: List[ImageRecord]
    audit: Dict[str, object]
    welding: List[WeldingSample]
    cv: Dict[str, object]
    metrics: Dict[str, object]
    classes: List[str]
    ood_threshold: float
    text: Dict[str, object]
    scenes: Dict[str, object]
    report: Dict[str, object] = field(default_factory=dict)
    out_dir: Path = REAL_RESULTS_DIR
    timing_s: Dict[str, float] = field(default_factory=dict)


def run_welding_only(cfg: AppConfig, symbols_dir: Path = SYMBOLS_DIR, out_dir: Path = REAL_RESULTS_DIR,
                     say=print) -> Tuple[List[ImageRecord], List[WeldingSample], Dict, Dict]:
    """Inventory + welding ROI/features + CV + metrics."""
    records = discover(symbols_dir, cfg)
    say(f"[REAL] discovered {len(records)} images under {rel(symbols_dir)}")
    samples = prepare_welding(records, cfg)
    say(f"[REAL] welding features: {len(samples)} images")
    cv = welding_cross_validation(samples, cfg)
    metrics = welding_metrics(cv["predictions"], cv["classes"])
    return records, samples, cv, metrics


def run_real_experiment(cfg: AppConfig, symbols_dir: Path = SYMBOLS_DIR, out_dir: Path = REAL_RESULTS_DIR,
                        welding_only: bool = False, say=print) -> ExperimentResult:
    """Full REAL SYMBOLS experiment; writes CSV/JSON/TXT/PNG into results/real_symbols/."""
    if not symbols_dir.exists():
        raise FileNotFoundError(f"Real data folder not found: {symbols_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    timing: Dict[str, float] = {}
    t0 = time.perf_counter()
    records, samples, cv, metrics = run_welding_only(cfg, symbols_dir, out_dir, say)
    timing["welding_cv"] = time.perf_counter() - t0
    audit = audit_summary(records, symbols_dir)
    (out_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "dataset_inventory.csv", [{"image_path": rel(r.path), "dataset_type": r.dataset_type,
                                                   "group_id": r.group_id, "variant": r.variant, "label": r.label,
                                                   "width": r.width, "height": r.height,
                                                   "readable": not r.failure_reason.startswith("unreadable")}
                                                  for r in records])
    # OOD threshold from OUT-OF-FOLD welding distances (fixed before any text image is probed)
    oof = list(cv["oof_distance"].values())
    ood_threshold = float(np.percentile(oof, cfg.real.ood_percentile)) if oof else float("inf")
    probe = fit_reference([s.features for s in samples if s.record.label in cv["classes"]],
                          [s.record.label for s in samples if s.record.label in cv["classes"]],
                          TOPO_FEATURES, GEO_FEATURES, (cfg.real.w_topology, cfg.real.w_geometry), cfg)
    for s in samples:
        s.record.used_for.append("ood_probe_reference")
    text = {"images": [], "segments": [], "ood": [], "character_cv": {"status": "SKIPPED (--welding-evaluate)"}}
    scenes = {"candidates": [], "overlays": {}}
    if not welding_only:
        t1 = time.perf_counter()
        say("[REAL] formal / informal segmentation + features + OOD probe (all crops)")
        text = text_analysis(records, cfg, probe, ood_threshold)
        timing["text_analysis"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        say("[REAL] scene candidate extraction (all scenes)")
        scenes = analyze_scenes(records, cfg, out_dir)
        timing["scene_analysis"] = time.perf_counter() - t2
    for r in records:
        if r.status == "PENDING":
            if r.used_for:
                r.status = "SUCCESS"
            elif welding_only and r.dataset_type != WELDING:
                r.status, r.failure_reason = "NOT_RUN", "--welding-evaluate runs welding images only"
            else:
                r.status, r.failure_reason = "FAILED", "no analysis path produced a result"
    result = ExperimentResult(records, audit, samples, cv, metrics, cv["classes"], ood_threshold, text, scenes,
                              out_dir=out_dir, timing_s=timing)
    write_outputs(result, cfg)
    return result


def write_outputs(r: ExperimentResult, cfg: AppConfig) -> None:
    """CSV + JSON outputs and the all-images integrity check."""
    out = r.out_dir
    write_csv(out / "welding_predictions.csv", r.cv["predictions"])
    write_csv(out / "welding_features.csv", [{"image_path": rel(s.record.path), "class": s.record.label,
                                              "sample_index": s.sample_index, **s.roi_info,
                                              **{k: v for k, v in s.features.items()}} for s in r.welding])
    images = r.text["images"]
    segs = r.text["segments"]
    for style, name in ((FORMAL, "formal_features.csv"), (INFORMAL, "informal_features.csv")):
        write_csv(out / name, [s for s in segs if s["style"] == style])
        write_csv(out / name.replace("_features", "_images"), [i for i in images if i["style"] == style])
    write_csv(out / "ood_probe.csv", r.text["ood"])
    write_csv(out / "scene_candidates.csv", r.scenes["candidates"])
    status_rows = [rec.row() for rec in r.records]
    write_csv(out / "all_images_status.csv", status_rows)
    discovered = len(r.records)
    success = sum(1 for x in r.records if x.status == "SUCCESS")
    failed = sum(1 for x in r.records if x.status == "FAILED")
    not_run = sum(1 for x in r.records if x.status == "NOT_RUN")
    assert discovered == success + failed + not_run == len(status_rows), "image accounting mismatch"
    r.report = build_report(r, cfg, success, failed, not_run)
    (out / "final_report.json").write_text(json.dumps(r.report, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")


def _dist_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """min / median / p95 / max of a distance list."""
    if not values:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None}
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def build_report(r: ExperimentResult, cfg: AppConfig, success: int, failed: int, not_run: int) -> Dict[str, object]:
    """final_report.json content (only computed values)."""
    m = r.metrics
    ood = r.text["ood"]
    def rate(style: str) -> Optional[float]:
        rows = [o for o in ood if o["style"] == style]
        return (sum(o["classification"] == "UNKNOWN_OOD" for o in rows) / len(rows)) if rows else None
    weld_ok = [s for s in r.welding]
    caption_ok = all(s.roi_info["caption_removed_verified"] for s in weld_ok) and bool(weld_ok)
    images = r.text["images"]
    transcribed = Counter(i["style"] for i in images if i.get("ground_truth", "UNKNOWN") != "UNKNOWN")
    cm = confusion(r.cv["predictions"], r.classes, "C_topology_geometry") if r.cv["predictions"] else None
    limitations = [
        "Formal/Informal transcriptions missing for {} of {} crops -> character recognition accuracy {}".format(
            sum(1 for i in images if i.get("ground_truth", "UNKNOWN") == "UNKNOWN"), len(images),
            r.text["character_cv"].get("status")),
        "Bayesian posteriors are UNCALIBRATED MODEL POSTERIORS (not real-world probabilities)",
        f"Small welding dataset: {len(r.welding)} images, {len(r.classes)} classes, 3 per class; "
        "per-fold reference has only 2 samples per class (variance shrinkage applied)",
        "Welding samples _02/_03 are rotated/skewed copies of _01 (near-duplicates): CV measures robustness "
        "to small warps of the same drawing, not generalization to new drawings",
        "Possible label leakage: class-name caption " + ("removed by fixed rule and verified empty"
                                                         if caption_ok else "NOT verifiably removed")
        + "; in-drawing dimension text (e.g. 'WPS-0', '6 50-150') is class-correlated drawing content",
        "No scene ground truth: detection accuracy NOT EVALUABLE",
        "No deep-learning detector/recognizer; scene candidates are OpenCV heuristics",
        "Welding-model outputs on text are an OUT-OF-DISTRIBUTION PROBE, not character recognition; "
        "the closed-set posterior is overconfident on OOD inputs (use distance for UNKNOWN)",
    ]
    return {
        "experiment": "REAL SYMBOLS DATA EXPERIMENT",
        "dataset_source": r.audit["dataset_source"],
        "synthetic_data_used": False,
        "total_images": r.audit["total_images"], "formal_images": r.audit["formal_images"],
        "informal_images": r.audit["informal_crop_images"], "scene_images": r.audit["informal_scene_images"],
        "welding_images": r.audit["welding_images"], "welding_classes": r.audit["welding_classes"],
        "posterior_type": POSTERIOR_LABEL,
        "cv_protocol": "3-fold leave-one-sample-per-class; per-fold reference statistics only",
        "fixed_hyperparameters": {
            "w_topology": cfg.real.w_topology, "w_geometry": cfg.real.w_geometry,
            "aggregation": "per-group mean log-likelihood",
            "variance_shrinkage_lambda": cfg.real.variance_shrinkage_lambda,
            "sigma_floor_relative": cfg.real.sigma_floor_relative,
            "sigma_floor_absolute": cfg.real.sigma_floor_absolute, "top_k_restoration": cfg.real.top_k,
            "ood_percentile": cfg.real.ood_percentile, "welding_canvas": cfg.real.welding_canvas,
            "topology_features": TOPO_FEATURES, "geometry_features": GEO_FEATURES,
            "persistence_features": {"ph_robust_h1": "count of H1 classes with lifetime >= "
                                     f"{cfg.topology.persistence_min_lifetime}px (signed-distance filtration)",
                                     "ph_near_holes": "robust H1 classes born at 0 < birth <= "
                                     f"{cfg.topology.persistence_max_gap_birth}px (gap-closed loops)"},
            "note": "fixed before evaluation; not tuned on test accuracy",
        },
        "geometry_accuracy": m["A_geometry_only"]["top1_accuracy"],
        "topology_accuracy": m["B_topology_only"]["top1_accuracy"],
        "combined_accuracy": m["C_topology_geometry"]["top1_accuracy"],
        "restoration_accuracy": m[MODEL_D]["top1_accuracy"],
        "top3_accuracy": m["C_topology_geometry"]["top3_accuracy"],
        "macro_accuracy": m["C_topology_geometry"]["macro_accuracy"],
        "mean_entropy": m["C_topology_geometry"]["mean_entropy"],
        "topology_contribution": m["topology_contribution"],
        "restoration_contribution": m["restoration_contribution"],
        "restoration_effect": m["restoration_effect_counts"],
        "restoration_operations": m["restoration_operations"],
        "model_metrics": {k: v for k, v in m.items() if k in list(MODELS) + [MODEL_D]},
        "most_confused_pairs_model_C": most_confused(cm, r.classes) if cm is not None else [],
        "folds": r.cv["folds"],
        "welding_ood_threshold": r.ood_threshold,
        "ood_threshold_rule": f"{cfg.real.ood_percentile}th percentile of out-of-fold welding nearest-class "
                              "combined distances (fixed before probing text)",
        "formal_ood_rate": rate(FORMAL), "informal_ood_rate": rate(INFORMAL),
        "ood_distance_summary": {
            "welding_out_of_fold": _dist_stats(list(r.cv["oof_distance"].values())),
            **{style: {kind: _dist_stats([o[f"{kind}_distance"] for o in ood if o["style"] == style])
                       for kind in ("combined", "topology", "geometry")} for style in (FORMAL, INFORMAL)},
        },
        "ood_segments": {"FORMAL": sum(1 for o in ood if o["style"] == FORMAL),
                         "INFORMAL": sum(1 for o in ood if o["style"] == INFORMAL)},
        "character_recognition": r.text["character_cv"],
        "transcribed_images": {"FORMAL": transcribed.get(FORMAL, 0), "INFORMAL": transcribed.get(INFORMAL, 0)},
        "scenes_processed": len({c for c in r.scenes["overlays"]}),
        "scene_candidate_regions": len(r.scenes["candidates"]),
        "scene_detection_accuracy": "NOT_EVALUABLE",
        "processed_images": success, "failed_images": failed, "not_run_images": not_run,
        "failed_list": [{"image": rel(x.path), "reason": x.failure_reason} for x in r.records if x.status == "FAILED"],
        "label_leakage_status": {
            "LABEL_LEAKAGE_RISK": not caption_ok,
            "caption_text": "REMOVED_AND_VERIFIED (fixed positional+intensity rule, all images)" if caption_ok
            else "NOT_VERIFIED",
            "NEAR_DUPLICATE_RISK": True,
            "near_duplicate_note": "variants are warped copies of one drawing per class",
            "in_drawing_dimension_text": "present in some classes (drawing content, not caption)",
        },
        "limitations": limitations,
        "timing_seconds": {k: round(v, 2) for k, v in r.timing_s.items()},
    }
