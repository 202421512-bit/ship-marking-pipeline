"""Topology-Constrained Character Restoration.

Instead of applying one morphology operation, several restoration candidates
are generated and scored by

    J = λ_data L_data + λ_topology L_topology + λ_geometry L_geometry + λ_change L_change

The topology/geometry targets are NOT hard-coded per label. A preliminary
Bayesian analysis of the observed mask yields Top-K candidate classes; their
prototype statistics serve as references. For each restoration candidate the
reference hypothesis that best explains it structurally is used:

    J(r) = λ_d L_d(r) + λ_c L_c(r) + min_k [ λ_t L_t(r, k) + λ_g L_g(r, k) ]

Bounded transforms d/(1+d) keep every loss term in [0, 1).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .bayesian import BayesianResult, BayesianShapeClassifier
from .config import AppConfig
from .prototypes import FeatureBundle, PrototypeModel, extract_features

GEOMETRY_LOSS_FEATURES: Tuple[str, ...] = (
    "aspect_ratio", "area_ratio", "perimeter_norm", "circularity", "solidity",
    "eccentricity", "hu_1", "hu_2", "hu_3", "hu_4", "skeleton_length_norm",
)


@dataclass
class RestorationCandidate:
    """One restoration candidate and its objective terms."""

    candidate_id: int
    operation: str
    mask: np.ndarray = field(repr=False)
    L_data: float = 0.0
    L_topology: float = 0.0
    L_geometry: float = 0.0
    L_change: float = 0.0
    J: float = float("inf")
    reference_class: str = ""
    beta_0: int = 0
    beta_1: int = 0
    valid: bool = True
    note: str = ""

    def row(self) -> Dict[str, object]:
        """CSV/JSON row."""
        return {
            "candidate_id": self.candidate_id, "operation": self.operation,
            "L_data": round(self.L_data, 6), "L_topology": round(self.L_topology, 6),
            "L_geometry": round(self.L_geometry, 6), "L_change": round(self.L_change, 6),
            "J": round(self.J, 6), "reference_class": self.reference_class,
            "beta_0": self.beta_0, "beta_1": self.beta_1, "valid": self.valid, "note": self.note,
        }


@dataclass
class RestorationResult:
    """Outcome of the restoration search."""

    preliminary: BayesianResult
    reference_classes: List[Tuple[str, float]]
    candidates: List[RestorationCandidate]
    best: RestorationCandidate

    def summary(self) -> Dict[str, object]:
        """JSON-serializable summary."""
        return {
            "method": "Topology-Constrained Character Restoration",
            "objective": "J = l_data*L_data + l_topology*L_topology + l_geometry*L_geometry + l_change*L_change",
            "preliminary_candidates": [{"class": c, "posterior": round(p, 6)} for c, p in self.preliminary.top5],
            "reference_classes_top_k": [{"class": c, "posterior": round(p, 6)} for c, p in self.reference_classes],
            "data_sources": {
                "observed": ["observed_mask (binary, from input image)",
                             "observed_ink_evidence (soft map from denoised input grayscale)"],
                "prototype_reference": ["per-class feature mean/std of rendered font prototypes "
                                        "(Top-K classes chosen by bayesian_before_restoration)"],
                "L_data": "candidate vs observed_ink_evidence (observed only)",
                "L_change": "candidate vs observed_mask (observed only)",
                "L_topology": "candidate features vs prototype_reference",
                "L_geometry": "candidate features vs prototype_reference",
                "uses_preliminary_posterior_values": False,
                "uses_preliminary_top_k_membership": True,
            },
            "normalization": {
                "L_data": "1 - soft Dice, in [0, 1]",
                "L_topology": "d/(1+d), d = weighted |Betti / skeleton count difference|, in [0, 1)",
                "L_geometry": "d/(1+d), d = mean clipped |z| (prototype sigma), in [0, 1)",
                "L_change": "changed_pixels / total_pixels, in [0, 1]",
            },
            "best_operation": self.best.operation,
            "best_reference_class": self.best.reference_class,
            "best": self.best.row(),
            "candidates": [c.row() for c in self.candidates],
        }


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _morph(op: int, size: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda m: cv2.morphologyEx(m.astype(np.uint8), op, _kernel(size)) > 0


def _remove_small(mask: np.ndarray, ratio: float) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= ratio * areas.max()
    return keep[labels]


def candidate_operations(cfg: AppConfig) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Map of operation name -> function(mask) -> mask (limited, fixed set)."""
    ops: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "original": lambda m: m.copy(),
        "closing_3x3": _morph(cv2.MORPH_CLOSE, 3),
        "closing_5x5": _morph(cv2.MORPH_CLOSE, 5),
        "closing_7x7": _morph(cv2.MORPH_CLOSE, 7),
        "opening_3x3": _morph(cv2.MORPH_OPEN, 3),
        "opening_5x5": _morph(cv2.MORPH_OPEN, 5),
        "dilation_3x3": lambda m: cv2.dilate(m.astype(np.uint8), _kernel(3)) > 0,
        "erosion_3x3": lambda m: cv2.erode(m.astype(np.uint8), _kernel(3)) > 0,
        "hole_filling": lambda m: ndimage.binary_fill_holes(m),
        "small_component_removal": lambda m: _remove_small(m, cfg.restoration.small_component_ratio),
    }
    return {name: ops[name] for name in cfg.restoration.operations if name in ops}


def data_loss(candidate: np.ndarray, soft: np.ndarray) -> float:
    """1 - soft Dice between the candidate mask and grayscale ink evidence."""
    c = candidate.astype(np.float32)
    denom = float(c.sum() + soft.sum())
    return 1.0 - 2.0 * float((c * soft).sum()) / denom if denom > 0 else 1.0


def change_loss(candidate: np.ndarray, observed: np.ndarray) -> float:
    """changed_pixels / total_pixels."""
    return float(np.count_nonzero(candidate != observed)) / float(observed.size)


def topology_loss(feats: Dict[str, float], model: PrototypeModel, label: str, cfg: AppConfig) -> float:
    """Bounded Betti-number (+ skeleton) distance to a class prototype."""
    r, mu = cfg.restoration, model.mean[label]
    d = (r.w_beta0 * abs(feats["beta_0"] - mu["beta_0"])
         + r.w_beta1 * abs(feats["beta_1"] - mu["beta_1"])
         + r.w_endpoints * abs(feats["endpoints"] - mu["endpoints"])
         + r.w_branch_points * abs(feats["branch_points"] - mu["branch_points"]))
    return d / (1.0 + d)


def geometry_loss(feats: Dict[str, float], clf: BayesianShapeClassifier, label: str, cfg: AppConfig) -> float:
    """Bounded mean |z| distance of normalized geometry features to a prototype."""
    clip = cfg.restoration.geometry_z_clip
    zs = [min(abs(feats[f] - clf.model.mean[label][f]) / clf.sigma(label, f), clip)
          for f in GEOMETRY_LOSS_FEATURES if f in feats]
    d = float(np.mean(zs)) if zs else 0.0
    return d / (1.0 + d)


def objective(cfg: AppConfig, l_data: float, l_topology: float, l_geometry: float, l_change: float) -> float:
    """J = λ_data L_data + λ_topology L_topology + λ_geometry L_geometry + λ_change L_change."""
    r = cfg.restoration
    return (r.lambda_data * l_data + r.lambda_topology * l_topology
            + r.lambda_geometry * l_geometry + r.lambda_change * l_change)


def restore(observed_mask: np.ndarray, observed_ink_evidence: np.ndarray, clf: BayesianShapeClassifier,
            cfg: AppConfig, observed_bundle: FeatureBundle, preliminary: BayesianResult) -> RestorationResult:
    """Generate, score and select restoration candidates.

    Data sources are kept separate:
      * OBSERVED: ``observed_mask`` and ``observed_ink_evidence`` come only from the input image;
        they define L_data and L_change.
      * PROTOTYPE REFERENCE: ``clf.model`` mean/std of rendered font prototypes; only the Top-K
        class *membership* of ``preliminary`` selects which references are compared (L_topology,
        L_geometry). Preliminary posterior *values* do not enter J.

    Args:
        observed_mask: bool mask after preprocessing (normalized canvas).
        observed_ink_evidence: float ink-evidence map on the same canvas.
        clf: Bayesian classifier holding the prototype reference model.
        cfg: application config.
        observed_bundle: features of the observed mask.
        preliminary: Bayesian result on the observed (un-restored) features.
    """
    r = cfg.restoration
    refs = list(preliminary.posteriors.items())[: r.top_k_reference]
    candidates: List[RestorationCandidate] = []
    for idx, (name, fn) in enumerate(candidate_operations(cfg).items()):
        cand = RestorationCandidate(candidate_id=idx, operation=name, mask=observed_mask)
        try:
            mask = fn(observed_mask)
            if mask.sum() == 0:
                raise ValueError("operation removed all ink")
            feats = observed_bundle.features if name == "original" else extract_features(mask, cfg).features
            cand.mask = mask
            cand.L_data = data_loss(mask, observed_ink_evidence)
            cand.L_change = change_loss(mask, observed_mask)
            # Reference hypothesis = Top-K prototype minimizing the structural part of J.
            best_struct, best_label, best_lt, best_lg = float("inf"), "", 0.0, 0.0
            for label, _ in refs:
                lt = topology_loss(feats, clf.model, label, cfg)
                lg = geometry_loss(feats, clf, label, cfg)
                s = r.lambda_topology * lt + r.lambda_geometry * lg
                if s < best_struct:
                    best_struct, best_label, best_lt, best_lg = s, label, lt, lg
            cand.L_topology, cand.L_geometry, cand.reference_class = best_lt, best_lg, best_label
            cand.J = objective(cfg, cand.L_data, cand.L_topology, cand.L_geometry, cand.L_change)
            cand.beta_0, cand.beta_1 = int(feats["beta_0"]), int(feats["beta_1"])
        except Exception as exc:
            cand.valid, cand.note = False, f"{type(exc).__name__}: {exc}"
        candidates.append(cand)
    valid = [c for c in candidates if c.valid]
    if not valid:
        raise RuntimeError("No valid restoration candidate (all operations failed).")
    best = min(valid, key=lambda c: c.J)
    return RestorationResult(preliminary=preliminary, reference_classes=refs,
                             candidates=candidates, best=best)


def write_candidates_csv(result: RestorationResult, path: Path) -> Path:
    """Write all candidates to CSV (required columns first)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["candidate_id", "operation", "L_data", "L_topology", "L_geometry", "L_change", "J",
               "reference_class", "beta_0", "beta_1", "valid", "note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for cand in result.candidates:
            writer.writerow(cand.row())
    return path
