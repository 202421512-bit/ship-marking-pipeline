"""Topology analysis of a binary character mask.

Computes Betti numbers (beta_0 = 8-connected ink components, beta_1 = holes,
i.e. bounded 4-connected background components), the Euler characteristic,
a pruned skeleton with endpoints / branch points and, when available, cubical
persistent homology (GUDHI) on the signed-distance filtration. Persistence
reveals "near holes": loops that are open by a small gap in the observed
image, which is exactly the evidence used by topology-constrained restoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .config import TopologyConfig

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class TopologyResult:
    """Topological description of a character mask."""

    beta_0: int
    beta_1: int
    euler_characteristic: int
    skeleton_length: int
    endpoints: int
    branch_points: int
    skeleton_density: float
    component_areas: List[int]
    hole_areas: List[int]
    component_labels: np.ndarray = field(repr=False)
    hole_labels: np.ndarray = field(repr=False)
    skeleton: np.ndarray = field(repr=False)
    endpoint_coords: List[Tuple[int, int]] = field(default_factory=list)
    branch_coords: List[Tuple[int, int]] = field(default_factory=list)
    persistence: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        """JSON-serializable summary (no arrays)."""
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "euler_characteristic": self.euler_characteristic,
            "skeleton_length": self.skeleton_length,
            "endpoints": self.endpoints,
            "branch_points": self.branch_points,
            "skeleton_density": round(self.skeleton_density, 4),
            "component_areas": self.component_areas,
            "hole_areas": self.hole_areas,
            "persistent_homology": self.persistence,
        }


def connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray, List[int]]:
    """8-connected foreground components: (count, label image, areas)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [int(a) for a in stats[1:, cv2.CC_STAT_AREA]]
    return n - 1, labels, areas


def holes(mask: np.ndarray, min_area: int) -> Tuple[int, np.ndarray, List[int]]:
    """Bounded 4-connected background components (dual connectivity to 8-ink)."""
    padded = np.pad(~mask, 1, constant_values=True).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(padded, connectivity=4)
    outer = labels[0, 0]
    hole_img = np.zeros_like(labels)
    areas: List[int] = []
    for lab in range(1, n):
        if lab == outer:
            continue
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        areas.append(area)
        hole_img[labels == lab] = len(areas)
    return len(areas), hole_img[1:-1, 1:-1], areas


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    """Number of 8-neighbors that are skeleton pixels, for each pixel."""
    counts = cv2.filter2D(skel.astype(np.uint8), -1, _NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT)
    return counts * skel


def prune_spurs(skel: np.ndarray, max_len: int) -> np.ndarray:
    """Remove short skeleton spurs (endpoint -> junction paths shorter than max_len)."""
    if max_len <= 0:
        return skel
    skel = skel.copy()
    h, w = skel.shape
    counts = neighbor_count(skel)
    ends = list(zip(*np.nonzero((counts == 1) & skel)))
    for ey, ex in ends:
        path = [(ey, ex)]
        prev, cur = None, (ey, ex)
        reached_junction = False
        while len(path) <= max_len:
            nbrs = [(cur[0] + dy, cur[1] + dx) for dy, dx in _OFFSETS
                    if 0 <= cur[0] + dy < h and 0 <= cur[1] + dx < w and skel[cur[0] + dy, cur[1] + dx]]
            nxt = [p for p in nbrs if p != prev and p not in path]
            if len(nbrs) >= 3 and cur != (ey, ex):
                reached_junction = True
                path.pop()  # keep the junction pixel itself
                break
            if len(nxt) >= 2:  # cur touches a junction region
                reached_junction = True
                path.pop()
                break
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        if reached_junction and len(path) < max_len:
            for py, px in path:
                skel[py, px] = False
    return skel


def skeleton_keypoints(skel: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Endpoints (1 neighbor) and branch-point clusters (>=3 neighbors)."""
    counts = neighbor_count(skel)
    ends = [(int(y), int(x)) for y, x in zip(*np.nonzero((counts == 1) & skel))]
    branch_mask = (counts >= 3) & skel
    lab, n = ndimage.label(branch_mask, structure=np.ones((3, 3)))
    branches: List[Tuple[int, int]] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        branches.append((int(round(ys.mean())), int(round(xs.mean()))))
    return ends, branches


def persistent_homology(mask: np.ndarray, cfg: TopologyConfig) -> Dict[str, object]:
    """Cubical persistent homology on the signed distance filtration (GUDHI).

    Sublevel sets of f = dist_to_ink - dist_to_background: t<0 is erosion,
    t=0 the observed mask, t>0 dilation. An H1 class born at small t>0 is a
    loop closed by bridging a gap of width ~2t (a "near hole").
    Falls back gracefully if GUDHI is unavailable or fails.
    """
    if not cfg.use_persistent_homology:
        return {"status": "DISABLED"}
    try:
        import gudhi  # noqa: WPS433 - optional heavy dependency

        inside = ndimage.distance_transform_edt(mask)
        outside = ndimage.distance_transform_edt(~mask)
        filtration = (outside - inside).astype(np.float64)
        padded = np.pad(filtration, 1, constant_values=float(filtration.max()) + 1.0)
        complex_ = gudhi.CubicalComplex(top_dimensional_cells=padded)
        pairs = complex_.persistence()
        h0, h1 = [], []
        for dim, (birth, death) in pairs:
            death_v = float(death) if np.isfinite(death) else None
            item = {"birth": round(float(birth), 3), "death": None if death_v is None else round(death_v, 3)}
            item["lifetime"] = None if death_v is None else round(death_v - float(birth), 3)
            (h0 if dim == 0 else h1 if dim == 1 else []).append(item)
        robust_h1 = [p for p in h1 if p["lifetime"] is not None
                     and p["lifetime"] >= cfg.persistence_min_lifetime]
        near_holes = [p for p in robust_h1 if 0.0 < p["birth"] <= cfg.persistence_max_gap_birth]
        robust_h0 = [p for p in h0 if p["lifetime"] is None or p["lifetime"] >= cfg.persistence_min_lifetime]
        return {
            "status": "OK",
            "backend": f"gudhi {getattr(gudhi, '__version__', '')}".strip(),
            "filtration": "signed distance (negative inside ink)",
            "H0": h0,
            "H1": h1,
            "robust_H0_count": len(robust_h0),
            "robust_H1_count": len(robust_h1),
            "near_hole_count": len(near_holes),
            "near_holes": near_holes,
        }
    except Exception as exc:  # GUDHI missing / failure -> fallback, never crash
        return {"status": "FALLBACK", "reason": f"{type(exc).__name__}: {exc}",
                "fallback": "connected components + holes + Euler + skeleton"}


def analyze_topology(mask: np.ndarray, cfg: TopologyConfig, with_persistence: bool = True) -> TopologyResult:
    """Compute topology features of a boolean ink mask.

    Args:
        mask: bool array, True = ink.
        cfg: Topology parameters.
        with_persistence: compute GUDHI persistence (skipped for prototype batches).
    """
    mask = mask.astype(bool)
    b0, comp_labels, comp_areas = connected_components(mask)
    b1, hole_labels, hole_areas = holes(mask, cfg.min_hole_area)
    try:
        skel = skeletonize(mask)
        skel = prune_spurs(skel, cfg.spur_prune_length)
    except Exception:
        skel = np.zeros_like(mask)
    ends, branches = skeleton_keypoints(skel)
    fg = int(mask.sum())
    persistence = persistent_homology(mask, cfg) if with_persistence else {"status": "SKIPPED"}
    return TopologyResult(
        beta_0=b0,
        beta_1=b1,
        euler_characteristic=b0 - b1,
        skeleton_length=int(skel.sum()),
        endpoints=len(ends),
        branch_points=len(branches),
        skeleton_density=float(skel.sum()) / fg if fg else 0.0,
        component_areas=comp_areas,
        hole_areas=hole_areas,
        component_labels=comp_labels,
        hole_labels=hole_labels,
        skeleton=skel,
        endpoint_coords=ends,
        branch_coords=branches,
        persistence=persistence,
    )


def topology_feature_dict(topo: TopologyResult, canvas: int) -> Dict[str, float]:
    """Feature values used by the Bayesian model (skeleton length normalized)."""
    return {
        "beta_0": float(topo.beta_0),
        "beta_1": float(topo.beta_1),
        "euler_characteristic": float(topo.euler_characteristic),
        "skeleton_length_norm": float(topo.skeleton_length) / float(canvas),
        "endpoints": float(topo.endpoints),
        "branch_points": float(topo.branch_points),
    }


def optional_int(value: Optional[float]) -> Optional[int]:
    """Round an optional float to int (helper for reports)."""
    return None if value is None else int(round(value))
