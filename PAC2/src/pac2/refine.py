"""Stage 2 of Phase 4 (refined): stroke-only pixel refinement INSIDE accepted candidate groups.

A group accepted as handwriting does not mean all its candidate pixels are strokes: the stroke-scale candidate generator
(top-hat up to 41 px) also responds to bright halos between clustered characters and near holes / edges. The refinement
keeps only candidate pixels that behave like strokes. Every method returns a SUBSET of the candidate pixels (no new
pixels, no hole filling, no bridging).

Methods (compared on field images with label-free proxies; no ground truth exists):
  M1 'small_tophat' : candidate ∩ (top/black-hat with a small kernel, 3 x stroke width + 1 > threshold)
  M2 'ridge'        : candidate ∩ dilate(Sato ridge response at sigmas 1-2.5 > robust threshold, 1 px)
  M3 'open_remove'  : candidate minus dilate(opening(candidate, disk r = 1.5 x stroke width)) - removes broad regions
  M4 'component_otsu+open' : per candidate component, keep pixels whose small-scale stroke response exceeds that
                      component's own Otsu threshold (separates strokes from the halo merged with them), then M3
  (M4 was added after M1-M3 left the inter-character halo of example1 in place; disclosed)
Stroke width w = median of 2 x distance transform on the skeleton of the ridge mask (the ridge mask has stroke-scale
structure even where the candidate is filled).
  M5 'hysteresis'   : seeds = M4 cores, grown inside the permissive M1 mask (components touching a seed), broad removed
Selection: the rule fixed beforehand (lowest broad-pixel fraction with thin-skeleton retention >= 0.80, see select())
did NOT capture the thin halo fans between characters and chose a different method per image; the adopted method
FINAL_METHOD = M5 was chosen by visual inspection of all 7 field images (no ground truth). Both are reported.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
from skimage.filters import sato
from skimage.morphology import skeletonize

FINAL_METHOD = "M5_hysteresis"
REFINE_CFG = {"sigmas": (1.0, 1.5, 2.0, 2.5), "ridge_k": 4.0, "min_area_px": 8, "broad_factor": 2.5, "open_factor": 1.5,
              "retention_min": 0.80, "otsu_min_px": 60}


def _robust_thr(r: np.ndarray, k: float) -> float:
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    return med + k * 1.4826 * mad


def _clean(m: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def ridge_mask(gray: np.ndarray, polarity: str, cfg: dict = REFINE_CFG) -> np.ndarray:
    r = sato(gray.astype(float) / 255.0, sigmas=cfg["sigmas"], black_ridges=(polarity == "dark"))
    return r > _robust_thr(r, cfg["ridge_k"])


def stroke_width(mask: np.ndarray) -> float:
    sk = skeletonize(mask)
    if not sk.any():
        return 2.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(max(2.0, 2.0 * np.median(dt[sk])))


def refine_methods(gray: np.ndarray, cand: np.ndarray, polarity: str, cfg: dict = REFINE_CFG) -> Dict[str, np.ndarray]:
    cand = cand > 0
    rid = ridge_mask(gray, polarity, cfg)
    w = stroke_width(rid)
    k = int(round(3 * w + 1)) | 1
    op = cv2.MORPH_BLACKHAT if polarity == "dark" else cv2.MORPH_TOPHAT
    small = cv2.morphologyEx(gray, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(float)
    m1 = cand & (small > _robust_thr(small, cfg["ridge_k"]))
    m2 = cand & cv2.dilate(rid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    r = max(1, int(round(cfg["open_factor"] * w)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    broad = cv2.dilate(cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, disk), np.ones((3, 3), np.uint8)).astype(bool)
    m3 = cand & ~broad
    # M4: per-component Otsu on the small-scale stroke response (strokes are brighter / darker than the halo that the
    # broad candidate generator merged with them), then M3's broad-region removal
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(cand)
    s8 = np.clip(small, 0, 255).astype(np.uint8)
    for i in range(1, n):
        sel = lab == i
        v = s8[sel]
        if v.size < cfg["otsu_min_px"] or v.max() == v.min():
            keep |= sel
            continue
        t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        keep |= sel & (s8 > t)
    m4 = keep & ~broad
    # M5: hysteresis - seeds = M4 cores, grown inside the permissive stroke mask M1 (connected components of M1 that
    # touch a seed), broad regions removed. Recovers strokes M4 cut while halo not connected through stroke-like pixels stays out.
    low = m1 & ~broad
    nl, labl = cv2.connectedComponents(low.astype(np.uint8), connectivity=8)
    hit = np.zeros(nl, bool)
    hit[np.unique(labl[m4 & low])] = True
    hit[0] = False
    m5 = hit[labl] | m4
    out = {"M1_small_tophat": m1, "M2_ridge": m2, "M3_open_remove": m3, "M4_component_otsu+open": m4, "M5_hysteresis": m5}
    out = {k_: _clean(v, cfg["min_area_px"]) for k_, v in out.items()}
    out["_w"] = w
    return out


def proxies(cand: np.ndarray, refined: np.ndarray, w: float, cfg: dict = REFINE_CFG) -> Dict[str, float]:
    """Label-free descriptors (NOT accuracy).
    broad_fraction   : share of kept pixels whose local width 2*DT exceeds broad_factor x w (filled / blob pixels)
    thin_retention   : skeleton length of the THIN part of the candidate (2*DT <= broad_factor x w) that is still covered
                       by the refined mask (dilated 1 px) / that skeleton length - a readability guard"""
    cand = cand > 0
    dt_c = cv2.distanceTransform(cand.astype(np.uint8), cv2.DIST_L2, 5)
    thin = cand & (2 * dt_c <= cfg["broad_factor"] * w)
    sk = skeletonize(thin)
    cov = cv2.dilate(refined.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    dt_r = cv2.distanceTransform(refined.astype(np.uint8), cv2.DIST_L2, 5)
    return {"kept_px": int(refined.sum()), "candidate_px": int(cand.sum()),
            "broad_fraction": float((2 * dt_r[refined] > cfg["broad_factor"] * w).mean()) if refined.any() else 0.0,
            "candidate_broad_fraction": float((2 * dt_c[cand] > cfg["broad_factor"] * w).mean()) if cand.any() else 0.0,
            "thin_retention": float((sk & cov).sum() / max(1, sk.sum())), "stroke_width_px": round(w, 2)}


def select(prox: Dict[str, dict], cfg: dict = REFINE_CFG) -> str:
    ok = [m for m, p in prox.items() if p["thin_retention"] >= cfg["retention_min"]]
    if not ok:
        return "M2_ridge"
    return min(ok, key=lambda m: (round(prox[m]["broad_fraction"], 3), m != "M2_ridge"))
