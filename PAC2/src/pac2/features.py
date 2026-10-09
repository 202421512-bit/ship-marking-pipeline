"""The 7 shape features (prompt numbering kept: 1, 2, 4, 5, 6, 10, 11) on a binary stroke mask.

Feature formulas follow the project prompt (the PDF '수기문자_정형비정형_특징설계안.pdf' was not available locally).
Connected components (CC) are used as character proxies: one CC is NOT always one character (touching characters
merge, broken strokes split), so CV_h, CV_g, S_theta and B are approximations.
Impossible values are never disguised as 0: they are NaN with a missing reason.
These features describe formal vs informal SHAPE; they do not by themselves separate handwriting from rust/scratches.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

FEATURES = ["CV_w", "CV_h", "CV_g", "S_theta", "B", "CV_A", "R"]
NUMBER = {"CV_w": 1, "CV_h": 2, "CV_g": 4, "S_theta": 5, "B": 6, "CV_A": 10, "R": 11}
NAN = float("nan")


def _cv(x: np.ndarray, eps: float) -> Tuple[float, str]:
    x = np.asarray(x, float)
    m = float(x.mean())
    if abs(m) <= eps:
        return NAN, "mean ~ 0 (CV undefined)"
    return float(x.std() / m), ""


def components(mask: np.ndarray, min_area: int = 1) -> List[dict]:
    m = (mask > 0).astype(np.uint8)
    n, lab, st, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h, a = (int(v) for v in st[i])
        out.append({"label": i, "x": x, "y": y, "w": w, "h": h, "area": a, "cx": float(cent[i][0]), "cy": float(cent[i][1]),
                    "pix": np.column_stack(np.nonzero(lab == i)[::-1])})   # (x, y)
    return out


def f_cv_w(mask, cfg) -> Tuple[float, str]:
    """1. stroke width variation: w = 2 * DT on the skeleton, CV_w = std(w) / mean(w)."""
    m = mask > 0
    sk = skeletonize(m)
    if sk.sum() < cfg["min_skeleton_px"]:
        return NAN, f"skeleton < {cfg['min_skeleton_px']} px"
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    return _cv(2.0 * dt[sk], cfg["eps"])


def f_cv_h(comps, cfg) -> Tuple[float, str]:
    """2. height variation of CC bounding boxes (character proxy)."""
    if len(comps) < cfg["min_components"]:
        return NAN, f"< {cfg['min_components']} components"
    return _cv(np.array([c["h"] for c in comps]), cfg["eps"])


def f_cv_g(comps, cfg) -> Tuple[float, str]:
    """4. gap variation: g'_i = gap_i / ((w_i + w_{i+1}) / 2) between x-adjacent CCs; needs >= 2 gaps."""
    cs = sorted(comps, key=lambda c: c["x"])
    if len(cs) - 1 < cfg["min_gaps"]:
        return NAN, f"< {cfg['min_gaps']} gaps ({max(0, len(cs) - 1)} available)"
    g = np.array([(b["x"] - (a["x"] + a["w"])) / ((a["w"] + b["w"]) / 2.0) for a, b in zip(cs, cs[1:])])
    return _cv(g, cfg["eps"])


def _axis(c) -> Tuple[float, float]:
    p = c["pix"].astype(float)
    p -= p.mean(0)
    cov = np.cov(p.T) if len(p) > 2 else np.eye(2)
    ev, evec = np.linalg.eigh(cov)
    elong = math.sqrt(ev[1] / max(ev[0], 1e-9))
    vx, vy = evec[:, 1]
    return math.degrees(math.atan2(-vy, vx)) % 180.0, elong        # image y points down -> -vy


def f_s_theta(comps, cfg) -> Tuple[float, str]:
    """5. slant variation: circular std of CC principal-axis angles (180-deg periodic, doubled-angle wrapping).
    The ROI's common rotation is removed by measuring deviations from the circular mean."""
    ang = [a for a, e in (_axis(c) for c in comps) if e >= cfg["orientation_min_elongation"]]
    if len(ang) < cfg["min_components"]:
        return NAN, f"< {cfg['min_components']} elongated components"
    t = np.radians(np.array(ang) * 2.0)
    mean = math.atan2(np.sin(t).mean(), np.cos(t).mean())
    dev = (np.degrees(np.angle(np.exp(1j * (t - mean)))) / 2.0)     # wrapped to (-90, 90]
    return float(dev.std()), ""


def f_baseline(comps, cfg) -> Tuple[float, str]:
    """6. baseline wobble: bottom points (cx, y+h) fitted by y = a x + b; B = RMSE / mean(h). Needs >= 3 points."""
    if len(comps) < cfg["min_baseline_points"]:
        return NAN, f"< {cfg['min_baseline_points']} components (2 points always fit exactly)"
    x = np.array([c["cx"] for c in comps])
    y = np.array([c["y"] + c["h"] for c in comps], float)
    if np.ptp(x) <= cfg["eps"]:
        return NAN, "components vertically stacked (x range 0)"
    a, b = np.polyfit(x, y, 1)
    rmse = float(np.sqrt(np.mean((y - (a * x + b)) ** 2)))
    mh = float(np.mean([c["h"] for c in comps]))
    return (rmse / mh, "") if mh > cfg["eps"] else (NAN, "mean height ~ 0")


def f_cv_a(comps, cfg) -> Tuple[float, str]:
    """10. CC pixel-area variation."""
    if len(comps) < cfg["min_components"]:
        return NAN, f"< {cfg['min_components']} components"
    return _cv(np.array([c["area"] for c in comps]), cfg["eps"])


def f_r(mask, cfg) -> Tuple[float, str]:
    """11. contour irregularity R = (P - P_s) / P_s, P_s = perimeter after approxPolyDP(eps = frac x P)."""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= cfg["contour_min_area_px"]]
    if not cnts:
        return NAN, f"no contour with area >= {cfg['contour_min_area_px']} px"
    P = sum(cv2.arcLength(c, True) for c in cnts)
    Ps = sum(cv2.arcLength(cv2.approxPolyDP(c, cfg["contour_smoothing_eps_frac"] * cv2.arcLength(c, True), True), True) for c in cnts)
    return ((P - Ps) / Ps, "") if Ps > cfg["eps"] else (NAN, "smoothed perimeter ~ 0")


def extract(mask: np.ndarray, cfg: dict) -> Dict[str, object]:
    """7 features + missing indicators + reasons + CC count for one binary mask."""
    comps = components(mask, 1)
    vals = {"CV_w": f_cv_w(mask, cfg), "CV_h": f_cv_h(comps, cfg), "CV_g": f_cv_g(comps, cfg),
            "S_theta": f_s_theta(comps, cfg), "B": f_baseline(comps, cfg), "CV_A": f_cv_a(comps, cfg), "R": f_r(mask, cfg)}
    out: Dict[str, object] = {"n_components": len(comps), "ink_fraction": float((mask > 0).mean())}
    for k in FEATURES:
        v, why = vals[k]
        out[k] = v
        out[f"{k}_missing"] = int(not np.isfinite(v))
        out[f"{k}_reason"] = why if not np.isfinite(v) else ""
    return out
