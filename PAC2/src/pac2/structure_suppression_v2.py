"""structure_suppression_v2 - ONE post-processing proposal on the frozen stable final mask (fixed before evaluation).

Inputs: stable final mask, stable structure_edges (large-object edges computed from the image), image size. No GT.
1. Long straight segments: HoughLinesP on the final mask (min length = 15 % of the longer image side).
2. A segment is STRUCTURAL if at least one cue holds:
     a. parallel partner: another long segment with angle difference <= 5 deg at 3-25 px perpendicular distance
        (repeated parallel borders of machinery / plate edges / seams)
     b. lies on large-object edges: >= 50 % of its pixels on structure_edges
     c. touches the image border (within 4 px)
3. Protection (handwriting-like zones): components of the final mask that do NOT touch the border, are not dominated by
   long segments (< 50 % of their skeleton on long segments) and have >= 3 skeleton endpoints+branches (character-like);
   dilated 6 px. A structural segment whose band intersects a protected zone is NOT removed -> recorded as UNCERTAIN.
4. Removal = final-mask pixels inside a band (segment width + 2 px) around structural, unprotected segments; leftover
   fragments < 8 px of the touched components are removed too. Output = subset of the stable final mask.
"""
from __future__ import annotations

import math
from typing import Dict

import cv2
import numpy as np
from skimage.morphology import skeletonize

CFG = {"min_len_frac": 0.15, "par_angle": 5.0, "par_dmin": 3, "par_dmax": 25, "edge_overlap": 0.5, "border_px": 4,
       "protect_line_frac": 0.5, "protect_min_keypoints": 3, "protect_dilate": 6, "min_fragment": 8}


def _segments(mask, cfg):
    L = int(cfg["min_len_frac"] * max(mask.shape))
    s = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, np.pi / 180, threshold=max(15, L // 3), minLineLength=L, maxLineGap=3)
    return [] if s is None else [tuple(map(int, v)) for v in s.reshape(-1, 4)]


def _band(shape, seg, width):
    b = np.zeros(shape, np.uint8)
    cv2.line(b, seg[:2], seg[2:], 1, max(1, int(width)))
    return b.astype(bool)


def _keypoints(skel):
    k = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32)) - skel.astype(np.uint8)
    return int(((k == 1) & skel).sum() + ((k >= 3) & skel).sum())


def suppress(final: np.ndarray, sedge: np.ndarray, cfg: dict = CFG) -> Dict[str, object]:
    H, W = final.shape
    segs = _segments(final, cfg)
    dt = cv2.distanceTransform(final.astype(np.uint8), cv2.DIST_L2, 5)
    info = []
    for s in segs:
        x1, y1, x2, y2 = s
        n = max(abs(x2 - x1), abs(y2 - y1)) + 1
        xs, ys = np.linspace(x1, x2, n).astype(int), np.linspace(y1, y2, n).astype(int)
        info.append({"seg": s, "ang": math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180, "width": 2 * float(np.median(dt[ys, xs])) + 1,
                     "edge": float(sedge[ys, xs].mean()), "mid": ((x1 + x2) / 2, (y1 + y2) / 2),
                     "border": min(x1, x2) <= cfg["border_px"] or min(y1, y2) <= cfg["border_px"] or max(x1, x2) >= W - 1 - cfg["border_px"] or max(y1, y2) >= H - 1 - cfg["border_px"]})
    for i, a in enumerate(info):
        a["parallel"] = False
        th = math.radians(a["ang"])
        nrm = np.array([-math.sin(th), math.cos(th)])
        for j, b in enumerate(info):
            if i == j:
                continue
            da = abs(a["ang"] - b["ang"]); da = min(da, 180 - da)
            d = abs(float((np.array(b["mid"]) - np.array(a["mid"])) @ nrm))
            if da <= cfg["par_angle"] and cfg["par_dmin"] <= d <= cfg["par_dmax"]:
                a["parallel"] = True
                break
        a["structural"] = a["parallel"] or a["edge"] >= cfg["edge_overlap"] or a["border"]
    # protection zones
    long_band = np.zeros_like(final)
    for a in info:
        long_band |= _band(final.shape, a["seg"], a["width"] + 2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(final.astype(np.uint8), connectivity=8)
    skel = skeletonize(final)
    protect = np.zeros_like(final)
    b = cfg["border_px"]
    for i in range(1, n):
        sel = lab == i
        x, y, w, h, _ = st[i]
        if x <= b or y <= b or x + w >= W - b or y + h >= H - b:
            continue
        sk = skel & sel
        if sk.sum() == 0:
            continue
        if (sk & long_band).sum() / sk.sum() < cfg["protect_line_frac"] and _keypoints(sk) >= cfg["protect_min_keypoints"]:
            protect |= sel
    pz = cv2.dilate(protect.astype(np.uint8), np.ones((2 * cfg["protect_dilate"] + 1,) * 2, np.uint8)).astype(bool)
    remove = np.zeros_like(final); uncertain = np.zeros_like(final)
    for a in info:
        if not a["structural"]:
            a["status"] = "kept (no structural cue)"
            continue
        band = _band(final.shape, a["seg"], a["width"] + 2) & final
        if (band & pz).any():
            uncertain |= band
            a["status"] = "UNCERTAIN (touches handwriting-like zone)"
        else:
            remove |= band
            a["status"] = "removed"
    out = final & ~remove
    # leftover fragments of touched components
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
    touched = np.unique(lab[remove])
    for i in range(1, n2):
        sel = lab2 == i
        if st2[i, cv2.CC_STAT_AREA] < cfg["min_fragment"] and np.isin(lab[sel], touched).all():
            out[sel] = False
    return {"final": out, "removed": final & ~out, "uncertain": uncertain & out, "protect": protect, "segments": info}
