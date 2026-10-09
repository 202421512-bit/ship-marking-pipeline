"""Phase 4 recovery: dual-polarity, stroke-scale candidates + structure suppression + stroke-evidence gating.

Root causes found in the previous generator (extraction.stroke_candidates), see reports/phase4_recovery/trace.json:
  * ONE polarity per image (ink_polarity). example1 (bright chalk) was judged 'dark' -> black-hat extracted the darker
    plate surface between the chalk strokes and the dark seam ("background leakage").
  * kernels up to 41 px: in field 1 / 2 86-93 % of kept pixels came from the 41 px kernel (large bright structures),
    which inflated the robust threshold (120.8 in field 2) so faint chalk (small-kernel response) was cut at the
    candidate stage.
New generator (no learning; every parameter in RECOVERY_CFG; parameters were set while looking at the 7 field images,
which are therefore development images, not an independent test):
  1. bright / dark channels separately: contrast to a local MEDIAN background AND a stroke-scale top-/black-hat
     (kernels <= 15 px) must both exceed robust thresholds (median + k * 1.4826 * MAD of that channel).
  2. low-contrast channel: hysteresis on contrast normalised by local noise (seed z >= z_high, grow z >= z_low).
  3. structure suppression (pixel level, recorded): long straight segments (Hough, >= seg_len_frac of the image side)
     that touch the border AND are thicker than 1.5 x the image's median stroke width, or dark segments spanning
     >= dark_span_frac of the image; afterwards components whose bounding box covers >= large_bbox_frac of the
     image and spans >= large_extent_frac of its side (machinery, plate outlines).
  4. components lying mostly (>= 50 %) on edges of LARGE objects are removed as structure: all candidate strokes are
     inpainted away first (so handwriting, even dense clusters, becomes background) and Canny is run on that image -
     only step edges of machinery, holes, plate edges, rust patches remain. (A 15 px median filter was tried first and
     deleted dense chalk clusters such as example1 "F55"; replaced.)
  5. component gating: specks (short and not elongated), blobs (not thin), no two-sided ridge evidence.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
from skimage.filters import sato

RECOVERY_CFG = {"bg_window": 31, "hat_kernels": (7, 11, 15), "k_contrast": 3.5, "k_hat": 3.5, "z_high": 4.0, "z_low": 3.0,
                "noise_window": 41, "min_area_px": 10, "ridge_sigmas": (1.0, 1.5, 2.0, 2.5), "ridge_k": 3.0,
                "ridge_support_min": 0.35, "seg_len_frac": 0.25, "seg_width_factor": 1.5, "dark_span_frac": 0.6, "border_px": 4,
                "large_bbox_frac": 0.20, "large_extent_frac": 0.40, "blob_thinness_max": 0.35, "blob_min_area_frac": 0.002,
                "speck_len_px": 12, "speck_elong_min": 2.5, "struct_median": 15, "struct_edge_dist": 2, "struct_overlap_max": 0.5}


def _thr(v: np.ndarray, k: float) -> float:
    med = float(np.median(v))
    return med + k * 1.4826 * float(np.median(np.abs(v - med)))


def _hat(gray: np.ndarray, op, kernels) -> np.ndarray:
    r = np.zeros(gray.shape, np.float32)
    for k in kernels:
        r = np.maximum(r, cv2.morphologyEx(gray, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32))
    return r


def _clean(m: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def channels(bgr: np.ndarray, cfg: dict = RECOVERY_CFG) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    bg = cv2.medianBlur(gray, cfg["bg_window"]).astype(np.float32)
    out = {"gray": gray, "background": bg}
    for pol, sign, op in (("bright", 1.0, cv2.MORPH_TOPHAT), ("dark", -1.0, cv2.MORPH_BLACKHAT)):
        c = sign * (g - bg)
        hat = _hat(gray, op, cfg["hat_kernels"])
        m = (c > max(2.0, _thr(c, cfg["k_contrast"]))) & (hat > max(2.0, _thr(hat, cfg["k_hat"])))
        out[f"{pol}_contrast"], out[f"{pol}_hat"], out[pol] = c, hat, _clean(m, cfg["min_area_px"])
        mad = cv2.medianBlur(np.clip(np.abs(c - np.median(c)), 0, 255).astype(np.uint8), cfg["noise_window"]).astype(np.float32)
        z = np.minimum(c, hat) / (1.4826 * np.maximum(mad, 1.0))
        low, high = z >= cfg["z_low"], z >= cfg["z_high"]
        n, lab = cv2.connectedComponents(low.astype(np.uint8), connectivity=8)
        hit = np.zeros(n, bool)
        hit[np.unique(lab[high & low])] = True
        hit[0] = False
        out[f"{pol}_lowcontrast"] = _clean(hit[lab], cfg["min_area_px"]) & ~out[pol]
        r = sato(g / 255.0, sigmas=cfg["ridge_sigmas"], black_ridges=(pol == "dark"))
        out[f"{pol}_ridge"] = r > _thr(r, cfg["ridge_k"])
    return out


def _median_stroke_width(mask: np.ndarray) -> float:
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    if not sk.any():
        return 2.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(max(2.0, 2.0 * np.median(dt[sk])))


def structural_segments(mask: np.ndarray, pol: str, w_med: float, cfg: dict = RECOVERY_CFG):
    """Pixel band of long straight segments judged structural; returns (band mask, segment records)."""
    H, W = mask.shape
    L = int(cfg["seg_len_frac"] * max(H, W))
    segs = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, np.pi / 180, threshold=max(20, L // 3), minLineLength=L, maxLineGap=3)
    band = np.zeros((H, W), bool)
    rec = []
    if segs is None:
        return band, rec
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    b = cfg["border_px"]
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        xs, ys = np.linspace(x1, x2, n).astype(int), np.linspace(y1, y2, n).astype(int)
        width = 2.0 * float(np.median(dt[ys, xs])) + 1.0
        length = float(np.hypot(x2 - x1, y2 - y1))
        border = min(x1, x2) <= b or min(y1, y2) <= b or max(x1, x2) >= W - 1 - b or max(y1, y2) >= H - 1 - b
        thick = width >= cfg["seg_width_factor"] * w_med
        dark_span = pol == "dark" and length >= cfg["dark_span_frac"] * max(H, W)
        structural = (border and thick) or dark_span
        rec.append({"polarity": pol, "segment": [int(x1), int(y1), int(x2), int(y2)], "length": round(length, 1), "width": round(width, 1),
                    "touches_border": bool(border), "thick": bool(thick), "dark_span": bool(dark_span), "structural": bool(structural)})
        if structural:
            tmp = np.zeros((H, W), np.uint8)
            cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), 1, max(3, int(round(width + 2))))
            band |= tmp.astype(bool)
    return band & mask, rec


def structure_edges(gray: np.ndarray, cand: np.ndarray, cfg: dict = RECOVERY_CFG) -> np.ndarray:
    """Edges of LARGE objects only: every candidate stroke pixel (both polarities, dilated) is inpainted away first, so
    handwriting - including dense clusters - becomes background, while step edges of machinery / holes / plate edges /
    rust patches remain; Canny with Otsu-derived thresholds on the inpainted image (median 5), dilated."""
    m = cv2.dilate(cand.astype(np.uint8), np.ones((5, 5), np.uint8))
    clean = cv2.medianBlur(cv2.inpaint(gray, m, 5, cv2.INPAINT_TELEA), 5)
    t, _ = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    e = cv2.Canny(clean, 0.25 * t, 0.5 * t)
    d = 2 * cfg["struct_edge_dist"] + 1
    return cv2.dilate(e, np.ones((d, d), np.uint8)) > 0

def component_table(mask: np.ndarray, ridge: np.ndarray, cfg: dict = RECOVERY_CFG, sedge=None):
    H, W = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    ridge_d = cv2.dilate(ridge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    rows = []
    for i in range(1, n):
        sel = lab == i
        w_, h_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        length = float(max(w_, h_))
        width = 2.0 * float(np.median(dt[sel]))
        elong = length / max(1.0, width)
        ridge_sup = float(ridge_d[sel].mean())
        large = (w_ * h_ >= cfg["large_bbox_frac"] * H * W) and (length >= cfg["large_extent_frac"] * max(H, W))
        blob = width / max(1.0, length) > cfg["blob_thinness_max"] and st[i, cv2.CC_STAT_AREA] >= cfg["blob_min_area_frac"] * H * W
        speck = length < cfg["speck_len_px"] or elong < cfg["speck_elong_min"]
        s_ov = float(sedge[sel].mean()) if sedge is not None else 0.0
        on_struct = s_ov >= cfg["struct_overlap_max"]
        reason = ("structure: large extent" if large else "structure: on large-object edges" if on_struct else "blob" if blob else "speck (short / not elongated)" if speck
                  else "no two-sided ridge evidence" if ridge_sup < cfg["ridge_support_min"] else "")
        x0, y0 = st[i, 0], st[i, 1]
        rows.append({"label": i, "area": int(st[i, cv2.CC_STAT_AREA]), "bbox": [int(x0), int(y0), int(x0 + w_), int(y0 + h_)],
                     "length": length, "median_width": round(width, 2), "elongation": round(elong, 2), "ridge_support": round(ridge_sup, 3), "structure_edge_overlap": round(s_ov, 3),
                     "removed": reason})
    return lab, rows


def recover(bgr: np.ndarray, cfg: dict = RECOVERY_CFG) -> Dict[str, object]:
    ch = channels(bgr, cfg)
    shape = ch["gray"].shape
    removed = {k: np.zeros(shape, bool) for k in ("structure_segment", "structure_large", "structure_edge", "blob", "speck", "no_ridge")}
    kept = np.zeros(shape, bool)
    comp_rows, seg_rows = [], []
    res = {"channels": ch}
    sedge = structure_edges(ch["gray"], ch["bright"] | ch["bright_lowcontrast"] | ch["dark"] | ch["dark_lowcontrast"], cfg)
    for pol in ("bright", "dark"):
        cand = ch[pol] | ch[f"{pol}_lowcontrast"]
        res[f"{pol}_candidate"] = cand
        w_med = _median_stroke_width(cand)
        band, segs = structural_segments(cand, pol, w_med, cfg)
        seg_rows += segs
        removed["structure_segment"] |= band
        rest = _clean(cand & ~band, cfg["min_area_px"])
        lab, rows = component_table(rest, ch[f"{pol}_ridge"], cfg, sedge)
        for r in rows:
            sel = lab == r["label"]
            r["polarity"] = pol
            r["source"] = "contrast" if (sel & ch[pol]).any() else "lowcontrast"
            key = {"structure: large extent": "structure_large", "structure: on large-object edges": "structure_edge", "blob": "blob", "speck (short / not elongated)": "speck",
                   "no two-sided ridge evidence": "no_ridge"}.get(r["removed"])
            if key:
                removed[key] |= sel
            else:
                kept |= sel
        comp_rows += rows
    res.update(combined=kept, removed=removed, structure_edges=sedge, components=comp_rows, segments=seg_rows)
    return res
