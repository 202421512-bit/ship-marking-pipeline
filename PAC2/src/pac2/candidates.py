"""Stroke-candidate generation. Output masks are CANDIDATES (uint8 0/255, same size as the input), not ground truth.

All methods are polarity- and colour-agnostic: ink may be darker or brighter than the surface and of any colour.
Polarity rule (used by gray methods): the ink is the minority whose intensity is far from the image median.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def read_bgr(path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return img


def _odd(n: float, lo: int = 3) -> int:
    n = max(lo, int(round(n)))
    return n if n % 2 else n + 1


def ink_polarity(gray: np.ndarray) -> str:
    """'dark' if the 10 % most deviating pixels are darker than the median, else 'bright'."""
    med = float(np.median(gray))
    dev = gray.astype(np.float32) - med
    k = max(1, int(0.10 * dev.size))
    idx = np.argpartition(np.abs(dev).ravel(), -k)[-k:]
    return "dark" if dev.ravel()[idx].mean() < 0 else "bright"


def _post(mask: np.ndarray, cfg: dict) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    k = int(cfg["open_kernel"])
    if k > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    min_area = max(int(cfg["min_cc_area_px"]), int(cfg["min_cc_area_frac"] * m.size))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[lab] * 255).astype(np.uint8)


def gray_otsu(bgr, cfg) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flag = cv2.THRESH_BINARY_INV if ink_polarity(g) == "dark" else cv2.THRESH_BINARY
    _, m = cv2.threshold(g, 0, 255, flag + cv2.THRESH_OTSU)
    return m


def clahe_adaptive(bgr, cfg) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    c = cv2.createCLAHE(cfg["clahe_clip"], (cfg["clahe_tile"], cfg["clahe_tile"])).apply(g)
    block = _odd(cfg["adaptive_block_frac"] * min(g.shape), 11)
    flag = cv2.THRESH_BINARY_INV if ink_polarity(g) == "dark" else cv2.THRESH_BINARY
    cval = cfg["adaptive_c"] if flag == cv2.THRESH_BINARY_INV else -cfg["adaptive_c"]
    return cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, block, cval)


def lab_bg_distance(bgr, cfg) -> np.ndarray:
    """Colour distance (CIE Lab) from a median-blurred local background; Otsu on the distance map."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = _odd(cfg["bg_median_frac"] * min(bgr.shape[:2]), 3)
    k = min(k, 255)
    bg = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB), k).astype(np.float32)
    d = np.sqrt(((lab - bg) ** 2).sum(axis=2))
    d8 = np.clip(d / max(d.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def morph_hat(bgr, cfg) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg["hat_kernel_frac"] * min(g.shape)),) * 2)
    op = cv2.MORPH_BLACKHAT if ink_polarity(g) == "dark" else cv2.MORPH_TOPHAT
    r = cv2.morphologyEx(g, op, k)
    _, m = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


METHODS = {"gray_otsu": gray_otsu, "clahe_adaptive": clahe_adaptive, "lab_bg_distance": lab_bg_distance, "morph_hat": morph_hat}


def candidates(bgr: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """All configured candidate masks after the shared clean-up (opening + speck removal)."""
    out = {}
    for name in cfg["methods"]:
        m = _post(METHODS[name](bgr, cfg), cfg)
        assert m.shape == bgr.shape[:2]
        out[name] = m
    return out


def mask_stats(mask: np.ndarray) -> Tuple[float, int]:
    n, _, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return float((mask > 0).mean()), n - 1
