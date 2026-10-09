"""Domain-matched mask normalization (experiment C).

Starts from the binary candidate mask (colour, saturation and polarity are already gone), crops the ink bounding box
and rescales it UNIFORMLY (aspect ratio kept) so the ink height equals a common target height, then re-binarizes.
This equalizes resolution, edge sharpness / anti-aliasing and pixel quantization between 43 px font renders and
photographs without changing letter shapes (no shear, no non-uniform scaling).
"""
from __future__ import annotations

import cv2
import numpy as np


def normalize_mask(mask: np.ndarray, target_h: int, pad: int = 4) -> np.ndarray:
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return np.zeros((target_h + 2 * pad, target_h + 2 * pad), np.uint8)
    crop = (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(np.float32)
    s = target_h / crop.shape[0]
    w = max(1, int(round(crop.shape[1] * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(crop, (w, target_h), interpolation=interp)
    out = ((r >= 0.5) * 255).astype(np.uint8)
    return cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
