"""Non-shape IMAGING-DOMAIN features (control experiment B only; never a deployment model)."""
from __future__ import annotations

import cv2
import numpy as np

from .candidates import ink_polarity

DOMAIN_FEATURES = ["saturation_mean", "saturation_std", "log_sharpness", "height", "width", "aspect",
                   "gray_mean", "gray_std", "polarity_dark"]


def domain_features(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return {"saturation_mean": float(hsv[..., 1].mean()), "saturation_std": float(hsv[..., 1].std()),
            "log_sharpness": float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())),
            "height": float(bgr.shape[0]), "width": float(bgr.shape[1]), "aspect": float(bgr.shape[1] / bgr.shape[0]),
            "gray_mean": float(gray.mean()), "gray_std": float(gray.std()),
            "polarity_dark": float(ink_polarity(gray) == "dark")}
