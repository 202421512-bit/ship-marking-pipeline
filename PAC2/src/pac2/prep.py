"""Fold-internal imputation and standardization (fit on the training fold only)."""
from __future__ import annotations

import warnings

import numpy as np


class FoldPrep:
    """Median imputation + z-standardization.

    mode='train_all'  : mu, sigma from all training-fold samples.
    mode='formal_ref' : mu, sigma from the training-fold FORMAL samples only (PDF proposal). With ~12 formal samples sigma
                        can be ~0 (font renders are nearly constant), so sigma is floored at max(sigma_formal,
                        floor_frac x sigma_train_all, eps) and every floored feature is recorded as a warning.
    """

    def __init__(self, mode: str = "train_all", floor_frac: float = 0.1, eps: float = 1e-6):
        self.mode, self.floor_frac, self.eps = mode, floor_frac, eps
        self.warnings: list = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        Xi = self._impute(X)
        sd_all = Xi.std(axis=0)
        if self.mode == "formal_ref":
            ref = Xi[y == 0]
            if len(ref) < 2:
                raise ValueError("formal_ref needs >= 2 formal training samples")
            self.mu, sd = ref.mean(axis=0), ref.std(axis=0)
            floor = np.maximum(self.floor_frac * sd_all, self.eps)
            low = sd < floor
            for j in np.nonzero(low)[0]:
                self.warnings.append(f"feature {j}: formal sigma {sd[j]:.4g} < floor {floor[j]:.4g} -> floored")
            self.sigma = np.where(low, floor, sd)
        else:
            self.mu, self.sigma = Xi.mean(axis=0), np.maximum(sd_all, self.eps)
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(X), X, self.median)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self._impute(X) - self.mu) / self.sigma
