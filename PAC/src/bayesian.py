"""Bayesian shape inference over the prototype statistical reference model.

    P(c|x) ∝ P(x|c) P(c)
    log P(x|c) = -0.5 * Σ_j [ (x_j - μ_cj)^2 / σ_cj^2 + log σ_cj^2 ]

Topology and geometry contributions are computed separately and reported.
The posterior is relative to the prototype reference model; it is NOT a
calibrated probability of being correct in industrial conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import logsumexp

from .config import BayesianConfig
from .prototypes import PrototypeModel


@dataclass
class BayesianResult:
    """Posterior distribution and diagnostics."""

    posteriors: Dict[str, float]                  # sorted descending
    top1: str
    top1_posterior: float
    top3: List[Tuple[str, float]]
    top5: List[Tuple[str, float]]
    entropy: float                                # nats
    normalized_entropy: float                     # entropy / log(K)
    margin: float                                 # p1 - p2
    contributions: Dict[str, Dict[str, float]]    # class -> topology/geometry log-lik
    feature_z: Dict[str, Dict[str, float]] = field(default_factory=dict)  # top classes only

    def summary(self) -> Dict[str, object]:
        """JSON-serializable summary."""
        return {
            "note": "posterior w.r.t. prototype statistical reference model; not calibrated",
            "top1": self.top1,
            "top1_posterior": round(self.top1_posterior, 6),
            "top3": [{"class": c, "posterior": round(p, 6)} for c, p in self.top3],
            "top5": [{"class": c, "posterior": round(p, 6)} for c, p in self.top5],
            "entropy_nats": round(self.entropy, 6),
            "normalized_entropy": round(self.normalized_entropy, 6),
            "margin_top1_top2": round(self.margin, 6),
            "posteriors": {c: round(p, 6) for c, p in self.posteriors.items()},
            "feature_contributions_loglik": {
                c: {k: round(v, 4) for k, v in d.items()} for c, d in self.contributions.items()
            },
            "feature_z_scores": {
                c: {k: round(v, 3) for k, v in d.items()} for c, d in self.feature_z.items()
            },
        }


class BayesianShapeClassifier:
    """Gaussian (naive) Bayes over topology + geometry prototype statistics."""

    def __init__(self, model: PrototypeModel, cfg: BayesianConfig) -> None:
        """Store the reference model and configuration."""
        self.model = model
        self.cfg = cfg
        self.classes = list(model.classes)
        self.topology_features = [f for f in cfg.topology_features if f in model.feature_names]
        self.geometry_features = [f for f in cfg.geometry_features if f in model.feature_names]

    def sigma(self, label: str, feature: str) -> float:
        """Standard deviation with floor + epsilon (avoids degenerate variance)."""
        floor = self.cfg.sigma_floor.get(feature, self.cfg.sigma_floor_default)
        return max(float(self.model.std[label][feature]), floor, math.sqrt(self.cfg.variance_epsilon))

    def z_scores(self, x: Dict[str, float], label: str, features: List[str]) -> Dict[str, float]:
        """Standardized deviation (x - μ)/σ of each feature for one class."""
        return {f: (x[f] - self.model.mean[label][f]) / self.sigma(label, f) for f in features}

    def _group_loglik(self, x: Dict[str, float], label: str, features: List[str]) -> float:
        """Gaussian log-likelihood of a feature group for one class."""
        total = 0.0
        for f in features:
            s = self.sigma(label, f)
            d = x[f] - self.model.mean[label][f]
            total += (d * d) / (s * s) + math.log(s * s)
        return -0.5 * total

    def log_priors(self) -> np.ndarray:
        """Log prior per class (uniform unless configured)."""
        priors = np.array([self.cfg.priors.get(c, 1.0) if self.cfg.priors else 1.0
                           for c in self.classes], dtype=np.float64)
        priors = np.clip(priors, 1e-12, None)
        return np.log(priors / priors.sum())

    def predict(self, x: Dict[str, float]) -> BayesianResult:
        """Compute the posterior over classes for a feature dict."""
        contributions: Dict[str, Dict[str, float]] = {}
        loglik = np.zeros(len(self.classes))
        for i, c in enumerate(self.classes):
            lt = self._group_loglik(x, c, self.topology_features)
            lg = self._group_loglik(x, c, self.geometry_features)
            if self.cfg.aggregation == "mean":
                lt /= max(1, len(self.topology_features))
                lg /= max(1, len(self.geometry_features))
            total = (self.cfg.topology_weight * lt + self.cfg.geometry_weight * lg) / self.cfg.likelihood_temperature
            contributions[c] = {"topology": lt, "geometry": lg, "weighted_total": total}
            loglik[i] = total
        log_post = loglik + self.log_priors()
        log_post -= logsumexp(log_post)
        post = np.exp(log_post)
        order = np.argsort(-post)
        posteriors = {self.classes[i]: float(post[i]) for i in order}
        ranked = list(posteriors.items())
        nz = post[post > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        k = len(self.classes)
        top_n = ranked[: self.cfg.top_n]
        return BayesianResult(
            posteriors=posteriors,
            top1=ranked[0][0],
            top1_posterior=ranked[0][1],
            top3=ranked[:3],
            top5=top_n,
            entropy=entropy,
            normalized_entropy=entropy / math.log(k) if k > 1 else 0.0,
            margin=ranked[0][1] - (ranked[1][1] if k > 1 else 0.0),
            contributions={c: contributions[c] for c, _ in top_n},
            feature_z={c: self.z_scores(x, c, self.topology_features + self.geometry_features)
                       for c, _ in ranked[:3]},
        )
