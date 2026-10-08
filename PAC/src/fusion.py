"""Heuristic evidence fusion of Bayesian posterior and VLM evidence.

    fusion(c) ∝ Bayes(c)^α × VLM_evidence(c)^(1-α)

Bayesian posterior and VLM model-reported confidence are NOT the same kind of
calibrated probability, so this is defined as a heuristic evidence fusion.
Without VLM evidence the final result equals the Bayesian posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .bayesian import BayesianResult
from .config import FusionConfig
from .vlm import VLMResult


@dataclass
class FusionResult:
    """Fused candidate ranking."""

    method: str
    alpha: float
    scores: Dict[str, float]           # sorted descending
    final_candidate: str
    fusion_score: float
    margin: float
    agreement: str                     # AGREEMENT | DISAGREEMENT | NOT_AVAILABLE
    bayesian_top1: str
    vlm_top1: Optional[str]
    vlm_evidence: Dict[str, float]

    def summary(self) -> Dict[str, object]:
        """JSON-serializable summary."""
        ranked = list(self.scores.items())
        return {
            "method": self.method,
            "note": "heuristic evidence fusion; not a calibrated probability",
            "alpha": self.alpha,
            "final_candidate": self.final_candidate,
            "fusion_score": round(self.fusion_score, 6),
            "margin_top1_top2": round(self.margin, 6),
            "agreement": self.agreement,
            "bayesian_top1": self.bayesian_top1,
            "vlm_top1": self.vlm_top1,
            "top5": [{"class": c, "score": round(s, 6)} for c, s in ranked[:5]],
            "vlm_evidence": {c: round(v, 6) for c, v in self.vlm_evidence.items()},
        }


def vlm_evidence(vlm: VLMResult, classes: Sequence[str], floor: float) -> Dict[str, float]:
    """Turn the VLM answer into a normalized evidence vector over classes."""
    ev = {c: floor for c in classes}
    pairs: List[tuple] = [(vlm.top_candidate, vlm.confidence or 0.0)]
    pairs += [(a.get("label"), float(a.get("confidence", 0.0))) for a in vlm.alternatives]
    lookup = {c.upper(): c for c in classes}
    for label, conf in pairs:
        key = lookup.get(str(label).strip().upper()) if label is not None else None
        if key is not None:
            ev[key] = max(ev[key], conf)
    total = sum(ev.values())
    return {c: v / total for c, v in ev.items()}


def fuse(bayes: BayesianResult, vlm: VLMResult, classes: Sequence[str], cfg: FusionConfig) -> FusionResult:
    """Combine evidences; falls back to Bayesian-only when VLM is not OK."""
    if vlm.status != "OK" or not vlm.top_candidate:
        ranked = list(bayes.posteriors.items())
        return FusionResult(
            method="bayesian_only (VLM not available)", alpha=1.0, scores=dict(bayes.posteriors),
            final_candidate=ranked[0][0], fusion_score=ranked[0][1],
            margin=ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0),
            agreement="NOT_AVAILABLE", bayesian_top1=bayes.top1, vlm_top1=None, vlm_evidence={},
        )
    ev = vlm_evidence(vlm, classes, cfg.vlm_evidence_floor)
    a = cfg.alpha
    raw = np.array([max(bayes.posteriors.get(c, 0.0), 1e-12) ** a * ev[c] ** (1.0 - a) for c in classes])
    raw /= raw.sum()
    order = np.argsort(-raw)
    scores = {classes[i]: float(raw[i]) for i in order}
    ranked = list(scores.items())
    vlm_top = vlm.top_candidate.strip()
    agree = "AGREEMENT" if vlm_top.upper() == bayes.top1.upper() else "DISAGREEMENT"
    return FusionResult(
        method="weighted geometric evidence fusion", alpha=a, scores=scores,
        final_candidate=ranked[0][0], fusion_score=ranked[0][1],
        margin=ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0),
        agreement=agree, bayesian_top1=bayes.top1, vlm_top1=vlm_top, vlm_evidence=ev,
    )
