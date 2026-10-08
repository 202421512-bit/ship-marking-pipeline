"""Confidence / ambiguity decision with human-review routing.

Thresholds are DEMO decision thresholds of a research prototype, not industrial
safety criteria. Whenever evidence is uncertain the outcome is
HUMAN_REVIEW_REQUIRED; welding conditions are never determined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .bayesian import BayesianResult, BayesianShapeClassifier
from .config import DecisionConfig
from .fusion import FusionResult
from .vlm import VLMResult

HIGH = "HIGH_CONFIDENCE"
REVIEW = "REVIEW_REQUIRED"
AMBIGUOUS = "AMBIGUOUS"


@dataclass
class DecisionResult:
    """Decision level, final routing and the reasons behind it."""

    level: str
    final_decision: str
    topology_consistency: Dict[str, object]
    geometry_consistency: Dict[str, object]
    reasons: List[str] = field(default_factory=list)
    review_candidates: List[Dict[str, object]] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        """JSON-serializable summary."""
        return {
            "level": self.level,
            "final_decision": self.final_decision,
            "threshold_type": "demo decision threshold (not an industrial safety criterion)",
            "topology_consistency": self.topology_consistency,
            "geometry_consistency": self.geometry_consistency,
            "reasons": self.reasons,
            "review_candidates": self.review_candidates,
        }


def topology_consistency(features: Dict[str, float], clf: BayesianShapeClassifier, label: str) -> Dict[str, object]:
    """Do the observed Betti numbers match the typical prototype of `label`?"""
    mu = clf.model.mean[label]
    exp_b0, exp_b1 = int(round(mu["beta_0"])), int(round(mu["beta_1"]))
    ok = int(features["beta_0"]) == exp_b0 and int(features["beta_1"]) == exp_b1
    return {"consistent": bool(ok), "candidate": label,
            "observed": {"beta_0": int(features["beta_0"]), "beta_1": int(features["beta_1"])},
            "prototype_typical": {"beta_0": exp_b0, "beta_1": exp_b1},
            "prototype_mean": {"beta_0": round(mu["beta_0"], 3), "beta_1": round(mu["beta_1"], 3)}}


def geometry_consistency(features: Dict[str, float], clf: BayesianShapeClassifier, label: str,
                         max_z: float) -> Dict[str, object]:
    """Mean |z| of geometry features against the candidate prototype."""
    z = clf.z_scores(features, label, clf.geometry_features)
    mean_abs = float(np.mean([abs(v) for v in z.values()])) if z else 0.0
    return {"consistent": bool(mean_abs <= max_z), "candidate": label,
            "mean_abs_z": round(mean_abs, 4), "max_allowed_mean_abs_z": max_z}


def decide(bayes: BayesianResult, vlm: VLMResult, fusion: FusionResult, features: Dict[str, float],
           clf: BayesianShapeClassifier, cfg: DecisionConfig,
           bayes_before: Optional[BayesianResult] = None,
           observed_features: Optional[Dict[str, float]] = None) -> DecisionResult:
    """Combine all evidence into HIGH_CONFIDENCE / REVIEW_REQUIRED / AMBIGUOUS.

    ``features`` / ``bayes`` describe the restored mask; ``observed_features`` /
    ``bayes_before`` the un-restored observation. Because the restoration is
    selected with prototype references, a result that only holds after
    restoration cannot reach HIGH_CONFIDENCE (self-reinforcement guard).
    """
    cand = fusion.final_candidate
    topo = topology_consistency(features, clf, cand)
    if observed_features is not None:
        obs_topo = topology_consistency(observed_features, clf, cand)
        topo["observed_mask_consistent"] = obs_topo["consistent"]
        topo["observed_mask"] = obs_topo["observed"]
    geo = geometry_consistency(features, clf, cand, cfg.geometry_consistency_max_z)
    reasons: List[str] = []
    vlm_ok = vlm.status == "OK"

    ambiguous = False
    if fusion.agreement == "DISAGREEMENT":
        ambiguous = True
        reasons.append(f"Bayesian Top-1 '{fusion.bayesian_top1}' and VLM Top-1 '{fusion.vlm_top1}' disagree")
    if fusion.margin < cfg.ambiguous_margin:
        ambiguous = True
        reasons.append(f"top-1/top-2 margin {fusion.margin:.3f} < {cfg.ambiguous_margin}")
    if bayes.normalized_entropy > cfg.ambiguous_normalized_entropy:
        ambiguous = True
        reasons.append(f"normalized Bayesian entropy {bayes.normalized_entropy:.3f} > {cfg.ambiguous_normalized_entropy}")

    high_checks = {
        f"fusion score >= {cfg.high_fusion_score}": fusion.fusion_score >= cfg.high_fusion_score,
        f"Bayesian posterior >= {cfg.high_bayes_posterior}": bayes.top1_posterior >= cfg.high_bayes_posterior,
        f"normalized entropy <= {cfg.max_normalized_entropy_high}": bayes.normalized_entropy <= cfg.max_normalized_entropy_high,
        "topology consistent with candidate prototype": bool(topo["consistent"]),
        "geometry consistent with candidate prototype": bool(geo["consistent"]),
    }
    if bayes_before is not None:
        high_checks["Bayesian Top-1 unchanged by restoration"] = bayes_before.top1 == bayes.top1
    if vlm_ok:
        high_checks[f"VLM agrees with confidence >= {cfg.high_vlm_confidence}"] = (
            fusion.agreement == "AGREEMENT" and (vlm.confidence or 0.0) >= cfg.high_vlm_confidence)
    elif cfg.require_vlm_for_high_confidence:
        high_checks["independent VLM verification available"] = False

    if ambiguous:
        level = AMBIGUOUS
    elif all(high_checks.values()):
        level = HIGH
        reasons.append("all demo high-confidence checks passed")
    else:
        level = REVIEW
    if level != HIGH:
        reasons += [f"check failed: {name}" for name, ok in high_checks.items() if not ok]

    final = "ACCEPTED_FOR_NEXT_STAGE" if level == HIGH else "HUMAN_REVIEW_REQUIRED"
    review = [{"class": c, "score": round(s, 6)} for c, s in list(fusion.scores.items())[:3]]
    return DecisionResult(level=level, final_decision=final, topology_consistency=topo,
                          geometry_consistency=geo, reasons=reasons, review_candidates=review)
