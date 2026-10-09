# -*- coding: utf-8 -*-
"""
decision.py
2차 공통 모듈 ②: 3단 판정 (스크래치·녹·얼룩 후보에 똑같이 적용)

  제거       : 아래 규칙을 모두 통과한 픽셀
  부분 제거  : 한 후보 안에서 통과한 픽셀만 제거하고 나머지는 남김
  보존+표시  : 제거하지 않고 '불확실(uncertain)'로 표시

근거: '불확실하면 지우지 말고 표시' [논문 4, 원문 확인], 문자 보호 마스크 [논문 3, 초록].
규칙의 순서·기준값은 모두 [우리 설정]이며 DECISION_CFG에 있다.

규칙 (위에서부터 적용)
  R1 확실한 보호(strong) 위        → 제거 안 함, 불확실
  R2 의심 영역(suspect) 위          → 제거 안 함, 불확실
  R3 보호 영역 주변 안전 거리 안     → 제거 안 함, 불확실
  R4 후보가 마킹과 닮음            → 후보 전체 제거 안 함, 불확실
  R5 후보가 확실하지 않음           → 후보 전체 제거 안 함, 불확실
  R6 남은 조각이 너무 짧음          → 그 조각 제거 안 함, 불확실
  R7 나머지                        → 제거
  여러 후보가 겹친 픽셀은 하나라도 '불확실'이면 불확실 (보수적)

안전 거리 = max(safety_px × 이미지 배율, safety_stroke_ratio × 글자 획 폭)
  → 해상도와 획 두께에 따라 조정 가능 (safety_stroke_ratio 기본 0 = 해상도만 반영)
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

DECISION_CFG = {
    "ref_long_side": 1600,
    "safety_px": 4,                 # 안전 거리 기본값 (px @1600) [실험 파라미터: 2/4/8 비교 대상]
    "safety_stroke_ratio": 0.0,     # 글자 획 폭에 비례한 안전 거리 (0이면 미사용)
    "min_run_px": 20,               # 이보다 짧은 제거 조각은 남김 (px @1600)
    "needs_check_radius": 20,       # '확실한 보호' 영역에서 이 거리 안에 불확실 후보가 있으면 사람 확인 (px @1600)
                                    # [실험 파라미터: 10/20/30 비교 대상]
}


@dataclass
class Decision:
    remove: np.ndarray
    uncertain: np.ndarray
    detect: np.ndarray
    needs_check: bool
    per_candidate: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _disk(r):
    r = max(0, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def needs_check_flag(uncertain, protected, radius_px):
    """불확실 픽셀이 보호 영역에서 radius_px 안에 있으면 True."""
    if not uncertain.any() or not protected.any():
        return False
    near = cv2.dilate(protected.astype(np.uint8), _disk(radius_px)) > 0
    return bool((uncertain & near).any())


def decide(candidates, protect, shape, cfg=None):
    cfg = {**DECISION_CFG, **(cfg or {})}
    s = max(shape[:2]) / cfg["ref_long_side"]
    text_w = protect.info.get("text_stroke_width_px", 0) or 0
    safety = int(round(max(cfg["safety_px"] * s, cfg["safety_stroke_ratio"] * text_w)))
    min_run = max(2, int(round(cfg["min_run_px"] * s)))
    nc_radius = int(round(cfg["needs_check_radius"] * s))

    protected = protect.strong | protect.suspect
    blocked = (cv2.dilate(protected.astype(np.uint8), _disk(safety)) > 0) if safety > 0 else protected.copy()

    h, w = shape[:2]
    remove = np.zeros((h, w), bool)
    uncertain = np.zeros((h, w), bool)
    detect = np.zeros((h, w), bool)
    per = []
    for i, c in enumerate(candidates):
        m = c.mask
        detect |= m
        reason = []
        if c.similar_to_marking:
            reason.append("R4_similar_to_marking")
        if not c.confident:
            reason.append("R5_not_confident")
        if reason:
            uncertain |= m
            per.append({"id": i, "kind": c.kind, "removed_px": 0, "uncertain_px": int(m.sum()),
                        "rules": reason, **c.geometry})
            continue
        ok = m & ~blocked                                    # R1~R3
        rules = []
        if (m & protect.strong).any():
            rules.append("R1_on_strong")
        if (m & protect.suspect).any():
            rules.append("R2_on_suspect")
        if (m & blocked & ~protected).any():
            rules.append("R3_safety_zone")
        # R6: 짧은 조각 제외. 가는 선의 픽셀은 군데군데 끊겨 있으므로 3px 이내 틈은 이어서 '조각 길이'를 잰다
        grown = cv2.dilate(ok.astype(np.uint8), np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
        keep = np.zeros(n, bool)
        for j in range(1, n):
            ext = max(st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]) - 6   # 7×7 팽창으로 늘어난 6px 제외
            keep[j] = ext >= min_run
        if n > 1 and not keep[1:].all():
            rules.append("R6_short_fragment")
        ok = ok & keep[lab] if n > 1 else ok
        remove |= ok
        uncertain |= m & ~ok
        per.append({"id": i, "kind": c.kind, "removed_px": int(ok.sum()), "uncertain_px": int((m & ~ok).sum()),
                    "rules": rules or ["R7_removed"], **c.geometry})

    remove &= ~uncertain                                     # 겹친 픽셀은 불확실 우선
    # needs_check 기준 영역 = '확실한 보호'(글자로 인식된 핵심 영역).
    # 의심 영역 전체를 기준으로 하면 흐린 스크래치 조각까지 기준이 되어, 겹침이 없는 C1에서도 90%가 확인 요청됨 (dev 관찰)
    needs = needs_check_flag(uncertain, protect.strong, nc_radius)
    info = {"safety_px_effective": safety, "min_run_px_effective": min_run,
            "needs_check_radius_effective": nc_radius, "n_candidates": len(candidates),
            "n_removed_candidates": sum(1 for p in per if p["removed_px"] > 0)}
    return Decision(remove=remove, uncertain=uncertain, detect=detect, needs_check=needs,
                    per_candidate=per, info=info)
