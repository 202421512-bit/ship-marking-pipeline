# -*- coding: utf-8 -*-
"""
pipeline.py
2차 구조적 노이즈 제거 전체 흐름  [우리 설계]

  입력(1차 결과) → ① 문자 보호 마스크 → 검출기들(스크래치 / 녹·얼룩은 후속) → ② 3단 판정
               → 선택적 제거(inpaint) → 보호 영역 원본 복원·검증 → Stage2Result

기존 1차 코드와 OCR 모듈은 수정하지 않으며, 이 함수 하나로 독립 실행된다.

  from pipeline import run_stage2
  result = run_stage2(image_bgr, protect_boxes=None, cfg={"decision": {"safety_px": 4}})
"""
import time

import cv2
import numpy as np

from decision import decide
from detectors import rust, scratch, stain
from protect_mask import build_protect_mask
from stage2_io import Stage2Result

DETECTORS = {"scratch": scratch, "rust": rust, "stain": stain}

PIPELINE_CFG = {
    "detectors": ["scratch"],       # MVP: 스크래치만. 녹·얼룩은 구현 후 추가
    "inpaint_radius": 3,            # [일반 기법 cv2.inpaint(Telea), 값은 우리 설정]
    "edge_grow_px": 1,              # 제거 영역을 1px 넓혀 선 가장자리(안티에일리어싱)까지 채움
}


def run_stage2(image_bgr, protect_boxes=None, cfg=None):
    cfg = cfg or {}
    pcfg = {**PIPELINE_CFG, **cfg.get("pipeline", {})}
    t = {}
    t0 = time.perf_counter()
    protect = build_protect_mask(image_bgr, protect_boxes, cfg.get("protect"))
    t["protect_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cands = []
    for name in pcfg["detectors"]:
        cands += DETECTORS[name].detect(image_bgr, protect, cfg.get(name))
    t["detect_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = decide(cands, protect, image_bgr.shape, cfg.get("decision"))
    t["decide_s"] = time.perf_counter() - t0

    # 선택적 제거: 제거 픽셀을 1px 넓히되 보호 영역·불확실 영역으로는 넓히지 않음
    t0 = time.perf_counter()
    protected = protect.strong | protect.suspect
    remove = dec.remove
    g = pcfg["edge_grow_px"]
    if g > 0 and remove.any():
        remove = (cv2.dilate(remove.astype(np.uint8), np.ones((2 * g + 1, 2 * g + 1), np.uint8)) > 0)
        remove &= ~protected & ~dec.uncertain
    out = image_bgr.copy()
    if remove.any():
        out = cv2.inpaint(image_bgr, remove.astype(np.uint8) * 255, pcfg["inpaint_radius"], cv2.INPAINT_TELEA)
    # 보호 영역 원본 복원과 검증: 제거로 선언하지 않은 픽셀은 원본 그대로 되돌린다
    keep = ~remove
    changed_before_restore = int(np.any(out[keep] != image_bgr[keep], axis=-1).sum())
    out[keep] = image_bgr[keep]
    protected_changed = int(np.any(out[protected] != image_bgr[protected], axis=-1).sum())
    t["remove_s"] = time.perf_counter() - t0

    report = {
        "protect": protect.info, "decision": dec.info,
        "candidates": [{**p, **{k: v for k, v in c.features.items()}, "confident": c.confident,
                        "similar_to_marking": c.similar_to_marking}
                       for p, c in zip(dec.per_candidate, cands)],
        "verification": {"pixels_changed_outside_remove_before_restore": changed_before_restore,
                         "protected_pixels_changed": protected_changed},
        "detectors_used": pcfg["detectors"],
        "detectors_not_implemented": [n for n, m in DETECTORS.items() if not getattr(m, "IMPLEMENTED", True)],
        "timing_s": {k: round(v, 4) for k, v in t.items()},
    }
    return Stage2Result(image=out, detect_mask=dec.detect, remove_mask=remove, uncertain_mask=dec.uncertain,
                        glare_mask=np.zeros(image_bgr.shape[:2], bool),   # 5단계에서 연결
                        needs_check=dec.needs_check, report=report)


def method_stage2(img, protect_boxes=None, gt=None, cfg=None):
    """evaluate_stage2.run()에 넘기는 형태. gt는 사용하지 않는다(정답을 보지 않음)."""
    return run_stage2(img, protect_boxes, cfg)
