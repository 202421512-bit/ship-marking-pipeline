# -*- coding: utf-8 -*-
"""
4단계 스크래치 검출 + 3단 판정 + 선택적 제거 자동 테스트  [우리 도구]
실행:  pytest stage2_structural/tests -v
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import evaluate_stage2 as ev  # noqa: E402
import make_stage2_samples as gen  # noqa: E402
from candidates import Candidate  # noqa: E402
from decision import decide, needs_check_flag  # noqa: E402
from detectors import scratch  # noqa: E402
from pipeline import run_stage2  # noqa: E402
from protect_mask import ProtectResult, build_protect_mask  # noqa: E402

H, W = 600, 800


def plate(seed=1):
    gen_h, gen_w = gen.H, gen.W
    img = gen.make_plate(np.random.default_rng(seed))[:H, :W]
    return np.clip(img, 0, 255).astype(np.uint8)


def fake_protect(strong=None, suspect=None, text_w=8.0):
    z = np.zeros((H, W), bool)
    return ProtectResult(strong=z.copy() if strong is None else strong, suspect=z.copy() if suspect is None else suspect,
                         thin_lines=z.copy(), thick_lines=z.copy(), weak=z.copy(), attached=z.copy(),
                         info={"polarity": "bright", "text_stroke_width_px": text_w})


def line_mask(p0, p1, w=2):
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, p0, p1, 1, w)
    return m > 0


# 1. 스크래치 없는 철판 질감에서는 '확실한' 후보가 없어야 함 (결 무늬 오검출 방지)
def test_no_confident_on_plain_plate():
    img = plate()
    cands = scratch.detect(img, fake_protect())
    assert not any(c.confident for c in cands)


# 2. 길고 가는 밝은 선 하나는 확실한 후보로 잡히고, 마킹과 닮지 않음
def test_detects_thin_scratch():
    img = plate().astype(np.float32)
    m = line_mask((100, 100), (700, 450), 2)
    img[m] += 50
    img = np.clip(img, 0, 255).astype(np.uint8)
    cands = [c for c in scratch.detect(img, fake_protect()) if c.confident]
    assert cands, "선을 찾지 못함"
    best = max(cands, key=lambda c: (c.mask & m).sum())
    assert (best.mask & m).sum() >= 0.5 * m.sum() * 0.5
    assert not best.similar_to_marking


# 3. 3단 판정 규칙
def test_decision_rules():
    cand_mask = line_mask((50, 300), (750, 300), 2)
    strong = np.zeros((H, W), bool)
    strong[280:320, 380:420] = True                      # 선이 지나가는 글자
    pr = fake_protect(strong=strong)
    c = Candidate("scratch", cand_mask, confident=True, similar_to_marking=False, geometry={"p0": [50, 300], "p1": [750, 300]})
    d = decide([c], pr, (H, W), {"safety_px": 4 * 1600 / W})   # 이 크기에서 실제 4px
    assert not (d.remove & strong).any()                                   # R1
    near = cv2.dilate(strong.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert not (d.remove & near).any()                                     # R3 안전 거리
    assert d.remove.sum() > 0.7 * (cand_mask & ~near).sum()                # 나머지는 제거
    assert d.needs_check                                                   # 글자 위 불확실 → 확인 요청
    # R4 닮음, R5 확신 부족 → 전혀 제거하지 않음
    for kw in ({"confident": True, "similar_to_marking": True}, {"confident": False, "similar_to_marking": False}):
        d2 = decide([Candidate("scratch", cand_mask, **kw)], fake_protect(), (H, W))
        assert d2.remove.sum() == 0 and d2.uncertain.sum() == cand_mask.sum()
    # R6 짧은 조각은 남김
    short = line_mask((100, 100), (104, 100), 2)          # 이 크기에서 최소 조각 길이는 10px
    d3 = decide([Candidate("scratch", short, True, False)], fake_protect(), (H, W))
    assert d3.remove.sum() == 0


# 4. needs_check 반경이 클수록 확인 요청이 줄지 않는다 (단조성)
def test_needs_check_radius_monotonic():
    unc = line_mask((100, 100), (300, 100), 2)
    strong = np.zeros((H, W), bool)
    strong[115:130, 150:200] = True                     # 선에서 약 15px 떨어진 글자
    assert not needs_check_flag(unc, strong, 10)
    assert needs_check_flag(unc, strong, 20) and needs_check_flag(unc, strong, 30)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage2_scratch_data")
    gen.generate("dev", 2, out_root=str(root), verbose=False)
    return str(root)


# 5. 파이프라인 보장: 보호 영역과 '제거' 밖 픽셀은 원본 그대로, 녹·얼룩은 미구현으로 표시
def test_pipeline_guarantees(data_root):
    for cond in gen.CONDITIONS:
        img = cv2.imread(os.path.join(data_root, "dev", cond, "000", "image.png"))
        res = run_stage2(img).validate(img)
        assert res.report["verification"]["protected_pixels_changed"] == 0
        keep = ~res.remove_mask
        assert np.array_equal(res.image[keep], img[keep])
        assert not (res.remove_mask & res.uncertain_mask).any()
        assert set(res.report["detectors_not_implemented"]) == {"rust", "stain"}
        pr = build_protect_mask(img)
        assert not (res.remove_mask & (pr.strong | pr.suspect)).any()


# 6. C3(구분 어려움)는 지우지 않고 불확실로 남기는 것이 의도된 동작
def test_c3_is_conservative(data_root):
    img = cv2.imread(os.path.join(data_root, "dev", "C3_hard", "000", "image.png"))
    gt = cv2.imread(os.path.join(data_root, "dev", "C3_hard", "000", "gt_scratch.png"), 0) > 0
    res = run_stage2(img)
    assert res.uncertain_mask.any()
    assert (res.remove_mask & gt).sum() <= 0.1 * gt.sum()


# 7. 평가 코드의 선분 단위 지표: 정답 기준선은 선분 재현율 1
def test_segment_metrics_with_oracle(data_root, tmp_path):
    rows, _, _ = ev.run("oracle_conservative", "dev", data_root=data_root, out_dir=str(tmp_path), verbose=False)
    for r in rows:
        assert r["seg_det_R"] == pytest.approx(1.0)
