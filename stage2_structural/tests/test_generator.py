# -*- coding: utf-8 -*-
"""
2차 가상 데이터 생성기 자동 테스트  [우리 도구]
실행 (저장소 최상위 폴더에서):  pytest stage2_structural/tests -v
OpenCV·NumPy만 사용하므로 팀 CI의 `pytest -v`에서도 함께 실행된다.
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import make_stage2_samples as gen  # noqa: E402

N_PER_COND = 3          # 테스트용 소량 생성 (조건당 3장)
MASKS = ["gt_text", "gt_symbol", "gt_text_faint", "gt_marking_line", "gt_protect", "gt_scratch",
         "gt_preexisting_damage", "gt_protect_intact", "gt_glare"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage2_data")
    gen.generate("dev", N_PER_COND, out_root=str(root), verbose=False)
    return root


def load(root, cond, i):
    d = os.path.join(root, "dev", cond, f"{i:03d}")
    m = {k: cv2.imread(os.path.join(d, k + ".png"), cv2.IMREAD_GRAYSCALE) > 0 for k in MASKS}
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    return m, meta


def all_samples(root):
    for cond in gen.CONDITIONS:
        for i in range(N_PER_COND):
            yield cond, i, *load(root, cond, i)


# 1. 파일이 모두 있고 필수 마스크가 비어 있지 않다
def test_files_and_nonempty(dataset):
    for cond, i, m, meta in all_samples(dataset):
        assert m["gt_text"].any() and m["gt_protect"].any() and m["gt_scratch"].any(), (cond, i)
        assert m["gt_marking_line"].any() == (cond == "C3_hard"), (cond, i)
        assert m["gt_glare"].any() == (meta["glare_center"] is not None), (cond, i)


# 2. 보호 대상 = 문자 ∪ 기호 ∪ 직선 마킹, 손상/온전 분리가 정확하다
def test_mask_algebra(dataset):
    for cond, i, m, _ in all_samples(dataset):
        union = m["gt_text"] | m["gt_symbol"] | m["gt_marking_line"]
        assert np.array_equal(union, m["gt_protect"]), (cond, i)
        assert np.array_equal(m["gt_preexisting_damage"], m["gt_protect"] & m["gt_scratch"])
        assert np.array_equal(m["gt_protect_intact"], m["gt_protect"] & ~m["gt_scratch"])
        assert not (m["gt_preexisting_damage"] & m["gt_protect_intact"]).any()


# 3. C1: 스크래치와 보호 대상이 전혀 겹치지 않는다
def test_c1_no_overlap(dataset):
    for i in range(N_PER_COND):
        m, _ = load(dataset, "C1_no_overlap", i)
        assert (m["gt_scratch"] & m["gt_protect"]).sum() == 0


# 4. C2: 모든 이미지에 겹침이 있고, 글자를 지나는 스크래치는 각각 겹친다. 폭은 1~2px
def test_c2_partial_overlap(dataset):
    for i in range(N_PER_COND):
        m, meta = load(dataset, "C2_partial", i)
        assert (m["gt_scratch"] & m["gt_protect"]).sum() > 0
        crossing = [s for s in meta["scratches"] if "like_mark" in s]
        assert crossing and all(s["overlap_px"] > 0 for s in crossing)
        assert all(s["width_px"] in (1, 2) for s in meta["scratches"])


# 5. C3: 스크래치의 실제 폭이 '닮은 대상 글자'의 실제 획 폭과 ±1.5px 이내
#    (두 마스크 모두 같은 방법으로 측정: 거리 변환 값의 90번째 백분위 × 2)
stroke_width = gen.stroke_width


def test_c3_width_like_stroke(dataset):
    for i in range(N_PER_COND):
        m, meta = load(dataset, "C3_hard", i)
        boxes = {mk["text"]: mk["box_xyxy"] for mk in meta["markings"]}
        for s in meta["scratches"]:
            assert s["mode"] == "like_stroke"
            x1, y1, x2, y2 = boxes[s["like_mark"]]
            w_text = stroke_width(m["gt_text"][y1:y2, x1:x2])
            w_scr = stroke_width(gen.scratch_layer(s["p0"], s["p1"], s["width_px"]) > 127)
            assert abs(w_scr - w_text) <= 1.5, (i, s["like_mark"], round(w_scr, 2), round(w_text, 2))


# 6. meta의 마킹 박스가 실제 문자·기호 픽셀을 모두 포함한다
def test_boxes_cover_strokes(dataset):
    for cond, i, m, meta in all_samples(dataset):
        inside = np.zeros_like(m["gt_text"])
        for mk in meta["markings"]:
            x1, y1, x2, y2 = mk["box_xyxy"]
            box = np.zeros_like(inside)
            box[y1:y2, x1:x2] = True
            assert (box & (m["gt_text"] | m["gt_symbol"])).any(), (cond, i, mk["text"])
            inside |= box
        assert not ((m["gt_text"] | m["gt_symbol"]) & ~inside).any(), (cond, i)
        assert len(meta["gt_text_strings"]) == len(meta["markings"]) >= 3


# 7. dev와 eval 시드가 겹치지 않고, 같은 시드는 같은 이미지를 만든다
def test_seed_split_and_determinism():
    dev = {gen.seed_for("dev", c, i) for c in gen.CONDITIONS for i in range(gen.MAX_PER_CONDITION)}
    ev = {gen.seed_for("eval", c, i) for c in gen.CONDITIONS for i in range(gen.MAX_PER_CONDITION)}
    assert not dev & ev
    a, _, _ = gen.make_one(1234, "C2_partial")
    b, _, _ = gen.make_one(1234, "C2_partial")
    assert np.array_equal(a, b)


# 8. 흐린 획: 문자·기호의 부분집합이고, 전체 마킹의 약 30%이며, 배경 대비가 선명한 획의 60% 미만
def test_faint_strokes(dataset):
    n_mark = n_faint = 0
    for cond, i, m, meta in all_samples(dataset):
        assert not (m["gt_text_faint"] & ~(m["gt_text"] | m["gt_symbol"])).any(), (cond, i)
        n_mark += len(meta["markings"])
        n_faint += sum(mk["faint"] for mk in meta["markings"])
        img = cv2.imread(os.path.join(dataset, "dev", cond, f"{i:03d}", "image.png"))
        L8 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
        contrast = L8.astype(float) - cv2.medianBlur(L8, 61).astype(float)   # 주변 배경 대비 밝기
        base = m["gt_protect_intact"] & ~m["gt_glare"]
        faint, clear = base & m["gt_text_faint"], base & ~m["gt_text_faint"] & ~m["gt_marking_line"]
        if faint.sum() > 200 and clear.sum() > 200:
            assert contrast[faint].mean() < 0.6 * contrast[clear].mean(), (cond, i)
    assert n_faint >= 1 and 0.1 <= n_faint / n_mark <= 0.55, (n_faint, n_mark)
