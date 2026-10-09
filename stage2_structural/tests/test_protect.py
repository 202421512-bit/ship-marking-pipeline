# -*- coding: utf-8 -*-
"""
3단계 문자 보호 마스크 자동 테스트  [우리 도구]
기준값(0.98 등)은 dev 결과보다 약간 낮게 잡은 '퇴보 감지용' 하한이다.
실행:  pytest stage2_structural/tests -v
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import eval_protect as ep  # noqa: E402
import make_stage2_samples as gen  # noqa: E402
import protect_mask as pm  # noqa: E402


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage2_protect_data")
    gen.generate("dev", 3, out_root=str(root), verbose=False)
    return str(root)


def text_image(fg, bg=120, size=(400, 900)):
    img = np.full((*size, 3), bg, np.uint8)
    cv2.putText(img, "AB12-7", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (fg, fg, fg), 14, cv2.LINE_AA)
    img = (img.astype(np.float32) + np.random.default_rng(0).normal(0, 2, img.shape)).clip(0, 255).astype(np.uint8)
    text = np.zeros(size, np.uint8)
    cv2.putText(text, "AB12-7", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 255, 14, cv2.LINE_AA)
    return img, text > 127


# 1. 출력 형식: bool, 같은 크기, 확실한 보호와 의심 영역은 겹치지 않음
def test_output_format(data_root):
    import json
    d = os.path.join(data_root, "dev", "C2_partial", "000")
    img = cv2.imread(os.path.join(d, "image.png"))
    pr = pm.build_protect_mask(img)
    for m in (pr.strong, pr.suspect, pr.thin_lines, pr.weak):
        assert m.dtype == bool and m.shape == img.shape[:2]
    assert not (pr.strong & pr.suspect).any()
    json.dumps(pr.info)      # 정보는 JSON으로 저장 가능해야 함


# 2. dev 성능 하한 (퇴보 감지)
def test_protect_quality(data_root):
    rows = ep.run("dev", data_root=data_root, verbose=False)
    s = {r["condition"]: r for r in ep.summarize(rows)}
    assert s["ALL"]["clear_any"] >= 0.98
    assert s["ALL"]["faint_any"] is not None and s["ALL"]["faint_any"] >= 0.85
    assert s["C3_hard"]["line_any"] >= 0.95
    assert s["C1_no_overlap"]["overprotect_any"] <= 0.25    # 지울 수 있는 스크래치를 너무 많이 막지 않음


# 3. 마킹 방향 설정: 밝은 글씨는 bright, 어두운 글씨는 dark로 잡힌다
def test_polarity():
    bright_img, text = text_image(fg=235)
    dark_img, _ = text_image(fg=30)
    rb = pm.build_protect_mask(bright_img, cfg={"polarity": "bright"})
    rd = pm.build_protect_mask(dark_img, cfg={"polarity": "dark"})
    wrong = pm.build_protect_mask(dark_img, cfg={"polarity": "bright"})
    assert (rb.protected & text).sum() / text.sum() >= 0.9
    assert (rd.protected & text).sum() / text.sum() >= 0.9
    assert (wrong.strong & text).sum() / text.sum() <= 0.1
    with pytest.raises(ValueError):
        pm.build_protect_mask(bright_img, cfg={"polarity": "both"})


# 4. 정형 문자 박스를 주면 박스(+여백) 전체가 확실한 보호
def test_protect_boxes():
    img = np.full((300, 400, 3), 120, np.uint8)
    pr = pm.build_protect_mask(img, protect_boxes=[(100, 50, 200, 120)])
    assert pr.strong[50:120, 100:200].all()


# 5. 작은 이미지에서도 동작하고 파라미터가 비율대로 줄어든다
def test_small_image_scaling(data_root):
    img = cv2.imread(os.path.join(data_root, "dev", "C1_no_overlap", "000", "image.png"))
    small = cv2.resize(img, (800, 600))
    pr = pm.build_protect_mask(small)
    assert pr.info["scale"] == pytest.approx(0.5)
    assert pr.strong.shape == (600, 800)


# 6. 직접 구현한 Sauvola가 scikit-image 결과와 같다 (scikit-image가 설치된 경우에만)
def test_sauvola_matches_skimage():
    sk = pytest.importorskip("skimage.filters")
    img, _ = text_image(fg=235)
    F = pm.foreground_image(img, "bright")
    ours = pm.sauvola_foreground(F, 51, 0.2, 128.0)
    D = 255.0 - F
    theirs = D < sk.threshold_sauvola(D.astype(np.float64), window_size=51, k=0.2, r=128.0)
    assert (ours == theirs).mean() >= 0.995
