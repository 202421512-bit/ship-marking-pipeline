"""size_variance_feature.py φ₆ (크기·종횡비 분산) 검증."""
import math

import cv2
import numpy as np
import pytest

import orientation_feature as of
import size_variance_feature as sv

STROKE = 6


def glyphs(sizes, size=(260, 640), gap=18, base=170, angle=0.0):
    """(폭, 높이) 목록대로 'ㅁ' 모양 글자를 한 줄로 (바닥 정렬). angle 이면 줄 전체 회전."""
    img = np.full(size, 210, np.uint8)
    x = 25
    for w, h in sizes:
        cv2.rectangle(img, (x + 3, base - h + 3), (x + w - 3, base - 3), 40, STROKE)
        x += w + gap
    M = cv2.getRotationMatrix2D((size[1] / 2, size[0] / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (size[1], size[0]), borderValue=210)


def test_identical_glyphs_give_zero():
    r = sv.calculate_phi_6_size_variance(glyphs([(36, 60)] * 7), robust="none")
    assert r["num_valid_chars"] == 7
    assert r["cv_area"] < 0.02 and r["cv_aspect_ratio"] < 0.02 and r["cv_height"] < 0.02
    assert r["phi_6"] < 0.1


def test_cv_matches_definition():
    sizes = [(30, 50), (40, 60), (50, 70), (36, 60), (44, 56)]
    r = sv.calculate_phi_6_size_variance(glyphs(sizes), robust="none")
    ar = np.array(r["aspect_ratios"])
    assert r["cv_aspect_ratio"] == pytest.approx(ar.std() / ar.mean(), rel=1e-6)   # 모집단 표준편차
    expected_ar = np.array([w / h for w, h in sizes])
    assert np.allclose(np.sort(ar), np.sort(expected_ar), atol=0.05)
    expected = 0.5 * (math.tanh(sv.ALPHA_A * r["cv_area"]) + math.tanh(sv.ALPHA_AR * r["cv_aspect_ratio"]))
    assert r["phi_6"] == pytest.approx(expected)


@pytest.mark.parametrize("angle", [0, 10, -15])
def test_aspect_ratio_measured_along_the_line(angle):
    r = sv.calculate_phi_6_size_variance(glyphs([(36, 60)] * 6, angle=angle), robust="none")
    assert r["num_valid_chars"] >= 5
    assert r["cv_aspect_ratio"] < 0.04
    assert np.allclose(r["aspect_ratios"], 36 / 60, atol=0.05)


def test_punctuation_scratch_and_specks_excluded():
    img = glyphs([(36, 60)] * 6)
    cv2.rectangle(img, (520, 145), (540, 151), 40, -1)            # '-' 처럼 낮은 성분
    cv2.line(img, (20, 200), (620, 205), 40, 2)                    # 글자 아래 긴 긁힘 (두께 2px)
    rng = np.random.default_rng(0)
    for _ in range(40):                                            # 녹 반점
        cv2.circle(img, (int(rng.integers(0, 640)), int(rng.integers(0, 180))), 1, 60, -1)
    r = sv.calculate_phi_6_size_variance(img, robust="none")
    assert r["num_valid_chars"] == 6
    assert r["cv_area"] < 0.03
    assert any(c["reason"] == "punctuation" for g in r["chars"] for c in g["members"])


def test_off_line_noise_is_not_merged_into_a_character():
    img = glyphs([(36, 60)] * 6)
    cv2.rectangle(img, (30, 60), (50, 74), 40, -1)                 # 첫 글자 바로 위 얼룩
    r = sv.calculate_phi_6_size_variance(img, robust="none")
    assert r["cv_height"] < 0.03
    assert any(c["reason"] == "noise:off_line" for c in r["components"])


def test_broken_character_is_one_character():
    img = glyphs([(36, 60)] * 6)
    img[135:140, 25 + 2 * 54: 25 + 2 * 54 + 36] = 210              # 세 번째 글자를 위아래로 끊음
    r = sv.calculate_phi_6_size_variance(img, robust="none")
    assert r["num_valid_chars"] == 6 and r["cv_height"] < 0.03


def test_iqr_drops_shape_outlier_and_huber_softens_it():
    sizes = [(36, 60)] * 7 + [(12, 60)]                            # '1' 이나 '/' 처럼 혼자 좁은 글자
    none = sv.calculate_phi_6_size_variance(glyphs(sizes), robust="none")
    iqr = sv.calculate_phi_6_size_variance(glyphs(sizes), robust="iqr")
    hub = sv.calculate_phi_6_size_variance(glyphs(sizes), robust="huber")
    assert iqr["num_valid_chars"] == 7 and iqr["cv_aspect_ratio"] < 0.03
    assert hub["cv_aspect_ratio"] < none["cv_aspect_ratio"]


def test_height_score_separates_printed_from_handwritten():
    p = sv.calculate_phi_6_size_variance(of.add_plate_noise(of.render_printed("HK357 B12", angle=5), seed=1),
                                         score="height")
    h = sv.calculate_phi_6_size_variance(of.add_plate_noise(
        of.render_handwritten("HK357 B12", seed=5, tilt_std=6, baseline_jitter=5), seed=2), score="height")
    assert p["phi_6"] < 0.5 < h["phi_6"]


@pytest.mark.parametrize("img", [np.full((80, 200), 200, np.uint8), glyphs([(36, 60)])])
def test_fewer_than_two_chars(img):
    r = sv.calculate_phi_6_size_variance(img)
    assert r["phi_6"] is None and r["reason"]


def test_debug_colors_green_and_red(tmp_path):
    sizes = [(36, 60)] * 5 + [(70, 60)]                            # 마지막 글자만 2배 넓음
    path = tmp_path / "크기.png"
    ok, buf = cv2.imencode(".png", glyphs(sizes))
    buf.tofile(str(path))
    r = sv.calculate_phi_6_size_variance(str(path), robust="none")
    vis = sv.draw_size_variance_debug(str(path), r, scale=1.0)
    b, g, rr = cv2.split(vis.astype(int))
    assert ((g > 200) & (rr < 80) & (b < 80)).any()                # 평균 근접 = 초록
    assert ((rr > 200) & (g < 120) & (b < 80)).any()               # 이상 편차 = 빨강
