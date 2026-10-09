"""spacing_feature.py φ₂ (문자 간격 편차) 검증."""
import math

import cv2
import numpy as np
import pytest

import orientation_feature as of
import spacing_feature as sp

STROKE = 6


def row(gaps, w=34, h=56, size=(240, 700), angle=0.0, x0=30):
    """'ㅁ' 글자를 지정한 간격(px)으로 한 줄. 음수면 겹침. angle 이면 줄 전체 회전."""
    img = np.full(size, 210, np.uint8)
    x, base = x0, size[0] // 2 + h // 2
    for i in range(len(gaps) + 1):
        cv2.rectangle(img, (x + 3, base - h + 3), (x + w - 4, base - 3), 40, STROKE)
        if i < len(gaps):
            x += w + gaps[i]
    M = cv2.getRotationMatrix2D((size[1] / 2, size[0] / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (size[1], size[0]), borderValue=210)


def test_equal_spacing_gives_zero():
    r = sp.calculate_phi_2_spacing(row([12] * 7))
    assert r["num_valid_chars"] == 8
    assert np.allclose(r["gaps_px"], 12, atol=1.5)
    assert r["sigma_d_px"] < 0.8 and r["phi_2"] < 0.2


def test_sigma_matches_population_std():
    gaps = [8, 16, 10, 20, 6, 14]
    r = sp.calculate_phi_2_spacing(row(gaps))
    assert r["sigma_d_px"] == pytest.approx(np.std(gaps), abs=0.8)
    expected = 1 - math.exp(-sp.ALPHA2 * r["sigma_d_px"] / r["mean_char_width"])
    assert r["phi_2"] == pytest.approx(expected)


def test_overlapping_letters_give_negative_gaps():
    r = sp.calculate_phi_2_spacing(row([12, -4, 12, 12, 12], w=40), spread="std")
    assert min(r["gaps_px"]) < 0 or r["num_valid_chars"] < 6   # 겹치면 음수 간격 (또는 한 덩어리로 붙음)


@pytest.mark.parametrize("angle", [0, 8, -12])
def test_gaps_measured_along_tilted_line(angle):
    r = sp.calculate_phi_2_spacing(row([12] * 6, angle=angle))
    assert r["num_valid_chars"] == 7
    assert np.allclose(r["gaps_px"], 12, atol=2.0)


def test_scale_invariance():
    small = row([8, 16, 10, 20, 6])
    big = cv2.resize(small, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    a, b = sp.calculate_phi_2_spacing(small), sp.calculate_phi_2_spacing(big)
    assert b["sigma_d_px"] == pytest.approx(2 * a["sigma_d_px"], rel=0.15)
    assert b["phi_2"] == pytest.approx(a["phi_2"], abs=0.05)


def test_word_space_and_punctuation_are_excluded():
    gaps = [12, 12, 60, 12, 12, 12]                         # 세 번째가 띄어쓰기
    img = row(gaps)
    r = sp.calculate_phi_2_spacing(img)
    assert r["num_word_gaps"] == 1 and r["sigma_d_px"] < 1.0
    # 하이픈: 글자 사이에 낮은 성분을 넣으면 그 양옆 간격은 통계에서 빠진다
    img2 = row([12, 40, 12, 12, 12])
    cv2.rectangle(img2, (30 + 2 * 34 + 12 + 8, 114), (30 + 2 * 34 + 12 + 32, 123), 40, -1)   # 24×9 하이픈
    r2 = sp.calculate_phi_2_spacing(img2)
    assert any(why == "punctuation_between" for *_, why in r2["all_gaps"])
    assert r2["sigma_d_px"] < 1.0


def test_broken_letters_and_specks_do_not_make_fake_gaps():
    img = row([12] * 6)
    img[118:123, 30 + 2 * 46: 30 + 2 * 46 + 34] = 210        # 세 번째 글자를 위아래로 끊음
    rng = np.random.default_rng(1)
    for _ in range(50):
        cv2.circle(img, (int(rng.integers(0, 700)), int(rng.integers(0, 240))), 1, 60, -1)
    r = sp.calculate_phi_2_spacing(img)
    assert r["num_valid_chars"] == 7 and r["sigma_d_px"] < 1.0


def test_monospace_pitch_metric():
    img = of.add_plate_noise(sp.render_monospace("B12-SP3 4500", pitch=40, angle=5), seed=2)
    r = sp.calculate_phi_2_spacing(img, metric="pitch")
    assert r["sigma_pitch_px"] / r["mean_char_width"] < 0.06 and r["phi_2"] < 0.5


def test_irregular_spacing_scores_higher_than_printed():
    p = sp.calculate_phi_2_spacing(of.add_plate_noise(of.render_printed("HK357 B12"), seed=3))
    q = sp.calculate_phi_2_spacing(of.add_plate_noise(sp.render_irregular_spacing("HK357 B12", jitter=9, seed=4), seed=4))
    assert p["phi_2"] < q["phi_2"]


@pytest.mark.parametrize("img", [np.full((80, 200), 200, np.uint8), row([])])
def test_fewer_than_two_chars(img):
    r = sp.calculate_phi_2_spacing(img)
    assert r["phi_2"] is None and r["reason"]


def test_debug_draws_gap_arrows(tmp_path):
    path = tmp_path / "간격.png"
    ok, buf = cv2.imencode(".png", row([10, 20, 14]))
    buf.tofile(str(path))
    r = sp.calculate_phi_2_spacing(str(path))
    vis = sp.draw_spacing_debug(str(path), r, scale=1.0)
    b, g, rr = cv2.split(vis.astype(int))
    assert ((b > 200) & (g < 120) & (rr < 60)).any()    # 간격 화살표 = 파랑
    assert ((g > 150) & (b < 60) & (rr < 60)).any()     # 글자 상자 = 초록
