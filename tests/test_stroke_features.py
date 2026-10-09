"""stroke_features.py φ₁ (획 두께 변화) 검증."""
import cv2
import numpy as np
import pytest

from stroke_features import (render_handwritten, render_printed, stroke_thickness_variation)


def bar(width, length=120, angle=0.0, size=200):
    """굵기가 일정한 직선 획 (이진 마스크)."""
    img = np.zeros((size, size), np.uint8)
    c = size // 2
    t = np.deg2rad(angle)
    dx, dy = int(length / 2 * np.cos(t)), int(length / 2 * np.sin(t))
    cv2.line(img, (c - dx, c - dy), (c + dx, c + dy), 255, width)
    return img > 0


def tapered(widths, length=150, size=220):
    """구간마다 굵기가 다른 획 (굵어졌다 가늘어짐)."""
    img = np.zeros((size, size), np.uint8)
    xs = np.linspace(35, 35 + length, len(widths) + 1).astype(int)
    for (x0, x1), w in zip(zip(xs[:-1], xs[1:]), widths):
        cv2.line(img, (x0, size // 2), (x1, size // 2), 255, w)
    return img > 0


def test_uniform_stroke_has_small_phi():
    r = stroke_thickness_variation(bar(10))   # cv2.line 굵기 10 -> 실제 11px
    assert r.valid
    # 거리 변환은 화소 중심 간 거리라 n px 폭 획의 중심 D = (n+1)/2 -> W = 2D 는 n+1 (정의상 +1px 편향)
    assert abs(r.mu_w - 12) <= 1
    assert r.cv < 0.08 and r.phi1 < 0.3


def test_varying_stroke_has_large_phi():
    r = stroke_thickness_variation(tapered([4, 8, 14, 20, 12, 6]))
    assert r.valid and r.cv > 0.3 and r.phi1 > 0.8


@pytest.mark.parametrize("angle", [0, 30, 45, 77])
def test_rotation_invariant(angle):
    assert stroke_thickness_variation(bar(10, angle=angle)).cv < 0.1


def test_phi_is_bounded_and_monotonic_in_alpha():
    img = tapered([4, 10, 16])
    lo = stroke_thickness_variation(img, alpha=1.0)
    hi = stroke_thickness_variation(img, alpha=8.0)
    assert 0 <= lo.phi1 < hi.phi1 < 1
    assert lo.cv == pytest.approx(hi.cv)


def test_thin_strokes_are_upscaled_and_scale_free():
    """가는 획은 확대 후 계산: 선폭은 입력 px 기준으로, CV 는 배율과 무관해야 한다."""
    gray = np.full((80, 80), 220, np.uint8)
    cv2.line(gray, (10, 40), (70, 40), 30, 3)       # 실제 5px 폭 -> W ≈ 6px (+1px 편향)
    r = stroke_thickness_variation(gray, min_width_px=10)
    assert r.debug["scale"] > 1
    assert abs(r.mu_w - 6) <= 1.5 and r.cv < 0.2     # 확대해도 선폭은 입력 px 기준


def test_polarity_auto_matches_explicit():
    dark = render_printed("B12")              # 밝은 바탕, 어두운 획
    light = 255 - dark                         # 어두운 바탕, 밝은 획
    a, b = stroke_thickness_variation(dark), stroke_thickness_variation(light)
    assert a.cv == pytest.approx(stroke_thickness_variation(dark, polarity="dark").cv)
    assert a.cv == pytest.approx(b.cv, abs=0.02)


def test_robust_to_pinholes_and_specks():
    clean = bar(14)
    noisy = clean.copy()
    rng = np.random.default_rng(0)
    ys, xs = np.nonzero(clean)
    pick = rng.choice(len(ys), 25, replace=False)
    noisy[ys[pick], xs[pick]] = False         # 획 안의 기포/녹 반점
    by, bx = rng.integers(0, 200, 40), rng.integers(0, 200, 40)
    noisy[by, bx] = True                       # 바탕의 점 잡음
    assert abs(stroke_thickness_variation(noisy).cv - stroke_thickness_variation(clean).cv) < 0.05


def test_closed_counters_do_not_fake_thin_strokes():
    """두꺼운 '0' 은 빗금 양옆 구멍이 거의 막힌다 -> 그 옆 골격이 가는 획으로 잡히면 안 됨."""
    assert stroke_thickness_variation(render_printed("4500")).cv < 0.15


def test_printed_vs_handwritten_on_synthetic_set():
    printed = [stroke_thickness_variation(render_printed(t)).phi1 for t in ("B12-SP3", "4500", "HK357")]
    hand = [stroke_thickness_variation(render_handwritten(t, seed=s)).phi1
            for s, t in enumerate(("B12-SP3", "4500", "HK357"))]
    assert max(printed) < 0.5 < min(hand)


@pytest.mark.parametrize("img", [np.zeros((50, 50), np.uint8), np.full((50, 50), 255, np.uint8)])
def test_empty_input_is_invalid_not_crash(img):
    r = stroke_thickness_variation(img)
    assert not r.valid and np.isnan(r.phi1)


def test_tiny_blob_is_invalid():
    img = np.zeros((40, 40), bool)
    img[18:22, 18:22] = True
    assert not stroke_thickness_variation(img).valid
