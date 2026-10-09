"""contour_roughness_feature.py φ₇ (테두리 거칠기) 검증."""
import math

import cv2
import numpy as np
import pytest

import contour_roughness_feature as cr
import orientation_feature as of


def disk(radius=60, size=200, scale=1):
    img = np.full((size * scale, size * scale), 210, np.uint8)
    cv2.circle(img, (size * scale // 2, size * scale // 2), radius * scale, 40, -1, cv2.LINE_AA)
    return img


def test_isoperimetric_of_circle_and_square():
    t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    circle = np.column_stack([100 + 50 * np.cos(t), 100 + 50 * np.sin(t)])
    r, a = cr.isoperimetric(circle)
    assert r == pytest.approx(1.0, abs=1e-3) and a == pytest.approx(math.pi * 2500, rel=1e-3)
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert cr.isoperimetric(square)[0] == pytest.approx(4 / math.pi)      # P²/(4πA) = 40²/(4π·100)


def test_pixel_staircase_is_not_roughness():
    """디지털 원(계단 외곽선)의 R 은 1 에 가깝게, ρ 도 1 에 가깝게 나와야 한다."""
    r = cr.calculate_phi_7_roughness(disk())
    assert r["num_valid_contours"] == 1
    assert r["roughness_values"][0] < 1.05
    assert r["relative_mean"] < 1.02 and r["phi_7"] < 0.1


def test_rough_edges_score_higher():
    printed = of.render_printed("B12-SP3 4500", thickness=8)
    smooth = cr.calculate_phi_7_roughness(of.add_plate_noise(printed, seed=1))
    rough = cr.calculate_phi_7_roughness(of.add_plate_noise(cr.roughen_edges(printed, 0.5, seed=2), seed=3))
    assert rough["relative_mean"] > smooth["relative_mean"] + 0.01
    assert smooth["phi_7"] < 0.3 < 0.7 < rough["phi_7"]


def test_relative_score_ignores_glyph_shape():
    """매끈한 '가는 획' 과 '굵은 원판' 은 등주비 R 이 크게 다르지만 ρ 는 둘 다 1 근처여야 한다."""
    bar = np.full((200, 300), 210, np.uint8)
    cv2.rectangle(bar, (30, 90), (270, 104), 40, -1)
    thin = cr.calculate_phi_7_roughness(bar)
    round_ = cr.calculate_phi_7_roughness(disk())
    assert thin["isoperimetric_mean"] > 4 * round_["isoperimetric_mean"]
    assert abs(thin["relative_mean"] - round_["relative_mean"]) < 0.02


def test_scale_invariance():
    printed = cr.roughen_edges(of.render_printed("HK357", size=(140, 300), thickness=8), 0.5, seed=4)
    big = cv2.resize(printed, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    a, b = cr.calculate_phi_7_roughness(printed), cr.calculate_phi_7_roughness(big)
    assert b["stroke_width"] == pytest.approx(2 * a["stroke_width"], rel=0.15)
    assert b["relative_mean"] == pytest.approx(a["relative_mean"], abs=0.03)


def test_rotation_invariance():
    img = of.render_printed("B12-SP3", size=(160, 420), thickness=8)
    rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((210, 80), 13, 1.0), (420, 160), borderValue=200)
    a, b = cr.calculate_phi_7_roughness(img), cr.calculate_phi_7_roughness(rot)
    assert abs(a["relative_mean"] - b["relative_mean"]) < 0.01


def test_specks_are_filtered_and_single_glyph_works():
    img = disk(50)
    rng = np.random.default_rng(0)
    for _ in range(60):                                    # 녹 반점
        cv2.circle(img, (int(rng.integers(0, 200)), int(rng.integers(0, 200))), 1, 60, -1)
    r = cr.calculate_phi_7_roughness(img)
    assert r["num_valid_contours"] == 1 and r["phi_7"] < 0.1


def test_isoperimetric_score_option_matches_formula():
    r = cr.calculate_phi_7_roughness(of.render_printed("B12", thickness=8), score="isoperimetric")
    w = np.array(r["areas"])
    rbar = float(np.sum(w * np.array(r["roughness_values"])) / w.sum())
    assert r["raw_roughness"] == pytest.approx(rbar)
    assert r["phi_7"] == pytest.approx(1 - math.exp(-cr.GAMMA_ISO * max(0, rbar - cr.R0_ISO)))


@pytest.mark.parametrize("img", [np.full((80, 80), 200, np.uint8), np.full((80, 80), 0, np.uint8)])
def test_empty_returns_none(img):
    r = cr.calculate_phi_7_roughness(img)
    assert r["phi_7"] is None and r["reason"]


def test_debug_marks_rough_segments(tmp_path):
    img = cr.roughen_edges(of.render_printed("B12", size=(140, 260), thickness=8), 0.6, seed=7)
    path = tmp_path / "거칠기.png"
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))
    r = cr.calculate_phi_7_roughness(str(path))
    vis = cr.draw_roughness_debug(str(path), r, scale=1.0)
    b, g, rr = cv2.split(vis.astype(int))
    assert ((g > 200) & (rr < 80) & (b < 80)).any()       # 매끈한 구간 = 초록
    assert ((rr > 200) & (b < 80)).any()                  # 거친 구간 = 노랑/빨강
