"""curvature_feature.py φ₅ (곡률 변화 에너지) 검증."""
import cv2
import numpy as np
import pytest

import curvature_feature as cf
import orientation_feature as of


def circle_img(radius=40, thickness=6, size=200, scale=1):
    img = np.full((size * scale, size * scale), 200, np.uint8)
    c = size * scale // 2
    cv2.circle(img, (c, c), radius * scale, 40, thickness * scale, cv2.LINE_AA)
    return img


def test_curvature_of_resampled_circle_is_constant():
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    pts = cf.resample(np.column_stack([100 + 50 * np.cos(t), 100 + 50 * np.sin(t)]), True)
    kappa, dkds, valid = cf.curvature_profile(pts, True, scale=50.0, window=15)
    assert np.allclose(np.abs(kappa), 1.0, atol=0.02)      # 반지름 = 기준 길이 -> κ = 1
    assert np.abs(dkds).max() < 0.5


def test_trace_skeleton_orders_a_line():
    skel = np.zeros((50, 120), bool)
    skel[25, 10:110] = True
    (path, closed), = cf.trace_skeleton(skel)
    assert not closed and len(path) == 100
    assert np.all(np.diff(path[:, 0]) != 0)                   # x 가 한 방향으로 순서대로


@pytest.mark.parametrize("curve", ["contour", "skeleton"])
def test_smooth_circle_has_low_energy(curve):
    r = cf.calculate_phi_5_curvature(circle_img(), curve=curve)
    assert r["num_valid_strokes"] >= 1
    assert r["phi_5"] < 0.3


@pytest.mark.parametrize("curve", ["contour", "skeleton"])
def test_tremor_raises_energy(curve):
    printed = of.add_plate_noise(cf.render_glyph_line(seed=1), seed=1)
    shaky = of.add_plate_noise(cf.render_glyph_line(tremor_amp=1.0, tremor_wavelength=10, seed=1), seed=2)
    a = cf.calculate_phi_5_curvature(printed, curve=curve)
    b = cf.calculate_phi_5_curvature(shaky, curve=curve)
    assert b["raw_energy"] > 1.5 * a["raw_energy"]
    assert a["phi_5"] < 0.5 < b["phi_5"]


def test_scale_invariance():
    small = cf.render_glyph_line("OS8C", seed=3, size=(110, 300))
    big = cv2.resize(small, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    a, b = cf.calculate_phi_5_curvature(small), cf.calculate_phi_5_curvature(big)
    assert b["char_height"] == pytest.approx(2 * a["char_height"], rel=0.1)
    assert 0.5 < b["raw_energy"] / a["raw_energy"] < 2.0     # 무차원화 안 하면 1/16 이 됨


def test_rotation_invariance():
    img = cf.render_glyph_line("OS83", seed=4, size=(110, 300))
    M = cv2.getRotationMatrix2D((150, 55), 20, 1.0)
    rot = cv2.warpAffine(img, M, (300, 110), borderValue=200)
    a, b = cf.calculate_phi_5_curvature(img), cf.calculate_phi_5_curvature(rot)
    assert 0.5 < b["raw_energy"] / a["raw_energy"] < 2.0


def test_square_corners_do_not_explode():
    img = np.full((200, 200), 200, np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 40, 6)           # 직각 모서리 4개
    sq = cf.calculate_phi_5_curvature(img)
    circ = cf.calculate_phi_5_curvature(circle_img(50))
    assert sq["phi_5"] < 0.5
    assert sq["raw_energy"] < 5 * circ["raw_energy"] + 200
    assert any(s["corner"] is not None and s["corner"].any() for s in sq["strokes"] if s["valid"])


@pytest.mark.parametrize("img", [np.full((80, 80), 200, np.uint8), np.full((80, 80), 255, np.uint8)])
def test_empty_returns_none(img):
    r = cf.calculate_phi_5_curvature(img)
    assert r["phi_5"] is None and r["reason"]


def test_debug_heatmap_draws_color(tmp_path):
    img = cf.render_glyph_line("OS", seed=5, size=(110, 200))
    path = tmp_path / "곡률.png"
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))
    r = cf.calculate_phi_5_curvature(str(path))
    vis = cf.draw_curvature_debug(str(path), r, scale=1.0)
    assert vis.shape == (*img.shape, 3)
    b, g, rr = cv2.split(vis.astype(int))
    assert ((np.abs(b - g) > 80) | (np.abs(g - rr) > 80)).sum() > 100   # 컬러맵으로 칠한 궤적
