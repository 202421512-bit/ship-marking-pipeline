"""baseline_feature.py φ₄ (기준선 흔들림) 검증."""
import math

import cv2
import numpy as np
import pytest

import baseline_feature as bf
import orientation_feature as of


STROKE = 6


def boxes(bottoms, size=(160, 560), w=34, h=50, angle=0.0):
    """바닥 y 를 지정한 'ㅁ' 모양 글자들 (획 굵기 STROKE). angle 이면 줄 전체 회전."""
    img = np.full(size, 210, np.uint8)
    for i, yb in enumerate(bottoms):
        x = 30 + i * 55
        cv2.rectangle(img, (x + STROKE // 2, yb - h + STROKE // 2), (x + w - STROKE // 2, yb - STROKE // 2),
                      40, STROKE)
    M = cv2.getRotationMatrix2D((size[1] / 2, size[0] / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (size[1], size[0]), borderValue=210)


@pytest.mark.parametrize("fitter", sorted(bf.FITTERS))
def test_fitters_recover_line(fitter):
    x = np.arange(10, dtype=float) * 30
    y = 0.2 * x + 50
    a, b = bf.fit_ransac(x, y, 1.0) if fitter == "ransac" else bf.FITTERS[fitter](x, y)
    assert a == pytest.approx(0.2, abs=1e-6) and b == pytest.approx(50, abs=1e-4)


@pytest.mark.parametrize("fitter", ["theil_sen", "ransac", "huber"])
def test_robust_fitters_ignore_one_outlier(fitter):
    x = np.arange(9, dtype=float) * 30
    y = np.full_like(x, 100.0)
    y[4] = 60                       # 한 점만 크게 벗어남
    a, b = bf.fit_ransac(x, y, 2.0) if fitter == "ransac" else bf.FITTERS[fitter](x, y)
    assert abs(a) < 0.01 and abs(b - 100) < 1.0
    a_ols, _ = bf.fit_ols(x, y)
    assert abs(a_ols) < 0.01      # 대칭 위치라 기울기는 같지만
    assert abs(bf.fit_ols(x, y)[1] - 100) > 3   # OLS 절편은 끌려감


def test_aligned_bottoms_give_zero():
    res = bf.calculate_phi_4_baseline(boxes([120] * 7))
    assert res["num_valid_chars"] == 7
    assert res["rmse_px"] < 0.5 and res["phi_4"] < 0.1
    assert abs(res["slope_a"]) < 0.005 and abs(res["intercept_b"] - 120) < 1.5


def test_known_jitter_gives_known_rmse():
    offsets = [0, 6, -6, 6, -6, 6, -6]
    res = bf.calculate_phi_4_baseline(boxes([120 + o for o in offsets]))
    expected = math.sqrt(np.mean(np.square(np.array(offsets) - np.mean(offsets))))
    assert res["rmse_px"] == pytest.approx(expected, abs=0.8)
    assert res["phi_4"] > 0.9


@pytest.mark.parametrize("angle", [0, 8, -15])
def test_line_rotation_does_not_count_as_fluctuation(angle):
    res = bf.calculate_phi_4_baseline(boxes([120] * 7, angle=angle))
    assert res["rmse_norm"] < 0.01
    assert math.degrees(math.atan(res["slope_a"])) == pytest.approx(-angle, abs=1.0)


def test_scale_invariance():
    small = boxes([120 + o for o in [0, 4, -4, 4, -4, 0]])
    big = cv2.resize(small, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    a, b = bf.calculate_phi_4_baseline(small), bf.calculate_phi_4_baseline(big)
    assert b["rmse_px"] == pytest.approx(2 * a["rmse_px"], rel=0.15)
    assert b["phi_4"] == pytest.approx(a["phi_4"], abs=0.05)


@pytest.mark.parametrize("drip_y", [85, 104])   # 글자에 붙은 / 떨어진 흘러내림
def test_drip_does_not_move_baseline(drip_y):
    clean = of.render_printed("B12-SP3 4500")
    dripped = bf.add_drip(clean, 40, drip_y, length=30)
    a = bf.calculate_phi_4_baseline(of.add_plate_noise(clean))
    b = bf.calculate_phi_4_baseline(of.add_plate_noise(dripped))
    assert b["rmse_norm"] < 0.02
    assert abs(b["rmse_norm"] - a["rmse_norm"]) < 0.01


def test_punctuation_and_descender_excluded():
    img = boxes([120] * 6)
    cv2.rectangle(img, (30 + 6 * 55, 112), (30 + 6 * 55 + 20, 120), 40, -1)     # '-' 처럼 낮은 성분
    cv2.rectangle(img, (30 + 7 * 55 + 3, 73), (30 + 7 * 55 + 27, 142), 40, STROKE)  # 'p' 처럼 아래로 긴 글자
    res = bf.calculate_phi_4_baseline(img)
    reasons = {c["reason"] for c in res["components"]}
    assert {"punctuation", "descender"} <= reasons
    assert res["num_valid_chars"] == 6 and res["rmse_norm"] < 0.01


def test_broken_character_is_merged():
    """위아래로 끊긴 글자(도트/스텐실)는 한 글자로 합쳐져 위 조각이 이상치가 되지 않아야 한다."""
    img = boxes([120] * 6)
    x = 30 + 2 * 55
    img[90:96, x:x + 35] = 210                     # 세 번째 글자의 양 옆 획을 끊어 위아래 두 조각으로
    res = bf.calculate_phi_4_baseline(img)
    assert res["num_valid_chars"] == 6 and res["rmse_norm"] < 0.01


def test_printed_vs_handwritten():
    p = bf.calculate_phi_4_baseline(of.add_plate_noise(of.render_printed("HK357 B12", angle=6), seed=1))
    h = bf.calculate_phi_4_baseline(of.add_plate_noise(
        of.render_handwritten("HK357 B12", seed=4, tilt_std=5, baseline_jitter=5), seed=2))
    assert p["phi_4"] < 0.5 < h["phi_4"]


@pytest.mark.parametrize("img", [np.full((80, 200), 200, np.uint8), boxes([120])])
def test_fewer_than_two_chars(img):
    res = bf.calculate_phi_4_baseline(img)
    assert res["phi_4"] is None and res["reason"]


def test_debug_overlay_and_korean_path(tmp_path):
    img = boxes([120, 126, 116, 122])
    path = tmp_path / "기준선.png"
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))
    res = bf.calculate_phi_4_baseline(str(path))
    vis = bf.draw_baseline_debug(str(path), res, scale=1.0)
    assert vis.shape == (*img.shape, 3)
    assert ((vis[..., 2] > 200) & (vis[..., 1] < 60)).any()     # 빨간 편차선
