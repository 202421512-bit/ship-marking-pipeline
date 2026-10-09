"""orientation_feature.py φ₃ (기울기 변화) 검증."""
import math

import cv2
import numpy as np
import pytest

import orientation_feature as of


def bars(angles_deg, size=(160, 600), length=60, width=8):
    """각도가 지정된 막대 '글자' 들 (어두운 획, 밝은 바탕)."""
    img = np.full(size, 210, np.uint8)
    for i, a in enumerate(angles_deg):
        cx, cy = 60 + i * 80, size[0] // 2
        t = math.radians(a)
        dx, dy = length / 2 * math.cos(t), length / 2 * math.sin(t)
        cv2.line(img, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 40, width, cv2.LINE_AA)
    return img


def test_moment_angle_matches_drawn_angle():
    res = of.calculate_phi_3_orientation(bars([20, 20, 20, 20]))
    assert res["num_valid_chars"] == 4
    assert all(abs(t - 20) < 2 for t in res["thetas"])
    assert res["sigma_theta_deg"] < 1 and res["phi_3"] < 0.1


def test_axial_wraparound_near_vertical():
    """+88° 와 −88° 는 4° 차이인 거의 같은 축: 일반 표준편차(88°)가 아니라 작아야 한다."""
    res = of.calculate_phi_3_orientation(bars([88, -88, 88, -88]), spread="circular")
    assert res["sigma_theta_deg"] < 5


@pytest.mark.parametrize("spread", ["circular", "robust"])
def test_scattered_angles_score_higher(spread):
    calm = of.calculate_phi_3_orientation(bars([80, 82, 79, 81, 80, 82]), spread=spread)
    wild = of.calculate_phi_3_orientation(bars([60, 95, 70, 110, 85, 65]), spread=spread)
    assert calm["phi_3"] < 0.2 < 0.6 < wild["phi_3"]


def test_robust_spread_ignores_single_shape_outlier():
    """인쇄체의 'S' 처럼 한 글자만 주축이 크게 다른 경우: robust 가 circular 보다 훨씬 덜 반응."""
    angles = [90, 90, 90, 90, 90, 90, 35]
    circ = of.calculate_phi_3_orientation(bars(angles), spread="circular")
    rob = of.calculate_phi_3_orientation(bars(angles), spread="robust")
    assert rob["sigma_theta_deg"] < 2 < 10 < circ["sigma_theta_deg"]


def test_line_rotation_does_not_change_spread():
    """줄 전체가 기운 것은 글자 간 편차가 아니다 (전체 회전에 불변)."""
    a = of.calculate_phi_3_orientation(of.render_printed("B12-SP3 4500"))
    b = of.calculate_phi_3_orientation(of.render_printed("B12-SP3 4500", angle=15))
    assert abs(a["mean_theta_deg"] % 180 - (b["mean_theta_deg"] + 15) % 180) < 4
    assert abs(a["sigma_theta_deg"] - b["sigma_theta_deg"]) < 3


def test_round_shapes_are_excluded():
    img = np.full((160, 500), 210, np.uint8)
    for i in range(4):
        cv2.circle(img, (60 + i * 110, 80), 35, 40, 8)
    res = of.calculate_phi_3_orientation(img)
    assert res["num_valid_chars"] == 0 and res["phi_3"] is None
    assert all(c["reason"] == "low_eccentricity" for c in res["components"])


def test_fewer_than_two_chars_returns_fallback():
    one = bars([30])
    assert of.calculate_phi_3_orientation(one)["phi_3"] is None
    assert of.calculate_phi_3_orientation(one, fallback=0.5)["phi_3"] == 0.5
    blank = np.full((100, 100), 200, np.uint8)
    res = of.calculate_phi_3_orientation(blank)
    assert res["phi_3"] is None and res["num_valid_chars"] == 0


def test_noise_and_scratch_do_not_change_char_count():
    clean = of.render_printed("B12-SP3 4500")
    noisy = of.add_plate_noise(clean, seed=1)
    a = of.calculate_phi_3_orientation(clean)
    b = of.calculate_phi_3_orientation(noisy)
    assert b["num_valid_chars"] == a["num_valid_chars"]
    assert abs(b["sigma_theta_deg"] - a["sigma_theta_deg"]) < 3


def test_light_strokes_on_dark_background():
    img = bars([20, 25, 15])
    a = of.calculate_phi_3_orientation(img)
    b = of.calculate_phi_3_orientation(255 - img)
    assert a["num_valid_chars"] == b["num_valid_chars"] == 3
    assert a["sigma_theta_deg"] == pytest.approx(b["sigma_theta_deg"], abs=0.5)


def test_reads_korean_path(tmp_path):
    path = tmp_path / "강판_크롭.png"
    ok, buf = cv2.imencode(".png", bars([10, 12, 14]))
    buf.tofile(str(path))
    res = of.calculate_phi_3_orientation(str(path))
    assert res["num_valid_chars"] == 3


def test_debug_overlay_draws_on_image():
    img = bars([10, 40, 70])
    res = of.calculate_phi_3_orientation(img)
    vis = of.draw_orientation_debug(img, res, scale=1.0)
    assert vis.shape == (*img.shape, 3)
    assert (vis[..., 1] > vis[..., 0] + 100).any()   # 초록 화살표가 그려졌는지
