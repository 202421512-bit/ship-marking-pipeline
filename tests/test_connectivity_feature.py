"""connectivity_feature.py φ₈ (획 연결성·분기점 밀도) 검증."""
import cv2
import numpy as np
import pytest

import connectivity_feature as cf
import contour_roughness_feature as cr
import orientation_feature as of


def canvas(size=(200, 200)):
    return np.full(size, 210, np.uint8)


def test_crossing_number_basic_patterns():
    skel = np.zeros((9, 9), bool)
    skel[4, 1:8] = True                    # 가로 선
    skel[1:4, 4] = True                    # 위로 뻗은 가지 -> T 자
    cn = cf.crossing_number(skel)
    assert cn[4, 1] == 1 and cn[4, 7] == 1 and cn[1, 4] == 1   # 끝점
    assert cn[4, 2] == 2                                        # 단순 연결점
    assert cn[4, 4] == 3                                        # 분기점
    cross = np.zeros((9, 9), bool)
    cross[4, 1:8] = cross[1:8, 4] = True
    assert cf.crossing_number(cross)[4, 4] == 4


def test_cluster_points_merges_nearby_pixels():
    m = np.zeros((40, 40), bool)
    m[10, 10] = m[11, 11] = m[10, 12] = True           # 한 교차점이 여러 픽셀에 걸침
    m[30, 30] = True
    assert len(cf.cluster_points(m, 3)) == 2


def test_plus_t_and_ring():
    plus = canvas()
    cv2.line(plus, (40, 100), (160, 100), 40, 8)
    cv2.line(plus, (100, 40), (100, 160), 40, 8)
    r = cf.calculate_phi_8_connectivity(plus)
    assert r["num_junctions"] == 1 and r["num_endpoints"] == 4

    tee = canvas()
    cv2.line(tee, (40, 60), (160, 60), 40, 8)
    cv2.line(tee, (100, 60), (100, 160), 40, 8)
    r = cf.calculate_phi_8_connectivity(tee)
    assert r["num_junctions"] == 1 and r["num_endpoints"] == 3

    ring = canvas()
    cv2.circle(ring, (100, 100), 50, 40, 8)
    r = cf.calculate_phi_8_connectivity(ring)
    assert r["num_junctions"] == 0 and r["num_endpoints"] == 0 and r["num_loops"] == 1


def test_rough_edges_do_not_create_junctions():
    """거친 테두리의 털 모양 잔가지는 가지치기로 사라져야 한다."""
    img = canvas((160, 400))
    cv2.line(img, (40, 80), (360, 80), 40, 12)
    rough = cr.roughen_edges(img, 0.6, seed=3, background=210)
    r = cf.calculate_phi_8_connectivity(rough)
    assert r["num_junctions"] <= 1 and r["num_endpoints"] <= 3


def test_density_is_scale_invariant_per_height():
    img = of.render_printed("A4B8", size=(140, 300))
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    a, b = cf.calculate_phi_8_connectivity(img), cf.calculate_phi_8_connectivity(big)
    assert b["junction_density"] == pytest.approx(a["junction_density"] / 2, rel=0.3)   # 픽셀 기준은 반으로
    assert b["junction_density_per_height"] == pytest.approx(a["junction_density_per_height"], rel=0.25)


def test_ligatures_detected_and_scored():
    printed = cf.calculate_phi_8_connectivity(of.add_plate_noise(of.render_printed("A4B8 R4A8"), seed=1))
    joined = cf.calculate_phi_8_connectivity(of.add_plate_noise(cf.render_connected("A4B8 R4A8", seed=2), seed=2))
    assert printed["num_ligatures"] == 0 and printed["phi_8"] < 0.1
    assert joined["num_ligatures"] >= 3 and joined["phi_8"] > 0.8


def test_score_options():
    img = cf.render_connected("HK357 B12", seed=4)
    d = cf.calculate_phi_8_connectivity(img, score="density")
    lg = cf.calculate_phi_8_connectivity(img, score="ligature")
    cb = cf.calculate_phi_8_connectivity(img, score="combined")
    assert d["phi_8"] == pytest.approx(np.tanh(cf.ALPHA8 * d["junction_density_per_height"]))
    assert cb["phi_8"] == pytest.approx(1 - (1 - d["phi_8"]) * (1 - lg["phi_8"]))


@pytest.mark.parametrize("img", [np.full((80, 80), 200, np.uint8), np.full((80, 80), 0, np.uint8)])
def test_empty_returns_none(img):
    r = cf.calculate_phi_8_connectivity(img)
    assert r["phi_8"] is None and r["reason"]


def test_debug_overlay_colors(tmp_path):
    img = canvas()
    cv2.line(img, (40, 60), (160, 60), 40, 8)
    cv2.line(img, (100, 60), (100, 160), 40, 8)
    path = tmp_path / "연결.png"
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))
    r = cf.calculate_phi_8_connectivity(str(path))
    vis = cf.draw_connectivity_debug(str(path), r, scale=1.0)
    b, g, rr = cv2.split(vis.astype(int))
    assert ((b > 200) & (g < 120) & (rr < 60)).any()     # 골격 = 파랑
    assert ((g > 150) & (b < 60) & (rr < 60)).any()      # 끝점 = 초록
    assert ((rr > 200) & (g < 60) & (b < 60)).any()      # 분기점 = 빨강
