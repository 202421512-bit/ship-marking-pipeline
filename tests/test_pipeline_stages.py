"""기계 패턴 사전 판별, 색상 전처리, 2단계 계층형 목적함수 검증."""
import cv2
import numpy as np
import pytest

import color_preprocess as cp
import mechanical_pattern_detector as md
import objective_function as of
import synthetic_dataset as sd
from orientation_feature import _remove_specks, binarize


# --- 1단계: 기계 패턴 ----------------------------------------------------------
def dot_grid(rows=7, cols=40, pitch=8, radius=3, angle=0.0):
    img = np.full((rows * pitch + 60, cols * pitch + 60), 200, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (c % 6) < 5:                       # 6칸마다 한 칸 비움 = 글자 사이
                cv2.circle(img, (30 + c * pitch, 30 + r * pitch), radius, 40, -1, cv2.LINE_AA)
    h, w = img.shape
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), (w, h), borderValue=200)


@pytest.mark.parametrize("radius", [3, 4])        # 4 이면 피치 8 에서 도트끼리 닿음
@pytest.mark.parametrize("angle", [0, 12])
def test_dot_grid_detected_even_when_dots_touch(radius, angle):
    r = md.detect_mechanical_pattern(dot_grid(radius=radius, angle=angle))
    assert r["is_dot_matrix"] and r["kind"] == "dot_matrix"
    assert r["dot"]["pitch"] == pytest.approx(8, abs=1.0)
    assert r["dot"]["radius"] == pytest.approx(radius, abs=1.2)


def test_solid_strokes_and_random_specks_are_not_dots():
    solid = np.full((140, 500), 200, np.uint8)
    cv2.putText(solid, "B12SP3 4500", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 40, 7, cv2.LINE_AA)
    assert not md.detect_mechanical_pattern(solid)["is_dot_matrix"]
    rng = np.random.default_rng(0)
    rust = np.full((140, 500), 200, np.uint8)
    for _ in range(300):
        cv2.circle(rust, (int(rng.integers(0, 500)), int(rng.integers(0, 140))), int(rng.integers(1, 4)), 60, -1)
    assert not md.detect_mechanical_pattern(rust)["is_mechanical"]


def test_stencil_detected_and_handwriting_not():
    rng = np.random.default_rng(3)
    hits = sum(md.detect_mechanical_pattern(sd.printed_stencil(sd.random_text(rng, 7, 9), rng))["is_stencil"]
               for _ in range(6))
    assert hits >= 4
    hand_hits = sum(md.detect_mechanical_pattern(sd.make_sample(1, rng)[0])["is_mechanical"] for _ in range(10))
    assert hand_hits == 0


def test_no_ink():
    r = md.detect_mechanical_pattern(np.full((50, 50), 200, np.uint8))
    assert not r["is_mechanical"]


# --- 0단계: 색상 전처리 --------------------------------------------------------
def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_color_path_handles_rust_with_black_ink():
    """녹이 많아도 색상 전처리 경로가 잉크 마스크를 지켜야 한다 (회색 이진화와 큰 차이 없이)."""
    rng = np.random.default_rng(5)
    gray_iou, color_iou = [], []
    for i in range(8):
        base = sd.printed_sans(sd.random_text(rng), rng) if i % 2 else sd.handwritten(sd.random_text(rng), rng)
        img, truth = sd.colorize(base, "black", rng, rust_patches=(8, 12))
        gray_iou.append(_iou(_remove_specks(binarize(img)), truth))
        color_iou.append(_iou(_remove_specks(binarize(cp.preprocess_color(img)["enhanced"])), truth))
    assert np.mean(color_iou) > 0.85
    assert np.mean(color_iou) >= np.mean(gray_iou) - 0.05


@pytest.mark.parametrize("marking", ["yellow", "white"])
def test_light_markings_recovered(marking):
    rng = np.random.default_rng(12)
    img, truth = sd.colorize(sd.printed_sans("AB12CD", rng), marking, rng)
    pre = cp.preprocess_color(img)
    assert pre["is_color"]
    assert _iou(_remove_specks(binarize(pre["enhanced"])), truth) > 0.8


def test_grayscale_passthrough():
    g = np.full((60, 60), 180, np.uint8)
    pre = cp.preprocess_color(g)
    assert not pre["is_color"] and np.array_equal(pre["enhanced"], g)


# --- 2단계 계층형 목적함수 -------------------------------------------------------
def test_dot_matrix_exits_early_as_printed():
    r = of.evaluate_handwritten_score(dot_grid(radius=3))
    assert r["stage"] == 1 and r["verdict"] == "Confirmed Printed" and r["score"] == 0.0
    assert r["mechanical"]["kind"] == "dot_matrix"
    assert all(v is None for v in r["features"].values())


def test_early_exit_can_be_disabled():
    r = of.evaluate_handwritten_score(dot_grid(radius=3), early_exit=False)
    assert r["stage"] == 2 and r["score"] is not None


def test_handwriting_goes_to_stage_two_with_color_input():
    rng = np.random.default_rng(13)
    img, _ = sd.colorize(sd.handwritten("HK357B", rng), "yellow", rng)
    r = of.evaluate_handwritten_score(img, color="auto")
    assert r["stage"] == 2 and r["preprocess"]["is_color"]
    assert 0 <= r["score"] <= 1 and 0 <= r["confidence"] <= 1
    assert set(r["confidence_parts"]) >= {"contrast", "mask", "coverage"}
    assert "glare" not in r["confidence_parts"]


def test_rust_near_text_lowers_reference_confidence():
    rng = np.random.default_rng(14)
    base = sd.printed_sans("AB12CD", rng)
    clean, _ = sd.colorize(base, "black", np.random.default_rng(1), rust_patches=(0, 1))
    rusty, _ = sd.colorize(base, "black", np.random.default_rng(1), rust_patches=(14, 16))
    a = of.evaluate_handwritten_score(clean, color="auto")
    b = of.evaluate_handwritten_score(rusty, color="auto")
    assert b["confidence_parts"]["rust_near_text"] > a["confidence_parts"]["rust_near_text"]
    assert b["confidence_parts"]["mask"] < a["confidence_parts"]["mask"]


def test_debug_view_shows_stage():
    r = of.evaluate_handwritten_score(dot_grid(radius=3))
    vis = of.draw_objective_debug(dot_grid(radius=3), r)
    assert vis.ndim == 3
