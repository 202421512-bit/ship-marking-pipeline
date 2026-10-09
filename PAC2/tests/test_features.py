"""Unit tests for the 7 features and candidate generation (synthetic inputs with known answers)."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2 import load_config  # noqa: E402
from pac2.candidates import candidates, ink_polarity  # noqa: E402
from pac2.features import extract, f_s_theta, components  # noqa: E402

CFG = load_config()
F = CFG["features"]


def boxes(heights, xs, y_bottom=80, w=10):
    m = np.zeros((100, 300), np.uint8)
    for h, x in zip(heights, xs):
        m[y_bottom - h:y_bottom, x:x + w] = 255
    return m


def test_uniform_bars_have_low_variation():
    f = extract(boxes([40, 40, 40, 40], [20, 60, 100, 140]), F)
    assert f["CV_h"] == pytest.approx(0.0, abs=1e-9)
    assert f["CV_A"] == pytest.approx(0.0, abs=1e-9)
    assert f["CV_g"] == pytest.approx(0.0, abs=1e-9)          # equal gaps
    assert f["B"] == pytest.approx(0.0, abs=1e-9)             # common baseline
    assert f["CV_w"] < 0.25


def test_irregular_layout_increases_values():
    reg = extract(boxes([40, 40, 40, 40], [20, 60, 100, 140]), F)
    irr = extract(boxes([20, 45, 30, 50], [20, 45, 110, 140]), F)
    assert irr["CV_h"] > reg["CV_h"] and irr["CV_g"] > reg["CV_g"] and irr["CV_A"] > reg["CV_A"]


def test_missing_not_zero():
    one = extract(boxes([40], [20]), F)
    for k in ("CV_h", "CV_g", "B", "CV_A", "S_theta"):
        assert math.isnan(one[k]) and one[f"{k}_missing"] == 1 and one[f"{k}_reason"]
    two = extract(boxes([40, 40], [20, 60]), F)
    assert math.isnan(two["CV_g"])                            # one gap -> CV undefined
    assert math.isnan(two["B"])                               # two points always fit exactly


def test_slant_wraps_180_degrees():
    m = np.zeros((200, 400), np.uint8)
    for x, ang in ((60, 89), (180, 91), (300, 90)):          # near-vertical strokes either side of 90 deg
        r = math.radians(ang)
        dx, dy = 40 * math.cos(r), -40 * math.sin(r)
        cv2.line(m, (int(x - dx), int(100 - dy)), (int(x + dx), int(100 + dy)), 255, 3)
    v, _ = f_s_theta(components(m, 1), F)
    assert v < 3.0                                            # not ~90 deg from a naive 0/180 jump


def test_contour_irregularity_jagged_above_smooth():
    smooth = np.zeros((200, 200), np.uint8)
    cv2.circle(smooth, (100, 100), 60, 255, -1)
    jag = smooth.copy()
    for a in range(0, 360, 10):
        r = math.radians(a)
        cv2.circle(jag, (int(100 + 62 * math.cos(r)), int(100 + 62 * math.sin(r))), 6, 255, -1)
    assert extract(jag, F)["R"] > extract(smooth, F)["R"]


def test_candidates_same_size_and_polarity_agnostic():
    dark = np.full((60, 160, 3), 200, np.uint8)
    cv2.putText(dark, "F6", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (120, 30, 20), 4)
    bright = 255 - dark
    assert ink_polarity(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)) == "dark"
    assert ink_polarity(cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)) == "bright"
    for img in (dark, bright):
        for name, m in candidates(img, CFG["candidates"]).items():
            assert m.shape == img.shape[:2], name
            ink = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) != cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[0, 0])
            hit = ((m > 0) & ink).sum() / max(1, (m > 0).sum())
            assert hit > 0.6, name                            # mask mostly lies on the drawn strokes
