"""pipeline_demo.py 분류 파이프라인 검증 (합성 샘플 기반)."""
import cv2
import numpy as np
import pytest

import pipeline_demo as p


def labels_of(img):
    rois, _, _, _ = p.run_pipeline(img)
    return [r.label for r in rois]


def test_sample_plate_classification():
    labels = labels_of(p.make_sample_plate())
    assert labels.count(p.STRUCTURED) == 2
    assert labels.count(p.UNSTRUCTURED) == 3
    assert labels.count(p.SYMBOL) == 3


@pytest.mark.parametrize("tilt", [6.0, -10.0])
def test_deskew_levels_tilted_plate(tilt):
    deskewed, _, valid, theta = p.deskew(p.make_sample_plate(tilt_deg=tilt))
    assert abs(theta) > p.DESKEW_MIN_ANGLE
    residual = p.estimate_skew(p.foreground_mask(deskewed, valid))
    assert abs(residual) < 1.0


def test_small_tilt_is_not_rotated():
    img = p.make_sample_plate(tilt_deg=2.0)
    deskewed, M, _, theta = p.deskew(img)
    assert abs(theta) <= p.DESKEW_MIN_ANGLE
    assert deskewed is img
    assert np.allclose(M, [[1, 0, 0], [0, 1, 0]])


def test_baseline_std_aligned_vs_jittered():
    aligned = [(i * 30, 10, 20, 40) for i in range(6)]
    jittered = [(i * 30, 10 + dy, 20, 40) for i, dy in enumerate([0, 7, -6, 8, -7, 5])]
    assert p.baseline_std(aligned) <= p.BASELINE_STD_MAX
    assert p.baseline_std(jittered) > p.BASELINE_STD_MAX
    assert p.baseline_std(aligned[:2]) == float("inf")  # 문자 수 부족 -> 정형 확정 불가


def test_stroke_width_uniform_vs_varying():
    uniform = np.zeros((60, 200), np.uint8)
    cv2.line(uniform, (10, 30), (190, 30), 255, 6)
    varying = np.zeros((60, 200), np.uint8)
    for x, t in zip(range(10, 190, 30), [2, 12, 3, 14, 2, 10]):
        cv2.line(varying, (x, 30), (x + 25, 30), 255, t)
    assert p.stroke_width_cv(uniform) <= p.STROKE_CV_MAX
    assert p.stroke_width_cv(varying) > p.STROKE_CV_MAX


def test_dark_pixels_count_as_achromatic():
    img = np.zeros((4, 4, 3), np.uint8)
    img[:2] = (20, 25, 35)    # 거의 검정이지만 HSV S 는 높게 나오는 픽셀
    img[2:] = (0, 200, 255)   # 노란 마커
    sat = p.effective_saturation(img)
    assert sat[:2].max() == 0
    assert sat[2:].min() > p.SAT_MAX
