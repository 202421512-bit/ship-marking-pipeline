"""글자 유착(Touching/Overlapping) 검출, φ₆ 오염 방지, 목적함수 가산점 검증."""
import cv2
import numpy as np
import pytest

import objective_function as of
import synthetic_dataset as sd
from size_variance_feature import calculate_phi_6_size_variance as phi6
from touching_detector import detect_touching_chars, split_touching_mask

from tests.test_pipeline_stages import dot_grid


def hershey_line(text, advance, scale=2.2, thick=6, size=(140, 520)):
    """글자를 advance(글자 폭 대비 전진 비율)로 놓는다. 1 미만이면 서로 겹친다."""
    img = np.full(size, 200, np.uint8)
    x = 30
    for ch in text:
        (w, _), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, ch, (x, 100), cv2.FONT_HERSHEY_SIMPLEX, scale, 40, thick, cv2.LINE_AA)
        x += int(w * advance) if ch != " " else w
    return img


def touching_hand(seed, text_len=(5, 8)):
    rng = np.random.default_rng(seed)
    return sd.field_noise(sd.handwritten(sd.random_text(rng, *text_len), rng, link_prob=0.0, touch_prob=1.0), rng)


# --- 검출기 ----------------------------------------------------------------------
@pytest.mark.parametrize("text,advance", [("HKE", 0.6), ("HKE", 0.7), ("B12 77", 0.6)])
def test_overlapping_letters_are_flagged(text, advance):
    d = detect_touching_chars(hershey_line(text, advance))
    assert d["has_touching_chars"] and d["num_touching"] >= 1
    c = d["components"][0]
    assert c["ratio"] > 1.8 and c["est_chars"] >= 2 and c["kind"] in ("fused", "bridged")
    assert d["est_joints"] >= 1


def test_separated_letters_are_not_flagged():
    assert not detect_touching_chars(hershey_line("HK357B", 1.25))["has_touching_chars"]


def test_printed_samples_have_no_touching():
    rng = np.random.default_rng(1)
    for _ in range(15):
        img = sd.field_noise(sd.printed_sans(sd.random_text(rng), rng), rng)
        assert not detect_touching_chars(img)["has_touching_chars"]


def test_dash_underline_and_plate_edge_are_not_touching():
    img = hershey_line("H K", 1.25)
    cv2.line(img, (30, 125), (480, 125), 40, 4)           # 글자 높이에 못 미치는 밑줄
    assert not detect_touching_chars(img)["has_touching_chars"]
    edge = hershey_line("HK357", 1.25)
    edge[:, :6] = 40                                      # 판 가장자리 (경계에 닿는 긴 성분)
    edge[:6, :] = 40
    assert not detect_touching_chars(edge)["has_touching_chars"]


def test_no_ink_and_blank_do_not_crash():
    d = detect_touching_chars(np.full((60, 60), 200, np.uint8))
    assert not d["has_touching_chars"] and d["reason"]


# --- φ₆ 오염 방지 ----------------------------------------------------------------
def test_split_cuts_blob_into_characters():
    img = hershey_line("HKE 357", 0.62)
    mask_info = detect_touching_chars(img)
    assert mask_info["has_touching_chars"]
    from orientation_feature import _remove_specks, binarize
    mask = _remove_specks(binarize(img))
    cut, n = split_touching_mask(mask, detect_touching_chars(mask))
    assert n >= 1 and cut.sum() < mask.sum()
    n_before = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0]
    n_after = cv2.connectedComponents(cut.astype(np.uint8), connectivity=8)[0]
    assert n_after > n_before


def mixed_line(text, joined):
    """joined 에 든 위치의 글자만 다음 글자와 겹치고(advance 0.6) 나머지는 분리(1.25)."""
    img = np.full((140, 520), 200, np.uint8)
    x = 30
    for i, ch in enumerate(text):
        (w, _), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 6)
        cv2.putText(img, ch, (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 40, 6, cv2.LINE_AA)
        x += int(w * (0.6 if i in joined else 1.25))
    return img


def test_phi6_not_inflated_by_touching_blob():
    """정상 글자 사이에 유착 덩어리(3글자) 하나가 끼면 CV_A 가 크게 부푼다. 분할 추정하면 분리본에 가까워야 한다."""
    text = "HKE3579"
    touching, separate = mixed_line(text, {0, 1}), mixed_line(text, set())
    assert detect_touching_chars(touching)["num_touching"] == 1
    ref = phi6(separate, robust="none")
    with_split = phi6(touching, robust="none")
    no_split = phi6(touching, robust="none", split_touching=False)
    assert ref["phi_6"] is not None and with_split["phi_6"] is not None and no_split["phi_6"] is not None
    assert no_split["cv_area"] > 0.2                                     # 오염: 덩어리가 평균을 왜곡
    assert abs(with_split["cv_area"] - ref["cv_area"]) < abs(no_split["cv_area"] - ref["cv_area"])
    assert with_split["num_valid_chars"] > no_split["num_valid_chars"]


# --- 목적함수 --------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(300, 312))
def test_touching_handwriting_is_judged_handwritten_without_error(seed):
    r = of.evaluate_handwritten_score(touching_hand(seed))
    assert r["stage"] == 2 and r["score"] is not None and 0 <= r["score"] <= 1
    if r["has_touching_chars"]:
        assert "TOUCHING_CHARS" in r["flags"]
        assert r["score"] > r["score_before_touch"] and r["touch_boost"] > 0
        assert r["verdict"] == "Confirmed Handwritten"


def test_touching_samples_are_actually_detected():
    hits = sum(of.evaluate_handwritten_score(touching_hand(s))["has_touching_chars"] for s in range(300, 312))
    assert hits >= 4


def test_touching_hershey_text_is_boosted_to_handwritten():
    """획 모양만 보면 인쇄체처럼 규칙적인 글자도, 서로 겹쳤다는 사실만으로 수기 쪽으로 끌려간다."""
    img = hershey_line("HKE 357", 0.62)
    r = of.evaluate_handwritten_score(img)
    assert r["has_touching_chars"] and r["stage"] == 2
    assert r["score"] > r["score_before_touch"]
    assert r["verdict"] == "Confirmed Handwritten"


def test_no_boost_without_touching():
    r = of.evaluate_handwritten_score(hershey_line("HK357B", 1.25))
    assert not r["has_touching_chars"] and r["touch_boost"] == 0.0
    assert r["score"] == r["score_before_touch"]


def test_mechanical_pattern_takes_precedence_over_touching():
    """도트 매트릭스는 도트가 닿아 있어도 1단계에서 인쇄체로 확정 (가산점 없음)."""
    r = of.evaluate_handwritten_score(dot_grid(radius=4))
    assert r["stage"] == 1 and r["verdict"] == "Confirmed Printed"
    assert not r["has_touching_chars"]


def test_apply_touching_boost_math():
    fused = [{"kind": "fused"}]
    none = {"has_touching_chars": False, "est_joints": 0, "components": []}
    one = {"has_touching_chars": True, "est_joints": 1, "components": fused}
    many = {"has_touching_chars": True, "est_joints": 2, "components": fused}
    bridged = {"has_touching_chars": True, "est_joints": 1, "components": [{"kind": "bridged"}]}
    assert of.apply_touching_boost(0.2, none) == (0.2, 0.0)
    assert of.apply_touching_boost(0.2, one)[0] == pytest.approx(0.2 + 0.5 * 0.8)
    assert of.apply_touching_boost(0.2, many)[0] == pytest.approx(0.2 + 0.7 * 0.8)
    assert of.apply_touching_boost(0.2, bridged)[0] == pytest.approx(0.2 + 0.3 * 0.8)
    assert of.apply_touching_boost(None, one)[0] == pytest.approx(0.75)      # 지표를 못 구해도 유착 근거로 판정 가능
