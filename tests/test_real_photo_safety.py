"""실사진 대응 안전장치: 반사광 기각, 줄 분리, 줄 지표 최소 글자 수, 도트 조각 병합 조건, 단독 추론 스크립트."""
import cv2
import numpy as np
import pytest

import infer_single_image as inf
import layout
import objective_function as of
import synthetic_dataset as sd
from orientation_feature import _merge_fragments


def write_lines(lines, size=(420, 900), scale=2.0, thick=6):
    """여러 줄·두 열 손글씨풍 시트 (OpenCV 글꼴, 줄마다 다른 높이)."""
    img = np.full(size, 210, np.uint8)
    for text, (x, y) in lines:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 40, thick, cv2.LINE_AA)
    return img


# --- 반사광 기각 가드 ------------------------------------------------------------
def test_specular_over_text_is_rejected():
    rng = np.random.default_rng(1)
    img = sd.printed_sans("AB12CD", rng)
    blown = img.copy()
    blown[img > 150] = 255                                       # 획은 남고 글자 주변 바탕이 포화
    r = of.evaluate_handwritten_score(blown)
    assert r["stage"] == 0 and r["flags"][0] == "REJECTED_SPECULAR_NOISE"
    assert r["verdict"] == "Uncertain (Need Review)" and r["score"] is None


def test_specular_outside_text_is_ignored():
    rng = np.random.default_rng(2)
    img = sd.printed_sans("AB12CD", rng)
    padded = cv2.copyMakeBorder(img, 0, 0, 0, 400, cv2.BORDER_CONSTANT, value=210)
    padded[:, -150:] = 255                                       # 글자와 떨어진 곳의 포화
    r = of.evaluate_handwritten_score(padded)
    assert "REJECTED_SPECULAR_NOISE" not in r["flags"] and r["stage"] in (1, 2)


# --- 줄 분리 -----------------------------------------------------------------------
def test_layout_splits_rows_and_columns_and_drops_page_edge():
    img = write_lines([("ABCD", (40, 100)), ("WXYZ", (520, 100)), ("EFGH", (40, 230)), ("1234", (40, 360))])
    cv2.rectangle(img, (0, 0), (12, img.shape[0] - 1), 40, -1)  # 종이 가장자리
    lay = layout.analyze_layout(img)
    assert lay["removed_border"] >= 1
    assert len(lay["lines"]) == 4                                # 두 열은 같은 높이여도 다른 줄
    assert all(ln["num_components"] == 4 for ln in lay["lines"])


def test_two_char_lines_do_not_count_as_printed_evidence():
    """2글자 줄은 기준선 오차가 항상 0, 간격이 하나뿐 -> 줄 지표는 계산 불가(None) 여야 한다."""
    img = write_lines([("AB", (40, 100)), ("CD", (40, 230)), ("EF", (40, 360))])
    phis, raw = of.extract_features(img)
    for k in ("phi_2", "phi_4", "phi_6"):
        assert phis[k] is None and raw[k]["lines_used"] == 0


def test_multi_line_sheet_uses_every_line():
    img = write_lines([("ABCDE", (40, 100)), ("FGHKL", (40, 230)), ("MNPRS", (40, 360))])
    phis, raw = of.extract_features(img)
    assert raw["phi_4"]["lines_used"] == 3 and phis["phi_4"] is not None


# --- 도트 조각 병합 조건 ------------------------------------------------------------
def test_large_letters_are_not_merged_as_dot_fragments():
    """여러 줄 사진에서 글자가 영상 높이에 비해 작아도, 둥글고 꽉 찬 도트가 아니면 붙이지 않는다."""
    img = write_lines([("ABCD", (40, 100)), ("EFGH", (40, 230)), ("KLMN", (40, 360))])
    mask = img < 120
    merged, k = _merge_fragments(mask)
    assert k == 0 and np.array_equal(merged, mask)


def test_dot_fragments_are_still_merged():
    rng = np.random.default_rng(3)
    img = sd.printed_dots("AB12", rng)
    _, k = _merge_fragments(img < 120)
    assert k > 0


# --- 단독 추론 스크립트 -------------------------------------------------------------
def test_infer_script_runs_on_korean_path(tmp_path, capsys):
    rng = np.random.default_rng(4)
    img, _ = sd.colorize(sd.handwritten("HK357B12", rng), "black", rng)
    src = tmp_path / "현장_사진.jpg"
    ok, buf = cv2.imencode(".jpg", img)
    buf.tofile(str(src))
    r = of.evaluate_handwritten_score(inf.read_image(str(src)))
    inf.print_report(str(src), img, r)
    out = capsys.readouterr().out
    assert "[0단계]" in out and "최종 판정" in out
    vis = inf.compose(img, r)
    assert vis.ndim == 3 and vis.shape[1] == 2 * inf.PANEL_W


@pytest.mark.parametrize("stage_img", ["dots", "blown"])
def test_compose_handles_stage0_and_stage1(stage_img):
    rng = np.random.default_rng(5)
    if stage_img == "dots":
        img = sd.printed_dots("AB12CD34", rng)
    else:
        img = sd.printed_sans("AB12CD", rng)
        img[:, : img.shape[1] * 3 // 4] = 255
    r = of.evaluate_handwritten_score(img)
    assert inf.compose(img, r).ndim == 3
