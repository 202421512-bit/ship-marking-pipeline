# -*- coding: utf-8 -*-
"""
2차 평가 코드 자동 테스트  [우리 도구]
'정답을 아는 기준선'을 넣어서 평가 숫자가 이론값대로 나오는지 확인한다.
실행:  pytest stage2_structural/tests -v
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import evaluate_stage2 as ev  # noqa: E402
import make_stage2_samples as gen  # noqa: E402
from stage2_io import Stage2Result, empty_result  # noqa: E402

N = 2   # 조건당 장수 (빠른 테스트용)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage2_eval_data")
    # 반사가 들어가는 시드(끝자리 2)가 포함되도록 3장 생성 후 0,2번만 써도 되지만, 간단히 3장 생성
    gen.generate("dev", 3, out_root=str(root), verbose=False)
    return str(root)


def run(method, root, tmp_path, **kw):
    rows, summary, _ = ev.run(method, "dev", data_root=root, out_dir=str(tmp_path), verbose=False, **kw)
    return rows, {s["condition"]: s for s in summary}


# 1. 아무것도 안 하면: 검출 0, 제거 0, 손상 0, 보존율 1
def test_identity(data_root, tmp_path):
    rows, _ = run("identity", data_root, tmp_path)
    for r in rows:
        assert r["det_F1"] == 0 and r["rem_F1"] == 0
        assert r["algo_damage_measured_px"] == 0 and r["preserve_rate_measured"] == 1.0
        assert r["undeclared_change_px"] == 0


# 2. 정답 기반 보수적 제거: 검출·제거 완벽, 알고리즘 손상 0, 기존 손상은 안 건드림
def test_oracle_conservative(data_root, tmp_path):
    rows, _ = run("oracle_conservative", data_root, tmp_path)
    for r in rows:
        assert r["det_F1"] == pytest.approx(1.0)
        assert r["rem_P"] == pytest.approx(1.0) and r["rem_R"] == pytest.approx(1.0)
        assert r["algo_damage_declared_px"] == 0 and r["algo_damage_measured_px"] == 0
        assert r["preexisting_removed_px"] == 0
        if r["has_glare"]:
            assert r["glare_F1"] == pytest.approx(1.0)
        else:
            assert r["glare_false_px_noglare_img"] == 0


# 3. 정답 기반 전부 제거: 겹친 부분까지 지우면 제거 정밀도가 떨어지고 '기존 손상 지움'으로 잡힌다
def test_oracle_remove_all(data_root, tmp_path):
    rows, _ = run("oracle_remove_all", data_root, tmp_path)
    for r in rows:
        assert r["preexisting_removed_px"] == r["preexisting_damage_px"]
        assert r["algo_damage_declared_px"] == 0     # 온전한 획은 안 건드렸으므로 0
        if r["condition"] == "C1_no_overlap":
            assert r["rem_P"] == pytest.approx(1.0)
        else:
            assert r["rem_P"] < 1.0
            assert r["rem_P_tol"] < 1.0     # 허용 거리가 보호 대상 위 제거를 봐주면 안 됨


# 4. 스크래치를 넓게(7px 팽창) 지우는 방법: C2·C3에서 '알고리즘이 만든 손상'이 잡히고, C1은 0
def test_overreaching_removal_is_counted_as_damage(data_root, tmp_path):
    def wide(img, protect_boxes=None, gt=None):
        rem = cv2.dilate(gt["gt_scratch"].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        res = empty_result(img)
        res.detect_mask = res.remove_mask = rem
        res.image = cv2.inpaint(img, rem.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        return res
    rows, _ = run(wide, data_root, tmp_path)
    for r in rows:
        if r["condition"] == "C1_no_overlap":
            assert r["algo_damage_declared_px"] == 0
        else:
            assert r["algo_damage_declared_px"] > 0 and r["preserve_rate_declared"] < 1.0


# 5. 선언하지 않은 부작용(전체 흐림)은 'undeclared_change'와 실측 손상으로 잡힌다
def test_undeclared_side_effect_is_measured(data_root, tmp_path):
    def blur(img, protect_boxes=None, gt=None):
        res = empty_result(img)
        res.image = cv2.GaussianBlur(img, (0, 0), 3)
        return res
    rows, _ = run(blur, data_root, tmp_path)
    for r in rows:
        assert r["algo_damage_declared_px"] == 0
        assert r["undeclared_change_px"] > 0 and r["algo_damage_measured_px"] > 0


# 6. CER 인터페이스: 정답을 돌려주는 OCR은 CER 0, 빈 문자열을 돌려주는 OCR은 CER 1
def test_cer_interface(data_root, tmp_path):

    def method(img, protect_boxes=None, gt=None):
        return empty_result(img)
    # 정답 OCR: 박스 순서대로 meta의 문자열을 돌려주도록 run 안에서 meta를 알 수 없으므로,
    # 크롭 개수만큼 '정답 목록'을 꺼내는 방식으로 흉내 낸다.
    import json
    manifest = json.load(open(os.path.join(data_root, "dev", "manifest.json"), encoding="utf-8"))
    queue = []
    for item in manifest:
        meta = json.load(open(os.path.join(data_root, item["path"], "meta.json"), encoding="utf-8"))
        queue += [meta["gt_text_strings"]] * 2      # before / after 각 1번

    def perfect_ocr(crops):
        return queue.pop(0)

    rows, summ = run(method, data_root, tmp_path, ocr_fn=perfect_ocr)
    assert all(r["cer_before"] == 0 and r["cer_after"] == 0 for r in rows)
    assert summ["ALL"]["cer_after_micro"] == 0

    rows, summ = run(method, data_root, tmp_path, ocr_fn=lambda crops: [""] * len(crops))
    assert all(r["cer_after"] == 1.0 and r["exact_after"] == 0 for r in rows)


# 7. 잘못된 결과 형식은 바로 오류
def test_validate_rejects_wrong_shapes():
    img = np.zeros((40, 50, 3), np.uint8)
    res = empty_result(img)
    res.detect_mask = np.zeros((10, 10), bool)
    with pytest.raises(ValueError):
        res.validate(img)
    res = empty_result(img)
    res.image = np.zeros((40, 50, 3), np.float32)
    with pytest.raises(ValueError):
        res.validate(img)
    assert isinstance(empty_result(img).validate(img), Stage2Result)
