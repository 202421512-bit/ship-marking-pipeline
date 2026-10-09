"""objective_function.py / benchmark_and_optimize.py / synthetic_dataset.py 검증."""
import numpy as np
import pytest

import benchmark_and_optimize as bo
import objective_function as of
import synthetic_dataset as sd


# --- 통계 도구 ---------------------------------------------------------------
def test_roc_auc_and_cohens_d():
    assert bo.roc_auc([0, 1, 2], [3, 4, 5]) == 1.0
    assert bo.roc_auc([3, 4, 5], [0, 1, 2]) == 0.0
    assert bo.roc_auc([1, 1], [1, 1]) == 0.5                      # 동점
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 4000), rng.normal(1, 1, 4000)
    assert bo.cohens_d(a, b) == pytest.approx(1.0, abs=0.06)


def test_correlation_handles_missing():
    x = np.arange(20, dtype=float)
    X = np.column_stack([x, 2 * x + 1, -x, x.copy()])
    X[3, 3] = np.nan
    C = bo.correlation(X)
    assert C[0, 1] == pytest.approx(1.0) and C[0, 2] == pytest.approx(-1.0) and C[0, 3] == pytest.approx(1.0)


def test_logistic_recovers_direction_and_weights_are_nonnegative():
    rng = np.random.default_rng(1)
    n = 400
    y = (rng.random(n) < 0.5).astype(int)
    X = np.column_stack([y + rng.normal(0, 0.5, n),          # 수기일수록 큼 -> 양의 계수
                         -y + rng.normal(0, 0.5, n),         # 수기일수록 작음 -> 음의 계수
                         rng.normal(0, 1, n)])               # 무관
    pp = bo.Preprocessor().fit(X)
    w, b = bo.fit_logistic(pp.transform(X), y, lam=1.0)
    assert w[0] > 1 and w[1] < -1 and abs(w[2]) < 0.3
    pos = bo.weights_from_coefficients(w, pp.std, "positive")
    ab = bo.weights_from_coefficients(w, pp.std, "abs")
    assert pos.sum() == pytest.approx(1) and ab.sum() == pytest.approx(1)
    assert pos[1] == 0 and ab[1] > 0.3                         # |c| 는 음의 지표에도 큰 가중치를 줌


def test_thresholds_bound_error_in_confirmed_zones():
    rng = np.random.default_rng(2)
    s = np.concatenate([rng.uniform(0, 0.6, 200), rng.uniform(0.4, 1.0, 200)])
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    th = bo.choose_thresholds(s, y, max_error=0.03)
    assert th["printed"] <= th["handwritten"]
    assert y[s <= th["printed"]].mean() <= 0.03
    assert (1 - y[s >= th["handwritten"]]).mean() <= 0.03


# --- 목적함수 ----------------------------------------------------------------
def test_dynamic_reweighting_sums_to_one():
    w = {k: 1 / 8 for k in of.FEATURES}
    phis = {k: 0.5 for k in of.FEATURES}
    for k in ("phi_2", "phi_3", "phi_4", "phi_6"):
        phis[k] = None                                          # 글자 1개: 배열 지표 없음
    phis["phi_1"], phis["phi_5"], phis["phi_7"], phis["phi_8"] = 0.2, 0.4, 0.6, 0.8
    s, used, coverage = of.combine(phis, w)
    assert sum(used.values()) == pytest.approx(1.0)
    assert set(used) == {"phi_1", "phi_5", "phi_7", "phi_8"}
    assert s == pytest.approx(0.5) and coverage == pytest.approx(0.5)
    assert of.combine({k: None for k in of.FEATURES}, w)[0] is None


def test_reference_confidence_drops_with_low_contrast():
    rng = np.random.default_rng(3)
    img = sd.printed_sans("B12SP34", rng)
    c_ok, m_ok, _ = of.image_confidence(img)
    faint = (200 - (200 - img.astype(np.float32)) * 0.1).astype(np.uint8)
    faint = np.clip(faint + rng.normal(0, 8, faint.shape), 0, 255).astype(np.uint8)
    c_bad, _, _ = of.image_confidence(faint)
    assert c_ok > 0.9 and m_ok == 1.0
    assert c_bad < c_ok


def test_verdict_does_not_depend_on_confidence():
    """판정은 점수로만: 같은 점수면 신뢰도가 낮아도 같은 판정 (신뢰도 평가는 이후 단계)."""
    rng = np.random.default_rng(8)
    img = sd.field_noise(sd.handwritten("HK357B12", rng), rng)
    r = of.evaluate_handwritten_score(img)
    th = r["thresholds"]
    expected = ("Confirmed Handwritten" if r["score"] >= th["handwritten"] else
                "Confirmed Printed" if r["score"] <= th["printed"] else "Uncertain (Need Review)")
    assert r["verdict"] == expected


def test_evaluate_returns_verdict_and_handles_single_char(tmp_path):
    rng = np.random.default_rng(4)
    r = of.evaluate_handwritten_score(sd.field_noise(sd.handwritten("HK357B", rng), rng))
    assert r["verdict"] in of.VERDICT_COLORS
    assert 0 <= r["score"] <= 1 and 0 <= r["confidence"] <= 1
    assert sum(r["weights_used"].values()) == pytest.approx(1.0)
    one = of.evaluate_handwritten_score(sd.handwritten("B", rng))
    assert one["features"]["phi_4"] is None and one["features"]["phi_2"] is None
    assert one["score"] is not None and one["coverage"] < 1
    vis = of.draw_objective_debug(sd.handwritten("B", rng), one)
    assert vis.ndim == 3 and vis.shape[1] > 900


def test_explicit_weights_override_calibration():
    rng = np.random.default_rng(5)
    img = sd.printed_sans("AB12CD", rng)
    only4 = {k: (1.0 if k == "phi_4" else 0.0) for k in of.FEATURES}
    r = of.evaluate_handwritten_score(img, weights=only4)
    assert r["weights_used"] == {"phi_4": 1.0}
    assert r["score"] == pytest.approx(r["features"]["phi_4"])


# --- 합성 데이터 --------------------------------------------------------------
@pytest.mark.parametrize("kind", ["sans", "stencil", "dots"])
def test_printed_generators(kind):
    rng = np.random.default_rng(6)
    img = sd.PRINTED_KINDS[kind]("AB12", rng)
    assert img.dtype == np.uint8 and (img < 100).any() and (img == sd.BG).any()


def test_dot_matrix_letters_do_not_touch():
    """도트 글자는 칸 사이를 2 피치 띄워 이웃 글자 도트가 붙지 않아야 한다."""
    rng = np.random.default_rng(7)
    img = sd.printed_dots("0000", rng)
    ink = (img < 120).astype(np.uint8)
    cols = ink.any(axis=0)
    runs = np.diff(np.r_[0, cols.astype(int), 0])
    assert (runs == 1).sum() >= 4                               # 글자 4개가 세로 빈 열로 나뉨


def test_dataset_is_balanced_and_reproducible():
    a = sd.make_dataset(6, seed=3)
    b = sd.make_dataset(6, seed=3)
    assert [m["label"] for _, m in a].count(0) == 6
    assert all(np.array_equal(x, y) for (x, _), (y, _) in zip(a, b))
    assert {m["kind"] for _, m in a if m["label"] == 0} == {"sans", "stencil", "dots"}
