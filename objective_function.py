"""
수기 판별 목적함수: 8개 특징 φ₁~φ₈ 를 가중 합해 수기 점수 S_hand ∈ [0, 1] 과 신뢰도 C, 판정을 낸다.

  S_hand = Σ_{i∈유효} w_i φ_i / Σ_{i∈유효} w_i
    - 가중치 w 는 benchmark_and_optimize.py 가 데이터로 구해 objective_weights.json 에 저장한다
      (없으면 균등 가중치).
    - 동적 재분배: 글자가 1개뿐이라 φ₂(간격)·φ₄(기준선)·φ₆(크기)·φ₃(기울기) 가 None 이면 그 항을 빼고
      남은 지표의 가중치 합이 1 이 되게 비례 재할당한다.
  신뢰도 C = C_glare × C_contrast × C_coverage ∈ [0, 1]
    - C_glare   : 반사광 포화(≥ 250) 픽셀 비율이 클수록 낮음
    - C_contrast: 배경-획 대비 잡음비 CNR = |배경 중앙값 − 획 중앙값| / 배경 표준편차 가 낮을수록 낮음
    - C_coverage: 계산된 지표의 가중치 비율 (지표가 많이 빠질수록 낮음)
  판정: C < C_MIN 이거나 S 가 두 임계값 사이면 "Low Confidence (Need Review)",
        S ≥ 수기 임계값이면 "Confirmed Handwritten", S ≤ 인쇄체 임계값이면 "Confirmed Printed".
"""
import json
import math
import os

import cv2
import numpy as np

from baseline_feature import calculate_phi_4_baseline
from connectivity_feature import calculate_phi_8_connectivity
from contour_roughness_feature import calculate_phi_7_roughness
from curvature_feature import calculate_phi_5_curvature
from orientation_feature import _remove_specks, binarize, calculate_phi_3_orientation, load_image
from size_variance_feature import calculate_phi_6_size_variance
from spacing_feature import calculate_phi_2_spacing
from stroke_features import stroke_thickness_variation

FEATURES = ["phi_1", "phi_2", "phi_3", "phi_4", "phi_5", "phi_6", "phi_7", "phi_8"]
FEATURE_NAMES = {
    "phi_1": "stroke width", "phi_2": "spacing", "phi_3": "orientation", "phi_4": "baseline",
    "phi_5": "curvature", "phi_6": "size", "phi_7": "roughness", "phi_8": "connectivity",
}
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objective_weights.json")

# 신뢰도 / 판정 (benchmark_and_optimize.py 가 임계값을 데이터로 정해 json 에 덮어씀)
GLARE_LEVEL = 250
GLARE_FULL = 0.15          # 포화 비율이 이 이상이면 C_glare = 0
CNR_LOW, CNR_HIGH = 2.0, 6.0
C_MIN = 0.5
DEFAULT_THRESHOLDS = {"printed": 0.35, "handwritten": 0.55}


def load_calibration(path=WEIGHTS_PATH):
    """저장된 가중치·임계값. 없으면 균등 가중치와 기본 임계값."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["weights"], data.get("thresholds", DEFAULT_THRESHOLDS)
    return {k: 1 / len(FEATURES) for k in FEATURES}, dict(DEFAULT_THRESHOLDS)


# ---------------------------------------------------------------------------
# 특징 추출
# ---------------------------------------------------------------------------
def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


def extract_features(image_path_or_array):
    """φ₁~φ₈ (계산 불가면 None) 과 각 모듈의 원 결과."""
    img = load_image(image_path_or_array)
    r1 = stroke_thickness_variation(img)
    raw = {
        "phi_1": r1,
        "phi_2": calculate_phi_2_spacing(img),
        "phi_3": calculate_phi_3_orientation(img),
        "phi_4": calculate_phi_4_baseline(img),
        "phi_5": calculate_phi_5_curvature(img),
        "phi_6": calculate_phi_6_size_variance(img),
        "phi_7": calculate_phi_7_roughness(img),
        "phi_8": calculate_phi_8_connectivity(img),
    }
    phis = {"phi_1": _num(r1.phi1) if r1.valid else None}
    for k in FEATURES[1:]:
        phis[k] = _num(raw[k][k])
    return phis, raw


def image_confidence(img):
    """반사광 포화 비율과 배경-획 CNR 로 영상 품질 신뢰도. (C_glare, C_contrast, 상세) 반환."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    glare = float((gray >= GLARE_LEVEL).mean())
    c_glare = float(np.clip(1 - glare / GLARE_FULL, 0, 1))
    ink = _remove_specks(binarize(img))
    if ink.sum() < 20 or (~ink).sum() < 20:
        return c_glare, 0.0, {"glare_ratio": glare, "cnr": 0.0}
    background = ~cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # 획 테두리 제외
    bg = gray[background] if background.any() else gray[~ink]
    cnr = abs(float(np.median(bg)) - float(np.median(gray[ink]))) / (float(bg.std()) + 1e-6)
    c_contrast = float(np.clip((cnr - CNR_LOW) / (CNR_HIGH - CNR_LOW), 0, 1))
    return c_glare, c_contrast, {"glare_ratio": glare, "cnr": cnr}


def combine(phis, weights):
    """동적 재분배 가중 합. (S, 실제 쓴 가중치, 유효 가중치 비율)"""
    avail = {k: w for k, w in weights.items() if phis.get(k) is not None and w > 0}
    total = sum(weights.values())
    if not avail:
        return None, {}, 0.0
    s = sum(avail.values())
    used = {k: w / s for k, w in avail.items()}
    return float(sum(used[k] * phis[k] for k in used)), used, float(s / total) if total > 0 else 0.0


def evaluate_handwritten_score(image_path_or_array, weights=None, thresholds=None) -> dict:
    """수기 점수 S_hand, 신뢰도 C, 판정.

    weights     {"phi_1": w1, ...}. None 이면 objective_weights.json (없으면 균등)
    thresholds  {"printed": t_p, "handwritten": t_h}. None 이면 json 의 값

    반환 dict: score(S_hand), confidence(C), verdict, features(φ 값, None 포함), weights_used(재분배 후),
              coverage, confidence_parts, raw(각 모듈 결과)
    """
    img = load_image(image_path_or_array)
    cal_w, cal_t = load_calibration()
    weights = dict(weights or cal_w)
    thresholds = dict(thresholds or cal_t)
    phis, raw = extract_features(img)
    score, used, coverage = combine(phis, weights)
    c_glare, c_contrast, detail = image_confidence(img)
    confidence = c_glare * c_contrast * coverage

    if score is None:
        verdict = "Low Confidence (Need Review)"
    elif confidence < C_MIN:
        verdict = "Low Confidence (Need Review)"
    elif score >= thresholds["handwritten"]:
        verdict = "Confirmed Handwritten"
    elif score <= thresholds["printed"]:
        verdict = "Confirmed Printed"
    else:
        verdict = "Low Confidence (Need Review)"
    return {"score": score, "confidence": float(confidence), "verdict": verdict, "features": phis,
            "weights_used": used, "coverage": coverage, "thresholds": thresholds,
            "confidence_parts": {"glare": c_glare, "contrast": c_contrast, "coverage": coverage, **detail},
            "raw": raw}


# ---------------------------------------------------------------------------
# 시각화: 원본 + 레이더 차트 + 지표별 막대
# ---------------------------------------------------------------------------
VERDICT_COLORS = {"Confirmed Handwritten": (40, 40, 220), "Confirmed Printed": (40, 160, 40),
                  "Low Confidence (Need Review)": (0, 150, 230)}


def _radar(phis, used, size=400):
    img = np.full((size, size, 3), 255, np.uint8)
    c = (size // 2, size // 2 + 10)
    r = int(size * 0.33)
    n = len(FEATURES)
    ang = [-math.pi / 2 + 2 * math.pi * i / n for i in range(n)]
    for level in (0.25, 0.5, 0.75, 1.0):
        ring = [(int(c[0] + level * r * math.cos(a)), int(c[1] + level * r * math.sin(a))) for a in ang]
        cv2.polylines(img, [np.array(ring)], True, (215, 215, 215), 1, cv2.LINE_AA)
    pts = []
    for i, (k, a) in enumerate(zip(FEATURES, ang)):
        end = (int(c[0] + r * math.cos(a)), int(c[1] + r * math.sin(a)))
        cv2.line(img, c, end, (200, 200, 200), 1, cv2.LINE_AA)
        v = phis.get(k)
        label = f"{k.replace('phi_', 'p')} {FEATURE_NAMES[k]}"
        val = "n/a" if v is None else f"{v:.2f}"
        lx = int(c[0] + (r + 14) * math.cos(a)) - (len(label) * 3 if math.cos(a) < -0.1 else 0 if math.cos(a) > 0.1 else len(label) * 3 // 2)
        ly = int(c[1] + (r + 14) * math.sin(a)) + (10 if math.sin(a) > 0.1 else -2)
        lx = int(np.clip(lx, 2, size - 7 * len(label) - 2))      # 캔버스 밖으로 잘리지 않게
        color = (60, 60, 60) if v is not None else (170, 170, 170)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        cv2.putText(img, val, (lx, ly + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        vv = 0.0 if v is None else v
        pts.append((int(c[0] + vv * r * math.cos(a)), int(c[1] + vv * r * math.sin(a))))
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(pts)], (230, 160, 90))
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(img, [np.array(pts)], True, (200, 110, 40), 2, cv2.LINE_AA)
    for (px, py), k in zip(pts, FEATURES):
        cv2.circle(img, (px, py), 3 if phis.get(k) is not None else 2,
                   (200, 110, 40) if phis.get(k) is not None else (170, 170, 170), -1, cv2.LINE_AA)
    return img


def _bars(phis, used, width=400, height=250):
    img = np.full((height, width, 3), 255, np.uint8)
    n = len(FEATURES)
    bw = (width - 60) // n
    base = height - 40
    cv2.line(img, (40, base), (width - 10, base), (120, 120, 120), 1)
    for t in (0.5, 1.0):
        y = int(base - t * (height - 80))
        cv2.line(img, (40, y), (width - 10, y), (225, 225, 225), 1)
        cv2.putText(img, f"{t:.1f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    for i, k in enumerate(FEATURES):
        x = 45 + i * bw
        v = phis.get(k)
        if v is None:
            cv2.putText(img, "n/a", (x + 2, base - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)
        else:
            top = int(base - v * (height - 80))
            cv2.rectangle(img, (x + 4, top), (x + bw - 6, base), (200, 110, 40), -1)
        w = used.get(k, 0.0)
        cv2.putText(img, f"p{k[-1]}", (x + 6, base + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(img, f"w{w:.2f}", (x + 1, base + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (60, 60, 60) if w > 0 else (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(img, "phi (bar) / weight used (w)", (40, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def draw_objective_debug(image_path_or_array, result):
    """왼쪽: 원본 + 판정 띠, 오른쪽: 레이더 차트(위) + 지표별 φ 막대와 실제 쓴 가중치(아래)."""
    img = load_image(image_path_or_array)
    src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    panel_h = 400 + 250
    scale = min(560 / src.shape[1], (panel_h - 80) / src.shape[0])
    src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    left = np.full((panel_h, max(560, src.shape[1]), 3), 255, np.uint8)
    left[80:80 + src.shape[0], :src.shape[1]] = src
    color = VERDICT_COLORS[result["verdict"]]
    cv2.rectangle(left, (0, 0), (left.shape[1], 70), color, -1)
    s = "n/a" if result["score"] is None else f"{result['score']:.3f}"
    cv2.putText(left, result["verdict"], (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cp = result["confidence_parts"]
    cv2.putText(left, f"S_hand={s}  C={result['confidence']:.2f}  (glare {cp['glare']:.2f} x contrast {cp['contrast']:.2f}"
                      f" x coverage {cp['coverage']:.2f})", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    right = np.vstack([_radar(result["features"], result["weights_used"]),
                       _bars(result["features"], result["weights_used"])])
    return np.hstack([left, right])


def _save(path, image):
    ok, buf = cv2.imencode(".png", image)
    buf.tofile(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="수기 판별 목적함수: 점수·신뢰도·판정 + 레이더 시각화")
    parser.add_argument("image", nargs="?", help="크롭 이미지 (없으면 합성 예시)")
    parser.add_argument("-o", "--output", default="objective_debug.png")
    args = parser.parse_args()
    if args.image:
        samples = [(args.image, args.image)]
    else:
        import synthetic_dataset as sd
        rng = np.random.default_rng(7)
        samples = [("printed", sd.make_sample(0, rng, "sans")[0]), ("handwritten", sd.make_sample(1, rng)[0])]
    panels = []
    for name, src in samples:
        res = evaluate_handwritten_score(src)
        feats = "  ".join(f"{k[-1]}:{'-' if v is None else f'{v:.2f}'}" for k, v in res["features"].items())
        s = "None" if res["score"] is None else f"{res['score']:.3f}"
        print(f"{name:12s} S={s}  C={res['confidence']:.2f}  {res['verdict']}  | {feats}")
        panels.append(draw_objective_debug(src, res))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 6, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for p in panels]
    _save(args.output, np.vstack(panels))
    print(f"시각화 저장: {args.output}")
