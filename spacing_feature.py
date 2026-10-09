"""
수기(Handwritten) vs 기계식 마킹(Printed) 판별 특징 - φ₂ 문자 간격 편차 (Character Spacing Deviation)

정의
  글자 k = 1..M 을 줄 방향으로 정렬, 인접 글자 간 순수 간격  d_k = x_min^(k+1) − x_max^(k)  (겹치면 음수)
  σ_d: 간격의 표준편차 또는 MAD,   φ₂ = 1 − exp(−α₂ · σ_d / W̄_char)   (또는 tanh)

구현상 결정 (이유)
  1. 글자 분리는 φ₆ 과 같은 함수: 점 잡음 제거, 끊긴 도트/스텐실 조각 병합, 줄 밖 잡음과 경계에 잘린 글자 제외.
     간격과 폭은 '줄 방향 좌표' 로 잰다 (줄이 기울면 이미지 x 기준 간격이 글자 높이에 따라 달라짐).
  2. 문장부호('.', '-', ',')는 빼고, 그 양옆 간격도 통계에서 뺀다. 빼고 나서 이으면 '2-S' 사이가
     부호 폭만큼 큰 가짜 간격이 된다.
  3. 띄어쓰기: 단어 사이 간격은 자간과 다른 범주라 섞으면 인쇄체도 퍼짐이 크게 뛴다 (합성 실험에서
     정의 그대로의 표준편차는 AUC 0.50). 간격이 '중앙 간격의 2.5배' 와 '글자 높이의 0.35배' 를 모두
     넘으면 띄어쓰기로 보고 통계에서 뺀다. 그러고 나면 정의 그대로의 표준편차(spread="std", 기본)가
     AUC 0.83~0.85 로 MAD(0.77~0.87)와 비슷해서 기본은 정의를 따른다.
  4. 고정폭 글꼴(잉크젯·도트·스텐실에 흔함)은 '1' 같은 좁은 글자 양옆 상자 간격이 원래 넓고 대신 글자
     중심 간 거리(pitch)가 일정하다. 상자 간격 편차와 함께 중심 간격 편차(sigma_pitch_px)를 계산해 돌려주고,
     metric="pitch" 로 점수에 쓸 수 있다.
"""
import math

import cv2
import numpy as np

from orientation_feature import load_image
from size_variance_feature import extract_characters

ALPHA2 = 12.0             # φ₂ = 1 − exp(−α₂·σ_d/W̄): σ_d/W̄ = 0.058 에서 0.5. 합성 인쇄체 중앙 0.033, 불규칙 간격 0.077
PUNCT_HEIGHT_RATIO = 0.5  # 높이가 중앙값의 절반 미만이면 문장부호
WORD_GAP_MEDIAN = 2.5     # 간격 > 이 값 × 중앙 간격 이고
WORD_GAP_HEIGHT = 0.35    # 간격 > 이 값 × 글자 높이 이면 띄어쓰기


def robust_spread(values, spread):
    v = np.asarray(values, np.float64)
    if spread == "std":
        return float(v.std())                                   # 모집단 표준편차 (정의)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))  # MAD (정규분포면 표준편차와 같음)


def calculate_phi_2_spacing(image_path_or_array, alpha=ALPHA2, spread="std", metric="gap",
                            mapping="exp", binarization="sauvola") -> dict:
    """단일 크롭 이미지의 φ₂ (문자 간격 편차).

    spread   "std"(기본, 정의 그대로 모집단 표준편차) | "mad"(MAD×1.4826, 소수 이상치에 강건)
    metric   "gap"(기본, 상자 사이 순수 간격 d_k) | "pitch"(글자 중심 간 거리, 고정폭 글꼴용)
    mapping  "exp": 1 − exp(−α·x) | "tanh": tanh(α·x),   x = σ / W̄_char

    반환 dict
      phi_2, sigma_d_px(간격 퍼짐), sigma_pitch_px(중심 간격 퍼짐), mean_char_width(px),
      gaps_px(통계에 쓴 간격), all_gaps(시각화용: (왼쪽 글자, 오른쪽 글자, 간격, 사용 여부, 사유)),
      pitches_px, num_valid_chars, chars, components, reason
    """
    img = load_image(image_path_or_array)
    comps, chars, frame = extract_characters(img, binarization)
    result = {"phi_2": None, "sigma_d_px": None, "sigma_pitch_px": None, "mean_char_width": None,
              "gaps_px": [], "all_gaps": [], "pitches_px": [], "num_valid_chars": 0,
              "num_word_gaps": 0, "chars": chars, "components": comps, "frame": frame,
              "spread": spread, "metric": metric,
              "reason": ""}

    def mark(g, reason):
        g["reason"] = reason
        for m in g["members"]:
            m["reason"] = reason

    if chars:
        median_h = np.median([g["height"] for g in chars])
        for g in chars:
            if g["height"] < PUNCT_HEIGHT_RATIO * median_h:
                mark(g, "punctuation")
    letters = [g for g in chars if not g["reason"]]
    result["num_valid_chars"] = len(letters)
    if len(letters) < 2:
        result["reason"] = f"유효 글자 {len(letters)}개 (인접 간격을 정의하려면 2개 이상 필요)"
        return result

    # 글자 순서대로 이웃 간격. 사이에 문장부호가 끼어 있으면 그 간격은 통계에서 뺀다.
    order = sorted(chars, key=lambda g: g["span"][0])
    pos = {id(g): i for i, g in enumerate(order)}
    gaps, pitches = [], []
    for a, b in zip(letters, letters[1:]):
        d = b["span"][0] - a["span"][1]
        between = order[pos[id(a)] + 1: pos[id(b)]]
        reason = "punctuation_between" if any(g["reason"] == "punctuation" for g in between) else ""
        result["all_gaps"].append((a, b, float(d), not reason, reason))
        if not reason:
            gaps.append(d)
            pitches.append((b["span"][0] + b["span"][1]) / 2 - (a["span"][0] + a["span"][1]) / 2)
    mean_w = float(np.mean([g["width"] for g in letters]))
    result["mean_char_width"] = mean_w

    # 띄어쓰기 간격 제외
    if gaps:
        med_gap = float(np.median(gaps))
        char_h = float(np.median([g["height"] for g in letters]))
        word_cut = max(WORD_GAP_MEDIAN * max(med_gap, 1.0), WORD_GAP_HEIGHT * char_h)
        keep = [d <= word_cut for d in gaps]
        result["num_word_gaps"] = int(len(gaps) - sum(keep))
        gaps = [d for d, k in zip(gaps, keep) if k]
        pitches = [p for p, k in zip(pitches, keep) if k]
        # 시각화용 목록 갱신: 앞서 쓰던 간격(used) 순서대로 keep 이 대응
        flags = iter(keep)
        updated = []
        for a, b, d, used, why in result["all_gaps"]:
            if used and not next(flags):
                used, why = False, "word_space"
            updated.append((a, b, d, used, why))
        result["all_gaps"] = updated
    if len(gaps) < 1:
        result["reason"] = "문장부호를 사이에 두지 않은 인접 간격이 없음"
        return result

    sigma_gap = robust_spread(gaps, spread) if len(gaps) >= 2 else 0.0
    sigma_pitch = robust_spread(pitches, spread) if len(pitches) >= 2 else 0.0
    x = (sigma_pitch if metric == "pitch" else sigma_gap) / mean_w
    phi = 1 - math.exp(-alpha * x) if mapping == "exp" else math.tanh(alpha * x)
    result.update(phi_2=float(phi), sigma_d_px=sigma_gap, sigma_pitch_px=sigma_pitch,
                  gaps_px=[float(g) for g in gaps], pitches_px=[float(p) for p in pitches])
    if len(gaps) < 2:
        result["reason"] = "간격이 1개뿐이라 퍼짐 = 0 (참고용)"
    return result


# ---------------------------------------------------------------------------
# 디버깅 시각화
# ---------------------------------------------------------------------------
def draw_spacing_debug(image_path_or_array, result, scale=None):
    """글자 상자(초록, 줄 방향), 인접 간격 d_k 를 양방향 화살표와 수치(px)로.
    파랑 화살표: 통계에 쓴 간격 / 보라: 띄어쓰기로 제외 / 회색: 문장부호를 사이에 둬서 제외 / 주황 상자: 문장부호."""
    img = load_image(image_path_or_array)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    if scale is None:
        scale = max(1.0, 600 / max(canvas.shape[:2]))
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    t = max(1, int(round(scale)))
    for g in result["chars"]:
        quad = (np.array(g["corners"]) * scale).astype(np.int32)
        color = (0, 140, 255) if g["reason"] == "punctuation" else (0, 190, 0)
        cv2.polylines(canvas, [quad], True, color, t, cv2.LINE_AA)
    if result["frame"] is not None:
        u, n = result["frame"]
        for a, b, d, used, why in result["all_gaps"]:
            # 두 글자 몸통 높이 중간에서, a 의 오른쪽 끝 -> b 의 왼쪽 끝
            mid = (max(a["across"][0], b["across"][0]) + min(a["across"][1], b["across"][1])) / 2
            p0 = (a["span"][1] * u + mid * n) * scale
            p1 = (b["span"][0] * u + mid * n) * scale
            color = (255, 90, 0) if used else ((200, 0, 200) if why == "word_space" else (150, 150, 150))
            p0i, p1i = tuple(int(v) for v in p0), tuple(int(v) for v in p1)
            if abs(d) >= 2:
                cv2.arrowedLine(canvas, p0i, p1i, color, t, cv2.LINE_AA, tipLength=min(0.4, 4 / max(abs(d), 1)))
                cv2.arrowedLine(canvas, p1i, p0i, color, t, cv2.LINE_AA, tipLength=min(0.4, 4 / max(abs(d), 1)))
            else:
                cv2.circle(canvas, p0i, 2 * t, color, -1)
            label_pos = (int((p0[0] + p1[0]) / 2) - 8 * t, int(max(a["across"][1], b["across"][1]) * scale * n[1]
                                                               + (p0[1] + p1[1]) / 2 * (1 - n[1])) + 14 * t)
            cv2.putText(canvas, f"{d:.0f}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, color, t, cv2.LINE_AA)
    phi = result["phi_2"]
    text = (f"phi2={phi:.3f}  sigma_d={result['sigma_d_px']:.1f}px  sigma_pitch={result['sigma_pitch_px']:.1f}px  "
            f"W={result['mean_char_width']:.1f}px  M={result['num_valid_chars']}  [{result['spread']}/{result['metric']}]"
            if phi is not None else f"phi2=None: {result['reason']}")
    cv2.rectangle(canvas, (0, 0), (12 + 8 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 합성 시험 데이터
# ---------------------------------------------------------------------------
def render_monospace(text, pitch=40, size=(140, 520), thickness=6, scale=2.0, angle=0.0):
    """고정폭 기계 마킹: 글자마다 같은 칸(pitch) 가운데에 놓는다 ('1' 은 좁아도 칸 중심에)."""
    img = np.full(size, 200, np.uint8)
    base = size[0] - 40
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        (w, h), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cx = 20 + pitch * i + pitch // 2
        cv2.putText(img, ch, (cx - w // 2, base), cv2.FONT_HERSHEY_SIMPLEX, scale, 40, thickness, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((size[1] / 2, size[0] / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (size[1], size[0]), borderValue=200)


def render_irregular_spacing(text, jitter=8, seed=0, size=(140, 520), thickness=6, scale=2.2):
    """간격만 불규칙한 글씨 (글자 모양·크기는 인쇄체 그대로): 간격 지표만 따로 시험하기 위함."""
    rng = np.random.default_rng(seed)
    img = np.full(size, 200, np.uint8)
    x, base = 20, size[0] - 40
    for ch in text:
        if ch == " ":
            x += 28
            continue
        (w, _), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if x + w > size[1] - 10:
            break
        cv2.putText(img, ch, (x, base), cv2.FONT_HERSHEY_SIMPLEX, scale, 40, thickness, cv2.LINE_AA)
        x += w + int(rng.integers(-2, jitter + 1))
    return img


def _save(path, image):
    ok, buf = cv2.imencode(".png", image)
    buf.tofile(path)


if __name__ == "__main__":
    import argparse

    from orientation_feature import add_plate_noise, render_handwritten, render_printed

    parser = argparse.ArgumentParser(description="φ₂ 문자 간격 편차 계산 + 디버그 시각화")
    parser.add_argument("image", nargs="?", help="크롭 이미지 (없으면 합성 예시)")
    parser.add_argument("-o", "--output", default="phi2_debug.png")
    parser.add_argument("--spread", default="std", choices=["std", "mad"])
    parser.add_argument("--metric", default="gap", choices=["gap", "pitch"])
    args = parser.parse_args()

    if args.image:
        samples = [(args.image, args.image)]
    else:
        samples = [("printed (prop.)", add_plate_noise(render_printed("B12-SP3 4500"))),
                   ("printed (mono)", add_plate_noise(render_monospace("B12-SP3 4500", angle=6))),
                   ("irregular gaps", add_plate_noise(render_irregular_spacing("B12-SP3 4500", seed=1))),
                   ("handwritten", add_plate_noise(render_handwritten("B12-SP3 4500", seed=3, tilt_std=5)))]
    panels = []
    for name, src in samples:
        res = calculate_phi_2_spacing(src, spread=args.spread, metric=args.metric)
        if res["phi_2"] is None:
            print(f"{name:16s} phi_2=None ({res['reason']})")
        else:
            print(f"{name:16s} phi_2={res['phi_2']:.3f}  sigma_d={res['sigma_d_px']:.2f}px  "
                  f"sigma_pitch={res['sigma_pitch_px']:.2f}px  W={res['mean_char_width']:.1f}px  "
                  f"M={res['num_valid_chars']}  gaps={[round(g) for g in res['gaps_px']]}")
        panels.append(draw_spacing_debug(src, res, scale=1.0))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 4, 0, width - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    _save(args.output, np.vstack(panels))
    print(f"시각화 저장: {args.output}")
