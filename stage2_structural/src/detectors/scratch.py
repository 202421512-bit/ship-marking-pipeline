# -*- coding: utf-8 -*-
"""
detectors/scratch.py
스크래치 검출기 — 후보만 찾고 지우지 않는다.

근거 구분
  [논문 2]  top-hat 필터로 강편 결함 검출 (제목·소장 기록만 확인, 세부 미확인)
  [논문 1]  스크래치의 길이·폭을 측정해 판별 (초록 수준 확인)
  [논문 5]  선의 폭 균일성을 판별 특징으로 사용 (괘선 제거, 초록 수준 확인)
  [일반 기법] 허프 변환 직선 검출
  [실험 파라미터] SCRATCH_CFG 의 수치 — dev 사전 실험으로 정함. 독립 eval로 검증해야 함.

처리
  1. L 채널에 밝은 쪽(white top-hat)·어두운 쪽(black top-hat)을 크기 2개로 적용
  2. 반응을 잡음 수준(MAD)으로 나눠 이진화 → 허프 변환으로 선분
  3. 같은 선을 여러 번 찾은 중복 선분 제거 (큰 top-hat → 긴 것 우선)
  4. 선분마다 특징 계산: 길이, 신호 대 잡음, 폭(수직 단면의 반치폭 FWHM), 폭 균일성(변동계수), 연속성,
     밝기 방향, 주변 마킹과의 색 차이
  5. 확신 여부와 '마킹과 닮음' 여부를 정해 Candidate로 반환
"""
import cv2
import numpy as np

from candidates import Candidate

SCRATCH_CFG = {
    "ref_long_side": 1600,
    "tophat_sizes": (9, 21),        # px @1600, 가는 선 / 굵은 선 [실험 파라미터]
    "bin_k_noise": 4.0,             # 이진화: 반응 > 잡음 × 이 값
    "hough_threshold": 40, "hough_max_gap": 8,
    "snr_min": 6.0, "len_min": 100,          # 후보로 인정 [실험 파라미터, dev 사전 실험]
    "snr_confident": 8.0, "len_confident": 150,   # '확실' 판정 [실험 파라미터]
    "width_cv_max": 0.5,            # 폭(FWHM) 변동계수가 이보다 크면 확실하지 않음 [실험 파라미터]
    "width_cv_min_width": 4.0,      # 이 폭(px) 미만의 가는 선은 폭 균일성 검사를 하지 않음 [실험 파라미터]
    "coverage_min": 0.8,            # 선분 위 이진 픽셀 비율(연속성) 하한 [실험 파라미터]
    "max_frac_in_strong": 0.5,      # 후보 픽셀의 이 비율 이상이 확실한 글자 위면 글자 획으로 보고 제외
    "large_scale_min_width": 6.0,
    "max_width_vs_kernel": 0.8,     # FWHM 폭이 top-hat 크기 × 이 값보다 넓으면 선이 아님   # 큰 top-hat에서는 FWHM 폭이 이 이상인 굵은 선만 후보 (px @1600)
    "dedup_overlap": 0.7,           # 이미 채택된 선분과 이 비율 이상 겹치면 중복
    # 마킹과 닮음 [실험 파라미터]
    "sim_width_ratio": 0.6,         # 선 폭 ≥ 글자 획 폭 × 이 값
    "sim_color_de": 25.0,           # 주변 마킹 평균색과의 LAB 거리가 이 값 이하
    "sim_color_radius": 150,        # 색 비교에 쓰는 주변 마킹 범위 (px @1600)
}


def _scaled(cfg, shape):
    s = max(shape[:2]) / cfg["ref_long_side"]
    c = dict(cfg)
    c["tophat_sizes"] = tuple(max(3, int(round(k * s)) | 1) for k in cfg["tophat_sizes"])
    c["large_scale_min_width"] = cfg["large_scale_min_width"] * s
    for k in ("len_min", "len_confident", "hough_max_gap", "sim_color_radius"):
        c[k] = max(2, int(round(cfg[k] * s)))
    c["hough_threshold"] = max(10, int(round(cfg["hough_threshold"] * s)))
    c["_scale"] = s
    return c


def _sample(p0, p1):
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    xs = np.linspace(p0[0], p1[0], n).round().astype(int)
    ys = np.linspace(p0[1], p1[1], n).round().astype(int)
    return xs, ys


def _profile_widths(resp, p0, p1, n_samples=15, half=15):
    """선분에 수직인 단면을 여러 곳에서 잘라, 반응이 중심값의 절반 이상인 폭(FWHM)을 잰다.
    이진화 마스크의 두께로 재면 흐림 때문에 실제보다 굵게 나와서 이 방법을 쓴다. [우리 방법]"""
    h, w = resp.shape
    d = np.array([p1[0] - p0[0], p1[1] - p0[1]], np.float32)
    ln = float(np.hypot(*d))
    if ln < 1:
        return np.array([])
    d /= ln
    nrm = np.array([-d[1], d[0]], np.float32)
    offs = np.arange(-half, half + 1, dtype=np.float32)
    widths = []
    for t in np.linspace(0.1, 0.9, n_samples):
        c = np.array(p0, np.float32) + t * ln * d
        xs = np.clip((c[0] + offs * nrm[0]).round().astype(int), 0, w - 1)
        ys = np.clip((c[1] + offs * nrm[1]).round().astype(int), 0, h - 1)
        prof = resp[ys, xs]
        peak = prof[half - 2:half + 3].max()
        if peak <= 0:
            continue
        above = prof >= 0.5 * peak
        # 중심에서 양쪽으로 연속된 구간만 센다
        l = half
        while l > 0 and above[l - 1]:
            l -= 1
        r = half
        while r < len(prof) - 1 and above[r + 1]:
            r += 1
        if above[half]:
            widths.append(r - l + 1)
    return np.array(widths, np.float32)


def _mad(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) + 1e-6


def detect(image_bgr, protect, cfg=None):
    cfg = _scaled({**SCRATCH_CFG, **(cfg or {})}, image_bgr.shape)
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    marking_polarity = protect.info.get("polarity", "bright")
    text_w = protect.info.get("text_stroke_width_px", 8.0) or 8.0

    # 주변 마킹 평균색 비교용: 확실한 보호 픽셀의 LAB
    strong = protect.strong
    attached = None
    if getattr(protect, "attached", None) is not None:
        attached = cv2.dilate(protect.attached.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    labf = lab.astype(np.float32)

    raw = []          # (길이, p0, p1, 응답영상, 이진, 거리변환, 잡음, 방향, 크기)
    for ks in cfg["tophat_sizes"]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        for pol, op in (("bright", cv2.MORPH_TOPHAT), ("dark", cv2.MORPH_BLACKHAT)):
            resp = cv2.GaussianBlur(cv2.morphologyEx(L, op, k).astype(np.float32), (0, 0), 1.0)
            noise = _mad(resp)
            binm = resp > cfg["bin_k_noise"] * noise
            dt = cv2.distanceTransform(binm.astype(np.uint8), cv2.DIST_L2, 5)
            hough_in = binm
            if ks > min(cfg["tophat_sizes"]):
                # 굵은 선은 띠 전체에 허프를 쓰면 띠를 비스듬히 가로지르는 선분이 대량으로 나옴 → 중심선(거리변환 능선)에만 적용
                ridge = (dt >= 2) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-3)
                hough_in = cv2.dilate(ridge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            segs = cv2.HoughLinesP(hough_in.astype(np.uint8) * 255, 1, np.pi / 180, cfg["hough_threshold"],
                                   minLineLength=cfg["len_min"], maxLineGap=cfg["hough_max_gap"])
            if segs is None:
                continue
            for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
                length = float(np.hypot(x2 - x1, y2 - y1))
                raw.append((length, (int(x1), int(y1)), (int(x2), int(y2)), resp, binm, dt, noise, pol, ks))

    # 큰 top-hat(굵은 선)부터 처리: 작은 top-hat은 굵은 선의 '양쪽 가장자리'를 가는 선 2개로 잡는데,
    # 이것이 먼저 채택되면 진짜 굵은 선이 중복으로 버려진다 (dev C3 관찰)
    raw.sort(key=lambda r: (-r[-1], -r[0]))
    covered = np.zeros((h, w), bool)
    cands = []
    for length, p0, p1, resp, binm, dt, noise, pol, ks in raw:
        xs, ys = _sample(p0, p1)
        if covered[ys, xs].mean() >= cfg["dedup_overlap"]:
            continue
        snr = float(resp[ys, xs].mean() / noise)
        if snr < cfg["snr_min"]:
            continue
        coverage = float(binm[ys, xs].mean())
        local_w = _profile_widths(resp, p0, p1, half=max(15, ks))
        width = float(np.median(local_w)) if len(local_w) else 1.0
        width_cv = float(np.std(local_w) / (np.mean(local_w) + 1e-6)) if len(local_w) > 3 else 1.0
        if width > cfg["max_width_vs_kernel"] * ks:
            continue      # top-hat 크기보다 넓은 구조는 선이 아니라 덩어리(글자 묶음·조명 경계 등)
        if ks > min(cfg["tophat_sizes"]) and (width < cfg["large_scale_min_width"]
                                              or coverage < cfg["coverage_min"]
                                              or width_cv > cfg["width_cv_max"]):
            continue      # 큰 top-hat은 '연속적이고 폭이 고른 굵은 선' 전용 (사전 실험에서 오검출이 대부분 여기서 나옴)

        line = np.zeros((h, w), np.uint8)
        cv2.line(line, p0, p1, 1, max(1, int(round(width))))
        mask = (line > 0) & binm
        if not mask.any():
            continue
        if (mask & strong).sum() >= cfg["max_frac_in_strong"] * mask.sum():
            continue      # 대부분이 확실한 글자 위 = 글자 획 자체를 선으로 본 것 → 후보 아님
        if attached is not None and (mask & attached).sum() >= cfg["max_frac_in_strong"] * mask.sum():
            continue      # 보호 마스크가 '글자에 딸린 선(화살표 등)'으로 이미 판단한 선 → 후보 아님
        covered |= cv2.dilate(line, np.ones((5, 5), np.uint8)) > 0

        # 마킹과 닮음 판단 [실험 파라미터]
        same_pol = (pol == marking_polarity)
        wide = width >= cfg["sim_width_ratio"] * text_w
        color_close = False
        r = cfg["sim_color_radius"]
        x0, x1b = max(0, min(p0[0], p1[0]) - r), min(w, max(p0[0], p1[0]) + r)
        y0, y1b = max(0, min(p0[1], p1[1]) - r), min(h, max(p0[1], p1[1]) + r)
        near_strong = strong[y0:y1b, x0:x1b] & ~mask[y0:y1b, x0:x1b]
        de = None
        if near_strong.sum() > 30:
            mark_col = labf[y0:y1b, x0:x1b][near_strong].mean(0)
            line_col = labf[mask & ~strong].mean(0) if (mask & ~strong).any() else labf[mask].mean(0)
            de = float(np.linalg.norm(mark_col - line_col))
            color_close = de <= cfg["sim_color_de"]
        similar = same_pol and (wide or color_close)

        # 폭 균일성은 굵은 선에서만 판단: 1~3px 선은 반치폭이 1px 단위로 흔들려 변동계수가 의미 없음 (dev 관찰)
        uniform = width_cv <= cfg["width_cv_max"] or width < cfg["width_cv_min_width"]
        confident = (snr >= cfg["snr_confident"] and length >= cfg["len_confident"]
                     and uniform and coverage >= cfg["coverage_min"])
        cands.append(Candidate(
            kind="scratch", mask=mask, confident=bool(confident), similar_to_marking=bool(similar),
            features={"length": round(length, 1), "snr": round(snr, 2), "width": round(width, 2),
                      "width_cv": round(width_cv, 3), "coverage": round(coverage, 3), "polarity": pol,
                      "tophat_size": int(ks), "color_de_to_marking": None if de is None else round(de, 1),
                      "same_polarity_as_marking": bool(same_pol), "wide_like_stroke": bool(wide)},
            geometry={"p0": list(p0), "p1": list(p1)}))
    return cands
