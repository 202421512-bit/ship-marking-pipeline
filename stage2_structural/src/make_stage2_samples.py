# -*- coding: utf-8 -*-
"""
make_stage2_samples.py
2차 노이즈 제거 평가용 가상 데이터 생성기

[중요] 이 생성기는 논문 내용이 아닙니다. 2차 알고리즘의 성능을 정답과 비교해
측정하기 위해 우리가 만든 평가 도구입니다. 아래 수치(스크래치 개수·길이·폭 등)는
모두 [우리 설정]입니다. 1차 폴더(stage1_denoise)의 코드는 사용하지도, 수정하지도 않습니다.

데이터셋 구분
  dev  : 개발·기능 검증용 (기본 시드 1000번대)
  eval : 독립 평가용   (기본 시드 900000번대, dev와 절대 겹치지 않음)
  → 알고리즘의 규칙과 임계값은 dev로만 정하고, 최종 성능은 eval로만 보고합니다.

중첩 조건
  C1_no_overlap : 스크래치가 보호 대상(문자·기호·직선 마킹)과 겹치지 않음
  C2_partial    : 모든 스크래치가 문자를 최소 1번 가로지름. 획보다 가늘고 밝기가 다름
  C3_hard       : 스크래치의 폭·밝기가 획과 같음 + 직선 획 문자 다수 + 긴 굵은 직선 마킹

이미지 한 장마다 저장하는 파일 (data/<split>/<조건>/<번호>/)
  image.png               입력 이미지
  gt_text.png             문자 획
  gt_symbol.png           보호해야 하는 기호 (원, 화살표)
  gt_marking_line.png     긴 굵은 직선 마킹 (C3만. 분필선·용접선으로 가정 → 보호 대상)
  gt_protect.png          보호 대상 전체 = 문자 ∪ 기호 ∪ 직선 마킹
  gt_scratch.png          스크래치 (보호 대상과 겹친 부분 포함)
  gt_preexisting_damage.png  보호 대상 ∩ 스크래치 = 입력 단계에서 이미 손상된 획 픽셀
  gt_protect_intact.png   보호 대상 − 스크래치 = 입력에서 온전한 획 픽셀
                          → 알고리즘이 이 픽셀을 바꾸면 '알고리즘이 추가로 만든 손상'
  gt_glare.png            반사로 밝기가 포화된 영역 (일부 이미지만, 없으면 빈 마스크)
  meta.json               시드, 조건, 정답 문자열, 마킹별 박스, 스크래치별 정보 등

실행 (저장소 최상위 폴더에서)
  python stage2_structural/src/make_stage2_samples.py                 # dev, 조건별 10장
  python stage2_structural/src/make_stage2_samples.py --split eval --n 30
"""

import argparse
import json
import os

import cv2
import numpy as np

# ------------------------------------------------------------------
# 기본 설정 [우리 설정]
# ------------------------------------------------------------------
H, W = 1200, 1600
CONDITIONS = ("C1_no_overlap", "C2_partial", "C3_hard")
SEED_BASE = {"dev": 1000, "eval": 900000}      # 두 구간은 절대 겹치지 않음
SEED_STRIDE = {"C1_no_overlap": 0, "C2_partial": 300, "C3_hard": 600}   # 조건별 시드 구간
MAX_PER_CONDITION = 300                         # 위 구간 간격과 맞춤
GLARE_SEED_MOD = (2, 5, 8)                      # 시드 끝자리가 이 값이면 반사 추가 → 조건마다 30%
GLARE_LEVEL = 250                               # 이 밝기 이상이 '포화'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX]
PAINT_COLORS = [(235, 235, 235), (60, 215, 240)]   # 흰색, 노란색 (B, G, R)

# 정답 문자열 후보 (현장 표기 흉내). C3는 직선 획이 많은 문자열을 섞음
CODES_NORMAL = ["B12-P3", "FR45", "No.7", "S203", "BLK 5A", "P-18", "W6", "R10", "V20", "A3-2"]
CODES_STRAIGHT = ["1-11", "IL-7", "L/14", "117", "T-1", "I/L", "11-41", "H-17"]


# ------------------------------------------------------------------
# 1. 철판 바탕
# ------------------------------------------------------------------
def make_plate(rng):
    base = np.full((H, W, 3), rng.uniform(100, 135), np.float32)
    base += rng.normal(0, 3, (1, 1, 3)).astype(np.float32)          # 약간의 색조 차이
    grain = rng.normal(0, 5, (H, W, 1)).astype(np.float32)
    streak = cv2.blur(rng.normal(0, 14, (H, W)).astype(np.float32), (61, 1))[..., None]
    img = base + grain + streak
    # 약한 녹 얼룩 (2차 MVP 대상 아님. 배경 현실감용)
    n = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 25)
    n = (n - n.min()) / (np.ptp(n) + 1e-6)
    a = (np.clip((n - 0.88) / 0.08, 0, 1) * 0.5)[..., None]
    img = img * (1 - a) + np.array([40, 75, 140], np.float32) * a
    return img


# ------------------------------------------------------------------
# 2. 보호 대상: 문자, 기호, 직선 마킹
# ------------------------------------------------------------------
def draw_arrow(layer, p0, p1, thick):
    cv2.arrowedLine(layer, p0, p1, 255, thick, cv2.LINE_AA, tipLength=0.25)


def place_markings(rng, condition):
    """문자열 마킹 3~4개를 겹치지 않게 배치. 마킹마다 문자·기호 마스크와 박스를 돌려줌."""
    n_mark = int(rng.integers(3, 5))
    codes = CODES_STRAIGHT if condition == "C3_hard" else CODES_NORMAL
    occupied = np.zeros((H, W), np.uint8)
    marks = []
    tries = 0
    while len(marks) < n_mark and tries < 200:
        tries += 1
        text = str(rng.choice(codes))
        if condition == "C3_hard" and rng.random() < 0.4:
            text = str(rng.choice(CODES_NORMAL))           # C3에도 일반 문자열 일부 섞음
        font = FONTS[int(rng.integers(len(FONTS)))]
        scale = float(rng.uniform(2.0, 2.8))
        thick = int(rng.integers(6, 10))
        angle = float(rng.uniform(-15, 15))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cx = int(rng.integers(tw // 2 + 120, W - tw // 2 - 220))
        cy = int(rng.integers(th + 140, H - th - 140))

        t_layer = np.zeros((H, W), np.uint8)
        s_layer = np.zeros((H, W), np.uint8)
        cv2.putText(t_layer, text, (cx - tw // 2, cy + th // 2), font, scale, 255, thick, cv2.LINE_AA)
        sym = str(rng.choice(["circle", "arrow", "none"]))
        if sym == "circle":
            cv2.ellipse(s_layer, (cx + tw // 2 + 55, cy), (32, 32), 0, 0, 330, 255, max(thick - 2, 3), cv2.LINE_AA)
        elif sym == "arrow":
            draw_arrow(s_layer, (cx - tw // 2, cy + th + 40), (cx + tw // 2, cy + th + 40), max(thick - 2, 3))
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        t_layer = cv2.warpAffine(t_layer, M, (W, H))
        s_layer = cv2.warpAffine(s_layer, M, (W, H))
        both = cv2.threshold(np.maximum(t_layer, s_layer), 127, 255, cv2.THRESH_BINARY)[1]
        if both.sum() == 0:
            continue
        # 다른 마킹과 겹치거나 너무 가까우면 다시 배치
        grown = cv2.dilate(both, np.ones((61, 61), np.uint8))
        if (grown > 0)[occupied > 0].any():
            continue
        occupied = np.maximum(occupied, grown)
        ys, xs = np.nonzero(both)
        tx1, ty1, tx2, ty2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        w_meas = stroke_width(t_layer[ty1:ty2, tx1:tx2] > 127)
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        marks.append({
            "text": text, "symbol": sym, "font": int(font), "scale": round(scale, 3),
            "stroke_thickness_px": thick, "stroke_width_measured_px": round(w_meas, 2), "angle_deg": round(angle, 2),
            "color_bgr": list(PAINT_COLORS[int(rng.integers(len(PAINT_COLORS)))]),
            "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "box_rotated": [[round(v, 1) for v in p] for p in cv2.boxPoints(rect).tolist()],
            "_t": t_layer, "_s": s_layer,
        })
    return marks


def draw_marking_line(rng, occupied):
    """C3 전용: 이미지를 가로지르는 긴 굵은 직선 마킹 (분필선·용접선 가정).
    문자를 덮어 가리지 않도록 문자 주변(occupied)을 피해서 배치한다."""
    for _ in range(200):
        layer = np.zeros((H, W), np.uint8)
        y0 = int(rng.integers(int(H * 0.15), int(H * 0.85)))
        y1 = y0 + int(rng.integers(-40, 40))
        width = int(rng.integers(10, 15))
        cv2.line(layer, (0, y0), (W - 1, y1), 255, width, cv2.LINE_AA)
        if not ((layer > 0) & (occupied > 0)).any():
            return layer, {"p0": [0, y0], "p1": [W - 1, y1], "width_px": width}
    return layer, {"p0": [0, y0], "p1": [W - 1, y1], "width_px": width, "warning": "overlaps_text"}


# ------------------------------------------------------------------
# 3. 스크래치
# ------------------------------------------------------------------
def stroke_width(mask):
    """마스크의 대표 선 폭(px) = 거리 변환 값 90번째 백분위 × 2. [우리 측정 방법]
    글자 획과 스크래치를 같은 방법으로 재서 '폭이 같은지' 비교하는 데 쓴다."""
    mask = mask.astype(bool)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(2 * np.percentile(dt[mask], 90))


def fit_thickness(p0, p1, target_width):
    """이 선분을 cv2.line으로 그렸을 때 실제 폭(stroke_width로 측정)이 target_width에
    가장 가까워지는 두께 값을 찾는다. 글자와 직선은 같은 두께 값이라도 실제 폭이
    다르고, 선의 기울기에 따라서도 달라지므로 실제로 그려서 맞춘다. [우리 방법]"""
    x0, y0 = min(p0[0], p1[0]), min(p0[1], p1[1])
    pad = 30
    q0 = (int(p0[0] - x0 + pad), int(p0[1] - y0 + pad))
    q1 = (int(p1[0] - x0 + pad), int(p1[1] - y0 + pad))
    canvas_hw = (int(abs(p1[1] - p0[1])) + 2 * pad + 1, int(abs(p1[0] - p0[0])) + 2 * pad + 1)
    best, best_err = 1, 1e9
    for t in range(max(1, int(target_width) - 5), int(target_width) + 4):
        lay = np.zeros(canvas_hw, np.uint8)
        cv2.line(lay, q0, q1, 255, t, cv2.LINE_AA)
        err = abs(stroke_width(lay > 127) - target_width)
        if err < best_err:
            best, best_err = t, err
    return best


def scratch_layer(p0, p1, width):
    layer = np.zeros((H, W), np.uint8)
    cv2.line(layer, tuple(map(int, p0)), tuple(map(int, p1)), 255, int(width), cv2.LINE_AA)
    return layer


def random_segment(rng, length, through=None):
    ang = rng.uniform(0, np.pi)
    if through is None:
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
    else:
        cx, cy = through
    dx, dy = np.cos(ang) * length / 2, np.sin(ang) * length / 2
    return (cx - dx, cy - dy), (cx + dx, cy + dy), float(np.degrees(ang))


def make_scratches(rng, condition, protect, marks):
    """조건 규칙에 맞는 스크래치 목록을 만든다. 각 항목: 마스크, 폭, 색, 겹친 픽셀 수."""
    out = []
    keep_out = cv2.dilate(protect, np.ones((51, 51), np.uint8)) > 0     # C1용 여백 25px
    if condition == "C1_no_overlap":
        target, tries = int(rng.integers(8, 15)), 0
        while len(out) < target and tries < 500:
            tries += 1
            width = int(rng.integers(1, 3))
            p0, p1, ang = random_segment(rng, rng.uniform(150, 600))
            m = scratch_layer(p0, p1, width)
            if (m > 0)[keep_out].any():
                continue
            out.append(dict(mask=m, width_px=width, mode="contrast", angle=ang, p0=p0, p1=p1))
    else:
        n_cross = int(rng.integers(4, 8))
        for _ in range(n_cross):
            mk = marks[int(rng.integers(len(marks)))]
            ys, xs = np.nonzero(mk["_t"] > 127)
            k = int(rng.integers(len(xs)))                                # 획 위의 한 점을 지나가게
            length = rng.uniform(250, 700)
            p0, p1, ang = random_segment(rng, length, through=(xs[k], ys[k]))
            if condition == "C2_partial":
                width, mode = int(rng.integers(1, 3)), "contrast"
            else:
                width, mode = fit_thickness(p0, p1, mk["stroke_width_measured_px"]), "like_stroke"
            m = scratch_layer(p0, p1, width)
            out.append(dict(mask=m, width_px=width, mode=mode, angle=ang, p0=p0, p1=p1,
                            like_mark=mk["text"], stroke_thickness_px=mk["stroke_thickness_px"],
                            color_bgr=mk["color_bgr"]))
        # 글자를 지나지 않는 배경 스크래치도 몇 개 (C2는 대비형, C3는 획 닮은형)
        for _ in range(int(rng.integers(3, 7))):
            for _try in range(100):
                p0, p1, ang = random_segment(rng, rng.uniform(150, 600))
                if condition == "C2_partial":
                    width, mode = int(rng.integers(1, 3)), "contrast"
                else:
                    ref = marks[int(rng.integers(len(marks)))]
                    width, mode = fit_thickness(p0, p1, ref["stroke_width_measured_px"]), "like_stroke"
                m = scratch_layer(p0, p1, width)
                if not (m > 0)[keep_out].any():
                    item = dict(mask=m, width_px=width, mode=mode, angle=ang, p0=p0, p1=p1)
                    if mode == "like_stroke":
                        item.update(like_mark=ref["text"], stroke_thickness_px=ref["stroke_thickness_px"],
                                    color_bgr=ref["color_bgr"])
                    out.append(item)
                    break
    for s in out:
        s["overlap_px"] = int(((s["mask"] > 127) & (protect > 0)).sum())
    return out


# ------------------------------------------------------------------
# 4. 조명·반사·카메라
# ------------------------------------------------------------------
def lighting(img, rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 1.0 + 0.25 * ((xx / W) - 0.5) * 2 * rng.choice([-1, 1])
    return img * g[..., None]


def add_glare(img, rng):
    cx, cy = rng.integers(W // 5, 4 * W // 5), rng.integers(H // 5, 4 * H // 5)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = ((xx - cx) / rng.uniform(150, 280)) ** 2 + ((yy - cy) / rng.uniform(90, 170)) ** 2
    return img + 255 * rng.uniform(0.8, 1.1) * np.exp(-d)[..., None], [int(cx), int(cy)]


# ------------------------------------------------------------------
# 5. 한 장 생성
# ------------------------------------------------------------------
def make_one(seed, condition):
    rng = np.random.default_rng(seed)
    img = make_plate(rng)

    marks = place_markings(rng, condition)
    gt_text = np.zeros((H, W), np.uint8)
    gt_symbol = np.zeros((H, W), np.uint8)
    for mk in marks:
        alpha = (np.maximum(mk["_t"], mk["_s"]).astype(np.float32) / 255.0)[..., None]
        img = img * (1 - alpha) + np.array(mk["color_bgr"], np.float32) * alpha
        gt_text = np.maximum(gt_text, (mk["_t"] > 127).astype(np.uint8) * 255)
        gt_symbol = np.maximum(gt_symbol, (mk["_s"] > 127).astype(np.uint8) * 255)

    gt_line = np.zeros((H, W), np.uint8)
    line_info = None
    if condition == "C3_hard":
        occ = np.zeros((H, W), np.uint8)
        for mk in marks:
            occ = np.maximum(occ, cv2.dilate(np.maximum(mk["_t"], mk["_s"]), np.ones((31, 31), np.uint8)))
        lay, line_info = draw_marking_line(rng, occ)
        a = (lay.astype(np.float32) / 255.0)[..., None]
        img = img * (1 - a) + np.array([225, 230, 230], np.float32) * a
        gt_line = (lay > 127).astype(np.uint8) * 255

    gt_protect = np.maximum(np.maximum(gt_text, gt_symbol), gt_line)

    # 스크래치는 보호 대상 '위에' 그림 → 겹친 픽셀은 입력에서 이미 손상됨
    scratches = make_scratches(rng, condition, gt_protect, marks)
    gt_scratch = np.zeros((H, W), np.uint8)
    for s in scratches:
        a = (s["mask"].astype(np.float32) / 255.0)[..., None]
        if s["mode"] == "contrast":
            delta = rng.choice([-1.0, 1.0]) * rng.uniform(30, 60)
            img = img + delta * a
            s["delta"] = round(float(delta), 1)
        else:   # 획과 같은 색 (구분 어려움)
            img = img * (1 - a) + np.array(s["color_bgr"], np.float32) * a
        gt_scratch = np.maximum(gt_scratch, (s["mask"] > 127).astype(np.uint8) * 255)

    img = lighting(img, rng)
    glare_center = None
    if seed % 10 in GLARE_SEED_MOD:
        img, glare_center = add_glare(img, rng)
    img = cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 3, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 반사 정답: 반사를 넣은 이미지에서 밝기가 포화된 픽셀
    gt_glare = np.zeros((H, W), np.uint8)
    if glare_center is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt_glare = (gray >= GLARE_LEVEL).astype(np.uint8) * 255

    pre = ((gt_protect > 0) & (gt_scratch > 0)).astype(np.uint8) * 255
    intact = ((gt_protect > 0) & (gt_scratch == 0)).astype(np.uint8) * 255

    meta = {
        "generator": "make_stage2_samples.py (우리 평가 도구, 논문 아님)",
        "seed": int(seed), "condition": condition, "size_hw": [H, W],
        "markings": [{k: v for k, v in m.items() if not k.startswith("_")} for m in marks],
        "gt_text_strings": [m["text"] for m in marks],
        "marking_line": line_info,
        "scratches": [{k: (round(v, 1) if isinstance(v, float) else v)
                       for k, v in s.items() if k not in ("mask", "p0", "p1")}
                      | {"p0": [round(float(c), 1) for c in s["p0"]], "p1": [round(float(c), 1) for c in s["p1"]]}
                      for s in scratches],
        "glare_center": glare_center,
        "pixel_counts": {
            "protect": int((gt_protect > 0).sum()), "scratch": int((gt_scratch > 0).sum()),
            "preexisting_damage": int((pre > 0).sum()), "protect_intact": int((intact > 0).sum()),
            "glare": int((gt_glare > 0).sum()),
            "protect_in_glare": int(((gt_protect > 0) & (gt_glare > 0)).sum()),
        },
    }
    masks = {"gt_text": gt_text, "gt_symbol": gt_symbol, "gt_marking_line": gt_line,
             "gt_protect": gt_protect, "gt_scratch": gt_scratch,
             "gt_preexisting_damage": pre, "gt_protect_intact": intact, "gt_glare": gt_glare}
    return img, masks, meta


def seed_for(split, condition, i):
    if i >= MAX_PER_CONDITION:
        raise ValueError(f"조건당 최대 {MAX_PER_CONDITION}장")
    return SEED_BASE[split] + SEED_STRIDE[condition] + i


def generate(split="dev", n=10, out_root=DATA_DIR, conditions=CONDITIONS, verbose=True):
    manifest = []
    for cond in conditions:
        for i in range(n):
            seed = seed_for(split, cond, i)
            img, masks, meta = make_one(seed, cond)
            d = os.path.join(out_root, split, cond, f"{i:03d}")
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "image.png"), img)
            for k, m in masks.items():
                cv2.imwrite(os.path.join(d, k + ".png"), m)
            meta["split"] = split
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            manifest.append({"path": os.path.relpath(d, out_root), "seed": seed, "condition": cond})
            if verbose:
                print(f"{split}/{cond}/{i:03d}  seed={seed}  scratch={meta['pixel_counts']['scratch']}"
                      f"  pre_damage={meta['pixel_counts']['preexisting_damage']}"
                      f"  glare={'Y' if meta['glare_center'] else '-'}")
    os.makedirs(os.path.join(out_root, split), exist_ok=True)
    with open(os.path.join(out_root, split, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SEED_BASE), default="dev")
    ap.add_argument("--n", type=int, default=10, help="조건당 장수")
    args = ap.parse_args()
    generate(args.split, args.n)
    print("\n저장 위치:", os.path.join(DATA_DIR, args.split))


if __name__ == "__main__":
    main()
