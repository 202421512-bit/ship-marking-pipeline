"""조선소 용접 마킹(석필/분필) 형식의 학습용 PNG를 대량으로 만든다.

실제 현장 사진에서 본 형식
    │                         기준선(용접선 옆 세로선)
    │──────▷  F6.0            지시선 + 화살표 → 윗줄 코드 (F + 숫자)
    │         U5.5            아랫줄 코드 (U + 숫자, 없을 수도 있음)

출력
  standard/  깨끗한 표준 기호 (코드별 1장)       예: F6.0.png, U5.5.png, F6.0_U5.5.png
  train/     손글씨·철판·손상 변형이 들어간 장면  예: 000123.png
  train/labels.csv  파일별 정답 (top, bottom, arrow, ref_line, 손상 정도)

사용법
  python marking_symbols/generate.py 저장폴더 --count 2000
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

SIZE = 256

# ---------------------------------------------------------------- 한 획 글꼴
# 글자 하나를 (0~1, 0~1) 상자 안의 선(점 목록) 몇 개로 정의한다. y는 아래로 증가.

def _arc(cx, cy, rx, ry, a0, a1, n=14):
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, n)]

GLYPHS: dict[str, list[list[tuple[float, float]]]] = {
    "0": [_arc(0.5, 0.5, 0.38, 0.5, 0, 360, 24)],
    "1": [[(0.3, 0.2), (0.55, 0.0), (0.55, 1.0)]],
    "2": [_arc(0.5, 0.28, 0.35, 0.28, 200, 370) + [(0.15, 1.0), (0.9, 1.0)]],
    "3": [_arc(0.48, 0.25, 0.33, 0.25, 210, 450) + _arc(0.48, 0.73, 0.38, 0.27, 270, 500)],
    "4": [[(0.7, 1.0), (0.7, 0.0), (0.1, 0.68), (0.92, 0.68)]],
    "5": [[(0.85, 0.0), (0.28, 0.0), (0.22, 0.45)] + _arc(0.5, 0.68, 0.36, 0.32, 230, 500)],
    "6": [[(0.78, 0.05)] + _arc(0.5, 0.55, 0.36, 0.45, 260, 180, 6)[1:] + _arc(0.5, 0.72, 0.36, 0.28, 180, 540)],
    "7": [[(0.1, 0.0), (0.9, 0.0), (0.4, 1.0)]],
    "8": [_arc(0.5, 0.25, 0.3, 0.25, 90, 450) + _arc(0.5, 0.74, 0.37, 0.26, -90, 270)],
    "9": [_arc(0.5, 0.3, 0.36, 0.3, 0, 360) + [(0.86, 0.3), (0.75, 1.0)]],
    "F": [[(0.25, 1.0), (0.25, 0.0), (0.88, 0.0)], [(0.25, 0.48), (0.72, 0.48)]],
    "U": [[(0.15, 0.0)] + _arc(0.5, 0.62, 0.35, 0.38, 180, 0) + [(0.85, 0.0)]],
    ".": [[(0.5, 0.92), (0.53, 0.97)]],
}
WIDTH = {".": 0.35, "1": 0.55}  # 글자 폭 (기본 0.8)


def text_strokes(text, x, y, h, rng, jitter):
    """문자열을 (x, y) 왼쪽 위에서 높이 h로 쓴 획 목록. jitter가 클수록 손글씨처럼 흔들림."""
    strokes = []
    slant = rng.uniform(-0.35, 0.05) * jitter          # 오른쪽으로 기울어진 필기
    cx = x
    for ch in text:
        w = WIDTH.get(ch, 0.8) * h * rng.uniform(1 - 0.25 * jitter, 1 + 0.25 * jitter)
        sy = h * rng.uniform(1 - 0.15 * jitter, 1 + 0.1 * jitter)
        base = y + rng.normal(0, 0.06 * h * jitter)
        for s in GLYPHS[ch]:
            pts = []
            for px, py in s:
                qx = cx + px * w + (1 - py) * sy * -slant
                qy = base + py * sy
                pts.append((qx + rng.normal(0, 0.03 * h * jitter), qy + rng.normal(0, 0.03 * h * jitter)))
            strokes.append(pts)
        cx += w + h * rng.uniform(0.12, 0.3 + 0.2 * jitter) * (0.4 if ch == "." else 1)
    return strokes, cx


def arrow_strokes(x0, y0, x1, y1, rng, jitter, open_head=True):
    """(x0,y0)에서 (x1,y1)로 가는 지시선과 화살촉."""
    ang = math.atan2(y1 - y0, x1 - x0)
    L = rng.uniform(9, 15)
    spread = math.radians(rng.uniform(25, 40))
    a = (x1 - L * math.cos(ang - spread), y1 - L * math.sin(ang - spread))
    b = (x1 - L * math.cos(ang + spread), y1 - L * math.sin(ang + spread))
    mid = [(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t + rng.normal(0, 1.2 * jitter)) for t in np.linspace(0, 1, 8)]
    head = [a, (x1, y1), b] if open_head else [a, (x1, y1), b, a]
    return [mid, head]


# ---------------------------------------------------------------- 렌더링

def steel_background(rng):
    base = rng.uniform(70, 175)
    tint = np.array([rng.uniform(-8, 6), rng.uniform(-6, 6), rng.uniform(0, 18)])  # BGR: 약간 붉은 녹 기운
    low = cv2.GaussianBlur(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), (0, 0), rng.uniform(15, 40))
    low = low / (np.abs(low).max() + 1e-6) * rng.uniform(10, 35)
    grain = cv2.GaussianBlur(rng.normal(0, rng.uniform(3, 9), (SIZE, SIZE)).astype(np.float32), (0, 0), 0.8)
    img = np.dstack([base + low + grain + t for t in tint])
    # 조명 그라데이션 (자연광, 그림자)
    gx, gy = np.meshgrid(np.linspace(-1, 1, SIZE), np.linspace(-1, 1, SIZE))
    d = rng.uniform(0, 2 * math.pi)
    img *= (1 + rng.uniform(0, 0.35) * (gx * math.cos(d) + gy * math.sin(d)))[..., None]
    return img


def add_distractors(img, rng):
    """용접선, 구멍, 오래된 석필 자국 등 글자와 무관한 것들."""
    if rng.random() < 0.5:                                  # 용접 비드 / 이음매 (어두운 띠)
        x = int(rng.uniform(150, 240)); w = int(rng.uniform(6, 16))
        img[:, x:x + w] *= rng.uniform(0.35, 0.7)
    if rng.random() < 0.3:                                  # 구멍 (어두운 원 + 밝은 테두리)
        c = (int(rng.uniform(-20, 40)), int(rng.uniform(40, 220))); r = int(rng.uniform(20, 40))
        cv2.circle(img, c, r, (40, 40, 40), -1); cv2.circle(img, c, r, (210, 210, 210), 2)
    for _ in range(rng.integers(0, 4)):                     # 희미한 옛 석필 자국
        p = rng.uniform(0, SIZE, 4).astype(int)
        cv2.line(img, (p[0], p[1]), (p[2], p[3]), [img.mean() + rng.uniform(15, 40)] * 3, 1, cv2.LINE_AA)
    return img


def _resample(pts, step=1.5):
    """꺾은선을 일정 간격의 촘촘한 점으로 바꾼다 (모서리 모양은 그대로)."""
    pts = np.asarray(pts, np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-3:
        return pts
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, total, max(2, int(total / step)))
    return np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], 1)


def render(strokes, rng, thick, chalk=True, damage=0.0, wobble=0.0):
    """획을 석필처럼 그린 알파 마스크 (0~1)."""
    m = np.zeros((SIZE, SIZE), np.float32)
    for s in strokes:
        pts = _resample(s)
        if wobble > 0 and len(pts) > 4:                     # 손 떨림: 획을 따라 천천히 변하는 흔들림
            n = cv2.GaussianBlur(rng.normal(0, 1, (len(pts), 2)).astype(np.float32), (1, 0), sigmaX=0, sigmaY=6)
            pts = pts + n / (np.abs(n).max() + 1e-6) * wobble
        cv2.polylines(m, [np.round(pts * 4).astype(np.int32)], False, 1.0,
                      max(1, int(round(thick))), cv2.LINE_AA, shift=2)
    if chalk:                                               # 석필 특유의 거친 질감
        tex = cv2.GaussianBlur(rng.random((SIZE, SIZE)).astype(np.float32), (0, 0), 0.7)
        m *= np.clip(0.55 + tex * 0.7, 0, 1)
    if damage > 0:                                          # 지워짐: 무작위 구간을 문질러 없앰
        er = np.ones_like(m)
        for _ in range(int(damage * 12)):
            c = rng.uniform(0, SIZE, 2).astype(int)
            cv2.ellipse(er, (int(c[0]), int(c[1])), (int(rng.uniform(4, 14)), int(rng.uniform(2, 6))),
                        rng.uniform(0, 180), 0, 360, rng.uniform(0, 0.4), -1)
        m *= cv2.GaussianBlur(er, (0, 0), 2)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 0.6), 0, 1)


def finish(img, rng, glare=False, blur=0.0):
    if glare:                                               # 철판 반사광
        c = rng.uniform(0, SIZE, 2); r = rng.uniform(30, 90)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        img += (np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * r * r)) * rng.uniform(40, 110))[..., None]
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img += rng.normal(0, rng.uniform(1, 5), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(55, 95))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 장면 구성

def random_code(rng, prefix):
    lo, hi = (4.0, 12.0) if prefix == "F" else (3.0, 10.0)
    v = rng.choice(np.arange(lo, hi + 0.01, 0.5))
    return f"{prefix}{v:.1f}"


def compose(rng, top, bottom, arrow, ref_line, jitter, h):
    strokes = []
    tx = rng.uniform(95, 115) if arrow == "left" else rng.uniform(70, 95)
    ty = rng.uniform(80, 120)
    longest = max(top, bottom or "", key=len)
    est = sum(WIDTH.get(c, 0.8) + 0.4 for c in longest) * h + 15
    if tx + est > SIZE - 6:                                 # 긴 코드는 화면 안에 들어오게 글자 크기를 줄임
        h *= (SIZE - 6 - tx) / est
    top_s, top_end = text_strokes(top, tx, ty, h, rng, jitter)
    strokes += top_s
    if bottom:
        b_s, b_end = text_strokes(bottom, tx + rng.uniform(-8, 12), ty + h * rng.uniform(1.4, 1.8), h, rng, jitter)
        strokes += b_s
        if rng.random() < 0.6:                              # 아랫줄 뒤로 길게 끄는 꼬리
            yb = ty + h * 2.5
            strokes.append([(b_end - 4, yb), (b_end + rng.uniform(20, 50), yb + rng.uniform(-4, 8))])
    mid_y = ty + h * 0.5
    if arrow == "left":                                     # 글자 → 왼쪽 기준선을 가리킴 (사진 1)
        line_x = rng.uniform(35, 60)
        strokes += arrow_strokes(tx - 6, mid_y, line_x + 3, mid_y + rng.normal(0, 3), rng, jitter)
    else:                                                   # 기준선 → 글자를 가리킴 (사진 2, 3)
        line_x = rng.uniform(15, 35)
        strokes += arrow_strokes(line_x, mid_y, tx - 6, mid_y + rng.normal(0, 3), rng, jitter)
    if ref_line:
        y0, y1 = rng.uniform(10, 50), rng.uniform(180, 245)
        strokes.append([(line_x + rng.normal(0, 2), y0), (line_x + rng.normal(0, 2), y1)])
    return strokes


def make_standard(out: Path):
    """깨끗한 표준 기호: 흔들림·손상 없음, 어두운 철판색 배경에 흰 획."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    codes = [f"F{v:.1f}" for v in np.arange(4.0, 12.01, 0.5)] + [f"U{v:.1f}" for v in np.arange(3.0, 10.01, 0.5)]
    for code in codes:
        _, end = text_strokes(code, 0, 0, 56, rng, 0.0)
        h = 56 * min(1.0, (SIZE - 40) / end)
        _, end = text_strokes(code, 0, 0, h, rng, 0.0)
        s, _ = text_strokes(code, (SIZE - end) / 2, (SIZE - h) / 2, h, rng, 0.0)
        img = np.full((SIZE, SIZE, 3), 90, np.float32)
        a = render(s, rng, 5.0, chalk=False)
        img = img * (1 - a[..., None]) + 235 * a[..., None]
        cv2.imwrite(str(out / f"{code}.png"), img.astype(np.uint8))
    for top in ("F5.5", "F6.0", "F7.5", "F8.0"):
        for bottom in ("U4.5", "U5.0", "U5.5"):
            for arrow in ("left", "right"):
                s = compose(rng, top, bottom, arrow, True, 0.0, 30)
                img = np.full((SIZE, SIZE, 3), 90, np.float32)
                a = render(s, rng, 3.0, chalk=False)
                img = img * (1 - a[..., None]) + 235 * a[..., None]
                cv2.imwrite(str(out / f"{top}_{bottom}_{arrow}.png"), img.astype(np.uint8))
    return len(list(out.glob("*.png")))


def make_train(out: Path, count: int, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "top", "bottom", "arrow", "ref_line", "damage", "glare", "blur"])
        for i in range(count):
            top = random_code(rng, "F")
            bottom = random_code(rng, "U") if rng.random() < 0.75 else ""
            arrow = "left" if rng.random() < 0.5 else "right"
            ref_line = rng.random() < 0.85
            damage = float(rng.choice([0, 0, 0.3, 0.6, 1.0]))
            glare = bool(rng.random() < 0.25)
            blur = float(rng.choice([0, 0, 0.6, 1.2]))
            strokes = compose(rng, top, bottom, arrow, ref_line, rng.uniform(0.6, 1.4), rng.uniform(22, 34))
            img = add_distractors(steel_background(rng), rng)
            a = render(strokes, rng, rng.uniform(1.5, 3.2), damage=damage, wobble=rng.uniform(0.5, 2.0))
            chalk = rng.uniform(190, 245)
            img = img * (1 - a[..., None]) + chalk * a[..., None]
            img = finish(img, rng, glare=glare, blur=blur)
            name = f"{i:06d}.png"
            cv2.imwrite(str(out / name), img)
            w.writerow([name, top, bottom, arrow, int(ref_line), damage, int(glare), blur])


def main(argv=None):
    ap = argparse.ArgumentParser(description="용접 마킹 학습용 PNG 생성")
    ap.add_argument("out", help="저장 폴더")
    ap.add_argument("--count", type=int, default=2000, help="train 이미지 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    out = Path(args.out)
    n = make_standard(out / "standard")
    make_train(out / "train", args.count, args.seed)
    print(f"표준 기호 {n}장 → {out / 'standard'}")
    print(f"학습 이미지 {args.count}장 + labels.csv → {out / 'train'}")


if __name__ == "__main__":
    main()
