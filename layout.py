"""
배치(레이아웃) 분석: 실제 사진을 특징 모듈에 넣기 전에 정리한다.

φ₂(간격)·φ₄(기준선)·φ₆(크기) 는 '한 줄짜리 크롭' 을 가정한다. 휴대폰 사진은 보통 여러 줄·여러 열이 한 장에
들어 있고 종이/철판 가장자리가 같이 찍혀서, 그대로 넣으면 한 줄만 남기고 나머지 글자를 '줄 밖 잡음' 으로 버린다.

  1. 영상 경계에 닿는 큰 성분(종이·판 가장자리, 배경 물체) 을 바탕색으로 지운다
  2. 글자 성분을 줄로 묶는다: 세로 중심이 비슷하고(글자 높이의 0.5배 이내) 가로로 가까운(1.0배 이내) 성분끼리
     -> 같은 높이에 나란히 있는 두 열은 가로 간격 때문에 다른 줄이 된다
  3. 줄마다 그 줄의 글자만 남긴 크롭 영상을 만든다 (다른 줄 획이 상자 안으로 들어오면 지움)
"""
import cv2
import numpy as np

from orientation_feature import _remove_specks, binarize

LINE_Y_TOL = 0.5      # 세로 중심 차 < 이 값 × 글자 높이
LINE_X_GAP = 1.0      # 가로 틈 < 이 값 × 글자 높이 (띄어쓰기 ≈ 0.5~0.7, 나란한 두 열 사이는 보통 그보다 넓음)
LINE_PAD = 0.3        # 줄 크롭 여유 = 이 값 × 글자 높이


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def clean_borders(img):
    """경계에 닿는 성분을 지운 회색 영상. (회색 영상, 잉크 마스크, 라벨, 통계, 지운 성분 수)"""
    gray = _to_gray(img).copy()
    ink = _remove_specks(binarize(gray))
    n, lab, st, cents = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    h, w = ink.shape
    bg = float(np.median(gray[~ink])) if (~ink).any() else 255.0
    removed = 0
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            comp = cv2.dilate((lab == i).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            gray[comp] = int(bg)
            ink[lab == i] = False
            removed += 1
    n, lab, st, cents = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    return gray, ink, lab, st, cents, removed, bg


def segment_lines(st, cents):
    """성분을 줄로 묶는다. [줄별 성분 번호 목록] (줄 안은 x 순, 줄은 위에서 아래 순)."""
    ids = [i for i in range(1, len(st))]
    if not ids:
        return []
    heights = st[ids, cv2.CC_STAT_HEIGHT].astype(float)
    char_h = float(np.median(heights[heights >= 0.5 * np.median(heights)]))
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in ids:
        for b in ids:
            if b <= a:
                continue
            if abs(cents[a, 1] - cents[b, 1]) > LINE_Y_TOL * char_h:
                continue
            ax0, ax1 = st[a, 0], st[a, 0] + st[a, 2]
            bx0, bx1 = st[b, 0], st[b, 0] + st[b, 2]
            gap = max(bx0 - ax1, ax0 - bx1, 0)
            if gap <= LINE_X_GAP * char_h:
                parent[find(a)] = find(b)
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    lines = [sorted(g, key=lambda i: cents[i, 0]) for g in groups.values()]
    return sorted(lines, key=lambda g: (np.mean(cents[g, 1]), np.mean(cents[g, 0]))), char_h


def line_image(gray, lab, st, line, char_h, bg):
    """그 줄 성분만 남긴 크롭 (다른 성분은 바탕색으로)."""
    pad = int(LINE_PAD * char_h) + 4
    x0 = max(0, int(min(st[i, 0] for i in line)) - pad)
    y0 = max(0, int(min(st[i, 1] for i in line)) - pad)
    x1 = min(gray.shape[1], int(max(st[i, 0] + st[i, 2] for i in line)) + pad)
    y1 = min(gray.shape[0], int(max(st[i, 1] + st[i, 3] for i in line)) + pad)
    crop = gray[y0:y1, x0:x1].copy()
    lab_c = lab[y0:y1, x0:x1]
    other = (lab_c > 0) & ~np.isin(lab_c, line)
    if other.any():
        crop[cv2.dilate(other.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = int(bg)
    return crop, (x0, y0, x1, y1)


def analyze_layout(img):
    """반환 dict: gray(경계 성분 지운 회색), lines([{ids, box, image}]), char_height, removed_border."""
    gray, ink, lab, st, cents, removed, bg = clean_borders(img)
    if len(st) <= 1:
        return {"gray": gray, "lines": [], "char_height": None, "removed_border": removed}
    groups, char_h = segment_lines(st, cents)
    lines = []
    for g in groups:
        crop, box = line_image(gray, lab, st, g, char_h, bg)
        lines.append({"ids": g, "box": box, "image": crop, "num_components": len(g)})
    return {"gray": gray, "lines": lines, "char_height": char_h, "removed_border": removed}
