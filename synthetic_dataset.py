"""
판별 벤치마크용 합성 데이터셋: 인쇄체 4종 + 수기, 현장 노이즈 공통 적용.

인쇄체 (기계 마킹)
  sans     : 고딕/산세리프 (OpenCV Hershey SIMPLEX/DUPLEX), 굵기·크기 무작위
  stencil  : 산세리프 글자에 스텐실 다리(가로·세로 절단 띠)를 넣어 글자가 조각남
  dots     : DOD 도트 매트릭스. 글자마다 자기 칸의 격자로 찍고 칸 사이를 2 피치 이상 띄움
             (전체를 한 격자로 찍으면 이웃 글자 도트가 붙고 둥근 바닥이 격자 위치에 따라 잘림)
  + 25% 는 줄 전체를 ±15° 회전
수기
  글자마다: 가는 중심선 -> 떨림(법선 방향 매끈한 변위) -> 필압(반경이 매끈하게 변하는 원 찍기)
  -> 방향 전환점(모서리) 잉크 뭉침 -> 글자별 기울기·크기 -> 바닥선·간격 흔들림 -> 확률적 이어 쓰기
  -> 테두리 거칠기
현장 노이즈 (공통): 숏블라스트 요철, 완만한 조도 불균일, 녹 반점, 긁힘 선 (포화 반사광은 다루지 않음)

이 데이터는 판별 특징을 만들 때 쓴 생성기와 같은 계열이라 성능이 실제보다 낙관적으로 나온다.
현장 라벨 사진으로 반드시 다시 검증할 것.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

CHARS = "ABCDEFHKLMNPRSTUVXYZ0123456789"
BG, INK = 200, 40


def random_text(rng, n_min=5, n_max=9):
    n = int(rng.integers(n_min, n_max + 1))
    s = [CHARS[i] for i in rng.integers(0, len(CHARS), n)]
    if n >= 6 and rng.random() < 0.5:                       # 하이픈 또는 띄어쓰기 하나
        s.insert(int(rng.integers(2, n - 2)), "-" if rng.random() < 0.5 else " ")
    return "".join(s)


def _canvas_for(width, height):
    return np.full((height, width), BG, np.uint8)


def _rotate(img, angle):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderValue=BG)


def _char_boxes_line(text, font, scale, thick, gap):
    """한 줄 배치: [(문자, x, 폭, 높이)] 와 전체 폭, 높이."""
    boxes, x, hmax = [], 30, 0
    for ch in text:
        if ch == " ":
            x += int(0.6 * cv2.getTextSize("H", font, scale, thick)[0][1])
            continue
        (w, h), _ = cv2.getTextSize(ch, font, scale, thick)
        boxes.append((ch, x, w, h))
        x += w + gap
        hmax = max(hmax, h)
    return boxes, x + 30, hmax


# ---------------------------------------------------------------------------
# 인쇄체
# ---------------------------------------------------------------------------
def printed_sans(text, rng):
    font = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX][int(rng.integers(0, 2))]
    scale, thick = rng.uniform(1.8, 2.4), int(rng.integers(5, 9))
    boxes, width, h = _char_boxes_line(text, font, scale, thick, gap=int(rng.integers(4, 9)))
    img = _canvas_for(width, int(h * 2.6))
    base = int(h * 1.8)
    for ch, x, w, _ in boxes:
        cv2.putText(img, ch, (x, base), font, scale, INK, thick, cv2.LINE_AA)
    return img


def printed_stencil(text, rng):
    """스텐실: 글자마다 가운데 높이 가로 띠, 넓은 글자는 세로 띠까지 배경색으로 잘라 다리를 만든다."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(1.9, 2.4), int(rng.integers(7, 10))
    boxes, width, h = _char_boxes_line(text, font, scale, thick, gap=int(rng.integers(6, 10)))
    img = _canvas_for(width, int(h * 2.6))
    base = int(h * 1.8)
    bridge = max(2, thick // 2)
    for ch, x, w, ch_h in boxes:
        cv2.putText(img, ch, (x, base), font, scale, INK, thick, cv2.LINE_AA)
        if ch == "-":
            continue
        y_mid = base - ch_h // 2
        cv2.line(img, (x - 2, y_mid), (x + w + 2, y_mid), BG, bridge)
        if w > 0.55 * ch_h:
            cv2.line(img, (x + w // 2, base - ch_h - 4), (x + w // 2, base + 4), BG, bridge)
    return img


def printed_dots(text, rng):
    """DOD 도트 매트릭스: 글자 칸마다 같은 격자 원점에서 도트를 찍고, 칸 사이를 2 피치 띄운다."""
    pitch = int(rng.integers(6, 9))
    radius = max(2, int(round(pitch * 0.42)))
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(1.9, 2.3), int(rng.integers(6, 9))
    (wh, hh), _ = cv2.getTextSize("H", font, scale, thick)
    cell_w = int(np.ceil((wh + thick) / pitch)) * pitch
    width = 30 + len(text) * (cell_w + 2 * pitch) + 30
    img = _canvas_for(width, int(hh * 2.6))
    base = int(hh * 1.8)
    x = 30
    for ch in text:
        if ch != " ":
            tile = np.zeros((hh + 2 * thick + 2, cell_w + thick), np.uint8)
            (w, _), _ = cv2.getTextSize(ch, font, scale, thick)
            cv2.putText(tile, ch, ((cell_w - w) // 2, hh + thick), font, scale, 255, thick, cv2.LINE_AA)
            ys, xs = np.mgrid[pitch // 2:tile.shape[0]:pitch, pitch // 2:tile.shape[1]:pitch]
            on = tile[ys, xs] > 127
            for yy, xx in zip(ys[on], xs[on]):
                cv2.circle(img, (x + int(xx), base - hh - thick + int(yy)), radius, INK, -1, cv2.LINE_AA)
        x += cell_w + 2 * pitch
    return img


PRINTED_KINDS = {"sans": printed_sans, "stencil": printed_stencil, "dots": printed_dots}


# ---------------------------------------------------------------------------
# 수기
# ---------------------------------------------------------------------------
def _hand_glyph(ch, rng, scale, base_r, tremor, tilt):
    """한 글자: 중심선 -> 떨림 -> 필압 원 찍기 -> 모서리 잉크 뭉침 -> 기울이기. (타일, 시작점, 끝점) 반환."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(ch, font, scale, 2)
    pad = int(base_r * 3 + 12)
    thin = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.putText(thin, ch, (pad, h + pad), font, scale, 255, 2, cv2.LINE_AA)
    ys, xs = np.nonzero(skeletonize(thin > 127))     # 1px 중심선 (2px 띠에 찍으면 획이 띠 폭만큼 굵어짐)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    # 떨림: 매끈한 무작위 변위장 (파장 8~14px)
    hh, ww = thin.shape
    field = [cv2.GaussianBlur(rng.normal(size=(hh, ww)).astype(np.float32), (0, 0), rng.uniform(2, 3.5))
             for _ in range(2)]
    field = [f / (f.std() + 1e-9) * tremor for f in field]
    pts[:, 0] += field[0][ys, xs]
    pts[:, 1] += field[1][ys, xs]
    # 필압: 반경이 매끈하게 변함 (±40%)
    pressure = cv2.GaussianBlur(rng.normal(size=(hh, ww)).astype(np.float32), (0, 0), 5)
    pressure /= pressure.std() + 1e-9
    tile = np.full(thin.shape, BG, np.uint8)
    for (px, py), p in zip(pts, pressure[ys, xs]):
        r = base_r * (1 + 0.4 * np.clip(p, -1.5, 1.5) / 1.5)
        cv2.circle(tile, (int(round(px)), int(round(py))), max(1, int(round(r))), INK, -1, cv2.LINE_AA)
    # 방향 전환점 잉크 뭉침: 모서리 검출 지점 일부에 굵은 점
    corners = cv2.goodFeaturesToTrack(thin, 6, 0.2, base_r * 3)
    if corners is not None:
        for cx, cy in corners[:, 0]:
            if rng.random() < 0.5:
                cv2.circle(tile, (int(cx), int(cy)), int(round(base_r * rng.uniform(1.2, 1.5))), INK, -1, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((ww / 2, hh / 2), tilt, 1.0)
    tile = cv2.warpAffine(tile, M, (ww, hh), borderValue=BG)
    start = M @ np.array([pad, h + pad - h * 0.5, 1.0])
    end = M @ np.array([pad + w, h + pad - 2, 1.0])
    return tile, start, end, pad


def handwritten(text, rng, link_prob=None):
    from contour_roughness_feature import roughen_edges

    scale = rng.uniform(1.8, 2.3)
    base_r = rng.uniform(1.8, 3.0)                   # 기본 획 반폭 (인쇄체 굵기 5~9px 와 비슷한 범위)
    tremor = rng.uniform(0.5, 1.4)
    tilt_std = rng.uniform(3, 9)
    jitter = rng.uniform(3, 8)
    link_prob = rng.uniform(0.0, 0.6) if link_prob is None else link_prob
    (_, h), _ = cv2.getTextSize("H", cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    width = int(len(text) * h * 1.0 + 80)
    img = _canvas_for(width, int(h * 3.0))
    base = int(h * 2.0)
    x, prev_end = 30, None
    for ch in text:
        if ch == " ":
            x += int(h * 0.6)
            prev_end = None
            continue
        s = scale * rng.uniform(0.85, 1.15)
        tile, start, end, pad = _hand_glyph(ch, rng, s, base_r, tremor, rng.normal(0, tilt_std))
        th, tw = tile.shape
        y0 = int(base - (th - pad) + rng.normal(0, jitter))
        y0 = int(np.clip(y0, 0, img.shape[0] - th))
        if x + tw > img.shape[1]:
            break
        region = img[y0:y0 + th, x:x + tw]
        np.minimum(region, tile, out=region)
        if prev_end is not None and ch != "-" and rng.random() < link_prob:
            cv2.line(img, prev_end, (int(x + start[0]), int(y0 + start[1])), INK, max(2, int(base_r * 1.6)),
                     cv2.LINE_AA)
        prev_end = (int(x + end[0]), int(y0 + end[1]))
        x += tw - 2 * pad + int(rng.uniform(4, 6 + 2 * jitter))
    img = img[:, : min(img.shape[1], x + 40)]
    return roughen_edges(img, amount=rng.uniform(0.1, 0.25), bleed=0.8, seed=int(rng.integers(1 << 30)),
                         background=BG, ink=INK)


# ---------------------------------------------------------------------------
# 현장 노이즈
# ---------------------------------------------------------------------------
def field_noise(img, rng):
    """숏블라스트 요철(고주파 잡음 + 미세 패턴), 완만한 조도 불균일, 녹 반점, 긁힘."""
    h, w = img.shape
    out = img.astype(np.float32)
    out += rng.normal(0, rng.uniform(5, 10), out.shape)
    out += cv2.GaussianBlur(rng.normal(0, 12, out.shape).astype(np.float32), (0, 0), 1.2)
    yy, xx = np.mgrid[0:h, 0:w]
    out += rng.uniform(-20, 20) * (xx / w - 0.5)                   # 완만한 조도 기울기
    for _ in range(int(rng.integers(30, 90))):                     # 녹 반점
        cv2.circle(out, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 3)),
                   float(rng.uniform(60, 110)), -1)
    for _ in range(int(rng.integers(0, 3))):                       # 긁힘
        y1, y2 = rng.uniform(0, h, 2)
        cv2.line(out, (0, int(y1)), (w - 1, int(y2)), float(rng.uniform(80, 150)), 1, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)


MARKING_COLORS = {"yellow": (20, 205, 235), "white": (232, 232, 228), "black": (38, 38, 42)}   # BGR


def colorize(gray, marking, rng, rust_patches=(3, 8)):
    """회색 합성 영상(잉크 < 120) -> 컬러 강판: 청회색 강판 + 블라스트 질감, 적갈색 녹 얼룩, 마킹 색.
    (컬러 영상, 정답 잉크 마스크) 반환."""
    h, w = gray.shape
    ink = gray < 120
    steel = np.array([128, 124, 118], np.float32) + rng.normal(0, 3, 3)
    out = np.ones((h, w, 3), np.float32) * steel
    out += cv2.GaussianBlur(rng.normal(0, 10, (h, w)).astype(np.float32), (0, 0), 1.0)[..., None]   # 블라스트 질감
    for _ in range(int(rng.integers(*rust_patches))):               # 녹 얼룩 (회색으로 보면 어두워 잉크처럼 보임)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.03, 0.09) * w
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xx - cx) / r) ** 2 + ((yy - cy) / (r * rng.uniform(0.5, 1.5))) ** 2))
        blob *= cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2) > 0.45
        rust = np.array([35, 70, 135], np.float32)
        out = out * (1 - blob[..., None] * 0.85) + rust * blob[..., None] * 0.85
    color = np.array(MARKING_COLORS[marking], np.float32)
    soft = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 0.7)[..., None]
    out = out * (1 - soft) + color * soft
    out += rng.normal(0, 4, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), ink


def make_sample(label, rng, kind=None):
    """label: 0 = 인쇄체, 1 = 수기. (영상, 메타) 반환."""
    text = random_text(rng)
    meta = {"label": label, "text": text}
    if label == 0:
        kind = kind or ["sans", "stencil", "dots"][int(rng.integers(0, 3))]
        img = PRINTED_KINDS[kind](text, rng)
        if rng.random() < 0.25:
            angle = rng.uniform(-15, 15)
            img = _rotate(img, angle)
            meta["angle"] = float(angle)
        meta["kind"] = kind
    else:
        img = handwritten(text, rng)
        meta["kind"] = "handwritten"
    return field_noise(img, rng), meta


def make_dataset(n_per_class=100, seed=0):
    """[(영상, 메타)] 인쇄체 n + 수기 n. 인쇄체 종류는 고르게."""
    rng = np.random.default_rng(seed)
    data = []
    kinds = ["sans", "stencil", "dots"]
    for i in range(n_per_class):
        data.append(make_sample(0, rng, kinds[i % 3]))
    for _ in range(n_per_class):
        data.append(make_sample(1, rng))
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="합성 데이터 예시 저장")
    parser.add_argument("-o", "--output", default="synthetic_examples.png")
    args = parser.parse_args()
    rng = np.random.default_rng(1)
    rows = [make_sample(0, rng, k)[0] for k in ("sans", "stencil", "dots")]
    rows.append(_rotate(printed_sans("B12-SP3", rng), 12))
    rows += [make_sample(1, rng)[0] for _ in range(3)]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 2, 2, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=255) for r in rows]
    ok, buf = cv2.imencode(".png", np.vstack(rows))
    buf.tofile(args.output)
    print(f"저장: {args.output}")
