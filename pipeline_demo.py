"""
조선소 선박 블록 철판 마킹 분류 데모 (OpenCV + NumPy only)

정형 마킹(잉크젯/타각)은 물리적으로 규칙성이 엄격하다는 점을 이용한 "통과 필터(Pass-Filter)" 방식:
  - 3개 게이트(베이스라인 정렬 / 획 두께 균일성 / 무채색 도료)를 모두 통과한 RoI만 STRUCTURED 로 확정
  - 하나라도 불통과 시 기하 통계(Solidity, 종횡비)로 SYMBOL 또는 UNSTRUCTURED 로 분기

처리 순서: 로드 -> 전처리(전경 마스크) -> 기울기 보정 -> RoI 추출 -> 분류 -> 원본 위 시각화

사용법:
  python pipeline_demo.py                         # test_plate.jpg (없으면 합성 샘플을 만들어 저장)
  python pipeline_demo.py my_plate.jpg -o out.jpg --no-show
"""
import argparse
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 임계값
# ---------------------------------------------------------------------------
# 1) 기울기 보정
DESKEW_MIN_ANGLE = 2.5      # deg. |θ| 가 이 값을 넘을 때만 warpAffine 으로 수평 보정

# 2) 정형 문자 게이트 (모두 통과해야 STRUCTURED)
BASELINE_STD_MAX = 2.5      # px.  Ha et al. (ICDAR 1995, Recursive X-Y Cut) - 라인 바닥 y 정렬
STROKE_CV_MAX = 0.28        # σ/μ. Epshtein et al. (CVPR 2010, SWT) - 획 두께 변동계수
SAT_MAX = 45                # HSV S(0~255) 전경 평균. 무채색/흑색 기계 잉크
MIN_CHARS_PER_LINE = 3      # 베이스라인 통계에 필요한 최소 문자 수 (미만이면 정형으로 확정 불가)
BASELINE_DETREND_MIN = 5    # 문자가 이 이상이면 잔여 기울기를 1차 회귀로 제거한 뒤 표준편차 계산

# 3) 기호(SYMBOL) 분기
SOLIDITY_MIN = 0.40         # 외곽선 면적 / Convex Hull 면적
SYMBOL_AR_MAX = 8.0         # 최소 외접 사각형 긴 변 / 짧은 변 이 이 값 이상이면 극단적 종횡비
SYMBOL_DOMINANT_RATIO = 0.8 # RoI 잉크의 80% 이상이 한 덩어리일 때만 기호 후보 (여러 글자 라인 제외)

# 전처리 / RoI
BG_KERNEL = 51              # 배경(강판) 밝기 추정용 median 커널. 문자 크기보다 커야 함
FG_DIFF_MIN = 40            # 배경 대비 밝기 차 (흑색 잉크, 흰 분필 모두 검출)
FG_SAT_MIN = 90             # 유채색 마커는 밝기 차가 작아도 전경으로
SAT_DARK_V = 50             # V 가 이보다 어두운 픽셀은 S 를 0 으로 간주 (HSV S 가 수치적으로 불안정)
MIN_COMPONENT_AREA = 30     # 녹/먼지 점 제거
LINE_MERGE_KERNEL = (25, 5) # 문자 -> 라인 병합용 팽창 커널 (w, h)
PUNCT_HEIGHT_RATIO = 0.5    # 중앙값 높이 대비 이보다 작은 성분('-', '.')은 베이스라인 계산 제외

STRUCTURED, UNSTRUCTURED, SYMBOL = "STRUCTURED", "UNSTRUCTURED", "SYMBOL"
COLORS = {  # BGR
    STRUCTURED: (0, 200, 0),
    UNSTRUCTURED: (255, 80, 0),
    SYMBOL: (0, 0, 255),
}


@dataclass
class RoI:
    box: tuple                  # (x, y, w, h), 보정된 이미지 좌표
    comp_ids: list              # 이 RoI 에 속한 연결 성분 라벨
    char_boxes: list            # 성분별 (x, y, w, h)
    label: str = ""
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------
def effective_saturation(bgr):
    """HSV S 채널. 아주 어두운 픽셀은 S=(max-min)/max 가 노이즈에 폭주하므로 0(무채색)으로 둔다."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].copy()
    sat[hsv[..., 2] < SAT_DARK_V] = 0
    return sat


def foreground_mask(bgr, valid=None):
    """강판 배경 대비 밝기 차 OR 높은 채도 -> 마킹 전경(255). valid 밖(회전 여백)은 제외."""
    gray = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 3)
    background = cv2.medianBlur(gray, BG_KERNEL)
    diff = cv2.absdiff(gray, background)
    sat = cv2.medianBlur(effective_saturation(bgr), 3)

    mask = ((diff >= FG_DIFF_MIN) | (sat >= FG_SAT_MIN)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if valid is not None:
        mask &= valid

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# 1) 기울기 보정
# ---------------------------------------------------------------------------
def long_side_angle(rect):
    """minAreaRect 결과에서 긴 변의 각도를 [-90, 90) 로 정규화."""
    (_, _), (w, h), angle = rect
    if w < h:
        angle -= 90
    return (angle + 90) % 180 - 90


def estimate_skew(mask):
    """텍스트 라인 덩어리들의 minAreaRect 각도를 면적 가중 중앙값으로 추정 (deg)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, LINE_MERGE_KERNEL)
    blobs = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles, weights = [], []
    for c in contours:
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if min(w, h) < 1 or max(w, h) / min(w, h) < 3:
            continue  # 가로로 긴 라인 덩어리만 사용
        angle = long_side_angle(rect)
        if abs(angle) <= 45:
            angles.append(angle)
            weights.append(w * h)
    if not angles:
        return 0.0

    order = np.argsort(angles)
    angles, weights = np.asarray(angles)[order], np.asarray(weights)[order]
    cum = np.cumsum(weights)
    return float(angles[np.searchsorted(cum, cum[-1] / 2)])


def rotate_bound(img, angle):
    """잘림 없이 캔버스를 키워 회전. (회전 이미지, 2x3 affine 행렬, 원본이 있던 영역 마스크) 반환."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5)
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    # 복제된 여백은 줄무늬가 생겨 전경으로 오검출되므로 마스크로 제외 (경계 몇 px 여유)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (new_w, new_h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
    return rotated, M, valid


def deskew(bgr):
    """|θ| > DESKEW_MIN_ANGLE 이면 수평으로 회전 보정. (보정 이미지, affine, 유효영역, θ) 반환."""
    theta = estimate_skew(foreground_mask(bgr))
    if abs(theta) <= DESKEW_MIN_ANGLE:
        return bgr, np.float64([[1, 0, 0], [0, 1, 0]]), None, theta
    rotated, M, valid = rotate_bound(bgr, theta)
    return rotated, M, valid, theta


# ---------------------------------------------------------------------------
# RoI 추출
# ---------------------------------------------------------------------------
def extract_rois(mask):
    """문자 성분을 가로 팽창으로 라인 단위 RoI 로 묶는다."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, LINE_MERGE_KERNEL)
    nb, blob_labels = cv2.connectedComponents(cv2.dilate(mask, kernel), connectivity=8)

    # 성분은 자신의 팽창 영역에 완전히 포함되므로 모든 픽셀이 같은 덩어리 라벨을 가진다
    comp_to_blob = np.zeros(n, dtype=np.int32)
    comp_to_blob[labels.ravel()] = blob_labels.ravel()

    rois = []
    for b in range(1, nb):
        comp_ids = (np.flatnonzero(comp_to_blob[1:] == b) + 1).tolist()
        if not comp_ids:
            continue
        boxes = [tuple(int(v) for v in stats[i, :4]) for i in comp_ids]
        x0 = min(x for x, _, _, _ in boxes)
        y0 = min(y for _, y, _, _ in boxes)
        x1 = max(x + w for x, _, w, _ in boxes)
        y1 = max(y + h for _, y, _, h in boxes)
        rois.append(RoI(box=(x0, y0, x1 - x0, y1 - y0), comp_ids=comp_ids, char_boxes=boxes))
    rois.sort(key=lambda r: (r.box[1], r.box[0]))
    return rois, labels


# ---------------------------------------------------------------------------
# 2) 정형 문자 게이트
# ---------------------------------------------------------------------------
def baseline_std(char_boxes):
    """[기준 1] 라인 내 문자 바닥 y 좌표의 표준편차 (px).

    보정 임계(2.5°) 이하의 잔여 기울기가 표준편차를 부풀리지 않도록, 문자가 충분하면
    바닥 y 를 x 에 대해 1차 회귀한 잔차의 표준편차를 쓴다. (점이 적으면 회귀가 지터까지
    흡수해 버리므로 그냥 표준편차)
    """
    if len(char_boxes) < MIN_CHARS_PER_LINE:
        return float("inf")
    median_h = np.median([h for _, _, _, h in char_boxes])
    chars = [b for b in char_boxes if b[3] >= PUNCT_HEIGHT_RATIO * median_h]
    if len(chars) < MIN_CHARS_PER_LINE:
        return float("inf")

    xs = np.array([x + w / 2 for x, _, w, _ in chars], dtype=np.float64)
    ys = np.array([y + h for _, y, _, h in chars], dtype=np.float64)
    if len(chars) >= BASELINE_DETREND_MIN:
        ys = ys - np.polyval(np.polyfit(xs, ys, 1), xs)
    return float(ys.std())


def stroke_width_cv(ink):
    """[기준 2] SWT 근사: distanceTransform 능선(획 중심선)에서 획 두께 2·d 의 변동계수."""
    ink = cv2.copyMakeBorder(ink, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    ridge = (dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    widths = 2.0 * dist[ridge]
    if widths.size < 10:
        return float("inf")
    lo, hi = np.percentile(widths, [5, 95])  # 끝점/교차점 극단값 제거
    widths = widths[(widths >= lo) & (widths <= hi)]
    if widths.size == 0 or widths.mean() == 0:
        return float("inf")
    return float(widths.std() / widths.mean())


def mean_saturation(sat_crop, ink):
    """[기준 3] 전경 픽셀의 평균 HSV 채도."""
    return float(sat_crop[ink > 0].mean())


# ---------------------------------------------------------------------------
# 3) 기호 분기용 기하 통계
# ---------------------------------------------------------------------------
def shape_stats(label_crop, roi):
    """가장 큰 성분의 Solidity, 최소외접사각형 종횡비, RoI 잉크 중 차지 비율."""
    pixel_counts = [int(np.count_nonzero(label_crop == i)) for i in roi.comp_ids]
    k = int(np.argmax(pixel_counts))
    dominant_ratio = pixel_counts[k] / max(sum(pixel_counts), 1)

    comp = (label_crop == roi.comp_ids[k]).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0.0

    w, h = cv2.minAreaRect(contour)[1]
    aspect = max(w, h) / max(min(w, h), 1.0)
    return solidity, aspect, dominant_ratio


def classify_roi(roi, labels, sat):
    x, y, w, h = roi.box
    label_crop = labels[y:y + h, x:x + w]
    ink = np.isin(label_crop, roi.comp_ids).astype(np.uint8) * 255

    m = roi.metrics
    m["baseline_std"] = baseline_std(roi.char_boxes)
    m["stroke_cv"] = stroke_width_cv(ink)
    m["mean_sat"] = mean_saturation(sat[y:y + h, x:x + w], ink)
    m["solidity"], m["aspect"], m["dominant"] = shape_stats(label_crop, roi)

    m["gates"] = (
        m["baseline_std"] <= BASELINE_STD_MAX,
        m["stroke_cv"] <= STROKE_CV_MAX,
        m["mean_sat"] < SAT_MAX,
    )
    if all(m["gates"]):
        return STRUCTURED
    if m["dominant"] >= SYMBOL_DOMINANT_RATIO and (
        m["solidity"] < SOLIDITY_MIN or m["aspect"] >= SYMBOL_AR_MAX
    ):
        return SYMBOL
    return UNSTRUCTURED


# ---------------------------------------------------------------------------
# 시각화 / 출력
# ---------------------------------------------------------------------------
def draw_results(original, rois, M):
    """보정 좌표의 RoI 를 역변환해 원본 이미지 위에 클래스별 색으로 그린다."""
    canvas = original.copy()
    M_inv = cv2.invertAffineTransform(M)
    for idx, roi in enumerate(rois, 1):
        x, y, w, h = roi.box
        pad = 4
        corners = np.float32([[x - pad, y - pad], [x + w + pad, y - pad],
                              [x + w + pad, y + h + pad], [x - pad, y + h + pad]])
        quad = cv2.transform(corners[None], M_inv)[0].astype(np.int32)
        color = COLORS[roi.label]
        cv2.polylines(canvas, [quad], True, color, 2, cv2.LINE_AA)

        text = f"#{idx} {roi.label}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 라벨은 박스의 (보정 좌표 기준) 왼쪽 위 꼭짓점에 붙이고, 화면 밖으로 나가지 않게 자른다
        tx = int(np.clip(quad[0, 0], 0, canvas.shape[1] - tw - 4))
        ty = int(np.clip(quad[0, 1] - 4, th + 4, canvas.shape[0] - base))
        cv2.rectangle(canvas, (tx, ty - th - 4), (tx + tw + 4, ty + base), color, -1)
        cv2.putText(canvas, text, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # 범례: 오른쪽 위, 반투명 흰 바탕 위에 표시
    x0 = canvas.shape[1] - 170
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0 - 8, 6), (canvas.shape[1] - 6, 14 + 22 * len(COLORS)), (255, 255, 255), -1)
    canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)
    for i, (name, color) in enumerate(COLORS.items()):
        y = 26 + 22 * i
        cv2.rectangle(canvas, (x0, y - 12), (x0 + 16, y + 2), color, -1)
        cv2.putText(canvas, name, (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def print_report(rois, theta, rotated):
    status = f"보정 적용 ({theta:+.2f}° 회전)" if rotated else f"보정 생략 (|{theta:.2f}°| <= {DESKEW_MIN_ANGLE}°)"
    print(f"기울기 추정 θ = {theta:+.2f}° -> {status}")
    print(f"{'#':>2}  {'label':<12} {'box(x,y,w,h)':<22} {'baseσ':>6} {'SW_CV':>6} {'S':>6}"
          f"  gate  {'solid':>5} {'AR':>5}")
    for idx, r in enumerate(rois, 1):
        m = r.metrics
        gates = "".join("O" if g else "X" for g in m["gates"])
        print(f"{idx:>2}  {r.label:<12} {str(r.box):<22} {m['baseline_std']:>6.2f} "
              f"{m['stroke_cv']:>6.2f} {m['mean_sat']:>6.1f}  {gates:<4}  "
              f"{m['solidity']:>5.2f} {m['aspect']:>5.1f}")
    counts = {k: sum(r.label == k for r in rois) for k in COLORS}
    print("요약:", ", ".join(f"{k}={v}" for k, v in counts.items()))


def show_or_save(image, out_path, show=True):
    """결과는 항상 파일로 저장하고, GUI 가 가능할 때만 창을 띄운다."""
    cv2.imwrite(out_path, image)
    print(f"결과 저장: {out_path}")
    if not show:
        return
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("디스플레이 없음 -> 창 표시 생략")
        return
    try:
        cv2.imshow("marking classification", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("GUI 미지원 OpenCV(opencv-python-headless 등) -> 창 표시 생략, 파일로 확인하세요")


# ---------------------------------------------------------------------------
# 합성 샘플 (test_plate.jpg 가 없을 때)
# ---------------------------------------------------------------------------
def _handwrite(img, text, origin, color, rng):
    """글자마다 크기/두께/높이를 흔들어 수기 마커처럼 쓴다."""
    x, y = origin
    for ch in text:
        scale = 1.4 + rng.uniform(-0.25, 0.25)
        thick = int(rng.integers(2, 7))
        dy = int(rng.integers(-8, 9))
        cv2.putText(img, ch, (x, y + dy), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        (w, _), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, thick)
        x += w + int(rng.integers(0, 6))


def make_sample_plate(seed=7, tilt_deg=6.0):
    rng = np.random.default_rng(seed)
    h, w = 620, 960
    plate = np.full((h, w, 3), (128, 132, 136), np.float32)
    plate += rng.normal(0, 6, (h, w, 1))                       # 강판 표면 노이즈
    plate = np.clip(plate, 0, 255).astype(np.uint8)
    for _ in range(300):                                       # 녹/먼지 점
        cv2.circle(plate, (int(rng.integers(0, w)), int(rng.integers(0, h))), 1, (70, 85, 105), -1)

    ink = (30, 30, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(plate, "B12-SP3 4500", (60, 110), font, 1.6, ink, 4, cv2.LINE_AA)    # 정형
    cv2.putText(plate, "PORT FR-07 T16", (60, 200), font, 1.6, ink, 4, cv2.LINE_AA)  # 정형
    _handwrite(plate, "CUT 2", (80, 330), (0, 200, 255), rng)                         # 노란 마커
    _handwrite(plate, "OK chk", (560, 330), (225, 225, 225), rng)                     # 흰 분필
    cv2.arrowedLine(plate, (560, 450), (860, 450), (40, 40, 210), 4, cv2.LINE_AA, tipLength=0.12)
    cv2.line(plate, (300, 420), (360, 480), (0, 200, 255), 5, cv2.LINE_AA)            # X 표시
    cv2.line(plate, (360, 420), (300, 480), (0, 200, 255), 5, cv2.LINE_AA)
    cv2.line(plate, (60, 545), (900, 545), ink, 3, cv2.LINE_AA)                       # 마킹선

    M = cv2.getRotationMatrix2D((w / 2, h / 2), tilt_deg, 1.0)
    return cv2.warpAffine(plate, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------
def run_pipeline(bgr):
    """(RoI 목록, 원본 위 시각화 이미지, θ, 보정 여부) 반환."""
    deskewed, M, valid, theta = deskew(bgr)
    mask = foreground_mask(deskewed, valid)
    rois, labels = extract_rois(mask)
    sat = effective_saturation(deskewed)
    for roi in rois:
        roi.label = classify_roi(roi, labels, sat)
    rotated = abs(theta) > DESKEW_MIN_ANGLE
    return rois, draw_results(bgr, rois, M), theta, rotated


def main():
    parser = argparse.ArgumentParser(description="철판 마킹 정형/비정형/기호 분류 데모")
    parser.add_argument("image", nargs="?", default="test_plate.jpg")
    parser.add_argument("-o", "--output", default="result.jpg")
    parser.add_argument("--no-show", action="store_true", help="결과 창을 띄우지 않음")
    args = parser.parse_args()

    if os.path.exists(args.image):
        bgr = cv2.imread(args.image)
        if bgr is None:
            sys.exit(f"이미지를 읽을 수 없습니다: {args.image}")
    else:
        print(f"{args.image} 없음 -> 합성 샘플 철판을 생성해 저장합니다")
        bgr = make_sample_plate()
        cv2.imwrite(args.image, bgr)

    rois, vis, theta, rotated = run_pipeline(bgr)
    print_report(rois, theta, rotated)
    show_or_save(vis, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
