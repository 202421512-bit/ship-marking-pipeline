"""
수기(Handwritten) vs 기계식 마킹(Printed) 판별 특징 - φ₈ 획 연결성 및 분기점 밀도

배경
  현장 수기는 펜을 떼지 않고 글자·획을 이어 쓰는 일이 잦아(연결, ligature) 골격선에 분기점(3·4갈래)과
  닫힌 고리가 늘어난다. 기계 마킹은 글자가 분리되어 있어 분기점이 글꼴 구조 이상으로 늘지 않는다.

정의
  골격선 S (1px), 각 골격 픽셀의 횡단수 CN(p) = ½ Σ |I(p_i) − I(p_{i+1})|  (시계방향 8-이웃, p₉ = p₁)
      CN = 1 끝점,  CN = 2 단순 연결점,  CN ≥ 3 분기점
  ρ_J = N_J / L_skel,   φ₈ = tanh(α₈ · ρ_J)

구현상 결정 (이유)
  1. 가짜 분기점: (a) 거친 표면 때문에 획 안에 생긴 작은 구멍을 메우고 (구멍마다 골격 고리 = 분기점 2개),
     (b) 테두리를 획 폭의 1/4 로 다듬고, (b') 남은 털 모양 잔가지를 가지치기(끝점에서 시작해 교차점에서
     끝나는 짧은 가지 제거)한 뒤,
     (c) 인접한 분기 픽셀은 반경 r(획 폭의 절반, 최소 3px) 안에서 하나로 묶는다.
  2. 스케일: ρ_J 를 픽셀 단위로 두면 사진을 2배 키울 때 절반이 된다. 점수에는 골격 길이를 글자 높이 H 로
     잰 밀도  ρ_J,H = N_J / (L_skel / H)  (글자 높이만큼의 획 길이당 분기점 수) 를 쓴다.
     픽셀 단위 ρ_J (요청 정의) 는 junction_density 로 함께 돌려준다.
  3. 인쇄체 고유 분기점('A', 'B', '4')을 감안해, 밀도와 별개로 글자 간 연결을 센다:
     한 연결 요소가 줄 방향으로 글자 여러 개 폭에 걸치면 (폭 / 보통 글자 폭 − 1) 개의 연결로 본다.
     합성 실험(인쇄체 vs 이어 쓰기 36쌍)에서 분기점 밀도는 글꼴 구조에 끌려가 구분력이 약했고
     (AUC 0.58~0.81, 'A4B8' 같은 인쇄체가 이어 쓴 글씨보다 높게 나옴), 연결 비율은 0.93~1.00 이었다.
     그래서 기본 점수는 연결 비율(score="ligature"), 요청 수식은 score="density" 로 둔다.
  4. 여러 글자를 가로지르는 가는 긁힘은 가짜 연결·분기점을 만들므로 φ₃ 이후 공통 전처리로 끊는다.
     (획보다 훨씬 가는 실제 연결선도 함께 끊길 수 있다는 한계가 있음)
"""
import math

import cv2
import numpy as np
from skimage.morphology import remove_small_holes, skeletonize

from baseline_feature import fit_theil_sen
from orientation_feature import _cut_thin_lines, _merge_fragments, _remove_specks, binarize, load_image
from stroke_features import _prune_spurs

ALPHA8 = 1.2              # score="density": tanh(α₈ · ρ_J,H). 합성 인쇄체 ρ_J,H 중앙 0.73 -> 0.70 (밀도만으론 폭이 좁음)
ALPHA_LIG = 3.0           # score="ligature": tanh(α_L · 연결 비율). 이어 쓰기 연결 비율 0.2 -> 0.54, 0.45 -> 0.87
SPUR_FACTOR = 2.5         # 끝 가지 길이 < 이 값 × 중앙 반폭 이면 잔가지
JUNCTION_MERGE_MIN = 3.0  # 분기 픽셀 병합 반경 하한 (px)
LIGATURE_WIDTH = 1.6      # 줄 방향 폭이 보통 글자 폭의 이 배수 이상이면 글자 둘 이상이 이어진 것
HOLE_FILL_RATIO = 0.5     # 면적 < 이 값 × 획 폭² 인 획 안 구멍은 메움 (표면 요철 구멍 -> 가짜 고리·분기점)
BOUNDARY_SMOOTH = 0.25    # 골격화 전 테두리 스무딩 σ = 이 값 × 획 폭

# 시계방향 8-이웃 (dy, dx): 위부터 시계방향
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


# ---------------------------------------------------------------------------
# 골격 그래프
# ---------------------------------------------------------------------------
def crossing_number(skel):
    """CN(p) = ½ Σ |I(p_i) − I(p_{i+1})|, 시계방향 8-이웃 (벡터화). 골격이 아닌 픽셀은 0."""
    s = np.pad(skel.astype(np.int8), 1)
    h, w = skel.shape
    ring = [s[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _RING]
    cn = sum(np.abs(ring[i] - ring[(i + 1) % 8]) for i in range(8)) // 2
    return np.where(skel, cn, 0)


def neighbor_count(skel):
    k = np.ones((3, 3), np.float32)
    return (cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) - skel).astype(int)


def cluster_points(mask, radius):
    """반경 radius 안의 픽셀들을 하나로 묶어 대표점(무게중심) 목록."""
    if not mask.any():
        return []
    r = max(1, int(round(radius)))
    grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab = cv2.connectedComponents(grown, connectivity=8)
    ys, xs = np.nonzero(mask)
    ids = lab[ys, xs]
    return [(float(xs[ids == i].mean()), float(ys[ids == i].mean())) for i in range(1, n) if (ids == i).any()]


def count_loops(skel):
    """골격이 둘러싼 닫힌 구멍 수 (배경 성분 중 영상 가장자리에 닿지 않는 것)."""
    bg = (~skel).astype(np.uint8)
    n, lab = cv2.connectedComponents(bg, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return sum(1 for i in range(1, n) if i not in border)


# ---------------------------------------------------------------------------
# 글자 간 연결 (ligature)
# ---------------------------------------------------------------------------
def count_ligatures(mask, char_height):
    """줄 방향 폭이 보통 글자 폭의 LIGATURE_WIDTH 배 이상인 성분 = 글자 여러 개가 이어진 것.
    (연결 수 합, 전체 글자 수 추정, 성분별 (상자, 이어진 글자 수)) 반환."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_HEIGHT] >= 0.5 * char_height]
    if len(big) == 0:
        return 0, 0, []
    if len(big) >= 2:
        a, _ = fit_theil_sen(cents[big, 0], cents[big, 1])
    else:
        a = 0.0
    u = np.array([1.0, a]) / math.hypot(1.0, a)
    widths = []
    for i in big:
        ys, xs = np.nonzero(lab == i)
        along = xs * u[0] + ys * u[1]
        widths.append(float(along.max() - along.min() + 1))
    widths = np.array(widths)
    # 보통 글자 폭: 글자 높이보다 좁은 성분들의 중앙값 (모두 넓으면 높이의 0.6 배로 가정)
    narrow = widths[widths < char_height]
    char_w = float(np.median(narrow)) if len(narrow) else 0.6 * char_height
    pitch = char_w * 1.15                                    # 글자 폭 + 보통 간격
    links, chars, info = 0, 0, []
    for i, wd in zip(big, widths):
        k = max(1, int(round((wd + 0.15 * char_w) / pitch))) if wd >= LIGATURE_WIDTH * char_w else 1
        links += k - 1
        chars += k
        info.append((tuple(int(v) for v in stats[i, :4]), k))
    return links, chars, info


# ---------------------------------------------------------------------------
# φ₈
# ---------------------------------------------------------------------------
def calculate_phi_8_connectivity(image_path_or_array, alpha=ALPHA8, score="ligature",
                                 alpha_ligature=ALPHA_LIG, binarization="sauvola") -> dict:
    """단일 크롭 이미지의 φ₈ (획 연결성·분기점 밀도).

    score  "ligature" (기본): tanh(α_L · 연결 비율),  연결 비율 = 글자 간 연결 수 / (추정 글자 수 − 1)
           "density"  (요청 수식): tanh(α₈ · ρ_J,H),  ρ_J,H = N_J / (L_skel / H)
           "combined"          : 1 − (1 − tanh(α₈·ρ_J,H)) · (1 − tanh(α_L · 연결 비율))

    반환 dict
      phi_8, num_junctions, num_endpoints, skeleton_length(px), junction_density(N_J / L_skel, 1/px),
      junction_density_per_height(N_J / (L_skel/H)), num_loops, num_ligatures, ligature_ratio,
      char_height, junctions/endpoints(좌표), skeleton(마스크), ligature_components, reason
    """
    img = load_image(image_path_or_array)
    mask = _remove_specks(binarize(img, binarization))
    mask, _ = _merge_fragments(mask)
    mask = _cut_thin_lines(mask)
    result = {"phi_8": None, "num_junctions": 0, "num_endpoints": 0, "skeleton_length": 0,
              "junction_density": None, "junction_density_per_height": None, "num_loops": 0,
              "num_ligatures": 0, "ligature_ratio": None, "char_height": None,
              "junctions": [], "endpoints": [], "skeleton": None, "ligature_components": [],
              "score": score, "reason": ""}
    if not mask.any():
        result["reason"] = "획 픽셀 없음"
        return result

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    heights, areas = stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_AREA]
    char_h = float(np.median(heights[areas >= 0.3 * np.median(areas)])) if len(areas) > 2 else float(heights.max())

    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    # 거친 표면에서 생긴 획 안의 작은 구멍: 골격이 구멍마다 고리를 둘러 가짜 분기점을 두 개씩 만든다.
    # 글자 속공간('B', '8')보다 훨씬 작은 (0.5 × 획 폭²) 구멍만 메운다.
    stroke = 2 * float(np.percentile(dist[dist > 0], 90))
    mask = remove_small_holes(mask, max_size=max(1, int(HOLE_FILL_RATIO * stroke * stroke)))
    # 테두리 요철 다듬기: 획 폭의 1/4 로 흐린 뒤 다시 문턱. 요철에서 뻗는 긴 잔가지는 가지치기만으로
    # 다 안 지워져서(거친 막대 하나에 가짜 분기점 11~15개) 골격화 전에 테두리를 먼저 매끈하게 한다.
    sigma = max(1.0, BOUNDARY_SMOOTH * stroke)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) > 0.5
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    skel = skeletonize(mask)
    half_w = float(np.median(dist[skel])) if skel.any() else 1.0
    skel = _prune_spurs(skel, dist, factor=SPUR_FACTOR, rounds=3)
    length = int(skel.sum())
    if length < max(10, 0.5 * char_h):
        result["reason"] = f"골격선이 너무 짧음 ({length}px)"
        return result

    cn = crossing_number(skel)
    nb = neighbor_count(skel)
    # 분기 후보: CN ≥ 3. 8-연결 골격에선 T 자의 이웃이 붙어 있으면 CN=2 로 나오므로 이웃 수 ≥ 3 도 후보
    junction_px = skel & ((cn >= 3) | (nb >= 4) | ((nb == 3) & (cn >= 2) & _not_staircase(skel)))
    endpoint_px = skel & (cn == 1) & (nb == 1)
    radius = max(JUNCTION_MERGE_MIN, half_w)
    junctions = cluster_points(junction_px, radius)
    endpoints = cluster_points(endpoint_px, 1)

    links, n_chars, lig_info = count_ligatures(mask, char_h)
    rho = len(junctions) / length
    rho_h = len(junctions) / (length / char_h)
    lig_ratio = links / max(1, n_chars - 1) if n_chars > 1 else 0.0
    phi_density = math.tanh(alpha * rho_h)
    phi_lig = math.tanh(alpha_ligature * lig_ratio)
    phi = {"density": phi_density, "ligature": phi_lig,
           "combined": 1 - (1 - phi_density) * (1 - phi_lig)}[score]
    result.update(phi_8=float(phi), num_junctions=len(junctions), num_endpoints=len(endpoints),
                  skeleton_length=length, junction_density=rho, junction_density_per_height=rho_h,
                  num_loops=count_loops(skel), num_ligatures=links, ligature_ratio=lig_ratio,
                  char_height=char_h, junctions=junctions, endpoints=endpoints, skeleton=skel,
                  ligature_components=lig_info)
    return result


def _not_staircase(skel):
    """이웃 3개 중 둘이 서로 붙어 있는 '계단 모서리' 패턴(실제 분기 아님)을 거르기 위한 마스크.
    이웃 3개가 모두 서로 떨어져 있지 않은데 CN=2 인 경우 중, 4-이웃 둘 + 대각 하나가 L 자로 붙은
    계단 모양만 False."""
    s = np.pad(skel.astype(np.uint8), 1)
    h, w = skel.shape
    get = lambda dy, dx: s[1 + dy:1 + dy + h, 1 + dx:1 + dx + w].astype(bool)
    up, down, left, right = get(-1, 0), get(1, 0), get(0, -1), get(0, 1)
    ul, ur, dl, dr = get(-1, -1), get(-1, 1), get(1, -1), get(1, 1)
    # 대각 이웃이 그 옆의 4-이웃과 붙어 있으면 그 둘은 같은 가지 (계단)
    stair = (ul & (up | left)) | (ur & (up | right)) | (dl & (down | left)) | (dr & (down | right))
    return ~stair


# ---------------------------------------------------------------------------
# 디버깅 시각화
# ---------------------------------------------------------------------------
def draw_connectivity_debug(image_path_or_array, result, scale=None):
    """골격선(파랑), 끝점(초록 원), 분기점(빨강 원), 글자 여러 개가 이어진 성분(주황 상자 + 이어진 글자 수)."""
    img = load_image(image_path_or_array)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    canvas = cv2.cvtColor((gray * 0.55 + 100).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if scale is None:
        scale = max(1.0, 600 / max(canvas.shape[:2]))
    if result["skeleton"] is not None:
        skel = result["skeleton"].astype(np.uint8)
        if scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            skel = cv2.resize(skel, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
            skel = cv2.dilate(skel, np.ones((max(1, int(scale)), max(1, int(scale))), np.uint8))
        canvas[skel > 0] = (255, 80, 0)
    elif scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    r = max(3, int(round(3 * scale)))
    for (box, k) in result["ligature_components"]:
        if k > 1:
            x, y, w, h = (int(v * scale) for v in box)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 140, 255), max(1, int(scale)))
            cv2.putText(canvas, f"x{k}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)
    for x, y in result["endpoints"]:
        cv2.circle(canvas, (int(x * scale), int(y * scale)), r, (0, 200, 0), 2, cv2.LINE_AA)
    for x, y in result["junctions"]:
        cv2.circle(canvas, (int(x * scale), int(y * scale)), r + 1, (0, 0, 255), 2, cv2.LINE_AA)
    phi = result["phi_8"]
    text = (f"phi8={phi:.3f}  J={result['num_junctions']}  E={result['num_endpoints']}  L={result['skeleton_length']}px  "
            f"rhoJ,H={result['junction_density_per_height']:.3f}  lig={result['num_ligatures']}  [{result['score']}]"
            if phi is not None else f"phi8=None: {result['reason']}")
    cv2.rectangle(canvas, (0, 0), (12 + 8 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 합성 시험 데이터
# ---------------------------------------------------------------------------
def render_connected(text, link_prob=0.7, overlap=4, seed=0, size=(140, 520), thickness=6):
    """이어 쓰기 흉내: 글자를 조금씩 겹쳐 붙이고, 확률 link_prob 로 앞 글자 오른쪽 아래에서 다음 글자
    왼쪽 중간으로 잇는 획을 더한다 (펜을 떼지 않은 연결선)."""
    rng = np.random.default_rng(seed)
    img = np.full(size, 200, np.uint8)
    x, prev = 20, None
    base = size[0] - 40
    for ch in text:
        if ch == " ":
            x += 26
            prev = None
            continue
        s = 2.2 * rng.uniform(0.9, 1.1)
        (w, h), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, s, thickness)
        if x + w > size[1] - 10:
            break
        cv2.putText(img, ch, (x, base), cv2.FONT_HERSHEY_SIMPLEX, s, 40, thickness, cv2.LINE_AA)
        if prev is not None and rng.random() < link_prob:
            cv2.line(img, prev, (x + int(0.1 * w), base - int(rng.uniform(0.3, 0.6) * h)), 40, thickness - 1,
                     cv2.LINE_AA)
        prev = (x + w - 3, base - 2)
        x += w - overlap
    return img


def _save(path, image):
    ok, buf = cv2.imencode(".png", image)
    buf.tofile(path)


if __name__ == "__main__":
    import argparse

    from orientation_feature import add_plate_noise, render_printed

    parser = argparse.ArgumentParser(description="φ₈ 획 연결성·분기점 밀도 계산 + 디버그 시각화")
    parser.add_argument("image", nargs="?", help="크롭 이미지 (없으면 합성 예시)")
    parser.add_argument("-o", "--output", default="phi8_debug.png")
    parser.add_argument("--score", default="ligature", choices=["ligature", "density", "combined"])
    args = parser.parse_args()

    if args.image:
        samples = [(args.image, args.image)]
    else:
        samples = [("printed", add_plate_noise(render_printed("B12-SP3 4500"))),
                   ("printed AB48", add_plate_noise(render_printed("A4B8 R4A8"))),
                   ("connected", add_plate_noise(render_connected("B12SP3 4500", seed=1))),
                   ("connected AB48", add_plate_noise(render_connected("A4B8 R4A8", seed=2)))]
    panels = []
    for name, src in samples:
        res = calculate_phi_8_connectivity(src, score=args.score)
        if res["phi_8"] is None:
            print(f"{name:15s} phi_8=None ({res['reason']})")
        else:
            print(f"{name:15s} phi_8={res['phi_8']:.3f}  J={res['num_junctions']}  E={res['num_endpoints']}  "
                  f"L={res['skeleton_length']}px  rhoJ={res['junction_density']:.4f}/px  "
                  f"rhoJ,H={res['junction_density_per_height']:.3f}  loops={res['num_loops']}  "
                  f"ligatures={res['num_ligatures']} ({res['ligature_ratio']:.2f})")
        panels.append(draw_connectivity_debug(src, res, scale=1.0))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 4, 0, width - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    _save(args.output, np.vstack(panels))
    print(f"시각화 저장: {args.output}")
