"""
글자 유착(Touching / Overlapping Characters) 검출

기계식 마킹은 폰트 메트릭스(자간)가 제어되어 글자끼리 닿거나 겹칠 수 없다. 반대로 수기는 좁은 영역에
글자를 써 넣다 보면 인접 글자가 서로 닿는 일이 흔하다. 따라서 '유착' 은 에러·결측이 아니라 수기의 강한 증거다.

검출 규칙 (연결요소 CC 하나마다)
  - 가로/세로 비율 W/H > RATIO (1.8): 한 글자로는 너무 납작한 덩어리 (정상 글자 'M'·'W' 는 1.0~1.3 정도)
  - 열 투영 V(x) = Σ_y I(x, y) 를 본다. V(x) ≤ valley_frac·H 인 열(글자 분리 계곡)이 없으면 "fused"
    (글자가 굵게 겹쳐 붙은 덩어리), 있으면 "bridged" (가는 연결선으로 이어진 글자들). 둘 다 유착이다.
  - 오검출 방지: 영상 경계에 닿는 성분(판 가장자리), 기준 글자 높이보다 훨씬 낮은 성분(하이픈·밑줄),
    기준 높이의 SCRATCH_H_RATIO 배보다 높거나 MAX_SPAN_RATIO 배보다 긴 성분(긁힘·줄 가로지르는 선)은 제외.
  - 기준 글자 높이 = W/H ≤ RATIO 인 정상 성분 높이의 중앙값 (정상 성분이 없으면 전체 중앙값).

split_touching_mask 는 φ₆ 가 쓰는 보조 함수다. 유착 덩어리를 추정 글자 수 k 로 나눠 V(x) 가 가장 낮은 열에서
잘라, 큰 덩어리 하나가 CV_A(면적 변동계수)를 부풀리지 않게 한다.
"""
import cv2
import numpy as np

from orientation_feature import _remove_specks, binarize

RATIO = 1.8               # W/H 가 이 값 초과면 한 글자가 아닌 덩어리
VALLEY_FRAC = 0.12        # V(x) ≤ 이 값 × H 인 열 = 글자 분리 계곡
EDGE_MARGIN = 0.1         # 계곡 탐색에서 덩어리 양 끝 10% 는 제외 (획 끝은 항상 V 가 작음)
DASH_H_RATIO = 0.6        # 높이가 기준의 이 배수 미만이면 하이픈·밑줄
SCRATCH_H_RATIO = 1.8     # 높이가 기준의 이 배수 초과면 긁힘/줄 가로지르는 선
MAX_SPAN_RATIO = 6.0      # 폭이 기준 높이의 이 배수 초과면 긁힘/밑줄
CUT_SEARCH = 0.15         # 분할 위치는 균등 분할점 ± 이 비율×(덩어리 폭/글자 수) 안에서 V(x) 최소 열
MAX_PARTS = 4             # 한 덩어리를 최대 몇 글자로 나눌지


def _mask_of(img_or_mask):
    a = np.asarray(img_or_mask)
    if a.dtype == bool:
        return a
    return _remove_specks(binarize(a))


def _reference_sizes(stats):
    """기준 글자 높이·폭: 정상(W/H ≤ RATIO) 성분 중 잉크 면적의 상위 70% 를 이루는 성분의 중앙값.
    (면적 기준이라 점 잡음·녹 반점이 아무리 많아도 글자가 기준이 된다.)
    정상 성분이 없으면 전체 중앙 높이와 0.7×높이."""
    w, h = stats[1:, cv2.CC_STAT_WIDTH].astype(float), stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    area = stats[1:, cv2.CC_STAT_AREA].astype(float)
    normal = w / np.maximum(h, 1) <= RATIO
    if normal.any():
        idx = np.nonzero(normal)[0]
        order = idx[np.argsort(area[idx])[::-1]]
        main = order[: np.searchsorted(np.cumsum(area[order]), 0.7 * area[order].sum()) + 1]
        return float(np.median(h[main])), float(np.median(w[main]))
    ref_h = float(np.median(h))
    return ref_h, 0.7 * ref_h


def detect_touching_chars(img_or_mask, ratio=RATIO, valley_frac=VALLEY_FRAC) -> dict:
    """유착 글자 검출.

    반환 dict: has_touching_chars(bool), components([{id, box, ratio, kind('fused'|'bridged'), valley_cols,
              est_chars}]), num_touching, est_joints(추정 접점 수 = Σ(est_chars-1)), ref_height, ref_width,
              reason
    """
    mask = _mask_of(img_or_mask)
    out = {"has_touching_chars": False, "components": [], "num_touching": 0, "est_joints": 0,
           "ref_height": None, "ref_width": None, "reason": ""}
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        out["reason"] = "잉크 없음"
        return out
    H, W = mask.shape
    ref_h, ref_w = _reference_sizes(st)
    out.update(ref_height=ref_h, ref_width=ref_w)
    for i in range(1, n):
        x, y, w, h, _ = (int(v) for v in st[i])
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue                                            # 판 가장자리·잘린 성분
        if w / max(h, 1) <= ratio:
            continue
        if h < DASH_H_RATIO * ref_h or h > SCRATCH_H_RATIO * ref_h or w > MAX_SPAN_RATIO * ref_h:
            continue                                            # 하이픈·밑줄 / 긁힘
        v = (lab[y:y + h, x:x + w] == i).sum(axis=0)           # 열 투영 V(x)
        m = int(EDGE_MARGIN * w)
        inner = v[m:w - m] if w - 2 * m > 0 else v
        valleys = int((inner <= valley_frac * h).sum())
        k = int(np.clip(round(w / max(ref_w, 1.0)), 2, MAX_PARTS))
        out["components"].append({"id": i, "box": (x, y, w, h), "ratio": w / h,
                                  "kind": "bridged" if valleys else "fused",
                                  "valley_cols": valleys, "est_chars": k})
    out["num_touching"] = len(out["components"])
    out["est_joints"] = sum(c["est_chars"] - 1 for c in out["components"])
    out["has_touching_chars"] = out["num_touching"] > 0
    return out


def split_touching_mask(mask, info=None):
    """유착 덩어리를 V(x) 최소 열에서 잘라 글자 단위로 나눈 마스크 (원본은 건드리지 않음).

    info: detect_touching_chars(mask) 결과 (없으면 새로 계산). 반환: (잘린 마스크, 자른 덩어리 수)"""
    mask = np.asarray(mask, bool)
    info = info or detect_touching_chars(mask)
    if not info["has_touching_chars"]:
        return mask, 0
    n, lab, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = mask.copy()
    for c in info["components"]:
        x, y, w, h = c["box"]
        comp = lab[y:y + h, x:x + w] == c["id"]
        v = cv2.GaussianBlur(comp.sum(axis=0).astype(np.float32).reshape(1, -1), (0, 0), max(1.0, w / 60)).ravel()
        k = c["est_chars"]
        half = max(1, int(CUT_SEARCH * w / k))
        for j in range(1, k):
            ideal = int(round(w * j / k))
            lo, hi = max(1, ideal - half), min(w, ideal + half + 1)
            if lo >= hi:
                continue
            cut = lo + int(np.argmin(v[lo:hi]))
            out[y:y + h, x + cut] &= ~comp[:, cut]
    return out, len(info["components"])
