# -*- coding: utf-8 -*-
"""
stage1_denoise.py
1차 노이즈 제거 프로토타입
(Hao et al., Electronics 2026, 15(14), 3068 — Section 2.3 구조 기반)

------------------------------------------------------------------
[논문에 명시된 것]  → 코드에서 "논문 명시" 로 표시
  - LAB 색공간에서 처리, 휘도(L) 채널만 조정하고 색(a, b)은 보존
  - 1단계: L 채널의 적응형(adaptive) 밝기 보정
  - 2단계: 엣지 보존 필터링 → 그 다음 국소 대비 강화 (순서)

[논문에 없는 것 = 우리가 선택한 것]  → 코드에서 "우리 선택" 으로 표시
  - 각 단계에 쓸 구체 알고리즘 (CLAHE, Bilateral, Unsharp mask 등)
  - 모든 파라미터 값
  - 결과 비교 그림, 획 대비(CNR) 측정
------------------------------------------------------------------

실행 방법 (VS Code 터미널, (venv) 상태에서)
  python src/stage1_denoise.py

입력: data/samples/ 와 data/real/ 안의 모든 이미지
출력: results/stage1/ 아래에
  - <이름>_final.png   : 최종 결과 이미지
  - <이름>_compare.png : 단계별 비교 그림 (원본/1단계/2단계-①/최종)
  - metrics.csv        : 획 대비 수치 (정답 마스크가 있는 이미지만)
"""

import os
import glob
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")            # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt


# ==================================================================
# ★ 설정값 — 실험할 때는 여기만 바꾸면 됩니다 ★
#   (모든 값은 '우리 선택'. 논문에는 수치가 없음)
# ==================================================================
CONFIG = {
    # --- 1단계: L 채널 적응형 밝기 보정 ---
    #   "clahe" : 영역별로 밝기 분포를 펴주는 방법 (기본값)
    #   "none"  : 이 단계 끄기 (비교 실험용)
    "step1_method": "clahe",
    "clahe1_clip": 2.0,          # 클수록 대비를 세게 올림 (노이즈도 같이 커질 수 있음)
    "clahe1_tile": 8,            # 이미지를 8x8 칸으로 나눠 칸마다 보정

    # --- 2단계-①: 엣지 보존 필터링 ---
    #   "bilateral" : 비슷한 밝기끼리만 섞어서 경계는 살리는 필터 (기본값)
    #   "nlm"       : Non-Local Means, 비슷한 무늬끼리 평균 (느리지만 강함)
    #   "none"      : 이 단계 끄기
    "step2a_method": "bilateral",
    "bil_d": 9,                  # 필터 크기(픽셀). 클수록 넓게 섞음
    "bil_sigma_color": 40,       # 이 값보다 밝기 차이가 크면 '경계'로 보고 안 섞음
    "bil_sigma_space": 9,        # 거리 가중치
    "nlm_h": 10,                 # nlm 강도

    # --- 2단계-②: 국소 대비 강화 ---
    #   "unsharp" : 흐린 버전과의 차이를 더해 경계를 또렷하게 (기본값)
    #   "clahe"   : CLAHE를 한 번 더 적용
    #   "none"    : 이 단계 끄기
    "step2b_method": "unsharp",
    "unsharp_sigma": 3.0,        # 어느 정도 굵기의 구조를 강조할지
    "unsharp_amount": 1.0,       # 강조 세기
    "clahe2_clip": 2.0,
    "clahe2_tile": 8,
}


# ------------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIRS = [os.path.join(ROOT, "data", "samples"), os.path.join(ROOT, "data", "real")]
GT_DIR = os.path.join(ROOT, "data", "samples_gt")
OUT_DIR = os.path.join(ROOT, "results", "stage1")
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# 한글 파일명/경로도 읽고 쓸 수 있도록 하는 함수
def imread_any(path, flag=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flag)


def imwrite_any(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok:
        buf.tofile(path)


# ==================================================================
# 파이프라인 각 단계
# ==================================================================
def step1_brightness(L, cfg):
    """[논문 명시] L 채널 적응형 밝기 보정 / [우리 선택] 구체 방법"""
    if cfg["step1_method"] == "clahe":
        t = cfg["clahe1_tile"]
        clahe = cv2.createCLAHE(clipLimit=cfg["clahe1_clip"], tileGridSize=(t, t))
        return clahe.apply(L)
    return L.copy()


def step2a_edge_preserving(L, cfg):
    """[논문 명시] 엣지 보존 필터링 / [우리 선택] 구체 방법"""
    m = cfg["step2a_method"]
    if m == "bilateral":
        return cv2.bilateralFilter(L, cfg["bil_d"], cfg["bil_sigma_color"], cfg["bil_sigma_space"])
    if m == "nlm":
        return cv2.fastNlMeansDenoising(L, None, h=cfg["nlm_h"],
                                        templateWindowSize=7, searchWindowSize=21)
    return L.copy()


def step2b_local_contrast(L, cfg):
    """[논문 명시] 국소 대비 강화 / [우리 선택] 구체 방법"""
    m = cfg["step2b_method"]
    if m == "unsharp":
        blur = cv2.GaussianBlur(L, (0, 0), cfg["unsharp_sigma"])
        # 결과 = 원본 + amount × (원본 - 흐린것)
        return cv2.addWeighted(L, 1 + cfg["unsharp_amount"], blur, -cfg["unsharp_amount"], 0)
    if m == "clahe":
        t = cfg["clahe2_tile"]
        clahe = cv2.createCLAHE(clipLimit=cfg["clahe2_clip"], tileGridSize=(t, t))
        return clahe.apply(L)
    return L.copy()


def run_pipeline(bgr, cfg):
    """전체 1차 노이즈 제거. 단계별 L 채널도 함께 돌려줌 (비교용)"""
    # [논문 명시] LAB 변환 후 L / a / b 분리
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    L1 = step1_brightness(L, cfg)          # 1단계
    L2 = step2a_edge_preserving(L1, cfg)   # 2단계-①
    L3 = step2b_local_contrast(L2, cfg)    # 2단계-②

    # [논문 명시] 색 정보(a, b)는 원래 것 그대로 다시 합침
    out = cv2.cvtColor(cv2.merge([L3, a, b]), cv2.COLOR_LAB2BGR)

    def to_bgr(Lx):
        return cv2.cvtColor(cv2.merge([Lx, a, b]), cv2.COLOR_LAB2BGR)

    stages = {
        "0 Original": bgr,
        "1 Brightness (L)": to_bgr(L1),
        "2a Edge-preserving filter": to_bgr(L2),
        "2b Local contrast = Final": out,
    }
    return out, stages, {"L0": L, "L1": L1, "L2": L2, "L3": L3}


# ==================================================================
# [우리 추가] 획 대비 측정 — 논문에 없는 분석용 지표
#   CNR (Contrast-to-Noise Ratio)
#     = |획 평균 밝기 - 주변 배경 평균 밝기| / 주변 배경의 표준편차
#   값이 클수록 '글자가 배경에서 잘 구별된다'는 뜻.
#   배경 노이즈만 줄고 획이 유지되면 커지고,
#   획까지 같이 뭉개지면 작아짐.
# ==================================================================
def cnr(L, mask):
    stroke = mask > 0
    ring = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
    bg = ring & ~(cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0)
    Lf = L.astype(np.float32)
    return abs(Lf[stroke].mean() - Lf[bg].mean()) / (Lf[bg].std() + 1e-6)


def save_compare(stages, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    for ax, (name, img) in zip(axes.ravel(), stages.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=13)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = []
    for d in INPUT_DIRS:
        if os.path.isdir(d):
            files += [f for f in sorted(glob.glob(os.path.join(d, "*"))) if f.lower().endswith(EXTS)]
    if not files:
        print("입력 이미지가 없습니다. data/samples 또는 data/real 폴더를 확인하세요.")
        return

    setting = (f"step1={CONFIG['step1_method']} | step2a={CONFIG['step2a_method']} | "
               f"step2b={CONFIG['step2b_method']}")
    print("설정:", setting, "\n")

    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        img = imread_any(f)
        if img is None:
            print("읽기 실패:", f)
            continue

        out, stages, Ls = run_pipeline(img, CONFIG)
        imwrite_any(os.path.join(OUT_DIR, name + "_final.png"), out)
        save_compare(stages, f"{name}   [{setting}]", os.path.join(OUT_DIR, name + "_compare.png"))

        # 정답 마스크가 있으면 획 대비 측정
        gt = os.path.join(GT_DIR, name + "_mask.png")
        if os.path.exists(gt):
            mask = imread_any(gt, cv2.IMREAD_GRAYSCALE)
            vals = [cnr(Ls[k], mask) for k in ("L0", "L1", "L2", "L3")]
            rows.append([name] + [f"{v:.2f}" for v in vals])
            print(f"{name:22s} CNR  원본 {vals[0]:5.2f} → 1단계 {vals[1]:5.2f} "
                  f"→ 2a {vals[2]:5.2f} → 최종 {vals[3]:5.2f}")
        else:
            print(f"{name:22s} 처리 완료 (정답 마스크 없음 → CNR 생략)")

    if rows:
        with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp)
            w.writerow(["setting", setting])
            w.writerow(["image", "CNR_original", "CNR_step1", "CNR_step2a", "CNR_final"])
            w.writerows(rows)

    print("\n결과 저장 위치:", OUT_DIR)


if __name__ == "__main__":
    main()
