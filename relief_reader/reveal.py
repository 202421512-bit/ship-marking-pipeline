"""빛 방향을 바꿔 찍은 사진 4장으로 눈에 안 보이는 각인·눌린 자국 글자를 드러낸다.

원리
  표면이 파이거나 튀어나온 곳은 빛이 비스듬히 들어올 때 한쪽은 밝고 반대쪽은 그림자가 진다.
  왼쪽/오른쪽 조명 사진의 차이 → 가로 방향 기울기, 위/아래 조명 사진의 차이 → 세로 방향 기울기.
  두 기울기의 크기를 합치면 색과 상관없이 '표면의 요철'만 남는다.

사용법
  폴더에 top / bottom / left / right 이름의 사진 4장을 넣고 (확장자 jpg, jpeg, png)
  평소 조명으로 찍은 normal 사진도 있으면 비교 화면 왼쪽에 그 사진을 씁니다 (없으면 4장 평균)
    python relief_reader/reveal.py 사진폴더
  결과: 사진폴더/result/ 에 compare.png(일반 사진 vs 드러난 글자)와 단계별 이미지 저장
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DIRECTIONS = ("top", "bottom", "left", "right")
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_image(folder: Path, name: str) -> Path:
    for ext in EXTS:
        p = folder / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{folder}에 {name}.jpg(또는 png) 사진이 없습니다.")


def load_gray(path: Path, max_side: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"사진을 열 수 없습니다: {path} (아이폰 HEIC라면 JPG로 바꿔주세요)")
    scale = max_side / max(img.shape)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def align_to(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """손떨림으로 조금 어긋난 사진을 기준 사진에 맞춘다 (평행 이동만)."""
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            cv2.GaussianBlur(ref, (5, 5), 0), cv2.GaussianBlur(img, (5, 5), 0), warp,
            cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5))
    except cv2.error:
        return img
    return cv2.warpAffine(img, warp, ref.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


def flatten(img: np.ndarray) -> np.ndarray:
    """조명이 한쪽만 밝은 큰 얼룩(그라데이션)을 없애고 작은 요철만 남긴다."""
    sigma = max(img.shape) / 25
    background = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.log(img + 1e-3) - np.log(background + 1e-3)


def stretch(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / (b - a + 1e-6), 0, 1)


def reveal(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    flat = {k: flatten(v) for k, v in images.items()}
    gx = flat["right"] - flat["left"]    # 가로 방향 기울기
    gy = flat["bottom"] - flat["top"]    # 세로 방향 기울기
    relief = np.sqrt(gx ** 2 + gy ** 2)  # 요철 세기 (방향 무관)
    relief = cv2.GaussianBlur(relief, (0, 0), 1.0)
    relief_u8 = (stretch(relief) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(relief_u8)
    # 비스듬한 조명 효과(엠보싱): 사람이 보기에 입체감이 있는 버전
    emboss = ((stretch(gx + gy) * 255)).astype(np.uint8)
    normal = np.mean([images[d] for d in DIRECTIONS], axis=0)
    return {
        "normal": (stretch(normal, 0.5, 99.5) * 255).astype(np.uint8),
        "relief": relief_u8,
        "relief_clahe": clahe,
        "emboss": emboss,
    }


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h = max(24, img.shape[0] // 14)
    bar = np.full((h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, int(h * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, h / 34, (0, 0, 0), max(1, h // 20))
    return np.vstack([bar, out])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="4방향 조명 사진으로 각인·눌린 자국 글자 드러내기")
    ap.add_argument("folder", help="top/bottom/left/right 사진이 있는 폴더")
    ap.add_argument("--max-side", type=int, default=1600, help="계산용으로 줄일 최대 크기(px)")
    ap.add_argument("--no-align", action="store_true", help="사진 위치 자동 맞춤 끄기")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    images = {d: load_gray(find_image(folder, d), args.max_side) for d in DIRECTIONS}
    shapes = {v.shape for v in images.values()}
    if len(shapes) != 1:
        raise ValueError(f"사진 4장의 크기가 다릅니다: {shapes}. 같은 위치·같은 설정으로 찍어주세요.")
    if not args.no_align:
        ref = images["top"]
        images = {k: (v if k == "top" else align_to(ref, v)) for k, v in images.items()}

    res = reveal(images)
    try:
        plain = load_gray(find_image(folder, "normal"), args.max_side)
        if plain.shape == images["top"].shape:
            res["normal"] = (stretch(align_to(images["top"], plain), 0.5, 99.5) * 255).astype(np.uint8)
    except FileNotFoundError:
        pass
    out = folder / "result"
    out.mkdir(exist_ok=True)
    for name, img in res.items():
        cv2.imwrite(str(out / f"{name}.png"), img)
    compare = np.hstack([label(res["normal"], "normal photo"), label(res["relief_clahe"], "revealed (4-light)")])
    cv2.imwrite(str(out / "compare.png"), compare)
    print(f"완료: {out / 'compare.png'}")


if __name__ == "__main__":
    main()
