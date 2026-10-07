"""CI 동작 확인용 최소 테스트: 합성 철판 이미지 → 노이즈 제거 → 마킹 검출."""
import cv2
import numpy as np


def make_plate(with_noise=True):
    """회색 철판 위에 흰색 사각형 마킹 1개를 그리고, 필요하면 점(salt & pepper) 노이즈를 뿌린다."""
    img = np.full((200, 300), 120, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 100), 255, thickness=-1)
    if with_noise:
        rng = np.random.default_rng(0)
        mask = rng.random(img.shape)
        img[mask < 0.02] = 0
        img[mask > 0.98] = 255
    return img


def detect_markings(img, min_area=500):
    """밝은 영역을 이진화하고, 일정 면적 이상인 외곽선만 마킹으로 본다."""
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def test_opencv_available():
    assert cv2.__version__
    assert np.__version__


def test_median_blur_reduces_noise():
    clean = make_plate(with_noise=False).astype(np.int16)
    noisy = make_plate(with_noise=True)
    denoised = cv2.medianBlur(noisy, 5)

    err_before = np.abs(noisy.astype(np.int16) - clean).mean()
    err_after = np.abs(denoised.astype(np.int16) - clean).mean()
    assert err_after < err_before


def test_detects_single_marking_after_denoise():
    denoised = cv2.medianBlur(make_plate(with_noise=True), 5)
    markings = detect_markings(denoised)

    assert len(markings) == 1
    x, y, w, h = cv2.boundingRect(markings[0])
    assert abs(x - 50) <= 3 and abs(y - 50) <= 3
    assert abs(w - 101) <= 5 and abs(h - 51) <= 5


def test_sync_check():
    """팀원 git pull 동기화 확인용 테스트 (2026-10-08 추가)."""
    message = "hello from pac2026"
    assert message.startswith("hello")
