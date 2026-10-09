# -*- coding: utf-8 -*-
"""
stage2_io.py
2차 노이즈 제거의 '입출력 약속(인터페이스)'  [우리 설계]

어떤 2차 방법이든 아래 형태로 결과를 돌려주면 evaluate_stage2.py로 똑같이 평가할 수 있다.
기존 1차 코드와 OCR 모듈은 이 파일을 몰라도 된다 (독립 동작).

    method(image_bgr, protect_boxes=None) -> Stage2Result

    image_bgr     : 입력 이미지 (1차 결과), uint8 BGR
    protect_boxes : (선택) 정형 문자 박스 목록 [(x1, y1, x2, y2), ...]. 없으면 None

Stage2Result 필드
    image          처리된 이미지 (입력과 같은 크기, uint8 BGR)
    detect_mask    노이즈로 '검출'한 픽셀       (bool, H×W)
    remove_mask    실제로 '제거'(값을 바꾼) 픽셀 (bool, H×W)  ⊆ detect_mask 권장
    uncertain_mask 검출했지만 불확실해서 남겨 둔 픽셀 (bool, H×W)
    glare_mask     반사로 판독 불가한 영역       (bool, H×W)
    needs_check    사람 확인이 필요한지         (bool)
    report         자유 형식 요약 (dict, JSON으로 저장 가능해야 함)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stage2Result:
    image: np.ndarray
    detect_mask: np.ndarray
    remove_mask: np.ndarray
    uncertain_mask: np.ndarray
    glare_mask: np.ndarray
    needs_check: bool = False
    report: dict = field(default_factory=dict)

    def validate(self, input_image):
        h, w = input_image.shape[:2]
        if self.image.shape != input_image.shape or self.image.dtype != np.uint8:
            raise ValueError(f"image는 입력과 같은 크기의 uint8이어야 함: {self.image.shape}, {self.image.dtype}")
        for name in ("detect_mask", "remove_mask", "uncertain_mask", "glare_mask"):
            m = getattr(self, name)
            if m.shape != (h, w):
                raise ValueError(f"{name} 크기가 이미지와 다름: {m.shape} != {(h, w)}")
            setattr(self, name, m.astype(bool))
        self.needs_check = bool(self.needs_check)
        return self


def empty_result(image_bgr, **kw):
    """아무것도 하지 않은 결과 (기준선·테스트용)."""
    h, w = image_bgr.shape[:2]
    z = np.zeros((h, w), bool)
    return Stage2Result(image=image_bgr.copy(), detect_mask=z.copy(), remove_mask=z.copy(),
                        uncertain_mask=z.copy(), glare_mask=z.copy(), **kw)
