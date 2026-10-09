# -*- coding: utf-8 -*-
"""
candidates.py
검출기 공통 출력 형식  [우리 설계]

모든 노이즈 검출기(스크래치·녹·얼룩)는 '후보' 목록만 돌려주고, 지우지 않는다.
지울지 말지는 공통 3단 판정(decision.py)이 정한다.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Candidate:
    kind: str                       # "scratch" | "rust" | "stain"
    mask: np.ndarray                # 후보 픽셀 (bool, H×W)
    confident: bool                 # 검출기가 '확실한 노이즈'라고 판단했는가
    similar_to_marking: bool        # 마킹(글자·기호)과 닮았는가 → 닮았으면 판정 모듈이 제거하지 않음
    features: dict = field(default_factory=dict)   # 판정 근거 수치 (JSON 저장 가능)
    geometry: dict = field(default_factory=dict)   # 선분이면 {"p0": [x, y], "p1": [x, y]} 등
