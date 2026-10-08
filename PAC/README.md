# PAC Mission 3 — Ship Block Marking Analysis (Research Prototype)

> **본 프로젝트는 PAC Mission 3의 전체 시스템이 아닌,
> 표기정보 인식의 구조적 분석 및 신뢰도 검증을 위한 연구용 prototype이며,
> 향후 표기 의미 해석 및 용접조건 판단 모듈과 연결할 수 있도록 설계되었다.**

Research Prototype · Proof of Concept · Demo — 산업 현장 검증을 거치지 않았으며, 어떤 결과도 정확도를 보장하지 않는다.

---

## 1. PAC Mission 3 개요

HD현대로보틱스 PAC Mission 3 「선박 블록 표기정보 인식·해석 기반 용접조건 자동 설정」은 다음을 요구한다.

1. 표기정보 검출
2. 정형·비정형 표기정보 인식 (수기 문자/숫자/기호/약어 포함)
3. 표기 의미 및 작업지시 해석
4. 용접 작업 및 조건 자동 판단
5. 결과 시각화

추가 요구: 회전·기울어짐/저조도/부분 훼손 대응, 결과별 신뢰도 표시, 모호한 결과의 후보 제시, 원본→인식→해석 추적성, 불확실 시 작업자 확인 요청, 로봇 시스템 연계 가능 구조.

## 2. 프로젝트 목적

**표기정보 인식의 정확도·신뢰도·모호성 판단을 강화하는 연구용 핵심 분석 모듈**을 구현한다.
픽셀 기반 인식만 쓰지 않고, 문자의 **topology**와 **geometry**를 분석하고, 손상된 표기를 **topology constraint**로 복원한 뒤,
**Bayesian inference**와 **VLM**을 서로 독립적인 evidence로 사용하여 최종 판단한다.

## 3. 본 prototype의 범위

| 포함 | 제외 (이번 단계) |
|---|---|
| crop된 단일 표기(문자/숫자/기호) 분석 | 전체 블록 이미지에서의 문자 영역 detection |
| 전처리, topology, geometry, 복원, Bayesian, VLM, fusion, decision | 완전한 OCR 시스템 (Tesseract/EasyOCR/PaddleOCR 미사용) |
| 구조화된 JSON 출력, 시각화, GUI | 표기 의미 해석, 용접조건 계산, 로봇 제어 (인터페이스만 정의) |

> **현재 분석 pipeline(`--input`, `--demo`, GUI)은 single-character 분석용이다.**
> 입력은 문자 하나만 crop된 이미지여야 한다. 여러 문자로 된 표기("WPS-01", "6 mm")는
> 아래 실제 데이터셋 경로에서 **문자 단위로 segmentation한 뒤** 문자별로 학습·평가한다.
> 전체 장면에서 표기 영역을 찾는 detection은 구현되어 있지 않다.

### 3-1. 실제 데이터셋 (Symbols/) — Real Dataset

직접 검사한 실제 구조 (`python Main.py --dataset-audit`):

| 폴더 | 파일 | 내용 | label |
|---|---|---|---|
| `Formal Text/` | 132 PNG = 66 그룹 × 2 (`formal_NNN_k.png`) | 정형 표기 문자열 ("F6", "6 mm", "WPS-01", "ARROW SIDE" …) | **없음** (폴더·파일명에 class 정보 없음) |
| `Informal Text/` | 165 PNG = 55 그룹 × 3 (`informal_NNN_k.png`) | 수기 표기 문자열 | **없음** |
| `Informal Text/` | 30 PNG (`scene_NNN.png`, 900×600) | 표기 + 배경 선이 있는 전체 장면 (crop 아님) | 학습·평가 제외 |
| `Welding Symbol/` | 54 PNG = 18 class × 3 (`<class>_NN.png`) | 용접 기호 도면 | **파일명에서 추출** |

모든 파일은 PNG이며 규정 문서(PDF/TXT/DOCX/CSV)는 없다.

**Label 정책 (추측 금지)**

1. 사용자가 `data/transcriptions.csv`의 `transcription` 열에 그룹별 문자열을 직접 입력한다 (그룹 내 이미지는 같은 문자열).
   특정 이미지만 다르면 `data/dataset_manifest.csv`의 해당 행 **`transcription_override`** 열에 입력한다 (이미지별 값이 그룹 값보다 우선).
   manifest의 `transcription` 열은 실제 적용된 값을 보여주는 **출력 전용** 열이므로 수정해도 반영되지 않는다.
   그룹 transcription을 지우면 다음 실행에서 해당 이미지는 `NO_TRANSCRIPTION`이 되고 학습·평가에서 제외된다.
2. 원본 `transcription`은 그대로 보존되고, 비교용 `normalized_transcription`이 따로 만들어진다.
   기본 `space_policy="ignore"`: 공백은 잉크가 아니므로 문자 수에서 제외 ("6 mm" → "6mm", 3자). `config.DatasetConfig`에서 변경 가능.
3. 각 이미지를 문자 단위로 segmentation한다 (배경 제거 binarization → contrast 필터로 녹/얼룩 제거 → 텍스트 줄 band 필터 →
   가로로 겹치는 성분 병합 ('=', ':', '%') → 가로로 긴 밑줄/화살표 선 제외).
4. **`segmentation_count == expected_character_count`일 때만** `label_source=manual`, `valid_for_training=true`,
   `valid_for_evaluation=true` (`validation_status=COUNT_MATCH`). 문자 i ↔ 왼쪽에서 i번째 segment로 대응한다.
5. transcription이 비었거나(`NO_TRANSCRIPTION`) 개수가 다르면(`COUNT_MISMATCH`) 자동 제외되지만 manifest에는 남는다.
   `results/segmentation_preview/`에서 박스(초록=문자, 빨강=제외된 선)를 보고 원인을 확인할 수 있다.
6. `scene_*.png`: `style=INFORMAL_SCENE`, `valid_for_training=false`, `valid_for_evaluation=false`.
   삭제하지 않으며, 향후 *전체 이미지 → 표기영역 detection → crop → character analysis* 확장용 qualitative demonstration data로 유지한다.
7. 화면에 보이는 문자만 입력한다. 글자 아래 밑줄 화살표는 자동 제외되므로 입력하지 않는다. "← F6"처럼 글자로 그린 화살표는 문자로 입력한다.

**입력 후 재분석 (한 명령)**

```bat
.venv\Scripts\python.exe Main.py --train-reference
```

이 명령은 매번 ① audit을 다시 실행해 새 transcription을 검증하고 ② 유효한 문자 segment의 topology+geometry feature를
`data/training_features.csv`에 기록하고 ③ **DATA-FITTED REFERENCE MODEL**(class별 mean/std; Model A=class, Model B=class+style)을
`data/prototypes/data_fitted_model.json`, `results/model_statistics.json`에 저장한다. 표본이 `min_samples_per_class`(기본 3) 미만인 class는 제외된다.
유효 데이터가 없으면 모델을 만들지 않고(이전 모델은 삭제), pipeline은 synthetic font prototype을 계속 쓴다.
`--input`은 기본(`--reference auto`)으로 data-fitted 모델이 있으면 그것을 사용한다. **`--demo`는 항상 synthetic 모델을 사용**하며
결과에 "SYNTHETIC DEMO"로 표시되어 실제 데이터셋 평가와 구분된다.

**Welding Symbol**: 문자와 섞지 않는 별도 `welding_symbol` dataset (`data/welding_symbol_manifest.csv`, label_source=filename).
도면 안에 class 이름 캡션이 그려져 있어 이미지 기반 학습 시 label leakage 위험이 있다. 규정 텍스트가 없으므로
`data/welding_rules.json`의 meaning/rule은 `null`, `status="unparsed"`로 기록하며 내용을 생성하지 않는다.

## 4. 전체 pipeline

```
Input Marking (원본은 수정하지 않음, SHA-256 기록)
 → Image Preprocessing        src/preprocessing.py
 → Topology Analysis          src/topology.py
 → Geometry Analysis          src/geometry.py
 → Topology-Constrained Restoration  src/restoration.py
 → Bayesian Shape Inference   src/bayesian.py   (prototype reference model: src/prototypes.py)
 → VLM Visual Verification    src/vlm.py
 → Evidence Fusion            src/fusion.py
 → Confidence / Ambiguity Decision   src/decision.py
 → Visualization              src/visualization.py
 → Structured Output          results/result.json
```

오케스트레이션과 단계별 시간 측정은 `src/pipeline.py`, 모든 파라미터는 `src/config.py`에 있다.

**전처리 순서**: grayscale → median + Non-local-means 잡음 제거(잡음 추정값에 비례한 강도) → 조명 보정(배경 나눗셈) + percentile 정규화 → CLAHE(`auto`: 저대비일 때만) → Otsu(또는 adaptive) → 전경/배경 자동 판별(테두리 기준) → 작은 noise component 제거 → (선택) closing → 크기 정규화(64×64, 종횡비 유지).
CLAHE를 잡음 제거보다 먼저 하면 speckle noise가 증폭되어 이 순서를 택했다.

## 5. Topology

`src/topology.py` — "Topology-Aware Character Recognition"

* **β0** = 8-연결 잉크 성분 수, **β1** = hole 수 (경계에 닿지 않는 4-연결 배경 성분, 쌍대 연결성)
* **Euler characteristic** χ = β0 − β1
* **Skeleton** (scikit-image) + 짧은 spur pruning, skeleton length, **endpoints**(이웃 1), **branch points**(이웃 ≥3 군집)
* component/hole 면적, skeleton density
* **Persistent homology (GUDHI)**: signed distance filtration(잉크 내부 음수) 위의 cubical complex.
  작은 양의 birth를 갖는 H1은 "간격을 메우면 닫히는 loop", 즉 **near-hole**을 뜻한다 — 복원의 근거가 된다.
  GUDHI 실패 시 β0/β1/χ/skeleton으로 fallback하며 프로그램은 계속 실행된다.

구조해석의 SIMP topology optimization을 문자 이미지에 적용하는 것이 **아니다**. topology는 시각화용 feature가 아니라 복원 목적함수와 Bayesian likelihood에 실제로 사용된다.

## 6. Geometry

`src/geometry.py` — width, height, aspect ratio, area ratio, perimeter, circularity, compactness, convex hull area, solidity,
centroid(bbox 내 정규화 위치), orientation, eccentricity, Hu moments(−sign·log10), quadrant density, radial distance histogram,
skeleton length / endpoints / branch points. 모든 비교는 정규화된 형태(prototype 평균·표준편차 기반 z-score)로 수행한다.

## 7. Topology-Constrained Restoration

`src/restoration.py`

후보: original, closing 3/5/7, opening 3/5, dilation, erosion, hole filling, small component removal (고정된 10개, 계산량 제한).

```
J = λ_data·L_data + λ_topology·L_topology + λ_geometry·L_geometry + λ_change·L_change
    (기본 0.30 / 0.30 / 0.25 / 0.15, config.py에서 변경)
```

* `L_data` = 1 − soft Dice(후보 mask, 회색조 잉크 evidence) — 관측 영상과의 일치도
* `L_topology` = d/(1+d), d = w0|β0−β0ᵗ| + w1|β1−β1ᵗ| (+ endpoint/branch 차이)
* `L_geometry` = d/(1+d), d = 정규화 geometry feature의 평균 |z| (aspect, area, perimeter, circularity, solidity, eccentricity, Hu, skeleton length)
* `L_change` = changed_pixels / total_pixels

**Target topology를 hard-code하지 않는다.** 관측 mask로 preliminary Bayesian 분석을 하고(예: 8=0.61, B=0.25, 0=0.05),
Top-K(기본 3) 후보 prototype을 reference로 쓴다. 각 복원 후보에 대해 구조 손실이 최소인 reference 가설을 사용한다:
`J(r) = λd·Ld(r) + λc·Lc(r) + min_k[λt·Lt(r,k) + λg·Lg(r,k)]`. 관측과 지나치게 다른 복원은 `L_data`, `L_change`가 억제한다.
모든 후보는 `results/restoration_candidates.csv`에 저장된다.

## 8. Bayesian inference

`src/prototypes.py`, `src/bayesian.py`

* **Prototype statistical reference model (학습 아님)**: 시스템 폰트(Arial, Calibri, Consolas, Verdana, Tahoma, Times, Segoe UI …; 없으면 matplotlib의 DejaVu)로
  클래스(0–9, A–F; `config.CLASSES`에 추가 가능)를 렌더링하고 회전/스케일/이동/팽창/침식/blur/threshold 변형을 적용,
  각 샘플의 topology+geometry feature의 클래스별 평균·표준편차를 `data/prototypes/`에 캐시한다.
* `P(c|x) ∝ P(x|c)P(c)`, Gaussian log-likelihood `−0.5 Σ[(x−μ)²/σ² + log σ²]`, σ에 floor/epsilon 적용, 기본 uniform prior(`BayesianConfig.priors`로 변경), log-sum-exp 정규화.
* topology / geometry log-likelihood를 **분리하여 출력**한다. feature 독립 가정으로 인한 과신을 줄이기 위해 temperature(기본 4)로 log-likelihood를 나눈다.
* 출력: Top-1, posterior, Top-3/Top-5, entropy H = −Σ p log p, 정규화 entropy, margin.

> 이 posterior는 **prototype reference model에 대한 posterior**이며, 산업 현장에서 calibration된 정답 확률이 아니다.

## 9. VLM verification

`src/vlm.py` — VLM은 OCR engine이 아니라 **독립적인 visual verification**이다.
입력은 original crop과 restored crop 뿐이며, 독립성을 위해 Bayesian 결과는 **전달하지 않는다**.
보이는 형태 분석, 가장 가능성 높은 label, alternatives, 손상 고려, 없는 정보 생성 금지, 모호성 표시를 요청하고 JSON으로 받는다.

* 화면 표기: Bayesian = **Posterior Probability**, VLM = **Model-Reported Confidence** (calibration되지 않음)
* `.env`의 `OPENAI_API_KEY`, `OPENAI_VLM_MODEL`을 사용 (`.env.example` 참고, 키를 코드에 쓰지 않음)
* 키가 없으면 `Status = UNAVAILABLE`, 호출 실패 시 `ERROR`, `--no-vlm`이면 `DISABLED` — 어떤 경우에도 나머지 pipeline은 정상 실행된다.

## 10. Evidence fusion

`src/fusion.py` — 두 값은 같은 의미의 확률이 아니므로 **heuristic evidence fusion**으로 정의한다.

```
fusion(c) ∝ Bayes(c)^α × VLM_evidence(c)^(1−α),  α = 0.70 (config)
```

VLM evidence는 top/alternatives의 reported confidence에 floor를 더해 정규화한 벡터다. VLM이 없으면 Final = Bayesian.
Bayesian Top-1과 VLM Top-1을 비교해 `AGREEMENT` / `DISAGREEMENT` / `NOT_AVAILABLE`을 기록한다.

## 11. Human review

`src/decision.py` — **demo decision threshold** (산업 안전 기준 아님, `DecisionConfig`에서 설정)

| Level | 조건 (요약) |
|---|---|
| `AMBIGUOUS` | Bayesian/VLM 불일치, top-1/top-2 margin < 0.15, 정규화 entropy > 0.70 중 하나 |
| `HIGH_CONFIDENCE` | fusion ≥ 0.80, posterior ≥ 0.70, 정규화 entropy ≤ 0.35, topology·geometry 일관성, VLM 일치(conf ≥ 0.70) 모두 충족 |
| `REVIEW_REQUIRED` | 그 외 |

`HIGH_CONFIDENCE`가 아니면 **`HUMAN_REVIEW_REQUIRED`**이며 후보 Top-3를 제시한다. 기본 설정에서는 독립 VLM 검증이 없으면
HIGH가 될 수 없다(`require_vlm_for_high_confidence=True`). 용접조건은 어떤 경우에도 자동 확정하지 않는다.

* Topology consistency: 복원된 (β0, β1)이 후보 prototype의 전형값과 일치하는가
* Geometry consistency: geometry feature 평균 |z| ≤ 1.5

## 12. Traceability

`results/result.json`에서 모든 단계가 `stage_id`/`input_ref`로 연결된다.

```
input(sha256) → S1 preprocessing → S2 topology → S3 geometry → S4 restoration
 → S5 bayesian → S6 vlm → S7 fusion → S8 decision
```

중간 mask는 `results/artifacts/`에 저장되며, prototype 모델 메타데이터, config 가중치, 단계별 시간이 함께 기록된다.

## 13. Demo 실행

```bat
.venv\Scripts\python.exe Main.py --demo
```

Pillow로 깨끗한 "8"을 만든 뒤 (seed = 42, 항상 동일) stroke removal, partial break, 윤곽 손상, 회전, Gaussian blur, 불균일 조명,
intensity variation, speckle noise, 작은 noise component를 적용한다 (`data/demo/`). "8"은 hole 2개 구조를 보여주기 좋아 선택했다.
결과는 알고리즘이 계산한 그대로 표시되며 특정 label에 유리하게 조작하지 않는다.
**Demo는 synthetic "8" 전용 시연이며 실제 데이터 평가가 아니다.** Demo 이미지는 `data/demo/`에만, 결과는
`results/demo_synthetic/`에만 저장된다 (실제 입력 폴더 `data/input/`에는 쓰지 않는다).

## 14. 실제 이미지 실행

실제 마킹 문자열 전체 인식 점검 (Formal / Informal 이미지, 분할 → topology/geometry → Bayesian → 문자열 재구성):

```bat
.venv\Scripts\python.exe Main.py --real-marking-recognition
```

결과: `results/real_marking_recognition/` (예시·실패 사례, 분할 감사 시트, CER / 문자열 정확도).

단일 문자 crop 하나만 분석 (single-character pipeline, 결과는 `results/single_image/`):

```bat
.venv\Scripts\python.exe Main.py --input <문자 하나만 crop된 이미지>.png --no-vlm
```

지원 형식: `.png .jpg .jpeg .bmp`. `--input`은 **문자 하나만 crop된 이미지**용이다.
옵션: `--output DIR`, `--rebuild-prototypes`, `--reference auto|synthetic|data`.

데이터셋 명령: `--dataset-audit` (manifest·검증·미리보기 갱신, transcription 보존), `--train-reference` (audit + data-fitted 모델).

## 15. GUI 실행

```bat
.venv\Scripts\python.exe Main.py --gui
```

* 상단: **[도면/이미지 열기] [Demo] [분석 시작] [결과 폴더 열기]**, VLM 사용 체크박스
* 중앙: 원본 / 복원 / Skeleton overlay — 마우스 휠 확대·축소, 드래그 패닝, 더블클릭 맞춤
* 우측: Topology, Restoration, Geometry, Bayesian, VLM, Fusion, Decision
* 분석은 별도 thread에서 실행되어 UI가 멈추지 않는다.
* 대형 도면은 `QImageReader.setScaledSize`로 **축소 preview만 메모리에 올린다**(`GUIConfig.preview_max_side`). 분석은 원본 파일에서 별도로 수행된다.

## 16. 결과 파일

| 파일 | 내용 |
|---|---|
| `results/result.json` | 구조화 출력 + 전 단계 추적 정보 + timing |
| `results/restoration_candidates.csv` | candidate_id, operation, L_data, L_topology, L_geometry, L_change, J (+ reference_class, β0, β1) |
| `results/preprocessing.png` | 원본과 전처리 각 단계 |
| `results/topology_analysis.png` | Original, Binary, Skeleton, Components, Holes, Endpoints, Branch points, persistence diagram, 수치 |
| `results/restoration_comparison.png` | 복원 후보 이미지와 J 구성 항목 |
| `results/geometry_analysis.png` | bbox/hull/centroid/축, radial histogram, prototype 대비 z-score |
| `results/bayesian_posterior.png` | preliminary vs final posterior, topology/geometry 기여도 |
| `results/final_dashboard.png` | 발표용 종합 dashboard |
| `results/artifacts/*.png` | 관측 mask, 복원 mask, skeleton |
| `data/dataset_manifest.csv` | 이미지별 style, group, transcription, 검증 상태 (재실행 시 사용자 입력 보존) |
| `data/transcriptions.csv` | 그룹별 transcription 입력 템플릿 (Formal 66 + Informal 55) |
| `data/welding_symbol_manifest.csv`, `data/welding_rules.json` | 별도 welding symbol dataset, 규칙은 unparsed |
| `data/training_features.csv` (+ `results/`) | 검증된 문자 segment별 topology/geometry feature |
| `results/dataset_summary.json`, `results/model_statistics.json` | 데이터셋 집계, data-fitted 모델 통계 |
| `results/segmentation_preview/*.png` | 문자 segmentation 확인용 이미지 |

## 17. 현재 구현 범위

구현: 전처리, topology(+GUDHI persistence), geometry, prototype reference model, topology-constrained restoration,
Bayesian inference, VLM verification(옵션), heuristic fusion, demo-threshold decision, 시각화, CLI, GUI.

한계: 단일 crop 입력만 지원; prototype은 인쇄체 폰트 기반이라 수기 표기의 변동은 충분히 반영되지 않음; posterior/threshold는
calibration되지 않음; restoration 후보는 고정된 morphology 집합; 약어(다문자)는 아직 클래스로 정의되지 않음.

## 18. 향후 semantic interpretation 연결

`result.json`의 `recognized_mark`, `decision`, `decision_detail.review_candidates`, `raw_image`가 입력 계약이다
(`downstream_interfaces.semantic_interpretation`). 다문자 표기는 문자 단위 결과를 순서대로 묶어 약어 사전/문법 규칙으로 해석하고,
`HUMAN_REVIEW_REQUIRED`인 항목은 해석 단계로 넘기지 않거나 확인 후 넘기는 구조를 권장한다.

## 19. 향후 welding rule / condition engine 연결

semantic 출력(용접 종류, 위치, 각장 등 작업지시)을 rule engine이 받아 WPS/사내 기준표와 대조해 조건을 결정하는 구조를 가정한다.
`welding_status`는 현재 `NOT_IMPLEMENTED`이며, 본 prototype은 어떤 용접조건도 계산하거나 산업 기준처럼 제시하지 않는다.

## 20. 향후 robot controller 데이터 연결

검증된 작업지시 + 조건을 로봇 컨트롤러 job/파라미터 형식으로 변환하는 adapter를 별도 모듈로 둔다
(`downstream_interfaces.robot_controller`). 추적성을 위해 `run_id`와 입력 `sha256`을 로봇 작업 로그까지 전달하는 것을 권장한다.

---

### 환경

기존 `.venv`(Python 3.12)와 `requirements.txt`를 그대로 사용한다 (`venv.bat`로 생성). 추가 패키지 없음.
