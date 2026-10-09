# 2차 구조적 노이즈 제거 (stage2_structural) — 개발 중

담당: 박서연 · 설계안: 「2차 구조적 노이즈 제거 설계안」 문서 참고

1차(`stage1_denoise/`)와 팀 공용 파일은 수정하지 않고, 이 폴더 안에서만 독립적으로 동작합니다.

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 평가용 가상 데이터 생성기 (`src/make_stage2_samples.py`) | 완료 |
| 2 | 평가 코드 (`src/evaluate_stage2.py`, 인터페이스 `src/stage2_io.py`) | 완료 |
| 3 | 문자 보호 마스크 (`src/protect_mask.py`, 평가 `src/eval_protect.py`) | 완료 |
| 4 | 스크래치 검출 + 보수적 판정 + 선택적 제거 | 예정 |
| 5 | 반사 영역 표시 | 예정 |
| 6 | 파라미터 실험 · ablation | 예정 |

## 최종 통합 구조 (녹·얼룩 포함)

```
입력(1차 결과) ─┬─ 공통 ① 문자 보호 마스크 (protect_mask.py)  ← 스크래치·녹·얼룩이 모두 사용
               │      확실한 보호 / 의심 영역 / 나머지
               ├─ 검출기: 스크래치 (MVP) │ 녹 (후속) │ 얼룩 (후속)   ← 서로 독립 모듈
               ├─ 공통 ② 3단 판정: 제거 / 부분 제거 / 보존+불확실 표시
               ├─ 반사 영역 표시 (MVP)
               └─ 출력: 처리 이미지 + 검출·제거·불확실·반사 마스크 + needs_check (stage2_io.py)
```
- 녹·얼룩 검출기는 스크래치 MVP 이후 같은 인터페이스로 추가합니다. 평가는 노이즈 종류별로 따로 냅니다.
- 마킹 방향은 `polarity` 설정(`bright` / `dark`)으로 분리했습니다. MVP는 밝은 마킹 기준이며, 최종 목표는 양쪽 지원입니다.

## 1단계: 가상 데이터 생성기 [우리 평가 도구, 논문 아님]

```
python stage2_structural/src/make_stage2_samples.py                    # 개발용 dev (조건당 10장)
python stage2_structural/src/make_stage2_samples.py --split eval --n 30 # 독립 평가용 eval
python stage2_structural/src/preview_stage2_samples.py                 # 조건별 미리보기
pytest stage2_structural/tests -v                                      # 자동 테스트
```

- **dev / eval 분리:** dev 시드는 1000번대, eval 시드는 900000번대라 절대 겹치지 않습니다.
  규칙과 임계값은 dev로만 정하고, 최종 성능은 eval로만 보고합니다.
- **중첩 조건:** C1 안 겹침 / C2 일부 겹침(가는 스크래치) / C3 구분 어려움(획과 같은 폭·색의 스크래치, 직선 획 문자, 긴 직선 마킹)
- **보호 대상(`gt_protect`) = 문자 ∪ 기호(원·화살표) ∪ 긴 직선 마킹.**
  긴 직선 마킹은 분필선·용접선으로 **가정**해 보호 대상으로 분류했습니다 (의미 확인 필요).
- **손상 구분:**
  - `gt_preexisting_damage` = 보호 대상 ∩ 스크래치 → 입력 단계에서 이미 손상된 픽셀
  - `gt_protect_intact` = 보호 대상 − 스크래치 → 알고리즘이 이 픽셀을 바꾸면 "알고리즘이 추가로 만든 손상"
- **반사:** 시드 끝자리가 2·5·8인 이미지(30%)에만 넣고, 밝기 250 이상 포화 픽셀을 `gt_glare`로 저장합니다.
- `meta.json`에 정답 문자열(`gt_text_strings`)과 마킹별 박스(`box_xyxy`, `box_rotated`)가 있어
  추후 OCR CER 평가와 정형 문자 박스 입력에 사용합니다.

생성된 데이터(`data/`)와 결과(`results/`)는 Git에 올리지 않습니다. 같은 시드로 언제든 똑같이 다시 만들 수 있습니다.

## 2단계: 평가 코드 [우리 평가 도구, 논문 아님]

```
python stage2_structural/src/evaluate_stage2.py --method identity             # 아무것도 안 함 (기준선)
python stage2_structural/src/evaluate_stage2.py --method oracle_conservative  # 정답 사용, 겹친 부분 남김 (이상적 보수 제거)
python stage2_structural/src/evaluate_stage2.py --method oracle_remove_all    # 정답 사용, 겹친 부분까지 지움 (공격적 제거)
```
`oracle_*`는 정답을 쓰는 **평가 코드 점검용 기준선**이지 알고리즘이 아닙니다.

- **인터페이스:** 2차 방법은 `method(image, protect_boxes=None) -> Stage2Result`
  (처리 이미지 + 검출/제거/불확실/반사 마스크 + needs_check + report). 기존 OCR 모듈과 독립.
- **검출 성능과 제거 성능 분리:** 검출 = `detect_mask` vs 스크래치 정답,
  제거 = `remove_mask` vs (스크래치 − 보호 대상). 허용 거리(2px) 지표는 보호 대상 위 제거를 봐주지 않음.
- **손상 구분:** 알고리즘 손상 = 입력에서 온전했던 보호 픽셀 중 지웠거나(선언) 밝기가 20 이상 바뀐(실측) 픽셀.
  기존 손상(입력 단계에서 스크래치가 덮은 획)과 그중 알고리즘이 지운 픽셀은 따로 집계.
  반사로 가려진 보호 픽셀은 보존율 분모에서 제외하고 따로 집계.
- **CER 인터페이스:** `run(..., ocr_fn=f)`에 `f(crops) -> [문자열]`을 넘기면 처리 전후 CER을 함께 계산.
- 기준값(허용 거리 2px, 변화 판정 20)은 `EVAL_CFG`에 있으며 모두 [우리 설정]입니다.

## 3단계: 문자 보호 마스크 (공통 모듈 ①)

```
python stage2_structural/src/eval_protect.py                 # 보호 마스크만 평가 + 미리보기
python stage2_structural/src/eval_protect.py --boxes         # 정형 문자 박스를 함께 줄 때
python stage2_structural/src/eval_protect.py --polarity dark # 어두운 마킹 기준
```
- 근거: 보호 마스크 구조 [논문 3], 지우지 말고 표시 [논문 4], Sauvola 이진화 [논문 9, scikit-image 문서의 공식].
  Sauvola는 팀 공용 requirements를 바꾸지 않도록 OpenCV로 직접 구현했고 scikit-image와 결과를 대조했습니다.
- 그 외(선 분리, 글자다움 규칙, 흐린 획 검출, 글자에 딸린 선 처리, 모든 임계값)는 [우리 선택]이며 `PROTECT_CFG`에 있습니다.
- **주의:** `attach_radius`·`attach_frac`(화살표 등 글자에 딸린 선 판정)은 우리 생성기의 화살표 배치를 보고 정한 값이라,
  실제 현장 기호 배치에서는 다시 확인해야 합니다.
