# 2차 구조적 노이즈 제거 (stage2_structural) — 개발 중

담당: 박서연 · 설계안: 「2차 구조적 노이즈 제거 설계안」 문서 참고

1차(`stage1_denoise/`)와 팀 공용 파일은 수정하지 않고, 이 폴더 안에서만 독립적으로 동작합니다.

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 평가용 가상 데이터 생성기 (`src/make_stage2_samples.py`) | 완료 |
| 2 | 평가 코드 (획 보존율, 검출/제거 성능 분리, 처리 시간, CER 인터페이스) | 예정 |
| 3 | 문자 보호 마스크 | 예정 |
| 4 | 스크래치 검출 + 보수적 판정 + 선택적 제거 | 예정 |
| 5 | 반사 영역 표시 | 예정 |
| 6 | 파라미터 실험 · ablation | 예정 |

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
