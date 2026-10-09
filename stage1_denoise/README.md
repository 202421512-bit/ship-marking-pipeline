# 1차 노이즈 제거 프로토타입 (stage1_denoise)

담당: 박서연

Hao et al., *Electronics* 2026, 15(14), 3068 (Section 2.3)의 전처리 구조를 기반으로 구현한 1차 노이즈 제거.

| 순서 | 논문 명시 | 우리 선택 (기본값) |
|---|---|---|
| 0 | BGR → LAB, L 채널만 처리, a·b(색) 보존 | — |
| 1 | L 채널 적응형 밝기 보정 | CLAHE (clip 2.0, tile 8×8) |
| 2-1 | 엣지 보존 필터링 | Bilateral (d 9, sigmaColor 40, sigmaSpace 9) |
| 2-2 | 그 다음 국소 대비 강화 | Unsharp mask (sigma 3, amount 1.0) |

논문에는 구체 알고리즘과 파라미터가 명시되어 있지 않아, 위 오른쪽 열은 우리가 선택한 것입니다.
설정은 `src/stage1_denoise.py` 맨 위 `CONFIG`에서 바꿀 수 있습니다.

## 실행 (저장소 최상위 폴더에서, 가상환경 활성화 후)

```
python stage1_denoise/src/make_samples.py     # 가상 테스트 이미지 6장 + 정답 마스크 → data/samples, data/samples_gt
python stage1_denoise/src/stage1_denoise.py   # 1차 노이즈 제거 + 단계별 비교 그림 + CNR → results/stage1
python stage1_denoise/src/sweep_params.py     # 파라미터를 하나씩 바꿔 비교 → results/sweep
python stage1_denoise/src/ablation.py         # 단계를 하나씩 끄고 비교 → results/ablation
```

실제 이미지는 `stage1_denoise/data/real/`에 넣으면 `stage1_denoise.py`가 함께 처리합니다 (Git에는 올라가지 않음).
팀 예시 데이터셋(zip)은 `stage1_denoise/data/team_examples/`에 풀어 넣으면 하위 폴더까지 처리합니다 (Git에는 올라가지 않음).

옵션:
```
python stage1_denoise/src/stage1_denoise.py --only team_examples   # 한 폴더만 처리 (samples / real / team_examples)
python stage1_denoise/src/stage1_denoise.py --no-autoscale         # 크기 자동 보정 끄기 → results/stage1_noscale
```

### 이미지 크기 자동 보정 (우리 선택, 논문에 없음)
픽셀 단위 파라미터(Bilateral d·sigmaSpace, Unsharp sigma)를 긴 변 1600px 기준으로 정하고,
이미지 크기에 비례해 자동 조정합니다. 1600px 가상 이미지에서는 값이 바뀌지 않으므로 기존 실험 결과는 그대로입니다.
CLAHE 타일은 '칸 개수'라 크기와 무관하여 보정하지 않습니다.

## 지금까지의 결과 요약 (가상 이미지 기준)

- 그림자·불균일 조명: 개선됨 (1단계 CLAHE가 담당)
- 녹: 거의 남음 (색으로 구별되는 노이즈인데, 구조상 색을 보존함)
- 스크래치: 남고 오히려 강조됨 (엣지로 인식되어 보존·강화됨)
- 강한 반사: 복원 안 됨 (픽셀 포화, 어떤 파라미터로도 개선 미미)
- 획 손상: 2-1단계(Bilateral)가 약한 획의 대비를 낮춤
- 가장 영향이 큰 변수: CLAHE clip. 약한 글씨엔 클수록 좋고, 녹·스크래치가 많으면 클수록 나쁨 → 단일 최적값 없음
- 단계별 끄기: 전체 구성이 6장 모두 최고. 2-1 없이 2-2만 하면 배경 노이즈가 커져 원본보다 나빠지는 경우도 있음

## 알려진 이슈

- 기본 파라미터는 1600×1200 가상 이미지 기준입니다. 팀 예시 데이터셋(640px 내외)처럼 작은 이미지에서는
  Unsharp mask(sigma 3)가 상대적으로 넓게 걸려 글자 주변에 어두운 테두리(halo)가 생겼습니다.
  → 크기 자동 보정으로 halo가 줄었지만 완전히 없어지지는 않습니다.
    대신 작은 이미지에서는 Bilateral 범위도 줄어 스크래치 등 배경 노이즈 억제가 약해지는 trade-off가 있습니다.
