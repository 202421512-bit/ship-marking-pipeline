"""Write the refined presentation notes / index from saved CSVs and copy the key figures into
results/final_presentation_refined/ (copy only, SHA-256 verified, nothing overwritten)."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main() -> int:
    s = pd.read_csv(ROOT / "reports" / "phase4_refined" / "refined_summary.csv")
    s["image"] = s.image.str.replace("예시", "example")
    ver = json.loads((R / "progress_evolution_refined" / "verification.json").read_text(encoding="utf-8"))
    h1 = pd.read_csv(R / "progress_evolution_refined" / "H1_feature_space_evolution.csv")
    h3 = pd.read_csv(R / "progress_evolution_refined" / "H3_optimization_progress_summary.csv")
    rows = "\n".join(f"| {r.image} | {r.accepted_group_px} | {r.refined_px} | {r.broad_fraction_before:.3f} → {r.broad_fraction_after:.3f} | "
                     f"{r.thin_retention:.2f} | {r.a_priori_rule_choice} |" for r in s.itertuples())
    p4 = f"""# Phase 4 refined - stroke-only extraction (발표 노트)

**두 단계**: ① 그룹 판정(기존 RF 체크포인트, recall-priority 임계값, 변경 없음) → ② 보존 그룹 내부에서만 획 픽셀 정제(`src/pac2/refine.py`).
정제 결과는 항상 후보 픽셀의 부분집합(새 픽셀·구멍 채우기·연결 생성 없음, 코드에서 assert).

## 정제 방법 비교 (정답 없음 → 라벨 없는 지표만)
- M1 소규모 top-hat, M2 Sato ridge, M3 opening 기반 넓은 영역 제거, M4 성분별 Otsu(+M3), **M5 hysteresis(M4 핵에서 M1 안으로 성장, +M3) = 채택**
- 사전 규칙(넓은 픽셀 비율 최소 & 얇은 골격 보존 ≥ 0.80)은 글자 사이 얇은 halo를 잡지 못해 이미지마다 다른 방법을 골랐음 → **M5는 7장 전체의 육안 비교로 채택**(공개). M4는 halo를 가장 잘 지우지만 획을 끊음(보존 0.48–0.66).

| 이미지 | 보존 그룹 px(전) | 정제 후 px | 넓은(채워진) 픽셀 비율 | 얇은 골격 보존 | 사전 규칙 선택 |
|---|---|---|---|---|---|
{rows}

- 넓은 픽셀 비율 = 국소 폭 2·DT가 2.5 × 획 폭을 넘는 픽셀 비율(채워진 영역 지표). 얇은 골격 보존 = 후보의 얇은 부분 골격 중 정제 결과가 덮는 비율(가독성 지표; 후보의 얇은 비획 구조도 포함하므로 1이 이상값은 아님).
- **정확도 아님**: 정답 마스크가 없어 Dice/IoU/정밀도 미계산. GT가 생기면 같은 파이프라인(`scripts/phase4_refined.py`)으로 재평가.

## H5 Before vs After ![H5](figures/H5_extraction_before_after.png)
- example1: 구멍 옆 glow와 글자 사이 부채꼴 영역 대부분 제거(3,569 → 2,460 px), 일부 halo 남음.
- example2: 구멍 옆 밝은 덩어리 제거(2,555 → 1,233 px), 손글씨 획 유지. 단, 희미한 분필 획 상당수는 처음부터 후보에 없음.
- weld_marking_W79: 얇은 분필 획 거의 그대로(10,602 → 10,122 px, 골격 보존 0.97).
- rusted_stencil_GBO: 변화 없음(후보 355 px의 녹 조각, 스텐실 글자는 후보 단계에서 빠짐).

## 남은 실패 (H6)
1. 후보가 넓음: example1 halo 일부 잔존. 2. 희미한 분필 누락(field 2 'Angle face …'). 3. 비수기 구조 잔존(field 2 상단 기계 구조 — 그룹 판정 단계 문제, 정제로 해결 불가).
4. 짧고 단정한 표기(V8): RF는 보존하지만 정제가 굵은 마커 교차점/끝을 지워 가독성 손실. 5. field 1의 검은 마커 글씨는 밝은 극성만 후보로 잡아 누락.
"""
    (R / "phase4_refined" / "presentation_notes.md").write_text(p4, encoding="utf-8")

    pe = f"""# 발전 과정 시각화 (H1–H4, H6) - 발표 노트

- 로그: `reports/phase3/training_history.csv` (S3 로지스틱, Adam), OOF 예측, Phase 4 refined 출력. **새 학습 없음, Test 30장 미사용.**
- fold {ver['fold']} (조기 종료 없는 가장 낮은 번호, 그리기 전 고정). fold 표준화 재구성 검증: 훈련 손실 최대 오차 {ver['train_loss_max_diff']:.1e}, 검증 BCE {ver['val_bce_max_diff']:.1e}.

## H1 Feature-space evolution ![H1](H1_feature_space_evolution.png)
- 방법: fold 학습 데이터만으로 7개 표준화 특징 PCA(PC1 {ver['pca_explained_variance'][0]:.0%}, PC2 {ver['pca_explained_variance'][1]:.0%}). 배경 = 평면 위 점 z = 평균 + u·PC1 + v·PC2(평면 밖 성분은 학습 평균)에서 **7차원 모델을 정확히 계산**한 값. 점 = 투영, 빨간 원 = **전체 7차원 모델** 오분류.
- 수치: """ + "; ".join(f"epoch {r.epoch}: train {r.train_loss:.3f}, val {r.val_loss:.3f}, 오류 {r.errors_full_model_all90}/90" for r in h1.itertuples()) + """
- 해석: epoch 0은 p = 0.5가 임계값 0.4보다 커서 전부 수기로 판정(정형 15장 오류). epoch 10은 경계가 생기기 시작하지만 아직 오류 19장(임계값 근처에 많은 점) → epoch 100·599에서 정형 군집만 남기고 분리.
- 한계: 2차원 평면은 분산의 61%만 설명. 점의 실제 확률은 배경과 다를 수 있어 오류는 전체 모델 기준으로 별도 표시. PC2 = 10.7 이상치 1개는 화면 밖(표시).

## H2 Top-feature probability evolution ![H2](H2_top_feature_probability_evolution.png)
- CV_h, S_theta, R 각각을 변화시키고 나머지 6개는 학습 중앙값 고정(조건부 단면). 곡선 = 모델 시그모이드(회귀 적합 아님). CV_w는 가중치가 더 크지만 표시 대상에서 제외(요청 목록 기준, 제목에 명시).
- 해석: epoch 0 평탄(0.5) → epoch 10 완만 → 최종에서 CV_h는 급한 S자, S_theta·R은 중앙값 고정 조건에서 이미 높은 확률(다른 특징이 수기 쪽으로 끌어올림).

## H3 Optimization progress summary ![H3](H3_optimization_progress_summary.png)
- 손실 5-fold 평균 감소 + OOF BA """ + " → ".join(f"{r.stage} {r.BA:.3f}" for r in h3.itertuples()) + " + 수기 누락 " + " → ".join(f"{r.FN_missed_handwriting}" for r in h3.itertuples()) + """.
- 메시지: 최적화는 전체 판별력을 올렸지만 수기를 더 많이 제거했다.

## H4 Domain gap ![H4](H4_domain_gap.png)
- formal(렌더링 폰트) = 채도 0·매우 선명·높이 43 px, handwritten(촬영) = 채도 높고 덜 선명, field(강판 사진) = 또 다른 분포. 학습 두 집단으로 만든 모델을 세 번째 집단에 적용한다는 점을 청중에게 보여주는 그림.

## H6 Failure cases ![H6](H6_failure_cases.png)
- 너무 넓은 후보, 희미한 분필 누락, 비수기 구조 잔존, 짧은 단정 표기(가독성 손실) — 정제 후에도 남는 실패를 그대로 표시.
"""
    (R / "progress_evolution_refined" / "presentation_notes.md").write_text(pe, encoding="utf-8")

    fp = R / "final_presentation_refined"
    fp.mkdir(parents=True, exist_ok=True)
    picks = [("progress_evolution_refined/H1_feature_space_evolution.png", "모델이 7차원 특징 공간에서 분리를 학습하는 과정(PCA 평면)"),
             ("progress_evolution_refined/H2_top_feature_probability_evolution.png", "주요 특징별 확률 곡선 변화"),
             ("progress_evolution_refined/H3_optimization_progress_summary.png", "판별력 향상과 수기 누락 증가의 상충"),
             ("progress_evolution_refined/H4_domain_gap.png", "학습 데이터와 강판 사진의 도메인 차이"),
             ("phase4_refined/figures/H5_extraction_before_after.png", "획 전용 정제 전후(글자 사이 채움 영역 감소)"),
             ("progress_evolution_refined/H6_failure_cases.png", "남은 실패 사례"),
             ("progress_evolution/G17_probability_curve_evolution.png", "(기존) CV_h 확률 곡선 단면"),
             ("phase3/figures/G6_domain_bias.png", "(기존) 도메인 편향 대조 실험")]
    lines = ["# Final presentation (refined) index", "", "| 파일 | 원본 경로 | 발표 설명 | SHA-256 일치 |", "|---|---|---|---|"]
    for rel, msg in picks:
        src = R / rel
        dst = fp / src.name
        if not src.exists():
            lines.append(f"| {src.name} | results/{rel} | {msg} | MISSING |")
            continue
        if dst.exists() and sha(dst) != sha(src):
            lines.append(f"| {src.name} | results/{rel} | {msg} | CONFLICT (not overwritten) |")
            continue
        if not dst.exists():
            shutil.copy2(src, dst)
        lines.append(f"| {src.name} | results/{rel} | {msg} | {'yes' if sha(dst) == sha(src) else 'NO'} |")
    lines += ["", "모든 Phase 3 수치 = 개발 90장 5-fold OOF. Phase 4 = data/field_test 7장 + 개발 V8 1장, 정답 마스크 없음 → 정확도 미계산. Test 30장 미사용."]
    (fp / "presentation_index.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
