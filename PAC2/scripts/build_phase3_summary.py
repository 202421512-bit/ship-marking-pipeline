"""Build results/phase3/phase3_summary.md: every reported number is re-read from the copied CSVs and compared with
the previously reported value (a mismatch is shown, never silently corrected). Also writes results/README.md."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "phase3" / "tables"


def main() -> int:
    mc = pd.read_csv(T / "model_comparison.csv").set_index("model")
    g7 = pd.read_csv(T / "G7_confusion_matrix.csv").set_index("model")
    g1 = pd.read_csv(T / "G1_loss_convergence.csv")
    cs = pd.read_csv(T / "coefficient_stability.csv")
    cs = cs[cs.model == "C_shape_domain_matched|train_all|torch_logistic"].set_index("feature")
    ba = lambda m: float(mc.loc[m, "oof_balanced_accuracy"])  # noqa: E731
    checks = [
        ("S1 BA", 0.787, ba("S1_basic_unweighted"), 0.0005),
        ("S3 BA", 0.927, ba("C_shape_domain_matched|train_all|torch_logistic"), 0.0005),
        ("S4 BA", 0.960, ba("S4_hard_example_reweighted"), 0.0005),
        ("Random Forest BA (C features)", 0.980, ba("C_shape_domain_matched|train_all|random_forest"), 0.0005),
        ("Domain-only control BA", 1.000, ba("B_domain_only|train_all|torch_logistic"), 0.0005),
        ("S1 handwritten missed (FN)", 2, int(g7.loc["S1_basic_unweighted", "fn"]), 0),
        ("S4 handwritten missed (FN)", 6, int(g7.loc["S4_hard_example_reweighted", "fn"]), 0),
        ("Train loss start", 0.693, float(g1.train_loss_mean.iloc[0]), 0.0005),
        ("Train loss end", 0.235, float(g1.train_loss_mean.iloc[-1]), 0.0005),
        ("Val loss start", 0.693, float(g1.val_loss_mean.iloc[0]), 0.0005),
        ("Val loss end", 0.271, float(g1.val_loss_mean.iloc[-1]), 0.0005),
    ]
    for f, rep in (("CV_h", 3.44), ("CV_w", 1.85), ("S_theta", 1.00), ("R", 0.87), ("CV_A", 0.68), ("B", 0.57), ("CV_g", -0.31)):
        checks.append((f"weight {f} (mean)", rep, float(cs.loc[f, "mean"]), 0.005))
    rows = []
    for name, rep, csv_v, tol in checks:
        ok = abs(float(rep) - float(csv_v)) <= tol
        rows.append(f"| {name} | {rep} | {csv_v:.4f} | {'일치' if ok else '**불일치**'} |")
    n_bad = sum("불일치" in r for r in rows)
    w = lambda f: f"{cs.loc[f, 'mean']:+.2f} ± {cs.loc[f, 'std']:.2f} ({'부호 일관' if cs.loc[f, 'sign_consistent'] else '부호 불안정'})"  # noqa: E731
    md = f"""# PAC2 Phase 3 요약 (발표용)

## 데이터
- formal 20장(label 0, 컴퓨터 렌더링 폰트), handwritten 100장(label 1, 촬영된 수기)
- 개발 90장(handwritten 75 / formal 15) 5-fold 교차검증, **Test 30장(25/5)은 미사용**
- 모든 수치 = 개발 데이터 out-of-fold(OOF) 결과

## 보고값 ↔ 원본 CSV 대조 (`results/phase3/tables/`)
| 항목 | 기존 보고값 | CSV 값 | 판정 |
|---|---|---|---|
{chr(10).join(rows)}

불일치 {n_bad}건.

## 주요 성능 (OOF balanced accuracy)
- S1 기본(무가중) {ba('S1_basic_unweighted'):.3f} → S3 정규화·임계값 최적화 {ba('C_shape_domain_matched|train_all|torch_logistic'):.3f} → S4 hard-example reweighting {ba('S4_hard_example_reweighted'):.3f}
- Random Forest {ba('C_shape_domain_matched|train_all|random_forest'):.3f}
- 촬영 도메인 특징만 사용한 대조 모델 {ba('B_domain_only|train_all|torch_logistic'):.3f} (배포 불가, 편향 점검용)

## S4의 상충관계 (중요)
- 수기 누락(수기→정형): S1 {int(g7.loc['S1_basic_unweighted', 'fn'])}장 → S4 {int(g7.loc['S4_hard_example_reweighted', 'fn'])}장
- 정형 오판(정형→수기): S1 {int(g7.loc['S1_basic_unweighted', 'fp'])}장 → S4 {int(g7.loc['S4_hard_example_reweighted', 'fp'])}장
- BA 향상은 소수 클래스(formal) 오류 감소에서 왔고, 그 대가로 수기 보존이 줄었다. 추출 단계에서 수기 누락 = 수기 획 손실.

## 7개 특징 가중치 (S3, 학습 fold z 공간, 5-fold 평균 ± SD)
- CV_h {w('CV_h')}, CV_w {w('CV_w')}, S_theta {w('S_theta')}, R {w('R')}, CV_A {w('CV_A')}, B {w('B')}, CV_g {w('CV_g')}
- 계수 크기 ≠ 인과적 중요도. formal_ref 표준화 계수와는 척도가 달라 비교하지 않음.

## Loss 수렴 (S3, Adam, 5-fold 평균)
- Train {g1.train_loss_mean.iloc[0]:.3f} → {g1.train_loss_mean.iloc[-1]:.3f}, Validation {g1.val_loss_mean.iloc[0]:.3f} → {g1.val_loss_mean.iloc[-1]:.3f}
- 조기 종료된 fold(epoch 208, 414)는 이후 epoch에 **마지막 값을 carry forward**하여 평균함 → 그 구간은 해당 fold의 새 관측값이 아님.

## 발표 시 주의
- G8은 실제 강판 추출 성능이 아니라 **개발 데이터 crop의 이미지 단위 분류 결과**(OOF 판정으로 전체 획 보존/제거)다.
- 현재 성능은 '손글씨 vs 컴퓨터 렌더링 폰트' 형상 분류 성능이며, **실제 강판의 녹·스크래치 제거 정확도가 아니다**.
- formal 15장 → formal 1장 = BA 약 3.3 pp. S3→S4 차이는 1장.
"""
    (ROOT / "results" / "phase3" / "phase3_summary.md").write_text(md, encoding="utf-8")
    readme = """# PAC2 results (발표용 자료)

- `phase3/` : G1–G9 그림(PNG 300 dpi, SVG), 그래프 원본 CSV·요약표(`tables/`), 발표 노트, 요약(`phase3_summary.md`)
- `phase4/` : G10–G15 그림, 추출 예시(`extraction_examples/`), 표(`tables/`), 발표 노트
- `final_presentation/` : 발표에서 직접 쓰는 핵심 그림 + `presentation_index.md`
- `manifest.csv` : 원본 경로·복사 경로·SHA-256·동기화 시각

원본 로그(대형 `training_history.csv` 등)와 이미지별 추론 결과는 `reports/`에 남아 있다.
갱신: `.venv\\Scripts\\python.exe scripts\\sync_presentation_results.py` (같은 파일은 건너뛰고, 내용이 다르면 덮어쓰지 않고 보고).
Test 30장은 아직 사용하지 않았다.
"""
    (ROOT / "results" / "README.md").write_text(readme, encoding="utf-8")
    print(f"phase3_summary.md written; mismatches: {n_bad}")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
