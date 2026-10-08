"""4단계: 비정형 그룹으로 흘러들어온 손상 정형 문자를 정형 그룹으로 돌려보낸다.

전체 흐름
  1단계  정형 / 비정형 분류
  2단계  정형 그룹을 손상 / 비손상으로 구분
  3단계  비정형 그룹에 여러 필터를 적용해 "사실은 심하게 손상된 정형"일 가능성(score)을 계산
  4단계  (이 파일) 3단계 score를 보고 비정형 → 정형(손상)으로 되돌린다

3단계 score 기준
  score >= return_threshold   → 정형(손상)으로 복귀
  review_threshold <= score   → 비정형에 남기되 사람 검토 대상으로 표시
  그 외                        → 비정형 유지

사용 예 (명령줄)
  python -m marking_pipeline.step4_return \
      --markings markings.json --rescue step3_results.json --out regrouped.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

STRUCTURED = "정형"
UNSTRUCTURED = "비정형"
DAMAGED = "손상"
INTACT = "비손상"

DEFAULT_RETURN_THRESHOLD = 0.7
DEFAULT_REVIEW_THRESHOLD = 0.5


@dataclass
class Marking:
    """마킹 하나. 1·2단계를 거친 상태로 들어온다."""

    id: str
    group: str                     # 정형 / 비정형 (1단계 결과)
    damage: str | None = None      # 손상 / 비손상 (2단계 결과, 정형만)
    text: str | None = None        # 인식된 문자열 (있으면)
    image_path: str | None = None  # 잘라낸 마킹 이미지 경로 (있으면)
    needs_review: bool = False     # 사람이 확인해야 하는지
    history: list[str] = field(default_factory=list)  # 단계별 처리 기록


@dataclass
class RescueResult:
    """3단계 결과: 비정형 마킹 하나가 '손상된 정형'일 가능성."""

    marking_id: str
    score: float                   # 0.0 ~ 1.0, 높을수록 정형일 가능성이 큼
    reasons: list[str] = field(default_factory=list)  # 판단 근거 (예: "획 두께 일정")


@dataclass
class ReturnReport:
    """4단계에서 무엇이 어디로 갔는지 요약."""

    returned: list[str] = field(default_factory=list)      # 정형(손상)으로 복귀
    needs_review: list[str] = field(default_factory=list)  # 비정형 유지 + 검토 필요
    kept: list[str] = field(default_factory=list)          # 비정형 유지
    skipped: list[str] = field(default_factory=list)       # 3단계 결과가 잘못 들어온 경우


def return_to_structured(
    markings: list[Marking],
    rescue_results: list[RescueResult],
    return_threshold: float = DEFAULT_RETURN_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> tuple[list[Marking], ReturnReport]:
    """3단계 score를 보고 비정형 마킹을 정형(손상) 그룹으로 되돌린다.

    입력 리스트는 바꾸지 않고, 바뀐 복사본과 요약 보고서를 돌려준다.
    """
    if not 0.0 <= review_threshold <= return_threshold <= 1.0:
        raise ValueError(
            "0 <= review_threshold <= return_threshold <= 1 이어야 합니다 "
            f"(review={review_threshold}, return={return_threshold})"
        )

    by_id = {m.id: m for m in markings}
    if len(by_id) != len(markings):
        raise ValueError("마킹 id가 중복되었습니다.")

    results: dict[str, RescueResult] = {}
    for r in rescue_results:
        if r.marking_id in results:
            raise ValueError(f"3단계 결과에 같은 id가 두 번 있습니다: {r.marking_id}")
        if not 0.0 <= r.score <= 1.0:
            raise ValueError(f"score는 0~1 사이여야 합니다: {r.marking_id}={r.score}")
        results[r.marking_id] = r

    report = ReturnReport()
    # 존재하지 않거나 이미 정형인 마킹에 대한 결과는 무시하고 기록만 남긴다.
    for marking_id in results:
        target = by_id.get(marking_id)
        if target is None or target.group != UNSTRUCTURED:
            report.skipped.append(marking_id)

    updated: list[Marking] = []
    for m in markings:
        m = replace(m, history=list(m.history))
        result = results.get(m.id)

        if m.group != UNSTRUCTURED:
            updated.append(m)
            continue

        if result is None:
            report.kept.append(m.id)
        elif result.score >= return_threshold:
            # 심한 손상 때문에 비정형으로 잘못 분류된 정형 → 정형(손상)으로 복귀
            m.group = STRUCTURED
            m.damage = DAMAGED
            m.needs_review = False
            m.history.append(_note("비정형 → 정형(손상) 복귀", result))
            report.returned.append(m.id)
        elif result.score >= review_threshold:
            m.needs_review = True
            m.history.append(_note("비정형 유지, 검토 필요", result))
            report.needs_review.append(m.id)
        else:
            m.history.append(_note("비정형 유지", result))
            report.kept.append(m.id)
        updated.append(m)

    return updated, report


def split_groups(markings: list[Marking]) -> dict[str, list[Marking]]:
    """정형-비손상 / 정형-손상 / 비정형 세 묶음으로 나눈다."""
    groups: dict[str, list[Marking]] = {
        f"{STRUCTURED}-{INTACT}": [],
        f"{STRUCTURED}-{DAMAGED}": [],
        UNSTRUCTURED: [],
    }
    for m in markings:
        if m.group == UNSTRUCTURED:
            groups[UNSTRUCTURED].append(m)
        elif m.damage == DAMAGED:
            groups[f"{STRUCTURED}-{DAMAGED}"].append(m)
        else:
            groups[f"{STRUCTURED}-{INTACT}"].append(m)
    return groups


def _note(action: str, result: RescueResult) -> str:
    reasons = ", ".join(result.reasons) if result.reasons else "근거 없음"
    return f"4단계: {action} (score={result.score:.2f}, 근거: {reasons})"


# ---------- JSON 입출력 ----------

def load_markings(path: str | Path) -> list[Marking]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Marking(**item) for item in data]


def load_rescue_results(path: str | Path) -> list[RescueResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RescueResult(**item) for item in data]


def save_output(path: str | Path, markings: list[Marking], report: ReturnReport) -> None:
    output = {
        "summary": {
            "returned": len(report.returned),
            "needs_review": len(report.needs_review),
            "kept": len(report.kept),
            "skipped": len(report.skipped),
        },
        "report": asdict(report),
        "groups": {
            name: [asdict(m) for m in items]
            for name, items in split_groups(markings).items()
        },
    }
    Path(path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="4단계: 손상 정형을 정형 그룹으로 복귀")
    parser.add_argument("--markings", required=True, help="1·2단계를 거친 마킹 목록 JSON")
    parser.add_argument("--rescue", required=True, help="3단계 결과 JSON")
    parser.add_argument("--out", required=True, help="결과를 저장할 JSON 경로")
    parser.add_argument("--return-threshold", type=float, default=DEFAULT_RETURN_THRESHOLD)
    parser.add_argument("--review-threshold", type=float, default=DEFAULT_REVIEW_THRESHOLD)
    args = parser.parse_args(argv)

    markings, report = return_to_structured(
        load_markings(args.markings),
        load_rescue_results(args.rescue),
        return_threshold=args.return_threshold,
        review_threshold=args.review_threshold,
    )
    save_output(args.out, markings, report)
    print(
        f"복귀 {len(report.returned)}개 / 검토 필요 {len(report.needs_review)}개 / "
        f"비정형 유지 {len(report.kept)}개 / 무시 {len(report.skipped)}개 → {args.out}"
    )


if __name__ == "__main__":
    main()
