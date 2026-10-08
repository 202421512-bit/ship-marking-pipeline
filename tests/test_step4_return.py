"""4단계(손상 정형 → 정형 그룹 복귀) 테스트."""
import json

import pytest

from marking_pipeline.step4_return import (
    DAMAGED,
    INTACT,
    STRUCTURED,
    UNSTRUCTURED,
    Marking,
    RescueResult,
    main,
    return_to_structured,
    split_groups,
)


def sample_markings():
    return [
        Marking(id="S1", group=STRUCTURED, damage=INTACT, text="F6"),
        Marking(id="S2", group=STRUCTURED, damage=DAMAGED, text="B12"),
        Marking(id="U1", group=UNSTRUCTURED, text="F6"),   # 녹 때문에 비정형으로 잘못 감
        Marking(id="U2", group=UNSTRUCTURED, text="A3"),   # 애매함
        Marking(id="U3", group=UNSTRUCTURED, text="ㅎ"),    # 진짜 손글씨
        Marking(id="U4", group=UNSTRUCTURED),              # 3단계 결과 없음
    ]


def sample_results():
    return [
        RescueResult("U1", 0.86, ["획 두께 일정", "기준선 정렬"]),
        RescueResult("U2", 0.55, ["글자 간격 균일"]),
        RescueResult("U3", 0.12, []),
    ]


def test_high_score_returns_to_structured_damaged():
    updated, report = return_to_structured(sample_markings(), sample_results())
    u1 = next(m for m in updated if m.id == "U1")

    assert u1.group == STRUCTURED
    assert u1.damage == DAMAGED
    assert "획 두께 일정" in u1.history[-1]
    assert report.returned == ["U1"]


def test_middle_score_stays_but_needs_review():
    updated, report = return_to_structured(sample_markings(), sample_results())
    u2 = next(m for m in updated if m.id == "U2")

    assert u2.group == UNSTRUCTURED
    assert u2.needs_review is True
    assert report.needs_review == ["U2"]


def test_low_score_and_missing_result_stay_unstructured():
    updated, report = return_to_structured(sample_markings(), sample_results())
    groups = {m.id: m.group for m in updated}

    assert groups["U3"] == UNSTRUCTURED
    assert groups["U4"] == UNSTRUCTURED
    assert sorted(report.kept) == ["U3", "U4"]


def test_structured_markings_are_untouched():
    updated, _ = return_to_structured(sample_markings(), sample_results())
    before = {m.id: m for m in sample_markings()}

    for m in updated:
        if m.id.startswith("S"):
            assert m == before[m.id]


def test_input_is_not_modified():
    markings = sample_markings()
    return_to_structured(markings, sample_results())

    assert markings[2].group == UNSTRUCTURED
    assert markings[2].history == []


def test_split_groups_puts_returned_marking_in_damaged_group():
    updated, _ = return_to_structured(sample_markings(), sample_results())
    groups = split_groups(updated)

    assert [m.id for m in groups[f"{STRUCTURED}-{INTACT}"]] == ["S1"]
    assert [m.id for m in groups[f"{STRUCTURED}-{DAMAGED}"]] == ["S2", "U1"]
    assert [m.id for m in groups[UNSTRUCTURED]] == ["U2", "U3", "U4"]


def test_results_for_unknown_or_structured_ids_are_skipped():
    results = sample_results() + [RescueResult("S1", 0.9), RescueResult("없는id", 0.9)]
    updated, report = return_to_structured(sample_markings(), results)

    assert sorted(report.skipped) == ["S1", "없는id"]
    assert next(m for m in updated if m.id == "S1").damage == INTACT


def test_thresholds_can_be_changed():
    _, report = return_to_structured(
        sample_markings(), sample_results(), return_threshold=0.5, review_threshold=0.1
    )
    assert report.returned == ["U1", "U2"]
    assert report.needs_review == ["U3"]


@pytest.mark.parametrize(
    "results, kwargs",
    [
        ([RescueResult("U1", 1.5)], {}),
        ([RescueResult("U1", 0.8), RescueResult("U1", 0.9)], {}),
        ([], {"return_threshold": 0.4, "review_threshold": 0.6}),
    ],
)
def test_invalid_input_raises(results, kwargs):
    with pytest.raises(ValueError):
        return_to_structured(sample_markings(), results, **kwargs)


def test_command_line(tmp_path):
    markings_path = tmp_path / "markings.json"
    rescue_path = tmp_path / "rescue.json"
    out_path = tmp_path / "out.json"
    markings_path.write_text(
        json.dumps([m.__dict__ for m in sample_markings()], ensure_ascii=False), encoding="utf-8"
    )
    rescue_path.write_text(
        json.dumps([r.__dict__ for r in sample_results()], ensure_ascii=False), encoding="utf-8"
    )

    main(["--markings", str(markings_path), "--rescue", str(rescue_path), "--out", str(out_path)])

    output = json.loads(out_path.read_text(encoding="utf-8"))
    assert output["summary"] == {"returned": 1, "needs_review": 1, "kept": 2, "skipped": 0}
    assert [m["id"] for m in output["groups"]["정형-손상"]] == ["S2", "U1"]
