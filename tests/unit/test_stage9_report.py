from __future__ import annotations

import csv
import json

from scripts import check_stage_9
from scripts import generate_stage_9_artifacts as generator


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_stage9_index_records_the_evidence_bounded_classification() -> None:
    index = _load_json(generator.INDEX_PATH)

    assert index["final_classification"] == {
        "code": "D",
        "decision_rule": "docs/stage_9_contract.md",
        "exact_reproduction": False,
        "label": "structural reproduction",
        "local_speedup_claimed": False,
        "paper_timing_reproduced": False,
    }
    assert index["coverage"] == {
        "allocated_benchmark_rows": 11,
        "dimension_matches": 18,
        "nnz_matches": 0,
        "resource_resolved_without_allocation": 3,
        "symbolic_table_ii_rows": 18,
        "validated_cpu_rows": 10,
        "validated_gpu_rows": 11,
    }
    assert index["stage_10"] == {
        "n_minus_1_work_performed": False,
        "status": "LOCKED",
    }


def test_stage9_resource_rows_preserve_the_fail_closed_boundaries() -> None:
    rows = _load_csv(generator.TABLE_ROOT / "stage_9_resource_boundaries.csv")
    by_key = {row["case_key"]: row for row in rows}

    t16 = by_key["case9241pegase:T16"]
    assert t16["status"] == "MEMORY_BLOCKED"
    assert float(t16["projected_unified_gib"]) > float(t16["host_budget_gib"])
    assert float(t16["projected_unified_gib"]) > float(t16["device_budget_gib"])

    for key in ("case9241pegase:T24", "case9241pegase:T32"):
        row = by_key[key]
        assert row["status"] == "INDEX_BLOCKED"
        assert int(row["planning_nnz"]) > generator.INT32_MAX
        assert row["full_lp_allocated"] == "False"


def test_stage9_independent_checker_passes_every_gate() -> None:
    result = check_stage_9.run_checks()

    assert result["all_passed"] is True
    assert result["checker_status"] == "PASS"
    assert result["summary"] == {"passed": 17, "total": 17}
    assert all(check["passed"] for check in result["checks"])
