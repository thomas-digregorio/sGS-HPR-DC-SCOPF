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
        "taxonomy": "preregistered project-specific A--E decision framework",
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
    assert float(t16["projected_unified_gib"]) < float(t16["nominal_80pct_gib"])
    assert t16["block_reasons"] == (
        "failed:within_host_safety_budget;failed:within_device_safety_budget"
    )

    for key in ("case9241pegase:T24", "case9241pegase:T32"):
        row = by_key[key]
        assert row["status"] == "INDEX_BLOCKED"
        assert int(row["planning_nnz"]) > generator.INT32_MAX
        assert row["full_lp_allocated"] == "False"
        assert row["block_reasons"] == "failed:signed_int32_csr_nnz_limit"

    assert int(by_key["case9241pegase:T24"]["exact_reconstructed_nnz"]) < generator.INT32_MAX
    assert int(by_key["case9241pegase:T32"]["exact_reconstructed_nnz"]) > generator.INT32_MAX


def test_stage9_benchmarks_retain_timing_dispersion_and_censoring() -> None:
    rows = _load_csv(generator.TABLE_ROOT / "stage_9_benchmarks.csv")
    allocated = [row for row in rows if row["full_lp_allocated"] == "True"]

    assert all(row["highs_minimum_seconds"] and row["highs_maximum_seconds"] for row in allocated)
    assert all(row["gpu_measured_repetitions"] in {"5", "9"} for row in allocated)
    t6 = next(row for row in allocated if row["case_key"] == "case9241pegase:T6")
    assert t6["cpu_median_seconds"] == ""
    assert abs(float(t6["cpu_timeout_seconds"]) - 3600.092738645966) < 1e-9


def test_stage9_independent_checker_passes_every_gate() -> None:
    result = check_stage_9.run_checks()

    assert result["all_passed"] is True
    assert result["checker_status"] == "PASS"
    assert result["summary"] == {"passed": 17, "total": 17}
    assert all(check["passed"] for check in result["checks"])
