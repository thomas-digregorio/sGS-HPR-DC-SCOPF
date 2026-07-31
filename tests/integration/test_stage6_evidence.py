from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_stage_6 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_EVIDENCE,
    run_checks,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_EVIDENCE.is_file(),
    reason="preserved DGX Stage 6 evidence has not been retrieved",
)


def _checks_by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(check["name"]): check for check in result["checks"]}


def _mutated_evidence(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    evidence_path = tmp_path / "stage_6_validation.json"
    trajectory_name = evidence["evidence_files"]["trajectories_and_policy_events"]
    shutil.copy2(DEFAULT_EVIDENCE.parent / trajectory_name, tmp_path / trajectory_name)
    return evidence, evidence_path


def _write_evidence(evidence: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_preserved_stage6_evidence_passes_independent_checker() -> None:
    result = run_checks(DEFAULT_EVIDENCE, DEFAULT_CONFIG)
    by_name = _checks_by_name(result)

    assert result["stage"] == 6
    assert result["passed"] is True
    assert result["summary"] == {"passed": len(by_name)}
    assert {
        "embedded_configuration_and_input_hashes_match_versioned_sources",
        "dgx_spark_gb10_fp64_device_and_pinned_cupy_environment",
        "both_t1_and_t2_cases_have_exact_stage5_preconditioning",
        "scaled_direct_production_and_guarded_unscaled_structural_paths",
        "fp64_a1_a2_actual_alg2_selection_and_repeatable_probes",
        "csr_normal_transpose_and_explicit_transpose_gates",
        "one_ten_and_one_hundred_step_cpu_gpu_state_parity",
        "three_run_fp64_state_and_policy_determinism",
        "resident_one_thousand_step_cadence_and_cuda_timing",
        "solver_transfer_phase_direction_and_device_residency_audit",
        "original_space_eq54_kkt_physics_objective_and_sweep_invariants",
        "cpu_gpu_policy_objective_and_normalized_residual_parity",
        "all_required_timing_boundaries_have_units_and_truthful_methods",
        "fp32_is_non_gating_after_fp64_and_mixed_precision_is_disabled",
        "trajectory_gzip_integrity_and_sha256",
        "trajectory_rows_cover_all_cases_runs_and_policy_schedules",
        "stage_seven_locked_and_no_timing_or_speedup_claim",
    }.issubset(by_name)
    assert all(check["passed"] is True for check in by_name.values())


def test_checker_rejects_original_space_residual_above_frozen_threshold(tmp_path: Path) -> None:
    evidence, evidence_path = _mutated_evidence(tmp_path)
    evidence["cases"][0]["gpu_full_fp64"]["original_residuals"]["paper_normalized_norms"][
        "primal_feasibility"
    ] = 1e-4
    _write_evidence(evidence, evidence_path)

    result = run_checks(evidence_path, DEFAULT_CONFIG)
    check = _checks_by_name(result)[
        "original_space_eq54_kkt_physics_objective_and_sweep_invariants"
    ]

    assert result["passed"] is False
    assert check["passed"] is False


def test_checker_rejects_false_alg2_selection_claim(tmp_path: Path) -> None:
    evidence, evidence_path = _mutated_evidence(tmp_path)
    kernel = evidence["cases"][0]["sparse_crosschecks"]["operators"][0]["kernel_selection"]
    kernel["uses_csr_alg2"] = False
    kernel["effective_label"] = "cupyx.cusparse.spmv CUSPARSE_MV_ALG_DEFAULT"
    kernel["fallback_reason"] = "mutated test fallback"
    _write_evidence(evidence, evidence_path)

    result = run_checks(evidence_path, DEFAULT_CONFIG)
    check = _checks_by_name(result)["fp64_a1_a2_actual_alg2_selection_and_repeatable_probes"]

    assert result["passed"] is False
    assert check["passed"] is False


def test_checker_rejects_unplanned_loop_vector_transfer(tmp_path: Path) -> None:
    evidence, evidence_path = _mutated_evidence(tmp_path)
    ledger = evidence["cases"][0]["resident_timing"]["transfer_ledger"]
    ledger["records"].append(
        {
            "phase": "iteration_loop",
            "direction": "device_to_host",
            "kind": "vector",
            "calls": 1,
            "bytes": 4096,
        }
    )
    ledger["totals"]["device_to_host"]["calls"] += 1
    ledger["totals"]["device_to_host"]["bytes"] += 4096
    _write_evidence(evidence, evidence_path)

    result = run_checks(evidence_path, DEFAULT_CONFIG)
    check = _checks_by_name(result)["solver_transfer_phase_direction_and_device_residency_audit"]

    assert result["passed"] is False
    assert check["passed"] is False


def test_checker_rejects_unlocked_stage_seven_claim(tmp_path: Path) -> None:
    evidence, evidence_path = _mutated_evidence(tmp_path)
    evidence["stage_boundary"]["stage_7_benchmarks_locked"] = False
    _write_evidence(evidence, evidence_path)

    result = run_checks(evidence_path, DEFAULT_CONFIG)
    check = _checks_by_name(result)["stage_seven_locked_and_no_timing_or_speedup_claim"]

    assert result["passed"] is False
    assert check["passed"] is False
