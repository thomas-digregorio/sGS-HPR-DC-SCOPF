"""Independently validate the Stage 8 HiGHS/GPU-only continuation evidence."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts import check_stage_7 as stage7_checker  # noqa: E402
from scripts import check_stage_8 as stage8_checker  # noqa: E402
from scripts import run_stage_7 as stage7  # noqa: E402
from scripts import run_stage_8 as stage8  # noqa: E402
from scripts import run_stage_8_gpu_only_completion as runner  # noqa: E402

DEFAULT_CONFIG = runner.DEFAULT_CONFIG
DEFAULT_EVIDENCE = runner.DEFAULT_OUTPUT / runner.FINAL_NAME
DEFAULT_OUTPUT = runner.DEFAULT_OUTPUT / "stage_8_gpu_only_completion_checks.json"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _load(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        return runner._load_json(path), None
    except Exception as error:
        return {}, f"{type(error).__name__}: {error}"


def _add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _skip_policy_valid(case: Mapping[str, Any], *, resource_blocked: bool) -> bool:
    tracks = _mapping(case.get("solver_tracks"))
    skipped = _mapping(case.get("skipped_solver_tracks"))
    if any(name in tracks for name in runner.SKIPPED_TRACKS):
        return False
    if set(skipped) != set(runner.SKIPPED_TRACKS):
        return False
    for name in runner.SKIPPED_TRACKS:
        row = _mapping(skipped.get(name))
        if not (
            row.get("name") == name
            and row.get("status") == "SKIPPED_BY_USER_SCOPE"
            and row.get("gating") is False
            and row.get("passed") is None
            and row.get("executed") is False
        ):
            return False
        reason = str(row.get("reason", ""))
        if resource_blocked and "resource guard" not in reason:
            return False
        if not resource_blocked and "explicit user scope" not in reason:
            return False
    return True


def _resource_ledger_valid(
    evidence: Mapping[str, Any],
    original_evidence: Mapping[str, Any],
    base_stage7_evidence: Mapping[str, Any],
) -> bool:
    try:
        expected = runner._resource_ledger(original_evidence, base_stage7_evidence)
    except Exception:
        return False
    actual = _sequence(evidence.get("resource_ledger"))
    if len(actual) != len(expected) or [row.get("key") for row in actual] != list(
        runner.REQUESTED_KEYS
    ):
        return False
    for actual_row, expected_row in zip(actual, expected, strict=True):
        candidate = dict(actual_row)
        candidate["allocation_permitted_this_invocation"] = False
        reference = dict(expected_row)
        reference["allocation_permitted_this_invocation"] = False
        if candidate != reference:
            return False
    return True


def _static_cases_valid(evidence: Mapping[str, Any]) -> bool:
    cases = {
        str(row.get("key")): _mapping(row) for row in _sequence(evidence.get("cases"))
    }
    history = _sequence(evidence.get("allocation_history"))
    for key in runner.STATIC_BLOCK_KEYS:
        case = cases.get(key, {})
        gate = _mapping(case.get("stage8_resource_gate"))
        estimate = _mapping(case.get("resource_estimate"))
        failure = _mapping(case.get("failure"))
        dispositions = _mapping(case.get("required_solver_track_disposition"))
        if not (
            case.get("status") == "INDEX_BLOCKED"
            and case.get("passed") is False
            and case.get("resolved") is True
            and case.get("full_lp_allocation_attempted") is False
            and estimate.get("signed_int32_csr_supported") is False
            and int(estimate.get("conservative_planning_nnz", 0)) > 2_147_483_647
            and gate.get("passed") is False
            and gate.get("evaluation_kind") == "static_signed_int32_csr_guard"
            and gate.get("block_reasons") == ["signed_int32_csr_nnz_limit"]
            and gate.get("evaluated_before_full_lp_allocation") is True
            and failure.get("type") == "SparseIndexSafetyBlock"
            and failure.get("full_lp_allocated") is False
            and not _mapping(case.get("solver_tracks"))
            and dispositions
            == {name: "NOT_RUN_STATIC_RESOURCE_BLOCK" for name in runner.REQUIRED_TRACKS}
            and _skip_policy_valid(case, resource_blocked=True)
            and all(_mapping(row).get("key") != key for row in history)
        ):
            return False
    return True


def _resource_gate_valid(case: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    gate = _mapping(case.get("stage8_resource_gate"))
    estimate = _mapping(case.get("resource_estimate"))
    observation = _mapping(gate.get("observation"))
    if not gate or not estimate or not observation:
        return False
    expected = stage8._resource_gate(estimate, observation, config)
    return gate == expected


def _preprocessing_valid(case: Mapping[str, Any]) -> bool:
    preprocessing = _mapping(case.get("preprocessing"))
    scaled = _mapping(preprocessing.get("scaled_equality"))
    spectral = _mapping(preprocessing.get("sparse_spectral_certificate"))
    dimensions = _mapping(_mapping(case.get("construction")).get("dimensions"))
    storage = 109
    periods = 16
    m1 = periods + storage
    return (
        _finite_nonnegative(preprocessing.get("wall_seconds"))
        and preprocessing.get("cpu_hpr_workspace_prepared") is False
        and preprocessing.get("cpu_hpr_solver_called") is False
        and scaled.get("periods") == periods
        and scaled.get("storage_count") == storage
        and scaled.get("equality_rows") == m1
        and scaled.get("dense_equality_gram_materialized") is False
        and spectral.get("rows") == dimensions.get("m2")
        and spectral.get("columns") == dimensions.get("n")
        and spectral.get("power_seed") == 20260803
        and spectral.get("finite_certificate") is True
        and spectral.get("dense_matrix_materialized") is False
        and spectral.get("normal_matrix_materialized") is False
        and isinstance(spectral.get("lambda_used"), (int, float))
        and float(spectral["lambda_used"]) > 0.0
    )


def _successful_t16_valid(
    case: Mapping[str, Any], config: Mapping[str, Any], base_config: Mapping[str, Any]
) -> bool:
    if not _resource_gate_valid(case, config):
        return False
    gate = _mapping(case.get("stage8_resource_gate"))
    construction = _mapping(case.get("construction"))
    dimensions = _mapping(construction.get("dimensions"))
    reconciliation = _mapping(case.get("structural_reconciliation"))
    tracks = _mapping(case.get("solver_tracks"))
    highs = _mapping(tracks.get("highs"))
    gpu = _mapping(tracks.get("gpu_fp64_sgs_hpr"))
    reference = _mapping(_mapping(highs.get("correctness")).get("candidate")).get(
        "objective"
    )
    timing = _mapping(case.get("timing_boundaries"))
    expected_m, expected_n, paper_nnz, expected_nnz, _ = stage7_checker.EXPECTED_ROWS[
        runner.REQUESTED_KEYS[0]
    ]
    structure_valid = (
        case.get("status") == "PASS"
        and case.get("passed") is True
        and case.get("resolved") is True
        and case.get("full_lp_allocation_attempted") is True
        and gate.get("passed") is True
        and case.get("case_name") == "case9241pegase"
        and case.get("periods") == 16
        and dimensions.get("periods") == 16
        and dimensions.get("m") == expected_m
        and dimensions.get("n") == expected_n
        and dimensions.get("nnz_A") == expected_nnz
        and reconciliation.get("dimension_match") is True
        and reconciliation.get("published_nnz") == paper_nnz
        and reconciliation.get("actual_nnz") == expected_nnz
        and reconciliation.get("symbolic_reconstructed_nnz") == expected_nnz
        and reconciliation.get("actual_matches_symbolic_nnz") is True
        and reconciliation.get("paper_time_comparable") is False
        and re.fullmatch(
            r"[0-9a-f]{64}", str(_mapping(construction.get("lp_fingerprint")).get("sha256", ""))
        )
        is not None
        and _preprocessing_valid(case)
    )
    tracks_valid = (
        set(tracks) == set(runner.REQUIRED_TRACKS)
        and isinstance(reference, (int, float))
        and math.isfinite(float(reference))
        and stage7_checker._track_valid(
            highs, track_name="highs", reference_objective=None
        )
        and stage7_checker._track_valid(
            gpu,
            track_name="gpu_fp64_sgs_hpr",
            reference_objective=float(reference),
        )
        and _mapping(gpu.get("kernel_checks")).get("passed") is True
        and _mapping(gpu.get("kernel_checks")).get("FP64") is True
        and all(
            stage7_checker._kernel_selection_valid(
                _mapping(gpu.get("kernel_selection")).get(name)
            )
            for name in ("A1", "A2")
        )
        and stage7_checker._gpu_memory_report_valid(gpu.get("memory_before"))
        and stage7_checker._gpu_memory_report_valid(gpu.get("memory_after"))
        and stage7_checker._transfer_ledger_valid(
            gpu.get("preparation_transfer_delta"), require_solver_phases=False
        )
        and stage7_checker._transfer_ledger_valid(
            gpu.get("cumulative_transfer_ledger"), require_solver_phases=False
        )
    )
    timing_valid = (
        all(
            _finite_nonnegative(timing.get(name))
            for name in (
                "model_construction_wall_seconds",
                "preprocessing_wall_seconds",
                "gpu_workspace_setup_wall_seconds",
                "end_to_end_case_wall_seconds",
            )
        )
        and timing.get("cpu_hpr_workspace_setup_wall_seconds") is None
        and timing.get("solver_core_samples_are_stored_per_track") is True
        and timing.get("speedup_computed") is False
        and base_config.get("timing", {}).get("measured_runs") == 5
    )
    return structure_valid and tracks_valid and timing_valid and _skip_policy_valid(
        case, resource_blocked=False
    )


def _t16_valid(
    evidence: Mapping[str, Any], config: Mapping[str, Any], base_config: Mapping[str, Any]
) -> tuple[bool, str]:
    case = next(
        (
            _mapping(row)
            for row in _sequence(evidence.get("cases"))
            if _mapping(row).get("key") == runner.REQUESTED_KEYS[0]
        ),
        {},
    )
    status = case.get("status")
    gate = _mapping(case.get("stage8_resource_gate"))
    if status == "MEMORY_BLOCKED":
        dispositions = _mapping(case.get("required_solver_track_disposition"))
        valid = (
            _resource_gate_valid(case, config)
            and gate.get("passed") is False
            and case.get("passed") is False
            and case.get("resolved") is True
            and case.get("full_lp_allocation_attempted") is False
            and _mapping(case.get("failure")).get("type") == "MemorySafetyBlock"
            and _mapping(case.get("failure")).get("full_lp_allocated") is False
            and not _mapping(case.get("solver_tracks"))
            and dispositions
            == {name: "NOT_RUN_MEMORY_SAFETY_BLOCK" for name in runner.REQUIRED_TRACKS}
            and _skip_policy_valid(case, resource_blocked=True)
        )
        return valid, f"status={status}, gate_passed={gate.get('passed')}"
    if status == "PASS":
        return _successful_t16_valid(case, config, base_config), "status=PASS"
    if status in {"FAIL", "TIME_LIMIT"}:
        tracks = _mapping(case.get("solver_tracks"))
        valid = (
            _resource_gate_valid(case, config)
            and gate.get("passed") is True
            and case.get("passed") is False
            and case.get("resolved") is True
            and case.get("full_lp_allocation_attempted") is True
            and set(tracks).issubset(set(runner.REQUIRED_TRACKS))
            and bool(_mapping(case.get("failure")))
            and _skip_policy_valid(case, resource_blocked=False)
        )
        return valid, f"status={status}, tracks={sorted(tracks)}"
    return False, f"nonterminal_or_unknown_status={status}"


def _allocation_valid(evidence: Mapping[str, Any]) -> tuple[bool, str]:
    history = [_mapping(row) for row in _sequence(evidence.get("allocation_history"))]
    invocations = [_mapping(row) for row in _sequence(evidence.get("invocations"))]
    boundary = _mapping(evidence.get("stage_boundary"))
    allocated = [str(row.get("key")) for row in history]
    valid = (
        len(history) <= 1
        and set(allocated).issubset({runner.REQUESTED_KEYS[0]})
        and boundary.get("continuation_allocation_attempt_count") == len(history)
        and boundary.get("unique_allocated_keys") == list(dict.fromkeys(allocated))
        and all(len(_sequence(row.get("allocated_keys"))) <= 1 for row in invocations)
        and all(
            set(str(key) for key in _sequence(row.get("allocated_keys"))).issubset(
                {runner.REQUESTED_KEYS[0]}
            )
            for row in invocations
        )
        and all(row.get("cpu_hpr_called") is False for row in invocations)
        and all(row.get("gurobi_called") is False for row in invocations)
    )
    return valid, f"allocation_history={allocated}, invocations={len(invocations)}"


def _terminal_valid(evidence: Mapping[str, Any]) -> tuple[bool, str]:
    cases = {
        str(row.get("key")): _mapping(row) for row in _sequence(evidence.get("cases"))
    }
    t16_status = cases.get(runner.REQUESTED_KEYS[0], {}).get("status")
    expected = {
        "PASS": "COMPLETE_WITH_STATIC_RESOURCE_LIMITS",
        "MEMORY_BLOCKED": "COMPLETE_WITH_RESOURCE_LIMITS",
        "FAIL": "STOPPED_ON_FAILURE",
        "TIME_LIMIT": "STOPPED_ON_FAILURE",
    }.get(str(t16_status))
    boundary = _mapping(evidence.get("stage_boundary"))
    valid = (
        expected is not None
        and evidence.get("status") == expected
        and evidence.get("all_passed") is False
        and evidence.get("all_requested_rows_resolved") is True
        and evidence.get("executable_scope_passed") is (t16_status == "PASS")
        and boundary.get("stage_8_gpu_only_continuation_complete") is True
        and isinstance(evidence.get("completed_utc"), str)
    )
    return valid, f"status={evidence.get('status')}, t16={t16_status}"


def run_checks(
    evidence_path: Path = DEFAULT_EVIDENCE,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    evidence, evidence_error = _load(evidence_path)
    config, config_error = _load(config_path)
    _add(
        checks,
        "evidence_and_configuration_load",
        evidence_error is None and config_error is None,
        f"evidence={evidence_error or 'loaded'}, config={config_error or 'loaded'}",
    )
    config_identity = stage8._portable_identity(config_path)
    config_valid = (
        config_error is None
        and config_identity.get("passed") is True
        and config_identity.get("canonical_git_blob_sha256") == runner.FROZEN_CONFIG_SHA256
        and runner._validate_config(config) == []
        and evidence.get("configuration") == config
        and _mapping(evidence.get("configuration_validation")).get("passed") is True
    )
    _add(
        checks,
        "frozen_gpu_only_continuation_configuration",
        config_valid,
        f"canonical_sha256={config_identity.get('canonical_git_blob_sha256')}",
    )

    try:
        original_contract, original = runner._original_contract(config)
        original_valid = (
            original_contract.get("passed") is True
            and evidence.get("original_stage_8_contract") == original_contract
        )
    except Exception as error:
        original_contract, original, original_valid = {}, {}, False
        original_error = f"{type(error).__name__}: {error}"
    else:
        original_error = "none"
    _add(
        checks,
        "original_stage8_terminal_evidence_and_cpu_failure_preserved",
        original_valid,
        f"status={original_contract.get('terminal_status')}, error={original_error}",
    )

    try:
        base_contract, base = stage8._base_contract(original["configuration"])
        base_valid = (
            base_contract.get("passed") is True
            and evidence.get("base_stage_7_contract") == base_contract
            and _mapping(evidence.get("base_stage_7_provenance")).get("passed") is True
            and _mapping(evidence.get("policy_contract")).get("passed") is True
        )
    except Exception:
        base_contract, base, base_valid = {}, {"configuration": {}, "evidence": {}}, False
    _add(
        checks,
        "frozen_stage7_reconstruction_and_acceptance_contract_preserved",
        base_valid,
        f"hashes={base_contract.get('sha256')}",
    )
    _add(
        checks,
        "executed_source_identity",
        stage8_checker._source_identity_valid(evidence),
        f"manifest_rows={len(_sequence(evidence.get('source_manifest')))}",
    )

    ledger_valid = _resource_ledger_valid(
        evidence,
        _mapping(original.get("evidence")),
        _mapping(base.get("evidence")),
    )
    _add(
        checks,
        "sequence_6_8_resource_projections_match_frozen_stage8",
        ledger_valid,
        f"rows={len(_sequence(evidence.get('resource_ledger')))}",
    )
    _add(
        checks,
        "sequence_7_8_signed_int32_blocks_resolved_without_allocation",
        _static_cases_valid(evidence),
        f"blocked={sorted(runner.STATIC_BLOCK_KEYS)}",
    )

    t16_valid, t16_detail = _t16_valid(evidence, config, _mapping(base.get("configuration")))
    _add(
        checks,
        "sequence_6_gate_or_highs_gpu_evidence_is_honest",
        t16_valid,
        t16_detail,
    )
    allocation_valid, allocation_detail = _allocation_valid(evidence)
    _add(
        checks,
        "at_most_one_sequence_6_allocation_and_no_cpu_or_gurobi_calls",
        allocation_valid,
        allocation_detail,
    )
    terminal_valid, terminal_detail = _terminal_valid(evidence)
    _add(
        checks,
        "all_requested_rows_reached_an_honest_terminal_protocol_state",
        terminal_valid,
        terminal_detail,
    )

    cases = [_mapping(row) for row in _sequence(evidence.get("cases"))]
    no_speedup = all(
        _mapping(case.get("timing_boundaries")).get("speedup_computed") is False
        for case in cases
        if case.get("status") == "PASS"
    )
    _add(
        checks,
        "no_unsupported_speedup_computed",
        no_speedup,
        "HiGHS and GPU retain separate timing boundaries",
    )
    boundary = _mapping(evidence.get("stage_boundary"))
    boundary_valid = (
        boundary.get("stage_8_only") is True
        and boundary.get("stage_9_locked") is True
        and boundary.get("stage_9_allocation_count") == 0
        and boundary.get("n_minus_1_extension_enabled") is False
        and boundary.get("exact_paper_reproduction_claimed") is False
        and boundary.get("paper_a100_timing_reproduction_claimed") is False
        and boundary.get("original_stage_8_terminal_status") == "STOPPED_ON_FAILURE"
        and boundary.get("original_t6_cpu_failure_preserved") is True
    )
    _add(
        checks,
        "stage9_locked_and_structural_claim_boundary_preserved",
        boundary_valid,
        f"stage9_locked={boundary.get('stage_9_locked')}",
    )
    failures = _sequence(evidence.get("failures"))
    failed_cases = [
        case
        for case in cases
        if case.get("status") in {"INDEX_BLOCKED", "MEMORY_BLOCKED", "FAIL", "TIME_LIMIT"}
    ]
    _add(
        checks,
        "all_resource_and_solver_failures_preserved",
        len(failures) >= len(failed_cases)
        and all(bool(_mapping(case.get("failure"))) for case in failed_cases),
        f"failed_cases={len(failed_cases)}, failure_records={len(failures)}",
    )

    passed = all(bool(row["passed"]) for row in checks)
    return {
        "schema_version": "1.0",
        "stage": 8,
        "campaign": "gpu_only_sequence_6_8_completion",
        "checker_status": "PASS" if passed else "FAIL",
        "all_passed": passed,
        "campaign_status": evidence.get("status"),
        "evidence": evidence_path.relative_to(PROJECT_ROOT).as_posix(),
        "configuration": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "checks": checks,
        "summary": {
            "passed": sum(bool(row["passed"]) for row in checks),
            "total": len(checks),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_checks(args.evidence.resolve(), args.config.resolve())
    stage7._atomic_write_json(args.output.resolve(), result)
    print(
        f"Stage 8 GPU-only continuation checker: {result['checker_status']} "
        f"({result['summary']['passed']}/{result['summary']['total']})"
    )
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
