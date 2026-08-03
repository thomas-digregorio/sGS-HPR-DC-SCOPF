"""Independently validate Stage 8 planning, order, safety, and solver evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpu_dcopf_hpr.stage7_scalable_model import all_stage7_preflights  # noqa: E402
from scripts import check_stage_7 as stage7_checker  # noqa: E402
from scripts import run_stage_8 as stage8  # noqa: E402

DEFAULT_CONFIG = stage8.DEFAULT_CONFIG
DEFAULT_EVIDENCE = stage8.DEFAULT_OUTPUT / stage8.FINAL_NAME
DEFAULT_OUTPUT = stage8.DEFAULT_OUTPUT / "stage_8_checks.json"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _load(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, f"{type(error).__name__}: {error}"
    return (value, None) if isinstance(value, dict) else ({}, "top-level JSON must be an object")


def _add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _resource_ledger_valid(evidence: Mapping[str, Any], base_evidence: Mapping[str, Any]) -> bool:
    rows = [_mapping(row) for row in _sequence(evidence.get("resource_ledger"))]
    indexed = {str(row.get("key")): row for row in rows}
    exact = stage8._exact_nnz_by_key(base_evidence)
    preflights = {_key(preflight): preflight for preflight in all_stage7_preflights()}
    if len(rows) != 18 or set(indexed) != set(stage7_checker.EXPECTED_ROWS):
        return False
    stage7_keys = set(stage7_checker.EXPECTED_CASES)
    for key, row in indexed.items():
        preflight = preflights[key]
        expected_estimate = stage8._resource_estimate(preflight, exact[key])
        if key in stage7_keys:
            disposition = "stage7_completed"
        elif key in stage8.CAMPAIGN_ORDER:
            disposition = "stage8_campaign"
        else:
            disposition = "reconciliation_only"
        expected_static = "BLOCKED" if key in stage8.STATIC_CSR32_BLOCKS else "ELIGIBLE"
        expected_reasons = (
            ["signed_int32_csr_nnz_limit"] if key in stage8.STATIC_CSR32_BLOCKS else []
        )
        if not (
            row.get("campaign_sequence")
            == (stage8.CAMPAIGN_ORDER.index(key) + 1 if key in stage8.CAMPAIGN_ORDER else None)
            and row.get("disposition") == disposition
            and isinstance(row.get("allocation_permitted_this_invocation"), bool)
            and row.get("static_preallocation_status") == expected_static
            and row.get("static_block_reasons") == expected_reasons
            and row.get("resource_estimate") == expected_estimate
        ):
            return False
    return True


def _key(preflight: Any) -> str:
    return f"{preflight.row.case_name}:T{preflight.row.periods}"


def _resource_gate_valid(case: Mapping[str, Any]) -> bool:
    gate = _mapping(case.get("stage8_resource_gate"))
    observation = _mapping(gate.get("observation"))
    host = _mapping(observation.get("host"))
    checks = _mapping(gate.get("checks"))
    host_available = host.get("available_bytes")
    device_free = observation.get("device_free_bytes")
    expected_host = (
        int(0.8 * host_available)
        if isinstance(host_available, int) and host_available > 0
        else None
    )
    expected_device = (
        int(0.8 * device_free) if isinstance(device_free, int) and device_free > 0 else None
    )
    projected = gate.get("projected_unified_peak_bytes")
    expected_checks = {
        "observation_available": observation.get("passed") is True,
        "dimensions_match_table": checks.get("dimensions_match_table") is True,
        "signed_int32_csr_supported": checks.get("signed_int32_csr_supported") is True,
        "within_host_safety_budget": (
            expected_host is not None and isinstance(projected, int) and projected <= expected_host
        ),
        "within_device_safety_budget": (
            expected_device is not None
            and isinstance(projected, int)
            and projected <= expected_device
        ),
    }
    return (
        gate.get("host_safety_budget_bytes") == expected_host
        and gate.get("device_safety_budget_bytes") == expected_device
        and gate.get("observed_device_total_bytes") == observation.get("device_total_bytes")
        and gate.get("observed_device_free_bytes") == device_free
        and checks == expected_checks
        and gate.get("block_reasons")
        == [name for name, passed in expected_checks.items() if not passed]
        and gate.get("passed") is all(expected_checks.values())
        and gate.get("evaluated_before_full_lp_allocation") is True
    )


def _successful_case_valid(
    case: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any],
    expected_nnz: int,
    gurobi_available: bool,
) -> bool:
    key = str(case.get("key"))
    parsed = stage8._split_key(key)
    expected_m, expected_n, paper_nnz, _, _ = stage7_checker.EXPECTED_ROWS[key]
    spec = next(item for item in base_config["cases"] if item["case"] == parsed.case_name)
    storage = int(spec["storage"])
    m1 = parsed.periods + storage
    m2 = expected_m - m1
    construction = _mapping(case.get("construction"))
    dimensions = _mapping(construction.get("dimensions"))
    reconciliation = _mapping(case.get("structural_reconciliation"))
    tracks = _mapping(case.get("solver_tracks"))
    highs = _mapping(tracks.get("highs"))
    reference = _mapping(_mapping(highs.get("correctness")).get("candidate")).get("objective")
    gpu = _mapping(tracks.get("gpu_fp64_sgs_hpr"))
    timing = _mapping(case.get("timing_boundaries"))
    structural = (
        case.get("status") == "PASS"
        and case.get("passed") is True
        and case.get("full_lp_allocation_attempted") is True
        and _resource_gate_valid(case)
        and _mapping(case.get("stage8_resource_gate")).get("passed") is True
        and case.get("case_name") == parsed.case_name
        and case.get("periods") == parsed.periods
        and _mapping(case.get("preflight")).get("passed") is True
        and dimensions.get("periods") == parsed.periods
        and dimensions.get("m") == expected_m
        and dimensions.get("n") == expected_n
        and dimensions.get("m1") == m1
        and dimensions.get("m2") == m2
        and dimensions.get("nnz_A") == expected_nnz
        and reconciliation
        == {
            "dimension_match": True,
            "published_nnz": paper_nnz,
            "actual_nnz": expected_nnz,
            "nnz_difference": expected_nnz - paper_nnz,
            "symbolic_reconstructed_nnz": expected_nnz,
            "actual_matches_symbolic_nnz": True,
            "paper_time_comparable": False,
            "classification": "structural_reproduction_not_author_instance",
        }
        and re.fullmatch(
            r"[0-9a-f]{64}", str(_mapping(construction.get("lp_fingerprint")).get("sha256", ""))
        )
        is not None
        and stage7_checker._preprocessing_valid(
            case.get("preprocessing"),
            periods=parsed.periods,
            storage=storage,
            m1=m1,
            m2=m2,
            n=expected_n,
        )
    )
    tracks_valid = (
        set(tracks) == {*stage7_checker.REQUIRED_TRACKS, "gurobi"}
        and stage7_checker._finite(reference)
        and stage7_checker._track_valid(highs, track_name="highs", reference_objective=None)
        and stage7_checker._track_valid(
            tracks.get("cpu_fp64_sgs_hpr"),
            track_name="cpu_fp64_sgs_hpr",
            reference_objective=float(reference) if stage7_checker._finite(reference) else math.nan,
        )
        and stage7_checker._track_valid(
            gpu,
            track_name="gpu_fp64_sgs_hpr",
            reference_objective=float(reference) if stage7_checker._finite(reference) else math.nan,
        )
        and stage7_checker._gurobi_track_valid(
            tracks.get("gurobi"),
            available=gurobi_available,
            reference_objective=float(reference) if stage7_checker._finite(reference) else math.nan,
        )
    )
    gpu_valid = (
        stage7_checker._gpu_memory_report_valid(gpu.get("memory_before"))
        and stage7_checker._gpu_memory_report_valid(gpu.get("memory_after"))
        and stage7_checker._transfer_ledger_valid(
            gpu.get("preparation_transfer_delta"), require_solver_phases=False
        )
        and stage7_checker._transfer_ledger_valid(
            gpu.get("cumulative_transfer_ledger"), require_solver_phases=False
        )
        and all(
            stage7_checker._kernel_selection_valid(_mapping(gpu.get("kernel_selection")).get(name))
            for name in ("A1", "A2")
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
        and timing.get("solver_core_samples_are_stored_per_track") is True
        and timing.get("speedup_computed") is False
    )
    return structural and tracks_valid and gpu_valid and timing_valid


def _blocked_case_valid(case: Mapping[str, Any]) -> bool:
    gate = _mapping(case.get("stage8_resource_gate"))
    failure = _mapping(case.get("failure"))
    return (
        case.get("status") == "MEMORY_BLOCKED"
        and case.get("passed") is False
        and case.get("full_lp_allocation_attempted") is False
        and _resource_gate_valid(case)
        and gate.get("passed") is False
        and failure.get("type") == "MemorySafetyBlock"
        and failure.get("full_lp_allocated") is False
    )


def _failure_case_valid(case: Mapping[str, Any]) -> bool:
    return (
        case.get("status") in {"FAIL", "TIME_LIMIT"}
        and case.get("passed") is False
        and case.get("full_lp_allocation_attempted") is True
        and _resource_gate_valid(case)
        and _mapping(case.get("stage8_resource_gate")).get("passed") is True
        and bool(_mapping(case.get("failure")))
    )


def _campaign_valid(
    evidence: Mapping[str, Any],
    base_config: Mapping[str, Any],
    base_evidence: Mapping[str, Any],
) -> tuple[bool, str]:
    cases = [_mapping(case) for case in _sequence(evidence.get("cases"))]
    indexed = {str(case.get("key")): case for case in cases}
    if len(indexed) != len(cases) or not set(indexed).issubset(set(stage8.CAMPAIGN_ORDER)):
        return False, "case keys are duplicated or unauthorized"
    if set(indexed) & set(stage8.RECONCILIATION_ONLY_ROWS):
        return False, "a reconciliation-only row appears in allocated cases"
    seen_nonpass = False
    for key in stage8.CAMPAIGN_ORDER:
        case = indexed.get(key)
        if case is None:
            seen_nonpass = True
            continue
        if seen_nonpass:
            return False, f"{key} bypasses an incomplete predecessor"
        status = case.get("status")
        if status != "PASS":
            seen_nonpass = True
        expected_nnz = stage8._exact_nnz_by_key(base_evidence)[key]
        gurobi_available = bool(
            _mapping(_mapping(evidence.get("solver_availability")).get("gurobi")).get("available")
        )
        if status == "PASS":
            valid = _successful_case_valid(
                case,
                base_config=base_config,
                expected_nnz=expected_nnz,
                gurobi_available=gurobi_available,
            )
        elif status == "MEMORY_BLOCKED":
            valid = _blocked_case_valid(case)
        else:
            valid = _failure_case_valid(case)
        if not valid:
            return False, f"invalid terminal evidence for {key}"
    return True, f"terminal_cases={len(cases)}"


def _allocation_order_valid(evidence: Mapping[str, Any]) -> tuple[bool, str]:
    history = [_mapping(row) for row in _sequence(evidence.get("allocation_history"))]
    invocations = [_mapping(row) for row in _sequence(evidence.get("invocations"))]
    invocation_index = {str(row.get("id")): row for row in invocations}
    unique: list[str] = []
    valid = len(invocation_index) == len(invocations)
    for row in history:
        key = str(row.get("key"))
        if key not in stage8.CAMPAIGN_ORDER:
            valid = False
            continue
        invocation = invocation_index.get(str(row.get("invocation_id")))
        valid = valid and bool(
            invocation
            and invocation.get("mode") == "run_next"
            and invocation.get("approval_token_matched") is True
            and _sequence(invocation.get("allocated_keys")) == [key]
            and row.get("preallocation_gate_passed") is True
            and row.get("sequence") == stage8.CAMPAIGN_ORDER.index(key) + 1
        )
        if key not in unique:
            unique.append(key)
    valid = valid and all(len(_sequence(row.get("allocated_keys"))) <= 1 for row in invocations)
    valid = valid and tuple(unique) == stage8.CAMPAIGN_ORDER[: len(unique)]
    boundary = _mapping(evidence.get("stage_boundary"))
    valid = valid and boundary.get("stage_8_allocation_attempt_count") == len(history)
    valid = valid and boundary.get("unique_allocated_keys") == unique
    valid = valid and boundary.get("reconciliation_only_allocation_count") == 0
    valid = valid and boundary.get("stage_9_allocation_count") == 0
    return valid, f"attempts={len(history)}, unique={unique}"


def _terminal_campaign_valid(evidence: Mapping[str, Any]) -> tuple[bool, str]:
    status = evidence.get("status")
    boundary = _mapping(evidence.get("stage_boundary"))
    cases = {
        str(_mapping(case).get("key")): _mapping(case) for case in _sequence(evidence.get("cases"))
    }
    passing_prefix = 0
    for key in stage8.CAMPAIGN_ORDER:
        if cases.get(key, {}).get("status") == "PASS":
            passing_prefix += 1
        else:
            break
    stopped = next(
        (
            cases[key]
            for key in stage8.CAMPAIGN_ORDER
            if cases.get(key, {}).get("status") in {"FAIL", "TIME_LIMIT", "MEMORY_BLOCKED"}
        ),
        None,
    )
    if status == "PASS":
        valid = (
            passing_prefix == len(stage8.CAMPAIGN_ORDER)
            and stopped is None
            and evidence.get("all_passed") is True
        )
    elif status == "COMPLETE_WITH_RESOURCE_LIMIT":
        valid = (
            stopped is not None
            and stopped.get("status") == "MEMORY_BLOCKED"
            and evidence.get("all_passed") is False
        )
    elif status == "STOPPED_ON_FAILURE":
        valid = (
            stopped is not None
            and stopped.get("status") in {"FAIL", "TIME_LIMIT"}
            and evidence.get("all_passed") is False
        )
    else:
        return False, f"nonterminal campaign status={status}"
    valid = valid and (
        boundary.get("stage_8_complete") is True
        and boundary.get("next_authorized_key") is None
        and boundary.get("passing_prefix_length") == passing_prefix
        and isinstance(evidence.get("completed_utc"), str)
    )
    return valid, f"status={status}, passing_prefix={passing_prefix}"


def _source_identity_valid(evidence: Mapping[str, Any]) -> bool:
    commit = _mapping(_mapping(evidence.get("environment")).get("git")).get("head")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        return False
    manifest = [_mapping(row) for row in _sequence(evidence.get("source_manifest"))]
    if not manifest:
        return False
    for row in manifest:
        path = str(row.get("path"))
        if not (
            row.get("passed") is True
            and row.get("sha256_definition") == stage7_checker.CANONICAL_GIT_BLOB_SHA256_DEFINITION
            and row.get("git_blob") == stage7_checker._git_blob_oid(commit, path)
            and row.get("sha256") == stage7_checker._git_blob_sha256(commit, path)
        ):
            return False
    return True


def run_checks(
    evidence_path: Path = DEFAULT_EVIDENCE,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    evidence, evidence_error = _load(evidence_path)
    config, config_error = _load(config_path)
    config_identity = stage8._portable_identity(config_path)
    _add(
        checks,
        "evidence_and_configuration_load",
        not evidence_error and not config_error,
        f"evidence={evidence_error or 'loaded'}, config={config_error or 'loaded'}",
    )
    config_valid = (
        config_error is None
        and config_identity.get("passed") is True
        and config_identity.get("canonical_git_blob_sha256") == stage8.FROZEN_CONFIG_SHA256
        and stage8._validate_stage8_config(config) == []
        and evidence.get("configuration") == config
        and _mapping(evidence.get("configuration_validation")).get("passed") is True
    )
    _add(
        checks,
        "frozen_stage8_configuration",
        config_valid,
        f"canonical_sha256={config_identity.get('canonical_git_blob_sha256')}",
    )

    try:
        base_contract, base = stage8._base_contract(config)
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
        "accepted_stage7_artifacts_preserved",
        base_valid,
        f"hashes={base_contract.get('sha256')}",
    )
    _add(
        checks,
        "executed_source_identity",
        _source_identity_valid(evidence),
        f"manifest_rows={len(_sequence(evidence.get('source_manifest')))}",
    )

    try:
        ledger_valid = _resource_ledger_valid(evidence, base["evidence"])
    except Exception:
        ledger_valid = False
    _add(
        checks,
        "all_18_rows_have_preallocation_resource_estimates",
        ledger_valid,
        f"rows={len(_sequence(evidence.get('resource_ledger')))}",
    )
    static = {
        str(row.get("key")): row
        for row in _sequence(evidence.get("resource_ledger"))
        if isinstance(row, dict)
    }
    static_blocks_valid = all(
        _mapping(static.get(key)).get("static_preallocation_status") == "BLOCKED"
        and _mapping(_mapping(static.get(key)).get("resource_estimate")).get(
            "signed_int32_csr_supported"
        )
        is False
        for key in stage8.STATIC_CSR32_BLOCKS
    )
    _add(
        checks,
        "t24_t32_signed_int32_blocks_recorded_without_allocation",
        static_blocks_valid,
        f"blocked={sorted(stage8.STATIC_CSR32_BLOCKS)}",
    )

    order_valid, order_detail = _allocation_order_valid(evidence)
    _add(
        checks,
        "one_case_per_invocation_strict_prefix_and_no_locked_allocations",
        order_valid,
        order_detail,
    )
    try:
        campaign_valid, campaign_detail = _campaign_valid(
            evidence, base["configuration"], base["evidence"]
        )
    except Exception as error:
        campaign_valid, campaign_detail = False, f"{type(error).__name__}: {error}"
    _add(
        checks,
        "terminal_success_failure_and_memory_evidence_is_honest",
        campaign_valid,
        campaign_detail,
    )
    terminal_valid, terminal_detail = _terminal_campaign_valid(evidence)
    _add(
        checks,
        "stage8_campaign_reached_an_honest_terminal_protocol_state",
        terminal_valid,
        terminal_detail,
    )
    no_speedup = all(
        _mapping(_mapping(case).get("timing_boundaries")).get("speedup_computed") is False
        for case in _sequence(evidence.get("cases"))
        if _mapping(case).get("status") == "PASS"
    )
    _add(
        checks,
        "no_unsupported_speedup_computed",
        no_speedup,
        "successful tracks retain separate timing boundaries",
    )
    boundary = _mapping(evidence.get("stage_boundary"))
    boundary_valid = (
        boundary.get("stage_8_only") is True
        and boundary.get("stage_9_locked") is True
        and boundary.get("n_minus_1_extension_enabled") is False
        and boundary.get("exact_paper_reproduction_claimed") is False
        and boundary.get("paper_a100_timing_reproduction_claimed") is False
    )
    _add(
        checks,
        "structural_claim_and_stage9_boundary_preserved",
        boundary_valid,
        f"status={evidence.get('status')}",
    )
    failures = _sequence(evidence.get("failures"))
    terminal_failures = [
        case
        for case in _sequence(evidence.get("cases"))
        if _mapping(case).get("status") in {"FAIL", "TIME_LIMIT", "MEMORY_BLOCKED"}
    ]
    failure_valid = not terminal_failures or len(failures) >= len(terminal_failures)
    _add(
        checks,
        "terminal_failures_preserved",
        failure_valid,
        f"terminal_failures={len(terminal_failures)}, failure_records={len(failures)}",
    )
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "schema_version": "1.0",
        "stage": 8,
        "checker_status": "PASS" if passed else "FAIL",
        "all_passed": passed,
        "campaign_status": evidence.get("status"),
        "evidence": evidence_path.relative_to(PROJECT_ROOT).as_posix(),
        "configuration": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "checks": checks,
        "summary": {
            "passed": sum(bool(check["passed"]) for check in checks),
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
    stage7_checker._atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
