from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpu_dcopf_hpr.dcopf_model import build_dcopf_model, load_dcopf_config
from gpu_dcopf_hpr.network_data import load_matpower_case
from gpu_dcopf_hpr.preconditioning import precondition_lp
from gpu_dcopf_hpr.residuals import evaluate_residuals
from gpu_dcopf_hpr.sgs_hpr import prepare_sgs_hpr
from gpu_dcopf_hpr.stage7_scalable_model import stage7_preflight
from gpu_dcopf_hpr.stage7_scaled_y1 import prepare_scaled_block_arrow_y1
from gpu_dcopf_hpr.stage7_spectral import estimate_sparse_spectral_norm_squared
from gpu_dcopf_hpr.structural_y1 import DCOPFEqualityStructure
from gpu_dcopf_hpr.toy_problems import analytic_toy_case
from scripts.run_stage_7 import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    _atomic_write_json,
    _attempt,
    _audit_gpu_solver_transfers,
    _compatible_partial,
    _fresh_case_for_retry,
    _memory_guard,
    _policy_contract,
    _requirements_freeze,
    _run_timed_track,
    _solve_cpu_hpr,
    _solve_highs,
    _symbolic_ledger,
    _timing_statistics,
    _transfer_delta,
    _validate_stage7_config,
)


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_frozen_stage7_config_and_implementation_policy_agree() -> None:
    config = _config()

    assert _validate_stage7_config(config) == []
    assert _policy_contract(config)["passed"]


def test_config_validation_rejects_execution_and_stage8_drift() -> None:
    config = _config()
    invalid = copy.deepcopy(config)
    invalid["algorithm"]["mixed_precision_enabled"] = True
    invalid["timing"]["measured_runs"] = 1
    invalid["stage_boundary"]["stage_8_large_runs_locked"] = False
    invalid["cases"][2]["rows"][0]["execute_stage_7"] = True

    errors = _validate_stage7_config(invalid)

    assert "algorithm.mixed_precision_enabled drifted from the frozen contract" in errors
    assert "timing.measured_runs drifted from the frozen contract" in errors
    assert "stage_boundary.stage_8_large_runs_locked is invalid" in errors
    assert "Stage 7 must execute exactly six preregistered rows" in errors
    assert "case9241pegase allocations remain locked to Stage 8" in errors


def test_config_validation_rejects_acceptance_gate_drift() -> None:
    invalid = copy.deepcopy(_config())
    replacements = {
        "paper_residual_tolerance": 1e-3,
        "raw_kkt_tolerance": 1.0,
        "maximum_physical_violation": 1.0,
        "maximum_scaled_objective_gap_to_highs": 1e-2,
        "gurobi_required_only_when_installed_and_licensed": False,
    }
    invalid["acceptance"].update(replacements)

    errors = _validate_stage7_config(invalid)

    for key in replacements:
        assert f"acceptance.{key} drifted from the frozen contract" in errors


def test_config_validation_rejects_same_count_but_wrong_case2868_horizons() -> None:
    invalid = copy.deepcopy(_config())
    case2868_rows = invalid["cases"][1]["rows"]
    case2868_rows[1]["execute_stage_7"] = False
    case2868_rows[2]["execute_stage_7"] = True

    errors = _validate_stage7_config(invalid)

    assert (
        sum(bool(row["execute_stage_7"]) for case in invalid["cases"] for row in case["rows"]) == 6
    )
    assert (
        "Stage 7 executable rows must remain exactly case1354pegase "
        "T4/T16/T48/T96 and case2868rte T4/T16"
    ) in errors


def test_symbolic_ledger_covers_all_rows_without_stage8_allocations() -> None:
    ledger = _symbolic_ledger(_config())

    assert len(ledger) == 18
    assert len({row["key"] for row in ledger}) == 18
    assert sum(row["execute_stage_7"] for row in ledger) == 6
    assert all(row["dimensions_match_table"] for row in ledger)
    assert all(row["paper_values_match_config"] for row in ledger)
    assert all(set(row["dimension_comparison"]) == {"m", "n"} for row in ledger)
    assert all(row["dimension_comparison"]["m"]["absolute_difference"] == 0 for row in ledger)
    assert all(row["dimension_comparison"]["n"]["absolute_difference"] == 0 for row in ledger)
    assert all("cause" in row["nnz_comparison"] for row in ledger)
    assert all(
        row["allocation_permitted_this_run"] is False
        and row["allocation_stage"] == "stage_8_locked"
        for row in ledger
        if not row["execute_stage_7"]
    )
    unsupported = {row["key"] for row in ledger if not row["csr32_supported"]}
    assert unsupported == {"case9241pegase:T24", "case9241pegase:T32"}
    assert all(row["csr32_supported"] for row in ledger if row["execute_stage_7"])


def test_stage7_requirements_freeze_contains_execution_critical_pins() -> None:
    freeze = _requirements_freeze()

    assert freeze["passed"]
    assert freeze["pins"]["cupy-cuda13x"] == "14.1.1"
    assert freeze["pins"]["numpy"] == "2.3.5"
    assert freeze["pins"]["scipy"] == "1.16.3"
    assert len(freeze["sha256"]) == 64


def test_memory_guard_is_explicit_and_fail_closed() -> None:
    preflight = stage7_preflight("case1354pegase", 4)
    enough = _memory_guard(
        preflight,
        host_memory={"available_bytes": 10 * preflight.host_assembly_peak_bytes},
        device_total_bytes=10 * preflight.gpu_planning_bytes,
    )
    too_small = _memory_guard(
        preflight,
        host_memory={"available_bytes": 1},
        device_total_bytes=1,
    )

    assert enough["passed"]
    assert not too_small["passed"]
    assert not too_small["checks"]["host_available"]
    assert not too_small["checks"]["device_available"]


def test_timing_statistics_preserve_raw_repetitions() -> None:
    result = _timing_statistics([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result["raw_seconds"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["count"] == 5
    assert result["median_seconds"] == 3.0
    assert result["minimum_seconds"] == 1.0
    assert result["maximum_seconds"] == 5.0
    assert result["interquartile_range_seconds"] == 2.0
    assert np.isclose(result["standard_deviation_seconds"], np.sqrt(2.5))


def test_highs_adapter_uses_compatible_bounds_and_preserves_dual_signs() -> None:
    case = analytic_toy_case()

    result = _solve_highs(SimpleNamespace(lp=case.lp), time_limit_seconds=30)

    assert result["status"] == "SUCCESS"
    state = result["state"]
    assert np.allclose(state.x, case.expected_state.x, atol=case.solution_tolerance)
    assert state.y[case.lp.m1] >= 0.0
    residuals = evaluate_residuals(
        case.lp,
        x=state.x,
        y=state.y,
        z=state.z,
        tolerance=1e-7,
    )
    assert residuals.conditions.all_satisfied


def test_cpu_adapter_reuses_exact_prepared_workspace() -> None:
    network = load_matpower_case(PROJECT_ROOT / "data/raw/matpower/case5.m")
    dcopf_config = load_dcopf_config(
        PROJECT_ROOT / "configs/dcopf/case5_base_stage_2.json",
        network,
    )
    model = build_dcopf_model(network, dcopf_config)
    config = _config()
    algorithm = config["algorithm"]
    preconditioner = precondition_lp(
        model.lp,
        ruiz_iterations=int(algorithm["ruiz_iterations"]),
        pock_chambolle=bool(algorithm["pock_chambolle"]),
        normalize=bool(algorithm["normalize_b_and_c"]),
    )
    scaled_solver = prepare_scaled_block_arrow_y1(
        preconditioner,
        DCOPFEqualityStructure.from_model(model),
    )
    spectral = estimate_sparse_spectral_norm_squared(preconditioner.scaled_lp.A2)
    workspace = prepare_sgs_hpr(
        preconditioner.scaled_lp,
        scaled_structural_y1=scaled_solver,
        spectral_certificate=spectral,
    )

    attempt = _solve_cpu_hpr(
        model,
        preconditioner,
        scaled_solver,
        spectral,
        workspace,
        config=config,
    )

    assert attempt["status"] == "SUCCESS"
    assert attempt["result"].workspace is workspace
    assert attempt["result"].preparation_elapsed_seconds == 0.0
    assert attempt["result"].converged


def test_atomic_json_and_resume_require_matching_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    payload = {"run_fingerprint": "same", "value": np.float64(1.25)}

    _atomic_write_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "run_fingerprint": "same",
        "value": 1.25,
    }
    assert _compatible_partial(path, "same") == {
        "run_fingerprint": "same",
        "value": 1.25,
    }
    assert _compatible_partial(path, "different") is None
    assert list(tmp_path.glob("*.tmp")) == []


def test_track_validates_before_warmup_and_five_measurements() -> None:
    config = _config()
    config["timing"]["relative_range_escalation_threshold"] = 10.0
    events: list[str] = []
    checkpoints: list[int] = []

    def solve() -> dict:
        events.append("solve")
        return {"status": "SUCCESS", "wall_seconds": 1.0}

    def summarize(result: dict) -> dict:
        return {**result, "passed": True}

    track = {"name": "test"}
    _run_timed_track(
        track=track,
        solve=solve,
        summarize=summarize,
        config=config,
        checkpoint=lambda: checkpoints.append(len(events)),
    )

    assert len(events) == 7  # correctness + one warm-up + five measured
    assert track["correctness"]["passed"]
    assert len(track["warmup"]) == 1
    assert len(track["measured_repetitions"]) == 5
    assert track["statistics"]["count"] == 5
    assert track["timing_status"] == "COMPLETE"
    assert track["first_run"] == track["correctness"]
    assert track["first_measured_run"] == track["measured_repetitions"][0]
    assert checkpoints[0] == 1


def test_track_escalates_to_nine_repetitions_only_after_variability() -> None:
    config = _config()
    values = iter([0.0, 0.0, 1.0, 3.0, 1.0, 3.0, 1.0, 2.0, 2.0, 2.0, 2.0])

    def solve() -> dict:
        return {"status": "SUCCESS", "wall_seconds": next(values)}

    track = {"name": "variable"}
    _run_timed_track(
        track=track,
        solve=solve,
        summarize=lambda result: {**result, "passed": True},
        config=config,
        checkpoint=lambda: None,
    )

    assert track["variability_escalated"]
    assert len(track["measured_repetitions"]) == 9
    assert track["statistics"]["count"] == 9


def test_failed_correctness_never_produces_timing_samples() -> None:
    calls = 0

    def solve() -> dict:
        nonlocal calls
        calls += 1
        return {"status": "FAIL", "wall_seconds": 0.1}

    track = {"name": "failed"}
    _run_timed_track(
        track=track,
        solve=solve,
        summarize=lambda result: {**result, "passed": False},
        config=_config(),
        checkpoint=lambda: None,
    )

    assert calls == 1
    assert track["timing_status"] == "NOT_RUN_CORRECTNESS_FAILED"
    assert track["warmup"] == []
    assert track["measured_repetitions"] == []
    assert not track["passed"]


def test_invalid_measured_candidate_stops_track_even_with_success_status() -> None:
    calls = 0

    def solve() -> dict:
        nonlocal calls
        calls += 1
        return {"status": "SUCCESS", "wall_seconds": 0.01}

    def summarize(result: dict) -> dict:
        # Correctness, warm-up, and the first measurement pass; the second
        # measurement exposes an invalid candidate despite a success status.
        return {**result, "passed": calls < 4}

    track = {"name": "invalid-late-candidate"}
    _run_timed_track(
        track=track,
        solve=solve,
        summarize=summarize,
        config=_config(),
        checkpoint=lambda: None,
    )

    assert calls == 4
    assert track["timing_status"] == "INVALID_MEASURED_CANDIDATE"
    assert len(track["measured_repetitions"]) == 2
    assert not track["passed"]


def test_resume_rejects_persisted_failed_warmup_without_new_solves() -> None:
    calls = 0

    def solve() -> dict:
        nonlocal calls
        calls += 1
        return {"status": "SUCCESS", "wall_seconds": 0.01}

    track = {
        "name": "resumed-warmup",
        "correctness": {"status": "SUCCESS", "passed": True},
        "warmup": [{"status": "FAIL", "passed": False, "wall_seconds": 0.01, "repetition": 1}],
        "measured_repetitions": [],
    }

    _run_timed_track(
        track=track,
        solve=solve,
        summarize=lambda result: {**result, "passed": True},
        config=_config(),
        checkpoint=lambda: None,
    )

    assert calls == 0
    assert track["timing_status"] == "INVALID_RESUMED_WARMUP_STATE"
    assert not track["passed"]


def test_resume_rejects_persisted_failed_measurement_without_new_solves() -> None:
    calls = 0

    def solve() -> dict:
        nonlocal calls
        calls += 1
        return {"status": "SUCCESS", "wall_seconds": 0.01}

    track = {
        "name": "resumed-measurement",
        "correctness": {"status": "SUCCESS", "passed": True},
        "warmup": [{"status": "SUCCESS", "passed": True, "wall_seconds": 0.01, "repetition": 1}],
        "measured_repetitions": [
            {"status": "FAIL", "passed": False, "wall_seconds": 0.02, "repetition": 1}
        ],
    }

    _run_timed_track(
        track=track,
        solve=solve,
        summarize=lambda result: {**result, "passed": True},
        config=_config(),
        checkpoint=lambda: None,
    )

    assert calls == 0
    assert track["timing_status"] == "INVALID_RESUMED_MEASUREMENT_STATE"
    assert not track["passed"]


def test_retry_restarts_terminal_case_without_reusing_solver_samples() -> None:
    prior = {
        "key": "case1354pegase:T4",
        "case_name": "case1354pegase",
        "periods": 4,
        "status": "FAIL",
        "passed": False,
        "failure": {"phase": "case:case1354pegase:T4", "type": "RuntimeError"},
        "solver_tracks": {
            "highs": {
                "timing_status": "WARMUP_FAILED_OR_INVALID",
                "warmup": [{"status": "FAIL"}],
            }
        },
    }

    reset = _fresh_case_for_retry(
        prior,
        key=SimpleNamespace(
            text="case1354pegase:T4",
            case_name="case1354pegase",
            periods=4,
        ),
        fingerprint="frozen-fingerprint",
    )

    assert reset["status"] == "PENDING"
    assert reset["run_fingerprint"] == "frozen-fingerprint"
    assert "solver_tracks" not in reset
    assert reset["retry_history"][0]["prior_status"] == "FAIL"
    assert reset["retry_history"][0]["prior_solver_timing_status"] == {
        "highs": "WARMUP_FAILED_OR_INVALID"
    }


def test_attempt_records_elapsed_time_and_process_memory_on_failure() -> None:
    def fail() -> dict:
        raise RuntimeError("expected")

    result = _attempt("unit", fail)

    assert result["status"] == "FAIL"
    assert result["wall_seconds"] >= 0.0
    assert result["attempt_wall_seconds"] >= 0.0
    assert set(result["process_memory"]) == {"before", "after"}
    assert result["failure"]["type"] == "RuntimeError"


def test_gpu_transfer_delta_and_residency_audit_reject_hidden_state_copy() -> None:
    before = {
        "records": [
            {
                "phase": "preparation",
                "direction": "host_to_device",
                "kind": "matrix",
                "calls": 1,
                "bytes": 32,
            }
        ]
    }
    after = {
        "records": [
            *before["records"],
            {
                "phase": "initial_state",
                "direction": "host_to_device",
                "kind": "vector",
                "calls": 3,
                "bytes": 96,
            },
            {
                "phase": "periodic_diagnostics",
                "direction": "device_to_host",
                "kind": "scalar_packet",
                "calls": 2,
                "bytes": 160,
            },
        ]
    }

    delta = _transfer_delta(before, after)
    assert _audit_gpu_solver_transfers(delta)["passed"]
    after["records"].append(
        {
            "phase": "hidden_state_copy",
            "direction": "device_to_host",
            "kind": "vector",
            "calls": 1,
            "bytes": 1024,
        }
    )
    rejected = _audit_gpu_solver_transfers(_transfer_delta(before, after))
    assert not rejected["passed"]
    assert rejected["full_state_copied_inside_resident_loop"]
