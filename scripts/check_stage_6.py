"""Independently validate preserved Stage 6 DGX evidence.

The checker deliberately recomputes acceptance decisions from recorded values.
It does not accept the runner's aggregate ``passed`` fields as proof.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sgs_hpr" / "stage_6_gpu_dgx.json"
DEFAULT_EVIDENCE = PROJECT_ROOT / "results" / "raw" / "stage_6" / "stage_6_validation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_6" / "stage_6_checks.json"
DEFAULT_TRAJECTORY_NAME = "stage_6_trajectories.jsonl.gz"
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "environment" / "dgx_stage6_requirements.txt"
DEFAULT_NETWORK = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
DEFAULT_DCOPF_CONFIGS = (
    PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json",
    PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json",
)
FROZEN_STAGE_6_COMMIT = "2a8e4936a66c7ea4dade4ca208419076d603b446"

EXPECTED_CASES = {
    "case5_base_t1": 1,
    "case5_synthetic_extension_t2": 2,
}
STATE_BLOCKS = {"x", "y", "z"}
RESIDUAL_BLOCKS = {"primal_feasibility", "box", "stationarity"}
ALG2_REQUESTED = "CUSPARSE_SPMV_CSR_ALG2"
ALG2_EFFECTIVE_FRAGMENT = "CUSPARSE_SPMV_CSR_ALG2 (enum 3"
DEFAULT_SPMV_LABEL = "cupyx.cusparse.spmv CUSPARSE_MV_ALG_DEFAULT"
SOLVER_TRANSFER_PHASES = {
    ("initial_state", "host_to_device"),
    ("periodic_diagnostics", "device_to_host"),
    ("policy_diagnostics", "device_to_host"),
    ("final_state", "device_to_host"),
    ("final_scaled_state", "device_to_host"),
    ("final_diagnostics", "device_to_host"),
}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative(value: Any) -> bool:
    return _finite(value) and float(value) >= 0.0


def _positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _close(left: Any, right: Any, *, rel_tol: float = 1e-12, abs_tol: float = 1e-14) -> bool:
    return (
        _finite(left)
        and _finite(right)
        and math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> bytes | None:
    """Return raw Git output, or ``None`` when the frozen object is unavailable."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    return completed.stdout if completed.returncode == 0 else None


def _frozen_stage6_source_hashes() -> dict[str, str] | None:
    """Hash the exact runner and package sources preserved by the Stage 6 commit."""

    tree = _git_output(
        "ls-tree",
        "-r",
        "--name-only",
        FROZEN_STAGE_6_COMMIT,
        "--",
        "src/gpu_dcopf_hpr",
    )
    if tree is None:
        return None
    package_paths = sorted(
        path
        for path in tree.decode("utf-8").splitlines()
        if path.startswith("src/gpu_dcopf_hpr/") and path.endswith(".py")
    )
    source_paths = ["scripts/run_stage_6.py", *package_paths]
    hashes: dict[str, str] = {}
    for path in source_paths:
        blob = _git_output("show", f"{FROZEN_STAGE_6_COMMIT}:{path}")
        if blob is None:
            return None
        hashes[path] = hashlib.sha256(blob).hexdigest()
    return hashes


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"missing {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {}, f"{type(error).__name__}: {error}"
    if not isinstance(value, dict):
        return {}, "top-level JSON value is not an object"
    return value, None


def _load_gzip_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        return [], f"missing {path}"
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return rows, f"line {line_number} is not an object"
                rows.append(value)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as error:
        return rows, f"{type(error).__name__}: {error}"
    return rows, None


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _comparison_valid(
    value: Any,
    *,
    relative_tolerance: float | None,
    absolute_tolerance: float | None,
) -> bool:
    comparison = _mapping(value)
    candidate_shape = _sequence(comparison.get("candidate_shape"))
    reference_shape = _sequence(comparison.get("reference_shape"))
    relative_error = comparison.get("relative_error")
    absolute_error = comparison.get("maximum_absolute_error")
    return (
        bool(candidate_shape)
        and candidate_shape == reference_shape
        and comparison.get("shape_matches") is True
        and comparison.get("finite") is True
        and _nonnegative(relative_error)
        and _nonnegative(absolute_error)
        and comparison.get("relative_tolerance") == relative_tolerance
        and comparison.get("absolute_tolerance") == absolute_tolerance
        and (relative_tolerance is None or float(relative_error) <= relative_tolerance)
        and (absolute_tolerance is None or float(absolute_error) <= absolute_tolerance)
    )


def _state_comparison_valid(
    value: Any,
    *,
    relative_tolerance: float | None,
    absolute_tolerance: float | None,
) -> bool:
    comparison = _mapping(value)
    blocks = _mapping(comparison.get("blocks"))
    if set(blocks) != STATE_BLOCKS:
        return False
    valid = all(
        _comparison_valid(
            blocks[name],
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        for name in STATE_BLOCKS
    )
    maximum_relative = max(float(_mapping(blocks[name])["relative_error"]) for name in STATE_BLOCKS)
    maximum_absolute = max(
        float(_mapping(blocks[name])["maximum_absolute_error"]) for name in STATE_BLOCKS
    )
    return (
        valid
        and _close(comparison.get("maximum_relative_error"), maximum_relative)
        and _close(comparison.get("maximum_absolute_error"), maximum_absolute)
    )


def _kernel_alg2_valid(value: Any, *, tolerance: float) -> bool:
    kernel = _mapping(value)
    requested = str(kernel.get("requested_label", ""))
    effective = str(kernel.get("effective_label", ""))
    return (
        ALG2_REQUESTED in requested
        and ALG2_EFFECTIVE_FRAGMENT in effective
        and kernel.get("uses_csr_alg2") is True
        and kernel.get("fallback_reason") is None
        and _nonnegative(kernel.get("probe_max_abs_error"))
        and float(kernel["probe_max_abs_error"]) <= tolerance
        and kernel.get("probe_repeat_bitwise_equal") is True
    )


def _cuda_timing_valid(value: Any, *, repetitions: int, warmup: int) -> bool:
    timing = _mapping(value)
    elapsed = timing.get("elapsed_seconds")
    return (
        _nonnegative(elapsed)
        and timing.get("repetitions") == repetitions
        and timing.get("warmup_calls") == warmup
        and _close(timing.get("elapsed_milliseconds"), 1_000.0 * float(elapsed))
        and _close(timing.get("mean_seconds"), float(elapsed) / repetitions)
    )


def _gpu_loop_timing_valid(
    value: Any,
    *,
    residual_interval: int | None = None,
    residual_count: int | None = None,
) -> bool:
    timing = _mapping(value)
    loop = timing.get("loop_gpu_seconds")
    residual = timing.get("residual_check_gpu_seconds")
    excluding = timing.get("iterations_excluding_residual_checks_gpu_seconds")
    wall = timing.get("loop_wall_seconds")
    count = timing.get("residual_check_count")
    interval = timing.get("residual_check_interval")
    return (
        _nonnegative(loop)
        and _nonnegative(residual)
        and _nonnegative(excluding)
        and _nonnegative(wall)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        and isinstance(interval, int)
        and not isinstance(interval, bool)
        and interval > 0
        and float(residual) <= float(loop) + 1e-12
        and _close(excluding, max(0.0, float(loop) - float(residual)), abs_tol=1e-10)
        and (residual_interval is None or interval == residual_interval)
        and (residual_count is None or count == residual_count)
    )


def _ledger_valid(value: Any, *, solver: bool) -> bool:
    ledger = _mapping(value)
    rows = [_mapping(row) for row in _sequence(ledger.get("records"))]
    totals = _mapping(ledger.get("totals"))
    if not rows:
        return False
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        direction = row.get("direction")
        key = (str(row.get("phase")), str(direction), str(row.get("kind")))
        if (
            key in keys
            or direction not in {"host_to_device", "device_to_host"}
            or not isinstance(row.get("calls"), int)
            or isinstance(row.get("calls"), bool)
            or int(row["calls"]) <= 0
            or not isinstance(row.get("bytes"), int)
            or isinstance(row.get("bytes"), bool)
            or int(row["bytes"]) < 0
        ):
            return False
        keys.add(key)
        if solver and (key[0], key[1]) not in SOLVER_TRANSFER_PHASES:
            return False

    for direction in ("host_to_device", "device_to_host"):
        actual = _mapping(totals.get(direction))
        matching = [row for row in rows if row.get("direction") == direction]
        if actual != {
            "calls": sum(int(row["calls"]) for row in matching),
            "bytes": sum(int(row["bytes"]) for row in matching),
        }:
            return False

    if not solver:
        return True
    by_phase = {str(row["phase"]): row for row in rows}
    initial = _mapping(by_phase.get("initial_state"))
    periodic = _mapping(by_phase.get("periodic_diagnostics"))
    final_state = _mapping(by_phase.get("final_state"))
    final_scaled = _mapping(by_phase.get("final_scaled_state"))
    final_diagnostics = _mapping(by_phase.get("final_diagnostics"))
    policy = _mapping(by_phase.get("policy_diagnostics"))
    compact_packets = (
        periodic.get("bytes") == 80 * int(periodic.get("calls", -1))
        and (not policy or policy.get("bytes") == 16 * int(policy.get("calls", -1)))
        and final_diagnostics.get("calls") == 1
        and final_diagnostics.get("bytes") == 24
    )
    return (
        initial.get("direction") == "host_to_device"
        and initial.get("calls") == 3
        and periodic.get("direction") == "device_to_host"
        and final_state.get("calls") == 3
        and final_scaled.get("calls") == 3
        and compact_packets
    )


def _transfer_timing_valid(value: Any) -> bool:
    timing = _mapping(value)
    return set(timing) == {"host_to_device_seconds", "device_to_host_seconds"} and all(
        _nonnegative(item) for item in timing.values()
    )


def _configuration_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
) -> None:
    crosscheck = _mapping(config.get("crosscheck"))
    precision = _mapping(config.get("precision_study"))
    boundary = _mapping(config.get("stage_boundary"))
    frozen = (
        config.get("stage") == 6
        and config.get("target") == "NVIDIA DGX Spark GB10"
        and config.get("gpu_backend") == "CuPy"
        and config.get("gpu_backend_version") == "14.1.1"
        and config.get("gpu_package") == "cupy-cuda13x==14.1.1"
        and config.get("precision") == "float64"
        and config.get("requested_spmv_algorithm") == ALG2_REQUESTED
        and config.get("store_explicit_transposes") is True
        and config.get("paper_tolerance") == 5e-5
        and config.get("dcopf_kkt_combined_target") == 0.01
        and config.get("dcopf_physical_tolerance") == 0.01
        and config.get("dcopf_maximum_scaled_objective_gap") == 2e-4
        and config.get("maximum_equality_infinity_residual") == 5e-10
        and config.get("maximum_z_x_identity_error") == 1e-10
        and config.get("resident_timing_iterations") == 1000
        and config.get("resident_timing_residual_check_interval") == 100
        and crosscheck.get("spmv_relative_tolerance") == 5e-13
        and crosscheck.get("spmv_absolute_tolerance") == 5e-13
        and crosscheck.get("one_step_relative_tolerance") == 2e-12
        and crosscheck.get("one_step_absolute_tolerance") == 2e-12
        and crosscheck.get("ten_step_relative_tolerance") == 2e-10
        and crosscheck.get("ten_step_absolute_tolerance") == 2e-11
        and crosscheck.get("fixed_horizon_iterations") == 100
        and crosscheck.get("fixed_horizon_state_relative_tolerance") == 1e-10
        and precision
        == {
            "fp64_required": True,
            "fp32_enabled_after_fp64_pass": True,
            "mixed_precision_enabled": False,
            "reduced_precision_is_gating": False,
        }
        and boundary
        == {
            "stage_6_only": True,
            "stage_7_benchmarks_locked": True,
            "paper_timing_reproduction_claimed": False,
            "gpu_speedup_claimed": False,
        }
    )

    inputs = _mapping(evidence.get("inputs"))
    config_input = _mapping(inputs.get("config"))
    network_input = _mapping(inputs.get("network"))
    requirement_input = _mapping(inputs.get("requirements"))
    dcopf_inputs = [_mapping(item) for item in _sequence(inputs.get("dcopf_configs"))]
    dcopf_hashes = {item.get("sha256") for item in dcopf_inputs}
    source_inputs = [_mapping(item) for item in _sequence(inputs.get("source_files"))]
    expected_source_hashes = _frozen_stage6_source_hashes()
    recorded_source_hashes = {str(item.get("path")): item.get("sha256") for item in source_inputs}
    hashes_valid = (
        config_input.get("sha256") == _sha256(config_path)
        and network_input.get("sha256") == _sha256(DEFAULT_NETWORK)
        and requirement_input.get("sha256") == _sha256(DEFAULT_REQUIREMENTS)
        and dcopf_hashes == {_sha256(path) for path in DEFAULT_DCOPF_CONFIGS}
        and len(dcopf_inputs) == 2
        and expected_source_hashes is not None
        and recorded_source_hashes == expected_source_hashes
        and len(source_inputs) == len(recorded_source_hashes) == len(expected_source_hashes) == 19
    )
    add_check(
        checks,
        "embedded_configuration_and_input_hashes_match_versioned_sources",
        frozen and evidence.get("configuration") == config and hashes_valid,
        (
            f"config_sha256={_sha256(config_path)}, "
            f"frozen_stage_6_commit={FROZEN_STAGE_6_COMMIT}, "
            f"executed_source_hashes={len(source_inputs)}, input_hashes_match={hashes_valid}"
        ),
    )

    config_validation = _mapping(evidence.get("configuration_validation"))
    identity = (
        evidence.get("stage") == 6
        and evidence.get("status") == "PASS"
        and evidence.get("all_passed") is True
        and config_validation.get("errors") == []
        and config_validation.get("passed") is True
        and _mapping(_mapping(evidence.get("precision_study")).get("fp64")).get("status") == "PASS"
    )
    add_check(
        checks,
        "stage_six_fp64_cupy_run_identity",
        identity,
        "stage=6, final status=PASS, required precision=float64, backend=CuPy",
    )


def _device_checks(
    checks: list[dict[str, Any]], evidence: dict[str, Any], config: dict[str, Any]
) -> None:
    environment = _mapping(evidence.get("environment"))
    packages = _mapping(environment.get("packages"))
    device = _mapping(evidence.get("device"))
    diagnostics = _mapping(device.get("diagnostics"))
    valid = (
        diagnostics.get("cupy_version") == config.get("gpu_backend_version")
        and "GB10" in str(diagnostics.get("device_name", "")).upper()
        and diagnostics.get("compute_capability") == [12, 1]
        and diagnostics.get("device_id") == 0
        and diagnostics.get("fp64_supported") is True
        and diagnostics.get("fp64_itemsize_bytes") == 8
        and diagnostics.get("csr_index_dtype") == "int32"
        and diagnostics.get("csr_indptr_dtype") == "int32"
        and diagnostics.get("csr_index_bits") == 32
        and isinstance(diagnostics.get("cuda_runtime_version"), int)
        and diagnostics.get("cuda_runtime_version") >= 13000
        and isinstance(diagnostics.get("cuda_driver_version"), int)
        and diagnostics.get("cuda_driver_version") >= 13000
        and _positive(diagnostics.get("total_global_memory_bytes"))
        and packages.get("cupy-cuda13x") == "14.1.1"
        and packages.get("cuda-pathfinder") == "1.6.0"
        and str(environment.get("machine", "")).lower() in {"aarch64", "arm64"}
    )
    add_check(
        checks,
        "dgx_spark_gb10_fp64_device_and_pinned_cupy_environment",
        valid,
        (
            f"device={diagnostics.get('device_name')}, "
            f"cc={diagnostics.get('compute_capability')}, "
            f"cupy={packages.get('cupy-cuda13x')}"
        ),
    )


def _case_map(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = [_mapping(case) for case in _sequence(evidence.get("cases"))]
    return {str(case.get("name")): case for case in cases if case.get("name") is not None}


def _case_coverage_checks(
    checks: list[dict[str, Any]], cases: dict[str, dict[str, Any]], config: dict[str, Any]
) -> None:
    valid = set(cases) == set(EXPECTED_CASES)
    details: list[str] = []
    for name, periods in EXPECTED_CASES.items():
        case = _mapping(cases.get(name))
        dimensions = _mapping(case.get("dimensions"))
        preconditioner = _mapping(case.get("preconditioner"))
        row_range = _sequence(preconditioner.get("row_denominator_range"))
        column_range = _sequence(preconditioner.get("column_denominator_range"))
        valid = valid and (
            dimensions.get("periods") == periods
            and isinstance(dimensions.get("n"), int)
            and int(dimensions.get("n", 0)) > 0
            and isinstance(dimensions.get("m1"), int)
            and isinstance(dimensions.get("m2"), int)
            and preconditioner.get("ruiz_iterations") == config.get("ruiz_iterations") == 10
            and preconditioner.get("pock_chambolle_applied") is True
            and preconditioner.get("normalization_applied") is True
            and preconditioner.get("nnz_preserved") is True
            and _positive(preconditioner.get("b_scale"))
            and _positive(preconditioner.get("c_scale"))
            and len(row_range) == 2
            and len(column_range) == 2
            and all(_positive(item) for item in row_range + column_range)
        )
        details.append(f"{name}:periods={dimensions.get('periods')},n={dimensions.get('n')}")
    add_check(
        checks,
        "both_t1_and_t2_cases_have_exact_stage5_preconditioning",
        valid,
        "; ".join(details),
    )


def _structural_checks(
    checks: list[dict[str, Any]], cases: dict[str, dict[str, Any]], config: dict[str, Any]
) -> None:
    relative = float(_mapping(config.get("crosscheck"))["one_step_relative_tolerance"])
    absolute = float(_mapping(config.get("crosscheck"))["one_step_absolute_tolerance"])
    valid = bool(cases)
    details: list[str] = []
    for name, case in cases.items():
        preparation = _mapping(case.get("gpu_preparation"))
        structural = _mapping(case.get("unscaled_structural_parity"))
        state_comparisons = _mapping(structural.get("state_comparisons"))
        rejections = _mapping(structural.get("incompatible_pairing_rejections"))
        direct_rejection = _mapping(rejections.get("raw_descriptor_with_scaled_direct"))
        scaled_rejection = _mapping(rejections.get("raw_descriptor_with_scaled_lp"))
        rejection_valid = all(
            record.get("rejected") is True
            and record.get("error_type") == "ValueError"
            and bool(str(record.get("message", "")).strip())
            for record in (direct_rejection, scaled_rejection)
        )
        case_valid = (
            preparation.get("equality_mode") == "scaled_direct"
            and structural.get("equality_mode") == "unscaled_structural"
            and set(state_comparisons) == {"proximal", "reflected", "next_state"}
            and all(
                _state_comparison_valid(
                    comparison,
                    relative_tolerance=relative,
                    absolute_tolerance=absolute,
                )
                for comparison in state_comparisons.values()
            )
            and _comparison_valid(
                structural.get("first_sweep_comparison"),
                relative_tolerance=relative,
                absolute_tolerance=absolute,
            )
            and _comparison_valid(
                structural.get("diagnostic_comparison"),
                relative_tolerance=relative,
                absolute_tolerance=absolute,
            )
            and rejection_valid
        )
        valid = valid and case_valid
        details.append(
            f"{name}:production={preparation.get('equality_mode')},"
            f"structural={structural.get('equality_mode')},rejections={rejection_valid}"
        )
    add_check(
        checks,
        "scaled_direct_production_and_guarded_unscaled_structural_paths",
        valid,
        "; ".join(details),
    )


def _sparse_checks(
    checks: list[dict[str, Any]], cases: dict[str, dict[str, Any]], config: dict[str, Any]
) -> None:
    crosscheck = _mapping(config.get("crosscheck"))
    sparse_config = _mapping(config.get("sparse_benchmark"))
    relative = float(crosscheck["spmv_relative_tolerance"])
    absolute = float(crosscheck["spmv_absolute_tolerance"])
    kernels_valid = bool(cases)
    operators_valid = bool(cases)
    operator_count = 0
    for case in cases.values():
        operators = [
            _mapping(row)
            for row in _sequence(_mapping(case.get("sparse_crosschecks")).get("operators"))
        ]
        by_name = {str(row.get("operator")): row for row in operators}
        kernels_valid = kernels_valid and set(by_name) == {"A1", "A2"}
        operators_valid = operators_valid and set(by_name) == {"A1", "A2"}
        operator_count += len(by_name)
        for row in by_name.values():
            kernel = _mapping(row.get("kernel_selection"))
            kernels_valid = kernels_valid and _kernel_alg2_valid(kernel, tolerance=absolute)
            benchmark = _mapping(row.get("normal_transpose_explicit_transpose_benchmark"))
            timings_valid = all(
                _cuda_timing_valid(
                    benchmark.get(key),
                    repetitions=int(sparse_config["repetitions"]),
                    warmup=int(sparse_config["warmup_iterations"]),
                )
                for key in ("normal_csr", "transpose_flag", "explicit_csr_transpose")
            )
            operators_valid = operators_valid and (
                _comparison_valid(
                    row.get("normal_cpu_gpu_comparison"),
                    relative_tolerance=relative,
                    absolute_tolerance=absolute,
                )
                and _comparison_valid(
                    row.get("transpose_cpu_gpu_comparison"),
                    relative_tolerance=relative,
                    absolute_tolerance=absolute,
                )
                and benchmark.get("high_level_kernel_label") == DEFAULT_SPMV_LABEL
                and _nonnegative(benchmark.get("transpose_max_abs_difference"))
                and float(benchmark["transpose_max_abs_difference"]) <= absolute
                and timings_valid
            )
    add_check(
        checks,
        "fp64_a1_a2_actual_alg2_selection_and_repeatable_probes",
        kernels_valid and operator_count == 4,
        f"verified_operator_kernels={operator_count}",
    )
    add_check(
        checks,
        "csr_normal_transpose_and_explicit_transpose_gates",
        operators_valid and operator_count == 4,
        f"operators={operator_count}, public_benchmark_label={DEFAULT_SPMV_LABEL}",
    )


def _fixed_horizon_checks(
    checks: list[dict[str, Any]], cases: dict[str, dict[str, Any]], config: dict[str, Any]
) -> None:
    crosscheck = _mapping(config.get("crosscheck"))
    expected = {1, 10, int(crosscheck["fixed_horizon_iterations"])}
    valid = bool(cases)
    detail: list[str] = []
    for name, case in cases.items():
        runs = [_mapping(run) for run in _sequence(case.get("fixed_horizon_crosschecks"))]
        by_horizon = {run.get("iterations"): run for run in runs}
        valid = valid and set(by_horizon) == expected
        for horizon, run in by_horizon.items():
            if horizon == 1:
                relative = float(crosscheck["one_step_relative_tolerance"])
                absolute: float | None = float(crosscheck["one_step_absolute_tolerance"])
            elif horizon == 10:
                relative = float(crosscheck["ten_step_relative_tolerance"])
                absolute = float(crosscheck["ten_step_absolute_tolerance"])
            else:
                relative = float(crosscheck["fixed_horizon_state_relative_tolerance"])
                absolute = None
            valid = valid and (
                _state_comparison_valid(
                    run.get("original_state"),
                    relative_tolerance=relative,
                    absolute_tolerance=absolute,
                )
                and _state_comparison_valid(
                    run.get("scaled_state"),
                    relative_tolerance=relative,
                    absolute_tolerance=absolute,
                )
                and _gpu_loop_timing_valid(
                    run.get("gpu_timing"), residual_interval=1, residual_count=int(horizon)
                )
            )
        detail.append(f"{name}:horizons={sorted(by_horizon)}")
    add_check(
        checks,
        "one_ten_and_one_hundred_step_cpu_gpu_state_parity",
        valid,
        "; ".join(detail),
    )


def _determinism_and_resident_checks(
    checks: list[dict[str, Any]], cases: dict[str, dict[str, Any]], config: dict[str, Any]
) -> None:
    determinism_config = _mapping(config.get("determinism"))
    repetitions = int(determinism_config["repetitions"])
    horizon = int(determinism_config["fixed_horizon_iterations"])
    tolerance = float(determinism_config["relative_tolerance"])
    determinism_valid = bool(cases) and repetitions == 3
    resident_valid = bool(cases)
    for case in cases.values():
        determinism = _mapping(case.get("determinism"))
        runs = [_mapping(run) for run in _sequence(determinism.get("runs"))]
        comparisons = [_mapping(row) for row in _sequence(determinism.get("repeat_comparisons"))]
        determinism_valid = determinism_valid and (
            [run.get("repetition") for run in runs] == [1, 2, 3]
            and all(
                run.get("iterations") == horizon
                and _gpu_loop_timing_valid(
                    run.get("timing"), residual_interval=1, residual_count=horizon
                )
                for run in runs
            )
            and [comparison.get("repetition") for comparison in comparisons] == [2, 3]
            and all(
                _state_comparison_valid(
                    comparison.get("state"),
                    relative_tolerance=tolerance,
                    absolute_tolerance=None,
                )
                and comparison.get("policy_schedule_matches") is True
                and set(_mapping(comparison.get("bitwise_equal_by_block"))) == STATE_BLOCKS
                and comparison.get("bitwise_required") is False
                for comparison in comparisons
            )
        )

        resident = _mapping(case.get("resident_timing"))
        resident_iterations = int(config["resident_timing_iterations"])
        interval = int(config["resident_timing_residual_check_interval"])
        expected_count = math.ceil(resident_iterations / interval)
        resident_valid = resident_valid and (
            resident.get("fixed_horizon") == resident_iterations == 1000
            and resident.get("no_speedup_claim") is True
            and _gpu_loop_timing_valid(
                resident.get("timing"),
                residual_interval=interval,
                residual_count=expected_count,
            )
        )
    add_check(
        checks,
        "three_run_fp64_state_and_policy_determinism",
        determinism_valid,
        f"repetitions={repetitions}, fixed_horizon={horizon}",
    )
    add_check(
        checks,
        "resident_one_thousand_step_cadence_and_cuda_timing",
        resident_valid,
        "1000 resident iterations with one compact residual check per 100 iterations",
    )


def _solver_ledgers(
    evidence: dict[str, Any], cases: dict[str, dict[str, Any]]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name, case in cases.items():
        for run in _sequence(case.get("fixed_horizon_crosschecks")):
            row = _mapping(run)
            result.append(
                (
                    f"{name}/fixed_{row.get('iterations')}",
                    _mapping(row.get("transfer_ledger")),
                    _mapping(row.get("transfer_timing")),
                )
            )
        for run in _sequence(_mapping(case.get("determinism")).get("runs")):
            row = _mapping(run)
            result.append(
                (
                    f"{name}/determinism_{row.get('repetition')}",
                    _mapping(row.get("transfer_ledger")),
                    _mapping(row.get("transfer_timing")),
                )
            )
        resident = _mapping(case.get("resident_timing"))
        full = _mapping(case.get("gpu_full_fp64"))
        result.append(
            (
                f"{name}/resident",
                _mapping(resident.get("transfer_ledger")),
                _mapping(resident.get("transfer_timing")),
            )
        )
        result.append(
            (
                f"{name}/full",
                _mapping(full.get("transfer_ledger")),
                _mapping(full.get("transfer_timing")),
            )
        )
        warmup = _mapping(case.get("warmup"))
        if "transfer_ledger" in warmup:
            result.append(
                (
                    f"{name}/warmup",
                    _mapping(warmup.get("transfer_ledger")),
                    _mapping(warmup.get("transfer_timing")),
                )
            )
    fp32 = _mapping(_mapping(evidence.get("precision_study")).get("fp32"))
    for row_value in _sequence(fp32.get("cases")):
        row = _mapping(row_value)
        if "transfer_ledger" in row:
            result.append(
                (
                    f"{row.get('case')}/fp32",
                    _mapping(row.get("transfer_ledger")),
                    _mapping(row.get("transfer_timing")),
                )
            )
    return result


def _transfer_checks(
    checks: list[dict[str, Any]], evidence: dict[str, Any], cases: dict[str, dict[str, Any]]
) -> None:
    ledgers = _solver_ledgers(evidence, cases)
    valid = bool(ledgers) and all(
        _ledger_valid(ledger, solver=True) and _transfer_timing_valid(timing)
        for _, ledger, timing in ledgers
    )
    preparation_valid = bool(cases)
    for case in cases.values():
        preparation = _mapping(case.get("gpu_preparation"))
        ledger = _mapping(preparation.get("transfer_ledger"))
        rows = [_mapping(row) for row in _sequence(ledger.get("records"))]
        preparation_valid = preparation_valid and (
            _ledger_valid(ledger, solver=False)
            and _transfer_timing_valid(preparation.get("transfer_timing"))
            and all(
                row.get("direction") == "host_to_device"
                or (
                    row.get("direction") == "device_to_host"
                    and row.get("phase") == "sparse_kernel_probe"
                    and row.get("kind") == "scalar"
                )
                for row in rows
            )
        )
    add_check(
        checks,
        "solver_transfer_phase_direction_and_device_residency_audit",
        valid and preparation_valid,
        (
            f"solver_ledgers={len(ledgers)}; only compact 10/2/3-scalar packets "
            "cross during loops; full states cross only at final_state phases"
        ),
    )


def _original_acceptance_valid(summary: Any, config: dict[str, Any]) -> bool:
    run = _mapping(summary)
    residuals = _mapping(run.get("original_residuals"))
    normalized = _mapping(residuals.get("paper_normalized_norms"))
    stopping = _mapping(residuals.get("paper_stopping"))
    physical = _mapping(run.get("physical_validation"))
    families = [_mapping(item) for item in _sequence(physical.get("families"))]
    paper_tolerance = float(config["paper_tolerance"])
    physical_tolerance = float(config["dcopf_physical_tolerance"])
    return (
        run.get("converged") is True
        and isinstance(run.get("iterations"), int)
        and int(run["iterations"]) > 0
        and residuals.get("tolerance") == paper_tolerance
        and set(normalized) == RESIDUAL_BLOCKS
        and all(
            _nonnegative(value) and float(value) <= paper_tolerance for value in normalized.values()
        )
        and stopping
        == {
            "primal_feasibility": True,
            "box": True,
            "stationarity": True,
            "all_satisfied": True,
        }
        and _nonnegative(residuals.get("kkt_combined_norm"))
        and float(residuals["kkt_combined_norm"]) <= float(config["dcopf_kkt_combined_target"])
        and _nonnegative(run.get("scaled_objective_gap_to_highs"))
        and float(run["scaled_objective_gap_to_highs"])
        <= float(config["dcopf_maximum_scaled_objective_gap"])
        and _nonnegative(run.get("maximum_physical_violation"))
        and float(run["maximum_physical_violation"]) <= physical_tolerance
        and _nonnegative(run.get("maximum_canonical_primal_violation"))
        and float(run["maximum_canonical_primal_violation"]) <= physical_tolerance
        and physical.get("passed") is True
        and bool(families)
        and all(
            _nonnegative(family.get("maximum_violation"))
            and float(family["maximum_violation"]) <= physical_tolerance
            for family in families
        )
        and _nonnegative(run.get("maximum_equality_solve_infinity_residual"))
        and float(run["maximum_equality_solve_infinity_residual"])
        <= float(config["maximum_equality_infinity_residual"])
        and _nonnegative(run.get("maximum_z_x_identity_error"))
        and float(run["maximum_z_x_identity_error"]) <= float(config["maximum_z_x_identity_error"])
        and _finite(run.get("objective"))
        and _finite(run.get("reference_objective"))
        and _positive(_mapping(run.get("sigma")).get("final"))
    )


def _acceptance_and_parity_checks(
    checks: list[dict[str, Any]], cases: dict[str, dict[str, Any]], config: dict[str, Any]
) -> None:
    acceptance_valid = bool(cases)
    parity_valid = bool(cases)
    details: list[str] = []
    crosscheck = _mapping(config.get("crosscheck"))
    for name, case in cases.items():
        cpu = _mapping(case.get("cpu_full_oracle"))
        gpu = _mapping(case.get("gpu_full_fp64"))
        acceptance_valid = (
            acceptance_valid
            and _original_acceptance_valid(cpu, config)
            and _original_acceptance_valid(gpu, config)
        )

        cpu_residuals = _mapping(
            _mapping(cpu.get("original_residuals")).get("paper_normalized_norms")
        )
        gpu_residuals = _mapping(
            _mapping(gpu.get("original_residuals")).get("paper_normalized_norms")
        )
        residual_differences_valid = (
            set(cpu_residuals) == RESIDUAL_BLOCKS
            and set(gpu_residuals) == RESIDUAL_BLOCKS
            and all(
                abs(float(gpu_residuals[key]) - float(cpu_residuals[key]))
                <= float(crosscheck["normalized_residual_absolute_tolerance"])
                for key in RESIDUAL_BLOCKS
            )
        )
        objective_gap = abs(
            float(gpu.get("objective", math.inf)) - float(cpu.get("objective", -math.inf))
        ) / max(1.0, abs(float(cpu.get("objective", 0.0))))
        comparison = _mapping(case.get("cpu_gpu_full_comparison"))
        policy_matches = _sequence(cpu.get("policy_schedule")) == _sequence(
            gpu.get("policy_schedule")
        )
        parity_valid = parity_valid and (
            objective_gap <= float(crosscheck["objective_relative_tolerance"])
            and residual_differences_valid
            and policy_matches
            and _comparison_valid(
                comparison.get("objective"),
                relative_tolerance=float(crosscheck["objective_relative_tolerance"]),
                absolute_tolerance=None,
            )
            and _comparison_valid(
                comparison.get("normalized_residuals"),
                relative_tolerance=None,
                absolute_tolerance=float(crosscheck["normalized_residual_absolute_tolerance"]),
            )
        )
        details.append(
            f"{name}:cpu_kkt={_mapping(cpu.get('original_residuals')).get('kkt_combined_norm')},"
            f"gpu_kkt={_mapping(gpu.get('original_residuals')).get('kkt_combined_norm')}"
        )
    add_check(
        checks,
        "original_space_eq54_kkt_physics_objective_and_sweep_invariants",
        acceptance_valid,
        "; ".join(details),
    )
    add_check(
        checks,
        "cpu_gpu_policy_objective_and_normalized_residual_parity",
        parity_valid,
        "objective and residual differences recomputed; policy schedules compared directly",
    )


def _timing_boundary_checks(
    checks: list[dict[str, Any]], evidence: dict[str, Any], config: dict[str, Any]
) -> None:
    section = _mapping(evidence.get("timing_boundaries"))
    boundaries = _mapping(section.get("boundaries"))
    expected = _sequence(config.get("timing_boundaries"))
    valid = set(boundaries) == set(expected) and len(boundaries) == len(expected)
    for name in expected:
        record = _mapping(boundaries.get(name))
        seconds = record.get("seconds")
        if name == "allocation":
            allocation_method = str(record.get("method", "")).lower()
            valid = valid and (
                seconds is None
                and record.get("unit") == "seconds"
                and allocation_method.startswith("not independently timed")
                and "initialization" in allocation_method
                and record.get("status") == "reported but not independently timed"
            )
        else:
            valid = valid and (
                _nonnegative(seconds)
                and record.get("unit") == "seconds"
                and bool(str(record.get("method", "")).strip())
                and "measured" in str(record.get("status", "")).lower()
            )
    policy = str(section.get("synchronization_policy", "")).lower()
    valid = valid and "cuda-event" in policy and "synchronized" in policy
    valid = valid and section.get("gpu_speedup_claimed") is False
    add_check(
        checks,
        "all_required_timing_boundaries_have_units_and_truthful_methods",
        valid,
        (
            f"boundaries={list(boundaries)}, "
            f"synchronization_policy={section.get('synchronization_policy')}"
        ),
    )


def _precision_checks(
    checks: list[dict[str, Any]], evidence: dict[str, Any], cases: dict[str, dict[str, Any]]
) -> None:
    study = _mapping(evidence.get("precision_study"))
    fp64 = _mapping(study.get("fp64"))
    fp32 = _mapping(study.get("fp32"))
    fp32_cases = [_mapping(case) for case in _sequence(fp32.get("cases"))]
    by_name = {str(case.get("case")): case for case in fp32_cases}
    optional_cases_valid = set(by_name) == set(cases)
    for case in by_name.values():
        if case.get("status") == "ERROR":
            optional_cases_valid = optional_cases_valid and case.get("gating") is False
            continue
        a1 = _mapping(case.get("A1_kernel_selection"))
        a2 = _mapping(case.get("A2_kernel_selection"))
        optional_cases_valid = optional_cases_valid and (
            case.get("precision") == "float32"
            and case.get("gating") is False
            and case.get("mixed_precision") is False
            and case.get("iterations") == 100
            and _mapping(case.get("state_comparison_to_cpu_fp64")).get("gating") is False
            and a1.get("uses_csr_alg2") is False
            and a2.get("uses_csr_alg2") is False
            and bool(a1.get("fallback_reason"))
            and bool(a2.get("fallback_reason"))
        )
    valid = (
        fp64 == {"status": "PASS", "passed": True, "gating": True}
        and fp32.get("status") in {"COMPLETED", "COMPLETED_WITH_DIAGNOSTIC_FAILURES"}
        and fp32.get("gating") is False
        and fp32.get("mixed_precision") is False
        and fp32.get("ran_after_fp64_pass") is True
        and optional_cases_valid
        and all(
            _mapping(failure).get("gating") is False
            for failure in _sequence(evidence.get("non_gating_failures"))
        )
    )
    add_check(
        checks,
        "fp32_is_non_gating_after_fp64_and_mixed_precision_is_disabled",
        valid,
        f"fp64={fp64.get('status')}, fp32={fp32.get('status')}, cases={sorted(by_name)}",
    )


def _expected_trajectory_groups(
    cases: dict[str, dict[str, Any]], evidence: dict[str, Any]
) -> dict[tuple[str, str, str, str], int | None]:
    expected: dict[tuple[str, str, str, str], int | None] = {}
    for name, case in cases.items():
        cpu_full = _mapping(case.get("cpu_full_oracle"))
        gpu_full = _mapping(case.get("gpu_full_fp64"))
        expected[(name, "full_fp64_correctness", "cpu_oracle", "float64")] = cpu_full.get(
            "iterations"
        )
        expected[(name, "full_fp64_correctness", "dgx_gpu", "float64")] = gpu_full.get("iterations")
        for horizon in (1, 10, 100):
            run = f"fixed_{horizon}_step_fp64_crosscheck"
            expected[(name, run, "cpu_oracle", "float64")] = horizon
            expected[(name, run, "dgx_gpu", "float64")] = horizon
        for repetition in (1, 2, 3):
            expected[(name, f"determinism_fp64_repetition_{repetition}", "dgx_gpu", "float64")] = (
                100
            )
        expected[(name, "resident_1000_step_timing", "dgx_gpu", "float64")] = 1000
    fp32 = _mapping(_mapping(evidence.get("precision_study")).get("fp32"))
    for case_value in _sequence(fp32.get("cases")):
        case = _mapping(case_value)
        if case.get("status") != "ERROR":
            expected[
                (str(case.get("case")), "optional_fp32_fixed_horizon", "dgx_gpu", "float32")
            ] = case.get("iterations")
    return expected


def _trajectory_row_valid(row: dict[str, Any]) -> bool:
    if (
        row.get("record_type") != "trajectory"
        or row.get("execution") not in {"cpu_oracle", "dgx_gpu"}
        or row.get("precision") not in {"float64", "float32"}
        or not isinstance(row.get("iteration"), int)
        or isinstance(row.get("iteration"), bool)
        or int(row["iteration"]) <= 0
        or not isinstance(row.get("inner_iteration"), int)
        or int(row["inner_iteration"]) <= 0
        or not _positive(row.get("sigma"))
        or not isinstance(row.get("restart_count"), int)
    ):
        return False
    if row.get("execution") == "dgx_gpu":
        residuals = _mapping(row.get("residuals"))
        objectives = (row.get("scaled_objective"), row.get("original_objective"))
    else:
        residuals = _mapping(row.get("original_residuals"))
        objectives = (
            row.get("scaled_variable_objective"),
            row.get("original_variable_objective"),
        )
    normalized = _mapping(residuals.get("paper_normalized"))
    raw = _mapping(residuals.get("paper_raw"))
    return (
        set(normalized) == RESIDUAL_BLOCKS
        and set(raw) == RESIDUAL_BLOCKS
        and all(_nonnegative(value) for value in (*normalized.values(), *raw.values()))
        and _nonnegative(residuals.get("kkt_combined_norm"))
        and all(_finite(value) for value in objectives)
    )


def _policy_row_valid(row: dict[str, Any], *, interval: int) -> bool:
    update = _mapping(row.get("sigma_update"))
    return (
        row.get("record_type") == "policy_event"
        and row.get("execution") in {"cpu_oracle", "dgx_gpu"}
        and row.get("precision") == "float64"
        and isinstance(row.get("iteration"), int)
        and int(row["iteration"]) > 0
        and int(row["iteration"]) % interval == 0
        and isinstance(row.get("inner_iteration"), int)
        and int(row["inner_iteration"]) > 0
        and _nonnegative(row.get("merit"))
        and _nonnegative(row.get("reference_merit"))
        and isinstance(row.get("restart_count"), int)
        and update.get("attempted") in {True, False}
        and update.get("accepted") in {True, False}
        and _positive(update.get("sigma_before"))
        and _positive(update.get("sigma_after"))
    )


def _trajectory_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    config: dict[str, Any],
    trajectory_path: Path,
) -> None:
    rows, load_error = _load_gzip_jsonl(trajectory_path)
    evidence_files = _mapping(evidence.get("evidence_files"))
    hash_valid = (
        load_error is None
        and bool(rows)
        and evidence_files.get("trajectory_sha256") == _sha256(trajectory_path)
        and evidence_files.get("trajectories_and_policy_events") == trajectory_path.name
    )
    add_check(
        checks,
        "trajectory_gzip_integrity_and_sha256",
        hash_valid,
        (
            f"rows={len(rows)}, sha256={_sha256(trajectory_path)}"
            if load_error is None
            else load_error
        ),
    )

    trajectories: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    policies: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rows_valid = hash_valid
    interval = int(config.get("policy_check_interval", -1))
    for row in rows:
        key = (
            str(row.get("case")),
            str(row.get("run")),
            str(row.get("execution")),
            str(row.get("precision")),
        )
        if row.get("record_type") == "trajectory":
            trajectories[key].append(row)
            rows_valid = rows_valid and _trajectory_row_valid(row)
        elif row.get("record_type") == "policy_event":
            policies[key].append(row)
            rows_valid = rows_valid and _policy_row_valid(row, interval=interval)
        else:
            rows_valid = False

    expected = _expected_trajectory_groups(cases, evidence)
    coverage_valid = (
        rows_valid and set(trajectories) == set(expected) and set(policies).issubset(expected)
    )
    policy_schedules: dict[tuple[str, str, str, str], list[tuple[Any, ...]]] = {}
    for key, final_iteration in expected.items():
        group = trajectories.get(key, [])
        iterations = [row.get("iteration") for row in group]
        coverage_valid = coverage_valid and (
            bool(group)
            and iterations == sorted(set(iterations))
            and isinstance(final_iteration, int)
            and iterations[-1] == final_iteration
        )
        event_group = policies.get(key, [])
        event_iterations = [row.get("iteration") for row in event_group]
        coverage_valid = coverage_valid and event_iterations == sorted(set(event_iterations))
        policy_schedules[key] = [
            (
                row.get("iteration"),
                row.get("inner_iteration"),
                row.get("restarted"),
                tuple(_sequence(row.get("restart_reasons"))),
            )
            for row in event_group
        ]

    for name in cases:
        cpu_key = (name, "full_fp64_correctness", "cpu_oracle", "float64")
        gpu_key = (name, "full_fp64_correctness", "dgx_gpu", "float64")
        coverage_valid = coverage_valid and policy_schedules.get(cpu_key) == policy_schedules.get(
            gpu_key
        )
        for repetition in (2, 3):
            baseline = (name, "determinism_fp64_repetition_1", "dgx_gpu", "float64")
            repeated = (
                name,
                f"determinism_fp64_repetition_{repetition}",
                "dgx_gpu",
                "float64",
            )
            coverage_valid = coverage_valid and policy_schedules.get(
                baseline
            ) == policy_schedules.get(repeated)
    add_check(
        checks,
        "trajectory_rows_cover_all_cases_runs_and_policy_schedules",
        coverage_valid,
        (
            f"trajectory_groups={len(trajectories)}, expected={len(expected)}, "
            f"policy_groups={len(policies)}"
        ),
    )


def _stage_boundary_checks(
    checks: list[dict[str, Any]], evidence: dict[str, Any], config: dict[str, Any]
) -> None:
    evidence_boundary = _mapping(evidence.get("stage_boundary"))
    config_boundary = _mapping(config.get("stage_boundary"))
    valid = (
        config_boundary
        == {
            "stage_6_only": True,
            "stage_7_benchmarks_locked": True,
            "paper_timing_reproduction_claimed": False,
            "gpu_speedup_claimed": False,
        }
        and evidence_boundary
        == {
            "stage_6_only": True,
            "stage_6_executed": True,
            "stage_6_complete": True,
            "stage_7_benchmarks_locked": True,
            "paper_timing_reproduction_claimed": False,
            "gpu_speedup_claimed": False,
        }
        and _mapping(evidence.get("timing_boundaries")).get("gpu_speedup_claimed") is False
        and not _sequence(evidence.get("failures"))
    )
    add_check(
        checks,
        "stage_seven_locked_and_no_timing_or_speedup_claim",
        valid,
        f"boundary={evidence_boundary}",
    )


def run_checks(
    evidence_path: Path = DEFAULT_EVIDENCE,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    config, config_error = _load_json(config_path)
    evidence, evidence_error = _load_json(evidence_path)
    evidence_files = _mapping(evidence.get("evidence_files"))
    trajectory_path = evidence_path.parent / str(
        evidence_files.get("trajectories_and_policy_events", DEFAULT_TRAJECTORY_NAME)
    )
    required_paths = (
        DEFAULT_NETWORK,
        *DEFAULT_DCOPF_CONFIGS,
        DEFAULT_REQUIREMENTS,
        config_path,
        evidence_path,
        trajectory_path,
        PROJECT_ROOT / "src" / "gpu_dcopf_hpr" / "gpu_backend.py",
        PROJECT_ROOT / "src" / "gpu_dcopf_hpr" / "gpu_sparse.py",
        PROJECT_ROOT / "src" / "gpu_dcopf_hpr" / "gpu_sgs_hpr.py",
        PROJECT_ROOT / "src" / "gpu_dcopf_hpr" / "gpu_stage5_control.py",
        PROJECT_ROOT / "scripts" / "run_stage_6.py",
        PROJECT_ROOT / "scripts" / "check_stage_6.py",
        PROJECT_ROOT / "tests" / "integration" / "test_stage6_evidence.py",
    )
    missing = [_display_path(path) for path in required_paths if not path.is_file()]
    add_check(
        checks,
        "required_stage_six_paths",
        not missing,
        "complete" if not missing else f"missing={missing}",
    )
    add_check(
        checks,
        "configuration_is_valid_json",
        config_error is None,
        "loaded" if config_error is None else str(config_error),
    )
    add_check(
        checks,
        "validation_is_valid_json",
        evidence_error is None,
        "loaded" if evidence_error is None else str(evidence_error),
    )

    cases = _case_map(evidence)
    _configuration_checks(checks, evidence, config, config_path)
    _device_checks(checks, evidence, config)
    _case_coverage_checks(checks, cases, config)
    _structural_checks(checks, cases, config)
    _sparse_checks(checks, cases, config)
    _fixed_horizon_checks(checks, cases, config)
    _determinism_and_resident_checks(checks, cases, config)
    _transfer_checks(checks, evidence, cases)
    _acceptance_and_parity_checks(checks, cases, config)
    _timing_boundary_checks(checks, evidence, config)
    _precision_checks(checks, evidence, cases)
    _trajectory_checks(checks, evidence, cases, config, trajectory_path)
    _stage_boundary_checks(checks, evidence, config)

    totals: Counter[str] = Counter()
    for check in checks:
        totals["passed" if check["passed"] else "failed"] += 1
    return {
        "stage": 6,
        "passed": totals["failed"] == 0,
        "configuration": _display_path(config_path),
        "evidence": _display_path(evidence_path),
        "evidence_files": {
            "trajectories_and_policy_events": _display_path(trajectory_path),
        },
        "summary": dict(totals),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_checks(args.evidence.resolve(), args.config.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
