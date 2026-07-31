"""Run the preregistered Stage 6 FP64 GPU validation on the DGX Spark.

This runner preserves the Stage 5 CPU solver as the oracle, records truthful
CSR kernel selection and transfer boundaries, and writes partial evidence even
when a later phase fails.  It is a correctness run, not a Stage 7 benchmark or
a GPU speedup experiment.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

import numpy as np
import scipy
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gpu_dcopf_hpr.dcopf_model import (  # noqa: E402
    DCOPFModel,
    build_dcopf_model,
    load_dcopf_config,
)
from gpu_dcopf_hpr.gpu_backend import (  # noqa: E402
    CuPyBackend,
    create_gpu_backend,
)
from gpu_dcopf_hpr.gpu_sgs_hpr import (  # noqa: E402
    GPUHPRState,
    gpu_sgs_hpr_step,
    prepare_gpu_sgs_hpr,
)
from gpu_dcopf_hpr.gpu_stage5_control import (  # noqa: E402
    GPUStage6Problem,
    GPUStage6Result,
    prepare_gpu_stage6_problem,
    solve_gpu_stage5_sgs_hpr,
)
from gpu_dcopf_hpr.hpr_generic import HPRState  # noqa: E402
from gpu_dcopf_hpr.network_data import load_matpower_case  # noqa: E402
from gpu_dcopf_hpr.preconditioning import LPPreconditioner, precondition_lp  # noqa: E402
from gpu_dcopf_hpr.sgs_hpr import prepare_sgs_hpr, sgs_hpr_step  # noqa: E402
from gpu_dcopf_hpr.stage5_control import (  # noqa: E402
    Stage5Control,
    Stage5SGSHPRResult,
    solve_stage5_sgs_hpr,
)
from gpu_dcopf_hpr.structural_y1 import prepare_dcopf_structural_y1  # noqa: E402
from gpu_dcopf_hpr.validation import (  # noqa: E402
    maximum_primal_violation,
    solve_with_highs,
    validate_dcopf_candidate,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sgs_hpr" / "stage_6_gpu_dgx.json"
DEFAULT_NETWORK = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
DEFAULT_DCOPF_CONFIGS = (
    PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json",
    PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_6"
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "environment" / "dgx_stage6_requirements.txt"


@dataclass(frozen=True, slots=True)
class _CaseRuntime:
    name: str
    model: DCOPFModel
    preconditioner: LPPreconditioner
    inequality_lambda: float | None
    cpu_fixed_100: Stage5SGSHPRResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument(
        "--dcopf-configs",
        type=Path,
        nargs=2,
        default=DEFAULT_DCOPF_CONFIGS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def _clean_json(value: Any) -> Any:
    """Convert evidence recursively to strict JSON without hiding nonfinite data."""

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0.0 else "-Infinity"
        return value
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def _compare_arrays(
    candidate: Any,
    reference: Any,
    *,
    relative_tolerance: float | None,
    absolute_tolerance: float | None,
) -> dict[str, Any]:
    """Compare two numeric arrays using only explicitly supplied gates."""

    actual = np.asarray(candidate, dtype=np.float64)
    expected = np.asarray(reference, dtype=np.float64)
    shape_matches = actual.shape == expected.shape
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)))
    if shape_matches and finite:
        difference = actual - expected
        relative_error = float(
            np.linalg.norm(difference) / max(1.0, float(np.linalg.norm(expected)))
        )
        maximum_absolute_error = float(np.max(np.abs(difference), initial=0.0))
    else:
        relative_error = math.inf
        maximum_absolute_error = math.inf
    relative_passed = relative_tolerance is None or relative_error <= relative_tolerance
    absolute_passed = absolute_tolerance is None or maximum_absolute_error <= absolute_tolerance
    return {
        "candidate_shape": list(actual.shape),
        "reference_shape": list(expected.shape),
        "shape_matches": shape_matches,
        "finite": finite,
        "relative_error": relative_error,
        "maximum_absolute_error": maximum_absolute_error,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "passed": bool(shape_matches and finite and relative_passed and absolute_passed),
    }


def _compare_states(
    candidate: HPRState,
    reference: HPRState,
    *,
    relative_tolerance: float | None,
    absolute_tolerance: float | None,
) -> dict[str, Any]:
    blocks = {
        name: _compare_arrays(
            getattr(candidate, name),
            getattr(reference, name),
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        for name in ("x", "y", "z")
    }
    return {
        "blocks": blocks,
        "maximum_relative_error": max(float(block["relative_error"]) for block in blocks.values()),
        "maximum_absolute_error": max(
            float(block["maximum_absolute_error"]) for block in blocks.values()
        ),
        "passed": all(bool(block["passed"]) for block in blocks.values()),
    }


def _scaled_gap(candidate: float, reference: float) -> float:
    return abs(float(candidate) - float(reference)) / max(1.0, abs(float(reference)))


def _ledger_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Subtract two cumulative transfer-ledger summaries."""

    def index(summary: dict[str, Any]) -> dict[tuple[str, str, str], tuple[int, int]]:
        return {
            (str(row["phase"]), str(row["direction"]), str(row["kind"])): (
                int(row["calls"]),
                int(row["bytes"]),
            )
            for row in summary.get("records", [])
        }

    first = index(before)
    second = index(after)
    records: list[dict[str, Any]] = []
    for key in sorted(set(first) | set(second)):
        old_calls, old_bytes = first.get(key, (0, 0))
        new_calls, new_bytes = second.get(key, (0, 0))
        calls = new_calls - old_calls
        byte_count = new_bytes - old_bytes
        if calls < 0 or byte_count < 0:
            raise ValueError("transfer ledgers must be cumulative and monotone")
        if calls or byte_count:
            records.append(
                {
                    "phase": key[0],
                    "direction": key[1],
                    "kind": key[2],
                    "calls": calls,
                    "bytes": byte_count,
                }
            )
    totals = {
        direction: {
            "calls": sum(int(row["calls"]) for row in records if row["direction"] == direction),
            "bytes": sum(int(row["bytes"]) for row in records if row["direction"] == direction),
        }
        for direction in ("host_to_device", "device_to_host")
    }
    return {"records": records, "totals": totals}


def _timing_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    result = {key: float(after[key]) - float(before.get(key, 0.0)) for key in after}
    if any(value < 0.0 for value in result.values()):
        raise ValueError("transfer timing summaries must be cumulative and monotone")
    return result


def _audit_solver_transfers(delta: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        ("initial_state", "host_to_device"),
        ("periodic_diagnostics", "device_to_host"),
        ("policy_diagnostics", "device_to_host"),
        ("final_state", "device_to_host"),
        ("final_scaled_state", "device_to_host"),
        ("final_diagnostics", "device_to_host"),
    }
    unexpected = [
        row
        for row in delta.get("records", [])
        if (str(row["phase"]), str(row["direction"])) not in allowed
    ]
    return {
        "allowed_phase_directions": [list(item) for item in sorted(allowed)],
        "unexpected_records": unexpected,
        "passed": not unexpected,
    }


def _validate_stage6_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        if int(config["stage"]) != 6:
            errors.append("stage must equal 6")
        if config["precision"] != "float64":
            errors.append("the required correctness precision must be float64")
        if config["gpu_backend"] != "CuPy":
            errors.append("the Stage 6 GPU backend must be CuPy")
        boundary = config["stage_boundary"]
        if not boundary["stage_6_only"]:
            errors.append("stage_6_only must remain true")
        if not boundary["stage_7_benchmarks_locked"]:
            errors.append("Stage 7 benchmarks must remain locked")
        if boundary["paper_timing_reproduction_claimed"]:
            errors.append("paper timing reproduction must remain unclaimed")
        if boundary["gpu_speedup_claimed"]:
            errors.append("GPU speedup must remain unclaimed")
        precision = config["precision_study"]
        if not precision["fp64_required"]:
            errors.append("FP64 must remain required")
        if precision["mixed_precision_enabled"]:
            errors.append("mixed precision must remain disabled")
        if precision["reduced_precision_is_gating"]:
            errors.append("reduced precision must remain non-gating")
        for key in (
            "maximum_iterations",
            "policy_check_interval",
            "correctness_residual_check_interval",
            "resident_timing_residual_check_interval",
            "resident_timing_iterations",
            "history_interval",
        ):
            if (
                not isinstance(config[key], int)
                or isinstance(config[key], bool)
                or config[key] <= 0
            ):
                errors.append(f"{key} must be a positive integer")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"configuration structure error: {error}")
    return errors


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    def command(*arguments: str) -> tuple[int, str]:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode, completed.stdout.strip()

    commit_code, commit = command("rev-parse", "HEAD")
    branch_code, branch = command("rev-parse", "--abbrev-ref", "HEAD")
    status_code, status = command("status", "--porcelain")
    return {
        "commit": commit if commit_code == 0 else "unavailable",
        "branch": branch if branch_code == 0 else "unavailable",
        "dirty": bool(status) if status_code == 0 else None,
        "status_entries": status.splitlines() if status_code == 0 else [],
    }


def _package_versions() -> dict[str, str | None]:
    names = (
        "cupy-cuda13x",
        "cuda-pathfinder",
        "numpy",
        "scipy",
        "pytest",
        "ruff",
        "pip",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _cuda_library_versions(backend: CuPyBackend) -> dict[str, Any]:
    cp = backend.cp
    values: dict[str, Any] = {}
    probes = {
        "cusparse": lambda: cp.cuda.cusparse.getVersion(),
        "cublas": lambda: cp.cuda.cublas.getVersion(),
        "cusolver": lambda: cp.cuda.cusolver.getVersion(),
    }
    for name, probe in probes.items():
        try:
            values[name] = int(probe())
        except Exception as error:
            values[name] = {
                "status": "unavailable",
                "error_type": type(error).__name__,
                "message": str(error),
            }
    return values


def _exception_record(phase: str, error: BaseException) -> dict[str, Any]:
    return {
        "phase": phase,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__)[-30:],
    }


def _write_json(path: Path, evidence: dict[str, Any]) -> None:
    evidence["last_updated_utc"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_clean_json(evidence), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(stream: TextIO, row: dict[str, Any]) -> None:
    stream.write(json.dumps(_clean_json(row), sort_keys=True, allow_nan=False) + "\n")


def _write_cpu_rows(
    stream: TextIO,
    *,
    case_name: str,
    run_name: str,
    result: Stage5SGSHPRResult,
) -> None:
    for entry in result.history:
        _write_jsonl(
            stream,
            {
                "record_type": "trajectory",
                "execution": "cpu_oracle",
                "precision": "float64",
                "case": case_name,
                "run": run_name,
                **entry.as_dict(),
            },
        )
    for event in result.policy_events:
        _write_jsonl(
            stream,
            {
                "record_type": "policy_event",
                "execution": "cpu_oracle",
                "precision": "float64",
                "case": case_name,
                "run": run_name,
                **event.as_dict(),
            },
        )


def _write_gpu_rows(
    stream: TextIO,
    *,
    case_name: str,
    run_name: str,
    precision: str,
    result: GPUStage6Result,
) -> None:
    for entry in result.history:
        _write_jsonl(
            stream,
            {
                "record_type": "trajectory",
                "execution": "dgx_gpu",
                "precision": precision,
                "case": case_name,
                "run": run_name,
                **entry.as_dict(),
            },
        )
    for event in result.policy_events:
        _write_jsonl(
            stream,
            {
                "record_type": "policy_event",
                "execution": "dgx_gpu",
                "precision": precision,
                "case": case_name,
                "run": run_name,
                **event.as_dict(),
            },
        )
    stream.flush()


def _control(config: dict[str, Any], *, adaptive_sigma: bool, restart: bool) -> Stage5Control:
    restart_parameters = config["restart_parameters"]
    guards = config["sigma_guards"]
    return Stage5Control(
        adaptive_sigma=adaptive_sigma,
        restart=restart,
        check_interval=int(config["policy_check_interval"]),
        alpha_sufficient=float(restart_parameters["alpha_sufficient"]),
        alpha_necessary=float(restart_parameters["alpha_necessary"]),
        alpha_long=float(restart_parameters["alpha_long"]),
        movement_minimum=float(guards["movement_minimum"]),
        movement_maximum=float(guards["movement_maximum"]),
        infeasibility_ratio_minimum=float(guards["infeasibility_ratio_minimum"]),
        infeasibility_ratio_maximum=float(guards["infeasibility_ratio_maximum"]),
    )


def _preconditioner_summary(value: LPPreconditioner) -> dict[str, Any]:
    diagnostics = value.diagnostics
    return {
        "ruiz_iterations": diagnostics.ruiz_iterations,
        "pock_chambolle_applied": diagnostics.pock_chambolle_applied,
        "normalization_applied": diagnostics.normalization_applied,
        "nnz_preserved": diagnostics.nnz_preserved,
        "b_scale": value.b_scale,
        "c_scale": value.c_scale,
        "row_denominator_range": [
            float(np.min(value.row_denominator)),
            float(np.max(value.row_denominator)),
        ],
        "column_denominator_range": [
            float(np.min(value.column_denominator)),
            float(np.max(value.column_denominator)),
        ],
    }


def _policy_schedule(events: Any) -> list[dict[str, Any]]:
    return [
        {
            "iteration": int(event.iteration),
            "inner_iteration": int(event.inner_iteration),
            "restarted": bool(event.restarted),
            "restart_reasons": list(event.restart_reasons),
        }
        for event in events
    ]


def _solution_summary(
    result: Stage5SGSHPRResult | GPUStage6Result,
    *,
    model: DCOPFModel,
    reference_objective: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    objective = model.objective(result.solution.x)
    physical = validate_dcopf_candidate(
        model,
        result.solution.x,
        tolerance=float(config["dcopf_physical_tolerance"]),
    )
    maximum_physical = max(
        (family.maximum_violation for family in physical.families),
        default=0.0,
    )
    canonical_violation = maximum_primal_violation(model.lp, result.solution)
    values_finite = all(
        np.all(np.isfinite(block))
        for block in (result.solution.x, result.solution.y, result.solution.z)
    )
    checks = {
        "converged": bool(result.converged),
        "original_paper_stopping_satisfied": bool(result.residuals.conditions.all_satisfied),
        "original_kkt_target_satisfied": (
            result.residuals.combined_norm <= float(config["dcopf_kkt_combined_target"])
        ),
        "objective_matches_highs": (
            _scaled_gap(objective, reference_objective)
            <= float(config["dcopf_maximum_scaled_objective_gap"])
        ),
        "physical_candidate_valid": bool(physical.passed),
        "maximum_physical_violation": (
            maximum_physical <= float(config["dcopf_physical_tolerance"])
        ),
        "canonical_primal_violation": (
            canonical_violation <= float(config["dcopf_physical_tolerance"])
        ),
        "equality_solves_accurate": (
            result.maximum_equality_solve_infinity_residual
            <= float(config["maximum_equality_infinity_residual"])
        ),
        "z_x_identity_accurate": (
            result.maximum_z_x_identity_error <= float(config["maximum_z_x_identity_error"])
        ),
        "values_finite": bool(values_finite),
    }
    summary: dict[str, Any] = {
        "iterations": result.iterations,
        "converged": result.converged,
        "objective": objective,
        "reference_objective": reference_objective,
        "scaled_objective_gap_to_highs": _scaled_gap(objective, reference_objective),
        "original_residuals": result.residuals.summary(),
        "maximum_canonical_primal_violation": canonical_violation,
        "maximum_physical_violation": maximum_physical,
        "physical_validation": physical.summary(),
        "sigma": {
            "initial": result.initial_sigma,
            "final": result.sigma,
            "minimum": result.minimum_sigma,
            "maximum": result.maximum_sigma,
        },
        "restart_count": result.restart_count,
        "policy_event_count": len(result.policy_events),
        "policy_schedule": _policy_schedule(result.policy_events),
        "maximum_equality_solve_relative_residual": (
            result.maximum_equality_solve_relative_residual
        ),
        "maximum_equality_solve_infinity_residual": (
            result.maximum_equality_solve_infinity_residual
        ),
        "maximum_z_x_identity_error": result.maximum_z_x_identity_error,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if isinstance(result, GPUStage6Result):
        summary["timing"] = result.timing.as_dict()
    else:
        summary["timing"] = {
            "preparation_elapsed_seconds": result.preparation_elapsed_seconds,
            "total_elapsed_seconds": result.total_elapsed_seconds,
        }
    return summary


def _cpu_fixed_horizon(
    model: DCOPFModel,
    preconditioner: LPPreconditioner,
    config: dict[str, Any],
    iterations: int,
) -> Stage5SGSHPRResult:
    return solve_stage5_sgs_hpr(
        model.lp,
        sigma=float(config["initial_sigma"]),
        tolerance=np.finfo(np.float64).tiny,
        kkt_tolerance=None,
        max_iterations=iterations,
        history_interval=iterations,
        preconditioner=preconditioner,
        control=Stage5Control(),
    )


def _gpu_fixed_horizon(
    problem: GPUStage6Problem,
    config: dict[str, Any],
    iterations: int,
    *,
    precision: str,
    control: Stage5Control | None = None,
    residual_check_interval: int = 1,
) -> tuple[GPUStage6Result, dict[str, Any], dict[str, float]]:
    ledger_before = problem.backend.ledger.summary()
    timing_before = problem.backend.transfer_timing_summary()
    result = solve_gpu_stage5_sgs_hpr(
        problem,
        sigma=float(config["initial_sigma"]),
        tolerance=float(config["paper_tolerance"]),
        kkt_tolerance=None,
        max_iterations=iterations,
        residual_check_interval=residual_check_interval,
        history_interval=min(iterations, int(config["history_interval"])),
        control=Stage5Control() if control is None else control,
        fixed_horizon=True,
    )
    ledger_delta = _ledger_delta(ledger_before, result.transfer_ledger)
    timing_delta = _timing_delta(
        timing_before,
        problem.backend.transfer_timing_summary(),
    )
    if precision not in {"float64", "float32"}:
        raise ValueError(f"unexpected precision label: {precision}")
    return result, ledger_delta, timing_delta


def _resident_array_inventory(problem: GPUStage6Problem) -> dict[str, Any]:
    """Count declared reusable device arrays without claiming allocator completeness."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def add(prefix: str, value: Any) -> None:
        if not hasattr(value, "nbytes"):
            return
        byte_count = int(value.nbytes)
        pointer = int(getattr(getattr(value, "data", None), "ptr", id(value)))
        identity = (pointer, byte_count)
        if identity in seen:
            return
        seen.add(identity)
        rows.append(
            {
                "name": prefix,
                "shape": list(getattr(value, "shape", ())),
                "dtype": str(getattr(value, "dtype", "unknown")),
                "bytes": byte_count,
            }
        )

    def inspect_dataclass(prefix: str, value: Any) -> None:
        if not is_dataclass(value):
            return
        for field in fields(value):
            add(f"{prefix}.{field.name}", getattr(value, field.name))

    inspect_dataclass("workspace.buffers", problem.workspace.buffers)
    inspect_dataclass("problem.buffers", problem.buffers)
    for name in (
        "c",
        "b1",
        "b2",
        "lower",
        "upper",
        "equality_gram",
        "equality_cholesky",
        "structural_coupling",
        "structural_inverse_storage_diagonal",
        "structural_weight",
        "row_denominator",
        "column_denominator",
        "original_b",
        "original_c",
        "original_lower",
        "original_upper",
    ):
        owner = problem.workspace if hasattr(problem.workspace, name) else problem
        add(f"resident.{name}", getattr(owner, name, None))
    for operator_name, operator in (
        ("A1", problem.workspace.A1_resident),
        ("A2", problem.workspace.A2_resident),
    ):
        for orientation, matrix in (
            ("normal", operator.matrix),
            ("explicit_transpose", operator.transpose),
        ):
            add(f"{operator_name}.{orientation}.data", matrix.data)
            add(f"{operator_name}.{orientation}.indices", matrix.indices)
            add(f"{operator_name}.{orientation}.indptr", matrix.indptr)
    return {
        "scope": (
            "Declared persistent and reusable arrays only; not an exhaustive CuPy allocation trace."
        ),
        "array_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "arrays": rows,
    }


def _sparse_crosschecks(
    *,
    problem: GPUStage6Problem,
    config: dict[str, Any],
    case_index: int,
    case_name: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    benchmark_config = config["sparse_benchmark"]
    crosscheck = config["crosscheck"]
    rng = np.random.default_rng(int(benchmark_config["seed"]) + case_index)
    for operator_name, resident, host_matrix in (
        (
            "A1",
            problem.workspace.A1_resident,
            sparse.csr_matrix(problem.preconditioner.scaled_lp.A1),
        ),
        (
            "A2",
            problem.workspace.A2_resident,
            sparse.csr_matrix(problem.preconditioner.scaled_lp.A2),
        ),
    ):
        normal_host = rng.standard_normal(host_matrix.shape[1])
        transpose_host = rng.standard_normal(host_matrix.shape[0])
        dtype = np.float64 if problem.dtype_name == "float64" else np.float32
        normal_device = problem.backend.to_device(
            normal_host.astype(dtype),
            phase=f"{case_name}_{operator_name}_sparse_vectors",
            kind="vector",
        )
        transpose_device = problem.backend.to_device(
            transpose_host.astype(dtype),
            phase=f"{case_name}_{operator_name}_sparse_vectors",
            kind="vector",
        )
        normal_actual = problem.backend.to_host(
            resident.matvec(normal_device),
            phase=f"{case_name}_{operator_name}_sparse_crosscheck",
            kind="vector",
        )
        transpose_actual = problem.backend.to_host(
            resident.matvec(transpose_device, transpose=True),
            phase=f"{case_name}_{operator_name}_sparse_crosscheck",
            kind="vector",
        )
        normal_expected = np.asarray(host_matrix @ normal_host).reshape(-1)
        transpose_expected = np.asarray(host_matrix.T @ transpose_host).reshape(-1)
        normal_comparison = _compare_arrays(
            normal_actual,
            normal_expected,
            relative_tolerance=float(crosscheck["spmv_relative_tolerance"]),
            absolute_tolerance=float(crosscheck["spmv_absolute_tolerance"]),
        )
        transpose_comparison = _compare_arrays(
            transpose_actual,
            transpose_expected,
            relative_tolerance=float(crosscheck["spmv_relative_tolerance"]),
            absolute_tolerance=float(crosscheck["spmv_absolute_tolerance"]),
        )
        benchmark = resident.benchmark_matvec(
            normal_device,
            transpose_device,
            warmup_calls=int(benchmark_config["warmup_iterations"]),
            repetitions=int(benchmark_config["repetitions"]),
        )
        kernel = resident.kernel.as_dict()
        requested_matches = str(config["requested_spmv_algorithm"]) in str(
            kernel["requested_label"]
        )
        fallback_disclosed = bool(kernel["uses_csr_alg2"]) or bool(kernel["fallback_reason"])
        transpose_variants_match = benchmark.transpose_max_abs_difference <= float(
            crosscheck["spmv_absolute_tolerance"]
        )
        checks.append(
            {
                "operator": operator_name,
                "shape": list(host_matrix.shape),
                "nnz": int(host_matrix.nnz),
                "kernel_selection": kernel,
                "kernel_fidelity": (
                    "exact requested ALG2"
                    if kernel["uses_csr_alg2"]
                    else "declared generic fallback; no ALG2 performance claim"
                ),
                "normal_cpu_gpu_comparison": normal_comparison,
                "transpose_cpu_gpu_comparison": transpose_comparison,
                "normal_transpose_explicit_transpose_benchmark": benchmark.as_dict(),
                "checks": {
                    "requested_algorithm_recorded": requested_matches,
                    "fallback_is_disclosed_when_used": fallback_disclosed,
                    "normal_matches_cpu": normal_comparison["passed"],
                    "transpose_matches_cpu": transpose_comparison["passed"],
                    "transpose_flag_matches_explicit_transpose": (transpose_variants_match),
                },
            }
        )
        checks[-1]["passed"] = all(checks[-1]["checks"].values())
    return {"operators": checks, "passed": all(row["passed"] for row in checks)}


def _pull_gpu_state(
    backend: CuPyBackend,
    state: GPUHPRState,
    *,
    phase: str,
) -> HPRState:
    return HPRState(
        y=backend.to_host(state.y, phase=phase, kind="vector"),
        z=backend.to_host(state.z, phase=phase, kind="vector"),
        x=backend.to_host(state.x, phase=phase, kind="vector"),
    )


def _structural_parity(
    *,
    model: DCOPFModel,
    preconditioner: LPPreconditioner,
    backend: CuPyBackend,
    config: dict[str, Any],
    case_name: str,
) -> dict[str, Any]:
    structural = prepare_dcopf_structural_y1(model)

    def rejected_pairing(*, lp: Any, equality_mode: str) -> dict[str, Any]:
        try:
            prepare_gpu_sgs_hpr(
                lp,
                equality_mode=equality_mode,
                structural_y1=structural,
                backend=backend,
                dtype="float64",
            )
        except ValueError as error:
            return {
                "rejected": True,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        return {
            "rejected": False,
            "error_type": None,
            "message": "the incompatible pairing was unexpectedly accepted",
        }

    incompatible_pairings = {
        "raw_descriptor_with_scaled_direct": rejected_pairing(
            lp=model.lp,
            equality_mode="scaled_direct",
        ),
        "raw_descriptor_with_scaled_lp": rejected_pairing(
            lp=preconditioner.scaled_lp,
            equality_mode="unscaled_structural",
        ),
    }
    cpu_workspace = prepare_sgs_hpr(model.lp, structural_y1=structural)
    inequality_lambda = (
        None if cpu_workspace.spectral is None else cpu_workspace.spectral.lambda_used
    )
    gpu_workspace = prepare_gpu_sgs_hpr(
        model.lp,
        equality_mode="unscaled_structural",
        structural_y1=structural,
        inequality_lambda=inequality_lambda,
        backend=backend,
        dtype="float64",
    )
    zero = HPRState(
        y=np.zeros(model.lp.m, dtype=np.float64),
        z=np.zeros(model.lp.n, dtype=np.float64),
        x=np.zeros(model.lp.n, dtype=np.float64),
    )
    cpu_step = sgs_hpr_step(
        model.lp,
        zero,
        zero,
        cpu_workspace,
        iteration=0,
        sigma=float(config["initial_sigma"]),
    )
    device_zero = GPUHPRState.from_host(
        zero,
        backend,
        phase=f"{case_name}_structural_initial_state",
        dtype="float64",
    )
    gpu_step = gpu_sgs_hpr_step(
        model.lp,
        device_zero,
        device_zero,
        gpu_workspace,
        iteration=0,
        sigma=float(config["initial_sigma"]),
    )
    phase = f"{case_name}_structural_parity"
    gpu_states = {
        "proximal": _pull_gpu_state(backend, gpu_step.proximal, phase=phase),
        "reflected": _pull_gpu_state(backend, gpu_step.reflected, phase=phase),
        "next_state": _pull_gpu_state(backend, gpu_step.next_state, phase=phase),
    }
    cpu_states = {
        "proximal": cpu_step.proximal,
        "reflected": cpu_step.reflected,
        "next_state": cpu_step.next_state,
    }
    crosscheck = config["crosscheck"]
    comparisons = {
        name: _compare_states(
            gpu_states[name],
            cpu_states[name],
            relative_tolerance=float(crosscheck["one_step_relative_tolerance"]),
            absolute_tolerance=float(crosscheck["one_step_absolute_tolerance"]),
        )
        for name in gpu_states
    }
    y1_half = _compare_arrays(
        backend.to_host(gpu_step.y1_half, phase=phase, kind="vector"),
        cpu_step.y1_half,
        relative_tolerance=float(crosscheck["one_step_relative_tolerance"]),
        absolute_tolerance=float(crosscheck["one_step_absolute_tolerance"]),
    )
    gpu_diagnostics = np.asarray(
        [
            backend.scalar_to_host(
                gpu_step.first_equality_infinity_residual,
                phase=phase,
            ),
            backend.scalar_to_host(
                gpu_step.second_equality_infinity_residual,
                phase=phase,
            ),
            backend.scalar_to_host(gpu_step.z_x_identity_error, phase=phase),
        ]
    )
    cpu_diagnostics = np.asarray(
        [
            cpu_step.first_equality_infinity_residual,
            cpu_step.second_equality_infinity_residual,
            cpu_step.z_x_identity_error,
        ]
    )
    diagnostic_comparison = _compare_arrays(
        gpu_diagnostics,
        cpu_diagnostics,
        relative_tolerance=float(crosscheck["one_step_relative_tolerance"]),
        absolute_tolerance=float(crosscheck["one_step_absolute_tolerance"]),
    )
    checks = {
        "all_states_match": all(row["passed"] for row in comparisons.values()),
        "first_sweep_matches": y1_half["passed"],
        "diagnostics_match": diagnostic_comparison["passed"],
        "structural_mode_selected": gpu_workspace.equality_mode == "unscaled_structural",
        "raw_descriptor_retained": gpu_workspace.structural_y1 is structural,
        "raw_descriptor_rejected_by_scaled_direct": incompatible_pairings[
            "raw_descriptor_with_scaled_direct"
        ]["rejected"],
        "raw_descriptor_rejected_for_scaled_lp": incompatible_pairings[
            "raw_descriptor_with_scaled_lp"
        ]["rejected"],
    }
    return {
        "equality_mode": gpu_workspace.equality_mode,
        "state_comparisons": comparisons,
        "first_sweep_comparison": y1_half,
        "diagnostic_comparison": diagnostic_comparison,
        "A1_kernel_selection": gpu_workspace.A1_resident.kernel.as_dict(),
        "A2_kernel_selection": gpu_workspace.A2_resident.kernel.as_dict(),
        "incompatible_pairing_rejections": incompatible_pairings,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_gpu_correctness(
    *,
    problem: GPUStage6Problem,
    config: dict[str, Any],
    control: Stage5Control,
) -> tuple[GPUStage6Result, dict[str, Any], dict[str, float], float]:
    ledger_before = problem.backend.ledger.summary()
    timing_before = problem.backend.transfer_timing_summary()
    started = perf_counter()
    result = solve_gpu_stage5_sgs_hpr(
        problem,
        sigma=float(config["initial_sigma"]),
        tolerance=float(config["paper_tolerance"]),
        kkt_tolerance=float(config["dcopf_kkt_combined_target"]),
        max_iterations=int(config["maximum_iterations"]),
        residual_check_interval=int(config["correctness_residual_check_interval"]),
        history_interval=int(config["history_interval"]),
        control=control,
        fixed_horizon=False,
    )
    wall = perf_counter() - started
    return (
        result,
        _ledger_delta(ledger_before, result.transfer_ledger),
        _timing_delta(timing_before, problem.backend.transfer_timing_summary()),
        wall,
    )


def _full_cpu_gpu_comparison(
    cpu: Stage5SGSHPRResult,
    gpu: GPUStage6Result,
    *,
    model: DCOPFModel,
    config: dict[str, Any],
) -> dict[str, Any]:
    crosscheck = config["crosscheck"]
    objective = _compare_arrays(
        [model.objective(gpu.solution.x)],
        [model.objective(cpu.solution.x)],
        relative_tolerance=float(crosscheck["objective_relative_tolerance"]),
        absolute_tolerance=None,
    )
    gpu_normalized = np.asarray(gpu.residuals.paper_normalized_norms)
    cpu_normalized = np.asarray(cpu.residuals.paper_normalized_norms)
    residuals = _compare_arrays(
        gpu_normalized,
        cpu_normalized,
        relative_tolerance=None,
        absolute_tolerance=float(crosscheck["normalized_residual_absolute_tolerance"]),
    )
    state_diagnostic = _compare_states(
        gpu.solution,
        cpu.solution,
        relative_tolerance=float(crosscheck["fixed_horizon_state_relative_tolerance"]),
        absolute_tolerance=None,
    )
    policy_matches = _policy_schedule(gpu.policy_events) == _policy_schedule(cpu.policy_events)
    return {
        "objective": objective,
        "normalized_residuals": residuals,
        "final_state_diagnostic": {
            **state_diagnostic,
            "gating": False,
            "note": (
                "Stopping iterations may differ; objective and residual gates are authoritative."
            ),
        },
        "policy_schedule_matches": policy_matches,
        "iteration_count_matches": gpu.iterations == cpu.iterations,
        "checks": {
            "objective_matches": objective["passed"],
            "normalized_residuals_match": residuals["passed"],
            "policy_schedule_matches": policy_matches,
        },
        "passed": bool(objective["passed"] and residuals["passed"] and policy_matches),
    }


def _run_case(
    *,
    case_index: int,
    dcopf_path: Path,
    network: Any,
    backend: CuPyBackend,
    config: dict[str, Any],
    stream: TextIO,
    timing_totals: dict[str, float],
) -> tuple[dict[str, Any], _CaseRuntime]:
    cpu_preparation_started = perf_counter()
    dcopf_config = load_dcopf_config(dcopf_path, network)
    model = build_dcopf_model(network, dcopf_config)
    preconditioner = precondition_lp(
        model.lp,
        ruiz_iterations=int(config["ruiz_iterations"]),
        pock_chambolle=True,
        normalize=bool(config["normalize_b_and_c"]),
    )
    timing_totals["CPU matrix construction and preprocessing"] += (
        perf_counter() - cpu_preparation_started
    )
    highs_started = perf_counter()
    highs = solve_with_highs(model.lp, tolerance=float(config["paper_tolerance"]))
    highs_wall = perf_counter() - highs_started
    reference_objective = model.objective(highs.state.x)
    full_control = _control(config, adaptive_sigma=True, restart=True)

    cpu_started = perf_counter()
    cpu_full = solve_stage5_sgs_hpr(
        model.lp,
        sigma=float(config["initial_sigma"]),
        tolerance=float(config["paper_tolerance"]),
        kkt_tolerance=float(config["dcopf_kkt_combined_target"]),
        max_iterations=int(config["maximum_iterations"]),
        history_interval=int(config["history_interval"]),
        preconditioner=preconditioner,
        control=full_control,
    )
    cpu_runner_wall = perf_counter() - cpu_started
    _write_cpu_rows(
        stream,
        case_name=dcopf_config.name,
        run_name="full_fp64_correctness",
        result=cpu_full,
    )
    cpu_summary = _solution_summary(
        cpu_full,
        model=model,
        reference_objective=reference_objective,
        config=config,
    )
    cpu_summary["runner_wall_seconds"] = cpu_runner_wall

    memory_before = backend.memory_report().as_dict()
    ledger_before_prepare = backend.ledger.summary()
    transfer_before_prepare = backend.transfer_timing_summary()
    gpu_prepare_started = perf_counter()
    inequality_lambda = (
        None if cpu_full.workspace.spectral is None else cpu_full.workspace.spectral.lambda_used
    )
    problem = prepare_gpu_stage6_problem(
        model.lp,
        preconditioner,
        backend=backend,
        dtype="float64",
        inequality_lambda=inequality_lambda,
    )
    backend.synchronize()
    gpu_prepare_wall = perf_counter() - gpu_prepare_started
    timing_totals["GPU solver initialization"] += gpu_prepare_wall
    memory_after = backend.memory_report().as_dict()
    preparation_transfers = _ledger_delta(
        ledger_before_prepare,
        backend.ledger.summary(),
    )
    preparation_transfer_timing = _timing_delta(
        transfer_before_prepare,
        backend.transfer_timing_summary(),
    )

    warmup: dict[str, Any]
    if case_index == 0:
        warmup_started = perf_counter()
        warmup_result, warmup_ledger, warmup_transfer_timing = _gpu_fixed_horizon(
            problem,
            config,
            1,
            precision="float64",
        )
        warmup_wall = perf_counter() - warmup_started
        timing_totals["first-run compilation and warm-up"] += warmup_wall
        warmup = {
            "iterations": warmup_result.iterations,
            "wall_seconds": warmup_wall,
            "loop_timing": warmup_result.timing.as_dict(),
            "transfer_ledger": warmup_ledger,
            "transfer_timing": warmup_transfer_timing,
            "passed": warmup_result.iterations == 1,
        }
    else:
        warmup = {
            "status": "already completed before the first case",
            "passed": True,
        }

    sparse_crosschecks = _sparse_crosschecks(
        problem=problem,
        config=config,
        case_index=case_index,
        case_name=dcopf_config.name,
    )

    fixed_crosschecks: list[dict[str, Any]] = []
    cpu_fixed_results: dict[int, Stage5SGSHPRResult] = {}
    for iterations in (1, 10, int(config["crosscheck"]["fixed_horizon_iterations"])):
        cpu_fixed = _cpu_fixed_horizon(model, preconditioner, config, iterations)
        cpu_fixed_results[iterations] = cpu_fixed
        gpu_fixed, transfer_delta, transfer_timing = _gpu_fixed_horizon(
            problem,
            config,
            iterations,
            precision="float64",
        )
        run_name = f"fixed_{iterations}_step_fp64_crosscheck"
        _write_cpu_rows(
            stream,
            case_name=dcopf_config.name,
            run_name=run_name,
            result=cpu_fixed,
        )
        _write_gpu_rows(
            stream,
            case_name=dcopf_config.name,
            run_name=run_name,
            precision="float64",
            result=gpu_fixed,
        )
        if iterations == 1:
            relative_tolerance = float(config["crosscheck"]["one_step_relative_tolerance"])
            absolute_tolerance: float | None = float(
                config["crosscheck"]["one_step_absolute_tolerance"]
            )
        elif iterations == 10:
            relative_tolerance = float(config["crosscheck"]["ten_step_relative_tolerance"])
            absolute_tolerance = float(config["crosscheck"]["ten_step_absolute_tolerance"])
        else:
            relative_tolerance = float(
                config["crosscheck"]["fixed_horizon_state_relative_tolerance"]
            )
            absolute_tolerance = None
        original_comparison = _compare_states(
            gpu_fixed.solution,
            cpu_fixed.solution,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        scaled_comparison = _compare_states(
            gpu_fixed.scaled_solution,
            cpu_fixed.scaled_solution,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        transfer_audit = _audit_solver_transfers(transfer_delta)
        checks = {
            "cpu_completed_horizon": cpu_fixed.iterations == iterations,
            "gpu_completed_horizon": gpu_fixed.iterations == iterations,
            "original_state_matches": original_comparison["passed"],
            "scaled_state_matches": scaled_comparison["passed"],
            "device_residency_audit": transfer_audit["passed"],
        }
        fixed_crosschecks.append(
            {
                "iterations": iterations,
                "original_state": original_comparison,
                "scaled_state": scaled_comparison,
                "gpu_timing": gpu_fixed.timing.as_dict(),
                "transfer_ledger": transfer_delta,
                "transfer_timing": transfer_timing,
                "transfer_audit": transfer_audit,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    structural = _structural_parity(
        model=model,
        preconditioner=preconditioner,
        backend=backend,
        config=config,
        case_name=dcopf_config.name,
    )

    determinism_results: list[GPUStage6Result] = []
    determinism_runs: list[dict[str, Any]] = []
    determinism_config = config["determinism"]
    for repetition in range(int(determinism_config["repetitions"])):
        result, transfer_delta, transfer_timing = _gpu_fixed_horizon(
            problem,
            config,
            int(determinism_config["fixed_horizon_iterations"]),
            precision="float64",
            control=full_control,
            residual_check_interval=int(config["correctness_residual_check_interval"]),
        )
        determinism_results.append(result)
        _write_gpu_rows(
            stream,
            case_name=dcopf_config.name,
            run_name=f"determinism_fp64_repetition_{repetition + 1}",
            precision="float64",
            result=result,
        )
        determinism_runs.append(
            {
                "repetition": repetition + 1,
                "iterations": result.iterations,
                "timing": result.timing.as_dict(),
                "transfer_ledger": transfer_delta,
                "transfer_timing": transfer_timing,
                "transfer_audit": _audit_solver_transfers(transfer_delta),
            }
        )
    baseline = determinism_results[0]
    repeat_comparisons: list[dict[str, Any]] = []
    for repetition, result in enumerate(determinism_results[1:], start=2):
        comparison = _compare_states(
            result.solution,
            baseline.solution,
            relative_tolerance=float(determinism_config["relative_tolerance"]),
            absolute_tolerance=None,
        )
        bitwise = {
            name: bool(
                np.array_equal(
                    getattr(result.solution, name),
                    getattr(baseline.solution, name),
                )
            )
            for name in ("x", "y", "z")
        }
        repeat_comparisons.append(
            {
                "repetition": repetition,
                "state": comparison,
                "policy_schedule_matches": _policy_schedule(result.policy_events)
                == _policy_schedule(baseline.policy_events),
                "bitwise_equal_by_block": bitwise,
                "bitwise_required": bool(determinism_config["bitwise_required"]),
            }
        )
    determinism_checks = {
        "all_horizons_complete": all(
            result.iterations == int(determinism_config["fixed_horizon_iterations"])
            for result in determinism_results
        ),
        "all_states_within_tolerance": all(
            comparison["state"]["passed"] for comparison in repeat_comparisons
        ),
        "all_policy_schedules_match": all(
            comparison["policy_schedule_matches"] for comparison in repeat_comparisons
        ),
        "all_transfer_audits_pass": all(
            row["transfer_audit"]["passed"] for row in determinism_runs
        ),
    }
    determinism = {
        "runs": determinism_runs,
        "repeat_comparisons": repeat_comparisons,
        "checks": determinism_checks,
        "passed": all(determinism_checks.values()),
    }

    resident_iterations = int(config["resident_timing_iterations"])
    resident_result, resident_transfers, resident_transfer_timing = _gpu_fixed_horizon(
        problem,
        config,
        resident_iterations,
        precision="float64",
        control=full_control,
        residual_check_interval=int(config["resident_timing_residual_check_interval"]),
    )
    _write_gpu_rows(
        stream,
        case_name=dcopf_config.name,
        run_name="resident_1000_step_timing",
        precision="float64",
        result=resident_result,
    )
    timing_totals["iteration loop"] += resident_result.timing.loop_gpu_seconds
    timing_totals["residual checks"] += resident_result.timing.residual_check_gpu_seconds
    resident_transfer_audit = _audit_solver_transfers(resident_transfers)
    expected_checks = math.ceil(
        resident_iterations / int(config["resident_timing_residual_check_interval"])
    )
    resident_checks = {
        "completed_fixed_horizon": resident_result.iterations == resident_iterations,
        "residual_check_count": (resident_result.timing.residual_check_count == expected_checks),
        "values_finite": all(
            np.all(np.isfinite(block))
            for block in (
                resident_result.solution.x,
                resident_result.solution.y,
                resident_result.solution.z,
            )
        ),
        "device_residency_audit": resident_transfer_audit["passed"],
    }
    resident_timing = {
        "fixed_horizon": resident_iterations,
        "no_speedup_claim": True,
        "timing": resident_result.timing.as_dict(),
        "transfer_ledger": resident_transfers,
        "transfer_timing": resident_transfer_timing,
        "transfer_audit": resident_transfer_audit,
        "checks": resident_checks,
        "passed": all(resident_checks.values()),
    }

    gpu_full, full_transfers, full_transfer_timing, gpu_runner_wall = _run_gpu_correctness(
        problem=problem,
        config=config,
        control=full_control,
    )
    _write_gpu_rows(
        stream,
        case_name=dcopf_config.name,
        run_name="full_fp64_correctness",
        precision="float64",
        result=gpu_full,
    )
    gpu_summary = _solution_summary(
        gpu_full,
        model=model,
        reference_objective=reference_objective,
        config=config,
    )
    gpu_summary["runner_wall_seconds"] = gpu_runner_wall
    full_transfer_audit = _audit_solver_transfers(full_transfers)
    cpu_gpu_comparison = _full_cpu_gpu_comparison(
        cpu_full,
        gpu_full,
        model=model,
        config=config,
    )

    checks = {
        "warmup_completed": warmup["passed"],
        "cpu_oracle_passed": cpu_summary["passed"],
        "sparse_crosschecks_passed": sparse_crosschecks["passed"],
        "fixed_horizon_crosschecks_passed": all(row["passed"] for row in fixed_crosschecks),
        "unscaled_structural_parity_passed": structural["passed"],
        "determinism_passed": determinism["passed"],
        "resident_timing_completed": resident_timing["passed"],
        "gpu_correctness_passed": gpu_summary["passed"],
        "cpu_gpu_full_comparison_passed": cpu_gpu_comparison["passed"],
        "full_run_device_residency_audit": full_transfer_audit["passed"],
    }
    case_record = {
        "name": dcopf_config.name,
        "classification": dcopf_config.classification,
        "dimensions": model.dimension_summary(),
        "input_config": str(dcopf_path),
        "highs": {
            **highs.summary(),
            "total_objective": reference_objective,
            "runner_wall_seconds": highs_wall,
        },
        "preconditioner": _preconditioner_summary(preconditioner),
        "gpu_preparation": {
            "equality_mode": problem.workspace.equality_mode,
            "wall_seconds": gpu_prepare_wall,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "runtime_used_bytes_change": (
                int(memory_after["runtime_used_bytes"]) - int(memory_before["runtime_used_bytes"])
            ),
            "declared_resident_array_inventory": _resident_array_inventory(problem),
            "transfer_ledger": preparation_transfers,
            "transfer_timing": preparation_transfer_timing,
            "allocation_timing_note": (
                "Allocation is included in GPU solver initialization; reusable bytes and "
                "allocator snapshots are reported without claiming exhaustive allocation timing."
            ),
        },
        "warmup": warmup,
        "sparse_crosschecks": sparse_crosschecks,
        "fixed_horizon_crosschecks": fixed_crosschecks,
        "unscaled_structural_parity": structural,
        "determinism": determinism,
        "resident_timing": resident_timing,
        "cpu_full_oracle": cpu_summary,
        "gpu_full_fp64": {
            **gpu_summary,
            "transfer_ledger": full_transfers,
            "transfer_timing": full_transfer_timing,
            "transfer_audit": full_transfer_audit,
        },
        "cpu_gpu_full_comparison": cpu_gpu_comparison,
        "checks": checks,
        "passed": all(checks.values()),
    }
    fixed_100 = cpu_fixed_results[int(config["crosscheck"]["fixed_horizon_iterations"])]
    return case_record, _CaseRuntime(
        name=dcopf_config.name,
        model=model,
        preconditioner=preconditioner,
        inequality_lambda=inequality_lambda,
        cpu_fixed_100=fixed_100,
    )


def _run_fp32_study(
    *,
    runtime: _CaseRuntime,
    backend: CuPyBackend,
    config: dict[str, Any],
    stream: TextIO,
) -> dict[str, Any]:
    started = perf_counter()
    problem = prepare_gpu_stage6_problem(
        runtime.model.lp,
        runtime.preconditioner,
        backend=backend,
        dtype="float32",
        inequality_lambda=runtime.inequality_lambda,
    )
    preparation_wall = perf_counter() - started
    iterations = int(config["crosscheck"]["fixed_horizon_iterations"])
    result, transfer_delta, transfer_timing = _gpu_fixed_horizon(
        problem,
        config,
        iterations,
        precision="float32",
    )
    _write_gpu_rows(
        stream,
        case_name=runtime.name,
        run_name="optional_fp32_fixed_horizon",
        precision="float32",
        result=result,
    )
    state_diagnostic = _compare_states(
        result.solution,
        runtime.cpu_fixed_100.solution,
        relative_tolerance=float(config["crosscheck"]["fixed_horizon_state_relative_tolerance"]),
        absolute_tolerance=None,
    )
    transfer_audit = _audit_solver_transfers(transfer_delta)
    checks = {
        "ran_only_after_global_fp64_pass": True,
        "float32_device_state": problem.dtype_name == "float32",
        "completed_fixed_horizon": result.iterations == iterations,
        "values_finite": all(
            np.all(np.isfinite(block))
            for block in (result.solution.x, result.solution.y, result.solution.z)
        ),
        "device_residency_audit": transfer_audit["passed"],
    }
    return {
        "precision": "float32",
        "gating": False,
        "mixed_precision": False,
        "preparation_wall_seconds": preparation_wall,
        "iterations": result.iterations,
        "state_comparison_to_cpu_fp64": {
            **state_diagnostic,
            "gating": False,
        },
        "A1_kernel_selection": problem.workspace.A1_resident.kernel.as_dict(),
        "A2_kernel_selection": problem.workspace.A2_resident.kernel.as_dict(),
        "timing": result.timing.as_dict(),
        "transfer_ledger": transfer_delta,
        "transfer_timing": transfer_timing,
        "transfer_audit": transfer_audit,
        "checks": checks,
        "diagnostic_completed": all(checks.values()),
    }


def _timing_boundary_records(
    config: dict[str, Any],
    totals: dict[str, float],
    *,
    backend: CuPyBackend | None,
    complete_wall_seconds: float,
) -> dict[str, Any]:
    transfer = (
        {
            "host_to_device_seconds": 0.0,
            "device_to_host_seconds": 0.0,
        }
        if backend is None
        else backend.transfer_timing_summary()
    )
    records: dict[str, Any] = {}
    methods = {
        "CUDA initialization": (
            "Host monotonic clock around backend/device initialization, ending with "
            "an explicit device synchronization."
        ),
        "CPU matrix construction and preprocessing": (
            "Host monotonic clock around DCOPF construction and Stage 5 preprocessing."
        ),
        "first-run compilation and warm-up": (
            "Host monotonic clock around the first completed one-step GPU run; the "
            "run's GPU and transfer boundaries synchronize before return."
        ),
        "GPU solver initialization": (
            "Host monotonic clock around resident problem preparation, ending with "
            "an explicit device synchronization; includes allocation."
        ),
        "iteration loop": (
            "Synchronized CUDA events for the resident 1,000-step diagnostics, "
            "summed across the two frozen cases."
        ),
        "residual checks": (
            "Synchronized CUDA event pairs nested within the resident loop, summed "
            "across scheduled checks and the two frozen cases."
        ),
    }
    for name in config.get("timing_boundaries", []):
        if name == "allocation":
            records[name] = {
                "seconds": None,
                "unit": "seconds",
                "method": (
                    "Not independently timed; included in synchronized GPU solver initialization."
                ),
                "status": "reported but not independently timed",
                "evidence": ("Per-case reusable array bytes and before/after allocator snapshots."),
            }
        elif name == "host-to-device transfer":
            records[name] = {
                "seconds": transfer["host_to_device_seconds"],
                "unit": "seconds",
                "method": "Host monotonic clock around synchronized explicit transfers.",
                "status": "measured with synchronized explicit transfers",
            }
        elif name == "device-to-host transfer":
            records[name] = {
                "seconds": transfer["device_to_host_seconds"],
                "unit": "seconds",
                "method": "Host monotonic clock around synchronized explicit transfers.",
                "status": "measured with synchronized explicit transfers",
            }
        elif name == "complete end-to-end wall time":
            records[name] = {
                "seconds": complete_wall_seconds,
                "unit": "seconds",
                "method": "Host monotonic clock around the complete Stage 6 runner.",
                "status": "measured by monotonic host clock",
            }
        else:
            records[name] = {
                "seconds": totals.get(name, 0.0),
                "unit": "seconds",
                "method": methods.get(
                    name, "Method unavailable because the boundary was not reached."
                ),
                "status": "measured" if totals.get(name, 0.0) > 0.0 else "not reached",
            }
    return {
        "boundaries": records,
        "synchronization_policy": (
            "Explicit transfers and CUDA-event intervals are synchronized. "
            "Nested boundaries are labeled and are not summed into a speedup."
        ),
        "gpu_speedup_claimed": False,
    }


def main() -> int:
    args = parse_args()
    started = perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "stage_6_validation.json"
    trajectory_path = output_dir / "stage_6_trajectories.jsonl.gz"
    config_path = args.config.resolve()
    network_path = args.network.resolve()
    dcopf_paths = tuple(path.resolve() for path in args.dcopf_configs)
    source_paths = (
        Path(__file__).resolve(),
        *sorted((SOURCE_ROOT / "gpu_dcopf_hpr").glob("*.py")),
    )
    backend: CuPyBackend | None = None
    config: dict[str, Any] = {}
    timing_totals = {
        "CUDA initialization": 0.0,
        "CPU matrix construction and preprocessing": 0.0,
        "first-run compilation and warm-up": 0.0,
        "GPU solver initialization": 0.0,
        "iteration loop": 0.0,
        "residual checks": 0.0,
    }
    evidence: dict[str, Any] = {
        "stage": 6,
        "status": "RUNNING",
        "all_passed": False,
        "started_utc": datetime.now(UTC).isoformat(),
        "configuration": None,
        "configuration_validation": {"errors": [], "passed": False},
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": _package_versions(),
            "git": _git_metadata(),
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "network": {"path": str(network_path), "sha256": _sha256(network_path)},
            "dcopf_configs": [{"path": str(path), "sha256": _sha256(path)} for path in dcopf_paths],
            "requirements": {
                "path": str(DEFAULT_REQUIREMENTS),
                "sha256": _sha256(DEFAULT_REQUIREMENTS),
            },
            "source_files": [
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in source_paths
            ],
        },
        "device": None,
        "cases": [],
        "precision_study": {
            "fp64": {"status": "pending", "passed": False},
            "fp32": {"status": "not_run", "gating": False},
        },
        "failures": [],
        "non_gating_failures": [],
        "evidence_files": {
            "validation": validation_path.name,
            "trajectories_and_policy_events": trajectory_path.name,
        },
        "stage_boundary": {
            "stage_6_only": True,
            "stage_6_executed": False,
            "stage_6_complete": False,
            "stage_7_benchmarks_locked": True,
            "paper_timing_reproduction_claimed": False,
            "gpu_speedup_claimed": False,
        },
    }
    _write_json(validation_path, evidence)

    runtimes: list[_CaseRuntime] = []
    with gzip.open(trajectory_path, "wt", encoding="utf-8") as stream:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            evidence["configuration"] = config
            config_errors = _validate_stage6_config(config)
            evidence["configuration_validation"] = {
                "errors": config_errors,
                "passed": not config_errors,
            }
            if config_errors:
                raise ValueError("; ".join(config_errors))
            for path in (network_path, *dcopf_paths):
                if not path.is_file():
                    raise FileNotFoundError(path)
            _write_json(validation_path, evidence)

            cuda_started = perf_counter()
            backend = create_gpu_backend(
                device_id=args.device_id,
                required_cupy_version=str(config["gpu_backend_version"]),
            )
            backend.synchronize()
            timing_totals["CUDA initialization"] = perf_counter() - cuda_started
            diagnostics = backend.diagnostics
            device_checks = {
                "cupy_version_matches_config": (
                    diagnostics.cupy_version == str(config["gpu_backend_version"])
                ),
                "fp64_supported": diagnostics.fp64_supported,
                "fp64_itemsize_is_eight": diagnostics.fp64_itemsize_bytes == 8,
                "csr_indices_are_int32": diagnostics.csr_index_bits == 32,
            }
            evidence["device"] = {
                "diagnostics": diagnostics.as_dict(),
                "cuda_library_versions": _cuda_library_versions(backend),
                "initial_memory": backend.memory_report().as_dict(),
                "checks": device_checks,
                "passed": all(device_checks.values()),
            }
            evidence["stage_boundary"]["stage_6_executed"] = True
            network = load_matpower_case(network_path)

            for case_index, dcopf_path in enumerate(dcopf_paths):
                try:
                    case_record, runtime = _run_case(
                        case_index=case_index,
                        dcopf_path=dcopf_path,
                        network=network,
                        backend=backend,
                        config=config,
                        stream=stream,
                        timing_totals=timing_totals,
                    )
                    evidence["cases"].append(case_record)
                    runtimes.append(runtime)
                except Exception as error:
                    failure = _exception_record(f"case:{dcopf_path.name}", error)
                    failure["gating"] = True
                    evidence["failures"].append(failure)
                    evidence["cases"].append(
                        {
                            "input_config": str(dcopf_path),
                            "status": "ERROR",
                            "passed": False,
                            "failure": failure,
                        }
                    )
                _write_json(validation_path, evidence)

            fp64_passed = bool(
                evidence["device"]["passed"]
                and len(evidence["cases"]) == len(dcopf_paths)
                and len(runtimes) == len(dcopf_paths)
                and all(case.get("passed", False) for case in evidence["cases"])
                and not evidence["failures"]
            )
            evidence["precision_study"]["fp64"] = {
                "status": "PASS" if fp64_passed else "FAIL",
                "passed": fp64_passed,
                "gating": True,
            }

            fp32_config = config["precision_study"]
            if fp64_passed and bool(fp32_config["fp32_enabled_after_fp64_pass"]):
                fp32_cases: list[dict[str, Any]] = []
                for runtime in runtimes:
                    try:
                        fp32_cases.append(
                            {
                                "case": runtime.name,
                                **_run_fp32_study(
                                    runtime=runtime,
                                    backend=backend,
                                    config=config,
                                    stream=stream,
                                ),
                            }
                        )
                    except Exception as error:
                        failure = _exception_record(f"fp32:{runtime.name}", error)
                        failure["gating"] = False
                        evidence["non_gating_failures"].append(failure)
                        fp32_cases.append(
                            {
                                "case": runtime.name,
                                "status": "ERROR",
                                "gating": False,
                                "failure": failure,
                            }
                        )
                evidence["precision_study"]["fp32"] = {
                    "status": "COMPLETED_WITH_DIAGNOSTIC_FAILURES"
                    if evidence["non_gating_failures"]
                    else "COMPLETED",
                    "gating": False,
                    "mixed_precision": False,
                    "ran_after_fp64_pass": True,
                    "cases": fp32_cases,
                }
            else:
                evidence["precision_study"]["fp32"] = {
                    "status": "NOT_RUN",
                    "gating": False,
                    "reason": (
                        "All FP64 gates must pass before the optional FP32 diagnostic."
                        if not fp64_passed
                        else "FP32 is disabled by the versioned configuration."
                    ),
                }
        except Exception as error:
            failure = _exception_record("stage_6_global", error)
            failure["gating"] = True
            evidence["failures"].append(failure)

    complete_wall = perf_counter() - started
    if config:
        evidence["timing_boundaries"] = _timing_boundary_records(
            config,
            timing_totals,
            backend=backend,
            complete_wall_seconds=complete_wall,
        )
    evidence["evidence_files"]["trajectory_sha256"] = _sha256(trajectory_path)
    fp64_passed = bool(evidence["precision_study"]["fp64"].get("passed", False))
    evidence["all_passed"] = fp64_passed and not evidence["failures"]
    evidence["status"] = "PASS" if evidence["all_passed"] else "FAIL"
    evidence["stage_boundary"]["stage_6_complete"] = evidence["all_passed"]
    evidence["stage_boundary"]["stage_7_benchmarks_locked"] = True
    evidence["stage_boundary"]["paper_timing_reproduction_claimed"] = False
    evidence["stage_boundary"]["gpu_speedup_claimed"] = False
    if backend is not None:
        evidence["final_memory"] = backend.memory_report().as_dict()
        evidence["final_transfer_ledger"] = backend.ledger.summary()
        evidence["final_transfer_timing"] = backend.transfer_timing_summary()
    _write_json(validation_path, evidence)
    print(
        json.dumps(
            _clean_json(
                {
                    "stage": 6,
                    "status": evidence["status"],
                    "all_passed": evidence["all_passed"],
                    "validation_file": str(validation_path),
                    "trajectory_file": str(trajectory_path),
                    "cases": len(evidence["cases"]),
                    "gating_failures": len(evidence["failures"]),
                    "stage_7_benchmarks_locked": True,
                    "gpu_speedup_claimed": False,
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if evidence["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
