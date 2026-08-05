"""Complete Stage 8 sequences 6--8 with only HiGHS and GPU solver tracks.

This is a separately scoped continuation of the immutable Stage 8 campaign.
The original sequence-5 CPU failure remains terminal evidence.  The continuation
may allocate only sequence 6, after the unchanged unified-memory gate, and it
records sequences 7--8 as static signed-int32 CSR blocks without allocating an
LP.  CPU sGS-HPR and Gurobi are explicitly skipped, never treated as passes.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts import check_stage_8 as stage8_checker  # noqa: E402
from scripts import run_stage_7 as stage7  # noqa: E402
from scripts import run_stage_8 as stage8  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "benchmarks" / "stage_8_gpu_only_completion.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_8" / "gpu_only_completion"
PARTIAL_NAME = "stage_8_gpu_only_completion.partial.json"
FINAL_NAME = "stage_8_gpu_only_completion_validation.json"
APPROVAL_GATE = (
    "for stage 8, proceed with completing sequence 6,7,8 but only do HiGHS and GPU "
    "(i.e. skip CPU which was timing out). then give an updated report. stop before "
    "proceeding to stage 9."
)
FROZEN_CONFIG_SHA256 = "76ff7cb76f70ff104d1691a152b57ebabaade3e26b784091b888c1d5918cc64c"
REQUESTED_KEYS = (
    "case9241pegase:T16",
    "case9241pegase:T24",
    "case9241pegase:T32",
)
STATIC_BLOCK_KEYS = frozenset(REQUESTED_KEYS[1:])
REQUIRED_TRACKS = ("highs", "gpu_fp64_sgs_hpr")
SKIPPED_TRACKS = ("cpu_fp64_sgs_hpr", "gurobi")
TERMINAL_T16_STATUSES = frozenset({"PASS", "FAIL", "TIME_LIMIT", "MEMORY_BLOCKED"})


class Stage8GPUOnlyContractError(ValueError):
    """The continuation contract or one of its immutable inputs drifted."""


class Stage8GPUOnlyResumeError(Stage8GPUOnlyContractError):
    """A checkpoint cannot be resumed without changing the evidence semantics."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device-id", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-next",
        action="store_true",
        help="Evaluate sequence 6 and resolve the static sequence 7--8 blocks.",
    )
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the no-allocation continuation plan (the default).",
    )
    parser.add_argument(
        "--approval-token",
        help="Exact continuation approval; required with --run-next.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume only an unallocated PLANNED checkpoint with the same fingerprint.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage8GPUOnlyContractError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _validate_config(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_top = {
        "schema_version": "1.0",
        "stage": 8,
        "name": "stage_8_gpu_only_sequence_6_8_completion",
        "classification": "structural_reproduction",
        "approval_gate": APPROVAL_GATE,
    }
    for name, expected in expected_top.items():
        if config.get(name) != expected:
            errors.append(f"{name} drifted from the GPU-only continuation contract")
    requested = config.get("requested_sequence")
    if not isinstance(requested, list) or [row.get("key") for row in requested] != list(
        REQUESTED_KEYS
    ):
        errors.append("requested_sequence differs from sequences 6--8")
    tracks = config.get("track_policy", {})
    if tracks.get("required_solver_tracks") != list(REQUIRED_TRACKS):
        errors.append("required solver tracks are not exactly HiGHS and GPU FP64 sGS-HPR")
    if tracks.get("explicitly_skipped_solver_tracks") != list(SKIPPED_TRACKS):
        errors.append("CPU and Gurobi are not both explicitly skipped")
    if tracks.get("cpu_skip_is_not_a_cpu_pass") is not True:
        errors.append("the CPU skip must not be represented as a pass")
    resource = config.get("resource_policy", {})
    expected_resource = {
        "estimate_before_every_allocation": True,
        "host_safety_fraction": stage7.HOST_SAFETY_FRACTION,
        "device_safety_fraction": stage7.DEVICE_SAFETY_FRACTION,
        "unified_memory_accounting": "sum_host_assembly_peak_and_device_planning",
        "require_observed_host_available_bytes": True,
        "require_observed_device_total_bytes": True,
        "require_observed_device_free_bytes": True,
        "required_sparse_index_bits": 32,
        "stop_before_allocation_if_unsafe": True,
        "memory_projection_unchanged_by_cpu_waiver": True,
    }
    if resource != expected_resource:
        errors.append("resource_policy drifted from the unchanged fail-closed contract")
    execution = config.get("execution_policy", {})
    if not (
        execution.get("one_new_full_lp_allocation_per_invocation") is True
        and execution.get("only_sequence_6_is_allocation_eligible") is True
        and execution.get("record_sequence_7_8_static_blocks_without_allocation") is True
        and execution.get("interrupted_run_requires_new_user_direction") is True
    ):
        errors.append("execution_policy does not preserve the continuation boundary")
    acceptance = config.get("inherited_acceptance", {})
    if not (
        acceptance.get("reuse_numerical_thresholds_without_changes") is True
        and acceptance.get("reuse_timing_repetitions_without_changes") is True
        and acceptance.get("per_solve_time_limit_seconds") == 3600
        and acceptance.get("correctness_run_count") == 1
        and acceptance.get("warmup_run_count") == 1
        and acceptance.get("minimum_measured_run_count") == 5
        and acceptance.get("speedup_computed") is False
    ):
        errors.append("inherited numerical or timing acceptance gates drifted")
    if config.get("expected_preallocation_blocks") != {
        "case9241pegase:T24": "signed_int32_csr_nnz_limit",
        "case9241pegase:T32": "signed_int32_csr_nnz_limit",
    }:
        errors.append("expected static CSR blocks drifted")
    boundary = config.get("stage_boundary", {})
    if not (
        boundary.get("stage_8_only") is True
        and boundary.get("stage_9_locked") is True
        and boundary.get("stage_9_allocation_count") == 0
        and boundary.get("n_minus_1_extension_enabled") is False
        and boundary.get("exact_paper_reproduction_claimed") is False
        and boundary.get("paper_a100_timing_reproduction_claimed") is False
    ):
        errors.append("Stage 9 or scientific-claim boundary drifted")
    return errors


def _solver_availability() -> dict[str, Any]:
    """Probe only the two authorized tracks; CPU and Gurobi are not probed."""

    gpu_signature = inspect.signature(stage7.prepare_gpu_stage6_problem)
    gpu_adapter = any(
        name in gpu_signature.parameters
        for name in ("scaled_structural_y1", "scaled_equality_solver")
    )
    cupy_version = stage7._package_version("cupy-cuda13x")
    return {
        "highs": {
            "installed": True,
            "available": True,
            "provider": "scipy.optimize.linprog(method='highs-ds')",
            "scipy_version": stage7.scipy.__version__,
        },
        "gpu_fp64_sgs_hpr": {
            "installed": cupy_version is not None,
            "adapter_available": gpu_adapter,
            "available": bool(cupy_version is not None and gpu_adapter),
            "cupy_cuda13x_version": cupy_version,
            "reason": (
                None
                if cupy_version is not None and gpu_adapter
                else "CuPy and the device scaled block-arrow adapter are both required"
            ),
        },
        "cpu_fp64_sgs_hpr": {
            "available": None,
            "probed": False,
            "status": "SKIPPED_BY_USER_SCOPE",
        },
        "gurobi": {
            "available": None,
            "probed": False,
            "status": "SKIPPED_BY_USER_SCOPE",
        },
    }


def _original_contract(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    original = config["original_stage_8"]
    paths = {
        "configuration": (PROJECT_ROOT / original["configuration_path"]).resolve(),
        "evidence": (PROJECT_ROOT / original["evidence_path"]).resolve(),
        "checker": (PROJECT_ROOT / original["checker_path"]).resolve(),
    }
    expected = {
        "configuration": str(original["configuration_sha256"]),
        "evidence": str(original["evidence_sha256"]),
        "checker": str(original["checker_sha256"]),
    }
    actual = {name: _sha256(path) for name, path in paths.items()}
    old_config = _load_json(paths["configuration"])
    old_evidence = _load_json(paths["evidence"])
    stored_checks = _load_json(paths["checker"])
    replay = stage8_checker.run_checks(paths["evidence"], paths["configuration"])
    t6 = next(
        (row for row in old_evidence.get("cases", []) if row.get("key") == "case9241pegase:T6"),
        {},
    )
    errors = [
        f"original Stage 8 {name} SHA-256 mismatch"
        for name in expected
        if actual.get(name) != expected[name]
    ]
    if stage8._validate_stage8_config(old_config):
        errors.append("original Stage 8 configuration no longer satisfies its frozen contract")
    if not (
        old_evidence.get("status") == "STOPPED_ON_FAILURE"
        and t6.get("status") in {"FAIL", "TIME_LIMIT"}
        and t6.get("passed") is False
    ):
        errors.append("original T6 CPU failure is no longer preserved")
    if not (
        stored_checks.get("checker_status") == "PASS"
        and stored_checks.get("all_passed") is True
        and replay.get("checker_status") == "PASS"
        and replay.get("all_passed") is True
    ):
        errors.append("original Stage 8 independent honesty checker did not pass")
    result = {
        "paths": {name: path.relative_to(PROJECT_ROOT).as_posix() for name, path in paths.items()},
        "expected_sha256": expected,
        "actual_sha256": actual,
        "terminal_status": old_evidence.get("status"),
        "t6_status": t6.get("status"),
        "stored_checker_summary": stored_checks.get("summary"),
        "replayed_checker_summary": replay.get("summary"),
        "errors": errors,
        "passed": not errors,
    }
    return result, {
        "configuration": old_config,
        "evidence": old_evidence,
        "checker": stored_checks,
    }


def _source_manifest(config_path: Path) -> list[dict[str, Any]]:
    tracked_bytes, tracked_error = stage7._git_bytes(
        "ls-tree", "-r", "--name-only", "HEAD", "--", "src/gpu_dcopf_hpr"
    )
    tracked = (
        set()
        if tracked_bytes is None
        else {
            PROJECT_ROOT / relative
            for relative in tracked_bytes.decode("utf-8", errors="replace").splitlines()
            if relative.startswith("src/gpu_dcopf_hpr/") and relative.endswith(".py")
        }
    )
    local = set((PROJECT_ROOT / "src" / "gpu_dcopf_hpr").glob("*.py"))
    paths = [
        Path(__file__).resolve(),
        (PROJECT_ROOT / "scripts" / "check_stage_8_gpu_only_completion.py").resolve(),
        (PROJECT_ROOT / "scripts" / "run_stage_8.py").resolve(),
        (PROJECT_ROOT / "scripts" / "check_stage_8.py").resolve(),
        (PROJECT_ROOT / "scripts" / "run_stage_7.py").resolve(),
        (PROJECT_ROOT / "scripts" / "check_stage_7.py").resolve(),
        config_path,
        (PROJECT_ROOT / "configs" / "benchmarks" / "stage_8_large.json").resolve(),
        (PROJECT_ROOT / "configs" / "benchmarks" / "stage_7_small_medium.json").resolve(),
        (PROJECT_ROOT / "results" / "raw" / "stage_8" / "stage_8_validation.json").resolve(),
        (PROJECT_ROOT / "results" / "raw" / "stage_8" / "stage_8_checks.json").resolve(),
        (PROJECT_ROOT / "results" / "raw" / "stage_7" / "stage_7_validation.json").resolve(),
        (PROJECT_ROOT / "environment" / "dgx_stage7_requirements.txt").resolve(),
        *sorted(tracked | local),
    ]
    rows: list[dict[str, Any]] = []
    if tracked_error is not None:
        rows.append(
            {
                "path": "src/gpu_dcopf_hpr",
                "git_blob": None,
                "sha256": None,
                "passed": False,
                "error": tracked_error,
            }
        )
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT)
        identity = stage7._canonical_git_blob_identity(relative, path)
        rows.append(
            {
                "path": relative.as_posix(),
                "git_blob": identity["expected_git_blob"],
                "sha256": identity["canonical_git_blob_sha256"],
                "sha256_definition": stage7.CANONICAL_GIT_BLOB_SHA256_DEFINITION,
                "passed": identity["passed"],
            }
        )
    return rows


def _run_fingerprint(
    config_sha256: str | None,
    original_contract: Mapping[str, Any],
    source_manifest: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "continuation_config_sha256": config_sha256,
        "original_stage8_sha256": original_contract.get("actual_sha256"),
        "sources": list(source_manifest),
        "python": platform.python_version(),
        "numpy": stage7._package_version("numpy"),
        "scipy": stage7._package_version("scipy"),
        "cupy_cuda13x": stage7._package_version("cupy-cuda13x"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resource_ledger(
    original_evidence: Mapping[str, Any],
    base_stage7_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    original_rows = {
        str(row.get("key")): row for row in original_evidence.get("resource_ledger", [])
    }
    recomputed_rows = {
        str(row.get("key")): row for row in stage8._resource_ledger(base_stage7_evidence)
    }
    rows: list[dict[str, Any]] = []
    for sequence, key in enumerate(REQUESTED_KEYS, start=6):
        if key not in original_rows or key not in recomputed_rows:
            raise Stage8GPUOnlyContractError(f"frozen resource ledger is missing {key}")
        original_row = original_rows[key]
        recomputed_row = recomputed_rows[key]
        if not (
            original_row.get("resource_estimate") == recomputed_row.get("resource_estimate")
            and original_row.get("static_preallocation_status")
            == recomputed_row.get("static_preallocation_status")
            and original_row.get("static_block_reasons")
            == recomputed_row.get("static_block_reasons")
        ):
            raise Stage8GPUOnlyContractError(f"frozen resource projection drifted for {key}")
        row = deepcopy(original_row)
        row.update(
            {
                "campaign_sequence": sequence,
                "continuation_scope": "gpu_only_sequences_6_8",
                "allocation_permitted_this_invocation": False,
                "required_solver_tracks_if_allocated": list(REQUIRED_TRACKS),
                "explicitly_skipped_solver_tracks": list(SKIPPED_TRACKS),
            }
        )
        rows.append(row)
    return rows


def _skip_records(*, resource_blocked: bool = False) -> dict[str, dict[str, Any]]:
    reason = (
        "not reached because the unchanged preallocation resource guard blocked the row"
        if resource_blocked
        else "explicit user scope for the Stage 8 sequence 6--8 continuation"
    )
    return {
        name: {
            "name": name,
            "status": "SKIPPED_BY_USER_SCOPE",
            "reason": reason,
            "gating": False,
            "passed": None,
            "executed": False,
        }
        for name in SKIPPED_TRACKS
    }


def _static_case(row: Mapping[str, Any]) -> dict[str, Any]:
    key = str(row["key"])
    parsed = stage8._split_key(key)
    failure = {
        "phase": f"preallocation:{key}",
        "type": "SparseIndexSafetyBlock",
        "message": "signed_int32_csr_nnz_limit",
        "recorded_utc": stage7._utc_now(),
        "full_lp_allocated": False,
    }
    return {
        "key": key,
        "sequence": int(row["campaign_sequence"]),
        "case_name": parsed.case_name,
        "periods": parsed.periods,
        "status": "INDEX_BLOCKED",
        "passed": False,
        "resolved": True,
        "full_lp_allocation_attempted": False,
        "resource_estimate": deepcopy(row["resource_estimate"]),
        "stage8_resource_gate": {
            "passed": False,
            "evaluated_before_full_lp_allocation": True,
            "evaluation_kind": "static_signed_int32_csr_guard",
            "block_reasons": ["signed_int32_csr_nnz_limit"],
        },
        "solver_tracks": {},
        "required_solver_track_disposition": {
            name: "NOT_RUN_STATIC_RESOURCE_BLOCK" for name in REQUIRED_TRACKS
        },
        "skipped_solver_tracks": _skip_records(resource_blocked=True),
        "failure": failure,
        "completed_utc": stage7._utc_now(),
    }


def _summarize_highs_attempt(
    attempt: Mapping[str, Any],
    *,
    model: Any,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    state = attempt.get("state")
    summary = {name: value for name, value in attempt.items() if name != "state"}
    if not isinstance(state, stage7.HPRState):
        summary["passed"] = False
        return summary
    candidate = stage7._candidate_summary(
        model,
        state,
        config=base_config,
        reference_objective=None,
    )
    summary["candidate"] = candidate
    summary["passed"] = bool(attempt.get("status") == "SUCCESS" and candidate["passed"])
    return summary


def _run_gpu_only_case(
    case: dict[str, Any],
    *,
    base_config: Mapping[str, Any],
    availability: Mapping[str, Any],
    device_id: int,
    checkpoint: Callable[[], None],
) -> None:
    key = stage8._split_key(str(case["key"]))
    preflight = next(
        row
        for row in stage7.all_stage7_preflights()
        if row.row.case_name == key.case_name and row.row.periods == key.periods
    )
    case_wall_started = perf_counter()
    case.update(
        {
            "status": "RUNNING",
            "passed": False,
            "resolved": False,
            "started_utc": case.get("started_utc", stage7._utc_now()),
            "required_solver_tracks": list(REQUIRED_TRACKS),
            "skipped_solver_tracks": _skip_records(),
            "solver_tracks": case.get("solver_tracks", {}),
        }
    )
    checkpoint()

    construction_started = perf_counter()
    network_path = stage7._input_for_case(base_config, key.case_name)
    network = stage7.load_matpower_case(network_path)
    gate = case["stage8_resource_gate"]
    model = stage7.build_stage7_scalable_model(
        network,
        key.periods,
        host_memory_budget_bytes=gate["host_safety_budget_bytes"],
    )
    case["construction"] = {
        "wall_seconds": perf_counter() - construction_started,
        "dimensions": model.dimension_summary(),
        "lp_fingerprint": stage7._lp_fingerprint(model),
        "policy_fingerprint": model.fleet.policy_fingerprint,
        "input_sha256": next(
            str(item["sha256"])
            for item in base_config["public_network_source"]["files"]
            if item["case"] == key.case_name
        ),
        "input_sha256_definition": base_config["public_network_source"]["sha256_definition"],
    }
    count_only = stage7.stage7_reconstructed_nnz_ledger(
        model.normalized,
        key.periods,
        fleet=model.fleet,
        ptdf=model.ptdf,
    )
    dimensions = model.dimension_summary()
    actual_nnz = int(model.lp.A1.nnz + model.lp.A2.nnz)
    case["structural_reconciliation"] = {
        "dimension_match": (
            dimensions["m"] == preflight.row.published_m
            and dimensions["n"] == preflight.row.published_n
        ),
        "published_nnz": preflight.row.published_nnz,
        "actual_nnz": actual_nnz,
        "nnz_difference": actual_nnz - preflight.row.published_nnz,
        "symbolic_reconstructed_nnz": count_only.reconstructed_nnz,
        "actual_matches_symbolic_nnz": actual_nnz == count_only.reconstructed_nnz,
        "paper_time_comparable": actual_nnz == preflight.row.published_nnz,
        "classification": "structural_reproduction_not_author_instance",
    }
    if not case["structural_reconciliation"]["dimension_match"]:
        raise Stage8GPUOnlyContractError(f"{key.text} dimensions differ from Table II")
    if not case["structural_reconciliation"]["actual_matches_symbolic_nnz"]:
        raise Stage8GPUOnlyContractError(f"{key.text} allocated nnz differs from the ledger")
    checkpoint()

    preprocessing_started = perf_counter()
    preconditioner = stage7._precondition(model, base_config)
    scaled_solver = stage7.prepare_scaled_block_arrow_y1(
        preconditioner, stage7._case_structure(model)
    )
    spectral = stage7.estimate_sparse_spectral_norm_squared(
        preconditioner.scaled_lp.A2,
        power_seed=int(base_config["reconstruction_protocol"]["seed"]),
    )
    case["preprocessing"] = {
        "wall_seconds": perf_counter() - preprocessing_started,
        "preconditioner": stage7._clean_json(preconditioner.diagnostics),
        "scaled_equality": scaled_solver.diagnostics.summary(),
        "sparse_spectral_certificate": spectral.summary(),
        "cpu_hpr_workspace_prepared": False,
        "cpu_hpr_solver_called": False,
    }
    checkpoint()

    tracks = case["solver_tracks"]
    limit = int(base_config["timing"]["per_solve_time_limit_seconds"])
    highs = tracks.setdefault(
        "highs",
        {
            "name": "highs",
            "timing_boundary": (
                "SciPy linprog call including HiGHS interface/model setup and solve"
            ),
        },
    )
    stage7._run_timed_track(
        track=highs,
        solve=lambda: stage7._solve_highs(model, time_limit_seconds=limit),
        summarize=lambda attempt: _summarize_highs_attempt(
            attempt, model=model, base_config=base_config
        ),
        config=base_config,
        checkpoint=checkpoint,
    )
    reference = highs.get("correctness", {}).get("candidate", {}).get("objective")
    if reference is None or highs.get("passed") is not True:
        status = highs.get("timing_status") or highs.get("status")
        case["status"] = "TIME_LIMIT" if status == "TIME_LIMIT" else "FAIL"
        case["passed"] = False
        case["resolved"] = True
        case["completed_utc"] = stage7._utc_now()
        case["end_to_end_case_wall_seconds"] = perf_counter() - case_wall_started
        checkpoint()
        return

    gpu = tracks.setdefault(
        "gpu_fp64_sgs_hpr",
        {
            "name": "gpu_fp64_sgs_hpr",
            "timing_boundary": (
                "prepared resident GPU workspace solve including zero-state upload, iteration "
                "loop, stopping checks, and final state recovery/transfer"
            ),
        },
    )
    if availability["gpu_fp64_sgs_hpr"].get("available") is not True:
        gpu.update(
            {
                "status": "UNAVAILABLE",
                "reason": availability["gpu_fp64_sgs_hpr"].get("reason"),
                "passed": False,
            }
        )
    else:
        try:
            backend = stage7.create_gpu_backend(device_id=device_id)
            backend.synchronize()
            device = backend.diagnostics.as_dict()
            memory = backend.memory_report().as_dict()
            gpu_guard = stage7._memory_guard(
                preflight,
                host_memory=stage7._host_memory(),
                device_total_bytes=int(device["total_global_memory_bytes"]),
            )
            free_budget = int(stage7.DEVICE_SAFETY_FRACTION * int(memory["free_device_bytes"]))
            free_check = preflight.gpu_planning_bytes <= free_budget
            gpu["device"] = device
            gpu["memory_before"] = memory
            gpu["preflight"] = gpu_guard
            gpu["free_device_workspace_gate"] = {
                "observed_free_device_bytes": int(memory["free_device_bytes"]),
                "device_safety_fraction": stage7.DEVICE_SAFETY_FRACTION,
                "device_free_budget_bytes": free_budget,
                "projected_device_bytes": preflight.gpu_planning_bytes,
                "passed": free_check,
                "evaluated_before_gpu_workspace_allocation": True,
            }
            if not gpu_guard["passed"] or not free_check:
                raise MemoryError("GPU workspace preallocation safety guard rejected allocation")
            ledger_before = backend.ledger.summary()
            setup_started = perf_counter()
            problem = stage7._prepare_gpu_problem(
                model, preconditioner, scaled_solver, spectral, backend
            )
            gpu["workspace_setup_wall_seconds"] = perf_counter() - setup_started
            ledger_after = backend.ledger.summary()
            gpu["preparation_transfer_delta"] = stage7._transfer_delta(ledger_before, ledger_after)
            kernels = {
                "A1": problem.workspace.A1_resident.kernel.as_dict(),
                "A2": problem.workspace.A2_resident.kernel.as_dict(),
            }
            kernel_checks = {
                "A1_uses_csr_alg2": bool(kernels["A1"].get("uses_csr_alg2", False)),
                "A2_uses_csr_alg2": bool(kernels["A2"].get("uses_csr_alg2", False)),
                "FP64": problem.dtype_name == "float64",
                "scaled_structural_equality": (
                    problem.workspace.equality_mode == "scaled_structural"
                ),
            }
            gpu["kernel_selection"] = kernels
            gpu["kernel_checks"] = {
                "requested_algorithm": base_config["algorithm"]["requested_spmv_algorithm"],
                **kernel_checks,
                "passed": all(kernel_checks.values()),
            }
            checkpoint()
            if gpu["kernel_checks"]["passed"]:
                stage7._run_timed_track(
                    track=gpu,
                    solve=lambda: stage7._solve_gpu_hpr(problem, config=base_config),
                    summarize=lambda attempt: stage7._summarize_hpr_attempt(
                        attempt,
                        model=model,
                        config=base_config,
                        reference_objective=float(reference),
                    ),
                    config=base_config,
                    checkpoint=checkpoint,
                )
            else:
                gpu.update(
                    {
                        "status": "FAIL",
                        "reason": "the frozen FP64 CSR_ALG2 path was not selected",
                        "passed": False,
                    }
                )
            gpu["memory_after"] = backend.memory_report().as_dict()
            gpu["cumulative_transfer_ledger"] = backend.ledger.summary()
        except Exception as error:
            gpu.update(
                {
                    "status": (
                        "UNAVAILABLE" if isinstance(error, stage7.GPUBackendUnavailable) else "FAIL"
                    ),
                    "failure": stage7._exception_record("gpu_preparation", error),
                    "passed": False,
                }
            )
    checkpoint()

    case["passed"] = all(tracks.get(name, {}).get("passed") is True for name in REQUIRED_TRACKS)
    statuses = {
        tracks.get(name, {}).get("timing_status") or tracks.get(name, {}).get("status")
        for name in REQUIRED_TRACKS
    }
    case["status"] = (
        "PASS" if case["passed"] else ("TIME_LIMIT" if "TIME_LIMIT" in statuses else "FAIL")
    )
    case["resolved"] = True
    case["completed_utc"] = stage7._utc_now()
    case["end_to_end_case_wall_seconds"] = perf_counter() - case_wall_started
    case["timing_boundaries"] = {
        "model_construction_wall_seconds": case["construction"]["wall_seconds"],
        "preprocessing_wall_seconds": case["preprocessing"]["wall_seconds"],
        "cpu_hpr_workspace_setup_wall_seconds": None,
        "gpu_workspace_setup_wall_seconds": gpu.get("workspace_setup_wall_seconds"),
        "end_to_end_case_wall_seconds": case["end_to_end_case_wall_seconds"],
        "solver_core_samples_are_stored_per_track": True,
        "speedup_computed": False,
    }
    checkpoint()


def _case_index(evidence: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("key")): row for row in evidence.get("cases", []) if isinstance(row, dict)}


def _update_status(evidence: dict[str, Any]) -> None:
    cases = _case_index(evidence)
    t16 = cases.get(REQUESTED_KEYS[0], {})
    status = t16.get("status")
    resolved = status in TERMINAL_T16_STATUSES
    if status == "PASS":
        campaign_status = "COMPLETE_WITH_STATIC_RESOURCE_LIMITS"
    elif status == "MEMORY_BLOCKED":
        campaign_status = "COMPLETE_WITH_RESOURCE_LIMITS"
    elif status in {"FAIL", "TIME_LIMIT"}:
        campaign_status = "STOPPED_ON_FAILURE"
    elif status == "RUNNING":
        campaign_status = "RUNNING"
    else:
        campaign_status = "PLANNED"
    evidence["status"] = campaign_status
    evidence["all_passed"] = False
    evidence["all_requested_rows_resolved"] = bool(
        resolved
        and all(cases.get(key, {}).get("status") == "INDEX_BLOCKED" for key in STATIC_BLOCK_KEYS)
    )
    evidence["executable_scope_passed"] = status == "PASS"
    boundary = evidence["stage_boundary"]
    boundary["stage_8_gpu_only_continuation_complete"] = evidence["all_requested_rows_resolved"]
    boundary["stage_9_locked"] = True
    boundary["stage_9_allocation_count"] = 0


def _resume_checkpoint(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    evidence = _load_json(path)
    if evidence.get("run_fingerprint") != fingerprint:
        raise Stage8GPUOnlyResumeError("checkpoint fingerprint differs from this execution")
    t16 = _case_index(evidence).get(REQUESTED_KEYS[0], {})
    if t16.get("status") == "RUNNING" or (
        evidence.get("status") == "RUNNING" and evidence.get("allocation_history")
    ):
        raise Stage8GPUOnlyResumeError(
            "an allocated sequence-6 run stopped interrupted; new user direction is required"
        )
    return evidence


def _run_main(args: argparse.Namespace) -> int:
    if args.run_next and args.approval_token != APPROVAL_GATE:
        raise SystemExit(f"--run-next requires --approval-token {APPROVAL_GATE!r}")
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    partial_path = output_dir / PARTIAL_NAME
    final_path = output_dir / FINAL_NAME
    if not args.resume and (partial_path.exists() or final_path.exists()):
        raise Stage8GPUOnlyResumeError("--no-resume requires a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(config_path)
    errors = _validate_config(config)
    config_identity = stage8._portable_identity(config_path)
    config_sha256 = config_identity.get("canonical_git_blob_sha256")
    if config_sha256 != FROZEN_CONFIG_SHA256:
        errors.append("GPU-only continuation configuration SHA-256 drifted")
    errors.extend(f"configuration identity: {item}" for item in config_identity.get("errors", []))
    original_contract, original = _original_contract(config)
    errors.extend(original_contract["errors"])
    base_contract, base = stage8._base_contract(original["configuration"])
    errors.extend(base_contract["errors"])
    base_provenance = stage7._verify_provenance(
        base["configuration"],
        (PROJECT_ROOT / "configs" / "benchmarks" / "stage_7_small_medium.json").resolve(),
    )
    errors.extend(str(item) for item in base_provenance.get("errors", []))
    policy_contract = stage7._policy_contract(base["configuration"])
    if policy_contract.get("passed") is not True:
        errors.append("frozen Stage 7 reconstruction policy differs from implementation")
    sources = _source_manifest(config_path)
    if any(row.get("passed") is not True for row in sources):
        errors.append("source manifest does not match the clean executed worktree")
    fingerprint = _run_fingerprint(config_sha256, original_contract, sources)
    ledger = _resource_ledger(original["evidence"], base["evidence"])
    environment = stage7._environment()
    availability = _solver_availability()

    resumed = _resume_checkpoint(partial_path, fingerprint) if args.resume else None
    if resumed is not None and errors:
        raise Stage8GPUOnlyResumeError(
            "current preflight differs from the checkpoint: " + "; ".join(errors)
        )
    if resumed is None:
        static_cases = [_static_case(row) for row in ledger if row["key"] in STATIC_BLOCK_KEYS]
        static_failures = [deepcopy(row["failure"]) for row in static_cases]
        evidence: dict[str, Any] = {
            "schema_version": "1.0",
            "stage": 8,
            "campaign": "gpu_only_sequence_6_8_completion",
            "status": "PLANNED",
            "all_passed": False,
            "all_requested_rows_resolved": False,
            "executable_scope_passed": False,
            "started_utc": stage7._utc_now(),
            "run_fingerprint": fingerprint,
            "approval": {
                "gate": APPROVAL_GATE,
                "provided_for_allocation": bool(args.run_next),
            },
            "configuration": config,
            "configuration_validation": {"errors": errors, "passed": not errors},
            "original_stage_8_contract": original_contract,
            "base_stage_7_contract": base_contract,
            "base_stage_7_provenance": base_provenance,
            "policy_contract": policy_contract,
            "environment": environment,
            "source_manifest": sources,
            "requirements_freeze": stage7._requirements_freeze(),
            "solver_availability": availability,
            "resource_ledger": ledger,
            "resource_observations": [],
            "invocations": [],
            "cases": [
                {
                    "key": REQUESTED_KEYS[0],
                    "sequence": 6,
                    "case_name": "case9241pegase",
                    "periods": 16,
                    "status": "PENDING",
                    "passed": False,
                    "resolved": False,
                    "full_lp_allocation_attempted": False,
                    "solver_tracks": {},
                    "skipped_solver_tracks": _skip_records(),
                },
                *static_cases,
            ],
            "allocation_history": [],
            "failures": static_failures,
            "stage_boundary": {
                **config["stage_boundary"],
                "original_stage_8_terminal_status": original_contract["terminal_status"],
                "original_t6_cpu_failure_preserved": True,
                "one_new_full_lp_allocation_per_invocation": True,
                "continuation_allocation_attempt_count": 0,
                "unique_allocated_keys": [],
                "stage_8_gpu_only_continuation_complete": False,
                "stage_9_allocation_count": 0,
                "stage_9_locked": True,
            },
        }
    else:
        evidence = resumed
        evidence["environment"] = environment
        evidence["solver_availability"] = availability
        evidence["approval"]["provided_for_allocation"] = bool(args.run_next)

    invocation_id = stage7._utc_now()
    invocation = {
        "id": invocation_id,
        "started_utc": invocation_id,
        "mode": "run_next" if args.run_next else "plan_only",
        "approval_token_matched": bool(args.run_next),
        "device_id": args.device_id,
        "allocated_keys": [],
        "cpu_hpr_called": False,
        "gurobi_called": False,
    }
    evidence.setdefault("invocations", []).append(invocation)

    def checkpoint() -> None:
        evidence["updated_utc"] = stage7._utc_now()
        stage7._atomic_write_json(partial_path, evidence)

    if errors:
        raise Stage8GPUOnlyContractError("; ".join(errors))
    checkpoint()
    if not args.run_next:
        _update_status(evidence)
        invocation["completed_utc"] = stage7._utc_now()
        invocation["outcome"] = evidence["status"]
        checkpoint()
        return 0

    _update_status(evidence)
    if evidence["all_requested_rows_resolved"]:
        invocation["completed_utc"] = stage7._utc_now()
        invocation["outcome"] = evidence["status"]
        checkpoint()
        stage7._atomic_write_json(final_path, evidence)
        return 0 if evidence["status"] != "STOPPED_ON_FAILURE" else 1

    rows = {str(row["key"]): row for row in evidence["resource_ledger"]}
    t16_row = rows[REQUESTED_KEYS[0]]
    t16_row["allocation_permitted_this_invocation"] = True
    observation = stage8._resource_observation(args.device_id)
    gate = stage8._resource_gate(t16_row["resource_estimate"], observation, config)
    evidence["resource_observations"].append(
        {"key": REQUESTED_KEYS[0], "observed_utc": stage7._utc_now(), **observation}
    )
    t16 = _case_index(evidence)[REQUESTED_KEYS[0]]
    t16["run_fingerprint"] = fingerprint
    t16["resource_estimate"] = deepcopy(t16_row["resource_estimate"])
    t16["stage8_resource_gate"] = gate
    t16["full_lp_allocation_attempted"] = False
    checkpoint()
    if not gate["passed"]:
        failure = {
            "phase": f"preallocation:{REQUESTED_KEYS[0]}",
            "type": "MemorySafetyBlock",
            "message": ", ".join(gate["block_reasons"]),
            "recorded_utc": stage7._utc_now(),
            "full_lp_allocated": False,
        }
        t16.update(
            {
                "status": "MEMORY_BLOCKED",
                "passed": False,
                "resolved": True,
                "failure": failure,
                "skipped_solver_tracks": _skip_records(resource_blocked=True),
                "required_solver_track_disposition": {
                    name: "NOT_RUN_MEMORY_SAFETY_BLOCK" for name in REQUIRED_TRACKS
                },
                "completed_utc": stage7._utc_now(),
            }
        )
        evidence["failures"].append(failure)
    else:
        t16["full_lp_allocation_attempted"] = True
        evidence["allocation_history"].append(
            {
                "key": REQUESTED_KEYS[0],
                "sequence": 6,
                "attempted_utc": stage7._utc_now(),
                "invocation_id": invocation_id,
                "preallocation_gate_passed": True,
            }
        )
        invocation["allocated_keys"].append(REQUESTED_KEYS[0])
        boundary = evidence["stage_boundary"]
        boundary["continuation_allocation_attempt_count"] = 1
        boundary["unique_allocated_keys"] = [REQUESTED_KEYS[0]]
        checkpoint()
        try:
            _run_gpu_only_case(
                t16,
                base_config=base["configuration"],
                availability=availability,
                device_id=args.device_id,
                checkpoint=checkpoint,
            )
        except Exception as error:
            failure = stage7._exception_record(f"case:{REQUESTED_KEYS[0]}", error)
            t16.update(
                {
                    "status": "FAIL",
                    "passed": False,
                    "resolved": True,
                    "failure": failure,
                    "completed_utc": stage7._utc_now(),
                }
            )
            evidence["failures"].append(failure)
        if t16.get("status") != "PASS" and t16.get("failure") is None:
            failure = {
                "phase": f"case:{REQUESTED_KEYS[0]}",
                "type": "RecordedSolverFailure",
                "message": f"terminal case status {t16.get('status')}",
                "recorded_utc": stage7._utc_now(),
                "full_lp_allocated": True,
            }
            t16["failure"] = failure
            evidence["failures"].append(failure)

    _update_status(evidence)
    invocation["completed_utc"] = stage7._utc_now()
    invocation["outcome"] = evidence["status"]
    evidence["completed_utc"] = stage7._utc_now()
    checkpoint()
    stage7._atomic_write_json(final_path, evidence)
    return 0 if evidence["status"] != "STOPPED_ON_FAILURE" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with stage8._exclusive_output_lock(args.output_dir.resolve()):
            return _run_main(args)
    except (Stage8GPUOnlyContractError, stage8.Stage8ConcurrentRunError) as error:
        print(f"Stage 8 GPU-only continuation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
