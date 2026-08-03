"""Run one fail-closed increment of the approved Stage 8 large campaign.

The runner deliberately permits at most one new Table II allocation per
invocation.  It authenticates and reuses the frozen Stage 7 construction,
algorithm, numerical gates, and timing protocol; creates a no-allocation
resource ledger first; requires every predecessor to have passed; and writes
an atomic checkpoint after every material event.  A failed or unsafe row is
preserved exactly and stops scale-up.  No case selector exists by design.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpu_dcopf_hpr.gpu_backend import create_gpu_backend  # noqa: E402
from gpu_dcopf_hpr.stage7_scalable_model import (  # noqa: E402
    Stage7Preflight,
    all_stage7_preflights,
)
from scripts import check_stage_7 as stage7_checker  # noqa: E402
from scripts import run_stage_7 as stage7  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "benchmarks" / "stage_8_large.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_8"
PARTIAL_NAME = "stage_8_validation.partial.json"
FINAL_NAME = "stage_8_validation.json"
APPROVAL_GATE = "APPROVE STAGE 7 AND RUN STAGE 8"
FROZEN_CONFIG_SHA256 = "ac61cf282fe3f146a1fbe5e2d1bff87b4ce36641d30f6feaaf3a64d48bd04284"
FROZEN_STAGE7_CONFIG_SHA256 = stage7.FROZEN_CONFIG_SHA256
FROZEN_STAGE7_EVIDENCE_SHA256 = "180699f6b34228c3e1a69b158677b12dd3242582a7225e1cdb44aeaac29931ae"
FROZEN_REQUIREMENTS_SHA256 = stage7.FROZEN_REQUIREMENTS_SHA256
CAMPAIGN_ORDER = (
    "case2868rte:T48",
    "case2868rte:T64",
    "case2868rte:T96",
    "case9241pegase:T4",
    "case9241pegase:T6",
    "case9241pegase:T16",
    "case9241pegase:T24",
    "case9241pegase:T32",
)
RECONCILIATION_ONLY_ROWS = (
    "case2868rte:T56",
    "case2868rte:T72",
    "case2868rte:T80",
    "case2868rte:T88",
)
STATIC_CSR32_BLOCKS = frozenset({"case9241pegase:T24", "case9241pegase:T32"})
TERMINAL_STATUSES = frozenset({"PASS", "FAIL", "TIME_LIMIT", "MEMORY_BLOCKED"})
INTERRUPTED_STATUSES = frozenset({"PENDING", "RUNNING"})


class Stage8ContractError(ValueError):
    """The approved Stage 8 execution or frozen Stage 7 base drifted."""


class Stage8ResumeError(Stage8ContractError):
    """An existing checkpoint is incompatible or internally inconsistent."""


class Stage8ConcurrentRunError(Stage8ContractError):
    """Another process already owns this output directory's campaign lock."""


@contextlib.contextmanager
def _exclusive_output_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".stage_8.lock"
    stream = lock_path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise Stage8ConcurrentRunError(
                    f"another Stage 8 process holds {lock_path}"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise Stage8ConcurrentRunError(
                    f"another Stage 8 process holds {lock_path}"
                ) from error
        yield
    finally:
        with contextlib.suppress(OSError):
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device-id", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-next",
        action="store_true",
        help="Run only the next authorized campaign row after resource preflight.",
    )
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the complete no-allocation resource plan (the default).",
    )
    parser.add_argument(
        "--approval-token",
        help="Exact Stage 8 approval gate; required with --run-next.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume only a checkpoint with the identical run fingerprint.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry the same stopped row while preserving its prior evidence.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _portable_identity(path: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return {
            "canonical_git_blob_sha256": None,
            "errors": ["path is outside the repository"],
            "passed": False,
        }
    return stage7._canonical_git_blob_identity(relative, path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage8ContractError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise Stage8ContractError(f"{path} must contain one JSON object")
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _key(preflight: Stage7Preflight) -> str:
    return f"{preflight.row.case_name}:T{preflight.row.periods}"


def _split_key(key: str) -> stage7.CaseKey:
    name, raw_periods = key.rsplit(":T", 1)
    return stage7.CaseKey(name, int(raw_periods))


def _validate_stage8_config(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_top = {
        "schema_version": "1.0",
        "stage": 8,
        "name": "stage_8_large_structural_reproduction",
        "classification": "structural_reproduction",
        "approval_gate": APPROVAL_GATE,
    }
    for name, expected in expected_top.items():
        if config.get(name) != expected:
            errors.append(f"{name} drifted from the frozen Stage 8 contract")

    base = config.get("base_stage_7", {})
    expected_base = {
        "configuration_path": "configs/benchmarks/stage_7_small_medium.json",
        "configuration_sha256": FROZEN_STAGE7_CONFIG_SHA256,
        "evidence_path": "results/raw/stage_7/stage_7_validation.json",
        "evidence_sha256": FROZEN_STAGE7_EVIDENCE_SHA256,
        "requirements_path": "environment/dgx_stage7_requirements.txt",
        "requirements_sha256": FROZEN_REQUIREMENTS_SHA256,
        "reuse_reconstruction_protocol_without_changes": True,
        "reuse_algorithm_without_changes": True,
        "reuse_acceptance_thresholds_without_changes": True,
        "reuse_timing_protocol_without_changes": True,
    }
    if base != expected_base:
        errors.append("base_stage_7 differs from the frozen Stage 7 inheritance contract")

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
    }
    if resource != expected_resource:
        errors.append("resource_policy drifted from the fail-closed memory contract")

    execution = config.get("execution_policy", {})
    expected_execution = {
        "one_new_case_per_invocation": True,
        "strict_predecessor_pass_required": True,
        "stop_after_failure_or_memory_block": True,
        "resume_by_run_fingerprint": True,
        "retry_requires_explicit_flag": True,
        "do_not_replace_failures_with_estimates": True,
    }
    if execution != expected_execution:
        errors.append("execution_policy drifted from the incremental campaign contract")
    if tuple(config.get("campaign_order", ())) != CAMPAIGN_ORDER:
        errors.append("campaign_order differs from the approved Stage 8 sequence")
    if tuple(config.get("reconciliation_only_rows", ())) != RECONCILIATION_ONLY_ROWS:
        errors.append("reconciliation_only_rows drifted")
    if config.get("expected_preallocation_blocks") != {
        "case9241pegase:T24": "signed_int32_csr_nnz_limit",
        "case9241pegase:T32": "signed_int32_csr_nnz_limit",
    }:
        errors.append("expected_preallocation_blocks drifted")
    if config.get("stage_boundary") != {
        "stage_8_only": True,
        "stage_9_locked": True,
        "exact_paper_reproduction_claimed": False,
        "paper_a100_timing_reproduction_claimed": False,
        "n_minus_1_extension_enabled": False,
        "speedup_claim_requires_matching_formulation_hardware_and_boundaries": True,
    }:
        errors.append("stage_boundary drifted")
    return errors


def _base_contract(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = config["base_stage_7"]
    paths = {
        name: (PROJECT_ROOT / str(base[f"{name}_path"])).resolve()
        for name in ("configuration", "evidence", "requirements")
    }
    identities = {name: _portable_identity(path) for name, path in paths.items()}
    hashes = {
        name: identity.get("canonical_git_blob_sha256") for name, identity in identities.items()
    }
    expected = {
        "configuration": FROZEN_STAGE7_CONFIG_SHA256,
        "evidence": FROZEN_STAGE7_EVIDENCE_SHA256,
        "requirements": FROZEN_REQUIREMENTS_SHA256,
    }
    errors = [
        f"frozen Stage 7 {name} SHA-256 mismatch"
        for name in expected
        if hashes[name] != expected[name]
    ]
    errors.extend(
        f"frozen Stage 7 {name} identity: {error}"
        for name, identity in identities.items()
        for error in identity.get("errors", [])
    )
    base_config = _load_json(paths["configuration"])
    errors.extend(stage7._validate_stage7_config(base_config))
    evidence = _load_json(paths["evidence"])
    if not (
        evidence.get("stage") == 7
        and evidence.get("status") == "PASS"
        and evidence.get("all_passed") is True
        and len(evidence.get("cases", [])) == stage7.EXECUTABLE_CASE_COUNT
    ):
        errors.append("frozen Stage 7 evidence is not the accepted six-case PASS artifact")
    result = {
        "paths": {name: path.relative_to(PROJECT_ROOT).as_posix() for name, path in paths.items()},
        "sha256": hashes,
        "expected_sha256": expected,
        "portable_identities": identities,
        "errors": errors,
        "passed": not errors,
    }
    return result, {"configuration": base_config, "evidence": evidence}


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
        (PROJECT_ROOT / "scripts" / "run_stage_7.py").resolve(),
        (PROJECT_ROOT / "scripts" / "check_stage_7.py").resolve(),
        config_path,
        (PROJECT_ROOT / "configs" / "benchmarks" / "stage_7_small_medium.json").resolve(),
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
    base_contract: Mapping[str, Any],
    source_manifest: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "stage8_config_sha256": config_sha256,
        "base_stage7_sha256": base_contract.get("sha256"),
        "sources": list(source_manifest),
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "cupy_cuda13x": _package_version("cupy-cuda13x"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _exact_nnz_by_key(stage7_evidence: Mapping[str, Any]) -> dict[str, int]:
    rows = stage7_evidence.get("symbolic_ledger", [])
    result: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = row.get("actual_reconstruction_nnz")
        if value is None:
            value = row.get("symbolic_nnz", {}).get("reconstructed_nnz")
        if value is not None:
            result[str(row.get("key"))] = int(value)
    if len(result) != stage7.TABLE_ROW_COUNT:
        raise Stage8ContractError("Stage 7 evidence does not contain all 18 exact nnz counts")
    return result


def _resource_estimate(preflight: Stage7Preflight, exact_nnz: int) -> dict[str, Any]:
    sparse = int(preflight.gpu_matrix_and_transpose_bytes)
    vectors = int(preflight.gpu_vector_bytes)
    projected = int(preflight.gpu_planning_bytes)
    temporary = projected - sparse - vectors
    combined = int(preflight.host_assembly_peak_bytes) + projected
    return {
        "case_name": preflight.row.case_name,
        "periods": preflight.row.periods,
        "row_count": preflight.computed_m,
        "column_count": preflight.computed_n,
        "paper_nnz": preflight.row.published_nnz,
        "exact_reconstructed_nnz": exact_nnz,
        "conservative_planning_nnz": preflight.planning_nnz,
        "csr_matrix_bytes": preflight.csr_one_orientation_bytes,
        "csr_transpose_bytes": preflight.csr_one_orientation_bytes,
        "iterate_and_workspace_vector_bytes": vectors,
        "temporary_buffers_and_headroom_bytes": temporary,
        "projected_device_bytes": projected,
        "projected_host_assembly_peak_bytes": preflight.host_assembly_peak_bytes,
        "projected_unified_peak_bytes": combined,
        "csr_index_bits": 32,
        "signed_int32_csr_supported": preflight.csr32_supported,
        "dimensions_match_table": preflight.dimensions_match_table,
        "estimate_kind": "conservative_dense_support_bound_with_stage7_headroom",
        "full_lp_allocated": False,
    }


def _resource_ledger(stage7_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    exact = _exact_nnz_by_key(stage7_evidence)
    stage7_keys = {str(case.get("key")) for case in stage7_evidence.get("cases", [])}
    rows: list[dict[str, Any]] = []
    for preflight in all_stage7_preflights():
        key = _key(preflight)
        if key in stage7_keys:
            disposition = "stage7_completed"
        elif key in CAMPAIGN_ORDER:
            disposition = "stage8_campaign"
        else:
            disposition = "reconciliation_only"
        static_reasons = [] if preflight.csr32_supported else ["signed_int32_csr_nnz_limit"]
        rows.append(
            {
                "key": key,
                "campaign_sequence": (
                    CAMPAIGN_ORDER.index(key) + 1 if key in CAMPAIGN_ORDER else None
                ),
                "disposition": disposition,
                "allocation_permitted_this_invocation": False,
                "static_preallocation_status": "BLOCKED" if static_reasons else "ELIGIBLE",
                "static_block_reasons": static_reasons,
                "resource_estimate": _resource_estimate(preflight, exact[key]),
            }
        )
    return rows


def _resource_observation(device_id: int) -> dict[str, Any]:
    host = stage7._host_memory()
    device_total: int | None = None
    device_free: int | None = None
    device: dict[str, Any] = {}
    errors = list(host.get("errors", []))
    try:
        backend = create_gpu_backend(device_id=device_id)
        backend.synchronize()
        device = backend.diagnostics.as_dict()
        device_total = int(device["total_global_memory_bytes"])
        device_free = int(backend.memory_report().free_device_bytes)
    except Exception as error:  # evidence must preserve backend/import/runtime failures
        errors.append(f"{type(error).__name__}: {error}")
    host_available = host.get("available_bytes")
    if not isinstance(host_available, int) or host_available <= 0:
        errors.append("positive observed host available bytes are required")
    if not isinstance(device_total, int) or device_total <= 0:
        errors.append("positive observed device total bytes are required")
    if not isinstance(device_free, int) or device_free <= 0:
        errors.append("positive observed device free bytes are required")
    return {
        "host": host,
        "device": device,
        "device_total_bytes": device_total,
        "device_free_bytes": device_free,
        "errors": errors,
        "passed": not errors,
    }


def _resource_gate(
    estimate: Mapping[str, Any],
    observation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    policy = config["resource_policy"]
    host_available = observation.get("host", {}).get("available_bytes")
    device_total = observation.get("device_total_bytes")
    device_free = observation.get("device_free_bytes")
    host_budget = (
        int(float(policy["host_safety_fraction"]) * host_available)
        if isinstance(host_available, int) and host_available > 0
        else None
    )
    device_budget = (
        int(float(policy["device_safety_fraction"]) * device_free)
        if isinstance(device_free, int) and device_free > 0
        else None
    )
    projected = int(estimate["projected_unified_peak_bytes"])
    checks = {
        "observation_available": observation.get("passed") is True,
        "dimensions_match_table": estimate.get("dimensions_match_table") is True,
        "signed_int32_csr_supported": estimate.get("signed_int32_csr_supported") is True,
        "within_host_safety_budget": host_budget is not None and projected <= host_budget,
        "within_device_safety_budget": device_budget is not None and projected <= device_budget,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "observation": observation,
        "host_safety_budget_bytes": host_budget,
        "device_safety_budget_bytes": device_budget,
        "observed_device_total_bytes": device_total,
        "observed_device_free_bytes": device_free,
        "projected_unified_peak_bytes": projected,
        "checks": checks,
        "block_reasons": reasons,
        "passed": all(checks.values()),
        "evaluated_before_full_lp_allocation": True,
    }


def _compatible_partial(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    candidate = _load_json(path)
    if candidate.get("run_fingerprint") != fingerprint:
        raise Stage8ResumeError(
            "existing Stage 8 checkpoint has a different run fingerprint; refusing overwrite"
        )
    return candidate


def _checkpoint_success_valid(
    case: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    ledger_row: Mapping[str, Any],
    solver_availability: Mapping[str, Any],
) -> bool:
    key = str(case.get("key"))
    parsed = _split_key(key)
    estimate = ledger_row.get("resource_estimate", {})
    gate = case.get("stage8_resource_gate", {})
    if not isinstance(estimate, Mapping) or not isinstance(gate, Mapping):
        return False
    observation = gate.get("observation", {})
    if not isinstance(observation, Mapping) or gate != _resource_gate(
        estimate, observation, config
    ):
        return False
    expected_m = int(estimate["row_count"])
    expected_n = int(estimate["column_count"])
    expected_nnz = int(estimate["exact_reconstructed_nnz"])
    paper_nnz = int(estimate["paper_nnz"])
    spec = next(
        (item for item in base_config.get("cases", []) if item.get("case") == parsed.case_name),
        None,
    )
    if not isinstance(spec, Mapping):
        return False
    storage = int(spec["storage"])
    m1 = parsed.periods + storage
    m2 = expected_m - m1
    construction = case.get("construction", {})
    dimensions = construction.get("dimensions", {}) if isinstance(construction, Mapping) else {}
    reconciliation = case.get("structural_reconciliation", {})
    tracks = case.get("solver_tracks", {})
    if not all(isinstance(value, Mapping) for value in (dimensions, reconciliation, tracks)):
        return False
    highs = tracks.get("highs", {})
    correctness = highs.get("correctness", {}) if isinstance(highs, Mapping) else {}
    candidate = correctness.get("candidate", {}) if isinstance(correctness, Mapping) else {}
    reference = candidate.get("objective") if isinstance(candidate, Mapping) else None
    gurobi = solver_availability.get("gurobi", {})
    gurobi_available = bool(gurobi.get("available")) if isinstance(gurobi, Mapping) else False
    if not (
        case.get("status") == "PASS"
        and case.get("passed") is True
        and case.get("full_lp_allocation_attempted") is True
        and gate.get("passed") is True
        and case.get("case_name") == parsed.case_name
        and case.get("periods") == parsed.periods
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
        and stage7_checker._preprocessing_valid(
            case.get("preprocessing"),
            periods=parsed.periods,
            storage=storage,
            m1=m1,
            m2=m2,
            n=expected_n,
        )
        and stage7_checker._finite(reference)
    ):
        return False
    reference_float = float(reference)
    timing = case.get("timing_boundaries", {})
    return bool(
        set(tracks) == {*stage7_checker.REQUIRED_TRACKS, "gurobi"}
        and stage7_checker._track_valid(highs, track_name="highs", reference_objective=None)
        and stage7_checker._track_valid(
            tracks.get("cpu_fp64_sgs_hpr"),
            track_name="cpu_fp64_sgs_hpr",
            reference_objective=reference_float,
        )
        and stage7_checker._track_valid(
            tracks.get("gpu_fp64_sgs_hpr"),
            track_name="gpu_fp64_sgs_hpr",
            reference_objective=reference_float,
        )
        and stage7_checker._gurobi_track_valid(
            tracks.get("gurobi"),
            available=gurobi_available,
            reference_objective=reference_float,
        )
        and isinstance(timing, Mapping)
        and timing.get("solver_core_samples_are_stored_per_track") is True
        and timing.get("speedup_computed") is False
    )


def _validate_resume_checkpoint(
    evidence: Mapping[str, Any],
    *,
    fingerprint: str,
    config: Mapping[str, Any],
    base_contract: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    expected_ledger: Sequence[Mapping[str, Any]],
    base_config: Mapping[str, Any],
    solver_availability: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not (
        evidence.get("schema_version") == "1.0"
        and evidence.get("stage") == 8
        and evidence.get("run_fingerprint") == fingerprint
    ):
        errors.append("checkpoint schema, stage, or fingerprint is invalid")
    if evidence.get("configuration") != config:
        errors.append("checkpoint embedded configuration drifted")
    if evidence.get("base_stage_7_contract") != base_contract:
        errors.append("checkpoint Stage 7 base contract drifted")
    if evidence.get("source_manifest") != list(sources):
        errors.append("checkpoint executable source manifest drifted")

    actual_ledger = deepcopy(evidence.get("resource_ledger", []))
    for row in actual_ledger:
        if isinstance(row, dict):
            row["allocation_permitted_this_invocation"] = False
    if actual_ledger != list(expected_ledger):
        errors.append("checkpoint resource ledger contains an immutable-field mutation")

    cases = evidence.get("cases", [])
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        errors.append("checkpoint cases must be a list of objects")
        cases = []
    keys = [str(case.get("key")) for case in cases]
    if len(keys) != len(set(keys)) or not set(keys).issubset(set(CAMPAIGN_ORDER)):
        errors.append("checkpoint cases are duplicated or outside the campaign")
    encountered_nonpass = False
    allowed_statuses = TERMINAL_STATUSES | INTERRUPTED_STATUSES
    ledger_by_key = {str(row.get("key")): row for row in expected_ledger}
    for key in CAMPAIGN_ORDER:
        case = next((item for item in cases if item.get("key") == key), None)
        if case is None:
            encountered_nonpass = True
            continue
        if encountered_nonpass:
            errors.append(f"checkpoint case {key} bypasses an incomplete predecessor")
            break
        status = case.get("status")
        if status not in allowed_statuses:
            errors.append(f"checkpoint case {key} has invalid status {status!r}")
        if (status == "PASS") is not (case.get("passed") is True):
            errors.append(f"checkpoint case {key} status/passed fields disagree")
        if status == "PASS" and not _checkpoint_success_valid(
            case,
            config=config,
            base_config=base_config,
            ledger_row=ledger_by_key[key],
            solver_availability=solver_availability,
        ):
            errors.append(f"checkpoint PASS case {key} failed deep numerical/timing validation")
        if case.get("run_fingerprint") != fingerprint:
            errors.append(f"checkpoint case {key} fingerprint drifted")
        if status != "PASS":
            encountered_nonpass = True

    history = evidence.get("allocation_history", [])
    if not isinstance(history, list) or not all(isinstance(row, dict) for row in history):
        errors.append("checkpoint allocation history must be a list of objects")
        history = []
    unique_history: list[str] = []
    for row in history:
        key = str(row.get("key"))
        if key not in CAMPAIGN_ORDER or row.get("preallocation_gate_passed") is not True:
            errors.append("checkpoint allocation history contains an unauthorized attempt")
            continue
        if key not in unique_history:
            unique_history.append(key)
    if tuple(unique_history) != CAMPAIGN_ORDER[: len(unique_history)]:
        errors.append("checkpoint allocation history is not a campaign prefix")
    boundary = evidence.get("stage_boundary", {})
    if not isinstance(boundary, Mapping):
        errors.append("checkpoint stage boundary is invalid")
        boundary = {}
    if boundary.get("stage_8_allocation_attempt_count") != len(history):
        errors.append("checkpoint allocation attempt count disagrees with history")
    if boundary.get("unique_allocated_keys") != unique_history:
        errors.append("checkpoint unique allocated keys disagree with history")
    if boundary.get("reconciliation_only_allocation_count") != 0:
        errors.append("checkpoint allocated a reconciliation-only row")
    if boundary.get("stage_9_allocation_count") != 0:
        errors.append("checkpoint crossed the Stage 9 boundary")

    invocations = evidence.get("invocations", [])
    if not isinstance(invocations, list) or not all(
        isinstance(invocation, dict) for invocation in invocations
    ):
        errors.append("checkpoint invocations must be a list of objects")
    else:
        ids = [str(invocation.get("id")) for invocation in invocations]
        if len(ids) != len(set(ids)):
            errors.append("checkpoint invocation identifiers are duplicated")
        if any(len(invocation.get("allocated_keys", [])) > 1 for invocation in invocations):
            errors.append("checkpoint records more than one allocation in an invocation")
        invocation_by_id = {str(invocation.get("id")): invocation for invocation in invocations}
        for row in history:
            key = str(row.get("key"))
            invocation = invocation_by_id.get(str(row.get("invocation_id")))
            if not (
                invocation
                and invocation.get("mode") == "run_next"
                and invocation.get("approval_token_matched") is True
                and invocation.get("allocated_keys") == [key]
                and invocation.get("retry_failed") is bool(row.get("retry"))
                and row.get("sequence") == CAMPAIGN_ORDER.index(key) + 1
            ):
                errors.append(f"checkpoint allocation {key} lacks an approved invocation link")
    return errors


def _assert_clean_no_resume_target(partial_path: Path, final_path: Path) -> None:
    existing = [path for path in (partial_path, final_path) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise Stage8ContractError(
            f"--no-resume refuses to overwrite existing Stage 8 artifacts: {joined}"
        )


def _case_index(evidence: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("key")): case for case in evidence.get("cases", []) if isinstance(case, dict)
    }


def _next_key(evidence: Mapping[str, Any], *, retry_failed: bool) -> str | None:
    cases = _case_index(evidence)
    for position, key in enumerate(CAMPAIGN_ORDER):
        predecessors = CAMPAIGN_ORDER[:position]
        if any(cases.get(previous, {}).get("status") != "PASS" for previous in predecessors):
            return None
        case = cases.get(key)
        if case is None:
            return key
        status = case.get("status")
        if status == "PASS":
            continue
        if retry_failed and status in TERMINAL_STATUSES | INTERRUPTED_STATUSES:
            return key
        return None
    return None


def _fresh_retry(case: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    history = list(case.get("retry_history", []))
    prior_case = deepcopy(dict(case))
    prior_case.pop("retry_history", None)
    history.append(
        {
            "recorded_utc": stage7._utc_now(),
            "prior_status": case.get("status"),
            "prior_passed": case.get("passed"),
            "prior_failure": deepcopy(case.get("failure")),
            "prior_resource_gate": deepcopy(case.get("stage8_resource_gate")),
            "prior_case": prior_case,
        }
    )
    key = str(case["key"])
    parsed = _split_key(key)
    return {
        "key": key,
        "case_name": parsed.case_name,
        "periods": parsed.periods,
        "run_fingerprint": fingerprint,
        "status": "PENDING",
        "passed": False,
        "retry_history": history,
    }


def _update_campaign_status(evidence: dict[str, Any]) -> None:
    cases = _case_index(evidence)
    passing_prefix = 0
    for key in CAMPAIGN_ORDER:
        if cases.get(key, {}).get("status") == "PASS":
            passing_prefix += 1
        else:
            break
    stopped = next(
        (
            cases[key]
            for key in CAMPAIGN_ORDER
            if cases.get(key, {}).get("status") in {"FAIL", "TIME_LIMIT", "MEMORY_BLOCKED"}
        ),
        None,
    )
    interrupted = next(
        (
            cases[key]
            for key in CAMPAIGN_ORDER
            if cases.get(key, {}).get("status") in INTERRUPTED_STATUSES
        ),
        None,
    )
    complete = bool(passing_prefix == len(CAMPAIGN_ORDER) or stopped is not None)
    evidence["stage_boundary"].update(
        {
            "passing_prefix_length": passing_prefix,
            "next_authorized_key": (
                None
                if complete
                else CAMPAIGN_ORDER[passing_prefix]
                if passing_prefix < len(CAMPAIGN_ORDER)
                else None
            ),
            "stage_8_complete": complete,
            "retry_required_key": None if interrupted is None else interrupted.get("key"),
        }
    )
    if stopped is not None:
        evidence["status"] = (
            "COMPLETE_WITH_RESOURCE_LIMIT"
            if stopped.get("status") == "MEMORY_BLOCKED"
            else "STOPPED_ON_FAILURE"
        )
        evidence["all_passed"] = False
    elif interrupted is not None:
        evidence["status"] = "RETRY_REQUIRED"
        evidence["all_passed"] = False
        evidence["stage_boundary"]["next_authorized_key"] = None
    elif passing_prefix == len(CAMPAIGN_ORDER):
        evidence["status"] = "PASS"
        evidence["all_passed"] = True
    elif passing_prefix:
        evidence["status"] = "PARTIAL_PASS"
        evidence["all_passed"] = False
    else:
        evidence["status"] = "PLANNED"
        evidence["all_passed"] = False


def _run_main(args: argparse.Namespace) -> int:
    if args.run_next and args.approval_token != APPROVAL_GATE:
        raise SystemExit(f"--run-next requires --approval-token {APPROVAL_GATE!r}")
    if args.retry_failed and not args.run_next:
        raise SystemExit("--retry-failed is valid only with --run-next")

    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    partial_path = output_dir / PARTIAL_NAME
    final_path = output_dir / FINAL_NAME
    if not args.resume:
        _assert_clean_no_resume_target(partial_path, final_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_json(config_path)
    errors = _validate_stage8_config(config)
    config_identity = _portable_identity(config_path)
    config_sha256 = config_identity.get("canonical_git_blob_sha256")
    if config_sha256 != FROZEN_CONFIG_SHA256:
        errors.append("Stage 8 configuration SHA-256 drifted")
    errors.extend(f"Stage 8 configuration identity: {item}" for item in config_identity["errors"])
    base_contract, base = _base_contract(config)
    errors.extend(base_contract["errors"])
    policy_contract = stage7._policy_contract(base["configuration"])
    if not policy_contract["passed"]:
        errors.append("Stage 7 reconstruction policy differs from its frozen implementation")
    base_provenance = stage7._verify_provenance(
        base["configuration"],
        (PROJECT_ROOT / config["base_stage_7"]["configuration_path"]).resolve(),
    )
    if not base_provenance["passed"]:
        errors.extend(str(item) for item in base_provenance["errors"])
    sources = _source_manifest(config_path)
    if any(row.get("passed") is not True for row in sources):
        errors.append("canonical Stage 8 source manifest does not match the executed worktree")
    fingerprint = _run_fingerprint(config_sha256, base_contract, sources)
    resumed = _compatible_partial(partial_path, fingerprint) if args.resume else None
    expected_ledger = _resource_ledger(base["evidence"])
    current_environment = stage7._environment()
    current_solver_availability = stage7._solver_availability()
    if resumed is not None:
        resume_errors = _validate_resume_checkpoint(
            resumed,
            fingerprint=fingerprint,
            config=config,
            base_contract=base_contract,
            sources=sources,
            expected_ledger=expected_ledger,
            base_config=base["configuration"],
            solver_availability=resumed.get("solver_availability", current_solver_availability),
        )
        if resume_errors:
            raise Stage8ResumeError("; ".join(resume_errors))
        if errors:
            raise Stage8ResumeError(
                "current preflight differs from the resumable checkpoint: " + "; ".join(errors)
            )
    evidence: dict[str, Any] = resumed or {
        "schema_version": "1.0",
        "stage": 8,
        "status": "PLANNING",
        "all_passed": False,
        "started_utc": stage7._utc_now(),
        "run_fingerprint": fingerprint,
        "approval": {
            "gate": APPROVAL_GATE,
            "provided_for_allocation": bool(args.run_next),
        },
        "configuration": config,
        "configuration_validation": {"errors": errors, "passed": not errors},
        "base_stage_7_contract": base_contract,
        "base_stage_7_provenance": base_provenance,
        "policy_contract": policy_contract,
        "environment": current_environment,
        "source_manifest": sources,
        "requirements_freeze": stage7._requirements_freeze(),
        "solver_availability": current_solver_availability,
        "resource_ledger": expected_ledger,
        "resource_observations": [],
        "invocations": [],
        "cases": [],
        "allocation_history": [],
        "failures": [],
        "stage_boundary": {
            **config["stage_boundary"],
            "one_new_case_per_invocation": True,
            "stage_8_allocation_attempt_count": 0,
            "unique_allocated_keys": [],
            "reconciliation_only_allocation_count": 0,
            "stage_9_allocation_count": 0,
            "passing_prefix_length": 0,
            "next_authorized_key": CAMPAIGN_ORDER[0],
            "retry_required_key": None,
            "stage_8_complete": False,
        },
    }
    for ledger_row in evidence["resource_ledger"]:
        ledger_row["allocation_permitted_this_invocation"] = False
    evidence["environment"] = current_environment
    evidence["solver_availability"] = current_solver_availability
    invocation_id = stage7._utc_now()
    invocation = {
        "id": invocation_id,
        "started_utc": invocation_id,
        "mode": "run_next" if args.run_next else "plan_only",
        "approval_token_matched": bool(args.run_next),
        "retry_failed": bool(args.retry_failed),
        "device_id": args.device_id,
        "environment": current_environment,
        "solver_availability": current_solver_availability,
        "base_provenance_passed": base_provenance.get("passed") is True,
        "source_manifest_passed": all(row.get("passed") is True for row in sources),
        "allocated_keys": [],
    }
    evidence.setdefault("invocations", []).append(invocation)
    evidence["approval"]["provided_for_allocation"] = bool(args.run_next)

    def checkpoint() -> None:
        evidence["updated_utc"] = stage7._utc_now()
        stage7._atomic_write_json(partial_path, evidence)

    try:
        if errors:
            raise Stage8ContractError("; ".join(errors))
        checkpoint()
        if not args.run_next:
            _update_campaign_status(evidence)
            invocation["completed_utc"] = stage7._utc_now()
            invocation["outcome"] = evidence["status"]
            evidence["completed_utc"] = stage7._utc_now()
            checkpoint()
            stage7._atomic_write_json(final_path, evidence)
            return 0

        next_key = _next_key(evidence, retry_failed=args.retry_failed)
        if next_key is None:
            _update_campaign_status(evidence)
            invocation["completed_utc"] = stage7._utc_now()
            invocation["outcome"] = evidence["status"]
            evidence["completed_utc"] = stage7._utc_now()
            checkpoint()
            stage7._atomic_write_json(final_path, evidence)
            return 1 if evidence["status"] in {"STOPPED_ON_FAILURE", "RETRY_REQUIRED"} else 0

        ledger = {str(row["key"]): row for row in evidence["resource_ledger"]}
        if next_key not in CAMPAIGN_ORDER or next_key in RECONCILIATION_ONLY_ROWS:
            raise Stage8ContractError(f"{next_key} is not an authorized Stage 8 allocation")
        row = ledger[next_key]
        row["allocation_permitted_this_invocation"] = True
        observation = _resource_observation(args.device_id)
        gate = _resource_gate(row["resource_estimate"], observation, config)
        evidence["resource_observations"].append(
            {"key": next_key, "observed_utc": stage7._utc_now(), **observation}
        )
        cases = _case_index(evidence)
        case = cases.get(next_key)
        if case is None:
            parsed = _split_key(next_key)
            case = {
                "key": next_key,
                "case_name": parsed.case_name,
                "periods": parsed.periods,
                "run_fingerprint": fingerprint,
                "status": "PENDING",
                "passed": False,
            }
            evidence["cases"].append(case)
        elif args.retry_failed:
            replacement = _fresh_retry(case, fingerprint)
            evidence["cases"][evidence["cases"].index(case)] = replacement
            case = replacement
        case["stage8_resource_gate"] = gate
        case["full_lp_allocation_attempted"] = False
        checkpoint()
        if not gate["passed"]:
            failure = {
                "phase": f"preallocation:{next_key}",
                "type": "MemorySafetyBlock",
                "message": ", ".join(gate["block_reasons"]),
                "recorded_utc": stage7._utc_now(),
                "full_lp_allocated": False,
            }
            case.update(
                {
                    "status": "MEMORY_BLOCKED",
                    "passed": False,
                    "failure": failure,
                    "completed_utc": stage7._utc_now(),
                }
            )
            evidence["failures"].append(failure)
            _update_campaign_status(evidence)
            invocation["completed_utc"] = stage7._utc_now()
            invocation["outcome"] = evidence["status"]
            evidence["completed_utc"] = stage7._utc_now()
            checkpoint()
            stage7._atomic_write_json(final_path, evidence)
            return 0

        case["full_lp_allocation_attempted"] = True
        evidence["allocation_history"].append(
            {
                "key": next_key,
                "sequence": CAMPAIGN_ORDER.index(next_key) + 1,
                "attempted_utc": stage7._utc_now(),
                "retry": bool(args.retry_failed),
                "invocation_id": invocation_id,
                "preallocation_gate_passed": True,
            }
        )
        invocation["allocated_keys"].append(next_key)
        boundary = evidence["stage_boundary"]
        boundary["stage_8_allocation_attempt_count"] += 1
        boundary["unique_allocated_keys"] = list(
            dict.fromkeys([*boundary["unique_allocated_keys"], next_key])
        )
        checkpoint()
        parsed = _split_key(next_key)
        try:
            stage7._run_case(
                case,
                key=parsed,
                config=base["configuration"],
                availability=evidence["solver_availability"],
                device_id=args.device_id,
                checkpoint=checkpoint,
            )
        except Exception as error:
            failure = stage7._exception_record(f"case:{next_key}", error)
            case.update({"status": "FAIL", "passed": False, "failure": failure})
            evidence["failures"].append(failure)
        if case.get("status") != "PASS" and case.get("failure") is None:
            failure = {
                "phase": f"case:{next_key}",
                "type": "RecordedSolverFailure",
                "message": f"terminal case status {case.get('status')}",
                "recorded_utc": stage7._utc_now(),
            }
            case["failure"] = failure
            evidence["failures"].append(failure)
        _update_campaign_status(evidence)
        invocation["completed_utc"] = stage7._utc_now()
        invocation["outcome"] = evidence["status"]
        evidence["completed_utc"] = stage7._utc_now()
        checkpoint()
        stage7._atomic_write_json(final_path, evidence)
        return 0 if case.get("status") == "PASS" else 1
    except Exception as error:
        failure = stage7._exception_record("stage8_runner", error)
        evidence.setdefault("failures", []).append(failure)
        evidence["status"] = "FAIL"
        evidence["all_passed"] = False
        invocation["completed_utc"] = stage7._utc_now()
        invocation["outcome"] = "FAIL"
        evidence["completed_utc"] = stage7._utc_now()
        checkpoint()
        stage7._atomic_write_json(final_path, evidence)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with _exclusive_output_lock(args.output_dir.resolve()):
            return _run_main(args)
    except Stage8ConcurrentRunError as error:
        print(f"Stage 8 concurrent start refused: {error}", file=sys.stderr)
        return 1
    except Stage8ResumeError as error:
        print(f"Stage 8 resume refused: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        output_dir = args.output_dir.resolve()
        partial_path = output_dir / PARTIAL_NAME
        final_path = output_dir / FINAL_NAME
        if partial_path.exists() or final_path.exists():
            print(f"Stage 8 start refused: {error}", file=sys.stderr)
            return 1
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = stage7._exception_record("stage8_initialization", error)
        evidence = {
            "schema_version": "1.0",
            "stage": 8,
            "status": "FAIL",
            "all_passed": False,
            "started_utc": stage7._utc_now(),
            "completed_utc": stage7._utc_now(),
            "initialization_completed": False,
            "failures": [failure],
            "stage_boundary": {
                "stage_8_only": True,
                "stage_8_complete": False,
                "stage_9_locked": True,
                "stage_9_allocation_count": 0,
                "next_authorized_key": None,
            },
        }
        stage7._atomic_write_json(partial_path, evidence)
        stage7._atomic_write_json(final_path, evidence)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
