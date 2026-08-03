"""Run the frozen Stage 7 small/medium structural benchmark campaign.

This runner is deliberately evidence-first.  It verifies pinned inputs, emits a
no-allocation ledger for every Table II row, refuses Stage 8 allocations, and
checkpoints partial JSON atomically after every material event.  A solver that
is absent, unlicensed, times out, or lacks the scalable Stage 7 adapter is
reported as such; no timing or success value is synthesized.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy
from scipy import optimize, sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gpu_dcopf_hpr.gpu_backend import (  # noqa: E402
    CuPyBackend,
    GPUBackendUnavailable,
    create_gpu_backend,
)
from gpu_dcopf_hpr.gpu_stage5_control import (  # noqa: E402
    GPUStage6Result,
    prepare_gpu_stage6_problem,
    solve_gpu_stage5_sgs_hpr,
)
from gpu_dcopf_hpr.hpr_generic import HPRState  # noqa: E402
from gpu_dcopf_hpr.network_data import load_matpower_case  # noqa: E402
from gpu_dcopf_hpr.preconditioning import (  # noqa: E402
    LPPreconditioner,
    precondition_lp,
)
from gpu_dcopf_hpr.residuals import evaluate_residuals  # noqa: E402
from gpu_dcopf_hpr.sgs_hpr import SGSHPRWorkspace, prepare_sgs_hpr  # noqa: E402
from gpu_dcopf_hpr.stage5_control import (  # noqa: E402
    Stage5Control,
    Stage5SGSHPRResult,
    solve_stage5_sgs_hpr,
)
from gpu_dcopf_hpr.stage7_scalable_model import (  # noqa: E402
    FROZEN_STAGE7_POLICY,
    Stage7Preflight,
    Stage7ScalableModel,
    all_stage7_preflights,
    all_stage7_symbolic_ledgers,
    assert_stage7_reconstruction_contract,
    build_stage7_scalable_model,
    stage7_reconstructed_nnz_ledger,
)
from gpu_dcopf_hpr.stage7_scaled_y1 import (  # noqa: E402
    ScaledBlockArrowY1Solver,
    prepare_scaled_block_arrow_y1,
)
from gpu_dcopf_hpr.stage7_spectral import (  # noqa: E402
    SparseSpectralCertificate,
    estimate_sparse_spectral_norm_squared,
)
from gpu_dcopf_hpr.structural_y1 import DCOPFEqualityStructure  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "benchmarks" / "stage_7_small_medium.json"
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "environment" / "dgx_stage7_requirements.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_7"
PARTIAL_NAME = "stage_7_validation.partial.json"
FINAL_NAME = "stage_7_validation.json"
CANONICAL_GIT_BLOB_SHA256_DEFINITION = "SHA-256 of canonical Git blob bytes with LF text content"
FROZEN_CONFIG_SHA256 = "06a172463049c519ab14c446d8b9ab632cd91c8afa4b44264e284b3a4f59a062"
FROZEN_REQUIREMENTS_SHA256 = "827065b5bfc2920492cfe653e922cd2d3b2b4289ade12b06d866bea83d32dacf"

EXECUTABLE_CASE_COUNT = 6
TABLE_ROW_COUNT = 18
EXPECTED_EXECUTABLE_KEYS = frozenset(
    {
        ("case1354pegase", 4),
        ("case1354pegase", 16),
        ("case1354pegase", 48),
        ("case1354pegase", 96),
        ("case2868rte", 4),
        ("case2868rte", 16),
    }
)
HOST_SAFETY_FRACTION = 0.80
DEVICE_SAFETY_FRACTION = 0.80
TECHNICAL_ITERATION_CEILING = 2_147_483_647
EXPECTED_REQUIREMENT_PINS = {
    "cupy-cuda13x": "14.1.1",
    "numpy": "2.3.5",
    "scipy": "1.16.3",
}
TERMINAL_CASE_STATUSES = frozenset({"PASS", "FAIL", "TIME_LIMIT", "MEMORY_BLOCKED"})


class Stage7ContractError(ValueError):
    """The frozen Stage 7 contract or its implementation drifted."""


class SolveTimeLimit(TimeoutError):
    """One solver call exceeded the preregistered wall-clock limit."""


@dataclass(frozen=True, slots=True)
class CaseKey:
    case_name: str
    periods: int

    @property
    def text(self) -> str:
        return f"{self.case_name}:T{self.periods}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--case",
        dest="case_keys",
        action="append",
        default=[],
        metavar="NAME:T",
        help="Run only a preregistered Stage 7 case/horizon; may be repeated.",
    )
    parser.add_argument(
        "--ledger-only",
        action="store_true",
        help="Verify provenance and write the 18-row no-allocation ledger only.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume compatible partial evidence (default: enabled).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry terminal failed/timed-out cases instead of preserving them.",
    )
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _clean_json(value: Any) -> Any:
    """Convert evidence to strict JSON while keeping nonfinite results explicit."""

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
    if is_dataclass(value) and not isinstance(value, type):
        return _clean_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_clean_json(item) for item in value]
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing a truncated checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            _clean_json(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(*arguments: str) -> tuple[bytes | None, str | None]:
    """Run one read-only Git command and retain an auditable failure reason."""

    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"{type(error).__name__}: {error}"
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        return None, message or f"git {' '.join(arguments)} exited {result.returncode}"
    return result.stdout, None


def _canonical_git_blob_identity(
    relative: Path,
    path: Path,
    *,
    expected_blob: str | None = None,
) -> dict[str, Any]:
    """Hash canonical Git blob bytes while proving the worktree is equivalent.

    Git's path-aware ``hash-object`` applies the repository's text conversion,
    so an LF checkout and an equivalent CRLF checkout resolve to the same blob.
    The committed blob, filtered worktree blob, and configured upstream blob
    must all agree before the canonical SHA-256 is accepted.
    """

    relative_text = relative.as_posix()
    errors: list[str] = []
    head_bytes, head_error = _git_bytes("rev-parse", f"HEAD:{relative_text}")
    head_blob = None if head_bytes is None else head_bytes.decode().strip()
    canonical_blob = expected_blob or head_blob
    worktree_bytes, worktree_error = _git_bytes(
        "hash-object",
        f"--path={relative_text}",
        str(path),
    )
    status_bytes, status_error = _git_bytes(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        relative_text,
    )
    diff_bytes, diff_error = _git_bytes(
        "diff",
        "--name-only",
        "--",
        relative_text,
    )
    if canonical_blob is None:
        blob_bytes, blob_error = None, "the canonical Git blob is unavailable"
    else:
        blob_bytes, blob_error = _git_bytes("cat-file", "blob", canonical_blob)
    for label, error in (
        ("HEAD blob", head_error),
        ("filtered worktree blob", worktree_error),
        ("worktree status", status_error),
        ("worktree diff", diff_error),
        ("canonical blob read", blob_error),
    ):
        if error is not None:
            errors.append(f"{label}: {error}")

    worktree_blob = None if worktree_bytes is None else worktree_bytes.decode().strip()
    worktree_status = (
        None if status_bytes is None else status_bytes.decode("utf-8", errors="replace").strip()
    )
    worktree_diff = (
        None if diff_bytes is None else diff_bytes.decode("utf-8", errors="replace").strip()
    )
    canonical_sha256 = None if blob_bytes is None else hashlib.sha256(blob_bytes).hexdigest()
    canonical_lf_text = blob_bytes is not None and b"\r" not in blob_bytes
    checks = {
        "head_blob_matches": head_blob == canonical_blob,
        "filtered_worktree_blob_matches": worktree_blob == canonical_blob,
        "worktree_clean": worktree_status == "" and worktree_diff == "",
        "canonical_blob_read": blob_bytes is not None,
        "canonical_blob_uses_lf_text": canonical_lf_text,
    }
    errors.extend(f"{name} check failed" for name, passed in checks.items() if not passed)
    return {
        "sha256_definition": CANONICAL_GIT_BLOB_SHA256_DEFINITION,
        "expected_git_blob": canonical_blob,
        "head_git_blob": head_blob,
        "filtered_worktree_git_blob": worktree_blob,
        "worktree_status": worktree_status,
        "worktree_diff": worktree_diff,
        "worktree_raw_sha256": _sha256(path) if path.is_file() else None,
        "canonical_git_blob_sha256": canonical_sha256,
        "canonical_git_blob_size_bytes": None if blob_bytes is None else len(blob_bytes),
        "checks": checks,
        "errors": errors,
        "passed": not errors and all(checks.values()),
    }


def _hash_array(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _hash_sparse(matrix: sparse.spmatrix) -> str:
    value = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    value.sum_duplicates()
    value.eliminate_zeros()
    value.sort_indices()
    digest = hashlib.sha256()
    digest.update(json.dumps(list(value.shape)).encode())
    for block in (value.indptr, value.indices, value.data):
        digest.update(_hash_array(block).encode())
    return digest.hexdigest()


def _lp_fingerprint(model: Stage7ScalableModel) -> dict[str, Any]:
    lp = model.lp
    blocks = {
        "A1": _hash_sparse(lp.A1),
        "A2": _hash_sparse(lp.A2),
        "b1": _hash_array(lp.b1),
        "b2": _hash_array(lp.b2),
        "c": _hash_array(lp.c),
        "lower": _hash_array(lp.lower),
        "upper": _hash_array(lp.upper),
    }
    canonical = json.dumps(blocks, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(canonical.encode()).hexdigest(), "blocks": blocks}


def _state_fingerprint(state: HPRState) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name in ("x", "y", "z"):
        values = np.asarray(getattr(state, name), dtype=np.float64)
        rows[name] = {
            "shape": list(values.shape),
            "sha256": _hash_array(values),
            "finite": bool(np.all(np.isfinite(values))),
            "minimum": float(np.min(values, initial=0.0)),
            "maximum": float(np.max(values, initial=0.0)),
            "l2_norm": float(np.linalg.norm(values)),
        }
    return rows


def _exception_record(phase: str, error: BaseException) -> dict[str, Any]:
    return {
        "phase": phase,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "utc": _utc_now(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--short"),
    }


def _host_memory() -> dict[str, Any]:
    try:
        psutil = importlib.import_module("psutil")
        memory = psutil.virtual_memory()
        return {
            "available_bytes": int(memory.available),
            "total_bytes": int(memory.total),
            "source": "psutil.virtual_memory",
        }
    except (ImportError, AttributeError, OSError):
        pass
    if hasattr(os, "sysconf"):
        try:
            page = int(os.sysconf("SC_PAGE_SIZE"))
            available = int(os.sysconf("SC_AVPHYS_PAGES")) * page
            total = int(os.sysconf("SC_PHYS_PAGES")) * page
            return {
                "available_bytes": available,
                "total_bytes": total,
                "source": "os.sysconf",
            }
        except (OSError, ValueError, TypeError):
            pass
    return {"available_bytes": None, "total_bytes": None, "source": "unavailable"}


def _process_memory() -> dict[str, Any]:
    """Sample process RSS and disclose the cumulative OS high-water mark."""

    result: dict[str, Any] = {
        "rss_bytes": None,
        "cumulative_process_peak_bytes": None,
        "peak_scope": "cumulative process lifetime; not isolated to this solve",
        "sources": [],
    }
    try:
        psutil = importlib.import_module("psutil")
        process = psutil.Process()
        information = process.memory_info()
        result["rss_bytes"] = int(information.rss)
        result["sources"].append("psutil.Process.memory_info.rss")
        peak_wset = getattr(information, "peak_wset", None)
        if peak_wset is not None:
            result["cumulative_process_peak_bytes"] = int(peak_wset)
            result["sources"].append("psutil.Process.memory_info.peak_wset")
    except (ImportError, AttributeError, OSError):
        pass
    try:
        resource = importlib.import_module("resource")
        usage = resource.getrusage(resource.RUSAGE_SELF)
        raw_peak = int(usage.ru_maxrss)
        # Linux reports KiB; macOS reports bytes.  The DGX execution is Linux.
        peak_bytes = raw_peak if sys.platform == "darwin" else raw_peak * 1024
        result["cumulative_process_peak_bytes"] = peak_bytes
        result["sources"].append("resource.getrusage(RUSAGE_SELF).ru_maxrss")
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    return result


def _environment() -> dict[str, Any]:
    return {
        "captured_utc": _utc_now(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "packages": {
            name: _package_version(name) for name in ("cupy-cuda13x", "gurobipy", "numpy", "scipy")
        },
        "host_memory": _host_memory(),
        "git": _git_metadata(),
    }


def _configured_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in config.get("cases", []):
        for row in case.get("rows", []):
            rows.append(
                {
                    "case_name": case.get("case"),
                    "buses": case.get("buses"),
                    "branches": case.get("branches"),
                    "generators": case.get("generators"),
                    "renewables": case.get("renewables"),
                    "storage": case.get("storage"),
                    **row,
                }
            )
    return rows


def _requirements_freeze(path: Path = DEFAULT_REQUIREMENTS) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "sha256": None,
            "pins": {},
            "expected_pins": EXPECTED_REQUIREMENT_PINS,
            "errors": ["Stage 7 DGX requirements file is missing"],
            "passed": False,
        }
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    identity = _canonical_git_blob_identity(relative, path.resolve())
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip()] = version.strip()
    errors = [
        f"requirement {name} must remain pinned to {expected}"
        for name, expected in EXPECTED_REQUIREMENT_PINS.items()
        if pins.get(name) != expected
    ]
    if identity["canonical_git_blob_sha256"] != FROZEN_REQUIREMENTS_SHA256:
        errors.append(
            f"Stage 7 DGX requirements canonical SHA-256 drifted from {FROZEN_REQUIREMENTS_SHA256}"
        )
    errors.extend(str(error) for error in identity["errors"])
    return {
        "path": relative.as_posix(),
        "sha256": identity["canonical_git_blob_sha256"],
        "sha256_definition": CANONICAL_GIT_BLOB_SHA256_DEFINITION,
        "portable_identity": identity,
        "pins": pins,
        "expected_pins": EXPECTED_REQUIREMENT_PINS,
        "errors": errors,
        "passed": not errors,
    }


def _validate_stage7_config(config: Mapping[str, Any]) -> list[str]:
    """Reject changes to every execution-critical Stage 7 gate."""

    errors: list[str] = []
    if config.get("schema_version") != "1.0" or config.get("stage") != 7:
        errors.append("Stage 7 requires schema_version 1.0 and stage 7")
    if config.get("classification") != "structural_reproduction":
        errors.append("Stage 7 must remain a structural reproduction")
    if config.get("precision") != "FP64":
        errors.append("Stage 7 execution precision must remain FP64")
    source = config.get("public_network_source", {})
    if source.get("sha256_definition") != CANONICAL_GIT_BLOB_SHA256_DEFINITION:
        errors.append("public_network_source.sha256_definition drifted from the portable contract")

    reconstruction = config.get("reconstruction_protocol", {})
    expected_reconstruction = {
        "frozen_before_benchmark_runs": True,
        "timing_tuning_prohibited": True,
        "load_profile": "flat MATPOWER active demand in every period",
        "load_definition": "MATPOWER Pd + Gs treated as active withdrawal",
        "generator_policy": (
            "retain every MATPOWER generator row; originally offline rows are fixed at zero "
            "output and zero reserve"
        ),
        "generator_ramp_fraction_of_pmax_per_hour": 0.1,
        "renewable_placement": (
            "cycle through MATPOWER generator buses in ascending generator-row order"
        ),
        "renewable_total_nameplate_fraction_of_base_load": 0.1,
        "renewable_nameplate_distribution": "equal across reconstructed renewable rows",
        "renewable_minimum_fraction": 0.0,
        "renewable_availability_fraction": 1.0,
        "storage_placement": (
            "use every second reconstructed renewable location, cycling deterministically "
            "when required"
        ),
        "storage_total_power_fraction_of_base_load": 0.01,
        "storage_power_distribution": "equal across reconstructed storage rows",
        "storage_duration_hours": 4.0,
        "storage_minimum_energy_fraction": 0.0,
        "storage_initial_state_fraction": 0.5,
        "storage_charge_efficiency": 0.95,
        "storage_discharge_efficiency": 0.95,
        "reserve_up_fraction_of_base_load": 0.01,
        "reserve_down_fraction_of_base_load": 0.01,
        "renewable_penalty_per_mwh": 1.0,
        "storage_loss_penalty_per_mwh": 1.0,
        "positive_rate_a_policy": "preserve the MATPOWER rateA value",
        "zero_rate_a_policy": (
            "retain the paper-required branch rows with a finite bound derived to be "
            "redundant over the variable box"
        ),
        "zero_rate_a_bound_proof": (
            "outward-rounded triangle bound over the complete variable box"
        ),
        "inactive_branch_policy": (
            "retain paper-count rows as zero-flow rows while excluding them from topology"
        ),
        "angle_limit_policy": (
            "ignore MATPOWER angle-difference limits because they are absent from the paper model"
        ),
        "objective_policy": (
            "use the linear and constant MATPOWER polynomial terms; reject nonzero "
            "higher-order terms"
        ),
        "interval_hours": 1.0,
        "ptdf_zero_atol": 1e-12,
        "ptdf_rhs_chunk_columns": 128,
        "ptdf_reference_policy": "use the public MATPOWER type-3 reference bus",
        "seed": 20260803,
    }
    for key, expected in expected_reconstruction.items():
        if reconstruction.get(key) != expected:
            errors.append(f"reconstruction_protocol.{key} drifted from the frozen contract")

    algorithm = config.get("algorithm", {})
    expected_algorithm = {
        "ruiz_iterations": 10,
        "pock_chambolle": True,
        "pock_chambolle_alpha": 1.0,
        "normalize_b_and_c": True,
        "initial_sigma": 1.0,
        "adaptive_sigma": True,
        "restart": True,
        "policy_check_interval": 100,
        "correctness_residual_check_interval": 1,
        "requested_spmv_algorithm": "CUSPARSE_SPMV_CSR_ALG2",
        "mixed_precision_enabled": False,
    }
    for key, expected in expected_algorithm.items():
        if algorithm.get(key) != expected:
            errors.append(f"algorithm.{key} drifted from the frozen contract")

    acceptance = config.get("acceptance", {})
    expected_acceptance = {
        "paper_residual_tolerance": 5e-5,
        "raw_kkt_tolerance": 1e-2,
        "maximum_physical_violation": 1e-2,
        "maximum_scaled_objective_gap_to_highs": 2e-4,
        "maximum_dimension_difference_for_execution": 0,
        "nonzero_nnz_difference_blocks_paper_time_comparability": True,
        "required_solver_tracks": [
            "highs",
            "cpu_fp64_sgs_hpr",
            "gpu_fp64_sgs_hpr",
        ],
        "gurobi_required_only_when_installed_and_licensed": True,
    }
    for key in sorted(set(acceptance) | set(expected_acceptance)):
        if acceptance.get(key) != expected_acceptance.get(key):
            errors.append(f"acceptance.{key} drifted from the frozen contract")

    timing = config.get("timing", {})
    expected_timing = {
        "warmup_runs": 1,
        "measured_runs": 5,
        "maximum_measured_runs_after_variability_escalation": 9,
        "relative_range_escalation_threshold": 0.2,
        "per_solve_time_limit_seconds": 3600,
        "report_first_run_separately": True,
        "speedup_requires_matching_boundaries": True,
    }
    for key, expected in expected_timing.items():
        if timing.get(key) != expected:
            errors.append(f"timing.{key} drifted from the frozen contract")

    boundary = config.get("stage_boundary", {})
    expected_boundary = {
        "stage_7_only": True,
        "stage_8_large_runs_locked": True,
        "exact_paper_reproduction_claimed": False,
        "paper_a100_timing_reproduction_claimed": False,
        "n_minus_1_extension_enabled": False,
    }
    for key, expected in expected_boundary.items():
        if boundary.get(key) != expected:
            errors.append(f"stage_boundary.{key} is invalid")

    rows = _configured_rows(config)
    keys = [(row.get("case_name"), row.get("periods")) for row in rows]
    if len(rows) != TABLE_ROW_COUNT or len(set(keys)) != TABLE_ROW_COUNT:
        errors.append("Stage 7 must contain 18 unique Table II rows")
    if sum(bool(row.get("execute_stage_7")) for row in rows) != EXECUTABLE_CASE_COUNT:
        errors.append("Stage 7 must execute exactly six preregistered rows")
    executable_keys = {
        (str(row.get("case_name")), int(row.get("periods")))
        for row in rows
        if row.get("execute_stage_7") is True
    }
    if executable_keys != EXPECTED_EXECUTABLE_KEYS:
        errors.append(
            "Stage 7 executable rows must remain exactly case1354pegase "
            "T4/T16/T48/T96 and case2868rte T4/T16"
        )
    for row in rows:
        if row.get("execute_stage_7") is not True and row.get("case_name") == "case1354pegase":
            errors.append("all four case1354pegase rows belong to Stage 7")
        if row.get("execute_stage_7") is True and row.get("case_name") == "case9241pegase":
            errors.append("case9241pegase allocations remain locked to Stage 8")
    errors.extend(_requirements_freeze()["errors"])
    return errors


def _policy_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the implementation policy to the JSON, which is sole authority."""

    protocol = dict(config["reconstruction_protocol"])
    error: str | None = None
    try:
        assert_stage7_reconstruction_contract(protocol)
    except ValueError as caught:
        error = str(caught)
    return {
        "json_is_sole_authority": True,
        "policy_fingerprint": FROZEN_STAGE7_POLICY.fingerprint,
        "error": error,
        "passed": error is None,
    }


def _verify_provenance(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    config_relative = config_path.resolve().relative_to(PROJECT_ROOT.resolve())
    config_identity = _canonical_git_blob_identity(config_relative, config_path.resolve())
    config_record = {
        "path": str(config_path),
        "sha256": config_identity["canonical_git_blob_sha256"],
        "sha256_definition": CANONICAL_GIT_BLOB_SHA256_DEFINITION,
        "portable_identity": config_identity,
        "sha256_matches_frozen": (
            config_identity["canonical_git_blob_sha256"] == FROZEN_CONFIG_SHA256
        ),
        "passed": bool(
            config_identity["passed"]
            and config_identity["canonical_git_blob_sha256"] == FROZEN_CONFIG_SHA256
        ),
    }
    if not config_record["passed"]:
        details = list(config_identity["errors"])
        if not config_record["sha256_matches_frozen"]:
            details.append(
                "canonical SHA-256 does not match the frozen Stage 7 configuration "
                f"{FROZEN_CONFIG_SHA256}"
            )
        errors.append("frozen config is not the clean canonical Git blob: " + "; ".join(details))
    for item in config.get("public_network_source", {}).get("files", []):
        relative = Path(str(item.get("path", "")))
        path = (PROJECT_ROOT / relative).resolve()
        inside = path.is_relative_to(PROJECT_ROOT.resolve())
        exists = inside and path.is_file()
        expected_sha = item.get("sha256")
        expected_blob = str(item.get("git_blob", ""))
        identity = (
            _canonical_git_blob_identity(relative, path, expected_blob=expected_blob)
            if exists
            else {
                "canonical_git_blob_sha256": None,
                "filtered_worktree_git_blob": None,
                "checks": {},
                "errors": ["input path is absent or outside the project"],
                "passed": False,
            }
        )
        actual_sha = identity["canonical_git_blob_sha256"]
        sha_matches = actual_sha == expected_sha
        blob = identity["filtered_worktree_git_blob"]
        blob_matches = blob == expected_blob
        record = {
            "case_name": item.get("case"),
            "path": relative.as_posix(),
            "inside_project": inside,
            "exists": exists,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "sha256_matches": sha_matches,
            "expected_git_blob": item.get("git_blob"),
            "actual_git_blob": blob,
            "git_blob_matches": blob_matches,
            "portable_identity": identity,
            "passed": bool(
                inside and exists and identity["passed"] and sha_matches and blob_matches
            ),
        }
        if not record["passed"]:
            detail = "; ".join(identity.get("errors", []))
            errors.append(
                f"provenance mismatch for {relative.as_posix()}" + (f": {detail}" if detail else "")
            )
        records.append(record)
    return {
        "config": config_record,
        "upstream": dict(config.get("public_network_source", {})),
        "files": records,
        "errors": errors,
        "passed": not errors and config_record["passed"] and len(records) == 3,
    }


def _symbolic_ledger(
    config: Mapping[str, Any],
    *,
    networks: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    configured = {
        (str(row["case_name"]), int(row["periods"])): row for row in _configured_rows(config)
    }
    preflights = all_stage7_preflights()
    if len(preflights) != TABLE_ROW_COUNT:
        raise Stage7ContractError("the symbolic implementation did not return all 18 rows")
    exact = (
        {}
        if networks is None
        else {(row.case_name, row.periods): row for row in all_stage7_symbolic_ledgers(networks)}
    )
    if networks is not None and len(exact) != TABLE_ROW_COUNT:
        raise Stage7ContractError("the exact symbolic nnz ledger did not return all 18 rows")
    ledger: list[dict[str, Any]] = []
    for preflight in preflights:
        key = (preflight.row.case_name, preflight.row.periods)
        if key not in configured:
            raise Stage7ContractError(f"symbolic row {key} is absent from the JSON contract")
        row = configured[key]
        paper_matches_config = (
            int(row["m"]) == preflight.row.published_m
            and int(row["n"]) == preflight.row.published_n
            and int(row["nnz"]) == preflight.row.published_nnz
        )
        execute = bool(row["execute_stage_7"])
        exact_row = exact.get(key)
        reconstructed_nnz = None if exact_row is None else exact_row.reconstructed_nnz
        nnz_difference = None if exact_row is None else exact_row.difference_from_paper
        m_difference = abs(preflight.computed_m - preflight.row.published_m)
        n_difference = abs(preflight.computed_n - preflight.row.published_n)
        nnz_absolute_difference = None if nnz_difference is None else abs(nnz_difference)
        ledger.append(
            {
                **preflight.as_dict(),
                "key": CaseKey(*key).text,
                "execute_stage_7": execute,
                "allocation_stage": "stage_7" if execute else "stage_8_locked",
                "allocation_permitted_this_run": execute,
                "paper_values_match_config": paper_matches_config,
                "symbolic_nnz": None if exact_row is None else exact_row.as_dict(),
                "dimension_comparison": {
                    "m": {
                        "paper": preflight.row.published_m,
                        "reproduced": preflight.computed_m,
                        "absolute_difference": m_difference,
                        "percentage_difference": (100.0 * m_difference / preflight.row.published_m),
                        "cause": (
                            "paper row-count formulas evaluated with the published network, "
                            "renewable, and storage counts"
                        ),
                    },
                    "n": {
                        "paper": preflight.row.published_n,
                        "reproduced": preflight.computed_n,
                        "absolute_difference": n_difference,
                        "percentage_difference": (100.0 * n_difference / preflight.row.published_n),
                        "cause": (
                            "paper variable-count formula evaluated with the published network, "
                            "renewable, and storage counts"
                        ),
                    },
                },
                "nnz_comparison": {
                    "paper": preflight.row.published_nnz,
                    "reproduced": reconstructed_nnz,
                    "signed_difference": nnz_difference,
                    "absolute_difference": nnz_absolute_difference,
                    "percentage_difference": (
                        None
                        if nnz_absolute_difference is None
                        else 100.0 * nnz_absolute_difference / preflight.row.published_nnz
                    ),
                    "cause": (
                        "author renewable/storage placement and PTDF construction are unavailable; "
                        "the frozen deterministic selected-bus PTDF support uses a 1e-12 zero "
                        "threshold and was not tuned to timing or Table II nnz"
                    ),
                },
                "actual_reconstruction_nnz": reconstructed_nnz,
                "nnz_difference_from_paper": nnz_difference,
                "paper_time_comparable": nnz_difference == 0,
                "paper_time_comparability_reason": (
                    "exact symbolic reconstruction nnz not counted"
                    if nnz_difference is None
                    else (
                        "exact reconstructed nnz match"
                        if nnz_difference == 0
                        else "reconstructed nnz differs; Table II timing is context only"
                    )
                ),
                "passed": bool(preflight.dimensions_match_table and paper_matches_config),
            }
        )
    if {row["key"] for row in ledger} != {CaseKey(*key).text for key in configured}:
        raise Stage7ContractError("the JSON and symbolic Table II row sets differ")
    return ledger


def _timing_statistics(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values:
        return {
            "count": 0,
            "raw_seconds": [],
            "median_seconds": None,
            "minimum_seconds": None,
            "maximum_seconds": None,
            "standard_deviation_seconds": None,
            "interquartile_range_seconds": None,
            "relative_range": None,
        }
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("timing samples must be finite and nonnegative")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    q1, q3 = np.percentile(ordered, [25.0, 75.0], method="linear")
    median = float(np.median(ordered))
    minimum = float(ordered[0])
    maximum = float(ordered[-1])
    relative_range = (maximum - minimum) / max(median, np.finfo(np.float64).tiny)
    return {
        "count": len(values),
        "raw_seconds": values,
        "median_seconds": median,
        "minimum_seconds": minimum,
        "maximum_seconds": maximum,
        "standard_deviation_seconds": (float(statistics.stdev(values)) if len(values) > 1 else 0.0),
        "interquartile_range_seconds": float(q3 - q1),
        "relative_range": float(relative_range),
    }


@contextlib.contextmanager
def _solve_deadline(seconds: int) -> Iterator[dict[str, Any]]:
    """Apply a hard POSIX alarm where available and disclose weaker platforms."""

    if seconds <= 0:
        raise ValueError("solve deadline must be positive")
    active = bool(os.name == "posix" and hasattr(signal, "SIGALRM"))
    evidence = {
        "seconds": seconds,
        "mechanism": "signal.setitimer" if active else "solver-native/cooperative",
        "hard_alarm_active": active,
    }
    previous_handler: Any = None
    if active:
        previous_handler = signal.getsignal(signal.SIGALRM)

        def expired(_signum: int, _frame: Any) -> None:
            raise SolveTimeLimit(f"solve exceeded {seconds} seconds")

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield evidence
    finally:
        if active:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)


def _memory_guard(
    preflight: Stage7Preflight,
    *,
    host_memory: Mapping[str, Any],
    device_total_bytes: int | None,
) -> dict[str, Any]:
    host_available = host_memory.get("available_bytes")
    host_budget = (
        None if host_available is None else int(HOST_SAFETY_FRACTION * int(host_available))
    )
    device_budget = (
        None
        if device_total_bytes is None
        else int(DEVICE_SAFETY_FRACTION * int(device_total_bytes))
    )
    checks = {
        "dimensions_match_table": preflight.dimensions_match_table,
        "csr32_supported": preflight.csr32_supported,
        "preflight_fits_nominal_dgx": preflight.fits_dgx_planning_budget,
        "host_available": (
            True if host_budget is None else preflight.host_assembly_peak_bytes <= host_budget
        ),
        "device_available": (
            True if device_budget is None else preflight.gpu_planning_bytes <= device_budget
        ),
    }
    return {
        "preflight": preflight.as_dict(),
        "host_available_bytes": host_available,
        "host_safety_fraction": HOST_SAFETY_FRACTION,
        "host_budget_bytes": host_budget,
        "device_total_bytes": device_total_bytes,
        "device_safety_fraction": DEVICE_SAFETY_FRACTION,
        "device_budget_bytes": device_budget,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _transfer_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Subtract cumulative transfer ledgers into one per-attempt record."""

    def indexed(summary: Mapping[str, Any]) -> dict[tuple[str, str, str], tuple[int, int]]:
        return {
            (str(row["phase"]), str(row["direction"]), str(row["kind"])): (
                int(row["calls"]),
                int(row["bytes"]),
            )
            for row in summary.get("records", [])
        }

    first = indexed(before)
    second = indexed(after)
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


def _transfer_timing_delta(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, float]:
    result = {key: float(after[key]) - float(before.get(key, 0.0)) for key in after}
    if any(value < 0.0 for value in result.values()):
        raise ValueError("transfer timing summaries must be cumulative and monotone")
    return result


def _audit_gpu_solver_transfers(delta: Mapping[str, Any]) -> dict[str, Any]:
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
        "allowed_phase_directions": [list(value) for value in sorted(allowed)],
        "unexpected_records": unexpected,
        "full_state_copied_inside_resident_loop": any(
            row.get("direction") == "device_to_host"
            and row.get("phase")
            not in {
                "periodic_diagnostics",
                "policy_diagnostics",
                "final_state",
                "final_scaled_state",
                "final_diagnostics",
            }
            for row in delta.get("records", [])
        ),
        "passed": not unexpected,
    }


def _probe_gurobi() -> dict[str, Any]:
    try:
        gp = importlib.import_module("gurobipy")
    except Exception as error:
        return {
            "installed": False,
            "licensed": False,
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
        }
    environment = None
    try:
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 0)
        environment.start()
        version = ".".join(str(part) for part in gp.gurobi.version())
        return {
            "installed": True,
            "licensed": True,
            "available": True,
            "version": version,
            "reason": None,
        }
    except Exception as error:
        return {
            "installed": True,
            "licensed": False,
            "available": False,
            "version": _package_version("gurobipy"),
            "reason": f"{type(error).__name__}: {error}",
        }
    finally:
        if environment is not None:
            with contextlib.suppress(Exception):
                environment.dispose()


def _solver_availability() -> dict[str, Any]:
    cpu_signature = inspect.signature(solve_stage5_sgs_hpr)
    gpu_prepare_signature = inspect.signature(prepare_gpu_stage6_problem)
    cpu_adapter = any(
        name in cpu_signature.parameters
        for name in ("scaled_structural_y1", "scaled_equality_solver")
    )
    gpu_adapter = any(
        name in gpu_prepare_signature.parameters
        for name in ("scaled_structural_y1", "scaled_equality_solver")
    )
    cupy_version = _package_version("cupy-cuda13x")
    return {
        "highs": {
            "installed": True,
            "available": True,
            "provider": "scipy.optimize.linprog(method='highs-ds')",
            "scipy_version": scipy.__version__,
        },
        "cpu_fp64_sgs_hpr": {
            "installed": True,
            "available": cpu_adapter,
            "reason": (
                None
                if cpu_adapter
                else "scaled block-arrow adapter is absent from solve_stage5_sgs_hpr"
            ),
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
        "gurobi": _probe_gurobi(),
    }


def _scaled_gap(candidate: float, reference: float) -> float:
    return abs(float(candidate) - float(reference)) / max(1.0, abs(float(reference)))


def _canonical_violations(model: Stage7ScalableModel, x: Any) -> dict[str, float]:
    vector = np.asarray(x, dtype=np.float64)
    lp = model.lp
    equality = (
        float(np.max(np.abs(np.asarray(lp.A1 @ vector).reshape(-1) - lp.b1), initial=0.0))
        if lp.m1
        else 0.0
    )
    inequality = (
        float(np.max(np.maximum(lp.b2 - np.asarray(lp.A2 @ vector).reshape(-1), 0.0), initial=0.0))
        if lp.m2
        else 0.0
    )
    lower = float(np.max(np.maximum(lp.lower - vector, 0.0), initial=0.0))
    upper = float(np.max(np.maximum(vector - lp.upper, 0.0), initial=0.0))
    box = max(lower, upper)
    return {
        "equality_inf": equality,
        "inequality_positive_max": inequality,
        "lower_violation_max": lower,
        "upper_violation_max": upper,
        "box_violation_max": box,
        "overall_max": max(equality, inequality, box),
    }


def _physical_validation(
    model: Stage7ScalableModel,
    x: Any,
    *,
    tolerance: float,
) -> dict[str, Any]:
    module = importlib.import_module("gpu_dcopf_hpr.stage7_scalable_model")
    validator = getattr(module, "validate_stage7_physical", None)
    if validator is None:
        return {
            "available": False,
            "passed": False,
            "maximum_violation": None,
            "reason": "validate_stage7_physical adapter is unavailable",
        }
    result = validator(model, np.asarray(x, dtype=np.float64))
    if hasattr(result, "summary"):
        summary = result.summary()
    elif hasattr(result, "as_dict"):
        summary = result.as_dict()
    else:
        summary = _clean_json(result)
    if not isinstance(summary, Mapping):
        raise TypeError("Stage 7 physical validator must return mapping-like evidence")
    maximum = summary.get("maximum_violation")
    if maximum is None and "families" in summary:
        maximum = max(
            (float(row.get("maximum_violation", 0.0)) for row in summary["families"]),
            default=0.0,
        )
    crosscheck = float(summary.get("angle_vs_compressed_ptdf_flow_max_abs_mw", 0.0))
    combined_maximum = max(float(maximum), crosscheck)
    return {
        **dict(summary),
        "available": True,
        "maximum_violation": combined_maximum,
        "tolerance": tolerance,
        "passed": bool(np.isfinite(combined_maximum) and combined_maximum <= tolerance),
    }


def _candidate_summary(
    model: Stage7ScalableModel,
    state: HPRState,
    *,
    config: Mapping[str, Any],
    reference_objective: float | None,
) -> dict[str, Any]:
    acceptance = config["acceptance"]
    tolerance = float(acceptance["paper_residual_tolerance"])
    residuals = evaluate_residuals(
        model.lp,
        x=state.x,
        y=state.y,
        z=state.z,
        tolerance=tolerance,
    )
    objective = model.objective(state.x)
    canonical = _canonical_violations(model, state.x)
    physical = _physical_validation(
        model,
        state.x,
        tolerance=float(acceptance["maximum_physical_violation"]),
    )
    objective_is_reference = reference_objective is None
    effective_reference = objective if objective_is_reference else float(reference_objective)
    gap = _scaled_gap(objective, effective_reference)
    checks = {
        "finite": all(np.all(np.isfinite(block)) for block in (state.x, state.y, state.z)),
        "paper_stopping_satisfied": residuals.conditions.all_satisfied,
        "raw_kkt_within_tolerance": (
            residuals.combined_norm <= float(acceptance["raw_kkt_tolerance"])
        ),
        "canonical_primal_within_tolerance": (
            canonical["overall_max"] <= float(acceptance["maximum_physical_violation"])
        ),
        "physical_validation_available": bool(physical["available"]),
        "physical_validation_passed": bool(physical["passed"]),
        "objective_gap_within_tolerance": (
            gap <= float(acceptance["maximum_scaled_objective_gap_to_highs"])
        ),
    }
    return {
        "objective": objective,
        "scaled_objective_gap_to_highs": gap,
        "scaled_objective_gap_to_reference": gap,
        "objective_reference": objective_is_reference,
        "reference_objective": effective_reference,
        "residuals": residuals.summary(),
        "canonical_violations": canonical,
        "maximum_canonical_primal_violation": canonical["overall_max"],
        "physical_validation": physical,
        "state_fingerprint": _state_fingerprint(state),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _highs_state(result: Any, model: Stage7ScalableModel) -> HPRState:
    lp = model.lp
    y1 = (
        np.asarray(result.eqlin.marginals, dtype=np.float64)
        if lp.m1
        else np.empty(0, dtype=np.float64)
    )
    y2 = (
        -np.asarray(result.ineqlin.marginals, dtype=np.float64)
        if lp.m2
        else np.empty(0, dtype=np.float64)
    )
    z = np.asarray(result.lower.marginals, dtype=np.float64) + np.asarray(
        result.upper.marginals, dtype=np.float64
    )
    return HPRState(y=np.concatenate((y1, y2)), z=z, x=np.asarray(result.x, dtype=np.float64))


def _solve_highs(
    model: Stage7ScalableModel,
    *,
    time_limit_seconds: int,
) -> dict[str, Any]:
    lp = model.lp
    started = perf_counter()
    result = optimize.linprog(
        lp.c,
        A_ub=-lp.A2 if lp.m2 else None,
        b_ub=-lp.b2 if lp.m2 else None,
        A_eq=lp.A1 if lp.m1 else None,
        b_eq=lp.b1 if lp.m1 else None,
        bounds=list(zip(lp.lower.tolist(), lp.upper.tolist(), strict=True)),
        method="highs-ds",
        options={
            "presolve": True,
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "time_limit": float(time_limit_seconds),
        },
    )
    elapsed = perf_counter() - started
    status = "SUCCESS"
    if not bool(result.success):
        timed_out = int(result.status) == 1 and "time" in str(result.message).lower()
        status = "TIME_LIMIT" if timed_out else "FAIL"
    return {
        "status": status,
        "wall_seconds": elapsed,
        "native_status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "state": _highs_state(result, model) if result.x is not None else None,
    }


def _call_with_adapter(
    function: Callable[..., Any],
    positional: tuple[Any, ...],
    base_kwargs: dict[str, Any],
    *,
    scaled_solver: ScaledBlockArrowY1Solver,
    spectral: SparseSpectralCertificate,
    time_limit_seconds: int,
) -> Any:
    parameters = inspect.signature(function).parameters
    kwargs = dict(base_kwargs)
    if "scaled_structural_y1" in parameters:
        kwargs["scaled_structural_y1"] = scaled_solver
    elif "scaled_equality_solver" in parameters:
        kwargs["scaled_equality_solver"] = scaled_solver
    else:
        raise RuntimeError(f"{function.__name__} lacks the Stage 7 scaled equality adapter")
    if "inequality_lambda" in parameters:
        kwargs["inequality_lambda"] = spectral.lambda_used
    elif "spectral_certificate" in parameters:
        kwargs["spectral_certificate"] = spectral
    else:
        raise RuntimeError(f"{function.__name__} lacks the sparse Stage 7 spectral adapter")
    if "time_limit_seconds" in parameters:
        kwargs["time_limit_seconds"] = time_limit_seconds
    return function(*positional, **kwargs)


def _stage5_control(config: Mapping[str, Any]) -> Stage5Control:
    algorithm = config["algorithm"]
    return Stage5Control(
        adaptive_sigma=bool(algorithm["adaptive_sigma"]),
        restart=bool(algorithm["restart"]),
        check_interval=int(algorithm["policy_check_interval"]),
    )


def _solve_cpu_hpr(
    model: Stage7ScalableModel,
    preconditioner: LPPreconditioner,
    scaled_solver: ScaledBlockArrowY1Solver,
    spectral: SparseSpectralCertificate,
    workspace: SGSHPRWorkspace,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    limit = int(config["timing"]["per_solve_time_limit_seconds"])
    kwargs = {
        "sigma": float(config["algorithm"]["initial_sigma"]),
        "tolerance": float(config["acceptance"]["paper_residual_tolerance"]),
        "kkt_tolerance": float(config["acceptance"]["raw_kkt_tolerance"]),
        "max_iterations": TECHNICAL_ITERATION_CEILING,
        "history_interval": int(config["algorithm"]["policy_check_interval"]),
        "preconditioner": preconditioner,
        "prepared_workspace": workspace,
        "control": _stage5_control(config),
    }
    started = perf_counter()
    with _solve_deadline(limit) as deadline:
        result = _call_with_adapter(
            solve_stage5_sgs_hpr,
            (model.lp,),
            kwargs,
            scaled_solver=scaled_solver,
            spectral=spectral,
            time_limit_seconds=limit,
        )
    return {
        "status": "SUCCESS" if result.converged else "FAIL",
        "wall_seconds": perf_counter() - started,
        "deadline": deadline,
        "result": result,
    }


def _prepare_gpu_problem(
    model: Stage7ScalableModel,
    preconditioner: LPPreconditioner,
    scaled_solver: ScaledBlockArrowY1Solver,
    spectral: SparseSpectralCertificate,
    backend: CuPyBackend,
) -> Any:
    return _call_with_adapter(
        prepare_gpu_stage6_problem,
        (model.lp, preconditioner),
        {"backend": backend, "dtype": "float64"},
        scaled_solver=scaled_solver,
        spectral=spectral,
        time_limit_seconds=1,
    )


def _solve_gpu_hpr(problem: Any, *, config: Mapping[str, Any]) -> dict[str, Any]:
    limit = int(config["timing"]["per_solve_time_limit_seconds"])
    device_memory_before = problem.backend.memory_report().as_dict()
    ledger_before = problem.backend.ledger.summary()
    transfer_timing_before = problem.backend.transfer_timing_summary()
    started = perf_counter()
    result: GPUStage6Result | None = None
    failure: dict[str, Any] | None = None
    status = "FAIL"
    deadline: dict[str, Any] = {}
    try:
        with _solve_deadline(limit) as deadline:
            result = solve_gpu_stage5_sgs_hpr(
                problem,
                sigma=float(config["algorithm"]["initial_sigma"]),
                tolerance=float(config["acceptance"]["paper_residual_tolerance"]),
                kkt_tolerance=float(config["acceptance"]["raw_kkt_tolerance"]),
                max_iterations=TECHNICAL_ITERATION_CEILING,
                residual_check_interval=int(
                    config["algorithm"]["correctness_residual_check_interval"]
                ),
                history_interval=int(config["algorithm"]["policy_check_interval"]),
                control=_stage5_control(config),
                fixed_horizon=False,
                **(
                    {"time_limit_seconds": limit}
                    if "time_limit_seconds"
                    in inspect.signature(solve_gpu_stage5_sgs_hpr).parameters
                    else {}
                ),
            )
        status = "SUCCESS" if result.converged else "FAIL"
    except SolveTimeLimit as error:
        status = "TIME_LIMIT"
        failure = _exception_record("gpu_fp64_sgs_hpr", error)
    except Exception as error:
        status = "FAIL"
        failure = _exception_record("gpu_fp64_sgs_hpr", error)
    device_memory_after = problem.backend.memory_report().as_dict()
    ledger_after = problem.backend.ledger.summary()
    transfer_timing_after = problem.backend.transfer_timing_summary()
    transfer_delta = _transfer_delta(ledger_before, ledger_after)
    response = {
        "status": status,
        "wall_seconds": perf_counter() - started,
        "deadline": deadline,
        "result": result,
        "device_memory": {
            "before": device_memory_before,
            "after": device_memory_after,
            "true_per_solve_peak_available": False,
            "note": (
                "Allocator/device snapshots bracket the solve; CUDA does not expose a "
                "per-solve high-water mark through this backend."
            ),
        },
        "transfer_delta": transfer_delta,
        "transfer_timing_delta": _transfer_timing_delta(
            transfer_timing_before, transfer_timing_after
        ),
        "transfer_audit": _audit_gpu_solver_transfers(transfer_delta),
    }
    if failure is not None:
        response["failure"] = failure
    return response


def _solve_gurobi(
    model: Stage7ScalableModel,
    *,
    time_limit_seconds: int,
) -> dict[str, Any]:
    gp = importlib.import_module("gurobipy")
    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", 0)
    environment.start()
    solver = gp.Model(env=environment)
    try:
        solver.Params.OutputFlag = 0
        solver.Params.TimeLimit = float(time_limit_seconds)
        lp = model.lp
        x = solver.addMVar(lp.n, lb=lp.lower, ub=lp.upper, obj=lp.c, name="x")
        if lp.m1:
            solver.addMConstr(lp.A1, x, "=", lp.b1, name="equality")
        if lp.m2:
            solver.addMConstr(lp.A2, x, ">", lp.b2, name="inequality")
        started = perf_counter()
        solver.optimize()
        elapsed = perf_counter() - started
        success = solver.Status == gp.GRB.OPTIMAL
        time_limit = solver.Status == gp.GRB.TIME_LIMIT
        vector = np.asarray(x.X, dtype=np.float64) if solver.SolCount else None
        return {
            "status": "SUCCESS" if success else ("TIME_LIMIT" if time_limit else "FAIL"),
            "wall_seconds": elapsed,
            "native_status": int(solver.Status),
            "message": "OPTIMAL" if success else f"Gurobi status {solver.Status}",
            "iterations": float(solver.IterCount),
            "x": vector,
            "objective": float(solver.ObjVal) if solver.SolCount else None,
        }
    finally:
        with contextlib.suppress(Exception):
            solver.dispose()
        with contextlib.suppress(Exception):
            environment.dispose()


def _attempt(
    phase: str,
    solve: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    started = perf_counter()
    memory_before = _process_memory()
    try:
        result = solve()
        result["attempt_wall_seconds"] = perf_counter() - started
        result["process_memory"] = {
            "before": memory_before,
            "after": _process_memory(),
        }
        return result
    except SolveTimeLimit as error:
        return {
            "status": "TIME_LIMIT",
            "wall_seconds": perf_counter() - started,
            "attempt_wall_seconds": perf_counter() - started,
            "process_memory": {"before": memory_before, "after": _process_memory()},
            "failure": _exception_record(phase, error),
        }
    except Exception as error:
        return {
            "status": "FAIL",
            "wall_seconds": perf_counter() - started,
            "attempt_wall_seconds": perf_counter() - started,
            "process_memory": {"before": memory_before, "after": _process_memory()},
            "failure": _exception_record(phase, error),
        }


def _summarize_hpr_attempt(
    attempt: Mapping[str, Any],
    *,
    model: Stage7ScalableModel,
    config: Mapping[str, Any],
    reference_objective: float | None,
) -> dict[str, Any]:
    result = attempt.get("result")
    summary = {key: value for key, value in attempt.items() if key != "result"}
    if not isinstance(result, (Stage5SGSHPRResult, GPUStage6Result)):
        return summary
    candidate = _candidate_summary(
        model,
        result.solution,
        config=config,
        reference_objective=reference_objective,
    )
    summary.update(
        {
            "iterations": result.iterations,
            "converged": result.converged,
            "restart_count": result.restart_count,
            "sigma": result.sigma,
            "candidate": candidate,
            "passed": bool(attempt.get("status") == "SUCCESS" and candidate["passed"]),
        }
    )
    if isinstance(result, GPUStage6Result):
        summary["native_gpu_timing"] = result.timing.as_dict()
        summary["transfer_ledger"] = result.transfer_ledger
    else:
        summary["native_cpu_timing"] = {
            "preparation_elapsed_seconds": result.preparation_elapsed_seconds,
            "total_elapsed_seconds": result.total_elapsed_seconds,
        }
    return summary


def _run_timed_track(
    *,
    track: dict[str, Any],
    solve: Callable[[], dict[str, Any]],
    summarize: Callable[[Mapping[str, Any]], dict[str, Any]],
    config: Mapping[str, Any],
    checkpoint: Callable[[], None],
) -> None:
    timing = config["timing"]
    track.setdefault("correctness", None)
    track.setdefault("warmup", [])
    track.setdefault("measured_repetitions", [])

    def valid_persisted_attempt(row: Any, *, repetition: int | None) -> bool:
        if not isinstance(row, Mapping):
            return False
        if row.get("status") != "SUCCESS" or row.get("passed") is not True:
            return False
        return repetition is None or row.get("repetition") == repetition

    if track["correctness"] is None:
        raw = _attempt(f"{track['name']}:correctness", solve)
        track["correctness"] = summarize(raw)
        track["first_run"] = track["correctness"]
        track["first_run_meaning"] = (
            "untimed-for-statistics correctness run executed before warm-up and measurement"
        )
        checkpoint()
    elif "first_run" not in track:
        track["first_run"] = track["correctness"]
        track["first_run_meaning"] = (
            "alias of the correctness run; excluded from measured statistics"
        )
    correctness = track["correctness"]
    if not valid_persisted_attempt(correctness, repetition=None):
        track["timing_status"] = "NOT_RUN_CORRECTNESS_FAILED"
        track["passed"] = False
        checkpoint()
        return

    warmups = int(timing["warmup_runs"])
    if not isinstance(track["warmup"], list) or len(track["warmup"]) > warmups:
        track["timing_status"] = "INVALID_RESUMED_WARMUP_STATE"
        track["passed"] = False
        checkpoint()
        return
    if any(
        not valid_persisted_attempt(row, repetition=index)
        for index, row in enumerate(track["warmup"], start=1)
    ):
        track["timing_status"] = "INVALID_RESUMED_WARMUP_STATE"
        track["passed"] = False
        checkpoint()
        return
    while len(track["warmup"]) < warmups:
        raw = _attempt(f"{track['name']}:warmup", solve)
        row = summarize(raw)
        row["repetition"] = len(track["warmup"]) + 1
        track["warmup"].append(row)
        checkpoint()
        if row.get("status") != "SUCCESS" or not bool(row.get("passed", False)):
            track["timing_status"] = "WARMUP_FAILED_OR_INVALID"
            track["passed"] = False
            return

    required = int(timing["measured_runs"])
    maximum = int(timing["maximum_measured_runs_after_variability_escalation"])
    threshold = float(timing["relative_range_escalation_threshold"])
    if (
        not isinstance(track["measured_repetitions"], list)
        or len(track["measured_repetitions"]) > maximum
    ):
        track["timing_status"] = "INVALID_RESUMED_MEASUREMENT_STATE"
        track["passed"] = False
        checkpoint()
        return
    if any(
        not valid_persisted_attempt(row, repetition=index)
        for index, row in enumerate(track["measured_repetitions"], start=1)
    ):
        track["timing_status"] = "INVALID_RESUMED_MEASUREMENT_STATE"
        track["passed"] = False
        checkpoint()
        return
    while len(track["measured_repetitions"]) < required:
        raw = _attempt(f"{track['name']}:measured", solve)
        row = summarize(raw)
        row["repetition"] = len(track["measured_repetitions"]) + 1
        track["measured_repetitions"].append(row)
        checkpoint()
        if row.get("status") != "SUCCESS" or not bool(row.get("passed", False)):
            track["timing_status"] = (
                row.get("status", "FAILED")
                if row.get("status") != "SUCCESS"
                else "INVALID_MEASURED_CANDIDATE"
            )
            track["passed"] = False
            return

    samples = [float(row["wall_seconds"]) for row in track["measured_repetitions"]]
    stats = _timing_statistics(samples)
    escalated = bool(stats["relative_range"] is not None and stats["relative_range"] > threshold)
    target = maximum if escalated else required
    while len(track["measured_repetitions"]) < target:
        raw = _attempt(f"{track['name']}:variability_escalation", solve)
        row = summarize(raw)
        row["repetition"] = len(track["measured_repetitions"]) + 1
        track["measured_repetitions"].append(row)
        checkpoint()
        if row.get("status") != "SUCCESS" or not bool(row.get("passed", False)):
            track["timing_status"] = (
                row.get("status", "FAILED")
                if row.get("status") != "SUCCESS"
                else "INVALID_MEASURED_CANDIDATE"
            )
            track["passed"] = False
            return
    samples = [float(row["wall_seconds"]) for row in track["measured_repetitions"]]
    track["statistics"] = _timing_statistics(samples)
    attempt_samples = [float(row["attempt_wall_seconds"]) for row in track["measured_repetitions"]]
    track["attempt_wall_statistics"] = _timing_statistics(attempt_samples)
    track["first_measured_run"] = track["measured_repetitions"][0]
    track["variability_escalated"] = escalated
    track.setdefault(
        "timing_boundary",
        (
            "solver call from zero initial state through stopping and final state; "
            "see attempt_wall_statistics for the Python call boundary"
        ),
    )
    track["timing_status"] = "COMPLETE"
    track["passed"] = True
    checkpoint()


def _input_for_case(config: Mapping[str, Any], case_name: str) -> Path:
    for item in config["public_network_source"]["files"]:
        if item["case"] == case_name:
            return (PROJECT_ROOT / item["path"]).resolve()
    raise Stage7ContractError(f"no pinned input for {case_name}")


def _case_structure(model: Stage7ScalableModel) -> DCOPFEqualityStructure:
    return DCOPFEqualityStructure(
        periods=model.fleet.periods,
        generator_count=model.normalized.spec.generators,
        renewable_count=model.normalized.spec.renewables,
        interval_hours=model.fleet.interval_hours,
        charge_efficiencies=tuple(model.fleet.storage_charge_efficiency),
        discharge_efficiencies=tuple(model.fleet.storage_discharge_efficiency),
    )


def _precondition(model: Stage7ScalableModel, config: Mapping[str, Any]) -> LPPreconditioner:
    algorithm = config["algorithm"]
    return precondition_lp(
        model.lp,
        ruiz_iterations=int(algorithm["ruiz_iterations"]),
        pock_chambolle=bool(algorithm["pock_chambolle"]),
        normalize=bool(algorithm["normalize_b_and_c"]),
    )


def _run_case(
    case: dict[str, Any],
    *,
    key: CaseKey,
    config: Mapping[str, Any],
    availability: Mapping[str, Any],
    device_id: int,
    checkpoint: Callable[[], None],
) -> None:
    case_wall_started = perf_counter()
    case["status"] = "RUNNING"
    case["started_utc"] = case.get("started_utc", _utc_now())
    checkpoint()
    preflight = next(
        item
        for item in all_stage7_preflights()
        if item.row.case_name == key.case_name and item.row.periods == key.periods
    )
    guard = _memory_guard(
        preflight,
        host_memory=_host_memory(),
        device_total_bytes=None,
    )
    case["preflight"] = guard
    if not guard["passed"]:
        case["status"] = "MEMORY_BLOCKED"
        case["passed"] = False
        checkpoint()
        return

    construction_started = perf_counter()
    network_path = _input_for_case(config, key.case_name)
    network = load_matpower_case(network_path)
    model = build_stage7_scalable_model(
        network,
        key.periods,
        host_memory_budget_bytes=guard["host_budget_bytes"],
    )
    case["construction"] = {
        "wall_seconds": perf_counter() - construction_started,
        "dimensions": model.dimension_summary(),
        "lp_fingerprint": _lp_fingerprint(model),
        "policy_fingerprint": model.fleet.policy_fingerprint,
        "input_sha256": next(
            str(item["sha256"])
            for item in config["public_network_source"]["files"]
            if item["case"] == key.case_name
        ),
        "input_sha256_definition": config["public_network_source"]["sha256_definition"],
    }
    expected_nnz = preflight.row.published_nnz
    actual_nnz = int(model.lp.A1.nnz + model.lp.A2.nnz)
    count_only_ledger = stage7_reconstructed_nnz_ledger(
        model.normalized,
        key.periods,
        fleet=model.fleet,
        ptdf=model.ptdf,
    )
    dimensions = model.dimension_summary()
    dimension_match = (
        dimensions["m"] == preflight.row.published_m
        and dimensions["n"] == preflight.row.published_n
    )
    case["structural_reconciliation"] = {
        "dimension_match": dimension_match,
        "published_nnz": expected_nnz,
        "actual_nnz": actual_nnz,
        "nnz_difference": actual_nnz - expected_nnz,
        "symbolic_reconstructed_nnz": count_only_ledger.reconstructed_nnz,
        "actual_matches_symbolic_nnz": actual_nnz == count_only_ledger.reconstructed_nnz,
        "paper_time_comparable": actual_nnz == expected_nnz,
        "classification": "structural_reproduction_not_author_instance",
    }
    if not dimension_match:
        raise Stage7ContractError(f"{key.text} dimensions differ from the frozen table")
    if actual_nnz != count_only_ledger.reconstructed_nnz:
        raise Stage7ContractError(f"{key.text} allocated nnz differs from the symbolic ledger")
    checkpoint()

    preprocessing_started = perf_counter()
    preconditioner = _precondition(model, config)
    structure = _case_structure(model)
    scaled_solver = prepare_scaled_block_arrow_y1(preconditioner, structure)
    spectral = estimate_sparse_spectral_norm_squared(
        preconditioner.scaled_lp.A2,
        power_seed=int(config["reconstruction_protocol"]["seed"]),
    )
    cpu_workspace_started = perf_counter()
    cpu_workspace = prepare_sgs_hpr(
        preconditioner.scaled_lp,
        scaled_structural_y1=scaled_solver,
        spectral_certificate=spectral,
    )
    cpu_workspace_setup_seconds = perf_counter() - cpu_workspace_started
    case["preprocessing"] = {
        "wall_seconds": perf_counter() - preprocessing_started,
        "preconditioner": _clean_json(preconditioner.diagnostics),
        "scaled_equality": scaled_solver.diagnostics.summary(),
        "sparse_spectral_certificate": spectral.summary(),
        "cpu_workspace_setup_wall_seconds": cpu_workspace_setup_seconds,
        "cpu_workspace": {
            "equality_backend": cpu_workspace.equality_backend,
            "prepared_once_and_reused": True,
            "dense_equality_gram_materialized": cpu_workspace.equality_gram is not None,
            "spectral_certificate_reused": True,
        },
    }
    checkpoint()

    tracks = case.setdefault("solver_tracks", {})
    limit = int(config["timing"]["per_solve_time_limit_seconds"])

    highs_track = tracks.setdefault(
        "highs",
        {
            "name": "highs",
            "timing_boundary": (
                "SciPy linprog call including HiGHS interface/model setup and solve"
            ),
        },
    )

    def summarize_highs(attempt: Mapping[str, Any]) -> dict[str, Any]:
        state = attempt.get("state")
        summary = {key_: value for key_, value in attempt.items() if key_ != "state"}
        if not isinstance(state, HPRState):
            summary["passed"] = False
            return summary
        candidate = _candidate_summary(
            model,
            state,
            config=config,
            reference_objective=None,
        )
        summary["candidate"] = candidate
        summary["passed"] = bool(attempt.get("status") == "SUCCESS" and candidate["passed"])
        return summary

    _run_timed_track(
        track=highs_track,
        solve=lambda: _solve_highs(model, time_limit_seconds=limit),
        summarize=summarize_highs,
        config=config,
        checkpoint=checkpoint,
    )
    reference_objective = highs_track.get("correctness", {}).get("candidate", {}).get("objective")
    if reference_objective is None:
        case["status"] = "FAIL"
        case["passed"] = False
        case["end_to_end_case_wall_seconds"] = perf_counter() - case_wall_started
        checkpoint()
        return

    cpu_track = tracks.setdefault(
        "cpu_fp64_sgs_hpr",
        {
            "name": "cpu_fp64_sgs_hpr",
            "timing_boundary": (
                "prepared CPU sparse workspace solve including iteration loop, stopping "
                "checks, and final original-space residual evaluation; one-time workspace "
                "setup is recorded in case preprocessing"
            ),
        },
    )
    if not availability["cpu_fp64_sgs_hpr"]["available"]:
        cpu_track.update(
            {
                "status": "UNAVAILABLE",
                "reason": availability["cpu_fp64_sgs_hpr"]["reason"],
                "passed": False,
            }
        )
        checkpoint()
    else:
        _run_timed_track(
            track=cpu_track,
            solve=lambda: _solve_cpu_hpr(
                model,
                preconditioner,
                scaled_solver,
                spectral,
                cpu_workspace,
                config=config,
            ),
            summarize=lambda attempt: _summarize_hpr_attempt(
                attempt,
                model=model,
                config=config,
                reference_objective=float(reference_objective),
            ),
            config=config,
            checkpoint=checkpoint,
        )

    gpu_track = tracks.setdefault(
        "gpu_fp64_sgs_hpr",
        {
            "name": "gpu_fp64_sgs_hpr",
            "timing_boundary": (
                "prepared resident GPU workspace solve including zero-state upload, iteration "
                "loop, stopping checks, and final state recovery/transfer"
            ),
        },
    )
    backend: CuPyBackend | None = None
    if not availability["gpu_fp64_sgs_hpr"]["available"]:
        gpu_track.update(
            {
                "status": "UNAVAILABLE",
                "reason": availability["gpu_fp64_sgs_hpr"]["reason"],
                "passed": False,
            }
        )
        checkpoint()
    else:
        try:
            backend = create_gpu_backend(device_id=device_id)
            backend.synchronize()
            device = backend.diagnostics.as_dict()
            memory = backend.memory_report().as_dict()
            device_total = int(device["total_global_memory_bytes"])
            gpu_guard = _memory_guard(
                preflight,
                host_memory=_host_memory(),
                device_total_bytes=device_total,
            )
            gpu_track["device"] = device
            gpu_track["memory_before"] = memory
            gpu_track["preflight"] = gpu_guard
            if not gpu_guard["passed"]:
                raise MemoryError("GPU preflight safety guard rejected the allocation")
            ledger_before = backend.ledger.summary()
            gpu_setup_started = perf_counter()
            problem = _prepare_gpu_problem(
                model,
                preconditioner,
                scaled_solver,
                spectral,
                backend,
            )
            gpu_track["workspace_setup_wall_seconds"] = perf_counter() - gpu_setup_started
            ledger_after_preparation = backend.ledger.summary()
            gpu_track["preparation_transfer_delta"] = _transfer_delta(
                ledger_before, ledger_after_preparation
            )
            kernels = {
                "A1": problem.workspace.A1_resident.kernel.as_dict(),
                "A2": problem.workspace.A2_resident.kernel.as_dict(),
            }
            kernel_checks = {
                "requested_algorithm": config["algorithm"]["requested_spmv_algorithm"],
                "A1_uses_csr_alg2": bool(kernels["A1"].get("uses_csr_alg2", False)),
                "A2_uses_csr_alg2": bool(kernels["A2"].get("uses_csr_alg2", False)),
                "FP64": problem.dtype_name == "float64",
                "scaled_structural_equality": (
                    problem.workspace.equality_mode == "scaled_structural"
                ),
            }
            gpu_track["kernel_selection"] = kernels
            gpu_track["kernel_checks"] = {
                **kernel_checks,
                "passed": all(
                    value for key_, value in kernel_checks.items() if key_ != "requested_algorithm"
                ),
            }
            checkpoint()
            if not gpu_track["kernel_checks"]["passed"]:
                gpu_track.update(
                    {
                        "status": "FAIL",
                        "reason": (
                            "the requested FP64 CSR_ALG2 scaled-structural path was not selected"
                        ),
                        "passed": False,
                    }
                )
            else:
                _run_timed_track(
                    track=gpu_track,
                    solve=lambda: _solve_gpu_hpr(problem, config=config),
                    summarize=lambda attempt: _summarize_hpr_attempt(
                        attempt,
                        model=model,
                        config=config,
                        reference_objective=float(reference_objective),
                    ),
                    config=config,
                    checkpoint=checkpoint,
                )
            gpu_track["memory_after"] = backend.memory_report().as_dict()
            gpu_track["cumulative_transfer_ledger"] = backend.ledger.summary()
        except (GPUBackendUnavailable, MemoryError, RuntimeError, ValueError) as error:
            gpu_track.update(
                {
                    "status": "UNAVAILABLE" if isinstance(error, GPUBackendUnavailable) else "FAIL",
                    "failure": _exception_record("gpu_preparation", error),
                    "passed": False,
                }
            )
        checkpoint()

    gurobi_track = tracks.setdefault(
        "gurobi",
        {
            "name": "gurobi",
            "timing_boundary": (
                "Gurobi optimize core; per-repetition Python/model-interface setup is "
                "separately retained in attempt_wall_seconds"
            ),
        },
    )
    if not availability["gurobi"]["available"]:
        gurobi_track.update(
            {
                "status": "NOT_REQUIRED_UNAVAILABLE_OR_UNLICENSED",
                "availability": availability["gurobi"],
                "passed": True,
                "gating": False,
            }
        )
        checkpoint()
    else:

        def summarize_gurobi(attempt: Mapping[str, Any]) -> dict[str, Any]:
            x = attempt.get("x")
            summary = {key_: value for key_, value in attempt.items() if key_ != "x"}
            if x is None:
                summary["passed"] = False
                return summary
            physical = _physical_validation(
                model,
                x,
                tolerance=float(config["acceptance"]["maximum_physical_violation"]),
            )
            canonical = _canonical_violations(model, x)
            objective = model.objective(x)
            gap = _scaled_gap(objective, float(reference_objective))
            summary.update(
                {
                    "objective": objective,
                    "scaled_objective_gap_to_highs": gap,
                    "scaled_objective_gap_to_reference": gap,
                    "objective_reference": False,
                    "reference_objective": float(reference_objective),
                    "canonical_violations": canonical,
                    "maximum_canonical_primal_violation": canonical["overall_max"],
                    "physical_validation": physical,
                    "x_fingerprint": _hash_array(x),
                    "passed": bool(
                        attempt.get("status") == "SUCCESS"
                        and physical["passed"]
                        and canonical["overall_max"]
                        <= float(config["acceptance"]["maximum_physical_violation"])
                        and gap
                        <= float(config["acceptance"]["maximum_scaled_objective_gap_to_highs"])
                    ),
                }
            )
            return summary

        _run_timed_track(
            track=gurobi_track,
            solve=lambda: _solve_gurobi(model, time_limit_seconds=limit),
            summarize=summarize_gurobi,
            config=config,
            checkpoint=checkpoint,
        )

    required = config["acceptance"]["required_solver_tracks"]
    required_passed = all(bool(tracks[name].get("passed", False)) for name in required)
    gurobi_passed = bool(gurobi_track.get("passed", False))
    case["passed"] = bool(required_passed and gurobi_passed)
    statuses = {
        tracks[name].get("timing_status") or tracks[name].get("status") for name in required
    }
    case["status"] = (
        "PASS" if case["passed"] else ("TIME_LIMIT" if "TIME_LIMIT" in statuses else "FAIL")
    )
    case["completed_utc"] = _utc_now()
    case["end_to_end_case_wall_seconds"] = perf_counter() - case_wall_started
    case["timing_boundaries"] = {
        "model_construction_wall_seconds": case["construction"]["wall_seconds"],
        "preprocessing_wall_seconds": case["preprocessing"]["wall_seconds"],
        "cpu_workspace_setup_wall_seconds": case["preprocessing"][
            "cpu_workspace_setup_wall_seconds"
        ],
        "gpu_workspace_setup_wall_seconds": gpu_track.get("workspace_setup_wall_seconds"),
        "end_to_end_case_wall_seconds": case["end_to_end_case_wall_seconds"],
        "solver_core_samples_are_stored_per_track": True,
        "speedup_computed": False,
    }
    checkpoint()


def _parse_requested_keys(values: Sequence[str]) -> set[str]:
    keys: set[str] = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"case selector must be NAME:T, received {value!r}")
        name, raw_periods = value.rsplit(":", 1)
        raw_periods = raw_periods.removeprefix("T").removeprefix("t")
        keys.add(CaseKey(name, int(raw_periods)).text)
    return keys


def _fresh_case_for_retry(
    case: Mapping[str, Any],
    *,
    key: CaseKey,
    fingerprint: str,
) -> dict[str, Any]:
    """Restart one terminal failed case without reusing invalid solver samples."""

    tracks = case.get("solver_tracks", {})
    prior_track_status: dict[str, Any] = {}
    if isinstance(tracks, Mapping):
        for name, value in tracks.items():
            if isinstance(value, Mapping):
                prior_track_status[str(name)] = value.get("timing_status") or value.get("status")
    retry_record = {
        "retried_utc": _utc_now(),
        "prior_status": case.get("status"),
        "prior_passed": case.get("passed"),
        "prior_failure": case.get("failure"),
        "prior_solver_timing_status": prior_track_status,
    }
    return {
        "key": key.text,
        "case_name": key.case_name,
        "periods": key.periods,
        "run_fingerprint": fingerprint,
        "status": "PENDING",
        "passed": False,
        "retry_history": [*_sequence_or_empty(case.get("retry_history")), retry_record],
    }


def _sequence_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _source_manifest(config_path: Path) -> list[dict[str, Any]]:
    paths = [
        Path(__file__).resolve(),
        config_path,
        DEFAULT_REQUIREMENTS,
        *sorted((SOURCE_ROOT / "gpu_dcopf_hpr").glob("*.py")),
    ]
    manifest: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        identity = _canonical_git_blob_identity(Path(relative), path)
        manifest.append(
            {
                "path": relative,
                "git_blob": identity["expected_git_blob"],
                "sha256": identity["canonical_git_blob_sha256"],
                "sha256_definition": CANONICAL_GIT_BLOB_SHA256_DEFINITION,
                "passed": identity["passed"],
            }
        )
    return manifest


def _run_fingerprint(
    provenance: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "config_sha256": provenance["config"]["sha256"],
        "inputs": [
            {"path": row["path"], "sha256": row["actual_sha256"]} for row in provenance["files"]
        ],
        "sources": list(sources),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cupy_cuda13x": _package_version("cupy-cuda13x"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _compatible_partial(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return candidate if candidate.get("run_fingerprint") == fingerprint else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    partial_path = output_dir / PARTIAL_NAME
    final_path = output_dir / FINAL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = _validate_stage7_config(config)
    provenance = _verify_provenance(config, config_path)
    policy = _policy_contract(config)
    sources = _source_manifest(config_path)
    if any(row.get("passed") is not True for row in sources):
        errors.append("the canonical source manifest does not match the executed worktree")
    fingerprint = _run_fingerprint(provenance, sources)
    resumed = _compatible_partial(partial_path, fingerprint) if args.resume else None
    evidence: dict[str, Any] = resumed or {
        "schema_version": "1.0",
        "stage": 7,
        "status": "RUNNING",
        "all_passed": False,
        "started_utc": _utc_now(),
        "run_fingerprint": fingerprint,
        "configuration": config,
        "configuration_validation": {"errors": errors, "passed": not errors},
        "policy_contract": policy,
        "provenance": provenance,
        "environment": _environment(),
        "source_manifest": sources,
        "requirements_freeze": _requirements_freeze(),
        "solver_availability": _solver_availability(),
        "symbolic_ledger": [],
        "cases": [],
        "failures": [],
        "stage_boundary": {
            "stage_7_only": True,
            "stage_7_executed": False,
            "stage_7_complete": False,
            "stage_8_large_runs_locked": True,
            "stage_8_allocation_count": 0,
            "n_minus_1_extension_enabled": False,
            "exact_paper_reproduction_claimed": False,
            "paper_a100_timing_reproduction_claimed": False,
        },
    }
    if resumed is not None and args.retry_failed and evidence.get("failures"):
        evidence.setdefault("retry_history", []).append(
            {
                "retried_utc": _utc_now(),
                "prior_status": evidence.get("status"),
                "prior_failures": list(evidence.get("failures", [])),
            }
        )
        evidence["failures"] = []

    def checkpoint() -> None:
        evidence["updated_utc"] = _utc_now()
        _atomic_write_json(partial_path, evidence)

    try:
        if errors:
            raise Stage7ContractError("; ".join(errors))
        if not provenance["passed"]:
            raise Stage7ContractError("; ".join(provenance["errors"]))
        if not policy["passed"]:
            raise Stage7ContractError(
                "the scalable model policy differs from the frozen JSON contract"
            )
        networks = {
            str(item["case"]): load_matpower_case((PROJECT_ROOT / str(item["path"])).resolve())
            for item in config["public_network_source"]["files"]
        }
        evidence["symbolic_ledger"] = _symbolic_ledger(config, networks=networks)
        checkpoint()
        if args.ledger_only:
            evidence["status"] = "LEDGER_ONLY_COMPLETE"
            evidence["all_passed"] = all(bool(row["passed"]) for row in evidence["symbolic_ledger"])
            evidence["completed_utc"] = _utc_now()
            checkpoint()
            _atomic_write_json(final_path, evidence)
            return 0 if evidence["all_passed"] else 1

        requested = _parse_requested_keys(args.case_keys)
        executable = [row for row in evidence["symbolic_ledger"] if row["execute_stage_7"]]
        if requested:
            allowed = {str(row["key"]) for row in executable}
            invalid = requested - allowed
            if invalid:
                raise Stage7ContractError(
                    "requested rows are not preregistered Stage 7 allocations: "
                    + ", ".join(sorted(invalid))
                )
            executable = [row for row in executable if row["key"] in requested]
        evidence["stage_boundary"]["stage_7_executed"] = True
        cases_by_key = {str(case["key"]): case for case in evidence["cases"]}
        for row in executable:
            key = CaseKey(str(row["case_name"]), int(row["periods"]))
            case = cases_by_key.get(key.text)
            if case is None:
                case = {
                    "key": key.text,
                    "case_name": key.case_name,
                    "periods": key.periods,
                    "run_fingerprint": fingerprint,
                    "status": "PENDING",
                    "passed": False,
                }
                evidence["cases"].append(case)
                cases_by_key[key.text] = case
            elif (
                args.retry_failed
                and case.get("status") in TERMINAL_CASE_STATUSES
                and case.get("status") != "PASS"
            ):
                replacement = _fresh_case_for_retry(
                    case,
                    key=key,
                    fingerprint=fingerprint,
                )
                position = evidence["cases"].index(case)
                evidence["cases"][position] = replacement
                case = replacement
                cases_by_key[key.text] = case
                checkpoint()
            if case.get("status") in TERMINAL_CASE_STATUSES and not (
                args.retry_failed and case.get("status") != "PASS"
            ):
                continue
            try:
                _run_case(
                    case,
                    key=key,
                    config=config,
                    availability=evidence["solver_availability"],
                    device_id=args.device_id,
                    checkpoint=checkpoint,
                )
                row["actual_reconstruction_nnz"] = case.get("structural_reconciliation", {}).get(
                    "actual_nnz"
                )
                if row["actual_reconstruction_nnz"] is not None:
                    row["nnz_difference_from_paper"] = int(row["actual_reconstruction_nnz"]) - int(
                        row["published_nnz"]
                    )
                    row["paper_time_comparable"] = row["nnz_difference_from_paper"] == 0
                    row["paper_time_comparability_reason"] = (
                        "exact reconstructed nnz match"
                        if row["paper_time_comparable"]
                        else "reconstructed nnz differs; Table II timing is context only"
                    )
            except Exception as error:
                failure = _exception_record(f"case:{key.text}", error)
                case.update({"status": "FAIL", "passed": False, "failure": failure})
                evidence["failures"].append(failure)
            checkpoint()

        selected_keys = {str(row["key"]) for row in executable}
        selected_cases = [case for case in evidence["cases"] if case["key"] in selected_keys]
        complete_selection = bool(selected_cases) and len(selected_cases) == len(selected_keys)
        evidence["all_passed"] = bool(
            complete_selection and all(bool(case.get("passed", False)) for case in selected_cases)
        )
        full_campaign = len(selected_keys) == EXECUTABLE_CASE_COUNT
        evidence["stage_boundary"]["stage_7_complete"] = bool(
            full_campaign and evidence["all_passed"]
        )
        evidence["status"] = "PASS" if evidence["all_passed"] else "FAIL"
        evidence["completed_utc"] = _utc_now()
        checkpoint()
        _atomic_write_json(final_path, evidence)
        return 0 if evidence["all_passed"] else 1
    except Exception as error:
        failure = _exception_record("stage7_runner", error)
        evidence.setdefault("failures", []).append(failure)
        evidence["status"] = "FAIL"
        evidence["all_passed"] = False
        evidence["completed_utc"] = _utc_now()
        checkpoint()
        _atomic_write_json(final_path, evidence)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
