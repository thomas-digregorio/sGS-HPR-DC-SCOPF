"""Independently validate the frozen Stage 7 DGX benchmark evidence.

The checker is deliberately fail-closed.  It recomputes every acceptance and
timing decision from recorded scalar evidence, authenticates the executed
source tree against the recorded Git commit, and never treats a runner-owned
``passed`` flag as sufficient proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "benchmarks" / "stage_7_small_medium.json"
DEFAULT_EVIDENCE = PROJECT_ROOT / "results" / "raw" / "stage_7" / "stage_7_validation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_7" / "stage_7_checks.json"

FROZEN_CONFIG_SHA256 = "06a172463049c519ab14c446d8b9ab632cd91c8afa4b44264e284b3a4f59a062"
FROZEN_REQUIREMENTS_SHA256 = "827065b5bfc2920492cfe653e922cd2d3b2b4289ade12b06d866bea83d32dacf"
POLICY_FINGERPRINT = "e6911ef7e5ccab32a8392c917b892eeabbed3df16a44b1e342cd8ef664274dcf"
CANONICAL_GIT_BLOB_SHA256_DEFINITION = "SHA-256 of canonical Git blob bytes with LF text content"
TIMING_BOUNDARIES = {
    "highs": "SciPy linprog call including HiGHS interface/model setup and solve",
    "cpu_fp64_sgs_hpr": (
        "prepared CPU sparse workspace solve including iteration loop, stopping checks, "
        "and final original-space residual evaluation; one-time workspace setup is recorded "
        "in case preprocessing"
    ),
    "gpu_fp64_sgs_hpr": (
        "prepared resident GPU workspace solve including zero-state upload, iteration loop, "
        "stopping checks, and final state recovery/transfer"
    ),
    "gurobi": (
        "Gurobi optimize core; per-repetition Python/model-interface setup is separately "
        "retained in attempt_wall_seconds"
    ),
}

PAPER_TOLERANCE = 5e-5
RAW_KKT_TOLERANCE = 1e-2
PHYSICAL_TOLERANCE = 1e-2
OBJECTIVE_GAP_TOLERANCE = 2e-4
SOLVE_TIME_LIMIT_SECONDS = 3600
MEASURED_RUNS = 5
ESCALATED_RUNS = 9
VARIABILITY_THRESHOLD = 0.2

REQUIRED_TRACKS = ("highs", "cpu_fp64_sgs_hpr", "gpu_fp64_sgs_hpr")
EXPECTED_CASES = {
    "case1354pegase:T4",
    "case1354pegase:T16",
    "case1354pegase:T48",
    "case1354pegase:T96",
    "case2868rte:T4",
    "case2868rte:T16",
}

# These two count-only Stage 8 rows exceed the signed 32-bit CSR nonzero
# planning limit.  They are intentionally audited without allocating an LP.
# Every Stage 7-executed row, and every other locked row, remains CSR32-safe.
CSR32_UNSUPPORTED_LOCKED_ROWS = {
    "case9241pegase:T24",
    "case9241pegase:T32",
}

# key: (published m, published n, published nnz, reconstructed nnz, execute Stage 7)
EXPECTED_ROWS: dict[str, tuple[int, int, int, int, bool]] = {
    "case1354pegase:T4": (20192, 4208, 7190640, 4799808, True),
    "case1354pegase:T16": (82124, 16832, 28791792, 19228464, True),
    "case1354pegase:T48": (247276, 50496, 86586352, 57896368, True),
    "case1354pegase:T96": (495004, 100992, 173800432, 116420464, True),
    "case2868rte:T4": (40163, 9488, 30111616, 19073056, True),
    "case2868rte:T16": (163823, 37952, 120508576, 76354336, True),
    "case2868rte:T48": (493583, 113856, 295998240, 229507104, False),
    "case2868rte:T56": (576023, 132832, 345459808, 267886816, False),
    "case2868rte:T64": (658463, 151808, 394957984, 306303136, False),
    "case2868rte:T72": (740903, 170784, 444492768, 344756064, False),
    "case2868rte:T80": (823343, 189760, 494064160, 383245600, False),
    "case2868rte:T88": (905783, 208736, 543672160, 421771744, False),
    "case2868rte:T96": (988223, 227712, 593316768, 460334496, False),
    "case9241pegase:T4": (152774, 24700, 373238888, 342863272, False),
    "case9241pegase:T6": (230376, 37050, 559872262, 514308838, False),
    "case9241pegase:T16": (618386, 98800, 1493149532, 1371647068, False),
    "case9241pegase:T24": (928794, 148200, 2239903828, 2057650132, False),
    "case9241pegase:T32": (1239202, 197600, 2986775884, 2743770956, False),
}

PROVENANCE = {
    "data/raw/matpower/stage7/case1354pegase.m": (
        "case1354pegase",
        "1b08b25a2f6c1d540d090009dfaff41ff2b05784a2d8d302a7ad695821557b89",
        "d6ede376f35af472b45b93ae771209c483427c26",
    ),
    "data/raw/matpower/stage7/case2868rte.m": (
        "case2868rte",
        "2b30e8943daf84ccb111cee30f19f4917afc9c3772cab3ce9eaf6193988a6861",
        "0223116b52b3bd10786ccd61a808c440826aacdc",
    ),
    "data/raw/matpower/stage7/case9241pegase.m": (
        "case9241pegase",
        "593a58ecddb5af509ff94410a6630f81021b48fa31da0694ff516acfa9ea5f3b",
        "cc9816b188ef38725c1e7c5b04cb9555b6b8a78e",
    ),
}

PHYSICAL_FIELDS = {
    "equation_1_power_balance_mw",
    "equation_2_line_limit_mw",
    "equation_3_reserve_box_mw",
    "equation_4_headroom_footroom_mw",
    "equation_5_reserve_requirement_mw",
    "equation_6_generator_ramp_mw",
    "equation_7_renewable_box_mw",
    "equation_8_storage_energy_mwh",
    "equation_9_terminal_energy_mwh",
    "equation_10_storage_power_box_mw",
    "angle_vs_compressed_ptdf_flow_max_abs_mw",
}

GPU_TRANSFER_PHASES = {
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


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _close(
    left: Any,
    right: Any,
    *,
    rel_tol: float = 1e-11,
    abs_tol: float = 1e-13,
) -> bool:
    return (
        _finite(left)
        and _finite(right)
        and math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _requirements_pins(path: Path) -> dict[str, str] | None:
    """Parse an all-pinned requirements file, failing closed on malformed rows."""

    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    pins: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            return None
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version or name in pins:
            return None
        pins[name] = version
    return pins


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"missing {path}"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {}, f"{type(error).__name__}: {error}"
    if not isinstance(result, dict):
        return {}, "top-level JSON value is not an object"
    return result, None


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _git_bytes(*arguments: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _git_blob_sha256(commit: str, path: str) -> str | None:
    value = _git_bytes("show", f"{commit}:{path}")
    return None if value is None else _sha256_bytes(value)


def _git_blob_oid(commit: str, path: str) -> str | None:
    value = _git_bytes("rev-parse", f"{commit}:{path}")
    return None if value is None else value.decode().strip()


def _git_tree_paths(commit: str) -> list[str] | None:
    value = _git_bytes("ls-tree", "-r", "--name-only", commit, "--", "src/gpu_dcopf_hpr")
    if value is None:
        return None
    return sorted(
        path
        for path in value.decode("utf-8").splitlines()
        if path.startswith("src/gpu_dcopf_hpr/") and path.endswith(".py")
    )


def _git_worktree_identity(
    path: str,
) -> tuple[str | None, str | None, str | None]:
    """Return the filtered blob, tracked diff, and path-specific status."""

    blob = _git_bytes("hash-object", f"--path={path}", path)
    difference = _git_bytes(
        "diff",
        "--name-only",
        "--",
        path,
    )
    status = _git_bytes(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        path,
    )
    return (
        None if blob is None else blob.decode().strip(),
        None if difference is None else difference.decode("utf-8", errors="replace").strip(),
        None if status is None else status.decode("utf-8", errors="replace").strip(),
    )


def _portable_identity_valid(
    value: Any,
    *,
    expected_blob: str | None,
    expected_sha256: str,
) -> bool:
    identity = _mapping(value)
    return (
        expected_blob is not None
        and identity.get("sha256_definition") == CANONICAL_GIT_BLOB_SHA256_DEFINITION
        and identity.get("expected_git_blob") == expected_blob
        and identity.get("head_git_blob") == expected_blob
        and identity.get("filtered_worktree_git_blob") == expected_blob
        and identity.get("worktree_status") == ""
        and identity.get("worktree_diff") == ""
        and re.fullmatch(r"[0-9a-f]{64}", str(identity.get("worktree_raw_sha256", ""))) is not None
        and identity.get("canonical_git_blob_sha256") == expected_sha256
        and _integer(identity.get("canonical_git_blob_size_bytes"), minimum=1)
        and identity.get("checks")
        == {
            "head_blob_matches": True,
            "filtered_worktree_blob_matches": True,
            "worktree_clean": True,
            "canonical_blob_read": True,
            "canonical_blob_uses_lf_text": True,
        }
        and identity.get("errors") == []
        and identity.get("passed") is True
    )


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _frozen_configuration_valid(config: dict[str, Any]) -> bool:
    reconstruction = _mapping(config.get("reconstruction_protocol"))
    algorithm = _mapping(config.get("algorithm"))
    acceptance = _mapping(config.get("acceptance"))
    timing = _mapping(config.get("timing"))
    boundary = _mapping(config.get("stage_boundary"))
    source = _mapping(config.get("public_network_source"))
    return (
        config.get("schema_version") == "1.0"
        and config.get("stage") == 7
        and config.get("classification") == "structural_reproduction"
        and config.get("precision") == "FP64"
        and reconstruction.get("frozen_before_benchmark_runs") is True
        and reconstruction.get("timing_tuning_prohibited") is True
        and reconstruction.get("seed") == 20260803
        and reconstruction.get("ptdf_zero_atol") == 1e-12
        and algorithm
        == {
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
        and acceptance
        == {
            "paper_residual_tolerance": PAPER_TOLERANCE,
            "raw_kkt_tolerance": RAW_KKT_TOLERANCE,
            "maximum_physical_violation": PHYSICAL_TOLERANCE,
            "maximum_scaled_objective_gap_to_highs": OBJECTIVE_GAP_TOLERANCE,
            "maximum_dimension_difference_for_execution": 0,
            "nonzero_nnz_difference_blocks_paper_time_comparability": True,
            "required_solver_tracks": list(REQUIRED_TRACKS),
            "gurobi_required_only_when_installed_and_licensed": True,
        }
        and timing
        == {
            "warmup_runs": 1,
            "measured_runs": MEASURED_RUNS,
            "maximum_measured_runs_after_variability_escalation": ESCALATED_RUNS,
            "relative_range_escalation_threshold": VARIABILITY_THRESHOLD,
            "per_solve_time_limit_seconds": SOLVE_TIME_LIMIT_SECONDS,
            "report_first_run_separately": True,
            "report_statistics": [
                "median",
                "minimum",
                "maximum",
                "standard_deviation",
                "interquartile_range",
            ],
            "speedup_requires_matching_boundaries": True,
        }
        and boundary
        == {
            "stage_7_only": True,
            "stage_8_large_runs_locked": True,
            "exact_paper_reproduction_claimed": False,
            "paper_a100_timing_reproduction_claimed": False,
            "n_minus_1_extension_enabled": False,
        }
        and source.get("release") == "8.1"
        and source.get("tag_object") == "3f8ecfdbc79b07697d6b45f8d868ac1c2d27f788"
        and source.get("resolved_commit") == "1a828c7af590714499284e36ee9c81273388c594"
        and source.get("release_doi") == "10.5281/zenodo.15871662"
        and source.get("sha256_definition") == CANONICAL_GIT_BLOB_SHA256_DEFINITION
    )


def _configuration_checks(
    checks: list[dict[str, Any]],
    config: dict[str, Any],
    evidence: dict[str, Any],
    config_path: Path,
) -> None:
    config_relative = config_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    environment = _mapping(evidence.get("environment"))
    commit = _mapping(environment.get("git")).get("head")
    commit_valid = isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None
    executed_blob = _git_blob_oid(str(commit), config_relative) if commit_valid else None
    actual_hash = _git_blob_sha256(str(commit), config_relative) if commit_valid else None
    filtered_blob, worktree_diff, worktree_status = _git_worktree_identity(config_relative)
    embedded = _mapping(evidence.get("configuration"))
    validation = _mapping(evidence.get("configuration_validation"))
    provenance_config = _mapping(_mapping(evidence.get("provenance")).get("config"))
    add_check(
        checks,
        "frozen_configuration_hash_content_and_thresholds",
        bool(config)
        and commit_valid
        and actual_hash == FROZEN_CONFIG_SHA256
        and provenance_config.get("sha256") == FROZEN_CONFIG_SHA256
        and provenance_config.get("sha256_matches_frozen") is True
        and provenance_config.get("sha256_definition") == CANONICAL_GIT_BLOB_SHA256_DEFINITION
        and provenance_config.get("passed") is True
        and _portable_identity_valid(
            provenance_config.get("portable_identity"),
            expected_blob=executed_blob,
            expected_sha256=FROZEN_CONFIG_SHA256,
        )
        and filtered_blob == executed_blob
        and worktree_diff == ""
        and worktree_status == ""
        and embedded == config
        and _frozen_configuration_valid(config)
        and validation == {"errors": [], "passed": True},
        (
            f"canonical_sha256={actual_hash}, expected={FROZEN_CONFIG_SHA256}, "
            f"commit_blob={executed_blob}, worktree_blob={filtered_blob}, "
            f"worktree_clean={worktree_diff == ''}"
        ),
    )

    policy = _mapping(evidence.get("policy_contract"))
    add_check(
        checks,
        "reconstruction_policy_contract",
        policy.get("json_is_sole_authority") is True
        and policy.get("policy_fingerprint") == POLICY_FINGERPRINT
        and policy.get("error") is None
        and policy.get("passed") is True,
        f"policy_fingerprint={policy.get('policy_fingerprint')}",
    )

    requirements_path = PROJECT_ROOT / "environment" / "dgx_stage7_requirements.txt"
    requirements_relative = requirements_path.relative_to(PROJECT_ROOT).as_posix()
    requirements = _mapping(evidence.get("requirements_freeze"))
    parsed_pins = _requirements_pins(requirements_path)
    requirements_blob = _git_blob_oid(str(commit), requirements_relative) if commit_valid else None
    requirements_sha = (
        _git_blob_sha256(str(commit), requirements_relative) if commit_valid else None
    )
    requirements_worktree_blob, requirements_diff, requirements_status = _git_worktree_identity(
        requirements_relative
    )
    critical_names = ("cupy-cuda13x", "numpy", "scipy")
    expected_pins = (
        {}
        if parsed_pins is None or any(name not in parsed_pins for name in critical_names)
        else {name: parsed_pins[name] for name in critical_names}
    )
    add_check(
        checks,
        "frozen_dgx_environment_requirements",
        requirements_sha == FROZEN_REQUIREMENTS_SHA256
        and requirements.get("path") == "environment/dgx_stage7_requirements.txt"
        and requirements.get("sha256") == FROZEN_REQUIREMENTS_SHA256
        and requirements.get("sha256_definition") == CANONICAL_GIT_BLOB_SHA256_DEFINITION
        and _portable_identity_valid(
            requirements.get("portable_identity"),
            expected_blob=requirements_blob,
            expected_sha256=FROZEN_REQUIREMENTS_SHA256,
        )
        and requirements_worktree_blob == requirements_blob
        and requirements_diff == ""
        and requirements_status == ""
        and requirements.get("expected_pins") == expected_pins
        and all(
            _mapping(requirements.get("pins")).get(key) == value
            for key, value in expected_pins.items()
        )
        and requirements.get("errors") == []
        and requirements.get("passed") is True,
        (
            f"canonical_sha256={requirements_sha}, commit_blob={requirements_blob}, "
            f"worktree_blob={requirements_worktree_blob}, "
            f"worktree_clean={requirements_diff == ''}"
        ),
    )

    environment = _mapping(evidence.get("environment"))
    packages = _mapping(environment.get("packages"))
    availability = _mapping(evidence.get("solver_availability"))
    gpu_availability = _mapping(availability.get("gpu_fp64_sgs_hpr"))
    python_version = environment.get("python")
    add_check(
        checks,
        "executed_python_and_numeric_packages_match_frozen_requirements",
        re.fullmatch(r"3\.12\.\d+", str(python_version)) is not None
        and len(expected_pins) == len(critical_names)
        and environment.get("numpy") == expected_pins.get("numpy")
        and environment.get("scipy") == expected_pins.get("scipy")
        and packages.get("numpy") == expected_pins.get("numpy")
        and packages.get("scipy") == expected_pins.get("scipy")
        and packages.get("cupy-cuda13x") == expected_pins.get("cupy-cuda13x")
        and gpu_availability.get("cupy_cuda13x_version") == expected_pins.get("cupy-cuda13x"),
        (
            f"python={python_version}, numpy={environment.get('numpy')}, "
            f"scipy={environment.get('scipy')}, "
            f"cupy-cuda13x={packages.get('cupy-cuda13x')}"
        ),
    )


def _provenance_checks(
    checks: list[dict[str, Any]], evidence: dict[str, Any], config: dict[str, Any]
) -> None:
    provenance = _mapping(evidence.get("provenance"))
    embedded_config = _mapping(provenance.get("config"))
    rows = [_mapping(row) for row in _sequence(provenance.get("files"))]
    rows_by_path = {str(row.get("path")): row for row in rows}
    upstream = _mapping(provenance.get("upstream"))
    valid = (
        len(rows) == len(rows_by_path) == 3
        and set(rows_by_path) == set(PROVENANCE)
        and embedded_config.get("sha256") == FROZEN_CONFIG_SHA256
        and upstream == _mapping(config.get("public_network_source"))
        and provenance.get("errors") == []
        and provenance.get("passed") is True
    )
    for path, (case_name, expected_sha, expected_blob) in PROVENANCE.items():
        row = rows_by_path.get(path, {})
        filtered_blob, worktree_diff, worktree_status = _git_worktree_identity(path)
        commit = _mapping(_mapping(evidence.get("environment")).get("git")).get("head")
        executed_blob = _git_blob_oid(str(commit), path) if isinstance(commit, str) else None
        valid = valid and (
            row.get("case_name") == case_name
            and row.get("inside_project") is True
            and row.get("exists") is True
            and row.get("expected_sha256") == expected_sha
            and row.get("actual_sha256") == expected_sha
            and row.get("sha256_matches") is True
            and row.get("expected_git_blob") == expected_blob
            and row.get("actual_git_blob") == expected_blob
            and row.get("git_blob_matches") is True
            and row.get("passed") is True
            and _portable_identity_valid(
                row.get("portable_identity"),
                expected_blob=expected_blob,
                expected_sha256=expected_sha,
            )
            and isinstance(commit, str)
            and executed_blob == expected_blob
            and _git_blob_sha256(commit, path) == expected_sha
            and filtered_blob == expected_blob
            and worktree_diff == ""
            and worktree_status == ""
        )
    add_check(
        checks,
        "pinned_matpower_provenance_hashes",
        valid,
        f"records={len(rows)}, expected=3",
    )


def _source_checks(checks: list[dict[str, Any]], evidence: dict[str, Any]) -> None:
    environment = _mapping(evidence.get("environment"))
    git = _mapping(environment.get("git"))
    commit = git.get("head")
    commit_valid = isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None
    manifest = [_mapping(row) for row in _sequence(evidence.get("source_manifest"))]
    manifest_map = {str(row.get("path")): row for row in manifest}
    tree_paths = _git_tree_paths(commit) if commit_valid else None
    expected_paths = (
        None
        if tree_paths is None
        else {
            "scripts/run_stage_7.py",
            "configs/benchmarks/stage_7_small_medium.json",
            "environment/dgx_stage7_requirements.txt",
            *tree_paths,
        }
    )
    valid = (
        commit_valid
        and expected_paths is not None
        and len(manifest) == len(manifest_map)
        and set(manifest_map) == expected_paths
    )
    if valid:
        for path, row in manifest_map.items():
            executed_blob = _git_blob_oid(str(commit), path)
            executed_sha = _git_blob_sha256(str(commit), path)
            filtered_blob, worktree_diff, worktree_status = _git_worktree_identity(path)
            valid = valid and (
                re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))) is not None
                and row.get("sha256") == executed_sha
                and row.get("git_blob") == executed_blob
                and row.get("sha256_definition") == CANONICAL_GIT_BLOB_SHA256_DEFINITION
                and row.get("passed") is True
                and filtered_blob == executed_blob
                and worktree_diff == ""
                and worktree_status == ""
            )
    add_check(
        checks,
        "source_manifest_matches_exact_clean_executed_git_commit",
        valid,
        (
            f"commit={commit}, execution_source_paths_clean={valid}, "
            f"source_count={len(manifest)}, repository_status={git.get('status_porcelain')!r}"
        ),
    )

    provenance = _mapping(evidence.get("provenance"))
    inputs = [
        {"path": row.get("path"), "sha256": row.get("actual_sha256")}
        for row in _sequence(provenance.get("files"))
        if isinstance(row, dict)
    ]
    packages = _mapping(environment.get("packages"))
    payload = {
        "config_sha256": FROZEN_CONFIG_SHA256,
        "inputs": inputs,
        "sources": manifest,
        "python": environment.get("python"),
        "numpy": environment.get("numpy"),
        "scipy": environment.get("scipy"),
        "cupy_cuda13x": packages.get("cupy-cuda13x"),
    }
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    add_check(
        checks,
        "run_fingerprint_recomputed",
        evidence.get("run_fingerprint") == recomputed
        and all(
            _mapping(case).get("run_fingerprint") == recomputed
            for case in _sequence(evidence.get("cases"))
        ),
        f"recorded={evidence.get('run_fingerprint')}, recomputed={recomputed}",
    )


def _ledger_checks(checks: list[dict[str, Any]], evidence: dict[str, Any]) -> None:
    rows = [_mapping(row) for row in _sequence(evidence.get("symbolic_ledger"))]
    indexed = {str(row.get("key")): row for row in rows}
    valid = len(rows) == len(indexed) == 18 and set(indexed) == set(EXPECTED_ROWS)
    for key, (m, n, paper_nnz, reconstructed_nnz, execute) in EXPECTED_ROWS.items():
        row = indexed.get(key, {})
        symbolic = _mapping(row.get("symbolic_nnz"))
        dimensions = _mapping(row.get("dimension_comparison"))
        m_comparison = _mapping(dimensions.get("m"))
        n_comparison = _mapping(dimensions.get("n"))
        nnz_comparison = _mapping(row.get("nnz_comparison"))
        difference = reconstructed_nnz - paper_nnz
        expected_csr32_supported = key not in CSR32_UNSUPPORTED_LOCKED_ROWS
        expected_cause = (
            "author renewable/storage placement and PTDF construction are unavailable; "
            "the frozen deterministic selected-bus PTDF support uses a 1e-12 zero "
            "threshold and was not tuned to timing or Table II nnz"
        )
        valid = valid and (
            row.get("computed_m") == m
            and row.get("computed_n") == n
            and row.get("published_m") == m
            and row.get("published_n") == n
            and row.get("published_nnz") == paper_nnz
            and row.get("actual_reconstruction_nnz") == reconstructed_nnz
            and row.get("nnz_difference_from_paper") == difference
            and row.get("execute_stage_7") is execute
            and row.get("allocation_stage") == ("stage_7" if execute else "stage_8_locked")
            and row.get("allocation_permitted_this_run") is execute
            and row.get("dimensions_match_table") is True
            and row.get("paper_values_match_config") is True
            and row.get("full_lp_allocated") is False
            and row.get("stage_8_large_allocation_locked") is True
            and row.get("csr32_supported") is expected_csr32_supported
            and row.get("paper_time_comparable") is False
            and row.get("paper_time_comparability_reason")
            == "reconstructed nnz differs; Table II timing is context only"
            and row.get("policy_fingerprint") == POLICY_FINGERPRINT
            and row.get("passed") is True
            and symbolic.get("case_name") == key.split(":", 1)[0]
            and symbolic.get("periods") == int(key.rsplit("T", 1)[1])
            and symbolic.get("published_nnz") == paper_nnz
            and symbolic.get("reconstructed_nnz") == reconstructed_nnz
            and symbolic.get("difference_from_paper") == difference
            and symbolic.get("matches_paper") is False
            and symbolic.get("count_kind")
            == "exact_reconstruction_selected_bus_batched_sparse_ptdf_support"
            and symbolic.get("ptdf_zero_atol") == 1e-12
            and symbolic.get("policy_fingerprint") == POLICY_FINGERPRINT
            and symbolic.get("full_lp_allocated") is False
            and symbolic.get("stage_8_large_allocation_locked") is True
            and m_comparison.get("paper") == m
            and m_comparison.get("reproduced") == m
            and m_comparison.get("absolute_difference") == 0
            and m_comparison.get("percentage_difference") == 0.0
            and isinstance(m_comparison.get("cause"), str)
            and n_comparison.get("paper") == n
            and n_comparison.get("reproduced") == n
            and n_comparison.get("absolute_difference") == 0
            and n_comparison.get("percentage_difference") == 0.0
            and isinstance(n_comparison.get("cause"), str)
            and nnz_comparison.get("paper") == paper_nnz
            and nnz_comparison.get("reproduced") == reconstructed_nnz
            and nnz_comparison.get("signed_difference") == difference
            and nnz_comparison.get("absolute_difference") == abs(difference)
            and _close(
                nnz_comparison.get("percentage_difference"),
                100.0 * abs(difference) / paper_nnz,
            )
            and nnz_comparison.get("cause") == expected_cause
        )
    add_check(
        checks,
        "all_eighteen_exact_dimension_and_reconstructed_nnz_rows",
        valid,
        f"rows={len(rows)}, unique={len(indexed)}, paper_comparable=0",
    )


def _memory_record_valid(value: Any) -> bool:
    record = _mapping(value)
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    return all(
        _nonnegative(_mapping(snapshot).get("rss_bytes"))
        and _nonnegative(_mapping(snapshot).get("cumulative_process_peak_bytes"))
        for snapshot in (before, after)
    )


def _physical_valid(value: Any) -> bool:
    physical = _mapping(value)
    values = [physical.get(name) for name in PHYSICAL_FIELDS]
    maximum = max((float(item) for item in values if _nonnegative(item)), default=math.inf)
    return (
        len(values) == len(PHYSICAL_FIELDS)
        and all(_nonnegative(item) for item in values)
        and _close(physical.get("maximum_violation"), maximum)
        and maximum <= PHYSICAL_TOLERANCE
        and physical.get("available") is True
        and physical.get("passed") is True
        and physical.get("tolerance") == PHYSICAL_TOLERANCE
        and physical.get("sparse_factorization_reused") is True
        and physical.get("batched_periods") is True
    )


def _state_fingerprint_valid(value: Any) -> bool:
    state = _mapping(value)
    if set(state) != {"x", "y", "z"}:
        return False
    return all(
        bool(_sequence(_mapping(state[name]).get("shape")))
        and re.fullmatch(r"[0-9a-f]{64}", str(_mapping(state[name]).get("sha256", ""))) is not None
        and _mapping(state[name]).get("finite") is True
        and _nonnegative(_mapping(state[name]).get("l2_norm"))
        for name in state
    )


def _candidate_valid(value: Any, *, reference_objective: float | None) -> bool:
    candidate = _mapping(value)
    residuals = _mapping(candidate.get("residuals"))
    normalized = _mapping(residuals.get("paper_normalized_norms"))
    raw = _mapping(residuals.get("paper_raw_norms"))
    stopping = _mapping(residuals.get("paper_stopping"))
    checks = _mapping(candidate.get("checks"))
    objective = candidate.get("objective")
    gap = candidate.get("scaled_objective_gap_to_highs")
    effective_reference = objective if reference_objective is None else reference_objective
    expected_gap = (
        math.inf
        if not _finite(objective) or not _finite(effective_reference)
        else abs(float(objective) - float(effective_reference))
        / max(1.0, abs(float(effective_reference)))
    )
    canonical = _mapping(candidate.get("canonical_violations"))
    canonical_values = [
        canonical.get("equality_inf"),
        canonical.get("inequality_positive_max"),
        canonical.get("lower_violation_max"),
        canonical.get("upper_violation_max"),
    ]
    expected_box = (
        max(float(canonical_values[2]), float(canonical_values[3]))
        if all(_nonnegative(item) for item in canonical_values)
        else math.inf
    )
    expected_overall = (
        max(float(canonical_values[0]), float(canonical_values[1]), expected_box)
        if math.isfinite(expected_box)
        else math.inf
    )
    return (
        _finite(objective)
        and _close(gap, expected_gap)
        and _close(candidate.get("scaled_objective_gap_to_reference"), expected_gap)
        and float(gap) <= OBJECTIVE_GAP_TOLERANCE
        and candidate.get("objective_reference") is (reference_objective is None)
        and _close(candidate.get("reference_objective"), effective_reference)
        and residuals.get("tolerance") == PAPER_TOLERANCE
        and set(normalized) == {"primal_feasibility", "box", "stationarity"}
        and set(raw) == {"primal_feasibility", "box", "stationarity"}
        and all(_nonnegative(item) for item in raw.values())
        and all(
            _nonnegative(item) and float(item) <= PAPER_TOLERANCE for item in normalized.values()
        )
        and _nonnegative(residuals.get("paper_normalized_combined_norm"))
        and _nonnegative(residuals.get("kkt_combined_norm"))
        and float(residuals["kkt_combined_norm"]) <= RAW_KKT_TOLERANCE
        and stopping
        == {
            "primal_feasibility": True,
            "box": True,
            "stationarity": True,
            "all_satisfied": True,
        }
        and all(_nonnegative(item) for item in canonical_values)
        and _close(canonical.get("box_violation_max"), expected_box)
        and _close(canonical.get("overall_max"), expected_overall)
        and expected_overall <= PHYSICAL_TOLERANCE
        and _close(candidate.get("maximum_canonical_primal_violation"), expected_overall)
        and _physical_valid(candidate.get("physical_validation"))
        and _state_fingerprint_valid(candidate.get("state_fingerprint"))
        and set(checks)
        == {
            "finite",
            "paper_stopping_satisfied",
            "raw_kkt_within_tolerance",
            "canonical_primal_within_tolerance",
            "physical_validation_available",
            "physical_validation_passed",
            "objective_gap_within_tolerance",
        }
        and all(item is True for item in checks.values())
        and candidate.get("passed") is True
    )


def _transfer_ledger_valid(value: Any, *, require_solver_phases: bool) -> bool:
    ledger = _mapping(value)
    records = [_mapping(row) for row in _sequence(ledger.get("records"))]
    totals = _mapping(ledger.get("totals"))
    if not records:
        return False
    unique: set[tuple[str, str, str]] = set()
    for row in records:
        key = (str(row.get("phase")), str(row.get("direction")), str(row.get("kind")))
        if (
            key in unique
            or key[1] not in {"host_to_device", "device_to_host"}
            or not _integer(row.get("calls"), minimum=1)
            or not _integer(row.get("bytes"))
            or (require_solver_phases and (key[0], key[1]) not in GPU_TRANSFER_PHASES)
        ):
            return False
        unique.add(key)
    for direction in ("host_to_device", "device_to_host"):
        matching = [row for row in records if row.get("direction") == direction]
        if _mapping(totals.get(direction)) != {
            "calls": sum(int(row["calls"]) for row in matching),
            "bytes": sum(int(row["bytes"]) for row in matching),
        }:
            return False
    return True


def _gpu_attempt_valid(value: Any) -> bool:
    row = _mapping(value)
    device_memory = _mapping(row.get("device_memory"))
    timing = _mapping(row.get("native_gpu_timing"))
    transfer_timing = _mapping(row.get("transfer_timing_delta"))
    audit = _mapping(row.get("transfer_audit"))
    return (
        all(
            _nonnegative(item)
            for snapshot in (
                _mapping(device_memory.get("before")),
                _mapping(device_memory.get("after")),
            )
            for item in snapshot.values()
        )
        and device_memory.get("true_per_solve_peak_available") is False
        and all(
            _nonnegative(timing.get(name))
            for name in (
                "loop_gpu_seconds",
                "residual_check_gpu_seconds",
                "iterations_excluding_residual_checks_gpu_seconds",
                "loop_wall_seconds",
            )
        )
        and timing.get("residual_check_interval") == 1
        and _integer(timing.get("residual_check_count"), minimum=1)
        and float(timing["residual_check_gpu_seconds"]) <= float(timing["loop_gpu_seconds"]) + 1e-12
        and _close(
            timing.get("iterations_excluding_residual_checks_gpu_seconds"),
            max(
                0.0, float(timing["loop_gpu_seconds"]) - float(timing["residual_check_gpu_seconds"])
            ),
            abs_tol=1e-9,
        )
        and _transfer_ledger_valid(row.get("transfer_delta"), require_solver_phases=True)
        and _transfer_ledger_valid(row.get("transfer_ledger"), require_solver_phases=False)
        and set(transfer_timing) == {"host_to_device_seconds", "device_to_host_seconds"}
        and all(_nonnegative(item) for item in transfer_timing.values())
        and audit.get("passed") is True
        and audit.get("unexpected_records") == []
        and audit.get("full_state_copied_inside_resident_loop") is False
    )


def _attempt_valid(
    value: Any,
    *,
    track_name: str,
    reference_objective: float | None,
    repetition: int | None,
) -> bool:
    row = _mapping(value)
    basic = (
        row.get("status") == "SUCCESS"
        and row.get("passed") is True
        and _nonnegative(row.get("wall_seconds"))
        and float(row["wall_seconds"]) <= SOLVE_TIME_LIMIT_SECONDS
        and _nonnegative(row.get("attempt_wall_seconds"))
        and (repetition is None or row.get("repetition") == repetition)
        and _memory_record_valid(row.get("process_memory"))
        and _candidate_valid(row.get("candidate"), reference_objective=reference_objective)
    )
    if not basic:
        return False
    if track_name == "highs":
        return _integer(row.get("iterations")) and row.get("failure") is None
    if track_name in {"cpu_fp64_sgs_hpr", "gpu_fp64_sgs_hpr"}:
        hpr_valid = (
            _integer(row.get("iterations"), minimum=1)
            and row.get("converged") is True
            and _integer(row.get("restart_count"))
            and _positive(row.get("sigma"))
        )
        if track_name == "cpu_fp64_sgs_hpr":
            native = _mapping(row.get("native_cpu_timing"))
            return hpr_valid and all(_nonnegative(item) for item in native.values())
        return hpr_valid and _gpu_attempt_valid(row)
    return False


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _statistics_valid(value: Any, samples: list[float]) -> bool:
    stats = _mapping(value)
    median = statistics.median(samples)
    minimum = min(samples)
    maximum = max(samples)
    relative_range = (maximum - minimum) / max(median, float.fromhex("0x1.0p-1022"))
    return (
        stats.get("count") == len(samples)
        and stats.get("raw_seconds") == samples
        and _close(stats.get("median_seconds"), median)
        and _close(stats.get("minimum_seconds"), minimum)
        and _close(stats.get("maximum_seconds"), maximum)
        and _close(stats.get("standard_deviation_seconds"), statistics.stdev(samples))
        and _close(
            stats.get("interquartile_range_seconds"),
            _percentile(samples, 0.75) - _percentile(samples, 0.25),
        )
        and _close(stats.get("relative_range"), relative_range)
    )


def _track_valid(
    value: Any,
    *,
    track_name: str,
    reference_objective: float | None,
) -> bool:
    track = _mapping(value)
    correctness = _mapping(track.get("correctness"))
    first_run = _mapping(track.get("first_run"))
    warmup = _sequence(track.get("warmup"))
    measured = _sequence(track.get("measured_repetitions"))
    if not (
        track.get("name") == track_name
        and track.get("passed") is True
        and track.get("timing_status") == "COMPLETE"
        and track.get("timing_boundary") == TIMING_BOUNDARIES[track_name]
        and correctness
        and first_run == correctness
        and isinstance(track.get("first_run_meaning"), str)
        and (
            "excluded" in str(track.get("first_run_meaning"))
            or "untimed-for-statistics" in str(track.get("first_run_meaning"))
        )
        and len(warmup) == 1
        and len(measured) in {MEASURED_RUNS, ESCALATED_RUNS}
        and _mapping(track.get("first_measured_run")) == _mapping(measured[0])
    ):
        return False
    first_five = [float(_mapping(row).get("wall_seconds", math.nan)) for row in measured[:5]]
    if not all(math.isfinite(item) and item >= 0.0 for item in first_five):
        return False
    initial_relative_range = (max(first_five) - min(first_five)) / max(
        statistics.median(first_five), float.fromhex("0x1.0p-1022")
    )
    escalated = initial_relative_range > VARIABILITY_THRESHOLD
    if track.get("variability_escalated") is not escalated:
        return False
    if len(measured) != (ESCALATED_RUNS if escalated else MEASURED_RUNS):
        return False
    attempts = [correctness, *warmup, *measured]
    repetitions: list[int | None] = [None, 1, *range(1, len(measured) + 1)]
    if not all(
        _attempt_valid(
            row,
            track_name=track_name,
            reference_objective=reference_objective,
            repetition=repetition,
        )
        for row, repetition in zip(attempts, repetitions, strict=True)
    ):
        return False
    samples = [float(_mapping(row)["wall_seconds"]) for row in measured]
    attempt_samples = [float(_mapping(row)["attempt_wall_seconds"]) for row in measured]
    return _statistics_valid(track.get("statistics"), samples) and _statistics_valid(
        track.get("attempt_wall_statistics"), attempt_samples
    )


def _gurobi_attempt_valid(value: Any, *, reference_objective: float) -> bool:
    row = _mapping(value)
    objective = row.get("objective")
    gap = row.get("scaled_objective_gap_to_highs")
    expected_gap = (
        abs(float(objective) - reference_objective) / max(1.0, abs(reference_objective))
        if _finite(objective)
        else math.inf
    )
    canonical = _mapping(row.get("canonical_violations"))
    canonical_values = [
        canonical.get("equality_inf"),
        canonical.get("inequality_positive_max"),
        canonical.get("lower_violation_max"),
        canonical.get("upper_violation_max"),
    ]
    expected_box = (
        max(float(canonical_values[2]), float(canonical_values[3]))
        if all(_nonnegative(item) for item in canonical_values)
        else math.inf
    )
    expected_overall = (
        max(float(canonical_values[0]), float(canonical_values[1]), expected_box)
        if math.isfinite(expected_box)
        else math.inf
    )
    return (
        row.get("status") == "SUCCESS"
        and row.get("passed") is True
        and _nonnegative(row.get("wall_seconds"))
        and float(row["wall_seconds"]) <= SOLVE_TIME_LIMIT_SECONDS
        and _nonnegative(row.get("attempt_wall_seconds"))
        and _memory_record_valid(row.get("process_memory"))
        and _finite(objective)
        and _close(gap, expected_gap)
        and _close(row.get("scaled_objective_gap_to_reference"), expected_gap)
        and float(gap) <= OBJECTIVE_GAP_TOLERANCE
        and row.get("objective_reference") is False
        and _close(row.get("reference_objective"), reference_objective)
        and all(_nonnegative(item) for item in canonical_values)
        and _close(canonical.get("box_violation_max"), expected_box)
        and _close(canonical.get("overall_max"), expected_overall)
        and expected_overall <= PHYSICAL_TOLERANCE
        and _close(row.get("maximum_canonical_primal_violation"), expected_overall)
        and _physical_valid(row.get("physical_validation"))
        and re.fullmatch(r"[0-9a-f]{64}", str(row.get("x_fingerprint", ""))) is not None
    )


def _gurobi_track_valid(value: Any, *, available: bool, reference_objective: float) -> bool:
    track = _mapping(value)
    if not available:
        return (
            track.get("name") == "gurobi"
            and track.get("status") == "NOT_REQUIRED_UNAVAILABLE_OR_UNLICENSED"
            and track.get("passed") is True
            and track.get("gating") is False
            and not _sequence(track.get("measured_repetitions"))
        )
    correctness = _mapping(track.get("correctness"))
    warmup = _sequence(track.get("warmup"))
    measured = _sequence(track.get("measured_repetitions"))
    if not (
        track.get("name") == "gurobi"
        and track.get("passed") is True
        and track.get("timing_status") == "COMPLETE"
        and track.get("timing_boundary") == TIMING_BOUNDARIES["gurobi"]
        and _mapping(track.get("first_run")) == correctness
        and len(warmup) == 1
        and len(measured) in {MEASURED_RUNS, ESCALATED_RUNS}
        and _mapping(track.get("first_measured_run")) == _mapping(measured[0])
    ):
        return False
    attempts = [correctness, *warmup, *measured]
    if not all(
        _gurobi_attempt_valid(row, reference_objective=reference_objective) for row in attempts
    ):
        return False
    samples = [float(_mapping(row)["wall_seconds"]) for row in measured]
    first_range = (max(samples[:5]) - min(samples[:5])) / max(
        statistics.median(samples[:5]), float.fromhex("0x1.0p-1022")
    )
    escalated = first_range > VARIABILITY_THRESHOLD
    return (
        track.get("variability_escalated") is escalated
        and len(measured) == (ESCALATED_RUNS if escalated else MEASURED_RUNS)
        and _statistics_valid(track.get("statistics"), samples)
        and _statistics_valid(
            track.get("attempt_wall_statistics"),
            [float(_mapping(row)["attempt_wall_seconds"]) for row in measured],
        )
    )


def _preprocessing_valid(
    value: Any, *, periods: int, storage: int, m1: int, m2: int, n: int
) -> bool:
    preprocessing = _mapping(value)
    scaled = _mapping(preprocessing.get("scaled_equality"))
    spectral = _mapping(preprocessing.get("sparse_spectral_certificate"))
    workspace = _mapping(preprocessing.get("cpu_workspace"))
    return (
        _nonnegative(preprocessing.get("wall_seconds"))
        and _nonnegative(preprocessing.get("cpu_workspace_setup_wall_seconds"))
        and scaled.get("periods") == periods
        and scaled.get("storage_count") == storage
        and scaled.get("equality_rows") == m1
        and scaled.get("dense_equality_gram_materialized") is False
        and scaled.get("dense_schur_shape") == [storage, storage]
        and all(_positive(item) for item in _sequence(scaled.get("balance_diagonal_range")))
        and all(_positive(item) for item in _sequence(scaled.get("storage_diagonal_range")))
        and all(_positive(item) for item in _sequence(scaled.get("schur_cholesky_diagonal_range")))
        and spectral.get("rows") == m2
        and spectral.get("columns") == n
        and spectral.get("power_seed") == 20260803
        and spectral.get("finite_certificate") is True
        and spectral.get("dense_matrix_materialized") is False
        and spectral.get("normal_matrix_materialized") is False
        and _positive(spectral.get("lambda_used"))
        and _close(spectral.get("inequality_lambda"), spectral.get("lambda_used"))
        and workspace.get("equality_backend") == "scaled_structural"
        and workspace.get("prepared_once_and_reused") is True
        and workspace.get("dense_equality_gram_materialized") is False
        and workspace.get("spectral_certificate_reused") is True
    )


def _gpu_memory_report_valid(value: Any) -> bool:
    report = _mapping(value)
    required = {
        "free_device_bytes",
        "total_device_bytes",
        "runtime_used_bytes",
        "device_pool_used_bytes",
        "device_pool_total_bytes",
        "device_pool_free_blocks",
        "pinned_pool_free_blocks",
    }
    return set(report) == required and all(_integer(report.get(name)) for name in required)


def _kernel_selection_valid(value: Any) -> bool:
    kernel = _mapping(value)
    return (
        "CUSPARSE_SPMV_CSR_ALG2" in str(kernel.get("requested_label", ""))
        and "CUSPARSE_SPMV_CSR_ALG2 (enum 3" in str(kernel.get("effective_label", ""))
        and kernel.get("uses_csr_alg2") is True
        and kernel.get("fallback_reason") is None
        and _nonnegative(kernel.get("probe_max_abs_error"))
        and float(kernel["probe_max_abs_error"]) <= 1e-12
        and kernel.get("probe_repeat_bitwise_equal") is True
    )


def _case_checks(
    checks: list[dict[str, Any]], evidence: dict[str, Any], config: dict[str, Any]
) -> None:
    boundary = _mapping(evidence.get("stage_boundary"))
    cases = [_mapping(case) for case in _sequence(evidence.get("cases"))]
    indexed = {str(case.get("key")): case for case in cases}
    boundary_valid = boundary == {
        "stage_7_only": True,
        "stage_7_executed": True,
        "stage_7_complete": True,
        "stage_8_large_runs_locked": True,
        "stage_8_allocation_count": 0,
        "n_minus_1_extension_enabled": False,
        "exact_paper_reproduction_claimed": False,
        "paper_a100_timing_reproduction_claimed": False,
    }
    add_check(
        checks,
        "exactly_six_stage7_cases_and_zero_stage8_allocations",
        len(cases) == len(indexed) == 6
        and set(indexed) == EXPECTED_CASES
        and boundary_valid
        and not any(key.startswith("case9241pegase:") for key in indexed),
        f"case_keys={sorted(indexed)}, boundary={boundary}",
    )

    availability = _mapping(evidence.get("solver_availability"))
    environment = _mapping(evidence.get("environment"))
    gurobi_availability = _mapping(availability.get("gurobi"))
    gurobi_available = gurobi_availability.get("available") is True
    availability_valid = (
        environment.get("machine") == "aarch64"
        and all(
            _mapping(availability.get(name)).get("available") is True for name in REQUIRED_TRACKS
        )
        and _mapping(availability.get("highs")).get("provider")
        == "scipy.optimize.linprog(method='highs-ds')"
        and _mapping(availability.get("gpu_fp64_sgs_hpr")).get("installed") is True
        and _mapping(availability.get("gpu_fp64_sgs_hpr")).get("adapter_available") is True
        and isinstance(
            _mapping(availability.get("gpu_fp64_sgs_hpr")).get("cupy_cuda13x_version"), str
        )
        and gurobi_available
        is bool(gurobi_availability.get("installed") and gurobi_availability.get("licensed"))
        and (gurobi_available or isinstance(gurobi_availability.get("reason"), str))
    )
    add_check(
        checks,
        "required_solver_availability_and_optional_gurobi_semantics",
        availability_valid,
        f"gurobi_available={gurobi_available}",
    )

    config_cases = {str(case["case"]): case for case in _sequence(config.get("cases"))}
    structural_valid = set(indexed) == EXPECTED_CASES
    tracks_valid = structural_valid
    timing_valid = structural_valid
    device_valid = structural_valid
    for key, case in indexed.items():
        case_name, period_text = key.split(":T", 1)
        periods = int(period_text)
        m, n, paper_nnz, reconstructed_nnz, _ = EXPECTED_ROWS[key]
        spec = _mapping(config_cases.get(case_name))
        storage = int(spec.get("storage", -1))
        m1 = periods + storage
        m2 = m - m1
        construction = _mapping(case.get("construction"))
        dimensions = _mapping(construction.get("dimensions"))
        reconciliation = _mapping(case.get("structural_reconciliation"))
        tracks = _mapping(case.get("solver_tracks"))
        highs = _mapping(tracks.get("highs"))
        reference = _mapping(_mapping(highs.get("correctness")).get("candidate")).get("objective")
        structural_valid = structural_valid and (
            case.get("case_name") == case_name
            and case.get("periods") == periods
            and case.get("status") == "PASS"
            and case.get("passed") is True
            and _mapping(case.get("preflight")).get("passed") is True
            and all(
                item is True
                for item in _mapping(_mapping(case.get("preflight")).get("checks")).values()
            )
            and _nonnegative(construction.get("wall_seconds"))
            and dimensions.get("periods") == periods
            and dimensions.get("n") == n
            and dimensions.get("m") == m
            and dimensions.get("m1") == m1
            and dimensions.get("m2") == m2
            and dimensions.get("nnz_A") == reconstructed_nnz
            and dimensions.get("nnz_A1") + dimensions.get("nnz_A2") == reconstructed_nnz
            and construction.get("policy_fingerprint") == POLICY_FINGERPRINT
            and construction.get("input_sha256")
            == next(value[1] for value in PROVENANCE.values() if value[0] == case_name)
            and construction.get("input_sha256_definition") == CANONICAL_GIT_BLOB_SHA256_DEFINITION
            and re.fullmatch(
                r"[0-9a-f]{64}", str(_mapping(construction.get("lp_fingerprint")).get("sha256", ""))
            )
            is not None
            and reconciliation
            == {
                "dimension_match": True,
                "published_nnz": paper_nnz,
                "actual_nnz": reconstructed_nnz,
                "nnz_difference": reconstructed_nnz - paper_nnz,
                "symbolic_reconstructed_nnz": reconstructed_nnz,
                "actual_matches_symbolic_nnz": True,
                "paper_time_comparable": False,
                "classification": "structural_reproduction_not_author_instance",
            }
            and _preprocessing_valid(
                case.get("preprocessing"),
                periods=periods,
                storage=storage,
                m1=m1,
                m2=m2,
                n=n,
            )
        )
        tracks_valid = tracks_valid and (
            set(tracks) == {*REQUIRED_TRACKS, "gurobi"}
            and _finite(reference)
            and _track_valid(highs, track_name="highs", reference_objective=None)
            and _track_valid(
                tracks.get("cpu_fp64_sgs_hpr"),
                track_name="cpu_fp64_sgs_hpr",
                reference_objective=float(reference) if _finite(reference) else math.nan,
            )
            and _track_valid(
                tracks.get("gpu_fp64_sgs_hpr"),
                track_name="gpu_fp64_sgs_hpr",
                reference_objective=float(reference) if _finite(reference) else math.nan,
            )
            and _gurobi_track_valid(
                tracks.get("gurobi"),
                available=gurobi_available,
                reference_objective=float(reference) if _finite(reference) else math.nan,
            )
        )
        gpu = _mapping(tracks.get("gpu_fp64_sgs_hpr"))
        device = _mapping(gpu.get("device"))
        gpu_preflight = _mapping(gpu.get("preflight"))
        kernels = _mapping(gpu.get("kernel_selection"))
        kernel_checks = _mapping(gpu.get("kernel_checks"))
        device_valid = device_valid and (
            isinstance(device.get("device_name"), str)
            and "NVIDIA" in device["device_name"].upper()
            and "GB10" in device["device_name"].upper()
            and device.get("fp64_itemsize_bytes") == 8
            and device.get("fp64_supported") is True
            and device.get("csr_index_bits") in {32, 64}
            and _integer(device.get("total_global_memory_bytes"), minimum=1)
            and gpu_preflight.get("passed") is True
            and all(item is True for item in _mapping(gpu_preflight.get("checks")).values())
            and _gpu_memory_report_valid(gpu.get("memory_before"))
            and _gpu_memory_report_valid(gpu.get("memory_after"))
            and _transfer_ledger_valid(
                gpu.get("preparation_transfer_delta"), require_solver_phases=False
            )
            and _transfer_ledger_valid(
                gpu.get("cumulative_transfer_ledger"), require_solver_phases=False
            )
            and set(kernels) == {"A1", "A2"}
            and all(_kernel_selection_valid(kernels.get(name)) for name in ("A1", "A2"))
            and kernel_checks
            == {
                "requested_algorithm": "CUSPARSE_SPMV_CSR_ALG2",
                "A1_uses_csr_alg2": True,
                "A2_uses_csr_alg2": True,
                "FP64": True,
                "scaled_structural_equality": True,
                "passed": True,
            }
        )
        case_timing = _mapping(case.get("timing_boundaries"))
        timing_valid = timing_valid and (
            all(
                _nonnegative(case_timing.get(name))
                for name in (
                    "model_construction_wall_seconds",
                    "preprocessing_wall_seconds",
                    "gpu_workspace_setup_wall_seconds",
                    "end_to_end_case_wall_seconds",
                )
            )
            and case_timing.get("solver_core_samples_are_stored_per_track") is True
            and case_timing.get("speedup_computed") is False
            and _close(
                case_timing.get("model_construction_wall_seconds"), construction.get("wall_seconds")
            )
            and _close(
                case_timing.get("preprocessing_wall_seconds"),
                _mapping(case.get("preprocessing")).get("wall_seconds"),
            )
            and _close(
                case_timing.get("gpu_workspace_setup_wall_seconds"),
                gpu.get("workspace_setup_wall_seconds"),
            )
            and _close(
                case_timing.get("end_to_end_case_wall_seconds"),
                case.get("end_to_end_case_wall_seconds"),
            )
        )
    add_check(
        checks,
        "executed_case_dimensions_nnz_and_sparse_preprocessing",
        structural_valid,
        f"validated_cases={len(indexed)}",
    )
    add_check(
        checks,
        "required_correctness_objective_residual_physical_and_timing_gates",
        tracks_valid,
        f"required_tracks={list(REQUIRED_TRACKS)}, cases={len(indexed)}",
    )
    add_check(
        checks,
        "fp64_nvidia_device_memory_and_transfer_evidence",
        device_valid,
        f"gpu_tracks={len(indexed)}",
    )
    add_check(
        checks,
        "first_run_warmup_core_and_end_to_end_timing_boundaries",
        timing_valid,
        f"case_timing_records={len(indexed)}",
    )


def _report_manifest_checks(checks: list[dict[str, Any]], evidence: dict[str, Any]) -> None:
    manifest = evidence.get("report_manifest")
    if manifest is None:
        add_check(checks, "report_manifest_if_present", True, "not present")
        return
    rows = (
        _sequence(_mapping(manifest).get("files"))
        if isinstance(manifest, dict)
        else _sequence(manifest)
    )
    valid = bool(rows)
    seen: set[str] = set()
    for raw in rows:
        row = _mapping(raw)
        relative = Path(str(row.get("path", "")))
        path = (PROJECT_ROOT / relative).resolve()
        inside = path.is_relative_to(PROJECT_ROOT.resolve())
        label = relative.as_posix()
        valid = valid and (
            label not in seen
            and inside
            and path.is_file()
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))) is not None
            and _sha256(path) == row.get("sha256")
        )
        seen.add(label)
    add_check(
        checks,
        "report_manifest_if_present",
        valid,
        f"files={len(rows)}",
    )


def run_checks(
    evidence_path: Path = DEFAULT_EVIDENCE,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    config, config_error = _load_json(config_path)
    evidence, evidence_error = _load_json(evidence_path)

    required = [
        config_path,
        evidence_path,
        PROJECT_ROOT / "scripts" / "run_stage_7.py",
        PROJECT_ROOT / "scripts" / "check_stage_7.py",
        PROJECT_ROOT / "environment" / "dgx_stage7_requirements.txt",
        *(PROJECT_ROOT / path for path in PROVENANCE),
    ]
    missing = [_display_path(path) for path in required if not path.is_file()]
    add_check(
        checks,
        "required_stage_seven_paths",
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
    add_check(
        checks,
        "stage_status_and_runner_aggregate",
        evidence.get("schema_version") == "1.0"
        and evidence.get("stage") == 7
        and evidence.get("status") == "PASS"
        and evidence.get("all_passed") is True
        and _sequence(evidence.get("failures")) == [],
        f"status={evidence.get('status')}, all_passed={evidence.get('all_passed')}",
    )

    _configuration_checks(checks, config, evidence, config_path)
    _provenance_checks(checks, evidence, config)
    _source_checks(checks, evidence)
    _ledger_checks(checks, evidence)
    _case_checks(checks, evidence, config)
    _report_manifest_checks(checks, evidence)

    totals: Counter[str] = Counter()
    for check in checks:
        totals["passed" if check["passed"] else "failed"] += 1
    return {
        "stage": 7,
        "passed": totals["failed"] == 0,
        "configuration": _display_path(config_path),
        "evidence": _display_path(evidence_path),
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
