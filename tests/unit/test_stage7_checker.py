from __future__ import annotations

import copy
import hashlib
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts import check_stage_7

CONFIG_BLOB = "b" * 40
REQUIREMENTS_BLOB = "c" * 40


def _portable_identity(blob: str, sha256: str) -> dict[str, Any]:
    return {
        "sha256_definition": check_stage_7.CANONICAL_GIT_BLOB_SHA256_DEFINITION,
        "expected_git_blob": blob,
        "head_git_blob": blob,
        "filtered_worktree_git_blob": blob,
        "worktree_status": "",
        "worktree_diff": "",
        "worktree_raw_sha256": sha256,
        "canonical_git_blob_sha256": sha256,
        "canonical_git_blob_size_bytes": 123,
        "checks": {
            "head_blob_matches": True,
            "filtered_worktree_blob_matches": True,
            "worktree_clean": True,
            "canonical_blob_read": True,
            "canonical_blob_uses_lf_text": True,
        },
        "errors": [],
        "passed": True,
    }


def _statistics(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)

    def percentile(quantile: float) -> float:
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    median = statistics.median(samples)
    minimum = min(samples)
    maximum = max(samples)
    return {
        "count": len(samples),
        "raw_seconds": samples,
        "median_seconds": median,
        "minimum_seconds": minimum,
        "maximum_seconds": maximum,
        "standard_deviation_seconds": statistics.stdev(samples),
        "interquartile_range_seconds": percentile(0.75) - percentile(0.25),
        "relative_range": (maximum - minimum) / median,
    }


def _memory_report() -> dict[str, int]:
    return {
        "free_device_bytes": 100,
        "total_device_bytes": 200,
        "runtime_used_bytes": 100,
        "device_pool_used_bytes": 10,
        "device_pool_total_bytes": 20,
        "device_pool_free_blocks": 1,
        "pinned_pool_free_blocks": 1,
    }


def _process_memory() -> dict[str, Any]:
    snapshot = {
        "rss_bytes": 100,
        "cumulative_process_peak_bytes": 200,
        "peak_scope": "test",
        "sources": ["test"],
    }
    return {"before": snapshot, "after": snapshot}


def _physical() -> dict[str, Any]:
    return {
        **{name: 0.0 for name in check_stage_7.PHYSICAL_FIELDS},
        "maximum_violation": 0.0,
        "sparse_factorization_reused": True,
        "batched_periods": True,
        "available": True,
        "tolerance": check_stage_7.PHYSICAL_TOLERANCE,
        "passed": True,
    }


def _canonical() -> dict[str, float]:
    return {
        "equality_inf": 0.0,
        "inequality_positive_max": 0.0,
        "lower_violation_max": 0.0,
        "upper_violation_max": 0.0,
        "box_violation_max": 0.0,
        "overall_max": 0.0,
    }


def _candidate(*, reference: bool) -> dict[str, Any]:
    return {
        "objective": 100.0,
        "scaled_objective_gap_to_highs": 0.0,
        "scaled_objective_gap_to_reference": 0.0,
        "objective_reference": reference,
        "reference_objective": 100.0,
        "residuals": {
            "kkt_combined_norm": 0.0,
            "paper_normalized_combined_norm": 0.0,
            "paper_raw_norms": {
                "primal_feasibility": 0.0,
                "box": 0.0,
                "stationarity": 0.0,
            },
            "paper_normalized_norms": {
                "primal_feasibility": 0.0,
                "box": 0.0,
                "stationarity": 0.0,
            },
            "paper_stopping": {
                "primal_feasibility": True,
                "box": True,
                "stationarity": True,
                "all_satisfied": True,
            },
            "tolerance": check_stage_7.PAPER_TOLERANCE,
        },
        "canonical_violations": _canonical(),
        "maximum_canonical_primal_violation": 0.0,
        "physical_validation": _physical(),
        "state_fingerprint": {
            name: {
                "shape": [1],
                "sha256": character * 64,
                "finite": True,
                "minimum": 0.0,
                "maximum": 0.0,
                "l2_norm": 0.0,
            }
            for name, character in (("x", "1"), ("y", "2"), ("z", "3"))
        },
        "checks": {
            "finite": True,
            "paper_stopping_satisfied": True,
            "raw_kkt_within_tolerance": True,
            "canonical_primal_within_tolerance": True,
            "physical_validation_available": True,
            "physical_validation_passed": True,
            "objective_gap_within_tolerance": True,
        },
        "passed": True,
    }


def _transfer_ledger() -> dict[str, Any]:
    records = [
        {
            "phase": "initial_state",
            "direction": "host_to_device",
            "kind": "vector",
            "calls": 3,
            "bytes": 24,
        },
        {
            "phase": "final_state",
            "direction": "device_to_host",
            "kind": "vector",
            "calls": 3,
            "bytes": 24,
        },
    ]
    return {
        "records": records,
        "totals": {
            "host_to_device": {"calls": 3, "bytes": 24},
            "device_to_host": {"calls": 3, "bytes": 24},
        },
    }


def _attempt(track: str, *, reference: bool, repetition: int | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": "SUCCESS",
        "wall_seconds": 0.1,
        "attempt_wall_seconds": 0.11,
        "process_memory": _process_memory(),
        "iterations": 1,
        "candidate": _candidate(reference=reference),
        "passed": True,
    }
    if repetition is not None:
        row["repetition"] = repetition
    if track == "cpu_fp64_sgs_hpr":
        row.update(
            {
                "converged": True,
                "restart_count": 0,
                "sigma": 1.0,
                "native_cpu_timing": {
                    "preparation_elapsed_seconds": 0.0,
                    "total_elapsed_seconds": 0.1,
                },
            }
        )
    elif track == "gpu_fp64_sgs_hpr":
        row.update(
            {
                "converged": True,
                "restart_count": 0,
                "sigma": 1.0,
                "native_gpu_timing": {
                    "loop_gpu_seconds": 0.08,
                    "residual_check_gpu_seconds": 0.01,
                    "iterations_excluding_residual_checks_gpu_seconds": 0.07,
                    "loop_wall_seconds": 0.09,
                    "residual_check_count": 1,
                    "residual_check_interval": 1,
                },
                "device_memory": {
                    "before": _memory_report(),
                    "after": _memory_report(),
                    "true_per_solve_peak_available": False,
                    "note": "test",
                },
                "transfer_delta": _transfer_ledger(),
                "transfer_ledger": _transfer_ledger(),
                "transfer_timing_delta": {
                    "host_to_device_seconds": 0.001,
                    "device_to_host_seconds": 0.001,
                },
                "transfer_audit": {
                    "allowed_phase_directions": [],
                    "unexpected_records": [],
                    "full_state_copied_inside_resident_loop": False,
                    "passed": True,
                },
            }
        )
    return row


def _track(name: str) -> dict[str, Any]:
    correctness = _attempt(name, reference=name == "highs")
    warmup = [_attempt(name, reference=name == "highs", repetition=1)]
    measured = [
        _attempt(name, reference=name == "highs", repetition=index) for index in range(1, 6)
    ]
    return {
        "name": name,
        "timing_boundary": check_stage_7.TIMING_BOUNDARIES[name],
        "correctness": correctness,
        "first_run": copy.deepcopy(correctness),
        "first_run_meaning": (
            "untimed-for-statistics correctness run executed before warm-up and measurement"
        ),
        "warmup": warmup,
        "measured_repetitions": measured,
        "first_measured_run": copy.deepcopy(measured[0]),
        "statistics": _statistics([0.1] * 5),
        "attempt_wall_statistics": _statistics([0.11] * 5),
        "variability_escalated": False,
        "timing_status": "COMPLETE",
        "passed": True,
    }


def _kernel() -> dict[str, Any]:
    return {
        "requested_label": "cuSPARSE CUSPARSE_SPMV_CSR_ALG2",
        "effective_label": "cupy.cuda.cusparse.spMV CUSPARSE_SPMV_CSR_ALG2 (enum 3)",
        "uses_csr_alg2": True,
        "fallback_reason": None,
        "probe_max_abs_error": 0.0,
        "probe_repeat_bitwise_equal": True,
    }


def _case(key: str, config: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    case_name, period_text = key.split(":T", 1)
    periods = int(period_text)
    m, n, paper_nnz, reconstructed_nnz, _ = check_stage_7.EXPECTED_ROWS[key]
    spec = next(row for row in config["cases"] if row["case"] == case_name)
    m1 = periods + int(spec["storage"])
    m2 = m - m1
    gpu = _track("gpu_fp64_sgs_hpr")
    gpu.update(
        {
            "device": {
                "device_name": "NVIDIA GB10",
                "fp64_itemsize_bytes": 8,
                "fp64_supported": True,
                "csr_index_bits": 32,
                "total_global_memory_bytes": 128_000_000_000,
            },
            "preflight": {"passed": True, "checks": {"device_available": True}},
            "memory_before": _memory_report(),
            "memory_after": _memory_report(),
            "preparation_transfer_delta": _transfer_ledger(),
            "cumulative_transfer_ledger": _transfer_ledger(),
            "kernel_selection": {"A1": _kernel(), "A2": _kernel()},
            "kernel_checks": {
                "requested_algorithm": "CUSPARSE_SPMV_CSR_ALG2",
                "A1_uses_csr_alg2": True,
                "A2_uses_csr_alg2": True,
                "FP64": True,
                "scaled_structural_equality": True,
                "passed": True,
            },
            "workspace_setup_wall_seconds": 0.01,
        }
    )
    input_sha = next(
        value[1] for value in check_stage_7.PROVENANCE.values() if value[0] == case_name
    )
    return {
        "key": key,
        "case_name": case_name,
        "periods": periods,
        "run_fingerprint": fingerprint,
        "status": "PASS",
        "passed": True,
        "preflight": {"passed": True, "checks": {"dimensions_match_table": True}},
        "construction": {
            "wall_seconds": 0.2,
            "dimensions": {
                "periods": periods,
                "n": n,
                "m": m,
                "m1": m1,
                "m2": m2,
                "nnz_A": reconstructed_nnz,
                "nnz_A1": m1,
                "nnz_A2": reconstructed_nnz - m1,
            },
            "lp_fingerprint": {"sha256": "4" * 64, "blocks": {}},
            "policy_fingerprint": check_stage_7.POLICY_FINGERPRINT,
            "input_sha256": input_sha,
            "input_sha256_definition": (check_stage_7.CANONICAL_GIT_BLOB_SHA256_DEFINITION),
        },
        "structural_reconciliation": {
            "dimension_match": True,
            "published_nnz": paper_nnz,
            "actual_nnz": reconstructed_nnz,
            "nnz_difference": reconstructed_nnz - paper_nnz,
            "symbolic_reconstructed_nnz": reconstructed_nnz,
            "actual_matches_symbolic_nnz": True,
            "paper_time_comparable": False,
            "classification": "structural_reproduction_not_author_instance",
        },
        "preprocessing": {
            "wall_seconds": 0.3,
            "cpu_workspace_setup_wall_seconds": 0.01,
            "preconditioner": {},
            "scaled_equality": {
                "periods": periods,
                "storage_count": spec["storage"],
                "equality_rows": m1,
                "balance_diagonal_range": [1.0, 2.0],
                "storage_diagonal_range": [1.0, 2.0],
                "schur_cholesky_diagonal_range": [1.0, 2.0],
                "dense_equality_gram_materialized": False,
                "dense_schur_shape": [spec["storage"], spec["storage"]],
            },
            "sparse_spectral_certificate": {
                "rows": m2,
                "columns": n,
                "power_seed": 20260803,
                "finite_certificate": True,
                "dense_matrix_materialized": False,
                "normal_matrix_materialized": False,
                "lambda_used": 2.0,
                "inequality_lambda": 2.0,
            },
            "cpu_workspace": {
                "equality_backend": "scaled_structural",
                "prepared_once_and_reused": True,
                "dense_equality_gram_materialized": False,
                "spectral_certificate_reused": True,
            },
        },
        "solver_tracks": {
            "highs": _track("highs"),
            "cpu_fp64_sgs_hpr": _track("cpu_fp64_sgs_hpr"),
            "gpu_fp64_sgs_hpr": gpu,
            "gurobi": {
                "name": "gurobi",
                "timing_boundary": check_stage_7.TIMING_BOUNDARIES["gurobi"],
                "status": "NOT_REQUIRED_UNAVAILABLE_OR_UNLICENSED",
                "availability": {
                    "installed": False,
                    "licensed": False,
                    "available": False,
                    "reason": "not installed",
                },
                "passed": True,
                "gating": False,
            },
        },
        "end_to_end_case_wall_seconds": 1.0,
        "timing_boundaries": {
            "model_construction_wall_seconds": 0.2,
            "preprocessing_wall_seconds": 0.3,
            "gpu_workspace_setup_wall_seconds": 0.01,
            "end_to_end_case_wall_seconds": 1.0,
            "solver_core_samples_are_stored_per_track": True,
            "speedup_computed": False,
        },
    }


def _ledger() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, (m, n, paper_nnz, reconstructed_nnz, execute) in check_stage_7.EXPECTED_ROWS.items():
        case_name, period_text = key.split(":T", 1)
        periods = int(period_text)
        difference = reconstructed_nnz - paper_nnz
        rows.append(
            {
                "key": key,
                "case_name": case_name,
                "periods": periods,
                "computed_m": m,
                "computed_n": n,
                "published_m": m,
                "published_n": n,
                "published_nnz": paper_nnz,
                "actual_reconstruction_nnz": reconstructed_nnz,
                "nnz_difference_from_paper": difference,
                "execute_stage_7": execute,
                "allocation_stage": "stage_7" if execute else "stage_8_locked",
                "allocation_permitted_this_run": execute,
                "dimensions_match_table": True,
                "paper_values_match_config": True,
                "full_lp_allocated": False,
                "stage_8_large_allocation_locked": True,
                "csr32_supported": key not in check_stage_7.CSR32_UNSUPPORTED_LOCKED_ROWS,
                "paper_time_comparable": False,
                "paper_time_comparability_reason": (
                    "reconstructed nnz differs; Table II timing is context only"
                ),
                "policy_fingerprint": check_stage_7.POLICY_FINGERPRINT,
                "passed": True,
                "symbolic_nnz": {
                    "case_name": case_name,
                    "periods": periods,
                    "published_nnz": paper_nnz,
                    "reconstructed_nnz": reconstructed_nnz,
                    "difference_from_paper": difference,
                    "matches_paper": False,
                    "count_kind": ("exact_reconstruction_selected_bus_batched_sparse_ptdf_support"),
                    "ptdf_zero_atol": 1e-12,
                    "policy_fingerprint": check_stage_7.POLICY_FINGERPRINT,
                    "full_lp_allocated": False,
                    "stage_8_large_allocation_locked": True,
                },
                "dimension_comparison": {
                    name: {
                        "paper": value,
                        "reproduced": value,
                        "absolute_difference": 0,
                        "percentage_difference": 0.0,
                        "cause": "paper formula",
                    }
                    for name, value in (("m", m), ("n", n))
                },
                "nnz_comparison": {
                    "paper": paper_nnz,
                    "reproduced": reconstructed_nnz,
                    "signed_difference": difference,
                    "absolute_difference": abs(difference),
                    "percentage_difference": 100.0 * abs(difference) / paper_nnz,
                    "cause": (
                        "author renewable/storage placement and PTDF construction are unavailable; "
                        "the frozen deterministic selected-bus PTDF support uses a 1e-12 zero "
                        "threshold and was not tuned to timing or Table II nnz"
                    ),
                },
            }
        )
    return rows


def _valid_evidence() -> tuple[dict[str, Any], dict[str, str]]:
    config = json.loads(check_stage_7.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    source_paths = [
        "scripts/run_stage_7.py",
        "configs/benchmarks/stage_7_small_medium.json",
        "environment/dgx_stage7_requirements.txt",
        *sorted(
            path.relative_to(check_stage_7.PROJECT_ROOT).as_posix()
            for path in (check_stage_7.PROJECT_ROOT / "src" / "gpu_dcopf_hpr").glob("*.py")
        ),
    ]
    git_hashes = {
        path: str(check_stage_7._sha256(check_stage_7.PROJECT_ROOT / path)) for path in source_paths
    }
    git_hashes["configs/benchmarks/stage_7_small_medium.json"] = check_stage_7.FROZEN_CONFIG_SHA256
    git_hashes["environment/dgx_stage7_requirements.txt"] = check_stage_7.FROZEN_REQUIREMENTS_SHA256
    source_manifest = [
        {
            "path": path,
            "git_blob": (
                CONFIG_BLOB
                if path == "configs/benchmarks/stage_7_small_medium.json"
                else (
                    REQUIREMENTS_BLOB
                    if path == "environment/dgx_stage7_requirements.txt"
                    else "d" * 40
                )
            ),
            "sha256": git_hashes[path],
            "sha256_definition": check_stage_7.CANONICAL_GIT_BLOB_SHA256_DEFINITION,
            "passed": True,
        }
        for path in source_paths
    ]
    provenance_rows = []
    for path, (case_name, sha256, blob) in check_stage_7.PROVENANCE.items():
        provenance_rows.append(
            {
                "case_name": case_name,
                "path": path,
                "inside_project": True,
                "exists": True,
                "expected_sha256": sha256,
                "actual_sha256": sha256,
                "sha256_matches": True,
                "expected_git_blob": blob,
                "actual_git_blob": blob,
                "git_blob_matches": True,
                "portable_identity": _portable_identity(blob, sha256),
                "passed": True,
            }
        )
    environment = {
        "python": "3.12.0",
        "machine": "aarch64",
        "numpy": "2.3.5",
        "scipy": "1.16.3",
        "packages": {
            "cupy-cuda13x": "14.1.1",
            "numpy": "2.3.5",
            "scipy": "1.16.3",
        },
        "git": {"head": "a" * 40, "branch": "", "status_porcelain": ""},
    }
    fingerprint_payload = {
        "config_sha256": check_stage_7.FROZEN_CONFIG_SHA256,
        "inputs": [
            {"path": row["path"], "sha256": row["actual_sha256"]} for row in provenance_rows
        ],
        "sources": source_manifest,
        "python": environment["python"],
        "numpy": environment["numpy"],
        "scipy": environment["scipy"],
        "cupy_cuda13x": environment["packages"]["cupy-cuda13x"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evidence = {
        "schema_version": "1.0",
        "stage": 7,
        "status": "PASS",
        "all_passed": True,
        "run_fingerprint": fingerprint,
        "configuration": config,
        "configuration_validation": {"errors": [], "passed": True},
        "policy_contract": {
            "json_is_sole_authority": True,
            "policy_fingerprint": check_stage_7.POLICY_FINGERPRINT,
            "error": None,
            "passed": True,
        },
        "requirements_freeze": {
            "path": "environment/dgx_stage7_requirements.txt",
            "sha256": check_stage_7.FROZEN_REQUIREMENTS_SHA256,
            "sha256_definition": check_stage_7.CANONICAL_GIT_BLOB_SHA256_DEFINITION,
            "portable_identity": _portable_identity(
                REQUIREMENTS_BLOB,
                check_stage_7.FROZEN_REQUIREMENTS_SHA256,
            ),
            "pins": {
                "cupy-cuda13x": "14.1.1",
                "numpy": "2.3.5",
                "scipy": "1.16.3",
            },
            "expected_pins": {
                "cupy-cuda13x": "14.1.1",
                "numpy": "2.3.5",
                "scipy": "1.16.3",
            },
            "errors": [],
            "passed": True,
        },
        "provenance": {
            "config": {
                "path": str(check_stage_7.DEFAULT_CONFIG),
                "sha256": check_stage_7.FROZEN_CONFIG_SHA256,
                "sha256_matches_frozen": True,
                "sha256_definition": check_stage_7.CANONICAL_GIT_BLOB_SHA256_DEFINITION,
                "portable_identity": _portable_identity(
                    CONFIG_BLOB,
                    check_stage_7.FROZEN_CONFIG_SHA256,
                ),
                "passed": True,
            },
            "upstream": config["public_network_source"],
            "files": provenance_rows,
            "errors": [],
            "passed": True,
        },
        "environment": environment,
        "source_manifest": source_manifest,
        "solver_availability": {
            "highs": {
                "installed": True,
                "available": True,
                "provider": "scipy.optimize.linprog(method='highs-ds')",
            },
            "cpu_fp64_sgs_hpr": {"installed": True, "available": True},
            "gpu_fp64_sgs_hpr": {
                "installed": True,
                "adapter_available": True,
                "available": True,
                "cupy_cuda13x_version": "14.1.1",
            },
            "gurobi": {
                "installed": False,
                "licensed": False,
                "available": False,
                "reason": "not installed",
            },
        },
        "symbolic_ledger": _ledger(),
        "cases": [_case(key, config, fingerprint) for key in sorted(check_stage_7.EXPECTED_CASES)],
        "failures": [],
        "stage_boundary": {
            "stage_7_only": True,
            "stage_7_executed": True,
            "stage_7_complete": True,
            "stage_8_large_runs_locked": True,
            "stage_8_allocation_count": 0,
            "n_minus_1_extension_enabled": False,
            "exact_paper_reproduction_claimed": False,
            "paper_a100_timing_reproduction_claimed": False,
        },
    }
    return evidence, git_hashes


def _checks_by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): row for row in result["checks"]}


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: dict[str, Any],
    git_hashes: dict[str, str],
    *,
    worktree_overrides: dict[str, tuple[str | None, str | None, str | None]] | None = None,
) -> dict[str, Any]:
    evidence_path = tmp_path / "stage_7_validation.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    package_paths = [path for path in git_hashes if path.startswith("src/gpu_dcopf_hpr/")]
    monkeypatch.setattr(check_stage_7, "_git_tree_paths", lambda _commit: package_paths)
    monkeypatch.setattr(
        check_stage_7,
        "_git_blob_sha256",
        lambda _commit, path: (
            git_hashes.get(path)
            or next(
                (value[1] for key, value in check_stage_7.PROVENANCE.items() if key == path),
                None,
            )
        ),
    )
    blob_by_path = {
        "configs/benchmarks/stage_7_small_medium.json": CONFIG_BLOB,
        "environment/dgx_stage7_requirements.txt": REQUIREMENTS_BLOB,
        **{path: value[2] for path, value in check_stage_7.PROVENANCE.items()},
        **{str(row["path"]): str(row["git_blob"]) for row in evidence["source_manifest"]},
    }
    overrides = worktree_overrides or {}
    monkeypatch.setattr(
        check_stage_7,
        "_git_blob_oid",
        lambda _commit, path: blob_by_path.get(path),
    )
    monkeypatch.setattr(
        check_stage_7,
        "_git_worktree_identity",
        lambda path: overrides.get(path, (blob_by_path.get(path), "", "")),
    )
    return check_stage_7.run_checks(evidence_path, check_stage_7.DEFAULT_CONFIG)


def test_checker_accepts_complete_recomputed_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, git_hashes = _valid_evidence()

    result = _run(tmp_path, monkeypatch, evidence, git_hashes)

    assert result["passed"] is True
    assert result["summary"].get("failed", 0) == 0


def test_checker_rejects_wrong_and_dirty_matpower_worktree_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, git_hashes = _valid_evidence()
    path = next(iter(check_stage_7.PROVENANCE))

    result = _run(
        tmp_path,
        monkeypatch,
        evidence,
        git_hashes,
        worktree_overrides={path: ("0" * 40, path, f" M {path}")},
    )

    assert result["passed"] is False
    assert _checks_by_name(result)["pinned_matpower_provenance_hashes"]["passed"] is False


def test_checker_rejects_staged_only_input_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, git_hashes = _valid_evidence()
    path, (_, _, expected_blob) = next(iter(check_stage_7.PROVENANCE.items()))

    result = _run(
        tmp_path,
        monkeypatch,
        evidence,
        git_hashes,
        worktree_overrides={path: (expected_blob, "", f"M  {path}")},
    )

    assert result["passed"] is False
    assert _checks_by_name(result)["pinned_matpower_provenance_hashes"]["passed"] is False


def test_checker_allows_untracked_result_artifacts_when_sources_are_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, git_hashes = _valid_evidence()
    evidence["environment"]["git"]["status_porcelain"] = (
        "?? results/raw/stage_7/stage_7_validation.partial.json"
    )

    result = _run(tmp_path, monkeypatch, evidence, git_hashes)

    assert result["passed"] is True
    assert (
        _checks_by_name(result)["source_manifest_matches_exact_clean_executed_git_commit"]["passed"]
        is True
    )


Mutation = Callable[[dict[str, Any]], None]


def _tamper_residual(evidence: dict[str, Any]) -> None:
    candidate = evidence["cases"][0]["solver_tracks"]["cpu_fp64_sgs_hpr"]["correctness"][
        "candidate"
    ]
    candidate["residuals"]["kkt_combined_norm"] = 1.0


def _drop_required_track(evidence: dict[str, Any]) -> None:
    del evidence["cases"][0]["solver_tracks"]["gpu_fp64_sgs_hpr"]


def _change_embedded_config(evidence: dict[str, Any]) -> None:
    evidence["configuration"]["reconstruction_protocol"]["seed"] = 1


def _tamper_source(evidence: dict[str, Any]) -> None:
    evidence["source_manifest"][0]["sha256"] = "0" * 64


def _allocate_stage8(evidence: dict[str, Any]) -> None:
    extra = copy.deepcopy(evidence["cases"][0])
    extra["key"] = "case9241pegase:T4"
    extra["case_name"] = "case9241pegase"
    extra["periods"] = 4
    evidence["cases"].append(extra)
    evidence["stage_boundary"]["stage_8_allocation_count"] = 1


def _alter_statistics(evidence: dict[str, Any]) -> None:
    evidence["cases"][0]["solver_tracks"]["highs"]["statistics"]["median_seconds"] = 99.0


def _relax_threshold(evidence: dict[str, Any]) -> None:
    evidence["configuration"]["acceptance"]["raw_kkt_tolerance"] = 1.0


def _tamper_provenance(evidence: dict[str, Any]) -> None:
    evidence["provenance"]["files"][0]["actual_sha256"] = "f" * 64


def _mark_executed_row_csr32_unsupported(evidence: dict[str, Any]) -> None:
    row = next(row for row in evidence["symbolic_ledger"] if row["key"] == "case1354pegase:T96")
    row["csr32_supported"] = False


def _mark_locked_oversize_row_csr32_supported(evidence: dict[str, Any]) -> None:
    row = next(row for row in evidence["symbolic_ledger"] if row["key"] == "case9241pegase:T24")
    row["csr32_supported"] = True


def _refresh_run_fingerprint(evidence: dict[str, Any]) -> None:
    environment = evidence["environment"]
    payload = {
        "config_sha256": check_stage_7.FROZEN_CONFIG_SHA256,
        "inputs": [
            {"path": row["path"], "sha256": row["actual_sha256"]}
            for row in evidence["provenance"]["files"]
        ],
        "sources": evidence["source_manifest"],
        "python": environment["python"],
        "numpy": environment["numpy"],
        "scipy": environment["scipy"],
        "cupy_cuda13x": environment["packages"]["cupy-cuda13x"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evidence["run_fingerprint"] = fingerprint
    for case in evidence["cases"]:
        case["run_fingerprint"] = fingerprint


def _drift_python(evidence: dict[str, Any]) -> None:
    evidence["environment"]["python"] = "3.13.0"
    _refresh_run_fingerprint(evidence)


def _drift_numpy(evidence: dict[str, Any]) -> None:
    evidence["environment"]["numpy"] = "2.4.0"
    evidence["environment"]["packages"]["numpy"] = "2.4.0"
    _refresh_run_fingerprint(evidence)


def _drift_scipy(evidence: dict[str, Any]) -> None:
    evidence["environment"]["scipy"] = "1.17.0"
    evidence["environment"]["packages"]["scipy"] = "1.17.0"
    _refresh_run_fingerprint(evidence)


def _drift_cupy(evidence: dict[str, Any]) -> None:
    evidence["environment"]["packages"]["cupy-cuda13x"] = "14.2.0"
    evidence["solver_availability"]["gpu_fp64_sgs_hpr"]["cupy_cuda13x_version"] = "14.2.0"
    _refresh_run_fingerprint(evidence)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (_tamper_residual, "required_correctness_objective_residual_physical_and_timing_gates"),
        (_drop_required_track, "required_correctness_objective_residual_physical_and_timing_gates"),
        (_change_embedded_config, "frozen_configuration_hash_content_and_thresholds"),
        (_tamper_source, "source_manifest_matches_exact_clean_executed_git_commit"),
        (_allocate_stage8, "exactly_six_stage7_cases_and_zero_stage8_allocations"),
        (_alter_statistics, "required_correctness_objective_residual_physical_and_timing_gates"),
        (_relax_threshold, "frozen_configuration_hash_content_and_thresholds"),
        (_tamper_provenance, "pinned_matpower_provenance_hashes"),
        (
            _mark_executed_row_csr32_unsupported,
            "all_eighteen_exact_dimension_and_reconstructed_nnz_rows",
        ),
        (
            _mark_locked_oversize_row_csr32_supported,
            "all_eighteen_exact_dimension_and_reconstructed_nnz_rows",
        ),
        (
            _drift_python,
            "executed_python_and_numeric_packages_match_frozen_requirements",
        ),
        (
            _drift_numpy,
            "executed_python_and_numeric_packages_match_frozen_requirements",
        ),
        (
            _drift_scipy,
            "executed_python_and_numeric_packages_match_frozen_requirements",
        ),
        (
            _drift_cupy,
            "executed_python_and_numeric_packages_match_frozen_requirements",
        ),
    ],
)
def test_checker_fails_closed_on_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Mutation,
    failed_check: str,
) -> None:
    evidence, git_hashes = _valid_evidence()
    mutation(evidence)

    result = _run(tmp_path, monkeypatch, evidence, git_hashes)

    assert result["passed"] is False
    assert _checks_by_name(result)[failed_check]["passed"] is False
