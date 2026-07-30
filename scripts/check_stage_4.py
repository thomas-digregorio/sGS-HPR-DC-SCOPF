"""Independently validate preserved Stage 4 evidence and later-stage boundaries."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sgs_hpr" / "stage_4_structural.json"
DEFAULT_EVIDENCE = PROJECT_ROOT / "results" / "raw" / "stage_4" / "stage_4_validation.json"

EXPECTED_STRUCTURAL_CASES = {
    "no_storage_t1",
    "no_storage_t17",
    "ideal_storage_t1",
    "one_storage_t2",
    "extreme_efficiency_t32",
    "heterogeneous_storage_t5",
    "many_ideal_storage_t16",
}
STRESS_STRUCTURAL_CASES = {
    "extreme_efficiency_t32",
    "heterogeneous_storage_t5",
    "many_ideal_storage_t16",
}
EXPECTED_DCOPF_CASES = {
    "case5_base_t1",
    "case5_synthetic_extension_t2",
}


def _display_path(path: Path) -> str:
    """Render repository evidence paths without leaking a local checkout prefix."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"missing {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
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
                row = json.loads(line)
                if not isinstance(row, dict):
                    return rows, f"line {line_number} is not a JSON object"
                rows.append(row)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        return rows, f"{type(exc).__name__}: {exc}"
    return rows, None


def _all_checks_true(values: Any) -> bool:
    return (
        isinstance(values, dict)
        and bool(values)
        and all(value is True for value in values.values())
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _matrix_free_summary(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    storage = summary.get("storage_count")
    return (
        summary.get("dense_gram_materialized") is False
        and summary.get("explicit_kronecker_materialized") is False
        and isinstance(storage, int)
        and summary.get("stored_float_count") == 3 * storage
        and summary.get("solve_complexity") == "O(T + N_ESS)"
    )


def _structural_case_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    section = evidence.get("structural_crosschecks", {})
    cases = section.get("cases", []) if isinstance(section, dict) else []
    by_name = {
        str(case.get("name")): case
        for case in cases
        if isinstance(case, dict) and case.get("name") is not None
    }
    standard_tolerance = float(config.get("structural_maximum_normalized_error", -1.0))
    stress_tolerance = float(
        config.get(
            "structural_stress_maximum_normalized_error",
            standard_tolerance,
        )
    )
    coverage = set(by_name) == EXPECTED_STRUCTURAL_CASES
    add_check(
        checks,
        "seven_structural_cases_present",
        coverage,
        f"cases={sorted(by_name)}",
    )

    required_metrics = (
        "relative_solution_error",
        "normalized_system_residual",
        "scaled_component_error",
    )
    cases_valid = coverage and section.get("passed") is True
    details: list[str] = []
    for name, case in by_name.items():
        expected_tolerance = (
            stress_tolerance if name in STRESS_STRUCTURAL_CASES else standard_tolerance
        )
        maxima = case.get("maximum_corrected_errors", {})
        structure = case.get("structure", {})
        case_valid = (
            case.get("passed") is True
            and case.get("stress_case") is (name in STRESS_STRUCTURAL_CASES)
            and case.get("acceptance_tolerance") == expected_tolerance
            and _all_checks_true(case.get("checks"))
            and structure.get("maximum_a1_pattern_error") == 0.0
            and _matrix_free_summary(structure)
            and all(
                _finite_number(maxima.get(metric)) and float(maxima[metric]) <= expected_tolerance
                for metric in required_metrics
            )
        )
        cases_valid = cases_valid and case_valid
        details.append(f"{name}:{'pass' if case_valid else 'fail'}")
    add_check(
        checks,
        "structural_oracle_tolerances_and_eq55_pattern",
        cases_valid,
        ", ".join(details),
    )
    return by_name


def _crosscheck_gzip_checks(
    checks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    error: str | None,
    structural_cases: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    scales = config.get("rhs_scales", [])
    trials_per_scale = config.get("rhs_vectors_per_scale")
    expected_per_case = (
        len(scales) * trials_per_scale
        if isinstance(scales, list) and isinstance(trials_per_scale, int)
        else -1
    )
    counts = Counter(str(row.get("case")) for row in rows)
    required = {
        "case",
        "trial",
        "seed",
        "rhs_scale",
        "rhs_sha256",
        "right_hand_side",
        "corrected",
        "printed_sign",
    }
    rows_valid = error is None and bool(rows)
    unique_trials: set[tuple[str, int, float, int]] = set()
    corrected_by_case: dict[str, list[float]] = defaultdict(list)
    printed_sign_errors: list[float] = []

    for row in rows:
        name = str(row.get("case"))
        corrected = row.get("corrected", {})
        printed = row.get("printed_sign", {})
        vector = row.get("right_hand_side")
        try:
            packed = struct.pack(
                f"<{len(vector)}d",
                *(float(value) for value in vector),
            )
            hash_matches = hashlib.sha256(packed).hexdigest() == row.get("rhs_sha256")
            identity = (
                name,
                int(row["seed"]),
                float(row["rhs_scale"]),
                int(row["trial"]),
            )
            corrected_error = float(corrected["relative_solution_error"])
            corrected_residual = float(corrected["normalized_system_residual"])
            corrected_component = float(corrected["scaled_component_error"])
            printed_error = float(printed["relative_solution_error"])
        except (KeyError, TypeError, ValueError, OverflowError, struct.error):
            rows_valid = False
            continue

        tolerance = float(structural_cases.get(name, {}).get("acceptance_tolerance", -1.0))
        rows_valid = rows_valid and (
            required.issubset(row)
            and name in EXPECTED_STRUCTURAL_CASES
            and identity not in unique_trials
            and hash_matches
            and all(
                math.isfinite(value)
                for value in (
                    corrected_error,
                    corrected_residual,
                    corrected_component,
                    printed_error,
                )
            )
            and corrected_error <= tolerance
            and corrected_residual <= tolerance
            and corrected_component <= tolerance
        )
        unique_trials.add(identity)
        corrected_by_case[name].append(corrected_error)
        if name == "one_storage_t2":
            printed_sign_errors.append(printed_error)

    expected_counts = {name: expected_per_case for name in EXPECTED_STRUCTURAL_CASES}
    rows_valid = rows_valid and dict(counts) == expected_counts
    add_check(
        checks,
        "structural_crosscheck_gzip_complete",
        rows_valid,
        (f"rows={len(rows)}, counts={dict(sorted(counts.items()))}" if error is None else error),
    )

    corrected_maximum = max(corrected_by_case.get("one_storage_t2", [float("inf")]))
    printed_median = (
        statistics.median(printed_sign_errors) if printed_sign_errors else -float("inf")
    )
    add_check(
        checks,
        "gzip_confirms_corrected_rank_one_sign",
        (
            corrected_maximum <= float(config.get("structural_maximum_normalized_error", -1.0))
            and printed_median >= 1e-3
        ),
        (f"corrected_max={corrected_maximum:.3e}, printed_median={printed_median:.3e}"),
    )


def _sign_resolution_check(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> None:
    sign = evidence.get("structural_crosschecks", {}).get(
        "sign_resolution",
        {},
    )
    corrected = sign.get("corrected_maximum_relative_error")
    printed = sign.get("printed_sign_median_relative_error")
    passed = (
        sign.get("fixture") == "one_storage_t2"
        and sign.get("corrected_matches_direct") is True
        and sign.get("printed_sign_is_materially_wrong") is True
        and _finite_number(corrected)
        and float(corrected) <= float(config.get("structural_maximum_normalized_error", -1.0))
        and _finite_number(printed)
        and float(printed) >= 1e-3
        and "positive correction" in str(sign.get("implemented_correction_sign", ""))
        and "minus rank-one" in str(sign.get("implemented_correction_sign", ""))
        and "Equation (43)" in str(sign.get("paper_discrepancy", ""))
    )
    add_check(
        checks,
        "proposition_five_sign_discrepancy_resolved",
        passed,
        (
            f"corrected={corrected}, printed={printed}, "
            f"sign={sign.get('implemented_correction_sign')}"
        ),
    )


def _full_solver_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    section = evidence.get("full_solver_crosschecks", {})
    cases = section.get("cases", []) if isinstance(section, dict) else []
    by_name = {
        str(case.get("name")): case
        for case in cases
        if isinstance(case, dict) and case.get("name") is not None
    }
    coverage = set(by_name) == EXPECTED_DCOPF_CASES
    backend_valid = coverage and section.get("passed") is True
    physics_valid = coverage
    trajectory_valid = coverage

    for case in by_name.values():
        lockstep = case.get("lockstep", {})
        convergence = case.get("convergence", {})
        direct = convergence.get("direct", {})
        structural = convergence.get("structural", {})
        highs = convergence.get("highs", {})
        direct_validation = convergence.get("direct_validation", {})
        structural_validation = convergence.get("structural_validation", {})
        comparison = convergence.get("comparison", {})
        lockstep_comparison = lockstep.get("comparison", {})
        lockstep_tolerances = evidence.get("configuration", {}).get(
            "lockstep_tolerances",
            {},
        )

        backend_valid = backend_valid and (
            case.get("passed") is True
            and lockstep.get("direct", {}).get("backend") == "direct"
            and lockstep.get("structural", {}).get("backend") == "structural"
            and direct.get("backend") == "direct"
            and structural.get("backend") == "structural"
            and highs.get("status") == 0
            and _all_checks_true(lockstep.get("checks"))
            and _all_checks_true(convergence.get("direct_checks"))
            and _all_checks_true(convergence.get("structural_checks"))
            and _all_checks_true(convergence.get("checks"))
        )
        physics_valid = physics_valid and (
            direct_validation.get("physical_validation", {}).get("passed") is True
            and structural_validation.get("physical_validation", {}).get("passed") is True
            and direct_validation.get("scaled_objective_gap_to_highs", float("inf"))
            <= config.get("dcopf_maximum_scaled_objective_gap", -1.0)
            and structural_validation.get(
                "scaled_objective_gap_to_highs",
                float("inf"),
            )
            <= config.get("dcopf_maximum_scaled_objective_gap", -1.0)
            and direct_validation.get("maximum_physical_violation", float("inf"))
            <= config.get("dcopf_physical_tolerance", -1.0)
            and structural_validation.get(
                "maximum_physical_violation",
                float("inf"),
            )
            <= config.get("dcopf_physical_tolerance", -1.0)
            and direct_validation.get(
                "maximum_canonical_primal_violation",
                float("inf"),
            )
            <= config.get("dcopf_physical_tolerance", -1.0)
            and structural_validation.get(
                "maximum_canonical_primal_violation",
                float("inf"),
            )
            <= config.get("dcopf_physical_tolerance", -1.0)
            and direct.get("residuals", {}).get(
                "kkt_combined_norm",
                float("inf"),
            )
            <= config.get("dcopf_kkt_combined_target", -1.0)
            and structural.get("residuals", {}).get(
                "kkt_combined_norm",
                float("inf"),
            )
            <= config.get("dcopf_kkt_combined_target", -1.0)
        )

        state_maximum = max(
            float(lockstep_comparison.get(name, float("inf")))
            for name in ("x_relative_error", "y_relative_error", "z_relative_error")
        )
        objective_maximum = max(
            float(
                lockstep_comparison.get(
                    "maximum_scaled_objective_gap",
                    float("inf"),
                )
            ),
            float(
                lockstep_comparison.get(
                    "final_scaled_objective_gap",
                    float("inf"),
                )
            ),
        )
        residual_maximum = max(
            float(
                lockstep_comparison.get(
                    "maximum_combined_residual_gap",
                    float("inf"),
                )
            ),
            float(
                lockstep_comparison.get(
                    "maximum_paper_component_gap",
                    float("inf"),
                )
            ),
            float(
                lockstep_comparison.get(
                    "final_combined_residual_gap",
                    float("inf"),
                )
            ),
        )
        trajectory_valid = trajectory_valid and (
            lockstep_comparison.get("iteration_grid_matches") is True
            and state_maximum <= lockstep_tolerances.get("state", -1.0)
            and objective_maximum <= lockstep_tolerances.get("objective", -1.0)
            and residual_maximum <= lockstep_tolerances.get("residual", -1.0)
            and comparison.get("scaled_objective_gap", float("inf"))
            <= config.get(
                "direct_structural_maximum_scaled_objective_difference",
                -1.0,
            )
            and comparison.get("x_relative_error", float("inf"))
            <= config.get(
                "direct_structural_maximum_solution_relative_difference",
                -1.0,
            )
            and comparison.get("iteration_difference", float("inf"))
            <= comparison.get("iteration_difference_limit", -1)
            and comparison.get("structural_repeat_deterministic") is True
        )

    add_check(
        checks,
        "two_full_solver_cases_and_backends",
        backend_valid,
        f"cases={sorted(by_name)}; direct, structural, and HiGHS required",
    )
    add_check(
        checks,
        "full_solver_highs_objective_kkt_and_physics",
        physics_valid,
        "both backends satisfy the configured HiGHS, KKT, and physical gates",
    )
    add_check(
        checks,
        "full_solver_trajectory_parity_and_determinism",
        trajectory_valid,
        "lockstep trajectories and converged structural repeat checked",
    )
    return by_name


def _trajectory_gzip_checks(
    checks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    error: str | None,
    full_cases: dict[str, dict[str, Any]],
) -> None:
    required = {
        "case",
        "classification",
        "phase",
        "backend",
        "iteration",
        "canonical_variable_objective",
        "total_objective",
        "iteration_loop_elapsed_seconds",
        "paper_raw",
        "paper_normalized",
        "sigma",
        "restart_count",
    }
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    valid = error is None and bool(rows)
    for row in rows:
        valid = valid and required.issubset(row)
        try:
            group = (
                str(row["case"]),
                str(row["phase"]),
                str(row["backend"]),
            )
            valid = valid and (
                group[0] in EXPECTED_DCOPF_CASES
                and int(row["iteration"]) > 0
                and float(row["sigma"]) == 1.0
                and int(row["restart_count"]) == 0
                and _finite_number(row["canonical_variable_objective"])
                and _finite_number(row["total_objective"])
                and _finite_number(row["iteration_loop_elapsed_seconds"])
            )
            groups[group].append(row)
        except (KeyError, TypeError, ValueError):
            valid = False

    expected_last: dict[tuple[str, str, str], int] = {}
    for name, case in full_cases.items():
        lockstep = case.get("lockstep", {})
        convergence = case.get("convergence", {})
        phase = f"lockstep_{lockstep.get('requested_iterations')}"
        expected_last[(name, phase, "direct")] = int(
            lockstep.get("direct", {}).get("iterations", -1)
        )
        expected_last[(name, phase, "structural")] = int(
            lockstep.get("structural", {}).get("iterations", -1)
        )
        expected_last[(name, "convergence", "direct")] = int(
            convergence.get("direct", {}).get("iterations", -1)
        )
        structural_iterations = int(convergence.get("structural", {}).get("iterations", -1))
        expected_last[(name, "convergence", "structural")] = structural_iterations
        expected_last[(name, "convergence", "structural_repeat")] = structural_iterations

    valid = valid and set(groups) == set(expected_last)
    last_iterations: dict[str, int] = {}
    for group, records in groups.items():
        iterations = [int(record.get("iteration", -1)) for record in records]
        valid = valid and (
            iterations == sorted(set(iterations)) and iterations[-1] == expected_last.get(group)
        )
        last_iterations["/".join(group)] = iterations[-1]
    add_check(
        checks,
        "solver_trajectory_gzip_complete",
        valid,
        (f"rows={len(rows)}, last_iterations={last_iterations}" if error is None else error),
    )


def _matrix_free_workspace_check(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> None:
    summaries: list[Any] = []
    summaries.extend(
        case.get("structure")
        for case in evidence.get("structural_crosschecks", {}).get("cases", [])
    )
    for case in evidence.get("full_solver_crosschecks", {}).get("cases", []):
        summaries.append(case.get("structure"))
        summaries.append(
            case.get("lockstep", {}).get("structural", {}).get("structural_diagnostics")
        )
        summaries.append(
            case.get("convergence", {}).get("structural", {}).get("structural_diagnostics")
        )
    performance = evidence.get("performance_and_complexity", {})
    for point in performance.get("direct_structural_comparison", []):
        summaries.append(point.get("structural_workspace"))
    for point in performance.get("complexity_points", []):
        summaries.append(point.get("structural_workspace"))

    passed = bool(summaries) and all(_matrix_free_summary(summary) for summary in summaries)
    add_check(
        checks,
        "no_dense_structural_gram_or_kronecker",
        passed,
        f"{len(summaries)} structural workspaces checked",
    )


def _performance_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> None:
    performance = evidence.get("performance_and_complexity", {})
    comparison = performance.get("direct_structural_comparison", [])
    complexity = performance.get("complexity_points", [])
    largest = max(
        comparison,
        key=lambda point: point.get("paper_work_proxy", -1),
        default={},
    )
    speedups = largest.get("speedup_direct_over_structural") or {}
    timing_protocol = performance.get("timing_protocol", {})
    distributions_valid = bool(comparison)
    for point in comparison:
        for backend in ("direct", "structural"):
            for phase in ("solve_only", "rhs_and_solve"):
                distribution = (
                    point.get("timings", {})
                    .get(backend, {})
                    .get(
                        phase,
                        {},
                    )
                )
                samples = distribution.get(
                    "samples_seconds_per_right_hand_side",
                    [],
                )
                distributions_valid = distributions_valid and (
                    bool(samples)
                    and all(_finite_number(value) and value > 0.0 for value in samples)
                    and _finite_number(
                        distribution.get(
                            "median_seconds_per_right_hand_side",
                        )
                    )
                )
    advantage = (
        performance.get("checks", {}).get("largest_solve_only_advantage_measured") is True
        and speedups.get("solve_only", 0.0) > 1.0
        and largest.get("periods") == 1_024
        and distributions_valid
        and timing_protocol.get("precision") == "FP64"
        and timing_protocol.get("device") == "local CPU"
        and timing_protocol.get("preparation_reported_separately") is True
    )
    add_check(
        checks,
        "largest_synthetic_structural_speed_advantage",
        advantage,
        f"periods={largest.get('periods')}, speedups={speedups}",
    )

    acceptance = performance.get("acceptance", {})
    slope_range = acceptance.get("slope_range", [])
    minimum_r_squared = acceptance.get("minimum_r_squared", -1.0)
    maximum_spread = acceptance.get(
        "maximum_normalized_time_per_work_spread",
        -1.0,
    )
    fits = performance.get("fits", {})
    expected_fit_names = {
        "structural_solve_only_O_T_plus_NESS",
        "rhs_and_structural_solve_O_nnz_A1",
    }
    solve_only_fit = fits.get("structural_solve_only_O_T_plus_NESS", {})
    gating_fit = fits.get("rhs_and_structural_solve_O_nnz_A1", {})
    fits_valid = (
        set(fits) == expected_fit_names
        and len(complexity) == 6
        and [point.get("periods") for point in complexity]
        == [2_048, 4_096, 8_192, 16_384, 32_768, 65_536]
        and isinstance(slope_range, list)
        and len(slope_range) == 2
        and acceptance.get("gating_fit") == "rhs_and_structural_solve_O_nnz_A1"
        and all(_finite_number(value) for value in solve_only_fit.values())
    )
    if fits_valid:
        fits_valid = (
            slope_range[0] <= gating_fit.get("slope", float("inf")) <= slope_range[1]
            and gating_fit.get("r_squared", -float("inf")) >= minimum_r_squared
            and gating_fit.get(
                "normalized_time_per_work_spread",
                float("inf"),
            )
            <= maximum_spread
        )
    fits_valid = fits_valid and (
        performance.get("checks", {}).get("solve_only_fit_reported") is True
        and performance.get("checks", {}).get("paper_work_proxy_rhs_and_solve_approximately_linear")
        is True
        and performance.get("checks", {}).get("all_timing_checksums_finite") is True
        and isinstance(performance.get("non_gating_diagnostics"), dict)
        and "not used to claim"
        in performance.get("non_gating_diagnostics", {}).get(
            "solve_only_interpretation",
            "",
        )
    )
    add_check(
        checks,
        "both_empirical_fits_reported_and_paper_boundary_fit_passes",
        fits_valid,
        (f"gating_fit={gating_fit}, non_gating_solve_only_fit={solve_only_fit}"),
    )

    decisions_path = PROJECT_ROOT / "docs" / "decisions.md"
    report_path = PROJECT_ROOT / "docs" / "stage_reports" / "stage_4_report.md"
    log_path = PROJECT_ROOT / "logs" / "stage_4" / "commands_and_results.txt"
    decisions_text = (
        decisions_path.read_text(encoding="utf-8").lower() if decisions_path.is_file() else ""
    )
    report_text = report_path.read_text(encoding="utf-8").lower() if report_path.is_file() else ""
    log_text = log_path.read_text(encoding="utf-8").lower() if log_path.is_file() else ""
    failures_disclosed = (
        "d-0033" in decisions_text
        and "two disclosed timing-protocol failures" in decisions_text
        and "solve-only" in report_text
        and "non-gating" in report_text
        and "two" in report_text
        and "0.636" in log_text
        and "0.552" in log_text
        and "4.889" in log_text
        and "4.860" in log_text
    )
    add_check(
        checks,
        "two_failed_solve_only_fits_preserved_and_disclosed",
        failures_disclosed,
        "D-0033 plus both failed fit records required in report and log",
    )


def _stability_and_boundary_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> None:
    stability = evidence.get("numerical_stability", {})
    limits = stability.get("documented_limits", [])
    limit_text = " ".join(str(limit).lower() for limit in limits)
    report_path = PROJECT_ROOT / "docs" / "stage_reports" / "stage_4_report.md"
    notes_path = PROJECT_ROOT / "docs" / "mathematical_notes.md"
    report_text = report_path.read_text(encoding="utf-8").lower() if report_path.is_file() else ""
    notes_text = notes_path.read_text(encoding="utf-8").lower() if notes_path.is_file() else ""
    documented = (
        isinstance(limits, list)
        and len(limits) >= 4
        and stability.get("extreme_efficiency_covered") is True
        and stability.get("many_ideal_storage_covered") is True
        and "schur" in limit_text
        and "efficien" in limit_text
        and "mean/zero-mean" in limit_text
        and "fp64" in limit_text
        and "numerical stability" in report_text
        and "corrected and stable proposition 5" in notes_text
    )
    add_check(
        checks,
        "numerical_stability_limits_documented_and_exercised",
        documented,
        f"documented_limits={len(limits)}",
    )

    boundary = evidence.get("reproduction_boundary", {})
    unsupported = config.get("unsupported_features", {})
    future_paths = (
        "src/gpu_dcopf_hpr/scaling.py",
        "src/gpu_dcopf_hpr/adaptive_sigma.py",
        "src/gpu_dcopf_hpr/restart.py",
        "src/gpu_dcopf_hpr/gpu_solver.py",
        "scripts/run_stage_5.py",
        "scripts/run_stage_6.py",
    )
    premature = [path for path in future_paths if (PROJECT_ROOT / path).exists()]
    gate_text = " ".join(
        path.read_text(encoding="utf-8")
        for path in (
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "project_state.md",
            report_path,
        )
        if path.is_file()
    )
    boundary_valid = (
        boundary.get("classification") == "paper-specific structural reproduction"
        and boundary.get("structural_y1_used") is True
        and boundary.get("direct_oracle_used") is True
        and boundary.get("dgx_spark_touched") is False
        and boundary.get("gpu_code_used") is False
        and boundary.get("adaptive_sigma_used") is False
        and boundary.get("restart_used") is False
        and boundary.get("scaling_used") is False
        and boundary.get("precision") == "FP64"
        and boundary.get("execution_device") == "local CPU"
        and unsupported
        == {
            "adaptive_sigma": False,
            "dgx_execution": False,
            "gpu": False,
            "restart": False,
            "scaling": False,
        }
        and config.get("sigma") == 1.0
        and not premature
        and "APPROVE STAGE 4 AND RUN STAGE 5" in gate_text
        and "Stage 6" in gate_text
    )
    add_check(
        checks,
        "stage_five_and_six_boundaries_preserved",
        boundary_valid,
        (
            "fixed sigma CPU/FP64; no scaling, adaptation, restart, GPU, or DGX"
            if not premature
            else f"premature paths={premature}"
        ),
    )


def run_checks(evidence_path: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    config, config_error = _load_json(DEFAULT_CONFIG)
    evidence, evidence_error = _load_json(evidence_path)
    evidence_files = evidence.get("evidence_files", {})
    crosscheck_path = evidence_path.parent / str(
        evidence_files.get(
            "structural_crosschecks",
            "structural_crosschecks.jsonl.gz",
        )
    )
    trajectory_path = evidence_path.parent / str(
        evidence_files.get(
            "solver_trajectories",
            "solver_trajectories.jsonl.gz",
        )
    )

    required_paths = (
        "data/raw/matpower/case5.m",
        "configs/dcopf/case5_base_stage_2.json",
        "configs/dcopf/case5_synthetic_extension_stage_2.json",
        "configs/sgs_hpr/stage_4_structural.json",
        "src/gpu_dcopf_hpr/structural_y1.py",
        "src/gpu_dcopf_hpr/sgs_hpr.py",
        "scripts/run_stage_4.py",
        "scripts/check_stage_4.py",
        "tests/unit/test_structural_y1.py",
        "tests/integration/test_stage4_structural_y1.py",
        "logs/stage_4/commands_and_results.txt",
        "docs/stage_reports/stage_4_report.md",
        "docs/project_state.md",
        "docs/decisions.md",
        "docs/mathematical_notes.md",
        "docs/paper_specification.md",
        "README.md",
    )
    missing = [path for path in required_paths if not (PROJECT_ROOT / path).is_file()]
    for path in (evidence_path, crosscheck_path, trajectory_path):
        if not path.is_file():
            missing.append(str(path))
    add_check(
        checks,
        "required_stage_four_paths",
        not missing,
        "complete" if not missing else f"missing={missing}",
    )
    add_check(
        checks,
        "configuration_is_valid_json",
        config_error is None,
        "loaded" if config_error is None else config_error,
    )
    add_check(
        checks,
        "validation_is_valid_json",
        evidence_error is None,
        "loaded" if evidence_error is None else evidence_error,
    )

    stage_passed = (
        evidence.get("stage") == 4
        and evidence.get("all_passed") is True
        and evidence.get("section_passes")
        == {
            "structural_crosschecks": True,
            "full_solver_crosschecks": True,
            "performance_and_complexity": True,
        }
    )
    add_check(
        checks,
        "stage_four_validation_sections_passed",
        stage_passed,
        str(evidence.get("section_passes", "unavailable")),
    )

    embedded_config = evidence.get("configuration", {}).get(
        "stage_4_structural",
        {},
    )
    config_matches = (
        bool(config)
        and embedded_config == config
        and config.get("stage") == 4
        and config.get("precision") == "FP64"
        and config.get("sigma") == 1.0
    )
    add_check(
        checks,
        "embedded_configuration_matches_versioned_config",
        config_matches,
        "exact match" if config_matches else "configuration drift detected",
    )

    structural_cases = _structural_case_checks(checks, evidence, config)
    crosscheck_rows, crosscheck_error = _load_gzip_jsonl(crosscheck_path)
    _crosscheck_gzip_checks(
        checks,
        crosscheck_rows,
        crosscheck_error,
        structural_cases,
        config,
    )
    _sign_resolution_check(checks, evidence, config)
    full_cases = _full_solver_checks(checks, evidence, config)
    trajectory_rows, trajectory_error = _load_gzip_jsonl(trajectory_path)
    _trajectory_gzip_checks(
        checks,
        trajectory_rows,
        trajectory_error,
        full_cases,
    )
    _matrix_free_workspace_check(checks, evidence)
    _performance_checks(checks, evidence)
    _stability_and_boundary_checks(checks, evidence, config)

    totals: dict[str, int] = defaultdict(int)
    for check in checks:
        totals["passed" if check["passed"] else "failed"] += 1
    return {
        "stage": 4,
        "passed": totals["failed"] == 0,
        "evidence": _display_path(evidence_path),
        "evidence_files": {
            "structural_crosschecks": _display_path(crosscheck_path),
            "solver_trajectories": _display_path(trajectory_path),
        },
        "summary": dict(totals),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Stage 4 validation JSON to inspect.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_checks(args.evidence.resolve())
    rendered = (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(rendered, end="")
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
