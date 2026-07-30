"""Independently validate preserved Stage 5 evidence and the Stage 6 boundary."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sgs_hpr" / "stage_5_preconditioning_controls.json"
DEFAULT_EVIDENCE = PROJECT_ROOT / "results" / "raw" / "stage_5" / "stage_5_validation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_5" / "stage_5_checks.json"

EXPECTED_COMPONENT_CASES = {
    "dense_analytic": False,
    "sparse_planted_random": True,
}
EXPECTED_DCOPF_CASES = {
    "case5_base_t1": 1,
    "case5_synthetic_extension_t2": 2,
}
EXPECTED_HPR_LP_PIN = "1941fbcfbf2dae14e4a439b22f0ea1e1c05f4a29"
FULL_PRECONDITIONING = "10 Ruiz + Pock-Chambolle alpha=1 + norm"
PREPROCESSING_RUN_NAMES = {
    "unscaled Stage 4 structural baseline": "unscaled_fixed_no_restart",
    "norm b/c only": "normalized_fixed_no_restart",
    "10 Ruiz + norm": "ruiz_fixed_no_restart",
    FULL_PRECONDITIONING: "full_fixed_no_restart",
}
PAPER_RESIDUAL_COMPONENTS = {
    "primal_feasibility",
    "box",
    "stationarity",
}
ROUNDTRIP_ERRORS = {
    "state_x_roundtrip",
    "state_y_roundtrip",
    "state_z_roundtrip",
    "matrix_roundtrip",
    "b_roundtrip",
    "c_roundtrip",
    "lower_roundtrip",
    "upper_roundtrip",
    "objective_identity",
}
IDENTITY_ERRORS = {
    "primal_transform_identity",
    "stationarity_transform_identity",
}


def _display_path(path: Path) -> str:
    """Render repository paths without embedding the local checkout prefix."""
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


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_nonnegative(value: Any) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def _finite_positive(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _numbers_close(left: Any, right: Any) -> bool:
    return (
        _finite_number(left)
        and _finite_number(right)
        and math.isclose(float(left), float(right), rel_tol=1e-13, abs_tol=1e-14)
    )


def _all_checks_true(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"missing {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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


def _exact_preconditioner_sequence(
    summary: Any,
    *,
    config: dict[str, Any],
) -> bool:
    value = _mapping(summary)
    steps = [_mapping(step) for step in _sequence(value.get("steps"))]
    ruiz_iterations = config.get("ruiz_iterations")
    expected = [("ruiz", iteration, "infinity") for iteration in range(1, 11)] + [
        ("pock_chambolle", 1, "l1")
    ]
    actual = [
        (
            step.get("method"),
            step.get("iteration"),
            step.get("norm"),
        )
        for step in steps
    ]
    row_range = _sequence(value.get("row_denominator_range"))
    column_range = _sequence(value.get("column_denominator_range"))
    ranges_valid = (
        len(row_range) == 2
        and len(column_range) == 2
        and all(_finite_positive(item) for item in row_range + column_range)
        and float(row_range[0]) <= float(row_range[1])
        and float(column_range[0]) <= float(column_range[1])
    )
    step_diagnostics_valid = all(
        (
            _finite_nonnegative(step.get("row_zero_count_before"))
            and _finite_nonnegative(step.get("column_zero_count_before"))
            and len(_sequence(step.get("row_range_before"))) == 2
            and len(_sequence(step.get("column_range_before"))) == 2
            and len(_sequence(step.get("row_range_after"))) == 2
            and len(_sequence(step.get("column_range_after"))) == 2
            and all(
                _finite_nonnegative(item)
                for key in (
                    "row_range_before",
                    "column_range_before",
                    "row_range_after",
                    "column_range_after",
                )
                for item in _sequence(step.get(key))
            )
        )
        for step in steps
    )
    return (
        ruiz_iterations == 10
        and config.get("pock_chambolle_alpha") == 1
        and config.get("normalize_b_and_c") is True
        and value.get("ruiz_iterations") == 10
        and value.get("pock_chambolle_applied") is True
        and value.get("normalization_applied") is True
        and value.get("nnz_preserved") is True
        and value.get("original_nnz") == value.get("scaled_nnz")
        and isinstance(value.get("original_nnz"), int)
        and value.get("original_nnz") > 0
        and _finite_positive(value.get("b_scale"))
        and _finite_positive(value.get("c_scale"))
        and ranges_valid
        and actual == expected
        and step_diagnostics_valid
    )


def _configuration_and_source_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> None:
    embedded = _mapping(evidence.get("configuration"))
    restart = _mapping(config.get("restart_parameters"))
    guards = _mapping(config.get("sigma_guards"))
    core_settings_valid = (
        config.get("stage") == 5
        and config.get("precision") == "FP64"
        and config.get("backend") == "NumPy and SciPy CPU"
        and config.get("policy_check_interval") == 100
        and config.get("ruiz_iterations") == 10
        and config.get("pock_chambolle_alpha") == 1
        and config.get("normalize_b_and_c") is True
        and restart
        == {
            "alpha_sufficient": 0.2,
            "alpha_necessary": 0.6,
            "alpha_long": 0.2,
        }
        and guards
        == {
            "movement_minimum": 1e-16,
            "movement_maximum": 1e12,
            "infeasibility_ratio_minimum": 1e-8,
            "infeasibility_ratio_maximum": 1e8,
        }
    )
    add_check(
        checks,
        "embedded_configuration_matches_versioned_config",
        bool(config) and embedded == config and core_settings_valid,
        "exact match and Stage 5 constants preserved",
    )

    source_audit = _mapping(evidence.get("source_audit"))
    sources = _mapping(config.get("sources"))
    audited_sources = _mapping(source_audit.get("sources"))
    pin = sources.get("hpr_lp_reconstruction_pin")
    current_commit = sources.get("hpr_lp_current_commit_audited_for_drift")
    pins_valid = (
        pin == EXPECTED_HPR_LP_PIN
        and source_audit.get("hpr_lp_reconstruction_pin") == pin
        and source_audit.get("hpr_lp_current_commit_audited_for_drift") == current_commit
        and audited_sources == sources
        and isinstance(current_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", current_commit) is not None
        and sources.get("hpr_lp_source_repository") == "https://github.com/PolyU-IOR/HPR-LP"
        and sources.get("hpr_lp_article") == "https://doi.org/10.1007/s12532-025-00292-0"
        and sources.get("pock_chambolle") == "https://doi.org/10.1109/ICCV.2011.6126441"
    )
    add_check(
        checks,
        "primary_sources_and_hpr_lp_pin_preserved",
        pins_valid,
        f"reconstruction_pin={pin}, audited_commit={current_commit}",
    )

    provenance = _mapping(source_audit.get("formula_provenance"))
    provenance_text = {name: str(value).lower() for name, value in provenance.items()}
    unsupported = _mapping(config.get("unsupported_or_explicitly_bounded"))
    formula_boundary_valid = (
        {
            "ten_ruiz_iterations",
            "pock_chambolle_alpha_one",
            "b_c_norm_normalization",
            "restart",
            "adaptive_sigma",
        }
        == set(provenance)
        and "dcopf manuscript" in provenance_text.get("ten_ruiz_iterations", "")
        and "pdlp" in provenance_text.get("pock_chambolle_alpha_one", "")
        and "pock-chambolle" in provenance_text.get("pock_chambolle_alpha_one", "")
        and "dcopf manuscript" in provenance_text.get("b_c_norm_normalization", "")
        and "eqs. (10)-(12)" in provenance_text.get("restart", "")
        and "100" in provenance_text.get("restart", "")
        and "eqs. (15)-(18)" in provenance_text.get("adaptive_sigma", "")
        and "sgs metric" in provenance_text.get("adaptive_sigma", "")
        and source_audit.get("exact_author_dcopf_policy_available") is False
        and unsupported.get("authors_exact_dcopf_policy_code_available") is False
        and unsupported.get("adaptive_without_restart_is_paper_algorithm") is False
        and "not byte-for-byte" in str(source_audit.get("implementation_claim", "")).lower()
        and "published equations" in str(source_audit.get("source_drift_note", "")).lower()
        and "extra sigma heuristics" in str(source_audit.get("source_drift_note", "")).lower()
    )
    add_check(
        checks,
        "published_formula_transfer_boundary_is_explicit",
        formula_boundary_valid,
        "published HPR-LP equations are distinguished from unavailable DCOPF author code",
    )

    manuscript_hash = source_audit.get("dcopf_manuscript_sha256")
    manuscript_valid = (
        source_audit.get("dcopf_manuscript_local_path_available") is True
        and isinstance(manuscript_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", manuscript_hash) is not None
    )
    add_check(
        checks,
        "local_dcopf_manuscript_fingerprint_recorded",
        manuscript_valid,
        f"sha256={manuscript_hash}",
    )


def _component_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> None:
    section = _mapping(evidence.get("component_validation"))
    cases = [_mapping(case) for case in _sequence(section.get("cases"))]
    by_name = {str(case.get("name")): case for case in cases if case.get("name") is not None}
    coverage = (
        set(by_name) == set(EXPECTED_COMPONENT_CASES)
        and section.get("passed") is True
        and all(
            by_name[name].get("source_matrix_sparse") is sparse_expected
            for name, sparse_expected in EXPECTED_COMPONENT_CASES.items()
        )
    )
    add_check(
        checks,
        "dense_and_sparse_component_fixtures_present",
        coverage,
        f"cases={sorted(by_name)}",
    )

    roundtrip_tolerance = config.get("transform_roundtrip_tolerance")
    identity_tolerance = config.get("transform_identity_tolerance")
    tolerance_valid = (
        coverage and _finite_positive(roundtrip_tolerance) and _finite_positive(identity_tolerance)
    )
    details: list[str] = []
    for name, case in by_name.items():
        errors = _mapping(case.get("errors"))
        preconditioner = _mapping(case.get("preconditioner"))
        roundtrip_values = [errors.get(metric) for metric in ROUNDTRIP_ERRORS]
        identity_values = [errors.get(metric) for metric in IDENTITY_ERRORS]
        case_valid = (
            case.get("passed") is True
            and _all_checks_true(case.get("checks"))
            and set(errors) == ROUNDTRIP_ERRORS | IDENTITY_ERRORS
            and all(_finite_nonnegative(value) for value in roundtrip_values)
            and all(_finite_nonnegative(value) for value in identity_values)
            and max(float(value) for value in roundtrip_values) <= float(roundtrip_tolerance)
            and max(float(value) for value in identity_values) <= float(identity_tolerance)
            and _numbers_close(
                preconditioner.get("b_scale"),
                1.0 + float(preconditioner.get("diagonally_scaled_b_norm", math.nan)),
            )
            and _numbers_close(
                preconditioner.get("c_scale"),
                1.0 + float(preconditioner.get("diagonally_scaled_c_norm", math.nan)),
            )
        )
        tolerance_valid = tolerance_valid and case_valid
        details.append(
            f"{name}:roundtrip={max(roundtrip_values, default='missing')},"
            f" identity={max(identity_values, default='missing')}"
        )
    add_check(
        checks,
        "component_roundtrips_and_algebraic_identities_within_tolerance",
        tolerance_valid,
        "; ".join(details),
    )

    sequence_valid = coverage and all(
        _exact_preconditioner_sequence(
            case.get("preconditioner"),
            config=config,
        )
        for case in by_name.values()
    )
    add_check(
        checks,
        "exact_ten_ruiz_then_one_pock_chambolle_order",
        sequence_valid,
        "Ruiz 1..10 (infinity norm), then Pock-Chambolle alpha=1 (L1), then normalization",
    )


def _summary_policy_cadence_valid(
    summary: dict[str, Any],
    *,
    interval: int,
) -> bool:
    control = _mapping(summary.get("control"))
    iterations = summary.get("iterations")
    actual = _sequence(summary.get("policy_event_iterations"))
    enabled = control.get("adaptive_sigma") is True or control.get("restart") is True
    if not isinstance(iterations, int) or iterations <= 0:
        return False
    if not enabled:
        expected: list[int] = []
    else:
        last_eligible = iterations - 1 if summary.get("converged") is True else iterations
        expected = list(range(interval, last_eligible + 1, interval))
    return (
        control.get("check_interval") == interval
        and actual == expected
        and summary.get("policy_event_count") == len(expected)
        and summary.get("policy_event_cadence_valid") is True
    )


def _original_space_acceptance(
    summary: Any,
    *,
    config: dict[str, Any],
) -> bool:
    run = _mapping(summary)
    residuals = _mapping(run.get("original_residuals"))
    normalized = _mapping(residuals.get("paper_normalized_norms"))
    stopping = _mapping(residuals.get("paper_stopping"))
    physical = _mapping(run.get("physical_validation"))
    paper_tolerance = config.get("paper_tolerance")
    kkt_target = config.get("dcopf_kkt_combined_target")
    physical_tolerance = config.get("dcopf_physical_tolerance")
    objective_tolerance = config.get("dcopf_maximum_scaled_objective_gap")
    equality_tolerance = config.get("maximum_equality_infinity_residual")
    identity_tolerance = config.get("maximum_z_x_identity_error")
    return (
        run.get("passed") is True
        and run.get("fixed_horizon") is False
        and run.get("converged") is True
        and run.get("backend") == "direct"
        and run.get("preconditioning") == FULL_PRECONDITIONING
        and _all_checks_true(run.get("checks"))
        and residuals.get("tolerance") == paper_tolerance
        and set(normalized) == PAPER_RESIDUAL_COMPONENTS
        and all(
            _finite_nonnegative(value) and float(value) <= float(paper_tolerance)
            for value in normalized.values()
        )
        and stopping
        == {
            "primal_feasibility": True,
            "box": True,
            "stationarity": True,
            "all_satisfied": True,
        }
        and _finite_nonnegative(residuals.get("kkt_combined_norm"))
        and float(residuals["kkt_combined_norm"]) <= float(kkt_target)
        and _finite_nonnegative(run.get("scaled_objective_gap_to_highs"))
        and float(run["scaled_objective_gap_to_highs"]) <= float(objective_tolerance)
        and _finite_nonnegative(run.get("maximum_physical_violation"))
        and float(run["maximum_physical_violation"]) <= float(physical_tolerance)
        and _finite_nonnegative(run.get("maximum_canonical_primal_violation"))
        and float(run["maximum_canonical_primal_violation"]) <= float(physical_tolerance)
        and physical.get("passed") is True
        and physical.get("mode") == "approximate_first_order_candidate"
        and _finite_nonnegative(run.get("maximum_equality_solve_infinity_residual"))
        and float(run["maximum_equality_solve_infinity_residual"]) <= float(equality_tolerance)
        and _finite_nonnegative(run.get("maximum_z_x_identity_error"))
        and float(run["maximum_z_x_identity_error"]) <= float(identity_tolerance)
        and _finite_number(run.get("objective"))
        and _finite_number(run.get("reference_objective"))
        and isinstance(run.get("scaled_residuals"), dict)
        and _summary_policy_cadence_valid(
            run,
            interval=int(config["policy_check_interval"]),
        )
    )


def _dcopf_case_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    section = _mapping(evidence.get("dcopf_ablation"))
    cases = [_mapping(case) for case in _sequence(section.get("cases"))]
    by_name = {str(case.get("name")): case for case in cases if case.get("name") is not None}
    coverage = (
        set(by_name) == set(EXPECTED_DCOPF_CASES)
        and section.get("passed") is True
        and all(
            _mapping(by_name[name].get("dimensions")).get("periods") == periods
            for name, periods in EXPECTED_DCOPF_CASES.items()
        )
        and "synthetic"
        in str(by_name.get("case5_synthetic_extension_t2", {}).get("classification", "")).lower()
    )
    add_check(
        checks,
        "both_t1_and_t2_dcopf_cases_present",
        coverage,
        f"cases={sorted(by_name)}",
    )

    sequences_valid = coverage and all(
        _exact_preconditioner_sequence(
            case.get("full_preconditioner"),
            config=config,
        )
        for case in by_name.values()
    )
    add_check(
        checks,
        "dcopf_full_preconditioners_have_exact_order",
        sequences_valid,
        "both DCOPF cases preserve ten Ruiz steps, one PC step, and norm scaling",
    )

    acceptance_valid = coverage
    controls_valid = coverage
    details: list[str] = []
    for name, case in by_name.items():
        main = _mapping(case.get("main_full_adaptive_restart"))
        highs = _mapping(case.get("highs"))
        control = _mapping(main.get("control"))
        sigma = _mapping(main.get("sigma"))
        case_acceptance = (
            case.get("passed") is True
            and highs.get("status") == 0
            and _original_space_acceptance(main, config=config)
        )
        control_valid = (
            control.get("adaptive_sigma") is True
            and control.get("restart") is True
            and _finite_positive(sigma.get("initial"))
            and _finite_positive(sigma.get("final"))
            and _finite_positive(sigma.get("minimum"))
            and _finite_positive(sigma.get("maximum"))
            and isinstance(sigma.get("attempts"), int)
            and sigma.get("attempts") > 0
            and isinstance(sigma.get("accepted"), int)
            and sigma.get("accepted") > 0
            and isinstance(main.get("restart_count"), int)
            and main.get("restart_count") > 0
        )
        acceptance_valid = acceptance_valid and case_acceptance
        controls_valid = controls_valid and control_valid
        details.append(
            f"{name}:kkt={_mapping(main.get('original_residuals')).get('kkt_combined_norm')}, "
            f"physical={main.get('maximum_physical_violation')}, "
            f"objective_gap={main.get('scaled_objective_gap_to_highs')}"
        )
    add_check(
        checks,
        "dcopf_original_space_stopping_kkt_objective_and_physics",
        acceptance_valid,
        "; ".join(details),
    )
    add_check(
        checks,
        "adaptive_sigma_and_restart_exercised_on_both_dcopf_cases",
        controls_valid,
        "both main runs contain accepted sigma updates and restarts",
    )
    return by_name


def _ablation_checks(
    checks: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    base = _mapping(cases.get("case5_base_t1"))
    preprocessing = [_mapping(run) for run in _sequence(base.get("preprocessing_ablation"))]
    preprocessing_by_label = {
        str(run.get("preconditioning")): run
        for run in preprocessing
        if run.get("preconditioning") is not None
    }
    fixed_horizon = config.get("fixed_horizon_iterations")
    preprocessing_valid = set(preprocessing_by_label) == set(PREPROCESSING_RUN_NAMES) and len(
        preprocessing
    ) == len(PREPROCESSING_RUN_NAMES)
    for label, run in preprocessing_by_label.items():
        expected_backend = (
            "structural" if label == "unscaled Stage 4 structural baseline" else "direct"
        )
        control = _mapping(run.get("control"))
        sigma = _mapping(run.get("sigma"))
        preprocessing_valid = preprocessing_valid and (
            run.get("passed") is True
            and run.get("fixed_horizon") is True
            and run.get("iterations") == fixed_horizon
            and run.get("backend") == expected_backend
            and control.get("adaptive_sigma") is False
            and control.get("restart") is False
            and run.get("policy_event_count") == 0
            and _sequence(run.get("policy_event_iterations")) == []
            and sigma
            == {
                "initial": config.get("initial_sigma"),
                "final": config.get("initial_sigma"),
                "minimum": config.get("initial_sigma"),
                "maximum": config.get("initial_sigma"),
                "attempts": 0,
                "accepted": 0,
            }
            and _all_checks_true(run.get("checks"))
        )
    add_check(
        checks,
        "preprocessing_ablation_covers_unscaled_norm_ruiz_and_full",
        preprocessing_valid,
        f"labels={sorted(preprocessing_by_label)}",
    )

    control_runs = [_mapping(run) for run in _sequence(base.get("control_ablation"))]
    control_by_flags: dict[tuple[bool, bool], dict[str, Any]] = {}
    duplicate_controls = False
    for run in control_runs:
        control = _mapping(run.get("control"))
        key = (
            control.get("adaptive_sigma") is True,
            control.get("restart") is True,
        )
        if key in control_by_flags:
            duplicate_controls = True
        control_by_flags[key] = run
    expected_controls = {
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    }
    controls_present = (
        not duplicate_controls
        and set(control_by_flags) == expected_controls
        and len(control_runs) == 4
    )
    for key in ((False, False), (False, True), (True, True)):
        controls_present = controls_present and _original_space_acceptance(
            control_by_flags.get(key),
            config=config,
        )
    add_check(
        checks,
        "all_four_control_combinations_present",
        controls_present,
        f"controls={sorted(control_by_flags)}",
    )

    adaptive_only = _mapping(control_by_flags.get((True, False)))
    adaptive_checks = _mapping(adaptive_only.get("checks"))
    adaptive_sigma = _mapping(adaptive_only.get("sigma"))
    adaptive_non_gating = (
        adaptive_only.get("paper_algorithm_claim") is False
        and adaptive_only.get("fixed_horizon") is True
        and adaptive_only.get("iterations") == fixed_horizon
        and adaptive_only.get("restart_count") == 0
        and adaptive_only.get("passed") is True
        and set(adaptive_checks)
        == {
            "completed_requested_horizon",
            "policy_event_cadence",
            "values_finite",
        }
        and _all_checks_true(adaptive_checks)
        and "non-gating" in str(adaptive_only.get("interpretation", "")).lower()
        and "controlled" in str(adaptive_only.get("interpretation", "")).lower()
        and isinstance(adaptive_sigma.get("attempts"), int)
        and adaptive_sigma.get("attempts") > 0
        and _summary_policy_cadence_valid(
            adaptive_only,
            interval=int(config["policy_check_interval"]),
        )
    )
    add_check(
        checks,
        "adaptive_without_restart_is_explicitly_nonpaper_and_nongating",
        adaptive_non_gating,
        str(adaptive_only.get("interpretation", "missing interpretation")),
    )

    matrix_checks = _mapping(base.get("ablation_checks"))
    add_check(
        checks,
        "recorded_ablation_matrix_checks_pass",
        (
            base.get("ablation_passed") is True
            and _all_checks_true(matrix_checks)
            and set(matrix_checks)
            == {
                "four_control_combinations_present",
                "unscaled_ruiz_and_full_present",
                "all_fixed_horizon_runs_complete",
                "unscaled_structural_baseline_preserved",
                "fixed_no_restart_converges",
                "fixed_restart_converges",
                "adaptive_restart_converges",
                "adaptive_no_restart_policy_exercised",
                "initial_sigma_sensitivity_converges",
            }
        ),
        f"matrix_checks={matrix_checks}",
    )

    sensitivity = [_mapping(run) for run in _sequence(base.get("initial_sigma_sensitivity"))]
    by_initial = {
        float(_mapping(run.get("sigma")).get("initial")): run
        for run in sensitivity
        if _finite_positive(_mapping(run.get("sigma")).get("initial"))
    }
    expected_sigmas = {
        float(value) for value in _sequence(config.get("sensitivity_initial_sigmas"))
    }
    sensitivity_valid = set(by_initial) == expected_sigmas and len(sensitivity) == len(
        expected_sigmas
    )
    for initial_sigma, run in by_initial.items():
        control = _mapping(run.get("control"))
        sensitivity_valid = sensitivity_valid and (
            _original_space_acceptance(run, config=config)
            and control.get("adaptive_sigma") is True
            and control.get("restart") is True
            and _numbers_close(_mapping(run.get("sigma")).get("initial"), initial_sigma)
        )
    add_check(
        checks,
        "initial_sigma_sensitivity_coverage_and_acceptance",
        sensitivity_valid,
        f"initial_sigmas={sorted(by_initial)}",
    )


def _expected_run_summaries(
    cases: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for case_name, case in cases.items():
        main = _mapping(case.get("main_full_adaptive_restart"))
        expected[(case_name, "full_adaptive_restart", "convergence")] = main

    base_name = "case5_base_t1"
    base = _mapping(cases.get(base_name))
    for run in _sequence(base.get("preprocessing_ablation")):
        summary = _mapping(run)
        label = str(summary.get("preconditioning"))
        run_name = PREPROCESSING_RUN_NAMES.get(label)
        if run_name is not None:
            expected[
                (
                    base_name,
                    run_name,
                    "fixed_horizon_preprocessing_ablation",
                )
            ] = summary

    control_names = {
        (False, False): (
            "full_fixed_no_restart",
            "convergence_control_ablation",
        ),
        (True, False): (
            "full_adaptive_no_restart",
            "fixed_horizon_control_ablation",
        ),
        (False, True): (
            "full_fixed_restart",
            "convergence_control_ablation",
        ),
    }
    for run in _sequence(base.get("control_ablation")):
        summary = _mapping(run)
        control = _mapping(summary.get("control"))
        key = (
            control.get("adaptive_sigma") is True,
            control.get("restart") is True,
        )
        name_and_phase = control_names.get(key)
        if name_and_phase is not None:
            expected[(base_name, *name_and_phase)] = summary

    for run in _sequence(base.get("initial_sigma_sensitivity")):
        summary = _mapping(run)
        initial = _mapping(summary.get("sigma")).get("initial")
        if _finite_positive(initial):
            expected[
                (
                    base_name,
                    f"full_adaptive_restart_sigma_{float(initial):g}",
                    "initial_sigma_sensitivity",
                )
            ] = summary
    return expected


def _residual_snapshot_valid(value: Any) -> bool:
    residual = _mapping(value)
    raw = _mapping(residual.get("paper_raw"))
    normalized = _mapping(residual.get("paper_normalized"))
    return (
        set(raw) == PAPER_RESIDUAL_COMPONENTS
        and set(normalized) == PAPER_RESIDUAL_COMPONENTS
        and _finite_nonnegative(residual.get("kkt_combined_norm"))
        and _finite_nonnegative(residual.get("normalized_combined_norm"))
        and all(_finite_nonnegative(item) for item in raw.values())
        and all(_finite_nonnegative(item) for item in normalized.values())
        and isinstance(residual.get("paper_stopping_satisfied"), bool)
    )


def _trajectory_row_valid(row: dict[str, Any]) -> bool:
    required = {
        "record_type",
        "case",
        "run",
        "phase",
        "iteration",
        "inner_iteration",
        "iteration_loop_elapsed_seconds",
        "scaled_variable_objective",
        "original_variable_objective",
        "scaled_residuals",
        "original_residuals",
        "kkt_target_satisfied",
        "maximum_equality_solve_relative_residual",
        "maximum_equality_solve_infinity_residual",
        "minimum_original_inequality_multiplier",
        "sigma",
        "restart_count",
    }
    minimum_multiplier = row.get("minimum_original_inequality_multiplier")
    return (
        required.issubset(row)
        and row.get("record_type") == "trajectory"
        and isinstance(row.get("iteration"), int)
        and row.get("iteration") > 0
        and isinstance(row.get("inner_iteration"), int)
        and row.get("inner_iteration") > 0
        and _finite_nonnegative(row.get("iteration_loop_elapsed_seconds"))
        and _finite_number(row.get("scaled_variable_objective"))
        and _finite_number(row.get("original_variable_objective"))
        and _residual_snapshot_valid(row.get("scaled_residuals"))
        and _residual_snapshot_valid(row.get("original_residuals"))
        and isinstance(row.get("kkt_target_satisfied"), bool)
        and _finite_nonnegative(row.get("maximum_equality_solve_relative_residual"))
        and _finite_nonnegative(row.get("maximum_equality_solve_infinity_residual"))
        and (minimum_multiplier is None or _finite_number(minimum_multiplier))
        and _finite_positive(row.get("sigma"))
        and isinstance(row.get("restart_count"), int)
        and row.get("restart_count") >= 0
    )


def _policy_row_valid(
    row: dict[str, Any],
    *,
    summary: dict[str, Any],
    interval: int,
) -> bool:
    required = {
        "record_type",
        "case",
        "run",
        "phase",
        "iteration",
        "inner_iteration",
        "merit",
        "reference_merit",
        "previous_checkpoint_merit",
        "restart_reasons",
        "restarted",
        "restart_count",
        "sigma_update",
    }
    update = _mapping(row.get("sigma_update"))
    reasons = _sequence(row.get("restart_reasons"))
    control = _mapping(summary.get("control"))
    attempted = update.get("attempted")
    accepted = update.get("accepted")
    restarted = row.get("restarted")
    ratio = update.get("infeasibility_ratio")
    previous_merit = row.get("previous_checkpoint_merit")
    adaptive = control.get("adaptive_sigma") is True
    restart_enabled = control.get("restart") is True
    return (
        required.issubset(row)
        and row.get("record_type") == "policy_event"
        and isinstance(row.get("iteration"), int)
        and row.get("iteration") > 0
        and row.get("iteration") % interval == 0
        and isinstance(row.get("inner_iteration"), int)
        and row.get("inner_iteration") > 0
        and _finite_nonnegative(row.get("merit"))
        and _finite_nonnegative(row.get("reference_merit"))
        and (previous_merit is None or _finite_nonnegative(previous_merit))
        and all(
            reason
            in {
                "forced_first",
                "sufficient_decay",
                "necessary_decay_no_local_progress",
                "long_inner_loop",
            }
            for reason in reasons
        )
        and isinstance(restarted, bool)
        and restarted is bool(reasons)
        and (restart_enabled or not restarted)
        and isinstance(row.get("restart_count"), int)
        and row.get("restart_count") >= 0
        and isinstance(attempted, bool)
        and isinstance(accepted, bool)
        and (attempted or not accepted)
        and attempted is (adaptive and (restarted or not restart_enabled))
        and _finite_nonnegative(update.get("delta_x"))
        and _finite_nonnegative(update.get("delta_y"))
        and _finite_nonnegative(update.get("primal_infeasibility"))
        and _finite_nonnegative(update.get("dual_infeasibility"))
        and (ratio is None or _finite_nonnegative(ratio))
        and _finite_positive(update.get("sigma_before"))
        and _finite_positive(update.get("sigma_after"))
        and isinstance(update.get("reason"), str)
        and bool(update.get("reason"))
    )


def _final_row_matches_summary(
    row: dict[str, Any],
    summary: dict[str, Any],
) -> bool:
    original = _mapping(row.get("original_residuals"))
    summary_residuals = _mapping(summary.get("original_residuals"))
    physical = _mapping(summary.get("physical_validation"))
    event_iterations = _sequence(summary.get("policy_event_iterations"))
    final_policy_update = bool(event_iterations) and event_iterations[-1] == summary.get(
        "iterations"
    )
    sigma_matches = (
        _finite_positive(row.get("sigma"))
        if final_policy_update
        else _numbers_close(
            row.get("sigma"),
            _mapping(summary.get("sigma")).get("final"),
        )
    )
    return (
        row.get("iteration") == summary.get("iterations")
        and _numbers_close(
            original.get("kkt_combined_norm"),
            summary_residuals.get("kkt_combined_norm"),
        )
        and _numbers_close(
            original.get("normalized_combined_norm"),
            summary_residuals.get("paper_normalized_combined_norm"),
        )
        and _mapping(original.get("paper_raw"))
        == _mapping(summary_residuals.get("paper_raw_norms"))
        and _mapping(original.get("paper_normalized"))
        == _mapping(summary_residuals.get("paper_normalized_norms"))
        and original.get("paper_stopping_satisfied")
        is _mapping(summary_residuals.get("paper_stopping")).get("all_satisfied")
        and _numbers_close(
            row.get("original_variable_objective"),
            physical.get("variable_objective"),
        )
        # A fixed-horizon run can update sigma after recording the last
        # proximal state.  Its final policy event is cross-checked separately.
        and sigma_matches
        and row.get("restart_count") == summary.get("restart_count")
        and _numbers_close(
            row.get("maximum_equality_solve_relative_residual"),
            summary.get("maximum_equality_solve_relative_residual"),
        )
        and _numbers_close(
            row.get("maximum_equality_solve_infinity_residual"),
            summary.get("maximum_equality_solve_infinity_residual"),
        )
    )


def _gzip_consistency_checks(
    checks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    load_error: str | None,
    cases: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    expected = _expected_run_summaries(cases)
    trajectories: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    events: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    rows_valid = load_error is None and bool(rows)
    for row in rows:
        key = (
            str(row.get("case")),
            str(row.get("run")),
            str(row.get("phase")),
        )
        record_type = row.get("record_type")
        if record_type == "trajectory":
            trajectories[key].append(row)
            rows_valid = rows_valid and _trajectory_row_valid(row)
        elif record_type == "policy_event":
            events[key].append(row)
        else:
            rows_valid = False

    coverage_valid = (
        rows_valid and set(trajectories) == set(expected) and set(events).issubset(expected)
    )
    add_check(
        checks,
        "trajectory_gzip_is_valid_and_covers_every_recorded_run",
        coverage_valid,
        (
            f"rows={len(rows)}, trajectory_groups={len(trajectories)}, event_groups={len(events)}"
            if load_error is None
            else load_error
        ),
    )

    interval = int(config.get("policy_check_interval", -1))
    history_interval = int(config.get("history_interval", -1))
    trajectories_consistent = coverage_valid
    policies_consistent = coverage_valid
    group_details: list[str] = []
    for key, summary in expected.items():
        trajectory_rows = trajectories.get(key, [])
        event_rows = events.get(key, [])
        trajectory_iterations = [row.get("iteration") for row in trajectory_rows]
        event_iterations = [row.get("iteration") for row in event_rows]
        summary_event_iterations = _sequence(summary.get("policy_event_iterations"))
        iterations = summary.get("iterations")

        expected_history: set[int] = {1}
        if isinstance(iterations, int) and iterations > 0:
            expected_history.add(iterations)
            expected_history.update(range(history_interval, iterations + 1, history_interval))
        expected_history.update(
            value for value in summary_event_iterations if isinstance(value, int)
        )
        trajectory_group_valid = (
            bool(trajectory_rows)
            and trajectory_iterations == sorted(set(trajectory_iterations))
            and trajectory_iterations == sorted(expected_history)
            and all(
                (
                    row.get("case"),
                    row.get("run"),
                    row.get("phase"),
                )
                == key
                for row in trajectory_rows
            )
            and _final_row_matches_summary(trajectory_rows[-1], summary)
        )
        trajectories_consistent = trajectories_consistent and trajectory_group_valid

        policy_group_valid = (
            event_iterations == summary_event_iterations
            and event_iterations == sorted(set(event_iterations))
            and len(event_rows) == summary.get("policy_event_count")
            and all(
                (
                    row.get("case"),
                    row.get("run"),
                    row.get("phase"),
                )
                == key
                and _policy_row_valid(
                    row,
                    summary=summary,
                    interval=interval,
                )
                for row in event_rows
            )
            and _summary_policy_cadence_valid(
                summary,
                interval=interval,
            )
        )

        sigma = _mapping(summary.get("sigma"))
        attempts = sum(
            _mapping(row.get("sigma_update")).get("attempted") is True for row in event_rows
        )
        accepted = sum(
            _mapping(row.get("sigma_update")).get("accepted") is True for row in event_rows
        )
        restarted = sum(row.get("restarted") is True for row in event_rows)
        reason_counts = {
            reason: sum(reason in _sequence(row.get("restart_reasons")) for row in event_rows)
            for reason in (
                "forced_first",
                "sufficient_decay",
                "necessary_decay_no_local_progress",
                "long_inner_loop",
            )
        }
        policy_group_valid = policy_group_valid and (
            attempts == sigma.get("attempts")
            and accepted == sigma.get("accepted")
            and restarted == summary.get("restart_count")
            and reason_counts == _mapping(summary.get("restart_reason_counts"))
        )
        if event_rows:
            restart_counts = [row.get("restart_count") for row in event_rows]
            policy_group_valid = policy_group_valid and (
                restart_counts == sorted(restart_counts)
                and restart_counts[-1] == summary.get("restart_count")
                and _numbers_close(
                    _mapping(event_rows[-1].get("sigma_update")).get("sigma_after"),
                    sigma.get("final"),
                )
            )
        policies_consistent = policies_consistent and policy_group_valid
        group_details.append(
            f"{'/'.join(key)}:trajectory={len(trajectory_rows)}, events={len(event_rows)}"
        )

    add_check(
        checks,
        "trajectory_rows_match_json_summaries_and_sampling_grid",
        trajectories_consistent,
        "; ".join(group_details),
    )
    add_check(
        checks,
        "policy_events_match_json_counts_values_and_100_iteration_cadence",
        policies_consistent,
        f"policy_rows={sum(len(group) for group in events.values())}",
    )


def _stage_boundary_checks(
    checks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> None:
    boundary = _mapping(evidence.get("stage_boundary"))
    unsupported = _mapping(config.get("unsupported_or_explicitly_bounded"))
    premature_paths = [
        path
        for path in (
            "scripts/run_stage_6.py",
            "src/gpu_dcopf_hpr/gpu_solver.py",
            "src/gpu_dcopf_hpr/dgx_runner.py",
        )
        if (PROJECT_ROOT / path).exists()
    ]
    boundary_valid = (
        evidence.get("all_passed") is True
        and _mapping(evidence.get("component_validation")).get("passed") is True
        and _mapping(evidence.get("dcopf_ablation")).get("passed") is True
        and boundary
        == {
            "stage_5_complete": True,
            "stage_6_started": False,
            "gpu_code_executed": False,
            "dgx_executed": False,
        }
        and unsupported.get("gpu") is False
        and unsupported.get("dgx_execution") is False
        and unsupported.get("scaled_structural_eq55_backend_available") is False
        and not premature_paths
    )
    add_check(
        checks,
        "stage_six_gpu_and_dgx_boundaries_remain_closed",
        boundary_valid,
        (
            "Stage 5 complete; Stage 6, GPU code, and DGX execution false"
            if not premature_paths
            else f"premature_paths={premature_paths}"
        ),
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
        evidence_files.get(
            "trajectories_and_policy_events",
            "stage_5_trajectories.jsonl.gz",
        )
    )

    required_paths = (
        "data/raw/matpower/case5.m",
        "configs/dcopf/case5_base_stage_2.json",
        "configs/dcopf/case5_synthetic_extension_stage_2.json",
        "configs/sgs_hpr/stage_5_preconditioning_controls.json",
        "src/gpu_dcopf_hpr/preconditioning.py",
        "src/gpu_dcopf_hpr/stage5_control.py",
        "scripts/run_stage_5.py",
        "scripts/check_stage_5.py",
        "tests/unit/test_preconditioning.py",
        "tests/unit/test_stage5_control.py",
        "tests/integration/test_stage5_controls.py",
        "tests/integration/test_stage5_evidence.py",
    )
    missing = [path for path in required_paths if not (PROJECT_ROOT / path).is_file()]
    for path in (config_path, evidence_path, trajectory_path):
        if not path.is_file():
            missing.append(str(path))
    add_check(
        checks,
        "required_stage_five_paths",
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

    _configuration_and_source_checks(checks, evidence, config)
    _component_checks(checks, evidence, config)
    dcopf_cases = _dcopf_case_checks(checks, evidence, config)
    _ablation_checks(checks, dcopf_cases, config)
    rows, rows_error = _load_gzip_jsonl(trajectory_path)
    _gzip_consistency_checks(
        checks,
        rows,
        rows_error,
        dcopf_cases,
        config,
    )
    _stage_boundary_checks(checks, evidence, config)

    totals: Counter[str] = Counter()
    for check in checks:
        totals["passed" if check["passed"] else "failed"] += 1
    return {
        "stage": 5,
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
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Versioned Stage 5 configuration to inspect.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Stage 5 validation JSON to inspect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the independent check summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_checks(
        args.evidence.resolve(),
        args.config.resolve(),
    )
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
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
