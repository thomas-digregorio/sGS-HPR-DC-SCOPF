"""Run and preserve the complete Stage 5 preprocessing/control validation.

This remains an FP64 CPU reproduction.  It validates the reversible Ruiz,
Pock--Chambolle, and norm transforms, preserves the Stage 4 unscaled
structural baseline, evaluates the required control/preprocessing ablations,
and validates all reported DCOPF solutions in recovered original coordinates.

The DCOPF manuscript does not publish its exact adaptive-policy code.  The
implemented policy is therefore an explicit transfer of published HPR-LP
restart and penalty equations.  The adaptive-without-restart run is a
controlled ablation at the manuscript's 100-iteration check cadence, not a
claim that it is a standalone paper algorithm.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
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

from gpu_dcopf_hpr.canonical_lp import CanonicalLP  # noqa: E402
from gpu_dcopf_hpr.dcopf_model import (  # noqa: E402
    build_dcopf_model,
    load_dcopf_config,
)
from gpu_dcopf_hpr.hpr_generic import HPRState  # noqa: E402
from gpu_dcopf_hpr.network_data import load_matpower_case  # noqa: E402
from gpu_dcopf_hpr.preconditioning import (  # noqa: E402
    LPPreconditioner,
    precondition_lp,
)
from gpu_dcopf_hpr.stage5_control import (  # noqa: E402
    Stage5Control,
    Stage5SGSHPRResult,
    solve_stage5_sgs_hpr,
)
from gpu_dcopf_hpr.structural_y1 import prepare_dcopf_structural_y1  # noqa: E402
from gpu_dcopf_hpr.toy_problems import (  # noqa: E402
    analytic_toy_case,
    planted_random_case,
)
from gpu_dcopf_hpr.validation import (  # noqa: E402
    maximum_primal_violation,
    solve_with_highs,
    validate_dcopf_candidate,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sgs_hpr" / "stage_5_preconditioning_controls.json"
DEFAULT_NETWORK = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
DEFAULT_DCOPF_CONFIGS = (
    PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json",
    PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_5"
PAPER_PATH = Path.home() / "Downloads" / "AnEfficientGPU-basedHalpernAccelerating.pdf"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
    return parser.parse_args()


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(1.0, float(np.linalg.norm(reference))))


def _scaled_gap(candidate: float, reference: float) -> float:
    return abs(candidate - reference) / max(1.0, abs(reference))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_parent_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _preconditioner_summary(value: LPPreconditioner) -> dict[str, Any]:
    diagnostics = value.diagnostics
    return {
        "ruiz_iterations": diagnostics.ruiz_iterations,
        "pock_chambolle_applied": diagnostics.pock_chambolle_applied,
        "normalization_applied": diagnostics.normalization_applied,
        "original_nnz": diagnostics.original_nnz,
        "scaled_nnz": diagnostics.scaled_nnz,
        "nnz_preserved": diagnostics.nnz_preserved,
        "diagonally_scaled_b_norm": diagnostics.b_norm,
        "diagonally_scaled_c_norm": diagnostics.c_norm,
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
        "steps": [
            {
                "method": step.method,
                "iteration": step.iteration,
                "norm": step.norm,
                "row_zero_count_before": step.row_before.zero_count,
                "column_zero_count_before": step.column_before.zero_count,
                "row_range_before": [step.row_before.minimum, step.row_before.maximum],
                "column_range_before": [
                    step.column_before.minimum,
                    step.column_before.maximum,
                ],
                "row_range_after": [step.row_after.minimum, step.row_after.maximum],
                "column_range_after": [
                    step.column_after.minimum,
                    step.column_after.maximum,
                ],
            }
            for step in diagnostics.iterations
        ],
    }


def _component_fixture(
    *,
    name: str,
    lp: CanonicalLP,
    state: HPRState,
    config: dict[str, Any],
) -> dict[str, Any]:
    transformed = precondition_lp(
        lp,
        ruiz_iterations=int(config["ruiz_iterations"]),
        pock_chambolle=True,
        normalize=bool(config["normalize_b_and_c"]),
    )
    scaled_state = transformed.scale_state(state)
    recovered_state = transformed.recover_state(scaled_state)
    recovered_lp = transformed.recover_lp()

    original_primal = np.asarray(lp.A @ state.x, dtype=np.float64).reshape(-1) - lp.b
    scaled_primal = (
        np.asarray(
            transformed.scaled_lp.A @ scaled_state.x,
            dtype=np.float64,
        ).reshape(-1)
        - transformed.scaled_lp.b
    )
    expected_scaled_primal = original_primal / (transformed.row_denominator * transformed.b_scale)
    original_stationarity = (
        lp.c - np.asarray(lp.A.T @ state.y, dtype=np.float64).reshape(-1) - state.z
    )
    scaled_stationarity = (
        transformed.scaled_lp.c
        - np.asarray(
            transformed.scaled_lp.A.T @ scaled_state.y,
            dtype=np.float64,
        ).reshape(-1)
        - scaled_state.z
    )
    expected_scaled_stationarity = original_stationarity / (
        transformed.column_denominator * transformed.c_scale
    )
    scaled_objective = float(transformed.scaled_lp.c @ scaled_state.x)
    original_objective = float(lp.c @ state.x)
    recovered_objective = transformed.original_objective_from_scaled(scaled_objective)
    iteration_methods = [
        (step.method, step.iteration) for step in transformed.diagnostics.iterations
    ]

    errors = {
        "state_x_roundtrip": _relative_error(recovered_state.x, state.x),
        "state_y_roundtrip": _relative_error(recovered_state.y, state.y),
        "state_z_roundtrip": _relative_error(recovered_state.z, state.z),
        "matrix_roundtrip": _relative_error(
            recovered_lp.dense_A(),
            lp.dense_A(),
        ),
        "b_roundtrip": _relative_error(recovered_lp.b, lp.b),
        "c_roundtrip": _relative_error(recovered_lp.c, lp.c),
        "lower_roundtrip": _relative_error(recovered_lp.lower, lp.lower),
        "upper_roundtrip": _relative_error(recovered_lp.upper, lp.upper),
        "primal_transform_identity": _relative_error(
            scaled_primal,
            expected_scaled_primal,
        ),
        "stationarity_transform_identity": _relative_error(
            scaled_stationarity,
            expected_scaled_stationarity,
        ),
        "objective_identity": abs(recovered_objective - original_objective)
        / max(1.0, abs(original_objective)),
    }
    roundtrip_names = (
        "state_x_roundtrip",
        "state_y_roundtrip",
        "state_z_roundtrip",
        "matrix_roundtrip",
        "b_roundtrip",
        "c_roundtrip",
        "lower_roundtrip",
        "upper_roundtrip",
        "objective_identity",
    )
    identity_names = (
        "primal_transform_identity",
        "stationarity_transform_identity",
    )
    checks = {
        "ten_ruiz_iterations": (
            iteration_methods[:10] == [("ruiz", index) for index in range(1, 11)]
        ),
        "pock_chambolle_alpha_one_step": (iteration_methods[10:] == [("pock_chambolle", 1)]),
        "normalization_uses_full_vector_norms": (
            transformed.b_scale == 1.0 + float(np.linalg.norm(lp.b / transformed.row_denominator))
            and transformed.c_scale
            == 1.0 + float(np.linalg.norm(lp.c / transformed.column_denominator))
        ),
        "positive_finite_denominators": (
            np.all(np.isfinite(transformed.row_denominator))
            and np.all(transformed.row_denominator > 0.0)
            and np.all(np.isfinite(transformed.column_denominator))
            and np.all(transformed.column_denominator > 0.0)
        ),
        "sparse_nonzeros_preserved": transformed.diagnostics.nnz_preserved,
        "roundtrip_within_tolerance": (
            max(errors[name] for name in roundtrip_names)
            <= float(config["transform_roundtrip_tolerance"])
        ),
        "algebraic_identities_within_tolerance": (
            max(errors[name] for name in identity_names)
            <= float(config["transform_identity_tolerance"])
        ),
    }
    return {
        "name": name,
        "dimensions": {"m1": lp.m1, "m2": lp.m2, "n": lp.n},
        "source_matrix_sparse": sparse.issparse(lp.A1) or sparse.issparse(lp.A2),
        "preconditioner": _preconditioner_summary(transformed),
        "errors": errors,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_component_validation(config: dict[str, Any]) -> dict[str, Any]:
    dense = analytic_toy_case()
    planted = planted_random_case()
    sparse_lp = CanonicalLP(
        c=planted.lp.c,
        A1=sparse.csr_matrix(planted.lp.A1),
        b1=planted.lp.b1,
        A2=sparse.csr_matrix(planted.lp.A2),
        b2=planted.lp.b2,
        lower=planted.lp.lower,
        upper=planted.lp.upper,
    )
    cases = [
        _component_fixture(
            name="dense_analytic",
            lp=dense.lp,
            state=dense.expected_state,
            config=config,
        ),
        _component_fixture(
            name="sparse_planted_random",
            lp=sparse_lp,
            state=planted.expected_state,
            config=config,
        ),
    ]
    return {"cases": cases, "passed": all(case["passed"] for case in cases)}


def _control(
    config: dict[str, Any],
    *,
    adaptive_sigma: bool,
    restart: bool,
) -> Stage5Control:
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


def _solve(
    lp: CanonicalLP,
    config: dict[str, Any],
    *,
    preconditioner: LPPreconditioner | None,
    control: Stage5Control,
    sigma: float,
    fixed_horizon: bool,
    structural_y1: Any = None,
) -> Stage5SGSHPRResult:
    return solve_stage5_sgs_hpr(
        lp,
        sigma=sigma,
        tolerance=(1e-20 if fixed_horizon else float(config["paper_tolerance"])),
        kkt_tolerance=(None if fixed_horizon else float(config["dcopf_kkt_combined_target"])),
        max_iterations=(
            int(config["fixed_horizon_iterations"])
            if fixed_horizon
            else int(config["maximum_iterations"])
        ),
        history_interval=int(config["history_interval"]),
        structural_y1=structural_y1,
        preconditioner=preconditioner,
        control=control,
    )


def _write_result_rows(
    stream: TextIO,
    *,
    case_name: str,
    run_name: str,
    phase: str,
    result: Stage5SGSHPRResult,
) -> None:
    for entry in result.history:
        stream.write(
            json.dumps(
                {
                    "record_type": "trajectory",
                    "case": case_name,
                    "run": run_name,
                    "phase": phase,
                    **entry.as_dict(),
                },
                default=_json_default,
                sort_keys=True,
            )
            + "\n"
        )
    for event in result.policy_events:
        stream.write(
            json.dumps(
                {
                    "record_type": "policy_event",
                    "case": case_name,
                    "run": run_name,
                    "phase": phase,
                    **event.as_dict(),
                },
                default=_json_default,
                sort_keys=True,
            )
            + "\n"
        )


def _run_summary(
    *,
    result: Stage5SGSHPRResult,
    model: Any,
    reference_objective: float,
    config: dict[str, Any],
    fixed_horizon: bool,
    preconditioning_name: str,
) -> dict[str, Any]:
    objective = model.objective(result.solution.x)
    physical = validate_dcopf_candidate(
        model,
        result.solution.x,
        tolerance=float(config["dcopf_physical_tolerance"]),
    )
    primal_violation = maximum_primal_violation(model.lp, result.solution)
    event_iterations = [event.iteration for event in result.policy_events]
    event_cadence_valid = all(
        iteration % int(config["policy_check_interval"]) == 0 for iteration in event_iterations
    )
    sigma_attempts = sum(event.sigma_update.attempted for event in result.policy_events)
    sigma_accepts = sum(event.sigma_update.accepted for event in result.policy_events)
    summary = {
        "fixed_horizon": fixed_horizon,
        "preconditioning": preconditioning_name,
        "control": result.control.summary(),
        "backend": result.workspace.equality_backend,
        "iterations": result.iterations,
        "converged": result.converged,
        "total_elapsed_seconds": result.total_elapsed_seconds,
        "preparation_elapsed_seconds": result.preparation_elapsed_seconds,
        "objective": objective,
        "reference_objective": reference_objective,
        "scaled_objective_gap_to_highs": _scaled_gap(
            objective,
            reference_objective,
        ),
        "original_residuals": result.residuals.summary(),
        "scaled_residuals": result.scaled_residuals.summary(),
        "maximum_canonical_primal_violation": primal_violation,
        "maximum_physical_violation": max(family.maximum_violation for family in physical.families),
        "physical_validation": physical.summary(),
        "sigma": {
            "initial": result.initial_sigma,
            "final": result.sigma,
            "minimum": result.minimum_sigma,
            "maximum": result.maximum_sigma,
            "attempts": sigma_attempts,
            "accepted": sigma_accepts,
        },
        "restart_count": result.restart_count,
        "policy_event_count": len(result.policy_events),
        "policy_event_iterations": event_iterations,
        "policy_event_cadence_valid": event_cadence_valid,
        "restart_reason_counts": {
            reason: sum(reason in event.restart_reasons for event in result.policy_events)
            for reason in (
                "forced_first",
                "sufficient_decay",
                "necessary_decay_no_local_progress",
                "long_inner_loop",
            )
        },
        "maximum_equality_solve_relative_residual": (
            result.maximum_equality_solve_relative_residual
        ),
        "maximum_equality_solve_infinity_residual": (
            result.maximum_equality_solve_infinity_residual
        ),
        "maximum_z_x_identity_error": result.maximum_z_x_identity_error,
    }
    if fixed_horizon:
        summary["checks"] = {
            "completed_requested_horizon": (
                result.iterations == int(config["fixed_horizon_iterations"])
            ),
            "values_finite": all(
                np.all(np.isfinite(vector))
                for vector in (
                    result.solution.x,
                    result.solution.y,
                    result.solution.z,
                )
            ),
            "policy_event_cadence": event_cadence_valid,
        }
    else:
        summary["checks"] = {
            "converged": result.converged,
            "original_paper_stopping_satisfied": (result.residuals.conditions.all_satisfied),
            "original_kkt_target_satisfied": (
                result.residuals.combined_norm <= float(config["dcopf_kkt_combined_target"])
            ),
            "objective_matches_highs": (
                summary["scaled_objective_gap_to_highs"]
                <= float(config["dcopf_maximum_scaled_objective_gap"])
            ),
            "physical_candidate_valid": physical.passed,
            "canonical_primal_violation": (
                primal_violation <= float(config["dcopf_physical_tolerance"])
            ),
            "equality_solves_accurate": (
                result.maximum_equality_solve_infinity_residual
                <= float(config["maximum_equality_infinity_residual"])
            ),
            "z_x_identity_accurate": (
                result.maximum_z_x_identity_error <= float(config["maximum_z_x_identity_error"])
            ),
            "policy_event_cadence": event_cadence_valid,
            "values_finite": all(
                np.all(np.isfinite(vector))
                for vector in (
                    result.solution.x,
                    result.solution.y,
                    result.solution.z,
                )
            ),
        }
    summary["passed"] = all(summary["checks"].values())
    return summary


def _execute_run(
    *,
    stream: TextIO,
    case_name: str,
    run_name: str,
    phase: str,
    model: Any,
    reference_objective: float,
    config: dict[str, Any],
    preconditioner: LPPreconditioner | None,
    preconditioning_name: str,
    control: Stage5Control,
    sigma: float,
    fixed_horizon: bool,
    structural_y1: Any = None,
) -> tuple[dict[str, Any], Stage5SGSHPRResult]:
    started = perf_counter()
    result = _solve(
        model.lp,
        config,
        preconditioner=preconditioner,
        control=control,
        sigma=sigma,
        fixed_horizon=fixed_horizon,
        structural_y1=structural_y1,
    )
    elapsed = perf_counter() - started
    _write_result_rows(
        stream,
        case_name=case_name,
        run_name=run_name,
        phase=phase,
        result=result,
    )
    summary = _run_summary(
        result=result,
        model=model,
        reference_objective=reference_objective,
        config=config,
        fixed_horizon=fixed_horizon,
        preconditioning_name=preconditioning_name,
    )
    summary["runner_wall_seconds"] = elapsed
    return summary, result


def _run_dcopf_ablation(
    *,
    config: dict[str, Any],
    network_path: Path,
    dcopf_paths: tuple[Path, ...],
    stream: TextIO,
) -> dict[str, Any]:
    network = load_matpower_case(network_path)
    case_summaries: list[dict[str, Any]] = []
    full_main_passed = True

    for case_index, dcopf_path in enumerate(dcopf_paths):
        dcopf_config = load_dcopf_config(dcopf_path, network)
        model = build_dcopf_model(network, dcopf_config)
        highs = solve_with_highs(
            model.lp,
            tolerance=float(config["paper_tolerance"]),
        )
        reference_objective = model.objective(highs.state.x)
        full = precondition_lp(
            model.lp,
            ruiz_iterations=int(config["ruiz_iterations"]),
            pock_chambolle=True,
            normalize=bool(config["normalize_b_and_c"]),
        )
        full_main, _ = _execute_run(
            stream=stream,
            case_name=dcopf_config.name,
            run_name="full_adaptive_restart",
            phase="convergence",
            model=model,
            reference_objective=reference_objective,
            config=config,
            preconditioner=full,
            preconditioning_name="10 Ruiz + Pock-Chambolle alpha=1 + norm",
            control=_control(config, adaptive_sigma=True, restart=True),
            sigma=float(config["initial_sigma"]),
            fixed_horizon=False,
        )
        full_main["checks"]["adaptive_sigma_exercised"] = (
            full_main["sigma"]["attempts"] > 0
            and full_main["sigma"]["accepted"] > 0
            and full_main["sigma"]["minimum"] > 0.0
        )
        full_main["checks"]["restart_exercised"] = full_main["restart_count"] > 0
        full_main["passed"] = all(full_main["checks"].values())
        full_main_passed = full_main_passed and full_main["passed"]

        case_summary: dict[str, Any] = {
            "name": dcopf_config.name,
            "classification": dcopf_config.classification,
            "dimensions": model.dimension_summary(),
            "highs": {
                **highs.summary(),
                "total_objective": reference_objective,
            },
            "full_preconditioner": _preconditioner_summary(full),
            "main_full_adaptive_restart": full_main,
        }

        if case_index == 0:
            structural = prepare_dcopf_structural_y1(model)
            normalized = precondition_lp(
                model.lp,
                ruiz_iterations=0,
                pock_chambolle=False,
                normalize=True,
            )
            ruiz = precondition_lp(
                model.lp,
                ruiz_iterations=int(config["ruiz_iterations"]),
                pock_chambolle=False,
                normalize=True,
            )
            preprocessing_runs: list[dict[str, Any]] = []
            for run_name, value, label, structural_solver in (
                (
                    "unscaled_fixed_no_restart",
                    None,
                    "unscaled Stage 4 structural baseline",
                    structural,
                ),
                (
                    "normalized_fixed_no_restart",
                    normalized,
                    "norm b/c only",
                    None,
                ),
                (
                    "ruiz_fixed_no_restart",
                    ruiz,
                    "10 Ruiz + norm",
                    None,
                ),
                (
                    "full_fixed_no_restart",
                    full,
                    "10 Ruiz + Pock-Chambolle alpha=1 + norm",
                    None,
                ),
            ):
                summary, _ = _execute_run(
                    stream=stream,
                    case_name=dcopf_config.name,
                    run_name=run_name,
                    phase="fixed_horizon_preprocessing_ablation",
                    model=model,
                    reference_objective=reference_objective,
                    config=config,
                    preconditioner=value,
                    preconditioning_name=label,
                    control=_control(
                        config,
                        adaptive_sigma=False,
                        restart=False,
                    ),
                    sigma=float(config["initial_sigma"]),
                    fixed_horizon=True,
                    structural_y1=structural_solver,
                )
                preprocessing_runs.append(summary)

            control_runs: list[dict[str, Any]] = []
            fixed, _ = _execute_run(
                stream=stream,
                case_name=dcopf_config.name,
                run_name="full_fixed_no_restart",
                phase="convergence_control_ablation",
                model=model,
                reference_objective=reference_objective,
                config=config,
                preconditioner=full,
                preconditioning_name="10 Ruiz + Pock-Chambolle alpha=1 + norm",
                control=_control(
                    config,
                    adaptive_sigma=False,
                    restart=False,
                ),
                sigma=float(config["initial_sigma"]),
                fixed_horizon=False,
            )
            control_runs.append(fixed)

            adaptive_only, _ = _execute_run(
                stream=stream,
                case_name=dcopf_config.name,
                run_name="full_adaptive_no_restart",
                phase="fixed_horizon_control_ablation",
                model=model,
                reference_objective=reference_objective,
                config=config,
                preconditioner=full,
                preconditioning_name="10 Ruiz + Pock-Chambolle alpha=1 + norm",
                control=_control(
                    config,
                    adaptive_sigma=True,
                    restart=False,
                ),
                sigma=float(config["initial_sigma"]),
                fixed_horizon=True,
            )
            adaptive_only["paper_algorithm_claim"] = False
            adaptive_only["interpretation"] = (
                "Controlled decoupling at the 100-iteration policy cadence; "
                "non-convergence is informative and non-gating."
            )
            control_runs.append(adaptive_only)

            restart_only, _ = _execute_run(
                stream=stream,
                case_name=dcopf_config.name,
                run_name="full_fixed_restart",
                phase="convergence_control_ablation",
                model=model,
                reference_objective=reference_objective,
                config=config,
                preconditioner=full,
                preconditioning_name="10 Ruiz + Pock-Chambolle alpha=1 + norm",
                control=_control(
                    config,
                    adaptive_sigma=False,
                    restart=True,
                ),
                sigma=float(config["initial_sigma"]),
                fixed_horizon=False,
            )
            control_runs.append(restart_only)
            control_runs.append(full_main)

            sensitivity_runs: list[dict[str, Any]] = []
            for initial_sigma in config["sensitivity_initial_sigmas"]:
                sensitivity, _ = _execute_run(
                    stream=stream,
                    case_name=dcopf_config.name,
                    run_name=f"full_adaptive_restart_sigma_{initial_sigma:g}",
                    phase="initial_sigma_sensitivity",
                    model=model,
                    reference_objective=reference_objective,
                    config=config,
                    preconditioner=full,
                    preconditioning_name=("10 Ruiz + Pock-Chambolle alpha=1 + norm"),
                    control=_control(
                        config,
                        adaptive_sigma=True,
                        restart=True,
                    ),
                    sigma=float(initial_sigma),
                    fixed_horizon=False,
                )
                sensitivity_runs.append(sensitivity)

            required_controls = {
                (
                    bool(run["control"]["adaptive_sigma"]),
                    bool(run["control"]["restart"]),
                )
                for run in control_runs
            }
            preprocessing_labels = {run["preconditioning"] for run in preprocessing_runs}
            matrix_checks = {
                "four_control_combinations_present": required_controls
                == {
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                },
                "unscaled_ruiz_and_full_present": {
                    "unscaled Stage 4 structural baseline",
                    "10 Ruiz + norm",
                    "10 Ruiz + Pock-Chambolle alpha=1 + norm",
                }.issubset(preprocessing_labels),
                "all_fixed_horizon_runs_complete": all(run["passed"] for run in preprocessing_runs)
                and adaptive_only["passed"],
                "unscaled_structural_baseline_preserved": (
                    preprocessing_runs[0]["backend"] == "structural"
                ),
                "fixed_no_restart_converges": fixed["passed"],
                "fixed_restart_converges": restart_only["passed"],
                "adaptive_restart_converges": full_main["passed"],
                "adaptive_no_restart_policy_exercised": (
                    adaptive_only["sigma"]["attempts"] > 0
                    and adaptive_only["policy_event_count"]
                    == int(config["fixed_horizon_iterations"])
                    // int(config["policy_check_interval"])
                ),
                "initial_sigma_sensitivity_converges": all(
                    run["passed"] for run in sensitivity_runs
                ),
            }
            case_summary["preprocessing_ablation"] = preprocessing_runs
            case_summary["control_ablation"] = control_runs
            case_summary["initial_sigma_sensitivity"] = sensitivity_runs
            case_summary["ablation_checks"] = matrix_checks
            case_summary["ablation_passed"] = all(matrix_checks.values())
            full_main_passed = full_main_passed and case_summary["ablation_passed"]

        case_summary["passed"] = full_main["passed"] and (case_summary.get("ablation_passed", True))
        case_summaries.append(case_summary)

    return {
        "cases": case_summaries,
        "passed": full_main_passed and all(case["passed"] for case in case_summaries),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_dir / "stage_5_trajectories.jsonl.gz"
    validation_path = output_dir / "stage_5_validation.json"

    component_validation = _run_component_validation(config)
    with gzip.open(trajectories_path, "wt", encoding="utf-8") as stream:
        dcopf_ablation = _run_dcopf_ablation(
            config=config,
            network_path=args.network.resolve(),
            dcopf_paths=tuple(path.resolve() for path in args.dcopf_configs),
            stream=stream,
        )

    source_audit = {
        "dcopf_manuscript_local_path_available": PAPER_PATH.is_file(),
        "dcopf_manuscript_sha256": _sha256(PAPER_PATH),
        "formula_provenance": {
            "ten_ruiz_iterations": "DCOPF manuscript numerical settings",
            "pock_chambolle_alpha_one": ("DCOPF manuscript, PDLP paper, and Pock-Chambolle source"),
            "b_c_norm_normalization": "DCOPF manuscript numerical settings",
            "restart": "HPR-LP Eqs. (10)-(12), interval changed to DCOPF value 100",
            "adaptive_sigma": "HPR-LP Eqs. (15)-(18) in the sGS metric",
        },
        "exact_author_dcopf_policy_available": False,
        "implementation_claim": (
            "Sourced HPR-LP transfer with explicit DCOPF interval, not "
            "byte-for-byte identity with unpublished author code."
        ),
        "hpr_lp_reconstruction_pin": config["sources"]["hpr_lp_reconstruction_pin"],
        "hpr_lp_current_commit_audited_for_drift": config["sources"][
            "hpr_lp_current_commit_audited_for_drift"
        ],
        "source_drift_note": (
            "The inspected HPR-LP repository commit postdates the published "
            "article and contains extra sigma heuristics; Stage 5 implements "
            "the published equations instead of those later additions."
        ),
        "sources": config["sources"],
    }
    evidence = {
        "all_passed": (component_validation["passed"] and dcopf_ablation["passed"]),
        "configuration": config,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "repository_parent_commit": _git_parent_commit(),
            "highs_interface": "scipy.optimize.linprog(method='highs-ds')",
        },
        "source_audit": source_audit,
        "component_validation": component_validation,
        "dcopf_ablation": dcopf_ablation,
        "evidence_files": {
            "trajectories_and_policy_events": trajectories_path.name,
        },
        "stage_boundary": {
            "stage_5_complete": (component_validation["passed"] and dcopf_ablation["passed"]),
            "stage_6_started": False,
            "gpu_code_executed": False,
            "dgx_executed": False,
        },
    }
    validation_path.write_text(
        json.dumps(
            evidence,
            default=_json_default,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "all_passed": evidence["all_passed"],
                "validation_file": str(validation_path),
                "trajectory_file": str(trajectories_path),
                "component_cases": len(component_validation["cases"]),
                "dcopf_cases": len(dcopf_ablation["cases"]),
            },
            default=_json_default,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if evidence["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
