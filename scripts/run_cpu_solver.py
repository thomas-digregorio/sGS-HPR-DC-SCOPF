"""Run and preserve the complete Stage 3 CPU sGS-HPR validation."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gpu_dcopf_hpr.dcopf_model import (  # noqa: E402
    build_dcopf_model,
    load_dcopf_config,
)
from gpu_dcopf_hpr.network_data import load_matpower_case  # noqa: E402
from gpu_dcopf_hpr.sgs_hpr import SGSHPRResult, solve_sgs_hpr  # noqa: E402
from gpu_dcopf_hpr.toy_problems import reference_cases  # noqa: E402
from gpu_dcopf_hpr.validation import (  # noqa: E402
    maximum_primal_violation,
    solve_with_highs,
    validate_dcopf_candidate,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sgs_hpr" / "stage_3_fixed_sigma.json"
DEFAULT_NETWORK = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
DEFAULT_DCOPF_CONFIGS = (
    PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json",
    PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _scaled_gap(candidate: float, reference: float) -> float:
    return abs(candidate - reference) / max(1.0, abs(reference))


def _deterministic(first: SGSHPRResult, second: SGSHPRResult) -> bool:
    if (
        first.iterations != second.iterations
        or not np.array_equal(first.solution.x, second.solution.x)
        or not np.array_equal(first.solution.y, second.solution.y)
        or not np.array_equal(first.solution.z, second.solution.z)
        or len(first.history) != len(second.history)
    ):
        return False
    for left, right in zip(first.history, second.history, strict=True):
        left_values = left.as_dict()
        right_values = right.as_dict()
        left_values.pop("iteration_loop_elapsed_seconds")
        right_values.pop("iteration_loop_elapsed_seconds")
        if left_values != right_values:
            return False
    return True


def _solver_summary(
    result: SGSHPRResult,
    canonical_variable_objective: float,
    objective_constant: float = 0.0,
) -> dict[str, Any]:
    minimum_y2 = (
        float(np.min(result.solution.y[result.workspace.equality.rows :]))
        if result.workspace.spectral is not None
        else None
    )
    return {
        "converged": result.converged,
        "iterations": result.iterations,
        "objective": {
            "canonical_variable": canonical_variable_objective,
            "constant": objective_constant,
            "total": canonical_variable_objective + objective_constant,
        },
        "timing_seconds": {
            "preparation": result.preparation_elapsed_seconds,
            "iteration_loop": result.history[-1].iteration_loop_elapsed_seconds,
            "total": result.total_elapsed_seconds,
        },
        "x": result.solution.x.tolist(),
        "y": result.solution.y.tolist(),
        "z": result.solution.z.tolist(),
        "minimum_inequality_multiplier": minimum_y2,
        "residuals": result.residuals.summary(),
        "maximum_equality_solve_relative_residual": (
            result.maximum_equality_solve_relative_residual
        ),
        "maximum_equality_solve_infinity_residual": (
            result.maximum_equality_solve_infinity_residual
        ),
        "maximum_z_x_identity_error": result.maximum_z_x_identity_error,
        "equality_system": result.workspace.equality.summary(),
        "inequality_spectrum": (
            result.workspace.spectral.summary() if result.workspace.spectral is not None else None
        ),
        "sigma": result.sigma,
        "restart_count": result.restart_count,
        "history_interval": result.history_interval,
    }


def _write_trajectory(
    stream: Any,
    *,
    case_name: str,
    classification: str,
    result: SGSHPRResult,
    objective_constant: float = 0.0,
) -> None:
    for entry in result.history:
        record = {
            "case": case_name,
            "classification": classification,
            **entry.as_dict(),
            "objective_constant": objective_constant,
            "total_objective": (entry.canonical_variable_objective + objective_constant),
        }
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
        stream.write("\n")


def _toy_case_summaries(
    config: dict[str, Any],
    trajectory: Any,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for case in reference_cases():
        settings = {
            "sigma": config["sigma"],
            "tolerance": config["paper_tolerance"],
            "kkt_tolerance": config["toy_kkt_combined_target"],
            "max_iterations": config["maximum_iterations"],
            "history_interval": config["history_interval"],
        }
        result = solve_sgs_hpr(case.lp, **settings)
        repeated = solve_sgs_hpr(case.lp, **settings)
        highs = solve_with_highs(case.lp, tolerance=config["paper_tolerance"])
        objective = float(case.lp.c @ result.solution.x)
        objective_gap = _scaled_gap(objective, highs.objective)
        primal_violation = maximum_primal_violation(case.lp, result.solution)
        expected_solution_error = float(
            np.linalg.norm(result.solution.x - case.expected_state.x, ord=np.inf)
        )
        minimum_y2 = float(np.min(result.solution.y[case.lp.m1 :])) if case.lp.m2 else None
        deterministic = _deterministic(result, repeated)
        checks = {
            "converged": result.converged,
            "paper_stopping_satisfied": result.residuals.conditions.all_satisfied,
            "kkt_target_satisfied": (
                result.residuals.combined_norm <= config["toy_kkt_combined_target"]
            ),
            "objective_matches_highs": (
                objective_gap <= config["toy_maximum_scaled_objective_gap"]
            ),
            "solution_matches_expected": (expected_solution_error <= case.solution_tolerance),
            "primal_feasible": (primal_violation <= config["toy_maximum_primal_violation"]),
            "inequality_multipliers_nonnegative": (minimum_y2 is None or minimum_y2 >= -1e-12),
            "equality_solves_accurate": (
                result.maximum_equality_solve_infinity_residual
                <= config["maximum_equality_infinity_residual"]
            ),
            "z_x_identity_accurate": (
                result.maximum_z_x_identity_error <= config["maximum_z_x_identity_error"]
            ),
            "fixed_sigma": result.sigma == 1.0,
            "no_restart": result.restart_count == 0,
            "all_values_finite": all(
                np.all(np.isfinite(block))
                for block in (result.solution.x, result.solution.y, result.solution.z)
            ),
            "deterministic_repeated_run": deterministic,
        }
        classification = "unit-scale mathematical reference LP"
        _write_trajectory(
            trajectory,
            case_name=case.name,
            classification=classification,
            result=result,
        )
        summaries.append(
            {
                "name": case.name,
                "classification": classification,
                "dimensions": {"n": case.lp.n, "m1": case.lp.m1, "m2": case.lp.m2},
                "highs": highs.summary(),
                "sgs_hpr": _solver_summary(result, objective),
                "comparison": {
                    "scaled_objective_gap": objective_gap,
                    "maximum_primal_violation": primal_violation,
                    "expected_solution_error_inf": expected_solution_error,
                },
                "deterministic_repeat": deterministic,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return summaries


def _dcopf_case_summaries(
    config: dict[str, Any],
    network_path: Path,
    trajectory: Any,
) -> list[dict[str, Any]]:
    network = load_matpower_case(network_path)
    summaries: list[dict[str, Any]] = []
    for config_path in DEFAULT_DCOPF_CONFIGS:
        dcopf_config = load_dcopf_config(config_path, network)
        model = build_dcopf_model(network, dcopf_config)
        settings = {
            "sigma": config["sigma"],
            "tolerance": config["paper_tolerance"],
            "kkt_tolerance": config["dcopf_kkt_combined_target"],
            "max_iterations": config["maximum_iterations"],
            "history_interval": config["history_interval"],
        }
        result = solve_sgs_hpr(model.lp, **settings)
        repeated = solve_sgs_hpr(model.lp, **settings)
        highs = solve_with_highs(model.lp, tolerance=config["paper_tolerance"])
        candidate_objective = model.objective(result.solution.x)
        reference_objective = model.objective(highs.state.x)
        objective_gap = _scaled_gap(candidate_objective, reference_objective)
        primal_violation = maximum_primal_violation(model.lp, result.solution)
        physical = validate_dcopf_candidate(
            model,
            result.solution.x,
            tolerance=config["dcopf_physical_tolerance"],
        )
        minimum_y2 = float(np.min(result.solution.y[model.lp.m1 :]))
        deterministic = _deterministic(result, repeated)
        spectral = result.workspace.spectral
        assert spectral is not None
        checks = {
            "converged": result.converged,
            "paper_stopping_satisfied": result.residuals.conditions.all_satisfied,
            "kkt_target_satisfied": (
                result.residuals.combined_norm <= config["dcopf_kkt_combined_target"]
            ),
            "objective_matches_highs": (
                objective_gap <= config["dcopf_maximum_scaled_objective_gap"]
            ),
            "approximate_physical_candidate_validation": physical.passed,
            "canonical_primal_violation": (primal_violation <= config["dcopf_physical_tolerance"]),
            "inequality_multipliers_nonnegative": minimum_y2 >= -1e-12,
            "equality_block_full_row_rank": result.workspace.equality.full_row_rank,
            "equality_gram_positive_definite": result.workspace.equality.positive_definite,
            "equality_solves_accurate": (
                result.maximum_equality_solve_infinity_residual
                <= config["maximum_equality_infinity_residual"]
            ),
            "dense_sparse_power_estimates_agree": (
                spectral.power_converged and spectral.maximum_estimate_difference <= 1e-10
            ),
            "lambda_is_conservative": (
                spectral.lambda_used > spectral.dense_eigendecomposition
                and spectral.s2_minimum_eigenvalue > 0.0
            ),
            "z_x_identity_accurate": (
                result.maximum_z_x_identity_error <= config["maximum_z_x_identity_error"]
            ),
            "fixed_sigma": result.sigma == 1.0,
            "no_restart": result.restart_count == 0,
            "all_values_finite": all(
                np.all(np.isfinite(block))
                for block in (result.solution.x, result.solution.y, result.solution.z)
            ),
            "deterministic_repeated_run": deterministic,
        }
        _write_trajectory(
            trajectory,
            case_name=dcopf_config.name,
            classification=dcopf_config.classification,
            result=result,
            objective_constant=model.objective_constant,
        )
        summaries.append(
            {
                "name": dcopf_config.name,
                "classification": dcopf_config.classification,
                "synthetic_extension": dcopf_config.synthetic_extension,
                "dimensions": model.dimension_summary(),
                "highs": {
                    **highs.summary(),
                    "total_objective": reference_objective,
                },
                "sgs_hpr": _solver_summary(
                    result,
                    float(model.lp.c @ result.solution.x),
                    model.objective_constant,
                ),
                "comparison": {
                    "scaled_objective_gap": objective_gap,
                    "maximum_canonical_primal_violation": primal_violation,
                    "maximum_physical_violation": max(
                        family.maximum_violation for family in physical.families
                    ),
                },
                "approximate_candidate_validation": physical.summary(),
                "deterministic_repeat": deterministic,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return summaries


def _git_parent_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "sgs_hpr_trajectories.jsonl.gz"
    validation_path = output_dir / "stage_3_validation.json"

    with gzip.open(trajectory_path, "wt", encoding="utf-8", newline="\n") as trajectory:
        toy_cases = _toy_case_summaries(config, trajectory)
        dcopf_cases = _dcopf_case_summaries(config, args.network.resolve(), trajectory)

    cases = toy_cases + dcopf_cases
    summary = {
        "stage": 3,
        "all_passed": all(case["passed"] for case in cases),
        "configuration": config,
        "reproduction_boundary": {
            "classification": "mathematical reproduction and structural validation",
            "dgx_spark_touched": False,
            "gpu_code_used": False,
            "structural_y1_used": False,
            "adaptive_sigma_used": False,
            "restart_used": False,
            "scaling_used": False,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "highs_interface": "scipy.optimize.linprog(method='highs-ds')",
            "highspy_installed": importlib.util.find_spec("highspy") is not None,
            "precision": "FP64",
            "execution_device": "local CPU",
            "repository_parent_commit": _git_parent_commit(),
        },
        "trajectory_file": trajectory_path.name,
        "cases": cases,
    }
    validation_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "all_passed": summary["all_passed"],
                "trajectory_file": summary["trajectory_file"],
                "cases": [
                    {
                        "name": case["name"],
                        "passed": case["passed"],
                        "iterations": case["sgs_hpr"]["iterations"],
                        "objective": case["sgs_hpr"]["objective"]["total"],
                        "scaled_objective_gap": case["comparison"]["scaled_objective_gap"],
                    }
                    for case in cases
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
