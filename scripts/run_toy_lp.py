"""Run and preserve the complete Stage 1 HPR/HiGHS comparison."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gpu_dcopf_hpr.hpr_generic import solve_hpr  # noqa: E402
from gpu_dcopf_hpr.toy_problems import reference_cases  # noqa: E402
from gpu_dcopf_hpr.validation import (  # noqa: E402
    maximum_primal_violation,
    solve_with_highs,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "toy_lp" / "stage_1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _state_equal(first: Any, second: Any) -> bool:
    return (
        first.iterations == second.iterations
        and np.array_equal(first.solution.x, second.solution.x)
        and np.array_equal(first.solution.y, second.solution.y)
        and np.array_equal(first.solution.z, second.solution.z)
        and np.array_equal(
            [entry.kkt_combined_norm for entry in first.history],
            [entry.kkt_combined_norm for entry in second.history],
        )
    )


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "hpr_trajectories.jsonl.gz"
    comparison_path = output_dir / "reference_comparison.json"

    case_summaries: list[dict[str, Any]] = []
    all_passed = True
    with gzip.open(trajectory_path, "wt", encoding="utf-8", newline="\n") as trajectory:
        for case in reference_cases():
            highs = solve_with_highs(case.lp, tolerance=config["paper_tolerance"])
            hpr = solve_hpr(
                case.lp,
                sigma=config["sigma"],
                tolerance=config["paper_tolerance"],
                kkt_tolerance=config["kkt_combined_target"],
                max_iterations=config["max_iterations"],
            )
            repeated = solve_hpr(
                case.lp,
                sigma=config["sigma"],
                tolerance=config["paper_tolerance"],
                kkt_tolerance=config["kkt_combined_target"],
                max_iterations=config["max_iterations"],
            )

            objective = float(case.lp.c @ hpr.solution.x)
            scaled_objective_gap = abs(objective - highs.objective) / (1.0 + abs(highs.objective))
            solution_error_inf = float(
                np.linalg.norm(hpr.solution.x - case.expected_state.x, ord=np.inf)
            )
            primal_violation = maximum_primal_violation(case.lp, hpr.solution)
            minimum_y2 = float(np.min(hpr.solution.y[case.lp.m1 :])) if case.lp.m2 else None
            finite = all(
                np.all(np.isfinite(block))
                for block in (hpr.solution.x, hpr.solution.y, hpr.solution.z)
            )
            deterministic = _state_equal(hpr, repeated)
            checks = {
                "hpr_converged": hpr.converged,
                "paper_stopping_satisfied": hpr.residuals.conditions.all_satisfied,
                "kkt_target_satisfied": (
                    hpr.residuals.combined_norm <= config["kkt_combined_target"]
                ),
                "objective_matches_highs": (
                    scaled_objective_gap <= config["maximum_scaled_objective_gap"]
                ),
                "solution_matches_expected": (solution_error_inf <= case.solution_tolerance),
                "primal_feasible": (primal_violation <= config["maximum_primal_violation"]),
                "inequality_multipliers_nonnegative": (minimum_y2 is None or minimum_y2 >= -1e-12),
                "all_values_finite": bool(finite),
                "deterministic_repeated_run": deterministic,
                "highs_kkt_reference_valid": highs.residuals.combined_norm <= 1e-8,
            }
            passed = all(checks.values())
            all_passed = all_passed and passed

            case_summaries.append(
                {
                    "name": case.name,
                    "description": case.description,
                    "provenance_seed": case.provenance_seed,
                    "dimensions": {"n": case.lp.n, "m1": case.lp.m1, "m2": case.lp.m2},
                    "expected": {
                        "x": case.expected_state.x.tolist(),
                        "objective": case.expected_objective,
                    },
                    "highs": highs.summary(),
                    "hpr": {
                        "converged": hpr.converged,
                        "iterations": hpr.iterations,
                        "objective": objective,
                        "x": hpr.solution.x.tolist(),
                        "y": hpr.solution.y.tolist(),
                        "z": hpr.solution.z.tolist(),
                        "residuals": hpr.residuals.summary(),
                        "proximal_operator": {
                            "tau": hpr.proximal.tau,
                            "lambda_max": hpr.proximal.lambda_max,
                            "margin": hpr.proximal.margin,
                            "minimum_eigenvalue": hpr.proximal.minimum_eigenvalue,
                        },
                    },
                    "comparison": {
                        "scaled_objective_gap": scaled_objective_gap,
                        "solution_error_inf": solution_error_inf,
                        "maximum_primal_violation": primal_violation,
                        "minimum_inequality_multiplier": minimum_y2,
                    },
                    "checks": checks,
                    "passed": passed,
                }
            )

            for entry in hpr.history:
                row = {"case": case.name, **entry.as_dict()}
                trajectory.write(json.dumps(row, sort_keys=True, allow_nan=False))
                trajectory.write("\n")

    summary = {
        "stage": 1,
        "all_passed": all_passed,
        "configuration": config,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "highs_interface": "scipy.optimize.linprog(method='highs-ds')",
            "highspy_installed": importlib.util.find_spec("highspy") is not None,
            "platform": platform.platform(),
        },
        "trajectory_file": trajectory_path.name,
        "cases": case_summaries,
    }
    comparison_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
