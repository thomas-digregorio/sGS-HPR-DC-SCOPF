"""Run and preserve the complete Stage 4 structural-equality validation.

This runner is intentionally CPU/FP64-only.  It resolves the manuscript's
rank-one sign discrepancy against a dense Cholesky oracle, compares complete
direct and structural sGS-HPR runs, and records fair warmed timing evidence.
It does not enable scaling, adaptive penalty updates, restart, GPU code, or
DGX execution.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

import numpy as np
import scipy
from scipy import linalg, sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gpu_dcopf_hpr.canonical_lp import CanonicalLP  # noqa: E402
from gpu_dcopf_hpr.dcopf_model import (  # noqa: E402
    build_dcopf_model,
    load_dcopf_config,
)
from gpu_dcopf_hpr.network_data import load_matpower_case  # noqa: E402
from gpu_dcopf_hpr.sgs_hpr import SGSHPRResult, solve_sgs_hpr  # noqa: E402
from gpu_dcopf_hpr.structural_y1 import (  # noqa: E402
    DCOPFEqualityStructure,
    StructuralY1Solver,
    prepare_dcopf_structural_y1,
    prepare_structural_y1,
)
from gpu_dcopf_hpr.validation import (  # noqa: E402
    maximum_primal_violation,
    solve_with_highs,
    validate_dcopf_candidate,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sgs_hpr" / "stage_4_structural.json"
DEFAULT_NETWORK = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
DEFAULT_DCOPF_CONFIGS = (
    PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json",
    PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_4"

LOCKSTEP_STATE_TOLERANCE = 1e-9
LOCKSTEP_OBJECTIVE_TOLERANCE = 1e-12
LOCKSTEP_RESIDUAL_TOLERANCE = 1e-9
COMPLEXITY_SLOPE_RANGE = (0.6, 1.4)
COMPLEXITY_MINIMUM_R_SQUARED = 0.85
COMPLEXITY_MAXIMUM_NORMALIZED_SPREAD = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Stage 4 structural validation settings.",
    )
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument(
        "--dcopf-configs",
        type=Path,
        nargs=2,
        default=DEFAULT_DCOPF_CONFIGS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lockstep-iterations",
        type=int,
        help="Override config fixed_trajectory_iterations.",
    )
    parser.add_argument("--timing-repeats", type=int, default=9)
    parser.add_argument("--timing-batch-size", type=int, default=16)
    parser.add_argument("--complexity-repeats", type=int, default=11)
    parser.add_argument("--complexity-batch-size", type=int, default=64)
    return parser.parse_args()


def _scaled_gap(candidate: float, reference: float) -> float:
    return abs(candidate - reference) / max(1.0, abs(reference))


def _relative_vector_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(1.0, float(np.linalg.norm(reference))))


def _classical_relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _git_parent_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _synthetic_lp(structure: DCOPFEqualityStructure) -> CanonicalLP:
    """Independently assemble the implemented Equation (55) in CSR form."""

    periods = structure.periods
    generators = structure.generator_count
    renewables = structure.renewable_count
    storage = structure.storage_count
    columns = structure.expected_variables
    generator_offset = 0
    renewable_offset = periods * generators
    discharge_offset = renewable_offset + periods * renewables
    charge_offset = discharge_offset + periods * storage

    rows: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for period in range(periods):
        for generator in range(generators):
            rows.append(period)
            column_indices.append(generator_offset + period * generators + generator)
            values.append(1.0)
        for renewable in range(renewables):
            rows.append(period)
            column_indices.append(renewable_offset + period * renewables + renewable)
            values.append(1.0)
        for storage_index in range(storage):
            rows.extend((period, period))
            column_indices.extend(
                (
                    discharge_offset + period * storage + storage_index,
                    charge_offset + period * storage + storage_index,
                )
            )
            values.extend((1.0, -1.0))

    charge = np.asarray(structure.charge_efficiencies, dtype=np.float64)
    discharge = np.asarray(structure.discharge_efficiencies, dtype=np.float64)
    for storage_index in range(storage):
        row = periods + storage_index
        for period in range(periods):
            rows.extend((row, row))
            column_indices.extend(
                (
                    discharge_offset + period * storage + storage_index,
                    charge_offset + period * storage + storage_index,
                )
            )
            values.extend(
                (
                    -structure.interval_hours / discharge[storage_index],
                    structure.interval_hours * charge[storage_index],
                )
            )

    A1 = sparse.coo_matrix(
        (values, (rows, column_indices)),
        shape=(structure.expected_equalities, columns),
        dtype=np.float64,
    ).tocsr()
    return CanonicalLP(
        c=np.zeros(columns, dtype=np.float64),
        A1=A1,
        b1=np.zeros(structure.expected_equalities, dtype=np.float64),
        A2=sparse.csr_matrix((0, columns), dtype=np.float64),
        b2=np.empty(0, dtype=np.float64),
        lower=-np.ones(columns, dtype=np.float64),
        upper=np.ones(columns, dtype=np.float64),
    )


def _crosscheck_fixtures() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "no_storage_t1",
            "periods": 1,
            "generators": 5,
            "renewables": 0,
            "interval_hours": 1.0,
            "efficiencies": (),
            "stress": False,
        },
        {
            "name": "no_storage_t17",
            "periods": 17,
            "generators": 5,
            "renewables": 3,
            "interval_hours": 1.0,
            "efficiencies": (),
            "stress": False,
        },
        {
            "name": "ideal_storage_t1",
            "periods": 1,
            "generators": 5,
            "renewables": 1,
            "interval_hours": 0.25,
            "efficiencies": ((1.0, 1.0),),
            "stress": False,
        },
        {
            "name": "one_storage_t2",
            "periods": 2,
            "generators": 5,
            "renewables": 1,
            "interval_hours": 1.0,
            "efficiencies": ((0.95, 0.90),),
            "stress": False,
        },
        {
            "name": "extreme_efficiency_t32",
            "periods": 32,
            "generators": 5,
            "renewables": 2,
            "interval_hours": 4.0,
            "efficiencies": ((0.05, 0.05),),
            "stress": True,
        },
        {
            "name": "heterogeneous_storage_t5",
            "periods": 5,
            "generators": 5,
            "renewables": 2,
            "interval_hours": 1.0,
            "efficiencies": (
                (0.05, 1.0),
                (0.95, 0.90),
                (1.0, 0.05),
                (1.0, 1.0),
            ),
            "stress": True,
        },
        {
            "name": "many_ideal_storage_t16",
            "periods": 16,
            "generators": 1,
            "renewables": 0,
            "interval_hours": 1.0,
            "efficiencies": ((1.0, 1.0),) * 32,
            "stress": True,
        },
    )


def _structure_from_fixture(fixture: dict[str, Any]) -> DCOPFEqualityStructure:
    efficiencies = fixture["efficiencies"]
    return DCOPFEqualityStructure(
        periods=fixture["periods"],
        generator_count=fixture["generators"],
        renewable_count=fixture["renewables"],
        interval_hours=fixture["interval_hours"],
        charge_efficiencies=[pair[0] for pair in efficiencies],
        discharge_efficiencies=[pair[1] for pair in efficiencies],
    )


def _printed_sign_solution(
    solver: StructuralY1Solver,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Apply the inconsistent minus correction printed in Eqs. (39)/(44)."""

    diagnostics = solver.diagnostics
    periods = diagnostics.periods
    storage = diagnostics.storage_count
    if storage == 0:
        return np.asarray(
            right_hand_side / diagnostics.balance_diagonal,
            dtype=np.float64,
        )

    balance_rhs = right_hand_side[:periods]
    storage_rhs = right_hand_side[periods:]
    coupling = diagnostics.coupling
    inverse_storage = solver.inverse_storage_diagonal
    reduced = balance_rhs - float(np.dot(coupling * inverse_storage, storage_rhs))
    diagonal = diagnostics.balance_diagonal
    diagonal_inverse_reduced = reduced / diagonal
    alpha = diagnostics.alpha
    denominator = 1.0 / alpha + periods / diagonal
    balance = diagonal_inverse_reduced - (
        np.ones(periods, dtype=np.float64)
        * float(np.sum(diagonal_inverse_reduced))
        / diagonal
        / denominator
    )
    terminal = inverse_storage * (storage_rhs - coupling * float(np.sum(balance)))
    return np.concatenate((balance, terminal))


def _oracle_metrics(
    gram: np.ndarray,
    right_hand_side: np.ndarray,
    candidate: np.ndarray,
    direct: np.ndarray,
) -> dict[str, float]:
    difference = candidate - direct
    residual = gram @ candidate - right_hand_side
    return {
        "absolute_error_l2": float(np.linalg.norm(difference)),
        "maximum_component_error": float(np.linalg.norm(difference, ord=np.inf)),
        "relative_solution_error": _classical_relative_error(candidate, direct),
        "scaled_solution_error": _relative_vector_error(candidate, direct),
        "normalized_system_residual": float(
            np.linalg.norm(residual) / max(1.0, float(np.linalg.norm(right_hand_side)))
        ),
        "scaled_component_error": float(
            np.linalg.norm(difference, ord=np.inf)
            / max(1.0, float(np.linalg.norm(direct, ord=np.inf)))
        ),
    }


def _run_structural_crosschecks(
    stream: TextIO,
    config: dict[str, Any],
) -> dict[str, Any]:
    case_summaries: list[dict[str, Any]] = []
    sign_fixture_printed_errors: list[float] = []
    sign_fixture_corrected_errors: list[float] = []
    random_seed = int(config["random_seed"])
    rhs_scales = tuple(float(value) for value in config["rhs_scales"])
    trials_per_scale = int(config["rhs_vectors_per_scale"])
    standard_tolerance = float(config["structural_maximum_normalized_error"])
    stress_tolerance = float(
        config.get(
            "structural_stress_maximum_normalized_error",
            standard_tolerance,
        )
    )

    for case_index, fixture in enumerate(_crosscheck_fixtures()):
        structure = _structure_from_fixture(fixture)
        lp = _synthetic_lp(structure)
        solver = prepare_structural_y1(lp, structure)
        gram = np.asarray((lp.A1 @ lp.A1.T).toarray(), dtype=np.float64)
        threshold = stress_tolerance if fixture["stress"] else standard_tolerance
        corrected_records: list[dict[str, float]] = []
        printed_records: list[dict[str, float]] = []

        for scale_index, rhs_scale in enumerate(rhs_scales):
            for trial in range(trials_per_scale):
                seed = random_seed + case_index * 10_000 + scale_index * 100 + trial
                generator = np.random.default_rng(seed)
                right_hand_side = (
                    generator.standard_normal(structure.expected_equalities).astype(np.float64)
                    * rhs_scale
                )
                direct = np.linalg.solve(gram, right_hand_side)
                corrected = solver.solve(right_hand_side)
                printed = _printed_sign_solution(solver, right_hand_side)
                corrected_metrics = _oracle_metrics(
                    gram,
                    right_hand_side,
                    corrected,
                    direct,
                )
                printed_metrics = _oracle_metrics(
                    gram,
                    right_hand_side,
                    printed,
                    direct,
                )
                corrected_records.append(corrected_metrics)
                printed_records.append(printed_metrics)
                if fixture["name"] == "one_storage_t2":
                    sign_fixture_corrected_errors.append(
                        corrected_metrics["relative_solution_error"]
                    )
                    sign_fixture_printed_errors.append(printed_metrics["relative_solution_error"])
                stream.write(
                    json.dumps(
                        {
                            "case": fixture["name"],
                            "trial": trial,
                            "seed": seed,
                            "rhs_scale": rhs_scale,
                            "rhs_sha256": hashlib.sha256(right_hand_side.tobytes()).hexdigest(),
                            "right_hand_side": right_hand_side.tolist(),
                            "corrected": corrected_metrics,
                            "printed_sign": printed_metrics,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                stream.write("\n")

        maxima = {
            metric: max(record[metric] for record in corrected_records)
            for metric in corrected_records[0]
        }
        printed_median_relative = float(
            np.median([record["relative_solution_error"] for record in printed_records])
        )
        checks = {
            "equation_55_pattern_exact": (solver.diagnostics.maximum_a1_pattern_error == 0.0),
            "relative_solution_error": (maxima["relative_solution_error"] <= threshold),
            "normalized_system_residual": (maxima["normalized_system_residual"] <= threshold),
            "scaled_component_error": (maxima["scaled_component_error"] <= threshold),
            "matrix_free_workspace": (
                solver.diagnostics.summary()["dense_gram_materialized"] is False
                and solver.diagnostics.summary()["explicit_kronecker_materialized"] is False
                and solver.diagnostics.stored_float_count == 3 * structure.storage_count
            ),
        }
        case_summaries.append(
            {
                "name": fixture["name"],
                "stress_case": fixture["stress"],
                "acceptance_tolerance": threshold,
                "dimensions": {
                    "n": lp.n,
                    "m1": lp.m1,
                    "storage": structure.storage_count,
                },
                "condition_number": float(np.linalg.cond(gram)),
                "structure": solver.diagnostics.summary(),
                "maximum_corrected_errors": maxima,
                "printed_sign_median_relative_error": printed_median_relative,
                "right_hand_side_trials": len(corrected_records),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    corrected_maximum = max(sign_fixture_corrected_errors)
    printed_median = float(np.median(sign_fixture_printed_errors))
    sign_resolution = {
        "fixture": "one_storage_t2",
        "corrected_maximum_relative_error": corrected_maximum,
        "printed_sign_median_relative_error": printed_median,
        "corrected_matches_direct": (corrected_maximum <= standard_tolerance),
        "printed_sign_is_materially_wrong": printed_median >= 1e-3,
        "implemented_correction_sign": "positive correction for a minus rank-one update",
        "paper_discrepancy": (
            "Equation (43) is a minus rank-one Schur complement, while the "
            "printed inverse in Equations (39), (44), and (45) has the sign "
            "pattern of a plus update."
        ),
    }
    return {
        "cases": case_summaries,
        "sign_resolution": sign_resolution,
        "passed": (
            all(case["passed"] for case in case_summaries)
            and sign_resolution["corrected_matches_direct"]
            and sign_resolution["printed_sign_is_materially_wrong"]
        ),
    }


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
    *,
    canonical_objective: float,
    objective_constant: float,
) -> dict[str, Any]:
    return {
        "backend": result.workspace.equality_backend,
        "converged": result.converged,
        "iterations": result.iterations,
        "objective": {
            "canonical_variable": canonical_objective,
            "constant": objective_constant,
            "total": canonical_objective + objective_constant,
        },
        "timing_seconds": {
            "preparation": result.preparation_elapsed_seconds,
            "iteration_loop": result.history[-1].iteration_loop_elapsed_seconds,
            "total": result.total_elapsed_seconds,
        },
        "x": result.solution.x.tolist(),
        "y": result.solution.y.tolist(),
        "z": result.solution.z.tolist(),
        "residuals": result.residuals.summary(),
        "maximum_equality_solve_relative_residual": (
            result.maximum_equality_solve_relative_residual
        ),
        "maximum_equality_solve_infinity_residual": (
            result.maximum_equality_solve_infinity_residual
        ),
        "maximum_z_x_identity_error": result.maximum_z_x_identity_error,
        "equality_system": result.workspace.equality.summary(),
        "structural_diagnostics": (
            result.workspace.structural_y1.diagnostics.summary()
            if result.workspace.structural_y1 is not None
            else None
        ),
        "sigma": result.sigma,
        "restart_count": result.restart_count,
        "history_interval": result.history_interval,
    }


def _write_trajectory(
    stream: TextIO,
    *,
    case_name: str,
    classification: str,
    phase: str,
    backend: str,
    result: SGSHPRResult,
    objective_constant: float,
) -> None:
    for entry in result.history:
        record = {
            "case": case_name,
            "classification": classification,
            "phase": phase,
            "backend": backend,
            **entry.as_dict(),
            "objective_constant": objective_constant,
            "total_objective": (entry.canonical_variable_objective + objective_constant),
        }
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
        stream.write("\n")


def _trajectory_comparison(
    direct: SGSHPRResult,
    structural: SGSHPRResult,
) -> dict[str, Any]:
    direct_by_iteration = {entry.iteration: entry for entry in direct.history}
    structural_by_iteration = {entry.iteration: entry for entry in structural.history}
    iteration_grid_matches = tuple(direct_by_iteration) == tuple(structural_by_iteration)
    common_iterations = sorted(direct_by_iteration.keys() & structural_by_iteration.keys())
    paired = [
        (
            direct_by_iteration[iteration],
            structural_by_iteration[iteration],
        )
        for iteration in common_iterations
    ]
    objective_gaps = [
        _scaled_gap(
            structural_entry.canonical_variable_objective,
            direct_entry.canonical_variable_objective,
        )
        for direct_entry, structural_entry in paired
    ]
    residual_gaps = [
        abs(structural_entry.kkt_combined_norm - direct_entry.kkt_combined_norm)
        for direct_entry, structural_entry in paired
    ]
    paper_component_gaps: list[float] = []
    for direct_entry, structural_entry in paired:
        direct_values = direct_entry.as_dict()["paper_normalized"]
        structural_values = structural_entry.as_dict()["paper_normalized"]
        paper_component_gaps.extend(
            abs(structural_values[name] - direct_values[name]) for name in direct_values
        )
    return {
        "iteration_grid_matches": iteration_grid_matches,
        "sample_count": len(paired),
        "direct_sample_count": len(direct.history),
        "structural_sample_count": len(structural.history),
        "maximum_scaled_objective_gap": max(objective_gaps, default=0.0),
        "maximum_combined_residual_gap": max(residual_gaps, default=0.0),
        "maximum_paper_component_gap": max(paper_component_gaps, default=0.0),
    }


def _backend_validations(
    model: Any,
    result: SGSHPRResult,
    reference_objective: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    objective = model.objective(result.solution.x)
    physical = validate_dcopf_candidate(
        model,
        result.solution.x,
        tolerance=config["dcopf_physical_tolerance"],
    )
    primal_violation = maximum_primal_violation(model.lp, result.solution)
    checks = {
        "converged": result.converged,
        "paper_stopping_satisfied": (result.residuals.conditions.all_satisfied),
        "kkt_target_satisfied": (
            result.residuals.combined_norm <= config["dcopf_kkt_combined_target"]
        ),
        "objective_matches_highs": (
            _scaled_gap(objective, reference_objective)
            <= config["dcopf_maximum_scaled_objective_gap"]
        ),
        "physical_candidate_valid": physical.passed,
        "canonical_primal_violation": (primal_violation <= config["dcopf_physical_tolerance"]),
        "equality_solves_accurate": (
            result.maximum_equality_solve_infinity_residual
            <= config["maximum_equality_infinity_residual"]
        ),
        "z_x_identity_accurate": (
            result.maximum_z_x_identity_error <= config.get("maximum_z_x_identity_error", 1e-12)
        ),
        "fixed_sigma": result.sigma == 1.0,
        "no_restart": result.restart_count == 0,
        "all_values_finite": all(
            np.all(np.isfinite(block))
            for block in (
                result.solution.x,
                result.solution.y,
                result.solution.z,
            )
        ),
    }
    return (
        {
            "scaled_objective_gap_to_highs": _scaled_gap(
                objective,
                reference_objective,
            ),
            "maximum_canonical_primal_violation": primal_violation,
            "maximum_physical_violation": max(
                family.maximum_violation for family in physical.families
            ),
            "physical_validation": physical.summary(),
        },
        checks,
    )


def _run_full_solver_crosschecks(
    config: dict[str, Any],
    network_path: Path,
    dcopf_paths: tuple[Path, ...],
    trajectory_stream: TextIO,
    *,
    lockstep_iterations: int,
) -> dict[str, Any]:
    network = load_matpower_case(network_path)
    summaries: list[dict[str, Any]] = []

    for config_path in dcopf_paths:
        dcopf_config = load_dcopf_config(config_path, network)
        model = build_dcopf_model(network, dcopf_config)
        structural_solver = prepare_dcopf_structural_y1(model)
        common = {
            "sigma": float(config.get("sigma", 1.0)),
            "history_interval": config["history_interval"],
        }
        lockstep_settings = {
            **common,
            "tolerance": 1e-14,
            "kkt_tolerance": None,
            "max_iterations": lockstep_iterations,
        }
        lockstep_direct = solve_sgs_hpr(model.lp, **lockstep_settings)
        lockstep_structural = solve_sgs_hpr(
            model.lp,
            structural_y1=structural_solver,
            **lockstep_settings,
        )
        _write_trajectory(
            trajectory_stream,
            case_name=dcopf_config.name,
            classification=dcopf_config.classification,
            phase=f"lockstep_{lockstep_iterations}",
            backend="direct",
            result=lockstep_direct,
            objective_constant=model.objective_constant,
        )
        _write_trajectory(
            trajectory_stream,
            case_name=dcopf_config.name,
            classification=dcopf_config.classification,
            phase=f"lockstep_{lockstep_iterations}",
            backend="structural",
            result=lockstep_structural,
            objective_constant=model.objective_constant,
        )
        trajectory_comparison = _trajectory_comparison(
            lockstep_direct,
            lockstep_structural,
        )
        lockstep_comparison = {
            **trajectory_comparison,
            "x_relative_error": _relative_vector_error(
                lockstep_structural.solution.x,
                lockstep_direct.solution.x,
            ),
            "y_relative_error": _relative_vector_error(
                lockstep_structural.solution.y,
                lockstep_direct.solution.y,
            ),
            "z_relative_error": _relative_vector_error(
                lockstep_structural.solution.z,
                lockstep_direct.solution.z,
            ),
            "final_scaled_objective_gap": _scaled_gap(
                model.objective(lockstep_structural.solution.x),
                model.objective(lockstep_direct.solution.x),
            ),
            "final_combined_residual_gap": abs(
                lockstep_structural.residuals.combined_norm
                - lockstep_direct.residuals.combined_norm
            ),
        }
        lockstep_checks = {
            "both_completed_requested_iterations": (
                lockstep_direct.iterations == lockstep_iterations
                and lockstep_structural.iterations == lockstep_iterations
            ),
            "trajectory_iteration_grid_matches": (trajectory_comparison["iteration_grid_matches"]),
            "state_trajectory_matches": max(
                lockstep_comparison["x_relative_error"],
                lockstep_comparison["y_relative_error"],
                lockstep_comparison["z_relative_error"],
            )
            <= LOCKSTEP_STATE_TOLERANCE,
            "objective_trajectory_matches": max(
                trajectory_comparison["maximum_scaled_objective_gap"],
                lockstep_comparison["final_scaled_objective_gap"],
            )
            <= LOCKSTEP_OBJECTIVE_TOLERANCE,
            "residual_trajectory_matches": max(
                trajectory_comparison["maximum_combined_residual_gap"],
                trajectory_comparison["maximum_paper_component_gap"],
                lockstep_comparison["final_combined_residual_gap"],
            )
            <= LOCKSTEP_RESIDUAL_TOLERANCE,
        }

        convergence_settings = {
            **common,
            "tolerance": config["paper_tolerance"],
            "kkt_tolerance": config["dcopf_kkt_combined_target"],
            "max_iterations": config["maximum_iterations"],
        }
        converged_direct = solve_sgs_hpr(model.lp, **convergence_settings)
        converged_structural = solve_sgs_hpr(
            model.lp,
            structural_y1=structural_solver,
            **convergence_settings,
        )
        repeated_structural = solve_sgs_hpr(
            model.lp,
            structural_y1=structural_solver,
            **convergence_settings,
        )
        for backend, result in (
            ("direct", converged_direct),
            ("structural", converged_structural),
            ("structural_repeat", repeated_structural),
        ):
            _write_trajectory(
                trajectory_stream,
                case_name=dcopf_config.name,
                classification=dcopf_config.classification,
                phase="convergence",
                backend=backend,
                result=result,
                objective_constant=model.objective_constant,
            )

        highs = solve_with_highs(
            model.lp,
            tolerance=config["paper_tolerance"],
        )
        reference_objective = model.objective(highs.state.x)
        direct_validation, direct_checks = _backend_validations(
            model,
            converged_direct,
            reference_objective,
            config,
        )
        structural_validation, structural_checks = _backend_validations(
            model,
            converged_structural,
            reference_objective,
            config,
        )
        iteration_limit = max(
            10,
            int(
                np.ceil(
                    config["direct_structural_iteration_fraction"] * converged_direct.iterations
                )
            ),
        )
        convergence_comparison = {
            "scaled_objective_gap": _scaled_gap(
                model.objective(converged_structural.solution.x),
                model.objective(converged_direct.solution.x),
            ),
            "x_relative_error": _relative_vector_error(
                converged_structural.solution.x,
                converged_direct.solution.x,
            ),
            "y_relative_error": _relative_vector_error(
                converged_structural.solution.y,
                converged_direct.solution.y,
            ),
            "z_relative_error": _relative_vector_error(
                converged_structural.solution.z,
                converged_direct.solution.z,
            ),
            "iteration_difference": abs(
                converged_structural.iterations - converged_direct.iterations
            ),
            "iteration_difference_limit": iteration_limit,
            "structural_repeat_deterministic": _deterministic(
                converged_structural,
                repeated_structural,
            ),
        }
        convergence_checks = {
            "direct_backend_valid": all(direct_checks.values()),
            "structural_backend_valid": all(structural_checks.values()),
            "objective_matches_direct": (
                convergence_comparison["scaled_objective_gap"]
                <= config["direct_structural_maximum_scaled_objective_difference"]
            ),
            "solution_matches_direct_at_stopping_accuracy": (
                convergence_comparison["x_relative_error"]
                <= config["direct_structural_maximum_solution_relative_difference"]
            ),
            "iteration_counts_agree": (
                convergence_comparison["iteration_difference"] <= iteration_limit
            ),
            "structural_repeat_deterministic": (
                convergence_comparison["structural_repeat_deterministic"]
            ),
            "structural_backend_selected": (
                converged_structural.workspace.equality_backend == "structural"
                and converged_structural.workspace.equality_gram is None
                and converged_structural.workspace.equality_cholesky is None
            ),
        }
        summaries.append(
            {
                "name": dcopf_config.name,
                "classification": dcopf_config.classification,
                "dimensions": model.dimension_summary(),
                "structure": structural_solver.diagnostics.summary(),
                "lockstep": {
                    "requested_iterations": lockstep_iterations,
                    "direct": _solver_summary(
                        lockstep_direct,
                        canonical_objective=float(model.lp.c @ lockstep_direct.solution.x),
                        objective_constant=model.objective_constant,
                    ),
                    "structural": _solver_summary(
                        lockstep_structural,
                        canonical_objective=float(model.lp.c @ lockstep_structural.solution.x),
                        objective_constant=model.objective_constant,
                    ),
                    "comparison": lockstep_comparison,
                    "checks": lockstep_checks,
                },
                "convergence": {
                    "highs": {
                        **highs.summary(),
                        "total_objective": reference_objective,
                    },
                    "direct": _solver_summary(
                        converged_direct,
                        canonical_objective=float(model.lp.c @ converged_direct.solution.x),
                        objective_constant=model.objective_constant,
                    ),
                    "structural": _solver_summary(
                        converged_structural,
                        canonical_objective=float(model.lp.c @ converged_structural.solution.x),
                        objective_constant=model.objective_constant,
                    ),
                    "direct_validation": direct_validation,
                    "structural_validation": structural_validation,
                    "direct_checks": direct_checks,
                    "structural_checks": structural_checks,
                    "comparison": convergence_comparison,
                    "checks": convergence_checks,
                },
                "passed": (all(lockstep_checks.values()) and all(convergence_checks.values())),
            }
        )
    return {
        "cases": summaries,
        "passed": all(case["passed"] for case in summaries),
    }


def _timing_distribution(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    return {
        "samples_seconds_per_right_hand_side": values.tolist(),
        "median_seconds_per_right_hand_side": median,
        "minimum_seconds_per_right_hand_side": float(np.min(values)),
        "maximum_seconds_per_right_hand_side": float(np.max(values)),
        "first_quartile_seconds_per_right_hand_side": float(np.quantile(values, 0.25)),
        "third_quartile_seconds_per_right_hand_side": float(np.quantile(values, 0.75)),
        "median_absolute_deviation_seconds_per_right_hand_side": float(
            np.median(np.abs(values - median))
        ),
    }


def _time_batch(
    solver: Callable[[np.ndarray], np.ndarray],
    right_hand_sides: np.ndarray,
    common_vectors: np.ndarray | None,
    A1: sparse.csr_matrix,
) -> tuple[float, float]:
    start = perf_counter()
    checksum = 0.0
    if common_vectors is None:
        for right_hand_side in right_hand_sides:
            result = solver(right_hand_side)
            checksum += float(result[0])
    else:
        for equality_seed in common_vectors:
            right_hand_side = np.asarray(
                A1 @ np.asarray(A1.T @ equality_seed, dtype=np.float64),
                dtype=np.float64,
            )
            result = solver(right_hand_side)
            checksum += float(result[0])
    elapsed = perf_counter() - start
    if not np.isfinite(checksum):
        raise RuntimeError("Timing checksum is not finite.")
    count = right_hand_sides.shape[0] if common_vectors is None else common_vectors.shape[0]
    return elapsed / count, checksum


def _timed_fixture(
    *,
    periods: int,
    batch_size: int,
    repeats: int,
    include_direct: bool,
    seed: int,
) -> dict[str, Any]:
    structure = DCOPFEqualityStructure(
        periods=periods,
        generator_count=2,
        renewable_count=1,
        interval_hours=1.0,
        charge_efficiencies=(0.95, 0.80),
        discharge_efficiencies=(0.90, 0.75),
    )
    lp = _synthetic_lp(structure)
    structural_preparation_start = perf_counter()
    structural = prepare_structural_y1(lp, structure)
    structural_preparation = perf_counter() - structural_preparation_start

    direct_preparation: float | None = None
    direct_solver: Callable[[np.ndarray], np.ndarray] | None = None
    if include_direct:
        direct_preparation_start = perf_counter()
        gram = np.asarray((lp.A1 @ lp.A1.T).toarray(), dtype=np.float64)
        factor = linalg.cho_factor(gram, lower=True, check_finite=True)
        direct_preparation = perf_counter() - direct_preparation_start

        def solve_direct(right_hand_side: np.ndarray) -> np.ndarray:
            return np.asarray(
                linalg.cho_solve(
                    factor,
                    right_hand_side,
                    check_finite=False,
                ),
                dtype=np.float64,
            )

        direct_solver = solve_direct

    generator = np.random.default_rng(seed)
    right_hand_sides = generator.standard_normal(
        (batch_size, structure.expected_equalities)
    ).astype(np.float64)
    common_vectors = generator.standard_normal((batch_size, structure.expected_equalities)).astype(
        np.float64
    )
    A1 = sparse.csr_matrix(lp.A1)

    structural_solver = structural.solve
    warmup_count = min(2, batch_size)
    _time_batch(
        structural_solver,
        right_hand_sides[:warmup_count],
        None,
        A1,
    )
    _time_batch(
        structural_solver,
        right_hand_sides[:warmup_count],
        common_vectors[:warmup_count],
        A1,
    )
    if direct_solver is not None:
        _time_batch(
            direct_solver,
            right_hand_sides[:warmup_count],
            None,
            A1,
        )
        _time_batch(
            direct_solver,
            right_hand_sides[:warmup_count],
            common_vectors[:warmup_count],
            A1,
        )

    samples: dict[str, dict[str, list[float]]] = {
        "structural": {"solve_only": [], "rhs_and_solve": []},
    }
    if direct_solver is not None:
        samples["direct"] = {"solve_only": [], "rhs_and_solve": []}
    checksums: list[float] = []
    for repeat in range(repeats):
        backends = (
            ("structural", structural_solver),
            ("direct", direct_solver),
        )
        if repeat % 2:
            backends = tuple(reversed(backends))
        for backend, selected_solver in backends:
            if selected_solver is None:
                continue
            solve_elapsed, solve_checksum = _time_batch(
                selected_solver,
                right_hand_sides,
                None,
                A1,
            )
            full_elapsed, full_checksum = _time_batch(
                selected_solver,
                right_hand_sides,
                common_vectors,
                A1,
            )
            samples[backend]["solve_only"].append(solve_elapsed)
            samples[backend]["rhs_and_solve"].append(full_elapsed)
            checksums.extend((solve_checksum, full_checksum))

    distributions = {
        backend: {
            phase: _timing_distribution(phase_samples)
            for phase, phase_samples in backend_samples.items()
        }
        for backend, backend_samples in samples.items()
    }
    speedups = None
    if "direct" in distributions:
        speedups = {
            phase: (
                distributions["direct"][phase]["median_seconds_per_right_hand_side"]
                / distributions["structural"][phase]["median_seconds_per_right_hand_side"]
            )
            for phase in ("solve_only", "rhs_and_solve")
        }
    return {
        "periods": periods,
        "resources": {
            "generators": structure.generator_count,
            "renewables": structure.renewable_count,
            "storage": structure.storage_count,
        },
        "n": lp.n,
        "m1": lp.m1,
        "nnz_A1": int(lp.A1.nnz),
        "paper_work_proxy": (
            periods
            * (structure.generator_count + structure.renewable_count + structure.storage_count)
        ),
        "batch_size": batch_size,
        "repeats": repeats,
        "preparation_seconds": {
            "structural": structural_preparation,
            "direct_gram_and_cholesky": direct_preparation,
        },
        "timings": distributions,
        "speedup_direct_over_structural": speedups,
        "finite_checksum": bool(np.all(np.isfinite(checksums))),
        "structural_workspace": structural.diagnostics.summary(),
    }


def _log_log_fit(
    points: list[dict[str, Any]],
    *,
    phase: str,
) -> dict[str, float]:
    work = np.asarray(
        [point["paper_work_proxy"] for point in points],
        dtype=np.float64,
    )
    timings = np.asarray(
        [
            point["timings"]["structural"][phase]["median_seconds_per_right_hand_side"]
            for point in points
        ],
        dtype=np.float64,
    )
    slope, intercept = np.polyfit(np.log(work), np.log(timings), deg=1)
    predicted = intercept + slope * np.log(work)
    observed = np.log(timings)
    residual_sum = float(np.sum(np.square(observed - predicted)))
    total_sum = float(np.sum(np.square(observed - np.mean(observed))))
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    normalized = timings / work
    normalized_spread = float(np.max(normalized) / np.min(normalized))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "normalized_time_per_work_spread": normalized_spread,
    }


def _complexity_fit_passed(fit: dict[str, float]) -> bool:
    return (
        COMPLEXITY_SLOPE_RANGE[0] <= fit["slope"] <= COMPLEXITY_SLOPE_RANGE[1]
        and fit["r_squared"] >= COMPLEXITY_MINIMUM_R_SQUARED
        and fit["normalized_time_per_work_spread"] <= COMPLEXITY_MAXIMUM_NORMALIZED_SPREAD
    )


def _run_performance_evidence(
    *,
    config: dict[str, Any],
    timing_repeats: int,
    timing_batch_size: int,
    complexity_repeats: int,
    complexity_batch_size: int,
) -> dict[str, Any]:
    random_seed = int(config["random_seed"])
    comparison_points = [
        _timed_fixture(
            periods=periods,
            batch_size=timing_batch_size,
            repeats=timing_repeats,
            include_direct=True,
            seed=random_seed + periods,
        )
        for periods in (128, 256, 512, 1_024)
    ]
    complexity_points = [
        _timed_fixture(
            periods=periods,
            batch_size=complexity_batch_size,
            repeats=complexity_repeats,
            include_direct=False,
            seed=random_seed + periods,
        )
        for periods in (2_048, 4_096, 8_192, 16_384, 32_768, 65_536)
    ]
    kernel_fit = _log_log_fit(complexity_points, phase="solve_only")
    full_update_fit = _log_log_fit(
        complexity_points,
        phase="rhs_and_solve",
    )
    largest_speedup = comparison_points[-1]["speedup_direct_over_structural"]
    assert largest_speedup is not None
    kernel_fit_passed = _complexity_fit_passed(kernel_fit)
    checks = {
        "largest_solve_only_advantage_measured": (largest_speedup["solve_only"] > 1.0),
        "all_timing_checksums_finite": all(
            point["finite_checksum"] for point in comparison_points + complexity_points
        ),
        "solve_only_fit_reported": all(np.isfinite(value) for value in kernel_fit.values()),
        "paper_work_proxy_rhs_and_solve_approximately_linear": (
            _complexity_fit_passed(full_update_fit)
        ),
    }
    return {
        "timing_protocol": {
            "device": "local CPU",
            "precision": "FP64",
            "warmup": "two right-hand sides per backend and timing boundary",
            "backend_order": "alternated structural/direct across repeats",
            "configured_boundaries": config["timing_boundary"],
            "preparation_reported_separately": True,
            "variability": "all samples, median, quartiles, and MAD",
        },
        "direct_structural_comparison": comparison_points,
        "complexity_points": complexity_points,
        "fits": {
            "structural_solve_only_O_T_plus_NESS": kernel_fit,
            "rhs_and_structural_solve_O_nnz_A1": full_update_fit,
        },
        "acceptance": {
            "gating_fit": "rhs_and_structural_solve_O_nnz_A1",
            "slope_range": list(COMPLEXITY_SLOPE_RANGE),
            "minimum_r_squared": COMPLEXITY_MINIMUM_R_SQUARED,
            "maximum_normalized_time_per_work_spread": (COMPLEXITY_MAXIMUM_NORMALIZED_SPREAD),
            "interpretation": (
                "The paper's Corollary 1 includes construction of R1 through "
                "A1 products, so the RHS-plus-structural-solve fit is the "
                "Stage 4 complexity gate. Empirical timing supports rather "
                "than proves the theoretical trend and is not a paper "
                "speedup reproduction."
            ),
        },
        "non_gating_diagnostics": {
            "solve_only_fit_meets_same_timing_heuristic": kernel_fit_passed,
            "solve_only_interpretation": (
                "Reported separately because short NumPy vector kernels are "
                "sensitive to fixed dispatch, allocation, cache, and memory "
                "bandwidth effects. Formal operation counting remains "
                "O(T + N_ESS), but this wall-time fit is not used to claim it."
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_positive_integer(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_stage4_config(config: dict[str, Any]) -> None:
    if config.get("stage") != 4:
        raise ValueError("The evidence runner requires a Stage 4 configuration.")
    if config.get("precision") != "FP64":
        raise ValueError("Stage 4 correctness evidence must use FP64.")
    if float(config.get("sigma", 1.0)) != 1.0:
        raise ValueError("Stage 4 must preserve the validated fixed sigma=1 baseline.")
    unsupported = config.get("unsupported_features", {})
    enabled = sorted(name for name, value in unsupported.items() if value is not False)
    if enabled:
        raise ValueError("Stage 4 cannot enable Stage 5/6 features: " + ", ".join(enabled) + ".")


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    _validate_stage4_config(config)
    lockstep_iterations = (
        int(config["fixed_trajectory_iterations"])
        if args.lockstep_iterations is None
        else args.lockstep_iterations
    )
    for name in (
        "timing_repeats",
        "timing_batch_size",
        "complexity_repeats",
        "complexity_batch_size",
    ):
        _validate_positive_integer(name, getattr(args, name))
    _validate_positive_integer(
        "lockstep_iterations",
        lockstep_iterations,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crosscheck_path = output_dir / "structural_crosschecks.jsonl.gz"
    trajectory_path = output_dir / "solver_trajectories.jsonl.gz"
    validation_path = output_dir / "stage_4_validation.json"

    with gzip.open(
        crosscheck_path,
        "wt",
        encoding="utf-8",
        newline="\n",
    ) as crosscheck_stream:
        structural_crosschecks = _run_structural_crosschecks(
            crosscheck_stream,
            config,
        )
    with gzip.open(
        trajectory_path,
        "wt",
        encoding="utf-8",
        newline="\n",
    ) as trajectory_stream:
        full_solver = _run_full_solver_crosschecks(
            config,
            args.network.resolve(),
            tuple(path.resolve() for path in args.dcopf_configs),
            trajectory_stream,
            lockstep_iterations=lockstep_iterations,
        )
    performance = _run_performance_evidence(
        config=config,
        timing_repeats=args.timing_repeats,
        timing_batch_size=args.timing_batch_size,
        complexity_repeats=args.complexity_repeats,
        complexity_batch_size=args.complexity_batch_size,
    )

    section_passes = {
        "structural_crosschecks": structural_crosschecks["passed"],
        "full_solver_crosschecks": full_solver["passed"],
        "performance_and_complexity": performance["passed"],
    }
    summary = {
        "stage": 4,
        "all_passed": all(section_passes.values()),
        "section_passes": section_passes,
        "configuration": {
            "stage_4_structural": config,
            "random_seed": config["random_seed"],
            "lockstep_iterations": lockstep_iterations,
            "oracle_tolerances": {
                "standard": config["structural_maximum_normalized_error"],
                "stress": config.get(
                    "structural_stress_maximum_normalized_error",
                    config["structural_maximum_normalized_error"],
                ),
            },
            "lockstep_tolerances": {
                "state": LOCKSTEP_STATE_TOLERANCE,
                "objective": LOCKSTEP_OBJECTIVE_TOLERANCE,
                "residual": LOCKSTEP_RESIDUAL_TOLERANCE,
            },
            "convergence_comparison_tolerances": {
                "objective": config["direct_structural_maximum_scaled_objective_difference"],
                "state": config["direct_structural_maximum_solution_relative_difference"],
                "iteration_difference": ("max(10, ceil(configured_fraction * direct_iterations))"),
            },
        },
        "reproduction_boundary": {
            "classification": "paper-specific structural reproduction",
            "dgx_spark_touched": False,
            "gpu_code_used": False,
            "structural_y1_used": True,
            "direct_oracle_used": True,
            "adaptive_sigma_used": False,
            "restart_used": False,
            "scaling_used": False,
            "precision": "FP64",
            "execution_device": "local CPU",
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "highs_interface": "scipy.optimize.linprog(method='highs-ds')",
            "highspy_installed": importlib.util.find_spec("highspy") is not None,
            "repository_parent_commit": _git_parent_commit(),
        },
        "evidence_files": {
            "structural_crosschecks": crosscheck_path.name,
            "solver_trajectories": trajectory_path.name,
        },
        "structural_crosschecks": structural_crosschecks,
        "full_solver_crosschecks": full_solver,
        "performance_and_complexity": performance,
        "numerical_stability": {
            "documented_limits": [
                (
                    "The Schur scalar can become small when non-storage "
                    "resources are scarce relative to many ideal-efficiency "
                    "storage devices."
                ),
                (
                    "Very small efficiencies enlarge terminal-row coefficients "
                    "and can increase the equality-system condition number."
                ),
                (
                    "The implementation uses the cancellation-resistant "
                    "mean/zero-mean split and a stable physical expression for "
                    "the Schur scalar."
                ),
                (
                    "All correctness and timing evidence is FP64; no FP32 or "
                    "mixed-precision conclusion is made."
                ),
            ],
            "extreme_efficiency_covered": True,
            "many_ideal_storage_covered": True,
        },
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
                "section_passes": section_passes,
                "validation_file": str(validation_path),
                "crosscheck_file": str(crosscheck_path),
                "trajectory_file": str(trajectory_path),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
