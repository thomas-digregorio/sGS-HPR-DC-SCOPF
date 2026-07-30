from pathlib import Path

import numpy as np
import pytest

from gpu_dcopf_hpr.dcopf_model import (
    build_dcopf_model,
    load_dcopf_config,
)
from gpu_dcopf_hpr.network_data import load_matpower_case
from gpu_dcopf_hpr.sgs_hpr import solve_sgs_hpr
from gpu_dcopf_hpr.toy_problems import reference_cases
from gpu_dcopf_hpr.validation import (
    maximum_primal_violation,
    solve_with_highs,
    validate_dcopf_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
CONFIGS = (
    PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json",
    PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json",
)
PAPER_TOLERANCE = 5e-5
TOY_KKT_TOLERANCE = 2.5e-4
TOY_OBJECTIVE_GAP_TOLERANCE = 5e-4
DCOPF_OBJECTIVE_GAP_TOLERANCE = 2e-4
DCOPF_PHYSICAL_TOLERANCE = 0.01
DCOPF_KKT_TOLERANCE = 0.02


def _scaled_gap(candidate: float, reference: float) -> float:
    return abs(candidate - reference) / max(1.0, abs(reference))


@pytest.mark.parametrize("case", reference_cases(), ids=lambda case: case.name)
def test_cpu_sgs_hpr_matches_toy_references(case) -> None:
    result = solve_sgs_hpr(
        case.lp,
        sigma=1.0,
        tolerance=PAPER_TOLERANCE,
        kkt_tolerance=TOY_KKT_TOLERANCE,
        max_iterations=100_000,
        history_interval=250,
    )
    highs = solve_with_highs(case.lp, tolerance=PAPER_TOLERANCE)
    objective = float(case.lp.c @ result.solution.x)

    assert result.converged
    assert result.residuals.conditions.all_satisfied
    assert result.residuals.combined_norm <= TOY_KKT_TOLERANCE
    assert maximum_primal_violation(case.lp, result.solution) <= TOY_KKT_TOLERANCE
    assert _scaled_gap(objective, highs.objective) <= TOY_OBJECTIVE_GAP_TOLERANCE
    assert objective == pytest.approx(
        case.expected_objective,
        abs=TOY_OBJECTIVE_GAP_TOLERANCE * max(1.0, abs(case.expected_objective)),
    )
    np.testing.assert_allclose(
        result.solution.x,
        highs.state.x,
        rtol=0.0,
        atol=case.solution_tolerance,
    )
    assert np.all(np.isfinite(result.solution.x))
    assert np.all(np.isfinite(result.solution.y))
    assert np.all(np.isfinite(result.solution.z))
    assert np.min(result.solution.y[case.lp.m1 :], initial=0.0) >= -1e-13
    assert result.sigma == 1.0
    assert result.restart_count == 0
    assert all(entry.sigma == 1.0 and entry.restart_count == 0 for entry in result.history)
    assert result.maximum_equality_solve_infinity_residual <= 1e-12
    assert result.maximum_z_x_identity_error <= 1e-12


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda path: path.stem)
def test_cpu_sgs_hpr_matches_highs_and_physics_on_dcopf(config_path: Path) -> None:
    network = load_matpower_case(CASE)
    config = load_dcopf_config(config_path, network)
    model = build_dcopf_model(network, config)
    result = solve_sgs_hpr(
        model.lp,
        sigma=1.0,
        tolerance=PAPER_TOLERANCE,
        kkt_tolerance=DCOPF_KKT_TOLERANCE,
        max_iterations=150_000,
        history_interval=250,
    )
    highs = solve_with_highs(model.lp, tolerance=PAPER_TOLERANCE)
    validation = validate_dcopf_candidate(
        model,
        result.solution.x,
        tolerance=DCOPF_PHYSICAL_TOLERANCE,
    )
    candidate_objective = model.objective(result.solution.x)
    reference_objective = model.objective(highs.state.x)

    assert result.converged
    assert result.residuals.conditions.all_satisfied
    assert result.residuals.combined_norm <= DCOPF_KKT_TOLERANCE
    assert _scaled_gap(candidate_objective, reference_objective) <= (DCOPF_OBJECTIVE_GAP_TOLERANCE)
    assert validation.passed
    assert validation.mode == "approximate_first_order_candidate"
    assert max(family.maximum_violation for family in validation.families) <= (
        DCOPF_PHYSICAL_TOLERANCE
    )
    assert maximum_primal_violation(model.lp, result.solution) <= DCOPF_PHYSICAL_TOLERANCE
    assert np.min(result.solution.y[model.lp.m1 :], initial=0.0) >= -1e-13
    assert result.maximum_equality_solve_infinity_residual <= 2e-12
    assert result.maximum_z_x_identity_error <= 1e-12
    assert result.workspace.equality.full_row_rank
    assert result.workspace.equality.positive_definite
    assert result.workspace.spectral is not None
    assert result.workspace.spectral.power_converged
    assert result.workspace.spectral.maximum_estimate_difference <= 1e-10
    assert result.workspace.spectral.lambda_used > (
        result.workspace.spectral.dense_eigendecomposition
    )
    assert result.workspace.spectral.s2_minimum_eigenvalue > 0.0
    assert result.sigma == 1.0
    assert result.restart_count == 0


def test_cpu_sgs_hpr_repeated_runs_are_numerically_deterministic() -> None:
    case = reference_cases()[0]
    settings = {
        "sigma": 1.0,
        "tolerance": PAPER_TOLERANCE,
        "kkt_tolerance": TOY_KKT_TOLERANCE,
        "max_iterations": 100_000,
        "history_interval": 250,
    }

    first = solve_sgs_hpr(case.lp, **settings)
    second = solve_sgs_hpr(case.lp, **settings)

    assert first.iterations == second.iterations
    np.testing.assert_array_equal(first.solution.x, second.solution.x)
    np.testing.assert_array_equal(first.solution.y, second.solution.y)
    np.testing.assert_array_equal(first.solution.z, second.solution.z)
    for left, right in zip(first.history, second.history, strict=True):
        left_values = left.as_dict()
        right_values = right.as_dict()
        left_values.pop("iteration_loop_elapsed_seconds")
        right_values.pop("iteration_loop_elapsed_seconds")
        assert left_values == right_values
