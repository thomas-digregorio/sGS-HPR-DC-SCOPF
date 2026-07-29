import numpy as np
import pytest

from gpu_dcopf_hpr.hpr_generic import solve_hpr
from gpu_dcopf_hpr.toy_problems import reference_cases
from gpu_dcopf_hpr.validation import maximum_primal_violation, solve_with_highs


@pytest.mark.parametrize("case", reference_cases(), ids=lambda case: case.name)
def test_generic_hpr_matches_analytic_and_highs_references(case: object) -> None:
    highs = solve_with_highs(case.lp)
    hpr = solve_hpr(case.lp, max_iterations=100_000, kkt_tolerance=2.5e-4)

    assert hpr.converged
    assert hpr.residuals.conditions.all_satisfied
    assert hpr.residuals.combined_norm <= 2.5e-4
    assert maximum_primal_violation(case.lp, hpr.solution) <= 2.5e-4
    assert np.all(np.isfinite(hpr.solution.x))
    assert np.all(np.isfinite(hpr.solution.y))
    assert np.all(np.isfinite(hpr.solution.z))
    if case.lp.m2:
        assert np.min(hpr.solution.y[case.lp.m1 :]) >= -1e-12

    scaled_objective_gap = abs(case.lp.c @ hpr.solution.x - highs.objective) / (
        1.0 + abs(highs.objective)
    )
    assert scaled_objective_gap <= 2e-4
    np.testing.assert_allclose(
        hpr.solution.x,
        case.expected_state.x,
        rtol=0.0,
        atol=case.solution_tolerance,
    )
    assert np.isclose(highs.objective, case.expected_objective, rtol=0.0, atol=1e-10)
    assert highs.residuals.combined_norm <= 1e-8


def test_repeated_hpr_runs_are_deterministic_without_monotonicity_assumption() -> None:
    case = reference_cases()[0]

    first = solve_hpr(case.lp, max_iterations=100_000)
    second = solve_hpr(case.lp, max_iterations=100_000)

    assert first.converged and second.converged
    assert first.iterations == second.iterations
    np.testing.assert_array_equal(first.solution.x, second.solution.x)
    np.testing.assert_array_equal(first.solution.y, second.solution.y)
    np.testing.assert_array_equal(first.solution.z, second.solution.z)
    np.testing.assert_array_equal(
        [entry.kkt_combined_norm for entry in first.history],
        [entry.kkt_combined_norm for entry in second.history],
    )
