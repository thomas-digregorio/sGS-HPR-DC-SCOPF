import numpy as np

from gpu_dcopf_hpr.residuals import evaluate_residuals
from gpu_dcopf_hpr.toy_problems import analytic_toy_case


def test_analytic_primal_dual_solution_has_zero_equation_28_residual() -> None:
    case = analytic_toy_case()
    state = case.expected_state

    residuals = evaluate_residuals(case.lp, x=state.x, y=state.y, z=state.z)

    np.testing.assert_allclose(residuals.kkt.dual_projection, 0.0, atol=1e-14)
    np.testing.assert_allclose(residuals.kkt.box, 0.0, atol=1e-14)
    np.testing.assert_allclose(residuals.kkt.stationarity, 0.0, atol=1e-14)
    assert residuals.combined_norm <= 1e-14
    assert residuals.conditions.all_satisfied


def test_equation_28_dual_block_and_equation_54a_have_distinct_signs() -> None:
    case = analytic_toy_case()

    residuals = evaluate_residuals(
        case.lp,
        x=[0.3, 0.7],
        y=[1.5, 0.5],
        z=[0.0, 0.0],
    )

    np.testing.assert_allclose(residuals.kkt.dual_projection, [0.0, -0.2], atol=1e-14)
    np.testing.assert_allclose(
        residuals.paper_raw.primal_feasibility,
        [0.0, 0.2],
        atol=1e-14,
    )
    np.testing.assert_allclose(residuals.kkt.stationarity, 0.0, atol=1e-14)
    assert not residuals.conditions.primal_feasibility


def test_paper_normalizations_use_equation_54_denominators() -> None:
    case = analytic_toy_case()
    x = np.array([0.3, 0.7])
    y = np.array([1.5, 0.5])
    z = np.zeros(2)

    residuals = evaluate_residuals(case.lp, x=x, y=y, z=z)

    expected = 0.2 / (1.0 + np.linalg.norm(case.lp.b))
    assert np.isclose(
        np.linalg.norm(residuals.paper_normalized.primal_feasibility),
        expected,
    )
