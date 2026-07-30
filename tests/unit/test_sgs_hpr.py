import numpy as np
import pytest

from gpu_dcopf_hpr.canonical_lp import CanonicalLP
from gpu_dcopf_hpr.hpr_generic import HPRState
from gpu_dcopf_hpr.projections import project_box, project_nonnegative
from gpu_dcopf_hpr.sgs_hpr import (
    ALGORITHM_2_UPDATE_ORDER,
    estimate_inequality_spectrum,
    prepare_sgs_hpr,
    sgs_hpr_step,
    solve_sgs_hpr,
)
from gpu_dcopf_hpr.toy_problems import (
    analytic_toy_case,
    inequality_inactive_case,
)


def test_algorithm_2_step_matches_independent_equation_by_equation_calculation() -> None:
    lp = analytic_toy_case().lp
    current = HPRState(y=[-0.3, 0.4], z=[0.2, -0.1], x=[0.7, 0.2])
    anchor = HPRState(y=[0.1, 0.2], z=[-0.4, 0.5], x=[0.3, 0.6])
    workspace = prepare_sgs_hpr(lp)
    sigma = 0.7
    iteration = 3

    step = sgs_hpr_step(
        lp,
        current,
        anchor,
        workspace,
        iteration=iteration,
        sigma=sigma,
    )

    A1 = np.asarray(lp.A1)
    A2 = np.asarray(lp.A2)
    q = current.x + sigma * (A1.T @ current.y[: lp.m1] + A2.T @ current.y[lp.m1 :] - lp.c)
    expected_z = (project_box(q, lp.lower, lp.upper) - q) / sigma
    expected_x = current.x + sigma * (
        A1.T @ current.y[: lp.m1] + A2.T @ current.y[lp.m1 :] + expected_z - lp.c
    )
    gram = A1 @ A1.T
    first_rhs = (
        lp.b1 - A1 @ (expected_x + sigma * (A2.T @ current.y[lp.m1 :] + expected_z - lp.c))
    ) / sigma
    expected_y1_half = np.linalg.solve(gram, first_rhs)
    ry = (
        expected_x / sigma + A1.T @ expected_y1_half + A2.T @ current.y[lp.m1 :] + expected_z - lp.c
    )
    assert workspace.spectral is not None
    expected_y2 = project_nonnegative(
        current.y[lp.m1 :] + (lp.b2 / sigma - A2 @ ry) / workspace.spectral.lambda_used
    )
    second_rhs = (
        lp.b1 - A1 @ (expected_x + sigma * (A2.T @ expected_y2 + expected_z - lp.c))
    ) / sigma
    expected_y1 = np.linalg.solve(gram, second_rhs)
    expected_y = np.concatenate((expected_y1, expected_y2))
    expected_reflected_y = 2.0 * expected_y - current.y
    expected_reflected_z = 2.0 * expected_z - current.z
    expected_reflected_x = 2.0 * expected_x - current.x
    anchor_weight = 1.0 / (iteration + 2.0)
    reflected_weight = (iteration + 1.0) / (iteration + 2.0)

    np.testing.assert_allclose(step.y1_half, expected_y1_half, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(step.proximal.y, expected_y, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(step.proximal.z, expected_z, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(step.proximal.x, expected_x, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(step.reflected.y, expected_reflected_y, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(step.reflected.z, expected_reflected_z, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(step.reflected.x, expected_reflected_x, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        step.next_state.y,
        anchor_weight * anchor.y + reflected_weight * expected_reflected_y,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        step.next_state.z,
        anchor_weight * anchor.z + reflected_weight * expected_reflected_z,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        step.next_state.x,
        anchor_weight * anchor.x + reflected_weight * expected_reflected_x,
        rtol=0.0,
        atol=1e-14,
    )
    assert step.update_order == ALGORITHM_2_UPDATE_ORDER
    assert step.update_order == (
        "z_bar",
        "x_bar",
        "y1_half",
        "y2_bar",
        "y1_bar",
        "reflection",
        "halpern_anchor",
    )
    assert step.z_x_identity_error <= 1e-15
    assert step.first_equality_relative_residual <= 1e-15
    assert step.second_equality_relative_residual <= 1e-15
    assert step.first_equality_infinity_residual <= 1e-15
    assert step.second_equality_infinity_residual <= 1e-15


def test_z_and_x_equations_match_on_seeded_random_states() -> None:
    lp = analytic_toy_case().lp
    workspace = prepare_sgs_hpr(lp)
    generator = np.random.default_rng(20260729)
    anchor = HPRState(y=np.zeros(lp.m), z=np.zeros(lp.n), x=np.zeros(lp.n))

    for iteration in range(25):
        current = HPRState(
            y=generator.normal(size=lp.m),
            z=generator.normal(size=lp.n),
            x=generator.normal(size=lp.n),
        )
        step = sgs_hpr_step(
            lp,
            current,
            anchor,
            workspace,
            iteration=iteration,
            sigma=0.3 + 0.1 * iteration,
        )
        assert step.z_x_identity_error <= 2e-15


def test_equality_system_is_spd_and_direct_sweeps_are_accurate() -> None:
    lp = analytic_toy_case().lp
    workspace = prepare_sgs_hpr(lp)
    state = HPRState(y=np.zeros(lp.m), z=np.zeros(lp.n), x=np.zeros(lp.n))
    step = sgs_hpr_step(lp, state, state, workspace, iteration=0, sigma=1.0)

    assert workspace.equality.full_row_rank
    assert workspace.equality.positive_definite
    assert workspace.equality.symmetry_error == 0.0
    assert workspace.equality.minimum_eigenvalue is not None
    assert workspace.equality.minimum_eigenvalue > 0.0
    assert step.first_equality_relative_residual <= 1e-15
    assert step.second_equality_relative_residual <= 1e-15


def test_rank_deficient_equality_block_is_rejected() -> None:
    lp = CanonicalLP(
        c=[1.0, 1.0],
        A1=[[1.0, 1.0], [2.0, 2.0]],
        b1=[1.0, 2.0],
        A2=[[1.0, 0.0]],
        b2=[0.0],
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
    )

    with pytest.raises(ValueError, match="full row rank"):
        prepare_sgs_hpr(lp)


def test_step_rejects_workspace_prepared_for_a_different_lp() -> None:
    first = analytic_toy_case().lp
    second = analytic_toy_case().lp
    workspace = prepare_sgs_hpr(first)
    state = HPRState(y=np.zeros(second.m), z=np.zeros(second.n), x=np.zeros(second.n))

    with pytest.raises(ValueError, match="same CanonicalLP instance"):
        sgs_hpr_step(second, state, state, workspace, iteration=0, sigma=1.0)


def test_three_spectral_estimators_agree_and_lambda_is_conservative() -> None:
    A2 = np.diag([3.0, 2.0, 0.5])
    diagnostics = estimate_inequality_spectrum(A2)

    assert diagnostics.power_converged
    assert diagnostics.power_iterations > 1
    assert diagnostics.dense_eigendecomposition == pytest.approx(9.0, abs=1e-13)
    assert diagnostics.sparse_eigsh == pytest.approx(9.0, abs=1e-12)
    assert diagnostics.power_iteration == pytest.approx(9.0, abs=1e-10)
    assert diagnostics.maximum_estimate_difference <= 1e-10
    assert diagnostics.lambda_used > diagnostics.dense_eigendecomposition
    assert diagnostics.safety_margin > 0.0
    assert diagnostics.s2_minimum_eigenvalue > 0.0


def test_solver_supports_no_inequality_rows() -> None:
    lp = CanonicalLP(
        c=[-1.0, 0.0],
        A1=[[1.0, 1.0]],
        b1=[1.0],
        A2=np.empty((0, 2)),
        b2=[],
        lower=[0.0, 0.0],
        upper=[0.75, 1.0],
    )
    workspace = prepare_sgs_hpr(lp)
    result = solve_sgs_hpr(lp, max_iterations=20_000)

    assert workspace.spectral is None
    assert result.converged
    assert result.solution.y.shape == (1,)
    assert result.solution.x.shape == (2,)


def test_step_supports_no_equality_rows() -> None:
    lp = CanonicalLP(
        c=[1.0],
        A1=np.empty((0, 1)),
        b1=[],
        A2=[[1.0]],
        b2=[0.25],
        lower=[0.0],
        upper=[1.0],
    )
    workspace = prepare_sgs_hpr(lp)
    state = HPRState(y=[0.0], z=[0.0], x=[0.0])
    step = sgs_hpr_step(lp, state, state, workspace, iteration=0, sigma=1.0)

    assert workspace.equality.full_row_rank
    assert workspace.equality.positive_definite
    assert step.y1_half.shape == (0,)
    assert step.proximal.y.shape == (1,)
    assert step.first_equality_infinity_residual == 0.0
    assert step.second_equality_infinity_residual == 0.0


def test_stopping_is_checked_every_iteration_but_history_is_sparse() -> None:
    lp = inequality_inactive_case().lp
    first = solve_sgs_hpr(lp, history_interval=100, max_iterations=100)
    second = solve_sgs_hpr(lp, history_interval=100, max_iterations=100)

    assert first.converged
    assert first.iterations == 4
    assert [entry.iteration for entry in first.history] == [1, 4]
    assert first.history_interval == 100
    np.testing.assert_array_equal(first.solution.x, second.solution.x)
    np.testing.assert_array_equal(first.solution.y, second.solution.y)
    np.testing.assert_array_equal(first.solution.z, second.solution.z)
    assert [entry.iteration for entry in first.history] == [
        entry.iteration for entry in second.history
    ]
