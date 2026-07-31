import numpy as np
import pytest

from gpu_dcopf_hpr.hpr_generic import HPRState
from gpu_dcopf_hpr.preconditioning import precondition_lp
from gpu_dcopf_hpr.residuals import evaluate_residuals
from gpu_dcopf_hpr.sgs_hpr import prepare_sgs_hpr, solve_sgs_hpr
from gpu_dcopf_hpr.stage5_control import (
    Stage5Control,
    choose_restart_reasons,
    hprlp_sigma_update,
    hprlp_sigma_update_from_scalars,
    sgs_metric_y_quadratic,
    sgs_restart_merit,
    solve_stage5_sgs_hpr,
)
from gpu_dcopf_hpr.toy_problems import analytic_toy_case, inequality_inactive_case


def test_sgs_metric_matches_explicit_block_matrix() -> None:
    lp = analytic_toy_case().lp
    workspace = prepare_sgs_hpr(lp)
    delta_y = np.array([0.7, -0.2])
    A1 = np.asarray(lp.A1)
    A2 = np.asarray(lp.A2)
    assert workspace.spectral is not None
    lambda_used = workspace.spectral.lambda_used

    h11 = A1 @ A1.T
    h12 = A1 @ A2.T
    h21 = h12.T
    explicit = np.block(
        [
            [h11, h12],
            [h21, lambda_used * np.eye(lp.m2) + h21 @ np.linalg.solve(h11, h12)],
        ]
    )
    expected = float(delta_y @ explicit @ delta_y)

    assert sgs_metric_y_quadratic(workspace, delta_y) == pytest.approx(
        expected,
        rel=0.0,
        abs=2e-14,
    )


def test_restart_merit_matches_explicit_metric_quadratic() -> None:
    lp = analytic_toy_case().lp
    workspace = prepare_sgs_hpr(lp)
    delta_x = np.array([0.4, -0.3])
    delta_y = np.array([0.7, -0.2])
    sigma = 0.8
    expected_squared = (
        float(delta_x @ delta_x) / sigma
        + 2.0 * float((lp.dense_A() @ delta_x) @ delta_y)
        + sigma * sgs_metric_y_quadratic(workspace, delta_y)
    )

    assert sgs_restart_merit(
        workspace,
        delta_x=delta_x,
        delta_y=delta_y,
        sigma=sigma,
    ) == pytest.approx(np.sqrt(expected_squared), rel=0.0, abs=2e-14)


def test_restart_criteria_report_every_satisfied_reason_at_boundaries() -> None:
    control = Stage5Control(restart=True)
    reasons = choose_restart_reasons(
        merit=0.2,
        reference_merit=1.0,
        previous_checkpoint_merit=0.19,
        inner_iteration=20,
        total_iteration=100,
        control=control,
    )

    assert reasons == (
        "sufficient_decay",
        "necessary_decay_no_local_progress",
        "long_inner_loop",
    )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("check_interval", 0, "positive integer"),
        ("alpha_sufficient", 0.7, "restart parameters"),
        ("alpha_long", 1.0, "less than 1"),
        ("movement_minimum", 1e13, "movement_minimum"),
        ("infeasibility_ratio_minimum", 1e9, "infeasibility_ratio_minimum"),
    ],
)
def test_stage5_control_rejects_invalid_policy_parameters(
    keyword: str,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Stage5Control(**{keyword: value})


def test_published_sigma_formula_and_guarded_reset() -> None:
    lp = analytic_toy_case().lp
    workspace = prepare_sgs_hpr(lp)
    reference = HPRState(y=[0.1, -0.3], z=[0.0, 0.0], x=[0.2, -0.1])
    candidate = HPRState(y=[0.6, 0.2], z=[0.0, 0.0], x=[0.7, 0.4])
    residuals = evaluate_residuals(
        lp,
        x=candidate.x,
        y=candidate.y,
        z=candidate.z,
    )
    permissive = Stage5Control(
        adaptive_sigma=True,
        movement_minimum=1e-20,
        movement_maximum=1e20,
        infeasibility_ratio_minimum=1e-20,
        infeasibility_ratio_maximum=1e20,
    )
    update = hprlp_sigma_update(
        workspace,
        reference=reference,
        candidate=candidate,
        residuals=residuals,
        sigma_before=3.0,
        control=permissive,
    )
    expected_delta_x = np.linalg.norm(candidate.x - reference.x)
    expected_delta_y = np.sqrt(sgs_metric_y_quadratic(workspace, candidate.y - reference.y))

    assert update.attempted
    assert update.accepted
    assert update.delta_x == pytest.approx(expected_delta_x)
    assert update.delta_y == pytest.approx(expected_delta_y)
    assert update.sigma_after == pytest.approx(expected_delta_x / expected_delta_y)

    reset = hprlp_sigma_update(
        workspace,
        reference=candidate,
        candidate=candidate,
        residuals=residuals,
        sigma_before=3.0,
        control=permissive,
    )
    assert reset.attempted
    assert not reset.accepted
    assert reset.sigma_after == 1.0
    assert "movement guard" in reset.reason


def test_scalar_sigma_update_matches_cpu_wrapper_decision() -> None:
    lp = analytic_toy_case().lp
    workspace = prepare_sgs_hpr(lp)
    reference = HPRState(y=[0.1, -0.3], z=[0.0, 0.0], x=[0.2, -0.1])
    candidate = HPRState(y=[0.6, 0.2], z=[0.0, 0.0], x=[0.7, 0.4])
    residuals = evaluate_residuals(
        lp,
        x=candidate.x,
        y=candidate.y,
        z=candidate.z,
    )
    control = Stage5Control(
        adaptive_sigma=True,
        movement_minimum=1e-20,
        movement_maximum=1e20,
        infeasibility_ratio_minimum=1e-20,
        infeasibility_ratio_maximum=1e20,
    )
    wrapper_update = hprlp_sigma_update(
        workspace,
        reference=reference,
        candidate=candidate,
        residuals=residuals,
        sigma_before=3.0,
        control=control,
    )
    scalar_update = hprlp_sigma_update_from_scalars(
        delta_x=float(np.linalg.norm(candidate.x - reference.x)),
        delta_y=float(np.sqrt(sgs_metric_y_quadratic(workspace, candidate.y - reference.y))),
        primal_infeasibility=residuals.paper_normalized_norms[0],
        dual_infeasibility=residuals.paper_normalized_norms[2],
        sigma_before=3.0,
        control=control,
    )

    assert scalar_update == wrapper_update


def test_scalar_sigma_update_preserves_disabled_and_guarded_decisions() -> None:
    disabled = hprlp_sigma_update_from_scalars(
        delta_x=2.0,
        delta_y=4.0,
        primal_infeasibility=0.25,
        dual_infeasibility=0.5,
        sigma_before=3.0,
        control=Stage5Control(),
    )

    assert not disabled.attempted
    assert not disabled.accepted
    assert disabled.reason == "adaptive sigma disabled"
    assert disabled.delta_x == 2.0
    assert disabled.delta_y == 4.0
    assert disabled.infeasibility_ratio is None
    assert disabled.sigma_after == 3.0

    guarded = hprlp_sigma_update_from_scalars(
        delta_x=0.0,
        delta_y=2.0,
        primal_infeasibility=0.0,
        dual_infeasibility=1.0,
        sigma_before=3.0,
        control=Stage5Control(adaptive_sigma=True),
    )

    assert guarded.attempted
    assert not guarded.accepted
    assert guarded.reason == ("reset to 1 after movement guard and infeasibility-ratio guard")
    assert guarded.infeasibility_ratio is None
    assert guarded.sigma_after == 1.0


def test_disabled_stage5_controls_reproduce_stage3_solver_exactly() -> None:
    lp = inequality_inactive_case().lp
    baseline = solve_sgs_hpr(lp, max_iterations=100, history_interval=100)
    controlled = solve_stage5_sgs_hpr(
        lp,
        max_iterations=100,
        history_interval=100,
        control=Stage5Control(),
    )

    assert controlled.converged == baseline.converged
    assert controlled.iterations == baseline.iterations
    assert controlled.policy_events == ()
    assert controlled.restart_count == 0
    np.testing.assert_array_equal(controlled.solution.x, baseline.solution.x)
    np.testing.assert_array_equal(controlled.solution.y, baseline.solution.y)
    np.testing.assert_array_equal(controlled.solution.z, baseline.solution.z)


def test_restart_policy_uses_exact_100_iteration_cadence() -> None:
    lp = analytic_toy_case().lp
    result = solve_stage5_sgs_hpr(
        lp,
        tolerance=1e-20,
        kkt_tolerance=1e-20,
        max_iterations=101,
        history_interval=100,
        control=Stage5Control(restart=True),
    )

    assert not result.converged
    assert [event.iteration for event in result.policy_events] == [100]
    assert result.policy_events[0].restarted
    assert result.policy_events[0].restart_reasons == ("forced_first",)
    assert result.restart_count == 1
    assert result.history[-1].iteration == 101
    assert result.history[-1].inner_iteration == 1


def test_preconditioned_solver_checks_convergence_in_original_coordinates() -> None:
    lp = inequality_inactive_case().lp
    preconditioner = precondition_lp(
        lp,
        ruiz_iterations=10,
        pock_chambolle=True,
        normalize=True,
    )
    result = solve_stage5_sgs_hpr(
        lp,
        max_iterations=10_000,
        history_interval=100,
        preconditioner=preconditioner,
    )

    assert result.converged
    assert result.preconditioner is preconditioner
    assert result.residuals.conditions.all_satisfied
    assert result.history[-1].original_residuals.stopping_satisfied
    assert result.history[-1].original_variable_objective == pytest.approx(
        float(lp.c @ result.solution.x)
    )
    np.testing.assert_allclose(
        preconditioner.recover_state(result.scaled_solution).x,
        result.solution.x,
        rtol=0.0,
        atol=1e-14,
    )


def test_preconditioned_solver_rejects_raw_structural_backend() -> None:
    lp = analytic_toy_case().lp
    preconditioner = precondition_lp(lp, ruiz_iterations=1)

    with pytest.raises(ValueError, match="cannot be reused"):
        solve_stage5_sgs_hpr(
            lp,
            preconditioner=preconditioner,
            structural_y1=object(),  # type: ignore[arg-type]
            max_iterations=1,
        )
