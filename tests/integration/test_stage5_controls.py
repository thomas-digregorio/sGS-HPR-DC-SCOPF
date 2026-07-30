import numpy as np
import pytest

from gpu_dcopf_hpr import (
    LPPreconditioner,
    Stage5Control,
    precondition_lp,
    solve_sgs_hpr,
    solve_stage5_sgs_hpr,
)
from gpu_dcopf_hpr.toy_problems import analytic_toy_case, inequality_inactive_case
from gpu_dcopf_hpr.validation import maximum_primal_violation, solve_with_highs


@pytest.mark.parametrize(
    ("adaptive_sigma", "restart"),
    (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ),
    ids=(
        "fixed_no_restart",
        "adaptive_no_restart",
        "fixed_restart",
        "adaptive_restart",
    ),
)
def test_all_penalty_and_restart_combinations_run_deterministically(
    adaptive_sigma: bool,
    restart: bool,
) -> None:
    lp = analytic_toy_case().lp
    control = Stage5Control(
        adaptive_sigma=adaptive_sigma,
        restart=restart,
    )
    settings = {
        "sigma": 1.0,
        "tolerance": 1e-20,
        "kkt_tolerance": 1e-20,
        "max_iterations": 101,
        "history_interval": 100,
        "control": control,
    }

    first = solve_stage5_sgs_hpr(lp, **settings)
    second = solve_stage5_sgs_hpr(lp, **settings)

    assert not first.converged
    assert first.iterations == second.iterations == 101
    assert first.control is control
    assert first.initial_sigma == 1.0
    assert np.isfinite(first.sigma) and first.sigma > 0.0
    assert np.isfinite(first.minimum_sigma) and first.minimum_sigma > 0.0
    assert np.isfinite(first.maximum_sigma) and first.maximum_sigma > 0.0
    np.testing.assert_array_equal(first.solution.x, second.solution.x)
    np.testing.assert_array_equal(first.solution.y, second.solution.y)
    np.testing.assert_array_equal(first.solution.z, second.solution.z)

    expected_event_iterations = [100] if control.enabled else []
    assert [event.iteration for event in first.policy_events] == expected_event_iterations
    assert [event.as_dict() for event in first.policy_events] == [
        event.as_dict() for event in second.policy_events
    ]
    if control.enabled:
        update = first.policy_events[0].sigma_update
        assert update.attempted is adaptive_sigma
    if restart:
        assert first.restart_count == 1
        assert first.policy_events[0].restarted
    else:
        assert first.restart_count == 0
        assert all(not event.restarted for event in first.policy_events)


def test_complete_preconditioning_converges_and_validates_in_original_coordinates() -> None:
    case = inequality_inactive_case()
    lp = case.lp
    preconditioner = precondition_lp(
        lp,
        ruiz_iterations=10,
        pock_chambolle=True,
        normalize=True,
    )

    result = solve_stage5_sgs_hpr(
        lp,
        preconditioner=preconditioner,
        tolerance=5e-5,
        kkt_tolerance=2.5e-4,
        max_iterations=10_000,
        history_interval=100,
    )
    highs = solve_with_highs(lp, tolerance=5e-5)
    candidate_objective = float(lp.c @ result.solution.x)
    reference_objective = float(highs.objective)
    scaled_objective = float(preconditioner.scaled_lp.c @ result.scaled_solution.x)

    assert isinstance(result.preconditioner, LPPreconditioner)
    assert result.preconditioner is preconditioner
    assert result.converged
    assert result.residuals.conditions.all_satisfied
    assert result.residuals.combined_norm <= 2.5e-4
    assert maximum_primal_violation(lp, result.solution) <= 2.5e-4
    assert (
        abs(candidate_objective - reference_objective)
        / max(
            1.0,
            abs(reference_objective),
        )
        <= 5e-4
    )
    assert preconditioner.original_objective_from_scaled(scaled_objective) == pytest.approx(
        candidate_objective, rel=0.0, abs=1e-14
    )
    np.testing.assert_allclose(
        preconditioner.recover_state(result.scaled_solution).x,
        result.solution.x,
        rtol=0.0,
        atol=1e-14,
    )
    assert result.history[-1].original_residuals.stopping_satisfied
    assert result.history[-1].original_variable_objective == pytest.approx(
        candidate_objective,
        rel=0.0,
        abs=1e-14,
    )


def test_fixed_sigma_no_restart_control_is_exact_stage3_regression() -> None:
    lp = analytic_toy_case().lp
    settings = {
        "sigma": 1.0,
        "tolerance": 1e-20,
        "kkt_tolerance": 1e-20,
        "max_iterations": 500,
        "history_interval": 100,
    }

    baseline = solve_sgs_hpr(lp, **settings)
    controlled = solve_stage5_sgs_hpr(
        lp,
        control=Stage5Control(),
        **settings,
    )

    assert not baseline.converged
    assert controlled.converged == baseline.converged
    assert controlled.iterations == baseline.iterations == 500
    assert controlled.sigma == baseline.sigma == 1.0
    assert controlled.restart_count == baseline.restart_count == 0
    assert controlled.policy_events == ()
    np.testing.assert_array_equal(controlled.solution.x, baseline.solution.x)
    np.testing.assert_array_equal(controlled.solution.y, baseline.solution.y)
    np.testing.assert_array_equal(controlled.solution.z, baseline.solution.z)
    np.testing.assert_array_equal(controlled.current_state.x, baseline.current_state.x)
    np.testing.assert_array_equal(controlled.current_state.y, baseline.current_state.y)
    np.testing.assert_array_equal(controlled.current_state.z, baseline.current_state.z)
    assert [entry.iteration for entry in controlled.history] == [
        entry.iteration for entry in baseline.history
    ]
    for stage5_entry, baseline_entry in zip(
        controlled.history,
        baseline.history,
        strict=True,
    ):
        assert stage5_entry.original_variable_objective == (
            baseline_entry.canonical_variable_objective
        )
        assert stage5_entry.original_residuals.kkt_combined_norm == (
            baseline_entry.kkt_combined_norm
        )
        assert stage5_entry.sigma == baseline_entry.sigma == 1.0
        assert stage5_entry.restart_count == baseline_entry.restart_count == 0


def test_policy_events_use_exact_completed_iteration_cadence() -> None:
    lp = analytic_toy_case().lp
    result = solve_stage5_sgs_hpr(
        lp,
        tolerance=1e-20,
        kkt_tolerance=1e-20,
        max_iterations=301,
        history_interval=250,
        control=Stage5Control(
            adaptive_sigma=True,
            restart=False,
            check_interval=100,
        ),
    )

    assert not result.converged
    assert result.iterations == 301
    assert [event.iteration for event in result.policy_events] == [100, 200, 300]
    assert [event.inner_iteration for event in result.policy_events] == [100, 200, 300]
    assert all(event.iteration % 100 == 0 for event in result.policy_events)
    assert all(not event.restarted for event in result.policy_events)
    assert all(event.sigma_update.attempted for event in result.policy_events)
    assert {entry.iteration for entry in result.history}.issuperset({1, 100, 200, 250, 300, 301})
