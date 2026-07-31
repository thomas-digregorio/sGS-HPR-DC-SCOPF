"""Sourced Stage 5 restart and penalty controls for CPU sGS-HPR.

The fixed-penalty, no-restart implementation in :mod:`sgs_hpr` remains the
reference baseline.  This module adds the restart criteria and penalty update
published for HPR-LP while keeping every policy decision explicit.

Sources:

* HPR-LP restart Eqs. (10)-(12), penalty Eqs. (15)-(18), and parameters:
  https://doi.org/10.1007/s12532-025-00292-0
* The DCOPF manuscript fixes the policy-check interval at 100 iterations.

The DCOPF paper does not publish its exact policy code.  Consequently, these
controls are a sourced HPR-LP transfer, not a claim of byte-for-byte identity
with the authors' unpublished sGS-HPR implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from .canonical_lp import CanonicalLP
from .hpr_generic import HPRState
from .preconditioning import LPPreconditioner
from .residuals import ResidualEvaluation, evaluate_residuals
from .sgs_hpr import (
    SGSHPRWorkspace,
    prepare_sgs_hpr,
    sgs_hpr_step,
)
from .structural_y1 import StructuralY1Solver

FloatVector = NDArray[np.float64]
RestartReason = Literal[
    "forced_first",
    "sufficient_decay",
    "necessary_decay_no_local_progress",
    "long_inner_loop",
]


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return result


@dataclass(frozen=True, slots=True)
class Stage5Control:
    """Configuration for the published HPR-LP policy transfer."""

    adaptive_sigma: bool = False
    restart: bool = False
    check_interval: int = 100
    alpha_sufficient: float = 0.2
    alpha_necessary: float = 0.6
    alpha_long: float = 0.2
    movement_minimum: float = 1e-16
    movement_maximum: float = 1e12
    infeasibility_ratio_minimum: float = 1e-8
    infeasibility_ratio_maximum: float = 1e8

    def __post_init__(self) -> None:
        if not isinstance(self.check_interval, int) or self.check_interval <= 0:
            raise ValueError("check_interval must be a positive integer.")
        alpha_sufficient = _positive_finite(
            self.alpha_sufficient,
            name="alpha_sufficient",
        )
        alpha_necessary = _positive_finite(
            self.alpha_necessary,
            name="alpha_necessary",
        )
        alpha_long = _positive_finite(self.alpha_long, name="alpha_long")
        if not alpha_sufficient < alpha_necessary < 1.0:
            raise ValueError(
                "restart parameters must satisfy 0 < alpha_sufficient < alpha_necessary < 1."
            )
        if alpha_long >= 1.0:
            raise ValueError("alpha_long must be less than 1.")
        movement_minimum = _positive_finite(
            self.movement_minimum,
            name="movement_minimum",
        )
        movement_maximum = _positive_finite(
            self.movement_maximum,
            name="movement_maximum",
        )
        if movement_minimum >= movement_maximum:
            raise ValueError("movement_minimum must be less than movement_maximum.")
        ratio_minimum = _positive_finite(
            self.infeasibility_ratio_minimum,
            name="infeasibility_ratio_minimum",
        )
        ratio_maximum = _positive_finite(
            self.infeasibility_ratio_maximum,
            name="infeasibility_ratio_maximum",
        )
        if ratio_minimum >= ratio_maximum:
            raise ValueError(
                "infeasibility_ratio_minimum must be less than infeasibility_ratio_maximum."
            )

    @property
    def enabled(self) -> bool:
        return self.adaptive_sigma or self.restart

    def summary(self) -> dict[str, Any]:
        return {
            "adaptive_sigma": self.adaptive_sigma,
            "restart": self.restart,
            "check_interval": self.check_interval,
            "alpha_sufficient": self.alpha_sufficient,
            "alpha_necessary": self.alpha_necessary,
            "alpha_long": self.alpha_long,
            "movement_range": [
                self.movement_minimum,
                self.movement_maximum,
            ],
            "infeasibility_ratio_range": [
                self.infeasibility_ratio_minimum,
                self.infeasibility_ratio_maximum,
            ],
            "source": "HPR-LP Eqs. (10)-(18); DCOPF check interval 100",
        }


@dataclass(frozen=True, slots=True)
class ResidualSnapshot:
    """Scalar residual record without retaining full residual vectors."""

    kkt_combined_norm: float
    normalized_combined_norm: float
    primal_raw: float
    box_raw: float
    stationarity_raw: float
    primal_normalized: float
    box_normalized: float
    stationarity_normalized: float
    stopping_satisfied: bool

    @classmethod
    def from_evaluation(cls, value: ResidualEvaluation) -> ResidualSnapshot:
        raw = value.paper_raw_norms
        normalized = value.paper_normalized_norms
        return cls(
            kkt_combined_norm=value.combined_norm,
            normalized_combined_norm=value.normalized_combined_norm,
            primal_raw=raw[0],
            box_raw=raw[1],
            stationarity_raw=raw[2],
            primal_normalized=normalized[0],
            box_normalized=normalized[1],
            stationarity_normalized=normalized[2],
            stopping_satisfied=value.conditions.all_satisfied,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kkt_combined_norm": self.kkt_combined_norm,
            "normalized_combined_norm": self.normalized_combined_norm,
            "paper_raw": {
                "primal_feasibility": self.primal_raw,
                "box": self.box_raw,
                "stationarity": self.stationarity_raw,
            },
            "paper_normalized": {
                "primal_feasibility": self.primal_normalized,
                "box": self.box_normalized,
                "stationarity": self.stationarity_normalized,
            },
            "paper_stopping_satisfied": self.stopping_satisfied,
        }


@dataclass(frozen=True, slots=True)
class SigmaUpdate:
    """One application or guarded reset of HPR-LP Eqs. (15)-(18)."""

    attempted: bool
    accepted: bool
    reason: str
    delta_x: float
    delta_y: float
    primal_infeasibility: float
    dual_infeasibility: float
    infeasibility_ratio: float | None
    sigma_before: float
    sigma_after: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "reason": self.reason,
            "delta_x": self.delta_x,
            "delta_y": self.delta_y,
            "primal_infeasibility": self.primal_infeasibility,
            "dual_infeasibility": self.dual_infeasibility,
            "infeasibility_ratio": self.infeasibility_ratio,
            "sigma_before": self.sigma_before,
            "sigma_after": self.sigma_after,
        }


@dataclass(frozen=True, slots=True)
class Stage5PolicyEvent:
    """One 100-iteration policy checkpoint."""

    iteration: int
    inner_iteration: int
    merit: float
    reference_merit: float
    previous_checkpoint_merit: float | None
    restart_reasons: tuple[RestartReason, ...]
    restarted: bool
    restart_count: int
    sigma_update: SigmaUpdate

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "inner_iteration": self.inner_iteration,
            "merit": self.merit,
            "reference_merit": self.reference_merit,
            "previous_checkpoint_merit": self.previous_checkpoint_merit,
            "restart_reasons": list(self.restart_reasons),
            "restarted": self.restarted,
            "restart_count": self.restart_count,
            "sigma_update": self.sigma_update.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Stage5HistoryEntry:
    """Sparse iteration history in scaled and recovered original coordinates."""

    iteration: int
    inner_iteration: int
    iteration_loop_elapsed_seconds: float
    scaled_variable_objective: float
    original_variable_objective: float
    scaled_residuals: ResidualSnapshot
    original_residuals: ResidualSnapshot
    kkt_target_satisfied: bool
    maximum_equality_solve_relative_residual: float
    maximum_equality_solve_infinity_residual: float
    minimum_original_inequality_multiplier: float | None
    sigma: float
    restart_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "inner_iteration": self.inner_iteration,
            "iteration_loop_elapsed_seconds": self.iteration_loop_elapsed_seconds,
            "scaled_variable_objective": self.scaled_variable_objective,
            "original_variable_objective": self.original_variable_objective,
            "scaled_residuals": self.scaled_residuals.as_dict(),
            "original_residuals": self.original_residuals.as_dict(),
            "kkt_target_satisfied": self.kkt_target_satisfied,
            "maximum_equality_solve_relative_residual": (
                self.maximum_equality_solve_relative_residual
            ),
            "maximum_equality_solve_infinity_residual": (
                self.maximum_equality_solve_infinity_residual
            ),
            "minimum_original_inequality_multiplier": (self.minimum_original_inequality_multiplier),
            "sigma": self.sigma,
            "restart_count": self.restart_count,
        }


@dataclass(frozen=True, slots=True)
class Stage5SGSHPRResult:
    """Controlled solver result with explicit scaled/original separation."""

    solution: HPRState
    scaled_solution: HPRState
    current_state: HPRState
    scaled_current_state: HPRState
    residuals: ResidualEvaluation
    scaled_residuals: ResidualEvaluation
    history: tuple[Stage5HistoryEntry, ...]
    policy_events: tuple[Stage5PolicyEvent, ...]
    iterations: int
    converged: bool
    sigma: float
    initial_sigma: float
    minimum_sigma: float
    maximum_sigma: float
    restart_count: int
    history_interval: int
    workspace: SGSHPRWorkspace
    control: Stage5Control
    preconditioner: LPPreconditioner | None
    preparation_elapsed_seconds: float
    total_elapsed_seconds: float
    maximum_equality_solve_relative_residual: float
    maximum_equality_solve_infinity_residual: float
    maximum_z_x_identity_error: float


def _validate_state(lp: CanonicalLP, state: HPRState, *, name: str) -> None:
    if state.y.shape != (lp.m,):
        raise ValueError(f"{name}.y must have shape ({lp.m},); received {state.y.shape}.")
    if state.z.shape != (lp.n,):
        raise ValueError(f"{name}.z must have shape ({lp.n},); received {state.z.shape}.")
    if state.x.shape != (lp.n,):
        raise ValueError(f"{name}.x must have shape ({lp.n},); received {state.x.shape}.")


def _matvec(matrix: Any, vector: FloatVector) -> FloatVector:
    return np.asarray(matrix @ vector, dtype=np.float64).reshape(-1)


def _solve_equality_metric(
    workspace: SGSHPRWorkspace,
    right_hand_side: FloatVector,
) -> FloatVector:
    if workspace.equality.rows == 0:
        return np.empty(0, dtype=np.float64)
    if workspace.equality_backend == "structural":
        assert workspace.structural_y1 is not None
        return workspace.structural_y1.solve(right_hand_side)
    assert workspace.equality_cholesky is not None
    return np.asarray(
        linalg.cho_solve(
            workspace.equality_cholesky,
            right_hand_side,
            check_finite=True,
        ),
        dtype=np.float64,
    )


def sgs_metric_y_quadratic(
    workspace: SGSHPRWorkspace,
    delta_y: FloatVector,
) -> float:
    """Evaluate ``dy' (AA' + T1) dy`` for the paper's sGS operator.

    The sGS decomposition theorem gives

    ``AA' + T1 = (D + U)' D^-1 (D + U)``.

    For Algorithm 2, the second diagonal block is ``lambda I``.  This
    expression therefore needs only sparse products and the already selected
    equality solve; it never materializes ``T1``.
    """

    vector = np.asarray(delta_y, dtype=np.float64)
    if vector.shape != (workspace.source_lp.m,):
        raise ValueError(
            f"delta_y must match the workspace multiplier dimension; received {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("delta_y must contain only finite values.")

    m1 = workspace.source_lp.m1
    dy1 = vector[:m1]
    dy2 = vector[m1:]
    at_dy = _matvec(workspace.A1_transpose, dy1) + _matvec(
        workspace.A2_transpose,
        dy2,
    )
    first_block = _matvec(workspace.A1, at_dy)
    equality_term = 0.0
    if m1:
        solved = _solve_equality_metric(workspace, first_block)
        equality_term = float(np.dot(first_block, solved))

    inequality_term = 0.0
    if workspace.source_lp.m2:
        assert workspace.spectral is not None
        inequality_term = workspace.spectral.lambda_used * float(np.dot(dy2, dy2))

    value = equality_term + inequality_term
    scale = max(1.0, abs(equality_term), abs(inequality_term))
    tolerance = 500.0 * np.finfo(np.float64).eps * scale
    if value < -tolerance:
        raise FloatingPointError(
            "sGS metric quadratic became materially negative; "
            f"value={value}, tolerance={tolerance}."
        )
    return max(value, 0.0)


def sgs_restart_merit(
    workspace: SGSHPRWorkspace,
    *,
    delta_x: FloatVector,
    delta_y: FloatVector,
    sigma: float,
) -> float:
    """Evaluate the HPR-LP restart merit ``||w - w_hat||_M``."""

    sigma_value = _positive_finite(sigma, name="sigma")
    dx = np.asarray(delta_x, dtype=np.float64)
    dy = np.asarray(delta_y, dtype=np.float64)
    if dx.shape != (workspace.source_lp.n,):
        raise ValueError(f"delta_x must match the workspace primal dimension; received {dx.shape}.")
    if not np.all(np.isfinite(dx)):
        raise ValueError("delta_x must contain only finite values.")

    h_quadratic = sgs_metric_y_quadratic(workspace, dy)
    a_dx = np.concatenate(
        (
            _matvec(workspace.A1, dx),
            _matvec(workspace.A2, dx),
        )
    )
    primal_term = float(np.dot(dx, dx)) / sigma_value
    cross_term = 2.0 * float(np.dot(a_dx, dy))
    dual_term = sigma_value * h_quadratic
    squared = primal_term + cross_term + dual_term
    scale = max(1.0, abs(primal_term), abs(cross_term), abs(dual_term))
    tolerance = 1000.0 * np.finfo(np.float64).eps * scale
    if squared < -tolerance:
        raise FloatingPointError(
            f"restart metric became materially negative; value={squared}, tolerance={tolerance}."
        )
    return float(np.sqrt(max(squared, 0.0)))


def choose_restart_reasons(
    *,
    merit: float,
    reference_merit: float,
    previous_checkpoint_merit: float | None,
    inner_iteration: int,
    total_iteration: int,
    control: Stage5Control,
) -> tuple[RestartReason, ...]:
    """Apply HPR-LP Eqs. (10)-(12) without hidden priority rules."""

    values = (merit, reference_merit)
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("merit values must be finite and nonnegative.")
    if previous_checkpoint_merit is not None and (
        not np.isfinite(previous_checkpoint_merit) or previous_checkpoint_merit < 0.0
    ):
        raise ValueError("previous_checkpoint_merit must be None or finite and nonnegative.")
    if inner_iteration <= 0 or total_iteration <= 0:
        raise ValueError("iteration counters must be positive.")

    reasons: list[RestartReason] = []
    if merit <= control.alpha_sufficient * reference_merit:
        reasons.append("sufficient_decay")
    if (
        previous_checkpoint_merit is not None
        and merit <= control.alpha_necessary * reference_merit
        and merit > previous_checkpoint_merit
    ):
        reasons.append("necessary_decay_no_local_progress")
    if inner_iteration >= control.alpha_long * total_iteration:
        reasons.append("long_inner_loop")
    return tuple(reasons)


def hprlp_sigma_update_from_scalars(
    *,
    delta_x: float,
    delta_y: float,
    primal_infeasibility: float,
    dual_infeasibility: float,
    sigma_before: float,
    control: Stage5Control,
) -> SigmaUpdate:
    """Apply HPR-LP Eqs. (15)-(18) from precomputed scalar diagnostics.

    This decision-only boundary is shared by CPU and accelerator control paths.
    Callers remain responsible for computing ``delta_y`` in the appropriate
    multiplier metric and for supplying normalized primal and dual infeasibility.
    """

    sigma_value = _positive_finite(sigma_before, name="sigma_before")
    if not control.adaptive_sigma:
        return SigmaUpdate(
            attempted=False,
            accepted=False,
            reason="adaptive sigma disabled",
            delta_x=delta_x,
            delta_y=delta_y,
            primal_infeasibility=primal_infeasibility,
            dual_infeasibility=dual_infeasibility,
            infeasibility_ratio=None,
            sigma_before=sigma_value,
            sigma_after=sigma_value,
        )

    ratio = dual_infeasibility / primal_infeasibility if primal_infeasibility > 0.0 else None

    movements_ok = (
        control.movement_minimum < delta_x < control.movement_maximum
        and control.movement_minimum < delta_y < control.movement_maximum
    )
    ratio_ok = (
        ratio is not None
        and np.isfinite(ratio)
        and control.infeasibility_ratio_minimum < ratio < control.infeasibility_ratio_maximum
    )
    if movements_ok and ratio_ok:
        sigma_after = delta_x / delta_y
        if not np.isfinite(sigma_after) or sigma_after <= 0.0:
            movements_ok = False
        else:
            return SigmaUpdate(
                attempted=True,
                accepted=True,
                reason="accepted HPR-LP Eq. (16)",
                delta_x=delta_x,
                delta_y=delta_y,
                primal_infeasibility=primal_infeasibility,
                dual_infeasibility=dual_infeasibility,
                infeasibility_ratio=ratio,
                sigma_before=sigma_value,
                sigma_after=float(sigma_after),
            )

    failed: list[str] = []
    if not movements_ok:
        failed.append("movement guard")
    if not ratio_ok:
        failed.append("infeasibility-ratio guard")
    return SigmaUpdate(
        attempted=True,
        accepted=False,
        reason="reset to 1 after " + " and ".join(failed),
        delta_x=delta_x,
        delta_y=delta_y,
        primal_infeasibility=primal_infeasibility,
        dual_infeasibility=dual_infeasibility,
        infeasibility_ratio=ratio,
        sigma_before=sigma_value,
        sigma_after=1.0,
    )


def hprlp_sigma_update(
    workspace: SGSHPRWorkspace,
    *,
    reference: HPRState,
    candidate: HPRState,
    residuals: ResidualEvaluation,
    sigma_before: float,
    control: Stage5Control,
) -> SigmaUpdate:
    """Compute sGS diagnostics and apply published HPR-LP Eqs. (15)-(18)."""

    sigma_value = _positive_finite(sigma_before, name="sigma_before")
    if not control.adaptive_sigma:
        return hprlp_sigma_update_from_scalars(
            delta_x=0.0,
            delta_y=0.0,
            primal_infeasibility=residuals.paper_normalized_norms[0],
            dual_infeasibility=residuals.paper_normalized_norms[2],
            sigma_before=sigma_value,
            control=control,
        )

    _validate_state(workspace.source_lp, reference, name="reference")
    _validate_state(workspace.source_lp, candidate, name="candidate")
    delta_x = float(np.linalg.norm(candidate.x - reference.x))
    delta_y = float(
        np.sqrt(
            sgs_metric_y_quadratic(
                workspace,
                candidate.y - reference.y,
            )
        )
    )
    primal = residuals.paper_normalized_norms[0]
    dual = residuals.paper_normalized_norms[2]
    return hprlp_sigma_update_from_scalars(
        delta_x=delta_x,
        delta_y=delta_y,
        primal_infeasibility=primal,
        dual_infeasibility=dual,
        sigma_before=sigma_value,
        control=control,
    )


def _recover(
    state: HPRState,
    preconditioner: LPPreconditioner | None,
) -> HPRState:
    if preconditioner is None:
        return state.detached_copy()
    return preconditioner.recover_state(state)


def _algorithm_state(
    state: HPRState,
    preconditioner: LPPreconditioner | None,
) -> HPRState:
    if preconditioner is None:
        return state.detached_copy()
    return preconditioner.scale_state(state)


def solve_stage5_sgs_hpr(
    lp: CanonicalLP,
    *,
    sigma: float = 1.0,
    tolerance: float = 5e-5,
    kkt_tolerance: float | None = None,
    max_iterations: int = 200_000,
    history_interval: int = 100,
    initial_state: HPRState | None = None,
    structural_y1: StructuralY1Solver | None = None,
    preconditioner: LPPreconditioner | None = None,
    control: Stage5Control | None = None,
) -> Stage5SGSHPRResult:
    """Run CPU sGS-HPR with optional reversible preprocessing and policies.

    Convergence is always decided in recovered original coordinates.  A raw
    Stage 4 structural equality solver is rejected for a preconditioned LP
    because general column scaling destroys its Equation (55) Gram structure.
    """

    sigma_value = _positive_finite(sigma, name="sigma")
    _positive_finite(tolerance, name="tolerance")
    if kkt_tolerance is not None:
        _positive_finite(kkt_tolerance, name="kkt_tolerance")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    if not isinstance(history_interval, int) or history_interval <= 0:
        raise ValueError("history_interval must be a positive integer.")
    policy = Stage5Control() if control is None else control

    if preconditioner is not None:
        if preconditioner.source_lp is not lp:
            raise ValueError("preconditioner must be prepared from the supplied CanonicalLP.")
        if structural_y1 is not None:
            raise ValueError(
                "a raw structural_y1 backend cannot be reused after Stage 5 "
                "row/column preprocessing; use the direct equality backend."
            )
        algorithm_lp = preconditioner.scaled_lp
    else:
        algorithm_lp = lp

    total_start = perf_counter()
    if initial_state is None:
        original_initial = HPRState(
            y=np.zeros(lp.m, dtype=np.float64),
            z=np.zeros(lp.n, dtype=np.float64),
            x=np.zeros(lp.n, dtype=np.float64),
        )
    else:
        original_initial = initial_state.detached_copy()
    _validate_state(lp, original_initial, name="initial_state")
    algorithm_initial = _algorithm_state(original_initial, preconditioner)
    _validate_state(algorithm_lp, algorithm_initial, name="scaled initial_state")

    anchor = algorithm_initial.detached_copy()
    current = algorithm_initial.detached_copy()
    sigma_reference = algorithm_initial.detached_copy()
    preparation_start = perf_counter()
    workspace = prepare_sgs_hpr(algorithm_lp, structural_y1=structural_y1)
    preparation_elapsed_seconds = perf_counter() - preparation_start

    history: list[Stage5HistoryEntry] = []
    policy_events: list[Stage5PolicyEvent] = []
    scaled_final = current.detached_copy()
    final_state = _recover(scaled_final, preconditioner)
    scaled_final_residuals = evaluate_residuals(
        algorithm_lp,
        x=scaled_final.x,
        y=scaled_final.y,
        z=scaled_final.z,
        tolerance=tolerance,
    )
    final_residuals = evaluate_residuals(
        lp,
        x=final_state.x,
        y=final_state.y,
        z=final_state.z,
        tolerance=tolerance,
    )
    converged = False
    maximum_equality_residual = 0.0
    maximum_equality_infinity_residual = 0.0
    maximum_z_x_identity_error = 0.0
    iteration_start = perf_counter()
    completed_iterations = 0
    inner_iteration = 0
    restart_count = 0
    reference_merit: float | None = None
    previous_checkpoint_merit: float | None = None
    initial_sigma = sigma_value
    minimum_sigma = sigma_value
    maximum_sigma = sigma_value

    for global_iteration in range(max_iterations):
        sigma_used = sigma_value
        step = sgs_hpr_step(
            algorithm_lp,
            current,
            anchor,
            workspace,
            iteration=inner_iteration,
            sigma=sigma_used,
        )
        completed_iterations = global_iteration + 1
        inner_completed = inner_iteration + 1
        scaled_final = step.proximal
        final_state = _recover(scaled_final, preconditioner)
        maximum_equality_residual = max(
            maximum_equality_residual,
            step.first_equality_relative_residual,
            step.second_equality_relative_residual,
        )
        maximum_equality_infinity_residual = max(
            maximum_equality_infinity_residual,
            step.first_equality_infinity_residual,
            step.second_equality_infinity_residual,
        )
        maximum_z_x_identity_error = max(
            maximum_z_x_identity_error,
            step.z_x_identity_error,
        )
        scaled_final_residuals = evaluate_residuals(
            algorithm_lp,
            x=scaled_final.x,
            y=scaled_final.y,
            z=scaled_final.z,
            tolerance=tolerance,
        )
        final_residuals = evaluate_residuals(
            lp,
            x=final_state.x,
            y=final_state.y,
            z=final_state.z,
            tolerance=tolerance,
        )
        kkt_satisfied = kkt_tolerance is None or final_residuals.combined_norm <= kkt_tolerance
        converged = final_residuals.conditions.all_satisfied and kkt_satisfied

        merit: float | None = None
        if policy.enabled:
            merit = sgs_restart_merit(
                workspace,
                delta_x=current.x - step.reflected.x,
                delta_y=current.y - step.reflected.y,
                sigma=sigma_used,
            )
            if reference_merit is None:
                reference_merit = merit

        policy_checkpoint = (
            policy.enabled and completed_iterations % policy.check_interval == 0 and not converged
        )
        restarted = False
        if policy_checkpoint:
            assert merit is not None
            assert reference_merit is not None
            if policy.restart and restart_count == 0:
                # HPR-LP v0.1.0 unconditionally restarts at its first policy
                # check. The DCOPF manuscript changes the cadence to 100.
                reasons: tuple[RestartReason, ...] = ("forced_first",)
            elif policy.restart:
                reasons = choose_restart_reasons(
                    merit=merit,
                    reference_merit=reference_merit,
                    previous_checkpoint_merit=previous_checkpoint_merit,
                    inner_iteration=inner_completed,
                    total_iteration=completed_iterations,
                    control=policy,
                )
            else:
                reasons = ()
            restarted = bool(reasons)
            should_update_sigma = policy.adaptive_sigma and (restarted or not policy.restart)
            update_control = (
                policy
                if should_update_sigma
                else Stage5Control(
                    adaptive_sigma=False,
                    restart=policy.restart,
                    check_interval=policy.check_interval,
                    alpha_sufficient=policy.alpha_sufficient,
                    alpha_necessary=policy.alpha_necessary,
                    alpha_long=policy.alpha_long,
                    movement_minimum=policy.movement_minimum,
                    movement_maximum=policy.movement_maximum,
                    infeasibility_ratio_minimum=(policy.infeasibility_ratio_minimum),
                    infeasibility_ratio_maximum=(policy.infeasibility_ratio_maximum),
                )
            )
            sigma_update = hprlp_sigma_update(
                workspace,
                reference=anchor if restarted else sigma_reference,
                candidate=scaled_final,
                residuals=final_residuals,
                sigma_before=sigma_used,
                control=update_control,
            )
            sigma_value = sigma_update.sigma_after
            minimum_sigma = min(minimum_sigma, sigma_value)
            maximum_sigma = max(maximum_sigma, sigma_value)
            if policy.adaptive_sigma and not policy.restart:
                sigma_reference = scaled_final.detached_copy()
            if restarted:
                restart_count += 1
            policy_events.append(
                Stage5PolicyEvent(
                    iteration=completed_iterations,
                    inner_iteration=inner_completed,
                    merit=merit,
                    reference_merit=reference_merit,
                    previous_checkpoint_merit=previous_checkpoint_merit,
                    restart_reasons=reasons,
                    restarted=restarted,
                    restart_count=restart_count,
                    sigma_update=sigma_update,
                )
            )
            previous_checkpoint_merit = merit

        should_record = (
            completed_iterations == 1
            or completed_iterations % history_interval == 0
            or completed_iterations == max_iterations
            or converged
            or policy_checkpoint
        )
        if should_record:
            minimum_y2 = float(np.min(final_state.y[lp.m1 :])) if lp.m2 else None
            history.append(
                Stage5HistoryEntry(
                    iteration=completed_iterations,
                    inner_iteration=inner_completed,
                    iteration_loop_elapsed_seconds=(perf_counter() - iteration_start),
                    scaled_variable_objective=float(algorithm_lp.c @ scaled_final.x),
                    original_variable_objective=float(lp.c @ final_state.x),
                    scaled_residuals=ResidualSnapshot.from_evaluation(scaled_final_residuals),
                    original_residuals=ResidualSnapshot.from_evaluation(final_residuals),
                    kkt_target_satisfied=kkt_satisfied,
                    maximum_equality_solve_relative_residual=(maximum_equality_residual),
                    maximum_equality_solve_infinity_residual=(maximum_equality_infinity_residual),
                    minimum_original_inequality_multiplier=minimum_y2,
                    sigma=sigma_used,
                    restart_count=restart_count,
                )
            )

        if converged:
            current = step.next_state
            break
        if restarted:
            anchor = scaled_final.detached_copy()
            current = scaled_final.detached_copy()
            sigma_reference = scaled_final.detached_copy()
            inner_iteration = 0
            reference_merit = None
            previous_checkpoint_merit = None
        else:
            current = step.next_state
            inner_iteration += 1

    current_state = _recover(current, preconditioner)
    return Stage5SGSHPRResult(
        solution=final_state.detached_copy(),
        scaled_solution=scaled_final.detached_copy(),
        current_state=current_state,
        scaled_current_state=current.detached_copy(),
        residuals=final_residuals,
        scaled_residuals=scaled_final_residuals,
        history=tuple(history),
        policy_events=tuple(policy_events),
        iterations=completed_iterations,
        converged=converged,
        sigma=float(sigma_value),
        initial_sigma=float(initial_sigma),
        minimum_sigma=float(minimum_sigma),
        maximum_sigma=float(maximum_sigma),
        restart_count=restart_count,
        history_interval=history_interval,
        workspace=workspace,
        control=policy,
        preconditioner=preconditioner,
        preparation_elapsed_seconds=preparation_elapsed_seconds,
        total_elapsed_seconds=perf_counter() - total_start,
        maximum_equality_solve_relative_residual=maximum_equality_residual,
        maximum_equality_solve_infinity_residual=(maximum_equality_infinity_residual),
        maximum_z_x_identity_error=maximum_z_x_identity_error,
    )
