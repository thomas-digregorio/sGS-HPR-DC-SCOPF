"""Device-resident Stage 5 controls for the Stage 6 CuPy port.

The validated NumPy/SciPy solver remains the CPU oracle.  This module keeps
the scaled Algorithm 2 state, original-space recovery factors, residual
buffers, restart metric, and adaptive-sigma movement calculations on the GPU.
Only a compact scalar packet is copied at an explicitly configured diagnostic
cadence; complete state vectors return to the host once, at the end of a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np

from .canonical_lp import CanonicalLP
from .gpu_backend import CuPyBackend, create_gpu_backend
from .gpu_sgs_hpr import (
    GPUHPRState,
    GPUPrecision,
    GPUSGSHPRWorkspace,
    gpu_sgs_hpr_step,
    gpu_sgs_metric_y_quadratic,
    gpu_sgs_restart_merit,
    prepare_gpu_sgs_hpr,
)
from .hpr_generic import HPRState
from .preconditioning import LPPreconditioner
from .residuals import ResidualEvaluation, evaluate_residuals
from .stage5_control import (
    ResidualSnapshot,
    Stage5Control,
    Stage5PolicyEvent,
    choose_restart_reasons,
    hprlp_sigma_update_from_scalars,
)
from .stage7_scaled_y1 import ScaledBlockArrowY1Solver


@dataclass(slots=True)
class _GPUOriginalSpaceBuffers:
    ax_scaled: Any
    aty_scaled: Any
    ax_original: Any
    aty_original: Any
    x_original: Any
    y_original: Any
    z_original: Any
    dual_projection: Any
    primal_feasibility: Any
    box: Any
    stationarity: Any
    n_work: Any
    m_work: Any
    delta_x: Any
    delta_y: Any
    diagnostics: Any
    policy: Any


@dataclass(slots=True)
class GPUStage6Problem:
    """One fully resident scaled problem plus original-space recovery data."""

    original_lp: CanonicalLP
    preconditioner: LPPreconditioner
    workspace: GPUSGSHPRWorkspace
    backend: CuPyBackend
    dtype_name: GPUPrecision
    row_denominator: Any
    column_denominator: Any
    original_b: Any
    original_c: Any
    original_lower: Any
    original_upper: Any
    primal_denominator: float
    stationarity_denominator: float
    buffers: _GPUOriginalSpaceBuffers


@dataclass(frozen=True, slots=True)
class GPUDeviceResiduals:
    """Original-space residual scalars that remain on the device."""

    kkt_combined_norm: Any
    normalized_combined_norm: Any
    primal_raw: Any
    box_raw: Any
    stationarity_raw: Any
    primal_normalized: Any
    box_normalized: Any
    stationarity_normalized: Any
    scaled_objective: Any
    original_objective: Any


@dataclass(frozen=True, slots=True)
class GPUStage6Timing:
    """Synchronized loop timing with residual work called out separately."""

    loop_gpu_seconds: float
    residual_check_gpu_seconds: float
    iterations_excluding_residual_checks_gpu_seconds: float
    loop_wall_seconds: float
    residual_check_count: int
    residual_check_interval: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loop_gpu_seconds": self.loop_gpu_seconds,
            "residual_check_gpu_seconds": self.residual_check_gpu_seconds,
            "iterations_excluding_residual_checks_gpu_seconds": (
                self.iterations_excluding_residual_checks_gpu_seconds
            ),
            "loop_wall_seconds": self.loop_wall_seconds,
            "residual_check_count": self.residual_check_count,
            "residual_check_interval": self.residual_check_interval,
        }


@dataclass(frozen=True, slots=True)
class GPUStage6HistoryEntry:
    iteration: int
    inner_iteration: int
    residuals: ResidualSnapshot
    scaled_objective: float
    original_objective: float
    sigma: float
    restart_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "inner_iteration": self.inner_iteration,
            "residuals": self.residuals.as_dict(),
            "scaled_objective": self.scaled_objective,
            "original_objective": self.original_objective,
            "sigma": self.sigma,
            "restart_count": self.restart_count,
        }


@dataclass(frozen=True, slots=True)
class GPUStage6Result:
    """Host-facing result of one resident GPU solve."""

    solution: HPRState
    scaled_solution: HPRState
    residuals: ResidualEvaluation
    iterations: int
    converged: bool
    sigma: float
    initial_sigma: float
    minimum_sigma: float
    maximum_sigma: float
    restart_count: int
    control: Stage5Control
    history: tuple[GPUStage6HistoryEntry, ...]
    policy_events: tuple[Stage5PolicyEvent, ...]
    timing: GPUStage6Timing
    maximum_equality_solve_relative_residual: float
    maximum_equality_solve_infinity_residual: float
    maximum_z_x_identity_error: float
    transfer_ledger: dict[str, Any]


def _empty_state(xp: Any, *, m: int, n: int, dtype: Any) -> GPUHPRState:
    return GPUHPRState(
        y=xp.empty(m, dtype=dtype),
        z=xp.empty(n, dtype=dtype),
        x=xp.empty(n, dtype=dtype),
    )


def _copy_state(destination: GPUHPRState, source: GPUHPRState) -> None:
    destination.y[...] = source.y
    destination.z[...] = source.z
    destination.x[...] = source.x


def _allocate_original_buffers(problem: GPUStage6Problem) -> _GPUOriginalSpaceBuffers:
    xp = problem.backend.xp
    dtype = problem.workspace.dtype
    m = problem.original_lp.m
    n = problem.original_lp.n
    return _GPUOriginalSpaceBuffers(
        ax_scaled=xp.empty(m, dtype=dtype),
        aty_scaled=xp.empty(n, dtype=dtype),
        ax_original=xp.empty(m, dtype=dtype),
        aty_original=xp.empty(n, dtype=dtype),
        x_original=xp.empty(n, dtype=dtype),
        y_original=xp.empty(m, dtype=dtype),
        z_original=xp.empty(n, dtype=dtype),
        dual_projection=xp.empty(m, dtype=dtype),
        primal_feasibility=xp.empty(m, dtype=dtype),
        box=xp.empty(n, dtype=dtype),
        stationarity=xp.empty(n, dtype=dtype),
        n_work=xp.empty(n, dtype=dtype),
        m_work=xp.empty(m, dtype=dtype),
        delta_x=xp.empty(n, dtype=dtype),
        delta_y=xp.empty(m, dtype=dtype),
        diagnostics=xp.empty(10, dtype=xp.float64),
        policy=xp.empty(4, dtype=xp.float64),
    )


def prepare_gpu_stage6_problem(
    original_lp: CanonicalLP,
    preconditioner: LPPreconditioner,
    *,
    backend: CuPyBackend | None = None,
    dtype: GPUPrecision = "float64",
    inequality_lambda: float | None = None,
    scaled_structural_y1: ScaledBlockArrowY1Solver | None = None,
) -> GPUStage6Problem:
    """Upload the full Stage 5 scaled problem and its recovery factors once."""

    if preconditioner.source_lp is not original_lp:
        raise ValueError("preconditioner must be prepared from original_lp.")
    selected_backend = create_gpu_backend() if backend is None else backend
    if (
        scaled_structural_y1 is not None
        and scaled_structural_y1.preconditioner is not preconditioner
    ):
        raise ValueError(
            "scaled_structural_y1 must be prepared from the exact supplied preconditioner."
        )
    workspace = prepare_gpu_sgs_hpr(
        preconditioner.scaled_lp,
        equality_mode=(
            "scaled_structural" if scaled_structural_y1 is not None else "scaled_direct"
        ),
        scaled_structural_y1=scaled_structural_y1,
        inequality_lambda=inequality_lambda,
        backend=selected_backend,
        dtype=dtype,
    )
    host_dtype = np.float64 if dtype == "float64" else np.float32

    def upload(values: Any, kind: str) -> Any:
        return selected_backend.to_device(
            np.asarray(values, dtype=host_dtype),
            phase="original_space_preparation",
            kind=kind,
        )

    problem = GPUStage6Problem(
        original_lp=original_lp,
        preconditioner=preconditioner,
        workspace=workspace,
        backend=selected_backend,
        dtype_name=dtype,
        row_denominator=upload(preconditioner.row_denominator, "vector"),
        column_denominator=upload(preconditioner.column_denominator, "vector"),
        original_b=upload(original_lp.b, "vector"),
        original_c=upload(original_lp.c, "vector"),
        original_lower=upload(original_lp.lower, "vector"),
        original_upper=upload(original_lp.upper, "vector"),
        primal_denominator=1.0 + float(np.linalg.norm(original_lp.b)),
        stationarity_denominator=1.0 + float(np.linalg.norm(original_lp.c)),
        buffers=None,  # type: ignore[arg-type]
    )
    problem.buffers = _allocate_original_buffers(problem)
    return problem


def recover_gpu_state(problem: GPUStage6Problem, state: GPUHPRState) -> GPUHPRState:
    """Recover a scaled state into reusable original-space device buffers."""

    xp = problem.backend.xp
    buffers = problem.buffers
    xp.multiply(state.x, problem.preconditioner.b_scale, out=buffers.x_original)
    buffers.x_original /= problem.column_denominator
    xp.multiply(state.y, problem.preconditioner.c_scale, out=buffers.y_original)
    buffers.y_original /= problem.row_denominator
    xp.multiply(state.z, problem.preconditioner.c_scale, out=buffers.z_original)
    buffers.z_original *= problem.column_denominator
    return GPUHPRState(
        y=buffers.y_original,
        z=buffers.z_original,
        x=buffers.x_original,
    )


def evaluate_gpu_original_residuals(
    problem: GPUStage6Problem,
    state: GPUHPRState,
) -> GPUDeviceResiduals:
    """Evaluate Eq. (28) and every Eq. (54) block entirely on the GPU."""

    xp = problem.backend.xp
    workspace = problem.workspace
    buffers = problem.buffers
    original = recover_gpu_state(problem, state)

    if workspace.m1:
        buffers.ax_scaled[: workspace.m1] = workspace.A1_resident.matvec(state.x, transpose=False)
        buffers.aty_scaled[...] = workspace.A1_resident.matvec(
            state.y[: workspace.m1], transpose=True
        )
    else:
        buffers.aty_scaled.fill(0.0)
    if workspace.m2:
        buffers.ax_scaled[workspace.m1 :] = workspace.A2_resident.matvec(state.x, transpose=False)
        buffers.aty_scaled += workspace.A2_resident.matvec(state.y[workspace.m1 :], transpose=True)

    xp.multiply(buffers.ax_scaled, problem.row_denominator, out=buffers.ax_original)
    buffers.ax_original *= problem.preconditioner.b_scale
    xp.multiply(
        buffers.aty_scaled,
        problem.column_denominator,
        out=buffers.aty_original,
    )
    buffers.aty_original *= problem.preconditioner.c_scale

    # Eq. (28) projected dual block.
    xp.subtract(original.y, buffers.ax_original, out=buffers.m_work)
    buffers.m_work += problem.original_b
    if workspace.m1:
        xp.subtract(
            buffers.ax_original[: workspace.m1],
            problem.original_b[: workspace.m1],
            out=buffers.dual_projection[: workspace.m1],
        )
    if workspace.m2:
        xp.maximum(
            buffers.m_work[workspace.m1 :],
            0.0,
            out=buffers.dual_projection[workspace.m1 :],
        )
        xp.subtract(
            original.y[workspace.m1 :],
            buffers.dual_projection[workspace.m1 :],
            out=buffers.dual_projection[workspace.m1 :],
        )

    # Eq. (54a), preserving free equality rows and projecting only inequalities.
    xp.subtract(problem.original_b, buffers.ax_original, out=buffers.primal_feasibility)
    if workspace.m2:
        xp.maximum(
            buffers.primal_feasibility[workspace.m1 :],
            0.0,
            out=buffers.primal_feasibility[workspace.m1 :],
        )

    # Eq. (54b) box block and Eq. (54c) stationarity block.
    xp.subtract(original.x, original.z, out=buffers.n_work)
    xp.maximum(buffers.n_work, problem.original_lower, out=buffers.box)
    xp.minimum(buffers.box, problem.original_upper, out=buffers.box)
    xp.subtract(original.x, buffers.box, out=buffers.box)
    xp.subtract(problem.original_c, buffers.aty_original, out=buffers.stationarity)
    buffers.stationarity -= original.z

    primal_raw = xp.linalg.norm(buffers.primal_feasibility)
    box_raw = xp.linalg.norm(buffers.box)
    stationarity_raw = xp.linalg.norm(buffers.stationarity)
    primal_normalized = primal_raw / problem.primal_denominator
    box_denominator = 1.0 + xp.linalg.norm(original.x) + xp.linalg.norm(original.z)
    box_normalized = box_raw / box_denominator
    stationarity_normalized = stationarity_raw / problem.stationarity_denominator
    kkt_combined = xp.sqrt(
        xp.dot(buffers.dual_projection, buffers.dual_projection)
        + xp.dot(buffers.box, buffers.box)
        + xp.dot(buffers.stationarity, buffers.stationarity)
    )
    normalized_combined = xp.sqrt(
        primal_normalized * primal_normalized
        + box_normalized * box_normalized
        + stationarity_normalized * stationarity_normalized
    )
    scaled_objective = xp.dot(workspace.c, state.x)
    original_objective = problem.preconditioner.objective_factor * scaled_objective
    return GPUDeviceResiduals(
        kkt_combined_norm=kkt_combined,
        normalized_combined_norm=normalized_combined,
        primal_raw=primal_raw,
        box_raw=box_raw,
        stationarity_raw=stationarity_raw,
        primal_normalized=primal_normalized,
        box_normalized=box_normalized,
        stationarity_normalized=stationarity_normalized,
        scaled_objective=scaled_objective,
        original_objective=original_objective,
    )


def _residual_packet(
    problem: GPUStage6Problem,
    residuals: GPUDeviceResiduals,
    *,
    phase: str,
    tolerance: float,
) -> tuple[ResidualSnapshot, float, float]:
    packet = problem.buffers.diagnostics
    values = (
        residuals.kkt_combined_norm,
        residuals.normalized_combined_norm,
        residuals.primal_raw,
        residuals.box_raw,
        residuals.stationarity_raw,
        residuals.primal_normalized,
        residuals.box_normalized,
        residuals.stationarity_normalized,
        residuals.scaled_objective,
        residuals.original_objective,
    )
    for index, value in enumerate(values):
        packet[index] = value
    host = problem.backend.to_host(packet, phase=phase, kind="vector")
    snapshot = ResidualSnapshot(
        kkt_combined_norm=float(host[0]),
        normalized_combined_norm=float(host[1]),
        primal_raw=float(host[2]),
        box_raw=float(host[3]),
        stationarity_raw=float(host[4]),
        primal_normalized=float(host[5]),
        box_normalized=float(host[6]),
        stationarity_normalized=float(host[7]),
        stopping_satisfied=bool(np.all(host[5:8] <= tolerance)),
    )
    return snapshot, float(host[8]), float(host[9])


def _host_state(
    problem: GPUStage6Problem,
    state: GPUHPRState,
    *,
    phase: str,
) -> HPRState:
    return HPRState(
        y=problem.backend.to_host(state.y, phase=phase, kind="vector"),
        z=problem.backend.to_host(state.z, phase=phase, kind="vector"),
        x=problem.backend.to_host(state.x, phase=phase, kind="vector"),
    )


def solve_gpu_stage5_sgs_hpr(
    problem: GPUStage6Problem,
    *,
    sigma: float = 1.0,
    tolerance: float = 5e-5,
    kkt_tolerance: float | None = None,
    max_iterations: int = 200_000,
    residual_check_interval: int = 1,
    history_interval: int = 100,
    initial_state: HPRState | None = None,
    control: Stage5Control | None = None,
    fixed_horizon: bool = False,
) -> GPUStage6Result:
    """Run the Stage 5 method with resident GPU state and explicit diagnostics."""

    if not isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite scalar.")
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite scalar.")
    if kkt_tolerance is not None and (not isfinite(kkt_tolerance) or kkt_tolerance <= 0.0):
        raise ValueError("kkt_tolerance must be None or a positive finite scalar.")
    for name, value in (
        ("max_iterations", max_iterations),
        ("residual_check_interval", residual_check_interval),
        ("history_interval", history_interval),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    policy = Stage5Control() if control is None else control
    lp = problem.original_lp
    if initial_state is None:
        initial_state = HPRState(
            y=np.zeros(lp.m, dtype=np.float64),
            z=np.zeros(lp.n, dtype=np.float64),
            x=np.zeros(lp.n, dtype=np.float64),
        )
    scaled_initial = problem.preconditioner.scale_state(initial_state)
    gpu_initial = GPUHPRState.from_host(
        scaled_initial,
        problem.backend,
        phase="initial_state",
        dtype=problem.dtype_name,
    )
    xp = problem.backend.xp
    dtype = problem.workspace.dtype
    anchor = _empty_state(xp, m=lp.m, n=lp.n, dtype=dtype)
    current = _empty_state(xp, m=lp.m, n=lp.n, dtype=dtype)
    sigma_reference = _empty_state(xp, m=lp.m, n=lp.n, dtype=dtype)
    final_candidate = _empty_state(xp, m=lp.m, n=lp.n, dtype=dtype)
    _copy_state(anchor, gpu_initial)
    _copy_state(current, gpu_initial)
    _copy_state(sigma_reference, gpu_initial)
    _copy_state(final_candidate, gpu_initial)

    reference_merit_device = xp.empty((), dtype=dtype)
    reference_merit_set = False
    previous_checkpoint_merit: float | None = None
    history: list[GPUStage6HistoryEntry] = []
    policy_events: list[Stage5PolicyEvent] = []
    sigma_value = float(sigma)
    initial_sigma = sigma_value
    minimum_sigma = sigma_value
    maximum_sigma = sigma_value
    restart_count = 0
    inner_iteration = 0
    converged = False
    completed_iterations = 0
    last_snapshot: ResidualSnapshot | None = None
    last_scaled_objective = 0.0
    last_original_objective = 0.0
    max_equality_relative = xp.asarray(0.0, dtype=dtype)
    max_equality_infinity = xp.asarray(0.0, dtype=dtype)
    max_z_x_identity = xp.asarray(0.0, dtype=dtype)
    residual_event_pairs: list[tuple[Any, Any]] = []

    problem.backend.synchronize()
    loop_wall_start = perf_counter()
    loop_start = xp.cuda.Event()
    loop_stop = xp.cuda.Event()
    loop_start.record()
    for global_iteration in range(max_iterations):
        step = gpu_sgs_hpr_step(
            problem.preconditioner.scaled_lp,
            current,
            anchor,
            problem.workspace,
            iteration=inner_iteration,
            sigma=sigma_value,
        )
        completed_iterations = global_iteration + 1
        inner_completed = inner_iteration + 1
        _copy_state(final_candidate, step.proximal)
        max_equality_relative = xp.maximum(
            max_equality_relative,
            xp.maximum(
                step.first_equality_relative_residual,
                step.second_equality_relative_residual,
            ),
        )
        max_equality_infinity = xp.maximum(
            max_equality_infinity,
            xp.maximum(
                step.first_equality_infinity_residual,
                step.second_equality_infinity_residual,
            ),
        )
        max_z_x_identity = xp.maximum(max_z_x_identity, step.z_x_identity_error)

        merit_device = None
        if policy.enabled:
            xp.subtract(current.x, step.reflected.x, out=problem.buffers.delta_x)
            xp.subtract(current.y, step.reflected.y, out=problem.buffers.delta_y)
            merit_device = gpu_sgs_restart_merit(
                problem.workspace,
                delta_x=problem.buffers.delta_x,
                delta_y=problem.buffers.delta_y,
                sigma=sigma_value,
            )
            if not reference_merit_set:
                reference_merit_device[...] = merit_device
                reference_merit_set = True

        policy_checkpoint = policy.enabled and completed_iterations % policy.check_interval == 0
        diagnostics_due = (
            completed_iterations % residual_check_interval == 0
            or completed_iterations == max_iterations
            or policy_checkpoint
        )
        device_residuals: GPUDeviceResiduals | None = None
        if diagnostics_due:
            residual_start = xp.cuda.Event()
            residual_stop = xp.cuda.Event()
            residual_start.record()
            device_residuals = evaluate_gpu_original_residuals(problem, final_candidate)
            residual_stop.record()
            residual_event_pairs.append((residual_start, residual_stop))
            last_snapshot, last_scaled_objective, last_original_objective = _residual_packet(
                problem,
                device_residuals,
                phase="periodic_diagnostics",
                tolerance=tolerance,
            )
            kkt_satisfied = (
                kkt_tolerance is None or last_snapshot.kkt_combined_norm <= kkt_tolerance
            )
            converged = not fixed_horizon and last_snapshot.stopping_satisfied and kkt_satisfied

        restarted = False
        if policy_checkpoint and not converged:
            assert merit_device is not None
            assert device_residuals is not None
            packet = problem.buffers.policy
            packet[0] = merit_device
            packet[1] = reference_merit_device
            merit_host = problem.backend.to_host(
                packet[:2],
                phase="policy_diagnostics",
                kind="vector",
            )
            merit = float(merit_host[0])
            reference_merit = float(merit_host[1])
            if policy.restart and restart_count == 0:
                reasons = ("forced_first",)
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
            movement_reference = anchor if restarted else sigma_reference
            xp.subtract(
                final_candidate.x,
                movement_reference.x,
                out=problem.buffers.delta_x,
            )
            xp.subtract(
                final_candidate.y,
                movement_reference.y,
                out=problem.buffers.delta_y,
            )
            packet[0] = xp.linalg.norm(problem.buffers.delta_x)
            packet[1] = xp.sqrt(
                gpu_sgs_metric_y_quadratic(
                    problem.workspace,
                    problem.buffers.delta_y,
                )
            )
            movement_host = problem.backend.to_host(
                packet[:2],
                phase="policy_diagnostics",
                kind="vector",
            )
            should_update_sigma = policy.adaptive_sigma and (restarted or not policy.restart)
            update_control = policy
            if not should_update_sigma:
                update_control = Stage5Control(
                    adaptive_sigma=False,
                    restart=policy.restart,
                    check_interval=policy.check_interval,
                    alpha_sufficient=policy.alpha_sufficient,
                    alpha_necessary=policy.alpha_necessary,
                    alpha_long=policy.alpha_long,
                    movement_minimum=policy.movement_minimum,
                    movement_maximum=policy.movement_maximum,
                    infeasibility_ratio_minimum=policy.infeasibility_ratio_minimum,
                    infeasibility_ratio_maximum=policy.infeasibility_ratio_maximum,
                )
            assert last_snapshot is not None
            sigma_update = hprlp_sigma_update_from_scalars(
                delta_x=float(movement_host[0]),
                delta_y=float(movement_host[1]),
                primal_infeasibility=last_snapshot.primal_normalized,
                dual_infeasibility=last_snapshot.stationarity_normalized,
                sigma_before=sigma_value,
                control=update_control,
            )
            sigma_value = sigma_update.sigma_after
            minimum_sigma = min(minimum_sigma, sigma_value)
            maximum_sigma = max(maximum_sigma, sigma_value)
            if policy.adaptive_sigma and not policy.restart:
                _copy_state(sigma_reference, final_candidate)
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

        if last_snapshot is not None and (
            completed_iterations == 1
            or completed_iterations % history_interval == 0
            or completed_iterations == max_iterations
            or converged
            or policy_checkpoint
        ):
            history.append(
                GPUStage6HistoryEntry(
                    iteration=completed_iterations,
                    inner_iteration=inner_completed,
                    residuals=last_snapshot,
                    scaled_objective=last_scaled_objective,
                    original_objective=last_original_objective,
                    sigma=sigma_value,
                    restart_count=restart_count,
                )
            )

        if converged:
            break
        if restarted:
            _copy_state(anchor, final_candidate)
            _copy_state(current, final_candidate)
            _copy_state(sigma_reference, final_candidate)
            inner_iteration = 0
            reference_merit_set = False
            previous_checkpoint_merit = None
        else:
            _copy_state(current, step.next_state)
            inner_iteration += 1

    loop_stop.record()
    loop_stop.synchronize()
    loop_wall_seconds = perf_counter() - loop_wall_start
    loop_gpu_seconds = float(xp.cuda.get_elapsed_time(loop_start, loop_stop)) / 1_000.0
    residual_gpu_seconds = sum(
        float(xp.cuda.get_elapsed_time(start, stop)) / 1_000.0
        for start, stop in residual_event_pairs
    )

    # A final device-side recovery is followed by the only full-vector D2H copy.
    original_device = recover_gpu_state(problem, final_candidate)
    solution = _host_state(problem, original_device, phase="final_state")
    scaled_solution = _host_state(problem, final_candidate, phase="final_scaled_state")
    final_residuals = evaluate_residuals(
        lp,
        x=solution.x,
        y=solution.y,
        z=solution.z,
        tolerance=tolerance,
    )
    maxima = problem.buffers.policy
    maxima[0] = max_equality_relative
    maxima[1] = max_equality_infinity
    maxima[2] = max_z_x_identity
    maxima_host = problem.backend.to_host(
        maxima[:3],
        phase="final_diagnostics",
        kind="vector",
    )
    return GPUStage6Result(
        solution=solution,
        scaled_solution=scaled_solution,
        residuals=final_residuals,
        iterations=completed_iterations,
        converged=converged,
        sigma=float(sigma_value),
        initial_sigma=initial_sigma,
        minimum_sigma=minimum_sigma,
        maximum_sigma=maximum_sigma,
        restart_count=restart_count,
        control=policy,
        history=tuple(history),
        policy_events=tuple(policy_events),
        timing=GPUStage6Timing(
            loop_gpu_seconds=loop_gpu_seconds,
            residual_check_gpu_seconds=residual_gpu_seconds,
            iterations_excluding_residual_checks_gpu_seconds=max(
                0.0,
                loop_gpu_seconds - residual_gpu_seconds,
            ),
            loop_wall_seconds=loop_wall_seconds,
            residual_check_count=len(residual_event_pairs),
            residual_check_interval=residual_check_interval,
        ),
        maximum_equality_solve_relative_residual=float(maxima_host[0]),
        maximum_equality_solve_infinity_residual=float(maxima_host[1]),
        maximum_z_x_identity_error=float(maxima_host[2]),
        transfer_ledger=problem.backend.ledger.summary(),
    )


__all__ = [
    "GPUDeviceResiduals",
    "GPUStage6HistoryEntry",
    "GPUStage6Problem",
    "GPUStage6Result",
    "GPUStage6Timing",
    "evaluate_gpu_original_residuals",
    "prepare_gpu_stage6_problem",
    "recover_gpu_state",
    "solve_gpu_stage5_sgs_hpr",
]
