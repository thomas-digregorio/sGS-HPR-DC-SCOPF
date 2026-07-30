"""FP64 CPU reference implementation of the paper's Algorithm 2."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from .canonical_lp import CanonicalLP
from .hpr_generic import HPRState, halpern_update, reflect_state
from .projections import project_box, project_nonnegative
from .residuals import ResidualEvaluation, evaluate_residuals

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]

ALGORITHM_2_UPDATE_ORDER = (
    "z_bar",
    "x_bar",
    "y1_half",
    "y2_bar",
    "y1_bar",
    "reflection",
    "halpern_anchor",
)


def _csr(matrix: Any, *, rows: int, columns: int, name: str) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    if result.shape != (rows, columns):
        raise ValueError(f"{name} must have shape {(rows, columns)}; received {result.shape}.")
    if not np.all(np.isfinite(result.data)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _matvec(matrix: sparse.spmatrix, vector: FloatVector) -> FloatVector:
    return np.asarray(matrix @ vector, dtype=np.float64).reshape(-1)


def _validate_state(lp: CanonicalLP, state: HPRState, *, name: str) -> None:
    if state.y.shape != (lp.m,):
        raise ValueError(f"{name}.y must have shape ({lp.m},); received {state.y.shape}.")
    if state.z.shape != (lp.n,):
        raise ValueError(f"{name}.z must have shape ({lp.n},); received {state.z.shape}.")
    if state.x.shape != (lp.n,):
        raise ValueError(f"{name}.x must have shape ({lp.n},); received {state.x.shape}.")


@dataclass(frozen=True, slots=True)
class EqualitySystemDiagnostics:
    """Numerical checks for the direct ``A1 A1^T`` reference solve."""

    rows: int
    rank: int
    symmetry_error: float
    minimum_eigenvalue: float | None
    maximum_eigenvalue: float | None
    condition_number: float

    @property
    def full_row_rank(self) -> bool:
        return self.rank == self.rows

    @property
    def positive_definite(self) -> bool:
        return self.rows == 0 or (
            self.minimum_eigenvalue is not None and self.minimum_eigenvalue > 0.0
        )

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "rank": self.rank,
            "full_row_rank": self.full_row_rank,
            "symmetry_error": self.symmetry_error,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "maximum_eigenvalue": self.maximum_eigenvalue,
            "condition_number": self.condition_number,
            "positive_definite": self.positive_definite,
        }


@dataclass(frozen=True, slots=True)
class SpectralEstimateDiagnostics:
    """Cross-checked estimates for ``lambda_max(A2 A2^T)`` in Equation (47)."""

    rows: int
    dense_eigendecomposition: float
    sparse_eigsh: float
    power_iteration: float
    power_iterations: int
    power_converged: bool
    power_residual: float
    lambda_used: float
    safety_margin: float
    s2_minimum_eigenvalue: float

    @property
    def maximum_estimate_difference(self) -> float:
        values = (
            self.dense_eigendecomposition,
            self.sparse_eigsh,
            self.power_iteration,
        )
        return max(values) - min(values)

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "dense_eigendecomposition": self.dense_eigendecomposition,
            "sparse_eigsh": self.sparse_eigsh,
            "power_iteration": self.power_iteration,
            "power_iterations": self.power_iterations,
            "power_converged": self.power_converged,
            "power_residual": self.power_residual,
            "maximum_estimate_difference": self.maximum_estimate_difference,
            "lambda_used": self.lambda_used,
            "safety_margin": self.safety_margin,
            "s2_minimum_eigenvalue": self.s2_minimum_eigenvalue,
        }


@dataclass(frozen=True, slots=True)
class SGSHPRWorkspace:
    """Prepared sparse operators and trusted direct-solve factors."""

    source_lp: CanonicalLP
    A1: sparse.csr_matrix
    A1_transpose: sparse.csr_matrix
    A2: sparse.csr_matrix
    A2_transpose: sparse.csr_matrix
    equality_gram: FloatMatrix
    equality_cholesky: tuple[FloatMatrix, bool] | None
    equality: EqualitySystemDiagnostics
    spectral: SpectralEstimateDiagnostics | None


@dataclass(frozen=True, slots=True)
class SGSHPRStep:
    """One Algorithm 2 step with all intermediate states kept distinct."""

    y1_half: FloatVector
    proximal: HPRState
    reflected: HPRState
    next_state: HPRState
    first_equality_relative_residual: float
    second_equality_relative_residual: float
    first_equality_infinity_residual: float
    second_equality_infinity_residual: float
    z_x_identity_error: float
    update_order: tuple[str, ...] = ALGORITHM_2_UPDATE_ORDER


@dataclass(frozen=True, slots=True)
class SGSHPRHistoryEntry:
    """One recorded Equation (54) check interval."""

    iteration: int
    canonical_variable_objective: float
    iteration_loop_elapsed_seconds: float
    kkt_combined_norm: float
    primal_raw: float
    box_raw: float
    stationarity_raw: float
    primal_normalized: float
    box_normalized: float
    stationarity_normalized: float
    paper_stopping_satisfied: bool
    kkt_target_satisfied: bool
    maximum_equality_solve_relative_residual: float
    maximum_equality_solve_infinity_residual: float
    minimum_inequality_multiplier: float | None
    sigma: float
    restart_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "canonical_variable_objective": self.canonical_variable_objective,
            "iteration_loop_elapsed_seconds": self.iteration_loop_elapsed_seconds,
            "kkt_combined_norm": self.kkt_combined_norm,
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
            "paper_stopping_satisfied": self.paper_stopping_satisfied,
            "kkt_target_satisfied": self.kkt_target_satisfied,
            "maximum_equality_solve_relative_residual": (
                self.maximum_equality_solve_relative_residual
            ),
            "maximum_equality_solve_infinity_residual": (
                self.maximum_equality_solve_infinity_residual
            ),
            "minimum_inequality_multiplier": self.minimum_inequality_multiplier,
            "sigma": self.sigma,
            "restart_count": self.restart_count,
        }


@dataclass(frozen=True, slots=True)
class SGSHPRResult:
    """Complete Stage 3 solver result and numerical evidence."""

    solution: HPRState
    current_state: HPRState
    residuals: ResidualEvaluation
    history: tuple[SGSHPRHistoryEntry, ...]
    iterations: int
    converged: bool
    sigma: float
    history_interval: int
    restart_count: int
    workspace: SGSHPRWorkspace
    preparation_elapsed_seconds: float
    total_elapsed_seconds: float
    maximum_equality_solve_relative_residual: float
    maximum_equality_solve_infinity_residual: float
    maximum_z_x_identity_error: float


def _power_iteration(
    operator: sparse_linalg.LinearOperator,
    size: int,
    *,
    tolerance: float,
    max_iterations: int,
    seed: int,
) -> tuple[float, int, bool, float]:
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(size).astype(np.float64)
    vector /= np.linalg.norm(vector)
    eigenvalue = 0.0
    residual = np.inf

    for iteration in range(1, max_iterations + 1):
        product = np.asarray(operator @ vector, dtype=np.float64).reshape(-1)
        product_norm = float(np.linalg.norm(product))
        if product_norm == 0.0:
            return 0.0, iteration, True, 0.0
        vector = product / product_norm
        refreshed = np.asarray(operator @ vector, dtype=np.float64).reshape(-1)
        next_eigenvalue = float(np.dot(vector, refreshed))
        residual = float(np.linalg.norm(refreshed - next_eigenvalue * vector))
        change = abs(next_eigenvalue - eigenvalue)
        eigenvalue = next_eigenvalue
        scale = max(1.0, abs(eigenvalue))
        if residual <= tolerance * scale and change <= tolerance * scale:
            return max(eigenvalue, 0.0), iteration, True, residual

    return max(eigenvalue, 0.0), max_iterations, False, residual


def estimate_inequality_spectrum(
    A2: sparse.spmatrix,
    *,
    relative_safety_margin: float = 1e-10,
    power_tolerance: float = 1e-11,
    power_max_iterations: int = 20_000,
    power_seed: int = 20260729,
) -> SpectralEstimateDiagnostics:
    """Cross-check three estimators and return a conservative Equation (47) lambda."""

    matrix = sparse.csr_matrix(A2, dtype=np.float64, copy=True)
    rows = int(matrix.shape[0])
    if rows <= 0:
        raise ValueError("A2 must contain at least one inequality row.")
    if not np.isfinite(relative_safety_margin) or relative_safety_margin <= 0.0:
        raise ValueError("relative_safety_margin must be a positive finite scalar.")
    if not np.isfinite(power_tolerance) or power_tolerance <= 0.0:
        raise ValueError("power_tolerance must be a positive finite scalar.")
    if not isinstance(power_max_iterations, int) or power_max_iterations <= 0:
        raise ValueError("power_max_iterations must be a positive integer.")

    dense = np.asarray(matrix.toarray(), dtype=np.float64)
    dense_gram = dense @ dense.T
    dense_gram = 0.5 * (dense_gram + dense_gram.T)
    dense_lambda = max(float(np.linalg.eigvalsh(dense_gram)[-1]), 0.0)
    if dense_lambda <= np.finfo(np.float64).eps:
        raise ValueError("A2 must have a positive spectral norm for Equation (50).")

    transpose = matrix.T.tocsr()

    def gram_matvec(vector: NDArray[np.float64]) -> FloatVector:
        return _matvec(matrix, _matvec(transpose, np.asarray(vector, dtype=np.float64)))

    operator = sparse_linalg.LinearOperator(
        shape=(rows, rows),
        matvec=gram_matvec,
        rmatvec=gram_matvec,
        dtype=np.float64,
    )
    initial = np.random.default_rng(power_seed).standard_normal(rows)
    if rows == 1:
        sparse_lambda = dense_lambda
    else:
        sparse_values = sparse_linalg.eigsh(
            operator,
            k=1,
            which="LA",
            v0=initial,
            tol=1e-12,
            maxiter=max(1_000, rows * 100),
            return_eigenvectors=False,
        )
        sparse_lambda = max(float(sparse_values[0]), 0.0)

    power_lambda, power_iterations, power_converged, power_residual = _power_iteration(
        operator,
        rows,
        tolerance=power_tolerance,
        max_iterations=power_max_iterations,
        seed=power_seed,
    )
    largest_estimate = max(dense_lambda, sparse_lambda, power_lambda)
    safety_margin = relative_safety_margin * max(1.0, largest_estimate)
    lambda_used = float(np.nextafter(largest_estimate + safety_margin, np.inf))
    return SpectralEstimateDiagnostics(
        rows=rows,
        dense_eigendecomposition=dense_lambda,
        sparse_eigsh=sparse_lambda,
        power_iteration=power_lambda,
        power_iterations=power_iterations,
        power_converged=power_converged,
        power_residual=power_residual,
        lambda_used=lambda_used,
        safety_margin=lambda_used - largest_estimate,
        s2_minimum_eigenvalue=lambda_used - dense_lambda,
    )


def prepare_sgs_hpr(lp: CanonicalLP) -> SGSHPRWorkspace:
    """Prepare Stage 3 reference linear algebra and verify paper assumptions."""

    A1 = _csr(lp.A1, rows=lp.m1, columns=lp.n, name="A1")
    A2 = _csr(lp.A2, rows=lp.m2, columns=lp.n, name="A2")
    A1_transpose = A1.T.tocsr()
    A2_transpose = A2.T.tocsr()

    if lp.m1:
        raw_equality_gram = np.asarray((A1 @ A1_transpose).toarray(), dtype=np.float64)
        symmetry_error = float(np.linalg.norm(raw_equality_gram - raw_equality_gram.T, ord=np.inf))
        symmetry_scale = max(1.0, float(np.linalg.norm(raw_equality_gram, ord=np.inf)))
        symmetry_tolerance = 100.0 * np.finfo(np.float64).eps * symmetry_scale
        if symmetry_error > symmetry_tolerance:
            raise ValueError(
                "A1 A1^T must be numerically symmetric; "
                f"infinity-norm error {symmetry_error} exceeds {symmetry_tolerance}."
            )
        equality_gram = raw_equality_gram
        equality_gram = 0.5 * (equality_gram + equality_gram.T)
        singular_values = np.linalg.svd(np.asarray(A1.toarray()), compute_uv=False)
        singular_tolerance = max(A1.shape) * np.finfo(np.float64).eps * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > singular_tolerance))
        eigenvalues = np.linalg.eigvalsh(equality_gram)
        minimum_eigenvalue = float(eigenvalues[0])
        maximum_eigenvalue = float(eigenvalues[-1])
        condition_number = float(np.linalg.cond(equality_gram))
        equality = EqualitySystemDiagnostics(
            rows=lp.m1,
            rank=rank,
            symmetry_error=symmetry_error,
            minimum_eigenvalue=minimum_eigenvalue,
            maximum_eigenvalue=maximum_eigenvalue,
            condition_number=condition_number,
        )
        if not equality.full_row_rank:
            raise ValueError(f"A1 must have full row rank; detected rank {rank} for {lp.m1} rows.")
        if not equality.positive_definite:
            raise ValueError(
                f"A1 A1^T must be positive definite; minimum eigenvalue is {minimum_eigenvalue}."
            )
        equality_cholesky = linalg.cho_factor(
            equality_gram,
            lower=True,
            check_finite=True,
        )
    else:
        equality_gram = np.empty((0, 0), dtype=np.float64)
        equality_cholesky = None
        equality = EqualitySystemDiagnostics(
            rows=0,
            rank=0,
            symmetry_error=0.0,
            minimum_eigenvalue=None,
            maximum_eigenvalue=None,
            condition_number=1.0,
        )

    spectral = estimate_inequality_spectrum(A2) if lp.m2 else None
    return SGSHPRWorkspace(
        source_lp=lp,
        A1=A1,
        A1_transpose=A1_transpose,
        A2=A2,
        A2_transpose=A2_transpose,
        equality_gram=equality_gram,
        equality_cholesky=equality_cholesky,
        equality=equality,
        spectral=spectral,
    )


def _solve_equality(
    workspace: SGSHPRWorkspace,
    right_hand_side: FloatVector,
) -> tuple[FloatVector, float, float]:
    if workspace.equality_cholesky is None:
        return np.empty(0, dtype=np.float64), 0.0, 0.0
    solution = linalg.cho_solve(
        workspace.equality_cholesky,
        right_hand_side,
        check_finite=True,
    )
    residual = workspace.equality_gram @ solution - right_hand_side
    relative_residual = float(np.linalg.norm(residual) / (1.0 + np.linalg.norm(right_hand_side)))
    infinity_residual = float(np.linalg.norm(residual, ord=np.inf))
    return np.asarray(solution, dtype=np.float64), relative_residual, infinity_residual


def sgs_hpr_step(
    lp: CanonicalLP,
    current: HPRState,
    anchor: HPRState,
    workspace: SGSHPRWorkspace,
    *,
    iteration: int,
    sigma: float,
) -> SGSHPRStep:
    """Perform one exact Algorithm 2 iteration using direct equality solves."""

    _validate_state(lp, current, name="current")
    _validate_state(lp, anchor, name="anchor")
    if workspace.source_lp is not lp:
        raise ValueError("workspace must be prepared from the same CanonicalLP instance.")
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer.")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite scalar.")

    y1_current = current.y[: lp.m1]
    y2_current = current.y[lp.m1 :]
    aty = _matvec(workspace.A1_transpose, y1_current) + _matvec(
        workspace.A2_transpose,
        y2_current,
    )

    projection_argument = current.x + sigma * (aty - lp.c)
    projected_argument = project_box(projection_argument, lp.lower, lp.upper)
    z_bar = (projected_argument - projection_argument) / sigma
    x_bar = current.x + sigma * (aty + z_bar - lp.c)
    z_x_identity_error = float(np.linalg.norm(x_bar - projected_argument, ord=np.inf))

    common_without_y1 = x_bar + sigma * (_matvec(workspace.A2_transpose, y2_current) + z_bar - lp.c)
    first_rhs = (lp.b1 - _matvec(workspace.A1, common_without_y1)) / sigma
    y1_half, first_relative_residual, first_infinity_residual = _solve_equality(
        workspace,
        first_rhs,
    )

    if lp.m2:
        assert workspace.spectral is not None
        ry = (
            x_bar / sigma
            + _matvec(workspace.A1_transpose, y1_half)
            + _matvec(workspace.A2_transpose, y2_current)
            + z_bar
            - lp.c
        )
        projected_argument = (
            y2_current
            + (lp.b2 / sigma - _matvec(workspace.A2, ry)) / workspace.spectral.lambda_used
        )
        y2_bar = project_nonnegative(projected_argument)
    else:
        y2_bar = np.empty(0, dtype=np.float64)

    common_with_new_y2 = x_bar + sigma * (_matvec(workspace.A2_transpose, y2_bar) + z_bar - lp.c)
    second_rhs = (lp.b1 - _matvec(workspace.A1, common_with_new_y2)) / sigma
    y1_bar, second_relative_residual, second_infinity_residual = _solve_equality(
        workspace,
        second_rhs,
    )

    proximal = HPRState(
        y=np.concatenate((y1_bar, y2_bar)),
        z=z_bar,
        x=x_bar,
    )
    reflected = reflect_state(current, proximal)
    next_state = halpern_update(anchor, reflected, iteration=iteration)
    return SGSHPRStep(
        y1_half=y1_half,
        proximal=proximal,
        reflected=reflected,
        next_state=next_state,
        first_equality_relative_residual=first_relative_residual,
        second_equality_relative_residual=second_relative_residual,
        first_equality_infinity_residual=first_infinity_residual,
        second_equality_infinity_residual=second_infinity_residual,
        z_x_identity_error=z_x_identity_error,
    )


def solve_sgs_hpr(
    lp: CanonicalLP,
    *,
    sigma: float = 1.0,
    tolerance: float = 5e-5,
    kkt_tolerance: float | None = None,
    max_iterations: int = 200_000,
    history_interval: int = 100,
    initial_state: HPRState | None = None,
) -> SGSHPRResult:
    """Run fixed-penalty, no-restart CPU Algorithm 2.

    Equation (54) is evaluated on ``w_bar`` every iteration. The trajectory is
    recorded at ``history_interval`` boundaries and at the stopping iteration.
    The returned ``solution`` is the final checked intermediate state, matching
    the manuscript's stopping convention.
    """

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite scalar.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite scalar.")
    if kkt_tolerance is not None and (not np.isfinite(kkt_tolerance) or kkt_tolerance <= 0.0):
        raise ValueError("kkt_tolerance must be None or a positive finite scalar.")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    if not isinstance(history_interval, int) or history_interval <= 0:
        raise ValueError("history_interval must be a positive integer.")

    total_start = perf_counter()
    if initial_state is None:
        initial_state = HPRState(
            y=np.zeros(lp.m, dtype=np.float64),
            z=np.zeros(lp.n, dtype=np.float64),
            x=np.zeros(lp.n, dtype=np.float64),
        )
    _validate_state(lp, initial_state, name="initial_state")
    anchor = initial_state.detached_copy()
    current = initial_state.detached_copy()
    preparation_start = perf_counter()
    prepared = prepare_sgs_hpr(lp)
    preparation_elapsed_seconds = perf_counter() - preparation_start

    history: list[SGSHPRHistoryEntry] = []
    final_state = current.detached_copy()
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

    for iteration in range(max_iterations):
        step = sgs_hpr_step(
            lp,
            current,
            anchor,
            prepared,
            iteration=iteration,
            sigma=sigma,
        )
        completed_iterations = iteration + 1
        final_state = step.proximal
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
        final_residuals = evaluate_residuals(
            lp,
            x=final_state.x,
            y=final_state.y,
            z=final_state.z,
            tolerance=tolerance,
        )
        kkt_satisfied = kkt_tolerance is None or final_residuals.combined_norm <= kkt_tolerance
        converged = final_residuals.conditions.all_satisfied and kkt_satisfied
        should_record = (
            completed_iterations == 1
            or completed_iterations % history_interval == 0
            or completed_iterations == max_iterations
            or converged
        )
        if should_record:
            raw = final_residuals.paper_raw_norms
            normalized = final_residuals.paper_normalized_norms
            minimum_y2 = float(np.min(final_state.y[lp.m1 :])) if lp.m2 else None
            history.append(
                SGSHPRHistoryEntry(
                    iteration=completed_iterations,
                    canonical_variable_objective=float(lp.c @ final_state.x),
                    iteration_loop_elapsed_seconds=(perf_counter() - iteration_start),
                    kkt_combined_norm=final_residuals.combined_norm,
                    primal_raw=raw[0],
                    box_raw=raw[1],
                    stationarity_raw=raw[2],
                    primal_normalized=normalized[0],
                    box_normalized=normalized[1],
                    stationarity_normalized=normalized[2],
                    paper_stopping_satisfied=final_residuals.conditions.all_satisfied,
                    kkt_target_satisfied=kkt_satisfied,
                    maximum_equality_solve_relative_residual=maximum_equality_residual,
                    maximum_equality_solve_infinity_residual=(maximum_equality_infinity_residual),
                    minimum_inequality_multiplier=minimum_y2,
                    sigma=float(sigma),
                    restart_count=0,
                )
            )
        if converged:
            current = step.next_state
            break
        current = step.next_state

    return SGSHPRResult(
        solution=final_state.detached_copy(),
        current_state=current.detached_copy(),
        residuals=final_residuals,
        history=tuple(history),
        iterations=completed_iterations,
        converged=converged,
        sigma=float(sigma),
        history_interval=history_interval,
        restart_count=0,
        workspace=prepared,
        preparation_elapsed_seconds=preparation_elapsed_seconds,
        total_elapsed_seconds=perf_counter() - total_start,
        maximum_equality_solve_relative_residual=maximum_equality_residual,
        maximum_equality_solve_infinity_residual=maximum_equality_infinity_residual,
        maximum_z_x_identity_error=maximum_z_x_identity_error,
    )
