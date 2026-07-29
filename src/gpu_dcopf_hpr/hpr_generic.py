"""Correctness-oriented implementation of the paper's generic Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .canonical_lp import CanonicalLP
from .projections import project_box, project_dual_set
from .residuals import ResidualEvaluation, evaluate_residuals

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


def _finite_vector(values: ArrayLike, *, name: str) -> FloatVector:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


@dataclass(frozen=True, slots=True)
class HPRState:
    """Algorithm state in the paper's exact order ``w = (y, z, x)``."""

    y: ArrayLike
    z: ArrayLike
    x: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "y", _finite_vector(self.y, name="y"))
        object.__setattr__(self, "z", _finite_vector(self.z, name="z"))
        object.__setattr__(self, "x", _finite_vector(self.x, name="x"))

    def detached_copy(self) -> HPRState:
        return HPRState(y=self.y.copy(), z=self.z.copy(), x=self.x.copy())


@dataclass(frozen=True, slots=True)
class SpectralProximal:
    """A verified ``T1 = tau I - A A^T`` that makes the y metric scalar."""

    matrix: FloatMatrix
    tau: float
    lambda_max: float
    margin: float
    minimum_eigenvalue: float


@dataclass(frozen=True, slots=True)
class HPRStep:
    proximal: HPRState
    reflected: HPRState
    next_state: HPRState


@dataclass(frozen=True, slots=True)
class HPRHistoryEntry:
    iteration: int
    objective: float
    kkt_combined_norm: float
    paper_primal_normalized: float
    paper_box_normalized: float
    paper_stationarity_normalized: float
    paper_stopping_satisfied: bool
    kkt_target_satisfied: bool
    minimum_inequality_multiplier: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "objective": self.objective,
            "kkt_combined_norm": self.kkt_combined_norm,
            "paper_normalized": {
                "primal_feasibility": self.paper_primal_normalized,
                "box": self.paper_box_normalized,
                "stationarity": self.paper_stationarity_normalized,
            },
            "paper_stopping_satisfied": self.paper_stopping_satisfied,
            "kkt_target_satisfied": self.kkt_target_satisfied,
            "minimum_inequality_multiplier": self.minimum_inequality_multiplier,
        }


@dataclass(frozen=True, slots=True)
class HPRResult:
    solution: HPRState
    current_state: HPRState
    residuals: ResidualEvaluation
    history: tuple[HPRHistoryEntry, ...]
    iterations: int
    converged: bool
    sigma: float
    proximal: SpectralProximal


def _validate_state(lp: CanonicalLP, state: HPRState, *, name: str) -> None:
    if state.y.shape != (lp.m,):
        raise ValueError(f"{name}.y must have shape ({lp.m},); received {state.y.shape}.")
    if state.z.shape != (lp.n,):
        raise ValueError(f"{name}.z must have shape ({lp.n},); received {state.z.shape}.")
    if state.x.shape != (lp.n,):
        raise ValueError(f"{name}.x must have shape ({lp.n},); received {state.x.shape}.")


def construct_spectral_proximal(
    lp: CanonicalLP,
    *,
    relative_margin: float = 1e-8,
) -> SpectralProximal:
    """Construct and numerically verify a valid positive-semidefinite ``T1``.

    The positive margin makes ``T1`` itself positive definite. More importantly,
    ``A A^T + T1 = tau I`` is positive definite and its constrained y solve is
    exactly an ordinary projection onto ``D``.
    """

    if not np.isfinite(relative_margin) or relative_margin <= 0.0:
        raise ValueError("relative_margin must be a positive finite scalar.")
    A = lp.dense_A()
    gram = A @ A.T
    if lp.m == 0:
        return SpectralProximal(
            matrix=np.empty((0, 0), dtype=np.float64),
            tau=1.0,
            lambda_max=0.0,
            margin=1.0,
            minimum_eigenvalue=1.0,
        )

    eigenvalues = np.linalg.eigvalsh(gram)
    lambda_max = max(float(eigenvalues[-1]), 0.0)
    margin = relative_margin * max(1.0, lambda_max)
    tau = lambda_max + margin
    matrix = tau * np.eye(lp.m, dtype=np.float64) - gram
    matrix = 0.5 * (matrix + matrix.T)
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
    numerical_tolerance = 1e-11 * max(1.0, tau)
    if minimum_eigenvalue < -numerical_tolerance:
        raise RuntimeError(
            f"constructed T1 is not positive semidefinite: minimum eigenvalue {minimum_eigenvalue}."
        )
    metric = gram + matrix
    if not np.allclose(metric, tau * np.eye(lp.m), rtol=1e-11, atol=numerical_tolerance):
        raise RuntimeError("constructed y metric is not the expected scalar identity.")
    if float(np.linalg.eigvalsh(metric)[0]) <= 0.0:
        raise RuntimeError("T1 + A A^T must be positive definite.")
    return SpectralProximal(
        matrix=matrix,
        tau=tau,
        lambda_max=lambda_max,
        margin=margin,
        minimum_eigenvalue=minimum_eigenvalue,
    )


def reflect_state(current: HPRState, proximal: HPRState) -> HPRState:
    """Return ``w_hat = 2 w_bar - w`` without mutating either input."""

    if (
        current.y.shape != proximal.y.shape
        or current.z.shape != proximal.z.shape
        or current.x.shape != proximal.x.shape
    ):
        raise ValueError("current and proximal states must have matching block shapes.")
    return HPRState(
        y=2.0 * proximal.y - current.y,
        z=2.0 * proximal.z - current.z,
        x=2.0 * proximal.x - current.x,
    )


def halpern_update(anchor: HPRState, reflected: HPRState, *, iteration: int) -> HPRState:
    """Apply the fixed-anchor Halpern weights for zero-based ``iteration``."""

    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer.")
    if (
        anchor.y.shape != reflected.y.shape
        or anchor.z.shape != reflected.z.shape
        or anchor.x.shape != reflected.x.shape
    ):
        raise ValueError("anchor and reflected states must have matching block shapes.")
    anchor_weight = 1.0 / (iteration + 2.0)
    reflected_weight = (iteration + 1.0) / (iteration + 2.0)
    return HPRState(
        y=anchor_weight * anchor.y + reflected_weight * reflected.y,
        z=anchor_weight * anchor.z + reflected_weight * reflected.z,
        x=anchor_weight * anchor.x + reflected_weight * reflected.x,
    )


def hpr_step(
    lp: CanonicalLP,
    current: HPRState,
    anchor: HPRState,
    proximal: SpectralProximal,
    *,
    iteration: int,
    sigma: float,
) -> HPRStep:
    """Perform one exact generic Algorithm 1 iteration."""

    _validate_state(lp, current, name="current")
    _validate_state(lp, anchor, name="anchor")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite scalar.")

    A = lp.dense_A()
    q = current.x + sigma * (A.T @ current.y - lp.c)
    x_bar = project_box(q, lp.lower, lp.upper)
    z_bar = (x_bar - q) / sigma

    if lp.m == 0:
        y_bar = np.empty(0, dtype=np.float64)
    else:
        metric = A @ A.T + proximal.matrix
        right_hand_side = (
            proximal.matrix @ current.y + lp.b / sigma - A @ (x_bar / sigma + z_bar - lp.c)
        )
        unconstrained = np.linalg.solve(metric, right_hand_side)
        y_bar = project_dual_set(unconstrained, lp.m1)

    bar_state = HPRState(y=y_bar, z=z_bar, x=x_bar)
    reflected = reflect_state(current, bar_state)
    next_state = halpern_update(anchor, reflected, iteration=iteration)
    return HPRStep(proximal=bar_state, reflected=reflected, next_state=next_state)


def solve_hpr(
    lp: CanonicalLP,
    *,
    sigma: float = 1.0,
    tolerance: float = 5e-5,
    kkt_tolerance: float | None = None,
    max_iterations: int = 100_000,
    initial_state: HPRState | None = None,
    relative_proximal_margin: float = 1e-8,
) -> HPRResult:
    """Run Algorithm 1 and return the proximal state used for stopping.

    The paper stops when all three Equation (54) tests pass. Callers may also
    require the complete, unnormalized Equation (28) norm to meet
    ``kkt_tolerance``. Its scale depends on the LP data, so the default follows
    the paper-only stopping rule.
    """

    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite scalar.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite scalar.")
    if kkt_tolerance is not None and (not np.isfinite(kkt_tolerance) or kkt_tolerance <= 0.0):
        raise ValueError("kkt_tolerance must be None or a positive finite scalar.")

    if initial_state is None:
        initial_state = HPRState(
            y=np.zeros(lp.m, dtype=np.float64),
            z=np.zeros(lp.n, dtype=np.float64),
            x=np.zeros(lp.n, dtype=np.float64),
        )
    _validate_state(lp, initial_state, name="initial_state")
    anchor = initial_state.detached_copy()
    current = initial_state.detached_copy()
    proximal = construct_spectral_proximal(lp, relative_margin=relative_proximal_margin)

    history: list[HPRHistoryEntry] = []
    final_state = current
    final_residuals = evaluate_residuals(
        lp,
        x=current.x,
        y=current.y,
        z=current.z,
        tolerance=tolerance,
    )
    converged = False

    for iteration in range(max_iterations):
        step = hpr_step(
            lp,
            current,
            anchor,
            proximal,
            iteration=iteration,
            sigma=sigma,
        )
        final_state = step.proximal
        final_residuals = evaluate_residuals(
            lp,
            x=final_state.x,
            y=final_state.y,
            z=final_state.z,
            tolerance=tolerance,
        )
        normalized = final_residuals.paper_normalized_norms
        kkt_satisfied = kkt_tolerance is None or final_residuals.combined_norm <= kkt_tolerance
        minimum_inequality_multiplier = float(np.min(final_state.y[lp.m1 :])) if lp.m2 else None
        history.append(
            HPRHistoryEntry(
                iteration=iteration + 1,
                objective=float(lp.c @ final_state.x),
                kkt_combined_norm=final_residuals.combined_norm,
                paper_primal_normalized=normalized[0],
                paper_box_normalized=normalized[1],
                paper_stationarity_normalized=normalized[2],
                paper_stopping_satisfied=final_residuals.conditions.all_satisfied,
                kkt_target_satisfied=kkt_satisfied,
                minimum_inequality_multiplier=minimum_inequality_multiplier,
            )
        )
        current = step.next_state
        if final_residuals.conditions.all_satisfied and kkt_satisfied:
            converged = True
            break

    return HPRResult(
        solution=final_state.detached_copy(),
        current_state=current.detached_copy(),
        residuals=final_residuals,
        history=tuple(history),
        iterations=len(history),
        converged=converged,
        sigma=float(sigma),
        proximal=proximal,
    )
