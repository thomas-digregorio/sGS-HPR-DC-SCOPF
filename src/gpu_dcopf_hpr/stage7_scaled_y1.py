"""Generalized block-arrow equality solve for diagonally scaled DCOPF models.

Diagonal row and column scaling destroys the rank-one symmetry used by the
unscaled Stage 4 formula.  It preserves the arrow sparsity, however.  This
module eliminates the diagonal time-balance block, factors only the storage
Schur complement, and never materializes the full ``A1 @ A1.T`` matrix.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse

from .gpu_backend import CuPyBackend, GPUBackendUnavailable
from .preconditioning import LPPreconditioner
from .structural_y1 import DCOPFEqualityStructure, prepare_structural_y1

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


def _readonly_vector(values: ArrayLike) -> FloatVector:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_matrix(values: ArrayLike) -> FloatMatrix:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _canonical_csr(matrix: Any) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _validate_diagonal_scaling(
    preconditioner: LPPreconditioner,
    *,
    tolerance_multiplier: float,
) -> sparse.csr_matrix:
    source = _canonical_csr(preconditioner.source_lp.A1)
    scaled = _canonical_csr(preconditioner.scaled_lp.A1)
    row = np.asarray(preconditioner.row_denominator[: preconditioner.source_lp.m1])
    column = np.asarray(preconditioner.column_denominator)
    expected = source.multiply((1.0 / row)[:, np.newaxis])
    expected = expected.multiply((1.0 / column)[np.newaxis, :]).tocsr()
    expected = _canonical_csr(expected)

    if scaled.shape != expected.shape:
        raise ValueError("scaled A1 has incompatible dimensions.")
    if not np.array_equal(scaled.indptr, expected.indptr) or not np.array_equal(
        scaled.indices,
        expected.indices,
    ):
        raise ValueError("scaled A1 has incompatible sparsity for diagonal DCOPF scaling.")
    scale = max(1.0, float(np.max(np.abs(expected.data), initial=0.0)))
    tolerance = tolerance_multiplier * np.finfo(np.float64).eps * scale
    error = float(np.max(np.abs(scaled.data - expected.data), initial=0.0))
    if error > tolerance:
        raise ValueError(
            "scaled A1 is not the stated row/column diagonal transform; "
            f"maximum coefficient error {error} exceeds {tolerance}."
        )
    return scaled


@dataclass(frozen=True, slots=True)
class ScaledBlockArrowDiagnostics:
    """Preparation evidence for the scaled equality block-arrow solve."""

    periods: int
    storage_count: int
    equality_rows: int
    scaled_a1_nonzeros: int
    balance_diagonal_minimum: float
    balance_diagonal_maximum: float
    storage_diagonal_minimum: float | None
    storage_diagonal_maximum: float | None
    schur_minimum_cholesky_diagonal: float | None
    schur_maximum_cholesky_diagonal: float | None
    stored_float_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "periods": self.periods,
            "storage_count": self.storage_count,
            "equality_rows": self.equality_rows,
            "scaled_a1_nonzeros": self.scaled_a1_nonzeros,
            "balance_diagonal_range": [
                self.balance_diagonal_minimum,
                self.balance_diagonal_maximum,
            ],
            "storage_diagonal_range": (
                None
                if self.storage_diagonal_minimum is None
                else [self.storage_diagonal_minimum, self.storage_diagonal_maximum]
            ),
            "schur_cholesky_diagonal_range": (
                None
                if self.schur_minimum_cholesky_diagonal is None
                else [
                    self.schur_minimum_cholesky_diagonal,
                    self.schur_maximum_cholesky_diagonal,
                ]
            ),
            "stored_float_count": self.stored_float_count,
            "solve_complexity": "O(T*N_ESS + N_ESS^2) after preparation",
            "dense_equality_gram_materialized": False,
            "dense_schur_shape": [self.storage_count, self.storage_count],
        }


@dataclass(frozen=True, slots=True)
class ScaledBlockArrowY1Solver:
    """Prepared CPU solver for one diagonally scaled DCOPF equality matrix."""

    preconditioner: LPPreconditioner
    structure: DCOPFEqualityStructure
    inverse_balance_diagonal: FloatVector
    coupling: FloatMatrix
    storage_schur_cholesky: FloatMatrix
    diagnostics: ScaledBlockArrowDiagnostics

    @property
    def backend(self) -> str:
        return "scaled_block_arrow_cpu"

    @property
    def source_lp(self) -> Any:
        """Scaled LP identity expected by the existing CPU workspace checks."""

        return self.preconditioner.scaled_lp

    def solve(self, right_hand_side: ArrayLike) -> FloatVector:
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        expected = self.diagnostics.equality_rows
        if rhs.shape != (expected,):
            raise ValueError(
                f"right_hand_side must have shape ({expected},); received {rhs.shape}."
            )
        if not np.all(np.isfinite(rhs)):
            raise ValueError("right_hand_side must contain only finite values.")

        periods = self.diagnostics.periods
        balance_rhs = rhs[:periods]
        weighted_balance_rhs = self.inverse_balance_diagonal * balance_rhs
        if self.diagnostics.storage_count == 0:
            return np.asarray(weighted_balance_rhs, dtype=np.float64)

        storage_rhs = rhs[periods:]
        reduced_storage_rhs = storage_rhs - self.coupling.T @ weighted_balance_rhs
        storage_solution = linalg.cho_solve(
            (self.storage_schur_cholesky, True),
            reduced_storage_rhs,
            check_finite=False,
        )
        balance_solution = self.inverse_balance_diagonal * (
            balance_rhs - self.coupling @ storage_solution
        )
        return np.concatenate((balance_solution, storage_solution))

    def solve_into(self, right_hand_side: ArrayLike, out: Any) -> Any:
        """Compatibility helper for workspaces that own a reusable output array."""

        expected = self.diagnostics.equality_rows
        if tuple(out.shape) != (expected,):
            raise ValueError(f"out must have shape ({expected},); received {out.shape}.")
        out[...] = self.solve(right_hand_side)
        return out

    def to_device(
        self,
        backend: CuPyBackend,
        *,
        phase: str = "stage7_scaled_equality_setup",
        triangular_solve: Any | None = None,
    ) -> DeviceScaledBlockArrowY1Solver:
        """Upload the prepared factors once for repeated device-resident solves."""

        if triangular_solve is None:
            try:
                triangular_solve = importlib.import_module("cupyx.scipy.linalg").solve_triangular
            except Exception as error:
                raise GPUBackendUnavailable(
                    "CuPy is available, but cupyx.scipy.linalg.solve_triangular is not."
                ) from error
        return DeviceScaledBlockArrowY1Solver(
            backend=backend,
            inverse_balance_diagonal=backend.to_device(
                self.inverse_balance_diagonal,
                phase=phase,
                kind="vector",
            ),
            coupling=backend.to_device(self.coupling, phase=phase, kind="matrix"),
            storage_schur_cholesky=backend.to_device(
                self.storage_schur_cholesky,
                phase=phase,
                kind="matrix",
            ),
            triangular_solve=triangular_solve,
            source_lp=self.source_lp,
            periods=self.diagnostics.periods,
            storage_count=self.diagnostics.storage_count,
            storage_work=backend.xp.empty(
                self.diagnostics.storage_count,
                dtype=backend.xp.float64,
            ),
        )


@dataclass(frozen=True, slots=True)
class DeviceScaledBlockArrowY1Solver:
    """Optional device-resident counterpart of :class:`ScaledBlockArrowY1Solver`."""

    backend: CuPyBackend
    inverse_balance_diagonal: Any
    coupling: Any
    storage_schur_cholesky: Any
    triangular_solve: Any
    source_lp: Any
    periods: int
    storage_count: int
    storage_work: Any

    @property
    def backend_name(self) -> str:
        return "scaled_block_arrow_device"

    def solve(self, right_hand_side: Any) -> Any:
        """Solve on device; callers retain responsibility for finite device inputs."""

        expected = self.periods + self.storage_count
        if tuple(right_hand_side.shape) != (expected,):
            raise ValueError(
                f"right_hand_side must have shape ({expected},); received {right_hand_side.shape}."
            )
        out = self.backend.xp.empty(expected, dtype=right_hand_side.dtype)
        return self.solve_into(right_hand_side, out)

    def solve_into(self, right_hand_side: Any, out: Any) -> Any:
        """Solve into an existing device vector for Stage 7 workspace integration."""

        expected = self.periods + self.storage_count
        if tuple(right_hand_side.shape) != (expected,):
            raise ValueError(
                f"right_hand_side must have shape ({expected},); received {right_hand_side.shape}."
            )
        if tuple(out.shape) != (expected,):
            raise ValueError(f"out must have shape ({expected},); received {out.shape}.")
        balance_rhs = right_hand_side[: self.periods]
        balance_solution = out[: self.periods]
        balance_solution[...] = balance_rhs
        balance_solution *= self.inverse_balance_diagonal
        if self.storage_count == 0:
            return out

        storage_solution = out[self.periods :]
        work = self.storage_work
        work[...] = right_hand_side[self.periods :]
        work -= self.coupling.T @ balance_solution
        intermediate = self.triangular_solve(
            self.storage_schur_cholesky,
            work,
            lower=True,
            check_finite=False,
            overwrite_b=True,
        )
        if intermediate is not work:
            work[...] = intermediate
        solved = self.triangular_solve(
            self.storage_schur_cholesky.T,
            work,
            lower=False,
            check_finite=False,
            overwrite_b=True,
        )
        storage_solution[...] = solved
        balance_solution[...] = balance_rhs
        balance_solution -= self.coupling @ storage_solution
        balance_solution *= self.inverse_balance_diagonal
        return out


def prepare_scaled_block_arrow_y1(
    preconditioner: LPPreconditioner,
    structure: DCOPFEqualityStructure,
    *,
    scaling_tolerance_multiplier: float = 512.0,
) -> ScaledBlockArrowY1Solver:
    """Validate diagonal scaling and prepare the generalized block-arrow solve."""

    if not isinstance(preconditioner, LPPreconditioner):
        raise TypeError("preconditioner must be an LPPreconditioner.")
    if not isinstance(structure, DCOPFEqualityStructure):
        raise TypeError("structure must be a DCOPFEqualityStructure.")
    if not np.isfinite(scaling_tolerance_multiplier) or scaling_tolerance_multiplier <= 0.0:
        raise ValueError("scaling_tolerance_multiplier must be positive and finite.")

    # Reuse the independently tested raw Equation (55) validator before
    # accepting a scaled descendant of that matrix.
    prepare_structural_y1(preconditioner.source_lp, structure)
    scaled = _validate_diagonal_scaling(
        preconditioner,
        tolerance_multiplier=float(scaling_tolerance_multiplier),
    )
    periods = structure.periods
    storage_count = structure.storage_count
    balance = scaled[:periods]
    storage = scaled[periods:]
    balance_diagonal = np.asarray(balance.multiply(balance).sum(axis=1)).reshape(-1)
    storage_diagonal = np.asarray(storage.multiply(storage).sum(axis=1)).reshape(-1)
    coupling = np.asarray((balance @ storage.T).toarray(), dtype=np.float64)

    if (
        balance_diagonal.shape != (periods,)
        or np.any(balance_diagonal <= 0.0)
        or not np.all(np.isfinite(balance_diagonal))
    ):
        raise ValueError("the scaled balance diagonal must be finite and positive.")
    if (
        storage_diagonal.shape != (storage_count,)
        or np.any(storage_diagonal <= 0.0)
        or not np.all(np.isfinite(storage_diagonal))
        or not np.all(np.isfinite(coupling))
    ):
        raise ValueError("the scaled storage arrow blocks must be finite and positive.")

    inverse_balance = 1.0 / balance_diagonal
    if storage_count:
        schur = np.diag(storage_diagonal) - coupling.T @ (inverse_balance[:, np.newaxis] * coupling)
        schur = np.asarray(0.5 * (schur + schur.T), dtype=np.float64)
        try:
            cholesky = np.linalg.cholesky(schur)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "the scaled storage Schur complement is not numerically positive definite."
            ) from error
        cholesky_diagonal = np.diag(cholesky)
        schur_minimum = float(np.min(cholesky_diagonal))
        schur_maximum = float(np.max(cholesky_diagonal))
        storage_minimum = float(np.min(storage_diagonal))
        storage_maximum = float(np.max(storage_diagonal))
    else:
        cholesky = np.empty((0, 0), dtype=np.float64)
        schur_minimum = None
        schur_maximum = None
        storage_minimum = None
        storage_maximum = None

    inverse_balance = _readonly_vector(inverse_balance)
    coupling = _readonly_matrix(coupling)
    cholesky = _readonly_matrix(cholesky)
    diagnostics = ScaledBlockArrowDiagnostics(
        periods=periods,
        storage_count=storage_count,
        equality_rows=structure.expected_equalities,
        scaled_a1_nonzeros=int(scaled.nnz),
        balance_diagonal_minimum=float(np.min(balance_diagonal)),
        balance_diagonal_maximum=float(np.max(balance_diagonal)),
        storage_diagonal_minimum=storage_minimum,
        storage_diagonal_maximum=storage_maximum,
        schur_minimum_cholesky_diagonal=schur_minimum,
        schur_maximum_cholesky_diagonal=schur_maximum,
        stored_float_count=(periods + periods * storage_count + storage_count * storage_count),
    )
    return ScaledBlockArrowY1Solver(
        preconditioner=preconditioner,
        structure=structure,
        inverse_balance_diagonal=inverse_balance,
        coupling=coupling,
        storage_schur_cholesky=cholesky,
        diagnostics=diagnostics,
    )


__all__ = [
    "DeviceScaledBlockArrowY1Solver",
    "ScaledBlockArrowDiagnostics",
    "ScaledBlockArrowY1Solver",
    "prepare_scaled_block_arrow_y1",
]
