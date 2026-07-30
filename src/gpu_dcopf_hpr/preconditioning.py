"""Reversible sparse diagonal preconditioning for the canonical LP.

The implemented order and formulas follow the preprocessing described by the
DCOPF manuscript and the related HPR-LP implementation:

* HPR-LP source project: https://github.com/PolyU-IOR/HPR-LP
* Pock--Chambolle diagonal preconditioning:
  https://doi.org/10.1109/ICCV.2011.6126441

Each simultaneous Ruiz iteration divides the current matrix by the square
roots of its row and column infinity norms.  The optional Pock--Chambolle
``alpha=1`` step similarly uses row and column L1 sums.  Zero rows and columns
receive a neutral denominator of one.

For cumulative positive denominators ``r`` and ``d``, first define
``b_d = b / r`` and ``c_d = c / d``.  The optional norm factors are
``B = 1 + ||b_d||_2`` and ``C = 1 + ||c_d||_2``.  The transformed data are

``A_s = diag(1/r) A diag(1/d)``,
``b_s = b / (r B)``, ``c_s = c / (d C)``, and
``[l_s, u_s] = [l, u] d / B``.

The corresponding state maps are retained explicitly by :class:`LPPreconditioner`.
All matrix operations remain sparse; only row and column diagnostic vectors
are materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .canonical_lp import CanonicalLP
from .hpr_generic import HPRState

FloatVector = NDArray[np.float64]
ScalingMethod = Literal["ruiz", "pock_chambolle"]


def _readonly_vector(values: object, *, name: str, expected: int) -> FloatVector:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},); received {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    vector.setflags(write=False)
    return vector


def _canonical_csr(matrix: object, *, rows: int, columns: int) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    if result.shape != (rows, columns):
        raise ValueError(f"matrix must have shape {(rows, columns)}; received {result.shape}.")
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if not np.all(np.isfinite(result.data)):
        raise ValueError("matrix must contain only finite values.")
    return result


def _axis_maximum(matrix: sparse.csr_matrix, *, axis: int) -> FloatVector:
    if matrix.shape[axis] == 0:
        output_size = matrix.shape[1] if axis == 0 else matrix.shape[0]
        return np.zeros(output_size, dtype=np.float64)
    reduced = abs(matrix).max(axis=axis)
    values = reduced.toarray() if sparse.issparse(reduced) else reduced
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _axis_sum(matrix: sparse.csr_matrix, *, axis: int) -> FloatVector:
    if matrix.shape[axis] == 0:
        output_size = matrix.shape[1] if axis == 0 else matrix.shape[0]
        return np.zeros(output_size, dtype=np.float64)
    return np.asarray(abs(matrix).sum(axis=axis), dtype=np.float64).reshape(-1)


def _positive_sqrt_denominator(values: FloatVector, *, name: str) -> FloatVector:
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} norms must be finite and nonnegative.")
    denominator = np.ones(values.size, dtype=np.float64)
    positive = values > 0.0
    denominator[positive] = np.sqrt(values[positive])
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError(f"{name} denominators must be finite and positive.")
    return denominator


def _divide_sparse(
    matrix: sparse.csr_matrix,
    row_denominator: FloatVector,
    column_denominator: FloatVector,
) -> sparse.csr_matrix:
    scaled = matrix.multiply((1.0 / row_denominator)[:, np.newaxis])
    scaled = scaled.multiply((1.0 / column_denominator)[np.newaxis, :]).tocsr()
    scaled.sum_duplicates()
    scaled.eliminate_zeros()
    scaled.sort_indices()
    if not np.all(np.isfinite(scaled.data)):
        raise ValueError("preconditioning produced nonfinite matrix coefficients.")
    return scaled


def _multiply_sparse(
    matrix: sparse.csr_matrix,
    row_multiplier: FloatVector,
    column_multiplier: FloatVector,
) -> sparse.csr_matrix:
    restored = matrix.multiply(row_multiplier[:, np.newaxis])
    restored = restored.multiply(column_multiplier[np.newaxis, :]).tocsr()
    restored.sum_duplicates()
    restored.eliminate_zeros()
    restored.sort_indices()
    if not np.all(np.isfinite(restored.data)):
        raise ValueError("inverse preconditioning produced nonfinite matrix coefficients.")
    return restored


@dataclass(frozen=True, slots=True)
class NormSummary:
    """Compact immutable summary of one row- or column-norm vector."""

    count: int
    zero_count: int
    minimum: float
    maximum: float
    minimum_positive: float | None


def _norm_summary(values: FloatVector) -> NormSummary:
    if values.size == 0:
        return NormSummary(
            count=0,
            zero_count=0,
            minimum=0.0,
            maximum=0.0,
            minimum_positive=None,
        )
    positive = values[values > 0.0]
    return NormSummary(
        count=int(values.size),
        zero_count=int(np.count_nonzero(values == 0.0)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        minimum_positive=(float(np.min(positive)) if positive.size else None),
    )


@dataclass(frozen=True, slots=True)
class ScalingIterationDiagnostics:
    """Norms before and after one simultaneous diagonal scaling step."""

    method: ScalingMethod
    iteration: int
    norm: Literal["infinity", "l1"]
    row_before: NormSummary
    column_before: NormSummary
    row_after: NormSummary
    column_after: NormSummary


@dataclass(frozen=True, slots=True)
class PreconditioningDiagnostics:
    """Auditable summary of the complete sparse preprocessing pipeline."""

    ruiz_iterations: int
    pock_chambolle_applied: bool
    normalization_applied: bool
    original_nnz: int
    scaled_nnz: int
    b_norm: float
    c_norm: float
    iterations: tuple[ScalingIterationDiagnostics, ...]

    @property
    def nnz_preserved(self) -> bool:
        return self.original_nnz == self.scaled_nnz


@dataclass(frozen=True, slots=True)
class LPPreconditioner:
    """Immutable transformed LP plus exact data and state recovery factors."""

    source_lp: CanonicalLP
    scaled_lp: CanonicalLP
    row_denominator: object
    column_denominator: object
    b_scale: float
    c_scale: float
    diagnostics: PreconditioningDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.source_lp, CanonicalLP):
            raise TypeError("source_lp must be a CanonicalLP.")
        if not isinstance(self.scaled_lp, CanonicalLP):
            raise TypeError("scaled_lp must be a CanonicalLP.")
        if (
            self.source_lp.n != self.scaled_lp.n
            or self.source_lp.m1 != self.scaled_lp.m1
            or self.source_lp.m2 != self.scaled_lp.m2
        ):
            raise ValueError("source_lp and scaled_lp dimensions must match.")
        row = _readonly_vector(
            self.row_denominator,
            name="row_denominator",
            expected=self.source_lp.m,
        )
        column = _readonly_vector(
            self.column_denominator,
            name="column_denominator",
            expected=self.source_lp.n,
        )
        b_scale = float(self.b_scale)
        c_scale = float(self.c_scale)
        if not np.isfinite(b_scale) or b_scale <= 0.0:
            raise ValueError("b_scale must be a positive finite scalar.")
        if not np.isfinite(c_scale) or c_scale <= 0.0:
            raise ValueError("c_scale must be a positive finite scalar.")
        if not isinstance(self.diagnostics, PreconditioningDiagnostics):
            raise TypeError("diagnostics must be PreconditioningDiagnostics.")
        object.__setattr__(self, "row_denominator", row)
        object.__setattr__(self, "column_denominator", column)
        object.__setattr__(self, "b_scale", b_scale)
        object.__setattr__(self, "c_scale", c_scale)

    @property
    def objective_factor(self) -> float:
        """Factor satisfying ``c.T x = objective_factor * c_s.T x_s``."""

        factor = self.b_scale * self.c_scale
        if not np.isfinite(factor):
            raise ValueError("objective recovery factor overflowed FP64.")
        return float(factor)

    def _validate_state(self, state: HPRState, *, space: str) -> None:
        if not isinstance(state, HPRState):
            raise TypeError(f"{space} state must be an HPRState.")
        expected = self.source_lp
        if state.x.shape != (expected.n,):
            raise ValueError(
                f"{space} state.x must have shape ({expected.n},); received {state.x.shape}."
            )
        if state.z.shape != (expected.n,):
            raise ValueError(
                f"{space} state.z must have shape ({expected.n},); received {state.z.shape}."
            )
        if state.y.shape != (expected.m,):
            raise ValueError(
                f"{space} state.y must have shape ({expected.m},); received {state.y.shape}."
            )

    def recover_state(self, state: HPRState) -> HPRState:
        """Map a scaled state back to the original canonical LP."""

        self._validate_state(state, space="scaled")
        recovered = HPRState(
            x=self.b_scale * state.x / self.column_denominator,
            y=self.c_scale * state.y / self.row_denominator,
            z=self.c_scale * state.z * self.column_denominator,
        )
        if not all(
            np.all(np.isfinite(vector)) for vector in (recovered.x, recovered.y, recovered.z)
        ):
            raise ValueError("state recovery produced nonfinite values.")
        return recovered

    def scale_state(self, state: HPRState) -> HPRState:
        """Map an original state into scaled coordinates."""

        self._validate_state(state, space="original")
        scaled = HPRState(
            x=state.x * self.column_denominator / self.b_scale,
            y=state.y * self.row_denominator / self.c_scale,
            z=state.z / (self.c_scale * self.column_denominator),
        )
        if not all(np.all(np.isfinite(vector)) for vector in (scaled.x, scaled.y, scaled.z)):
            raise ValueError("state scaling produced nonfinite values.")
        return scaled

    def original_objective_from_scaled(self, value: float) -> float:
        """Recover an original variable objective from its scaled value."""

        scaled_value = float(value)
        if not np.isfinite(scaled_value):
            raise ValueError("scaled objective must be a finite scalar.")
        original = self.objective_factor * scaled_value
        if not np.isfinite(original):
            raise ValueError("objective recovery produced a nonfinite value.")
        return float(original)

    def recover_lp(self) -> CanonicalLP:
        """Invert the data transform without densifying either matrix block."""

        scaled_matrix = _canonical_csr(
            self.scaled_lp.A,
            rows=self.scaled_lp.m,
            columns=self.scaled_lp.n,
        )
        original_matrix = _multiply_sparse(
            scaled_matrix,
            self.row_denominator,
            self.column_denominator,
        )
        original_b = self.scaled_lp.b * self.row_denominator * self.b_scale
        original_c = self.scaled_lp.c * self.column_denominator * self.c_scale
        original_lower = self.scaled_lp.lower * self.b_scale / self.column_denominator
        original_upper = self.scaled_lp.upper * self.b_scale / self.column_denominator
        if not all(
            np.all(np.isfinite(vector))
            for vector in (original_b, original_c, original_lower, original_upper)
        ):
            raise ValueError("inverse preconditioning produced nonfinite LP data.")
        return CanonicalLP(
            c=original_c,
            A1=original_matrix[: self.source_lp.m1],
            b1=original_b[: self.source_lp.m1],
            A2=original_matrix[self.source_lp.m1 :],
            b2=original_b[self.source_lp.m1 :],
            lower=original_lower,
            upper=original_upper,
        )


def _apply_scaling_step(
    matrix: sparse.csr_matrix,
    *,
    method: ScalingMethod,
    iteration: int,
) -> tuple[
    sparse.csr_matrix,
    FloatVector,
    FloatVector,
    ScalingIterationDiagnostics,
]:
    if method == "ruiz":
        row_before = _axis_maximum(matrix, axis=1)
        column_before = _axis_maximum(matrix, axis=0)
        norm_name: Literal["infinity", "l1"] = "infinity"
    else:
        row_before = _axis_sum(matrix, axis=1)
        column_before = _axis_sum(matrix, axis=0)
        norm_name = "l1"
    row_step = _positive_sqrt_denominator(row_before, name=f"{method} row")
    column_step = _positive_sqrt_denominator(
        column_before,
        name=f"{method} column",
    )
    scaled = _divide_sparse(matrix, row_step, column_step)
    if method == "ruiz":
        row_after = _axis_maximum(scaled, axis=1)
        column_after = _axis_maximum(scaled, axis=0)
    else:
        row_after = _axis_sum(scaled, axis=1)
        column_after = _axis_sum(scaled, axis=0)
    diagnostics = ScalingIterationDiagnostics(
        method=method,
        iteration=iteration,
        norm=norm_name,
        row_before=_norm_summary(row_before),
        column_before=_norm_summary(column_before),
        row_after=_norm_summary(row_after),
        column_after=_norm_summary(column_after),
    )
    return scaled, row_step, column_step, diagnostics


def precondition_lp(
    lp: CanonicalLP,
    *,
    ruiz_iterations: int = 0,
    pock_chambolle: bool = False,
    normalize: bool = False,
) -> LPPreconditioner:
    """Build a sparse, reversible positive-diagonal LP transformation."""

    if not isinstance(lp, CanonicalLP):
        raise TypeError("lp must be a CanonicalLP.")
    if (
        not isinstance(ruiz_iterations, int)
        or isinstance(ruiz_iterations, bool)
        or ruiz_iterations < 0
    ):
        raise ValueError("ruiz_iterations must be a nonnegative integer.")
    if not isinstance(pock_chambolle, bool):
        raise TypeError("pock_chambolle must be a bool.")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a bool.")

    matrix = _canonical_csr(lp.A, rows=lp.m, columns=lp.n)
    original_nnz = int(matrix.nnz)
    row_denominator = np.ones(lp.m, dtype=np.float64)
    column_denominator = np.ones(lp.n, dtype=np.float64)
    iteration_diagnostics: list[ScalingIterationDiagnostics] = []

    for iteration in range(1, ruiz_iterations + 1):
        matrix, row_step, column_step, step_diagnostics = _apply_scaling_step(
            matrix,
            method="ruiz",
            iteration=iteration,
        )
        row_denominator *= row_step
        column_denominator *= column_step
        if (
            not np.all(np.isfinite(row_denominator))
            or not np.all(np.isfinite(column_denominator))
            or np.any(row_denominator <= 0.0)
            or np.any(column_denominator <= 0.0)
        ):
            raise ValueError("cumulative Ruiz denominators left the positive FP64 range.")
        iteration_diagnostics.append(step_diagnostics)

    if pock_chambolle:
        matrix, row_step, column_step, step_diagnostics = _apply_scaling_step(
            matrix,
            method="pock_chambolle",
            iteration=1,
        )
        row_denominator *= row_step
        column_denominator *= column_step
        if (
            not np.all(np.isfinite(row_denominator))
            or not np.all(np.isfinite(column_denominator))
            or np.any(row_denominator <= 0.0)
            or np.any(column_denominator <= 0.0)
        ):
            raise ValueError(
                "cumulative Pock--Chambolle denominators left the positive FP64 range."
            )
        iteration_diagnostics.append(step_diagnostics)

    if matrix.nnz != original_nnz:
        raise ValueError(
            "positive diagonal preconditioning changed the sparse nonzero count: "
            f"{original_nnz} -> {matrix.nnz}."
        )

    diagonally_scaled_b = lp.b / row_denominator
    diagonally_scaled_c = lp.c / column_denominator
    b_norm = float(np.linalg.norm(diagonally_scaled_b))
    c_norm = float(np.linalg.norm(diagonally_scaled_c))
    if not np.isfinite(b_norm) or not np.isfinite(c_norm):
        raise ValueError("b and c norms must remain finite in FP64.")
    b_scale = 1.0 + b_norm if normalize else 1.0
    c_scale = 1.0 + c_norm if normalize else 1.0
    if not np.isfinite(b_scale) or not np.isfinite(c_scale) or b_scale <= 0.0 or c_scale <= 0.0:
        raise ValueError("normalization scales must be finite and positive.")

    scaled_b = diagonally_scaled_b / b_scale
    scaled_c = diagonally_scaled_c / c_scale
    scaled_lower = lp.lower * column_denominator / b_scale
    scaled_upper = lp.upper * column_denominator / b_scale
    if not all(
        np.all(np.isfinite(vector)) for vector in (scaled_b, scaled_c, scaled_lower, scaled_upper)
    ):
        raise ValueError("preconditioning produced nonfinite LP vectors.")

    scaled_lp = CanonicalLP(
        c=scaled_c,
        A1=matrix[: lp.m1],
        b1=scaled_b[: lp.m1],
        A2=matrix[lp.m1 :],
        b2=scaled_b[lp.m1 :],
        lower=scaled_lower,
        upper=scaled_upper,
    )
    diagnostics = PreconditioningDiagnostics(
        ruiz_iterations=ruiz_iterations,
        pock_chambolle_applied=pock_chambolle,
        normalization_applied=normalize,
        original_nnz=original_nnz,
        scaled_nnz=int(matrix.nnz),
        b_norm=b_norm,
        c_norm=c_norm,
        iterations=tuple(iteration_diagnostics),
    )
    return LPPreconditioner(
        source_lp=lp,
        scaled_lp=scaled_lp,
        row_denominator=row_denominator,
        column_denominator=column_denominator,
        b_scale=b_scale,
        c_scale=c_scale,
        diagnostics=diagnostics,
    )
