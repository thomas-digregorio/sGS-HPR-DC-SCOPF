"""Validated canonical linear-program representation used by the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

FloatVector = NDArray[np.float64]
Matrix = NDArray[np.float64] | sparse.spmatrix


def _as_vector(values: ArrayLike, *, name: str, expected: int | None = None) -> FloatVector:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received shape {vector.shape}.")
    if expected is not None and vector.size != expected:
        raise ValueError(f"{name} must have length {expected}; received {vector.size}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    vector.setflags(write=False)
    return vector


def _as_matrix(values: Any, *, name: str, columns: int) -> Matrix:
    if sparse.issparse(values):
        matrix = values.astype(np.float64).tocsr(copy=True)
        if matrix.shape[1] != columns:
            raise ValueError(f"{name} must have {columns} columns; received shape {matrix.shape}.")
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError(f"{name} must contain only finite values.")
        return matrix

    matrix = np.array(values, dtype=np.float64, copy=True)
    if matrix.size == 0 and matrix.ndim in {1, 2}:
        matrix = np.empty((0, columns), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; received shape {matrix.shape}.")
    if matrix.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns; received shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    matrix.setflags(write=False)
    return matrix


def dense_matrix(matrix: Matrix) -> NDArray[np.float64]:
    """Return a detached dense FP64 copy of a dense or sparse matrix."""

    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.array(matrix, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class CanonicalLP:
    """The paper's LP: ``min c^T x`` with equality, >=, and box constraints."""

    c: ArrayLike
    A1: Any
    b1: ArrayLike
    A2: Any
    b2: ArrayLike
    lower: ArrayLike
    upper: ArrayLike

    def __post_init__(self) -> None:
        c = _as_vector(self.c, name="c")
        if c.size == 0:
            raise ValueError("c must describe at least one decision variable.")
        lower = _as_vector(self.lower, name="lower", expected=c.size)
        upper = _as_vector(self.upper, name="upper", expected=c.size)
        if np.any(lower > upper):
            bad = int(np.flatnonzero(lower > upper)[0])
            raise ValueError(
                f"lower must not exceed upper; index {bad} has {lower[bad]} > {upper[bad]}."
            )

        A1 = _as_matrix(self.A1, name="A1", columns=c.size)
        A2 = _as_matrix(self.A2, name="A2", columns=c.size)
        b1 = _as_vector(self.b1, name="b1", expected=A1.shape[0])
        b2 = _as_vector(self.b2, name="b2", expected=A2.shape[0])

        object.__setattr__(self, "c", c)
        object.__setattr__(self, "A1", A1)
        object.__setattr__(self, "b1", b1)
        object.__setattr__(self, "A2", A2)
        object.__setattr__(self, "b2", b2)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def n(self) -> int:
        return int(self.c.size)

    @property
    def m1(self) -> int:
        return int(self.b1.size)

    @property
    def m2(self) -> int:
        return int(self.b2.size)

    @property
    def m(self) -> int:
        return self.m1 + self.m2

    @property
    def A(self) -> Matrix:
        """Return ``[A1; A2]`` while retaining sparsity when either block is sparse."""

        if sparse.issparse(self.A1) or sparse.issparse(self.A2):
            return sparse.vstack((self.A1, self.A2), format="csr", dtype=np.float64)
        return np.vstack((self.A1, self.A2))

    @property
    def b(self) -> FloatVector:
        return np.concatenate((self.b1, self.b2))

    def dense_A(self) -> NDArray[np.float64]:
        return dense_matrix(self.A)
