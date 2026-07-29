"""Small, explicit projections used by the HPR reference implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatVector = NDArray[np.float64]


def _finite_vector(values: ArrayLike, *, name: str) -> FloatVector:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def project_box(values: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> FloatVector:
    """Project a vector onto the componentwise box ``[lower, upper]``."""

    vector = _finite_vector(values, name="values")
    lower_vector = _finite_vector(lower, name="lower")
    upper_vector = _finite_vector(upper, name="upper")
    if vector.shape != lower_vector.shape or vector.shape != upper_vector.shape:
        raise ValueError("values, lower, and upper must have the same shape.")
    if np.any(lower_vector > upper_vector):
        raise ValueError("lower must not exceed upper.")
    return np.clip(vector, lower_vector, upper_vector)


def project_nonnegative(values: ArrayLike) -> FloatVector:
    """Project a vector onto the nonnegative orthant."""

    vector = _finite_vector(values, name="values")
    return np.maximum(vector, 0.0)


def project_dual_set(values: ArrayLike, equality_rows: int) -> FloatVector:
    """Project onto ``D = R^m1 x R_+^m2``.

    Equality multipliers remain free; only the inequality-multiplier tail is
    clipped to zero.
    """

    vector = _finite_vector(values, name="values")
    if not isinstance(equality_rows, int):
        raise TypeError("equality_rows must be an integer.")
    if not 0 <= equality_rows <= vector.size:
        raise ValueError(
            f"equality_rows must be between 0 and {vector.size}; received {equality_rows}."
        )
    projected = vector.copy()
    projected[equality_rows:] = np.maximum(projected[equality_rows:], 0.0)
    return projected
