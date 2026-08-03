"""Sparse-only spectral safeguards for the scalable Stage 7 solver.

The power iteration in this module is an estimate, not a certificate.  The
value exposed as :attr:`SparseSpectralCertificate.lambda_used` is instead
inflated above the best of three sparse upper bounds: the Frobenius bound, the
induced one/infinity-norm bound, and a Collatz bound for the absolute normal
operator.  No dense copy or sparse Gram matrix is formed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

FloatVector = NDArray[np.float64]
_HASH_CHUNK_ELEMENTS = 131_072


def _contains_explicit_zero(values: Any) -> bool:
    source = np.asarray(values).reshape(-1)
    for start in range(0, int(source.size), _HASH_CHUNK_ELEMENTS):
        stop = min(start + _HASH_CHUNK_ELEMENTS, int(source.size))
        if np.any(source[start:stop] == 0.0):
            return True
    return False


def _update_hash_array(
    digest: Any,
    name: bytes,
    values: Any,
    *,
    dtype: str,
    require_finite: bool = False,
) -> None:
    source = np.asarray(values).reshape(-1)
    target_dtype = np.dtype(dtype)
    digest.update(name)
    digest.update(b"\0")
    byte_count = int(source.size) * int(target_dtype.itemsize)
    digest.update(byte_count.to_bytes(8, byteorder="little", signed=False))
    for start in range(0, int(source.size), _HASH_CHUNK_ELEMENTS):
        stop = min(start + _HASH_CHUNK_ELEMENTS, int(source.size))
        chunk = np.asarray(source[start:stop], dtype=target_dtype)
        if require_finite and not np.all(np.isfinite(chunk)):
            raise ValueError("matrix must contain only finite values.")
        digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))


def canonical_csr_sha256(matrix: sparse.spmatrix) -> str:
    """Fingerprint canonical CSR bytes without a matrix-sized hash payload copy."""

    if not sparse.issparse(matrix):
        raise TypeError("matrix must be a SciPy sparse matrix.")
    can_reuse = (
        sparse.isspmatrix_csr(matrix)
        and matrix.dtype == np.dtype(np.float64)
        and matrix.has_canonical_format
        and matrix.has_sorted_indices
        and not _contains_explicit_zero(matrix.data)
    )
    if can_reuse:
        canonical = matrix
    else:
        canonical = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
        canonical.sum_duplicates()
        canonical.eliminate_zeros()
        canonical.sort_indices()

    digest = hashlib.sha256()
    digest.update(b"stage7-canonical-csr-v1\0")
    _update_hash_array(
        digest,
        b"shape",
        canonical.shape,
        dtype="<i8",
    )
    _update_hash_array(
        digest,
        b"indptr",
        canonical.indptr,
        dtype="<i8",
    )
    _update_hash_array(
        digest,
        b"indices",
        canonical.indices,
        dtype="<i8",
    )
    _update_hash_array(
        digest,
        b"data",
        canonical.data,
        dtype="<f8",
        require_finite=True,
    )
    return digest.hexdigest()


def _positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _normal_matvec(
    matrix: sparse.csr_matrix,
    transpose: sparse.csr_matrix,
    vector: FloatVector,
    *,
    row_space: bool,
) -> FloatVector:
    if row_space:
        return np.asarray(matrix @ (transpose @ vector), dtype=np.float64).reshape(-1)
    return np.asarray(transpose @ (matrix @ vector), dtype=np.float64).reshape(-1)


@dataclass(frozen=True, slots=True)
class SparseSpectralCertificate:
    """Auditable sparse estimate and conservative value for ``||A2||_2^2``."""

    rows: int
    columns: int
    nonzeros: int
    matrix_sha256: str
    operator_dimension: int
    power_seed: int
    power_iterations: int
    power_converged: bool
    rayleigh_estimate: float
    power_residual: float
    frobenius_upper_bound: float
    induced_norm_upper_bound: float
    collatz_upper_bound: float
    collatz_iterations: int
    certified_upper_bound: float
    relative_safety_margin: float
    requested_safety_margin: float
    roundoff_safety_margin: float
    lambda_used: float

    @property
    def inequality_lambda(self) -> float:
        """Alias accepted by :func:`gpu_sgs_hpr.prepare_gpu_sgs_hpr`."""

        return self.lambda_used

    @property
    def certificate_gap(self) -> float:
        return self.lambda_used - self.rayleigh_estimate

    @property
    def finite_certificate(self) -> bool:
        values = (
            self.rayleigh_estimate,
            self.power_residual,
            self.frobenius_upper_bound,
            self.induced_norm_upper_bound,
            self.collatz_upper_bound,
            self.certified_upper_bound,
            self.relative_safety_margin,
            self.requested_safety_margin,
            self.roundoff_safety_margin,
            self.lambda_used,
        )
        return all(np.isfinite(value) for value in values) and self.lambda_used > 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "nonzeros": self.nonzeros,
            "matrix_sha256": self.matrix_sha256,
            "operator_dimension": self.operator_dimension,
            "power_seed": self.power_seed,
            "power_iterations": self.power_iterations,
            "power_converged": self.power_converged,
            "rayleigh_estimate": self.rayleigh_estimate,
            "power_residual": self.power_residual,
            "frobenius_upper_bound": self.frobenius_upper_bound,
            "induced_norm_upper_bound": self.induced_norm_upper_bound,
            "collatz_upper_bound": self.collatz_upper_bound,
            "collatz_iterations": self.collatz_iterations,
            "certified_upper_bound": self.certified_upper_bound,
            "relative_safety_margin": self.relative_safety_margin,
            "requested_safety_margin": self.requested_safety_margin,
            "roundoff_safety_margin": self.roundoff_safety_margin,
            "lambda_used": self.lambda_used,
            "inequality_lambda": self.inequality_lambda,
            "certificate_gap": self.certificate_gap,
            "finite_certificate": self.finite_certificate,
            "dense_matrix_materialized": False,
            "normal_matrix_materialized": False,
        }


def _power_estimate(
    matrix: sparse.csr_matrix,
    transpose: sparse.csr_matrix,
    *,
    dimension: int,
    row_space: bool,
    tolerance: float,
    max_iterations: int,
    seed: int,
) -> tuple[float, int, bool, float]:
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(dimension).astype(np.float64)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0 or not np.isfinite(vector_norm):
        raise ValueError("the deterministic spectral start vector is numerically invalid.")
    vector /= vector_norm

    eigenvalue = 0.0
    residual = np.inf
    for iteration in range(1, max_iterations + 1):
        product = _normal_matvec(
            matrix,
            transpose,
            vector,
            row_space=row_space,
        )
        product_norm = float(np.linalg.norm(product))
        if product_norm == 0.0:
            if iteration != 1:
                raise ValueError("power iteration unexpectedly entered the nullspace of A2.")
            squared = matrix.copy()
            squared.data = np.square(squared.data)
            axis = 1 if row_space else 0
            diagonal = np.asarray(squared.sum(axis=axis), dtype=np.float64).reshape(-1)
            vector.fill(0.0)
            vector[int(np.argmax(diagonal))] = 1.0
            product = _normal_matvec(
                matrix,
                transpose,
                vector,
                row_space=row_space,
            )
            product_norm = float(np.linalg.norm(product))
            if product_norm == 0.0:
                raise ValueError("A2 has no usable nonzero row or column for power iteration.")
        if not np.isfinite(product_norm):
            raise ValueError("power iteration overflowed FP64.")
        vector = product / product_norm
        refreshed = _normal_matvec(
            matrix,
            transpose,
            vector,
            row_space=row_space,
        )
        next_eigenvalue = max(float(np.dot(vector, refreshed)), 0.0)
        residual = float(np.linalg.norm(refreshed - next_eigenvalue * vector))
        if not np.isfinite(next_eigenvalue) or not np.isfinite(residual):
            raise ValueError("power iteration produced a nonfinite estimate.")
        scale = max(1.0, next_eigenvalue)
        change = abs(next_eigenvalue - eigenvalue)
        eigenvalue = next_eigenvalue
        if residual <= tolerance * scale and change <= tolerance * scale:
            return eigenvalue, iteration, True, residual

    return eigenvalue, max_iterations, False, residual


def _collatz_upper_bound(
    absolute_matrix: sparse.csr_matrix,
    absolute_transpose: sparse.csr_matrix,
    *,
    dimension: int,
    row_space: bool,
    iterations: int,
    seed: int,
    fallback: float,
) -> tuple[float, int]:
    """Bound the signed normal operator through its nonnegative majorant."""

    generator = np.random.default_rng(seed ^ 0x5A17_7A9D)
    vector = generator.uniform(0.5, 1.5, size=dimension).astype(np.float64)
    vector /= float(np.max(vector))
    floor = 64.0 * np.finfo(np.float64).eps
    best = fallback
    completed = 0

    for iteration in range(1, iterations + 1):
        product = _normal_matvec(
            absolute_matrix,
            absolute_transpose,
            vector,
            row_space=row_space,
        )
        if not np.all(np.isfinite(product)) or np.any(product < 0.0):
            break
        ratios = product / vector
        candidate = float(np.max(ratios))
        if not np.isfinite(candidate) or candidate <= 0.0:
            break
        best = min(best, candidate)
        completed = iteration
        product_scale = float(np.max(product))
        if product_scale <= 0.0 or not np.isfinite(product_scale):
            break
        vector = np.maximum(product / product_scale, floor)
        vector /= float(np.max(vector))

    return best, completed


def estimate_sparse_spectral_norm_squared(
    A2: sparse.spmatrix,
    *,
    relative_safety_margin: float = 1e-10,
    power_tolerance: float = 1e-11,
    power_max_iterations: int = 64,
    power_seed: int = 20260803,
    collatz_iterations: int = 32,
) -> SparseSpectralCertificate:
    """Estimate and conservatively safeguard ``||A2||_2^2`` using sparse operations.

    ``rayleigh_estimate`` is useful for assessing tightness.  Solver code must
    use ``lambda_used``, which is based on sparse upper bounds and explicit
    floating-point inflation rather than on the underestimating power iterate.
    """

    if not sparse.issparse(A2):
        raise TypeError("A2 must be a SciPy sparse matrix; dense input is unsupported.")
    if not np.isfinite(relative_safety_margin) or relative_safety_margin <= 0.0:
        raise ValueError("relative_safety_margin must be a positive finite scalar.")
    if not np.isfinite(power_tolerance) or power_tolerance <= 0.0:
        raise ValueError("power_tolerance must be a positive finite scalar.")
    power_max_iterations = _positive_integer(
        power_max_iterations,
        name="power_max_iterations",
    )
    collatz_iterations = _positive_integer(collatz_iterations, name="collatz_iterations")
    if not isinstance(power_seed, int) or isinstance(power_seed, bool):
        raise ValueError("power_seed must be an integer.")

    matrix = sparse.csr_matrix(A2, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    rows, columns = (int(value) for value in matrix.shape)
    if rows <= 0 or columns <= 0:
        raise ValueError("A2 must have at least one row and one column.")
    if matrix.nnz <= 0:
        raise ValueError("A2 must have a positive spectral norm.")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("A2 must contain only finite values.")

    with np.errstate(over="ignore", invalid="ignore"):
        frobenius_upper = float(np.dot(matrix.data, matrix.data))
    absolute = matrix.copy()
    absolute.data = np.abs(absolute.data)
    row_sums = np.asarray(absolute.sum(axis=1), dtype=np.float64).reshape(-1)
    column_sums = np.asarray(absolute.sum(axis=0), dtype=np.float64).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        induced_upper = float(np.max(row_sums) * np.max(column_sums))
    if (
        not np.isfinite(frobenius_upper)
        or not np.isfinite(induced_upper)
        or frobenius_upper <= 0.0
        or induced_upper <= 0.0
    ):
        raise ValueError("the sparse spectral upper bounds are nonfinite or nonpositive.")

    transpose = matrix.T.tocsr()
    absolute_transpose = absolute.T.tocsr()
    row_space = rows <= columns
    dimension = rows if row_space else columns
    rayleigh, power_iterations, power_converged, residual = _power_estimate(
        matrix,
        transpose,
        dimension=dimension,
        row_space=row_space,
        tolerance=power_tolerance,
        max_iterations=power_max_iterations,
        seed=power_seed,
    )
    sparse_fallback = min(frobenius_upper, induced_upper)
    collatz_upper, completed_collatz = _collatz_upper_bound(
        absolute,
        absolute_transpose,
        dimension=dimension,
        row_space=row_space,
        iterations=collatz_iterations,
        seed=power_seed,
        fallback=sparse_fallback,
    )

    upper = float(np.nextafter(max(min(sparse_fallback, collatz_upper), rayleigh), np.inf))
    epsilon = np.finfo(np.float64).eps
    accumulation_depth = max(1, int(np.ceil(np.log2(matrix.nnz + 1))))
    roundoff_margin = 128.0 * epsilon * accumulation_depth * max(1.0, upper)
    requested_margin = relative_safety_margin * max(1.0, upper)
    lambda_used = float(np.nextafter(upper + roundoff_margin + requested_margin, np.inf))
    result = SparseSpectralCertificate(
        rows=rows,
        columns=columns,
        nonzeros=int(matrix.nnz),
        matrix_sha256=canonical_csr_sha256(matrix),
        operator_dimension=dimension,
        power_seed=power_seed,
        power_iterations=power_iterations,
        power_converged=power_converged,
        rayleigh_estimate=rayleigh,
        power_residual=residual,
        frobenius_upper_bound=frobenius_upper,
        induced_norm_upper_bound=induced_upper,
        collatz_upper_bound=collatz_upper,
        collatz_iterations=completed_collatz,
        certified_upper_bound=upper,
        relative_safety_margin=float(relative_safety_margin),
        requested_safety_margin=requested_margin,
        roundoff_safety_margin=roundoff_margin,
        lambda_used=lambda_used,
    )
    if not result.finite_certificate or result.lambda_used <= result.rayleigh_estimate:
        raise ValueError("failed to construct a finite conservative spectral safeguard.")
    return result


__all__ = [
    "SparseSpectralCertificate",
    "canonical_csr_sha256",
    "estimate_sparse_spectral_norm_squared",
]
