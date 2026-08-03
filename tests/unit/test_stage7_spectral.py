from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from gpu_dcopf_hpr.stage7_spectral import (
    canonical_csr_sha256,
    estimate_sparse_spectral_norm_squared,
)


@pytest.mark.parametrize(
    "dense",
    (
        np.diag([3.0, 2.0, 0.5]),
        np.array(
            [
                [1.0, -4.0, 0.0, 2.0],
                [0.0, 3.0, -1.0, 0.0],
                [2.0, 0.0, 5.0, -2.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [1.0, -1.0],
                [-1.0, 1.0],
                [2.0, -2.0],
                [-3.0, 3.0],
            ],
            dtype=np.float64,
        ),
    ),
    ids=("diagonal", "wide_signed", "tall_cancellation"),
)
def test_sparse_certificate_cross_checks_small_matrix_exact_svd(dense: np.ndarray) -> None:
    exact = float(np.linalg.norm(dense, ord=2) ** 2)
    result = estimate_sparse_spectral_norm_squared(sparse.csr_matrix(dense))

    assert result.finite_certificate
    assert result.rayleigh_estimate == pytest.approx(exact, rel=2e-10, abs=2e-12)
    assert result.frobenius_upper_bound + 5e-13 >= exact
    assert result.induced_norm_upper_bound + 5e-13 >= exact
    assert result.collatz_upper_bound + 5e-12 >= exact
    assert result.lambda_used > exact
    assert result.inequality_lambda == result.lambda_used
    assert result.relative_safety_margin == 1e-10
    assert result.requested_safety_margin > 0.0
    assert result.certificate_gap > 0.0
    assert not result.summary()["dense_matrix_materialized"]
    assert not result.summary()["normal_matrix_materialized"]
    assert result.summary()["inequality_lambda"] == result.lambda_used
    assert result.summary()["matrix_sha256"] == result.matrix_sha256


def test_canonical_fingerprint_is_independent_of_csr_index_width() -> None:
    int32_matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 1.25, -2.5, 0.0],
                [3.75, 0.0, 0.0, -4.5],
                [0.0, 0.0, 5.25, 6.5],
            ],
            dtype=np.float64,
        )
    )
    int32_matrix.indices = int32_matrix.indices.astype(np.int32)
    int32_matrix.indptr = int32_matrix.indptr.astype(np.int32)
    int64_matrix = int32_matrix.copy()
    int64_matrix.indices = int64_matrix.indices.astype(np.int64)
    int64_matrix.indptr = int64_matrix.indptr.astype(np.int64)

    assert int32_matrix.indices.dtype == np.int32
    assert int64_matrix.indices.dtype == np.int64
    assert canonical_csr_sha256(int32_matrix) == canonical_csr_sha256(int64_matrix)


def test_sparse_certificate_is_bitwise_deterministic_for_fixed_seed() -> None:
    matrix = sparse.random(
        31,
        17,
        density=0.18,
        random_state=np.random.default_rng(55),
        data_rvs=np.random.default_rng(91).standard_normal,
        format="csr",
    )

    first = estimate_sparse_spectral_norm_squared(matrix, power_seed=123456)
    second = estimate_sparse_spectral_norm_squared(matrix, power_seed=123456)

    assert first == second
    exact = float(np.linalg.norm(matrix.toarray(), ord=2) ** 2)
    assert first.lambda_used > exact


def test_estimator_never_calls_sparse_toarray(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix = sparse.diags([1.0, 4.0, 2.0], format="csr")

    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("dense conversion is forbidden")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    result = estimate_sparse_spectral_norm_squared(matrix)

    assert result.lambda_used > 16.0


@pytest.mark.parametrize(
    ("matrix", "exception", "message"),
    (
        (np.eye(2), TypeError, "sparse"),
        (sparse.csr_matrix((2, 3)), ValueError, "positive spectral norm"),
        (sparse.csr_matrix([[1.0, np.inf]]), ValueError, "finite"),
        (sparse.csr_matrix((0, 3)), ValueError, "at least one row"),
    ),
)
def test_invalid_spectral_inputs_are_rejected(
    matrix: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        estimate_sparse_spectral_norm_squared(matrix)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "options",
    (
        {"relative_safety_margin": 0.0},
        {"power_tolerance": np.nan},
        {"power_max_iterations": True},
        {"power_seed": 1.5},
        {"collatz_iterations": 0},
    ),
)
def test_invalid_spectral_options_are_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        estimate_sparse_spectral_norm_squared(
            sparse.eye(2, format="csr"),
            **options,  # type: ignore[arg-type]
        )
