from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import sparse

import gpu_dcopf_hpr.gpu_sparse as gpu_sparse
from gpu_dcopf_hpr.gpu_backend import CUDATiming, TransferLedger
from gpu_dcopf_hpr.gpu_sparse import DEFAULT_SPMV_LABEL, prepare_resident_csr


class _FakeBackend:
    cp = np

    def __init__(self) -> None:
        self.ledger = TransferLedger()

    def to_device(
        self,
        values: Any,
        *,
        phase: str,
        dtype: Any,
        kind: Any,
    ) -> np.ndarray:
        result = np.asarray(values, dtype=dtype)
        self.ledger.record(
            phase=phase,
            direction="host_to_device",
            kind=kind,
            bytes=result.nbytes,
        )
        return result

    @staticmethod
    def scalar_to_host(value: Any, *, phase: str) -> float:
        del phase
        return float(np.asarray(value).reshape(-1)[0])

    @staticmethod
    def timed_call(
        operation: Any, *, warmup_calls: int, repetitions: int
    ) -> tuple[Any, CUDATiming]:
        for _ in range(warmup_calls):
            operation()
        result = None
        for _ in range(repetitions):
            result = operation()
        return result, CUDATiming(
            elapsed_seconds=0.001 * repetitions,
            repetitions=repetitions,
            warmup_calls=warmup_calls,
        )


class _FakePublicCuSparse:
    @staticmethod
    def spmv(
        matrix: sparse.spmatrix,
        vector: np.ndarray,
        y: np.ndarray | None,
        alpha: float,
        beta: float,
        transa: bool,
    ) -> np.ndarray:
        operator = matrix.T if transa else matrix
        result = alpha * np.asarray(operator @ vector).reshape(-1)
        if y is None:
            return result
        y[...] = result + beta * y
        return y


@pytest.fixture
def fake_sparse_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_sparse, "_import_cupyx_sparse", lambda: sparse)
    monkeypatch.setattr(gpu_sparse, "_import_cupyx_cusparse", lambda: _FakePublicCuSparse)


def _matrix() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        np.array(
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, 4.0],
            ],
            dtype=np.float64,
        )
    )


def test_prepare_resident_csr_records_three_transfers_and_explicit_transpose(
    fake_sparse_modules: None,
) -> None:
    del fake_sparse_modules
    backend = _FakeBackend()
    resident = prepare_resident_csr(
        backend, _matrix(), phase="operator_setup", prefer_csr_alg2=False
    )

    assert resident.shape == (2, 3)
    assert resident.nnz == 4
    assert resident.index_dtype == "int32"
    assert sparse.isspmatrix_csr(resident.matrix)
    assert sparse.isspmatrix_csr(resident.transpose)
    np.testing.assert_array_equal(resident.transpose.toarray(), _matrix().toarray().T)
    assert resident.kernel_label == DEFAULT_SPMV_LABEL
    assert resident.kernel.fallback_reason == "CSR_ALG2 was disabled by the caller."
    assert [(record.kind, record.calls) for record in backend.ledger.records] == [
        ("sparse_data", 1),
        ("sparse_indices", 1),
        ("sparse_indptr", 1),
    ]
    assert backend.ledger.total_bytes() == 32 + 16 + 12


def test_resident_matvec_uses_normal_and_explicit_transpose_without_host_roundtrip(
    fake_sparse_modules: None,
) -> None:
    del fake_sparse_modules
    backend = _FakeBackend()
    resident = prepare_resident_csr(backend, _matrix(), prefer_csr_alg2=False)
    transfers_before = backend.ledger.summary()

    normal = resident.matvec(np.array([1.0, 2.0, 3.0]))
    transposed = resident.matvec(np.array([5.0, 6.0]), transpose=True)

    np.testing.assert_allclose(normal, [-1.0, 18.0])
    np.testing.assert_allclose(transposed, [10.0, 18.0, 19.0])
    assert backend.ledger.summary() == transfers_before


def test_fp32_is_a_labeled_default_algorithm_diagnostic(
    fake_sparse_modules: None,
) -> None:
    del fake_sparse_modules
    resident = prepare_resident_csr(_FakeBackend(), _matrix(), dtype=np.float32)

    assert resident.matrix.dtype == np.float32
    assert not resident.kernel.uses_csr_alg2
    assert resident.kernel_label == DEFAULT_SPMV_LABEL
    assert "verified only for FP64" in (resident.kernel.fallback_reason or "")


def test_public_sparse_benchmark_compares_transpose_flag_with_explicit_csr(
    fake_sparse_modules: None,
) -> None:
    del fake_sparse_modules
    resident = prepare_resident_csr(_FakeBackend(), _matrix(), prefer_csr_alg2=False)
    result = resident.benchmark_matvec(
        np.array([1.0, 2.0, 3.0]),
        np.array([5.0, 6.0]),
        warmup_calls=1,
        repetitions=4,
    )

    assert result.high_level_kernel_label == DEFAULT_SPMV_LABEL
    assert result.normal_csr.repetitions == 4
    assert result.transpose_flag.repetitions == 4
    assert result.explicit_csr_transpose.repetitions == 4
    assert result.transpose_max_abs_difference == 0.0


def test_rejects_nonfinite_matrix_and_unsupported_dtype(
    fake_sparse_modules: None,
) -> None:
    del fake_sparse_modules
    matrix = _matrix()
    matrix.data[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        prepare_resident_csr(_FakeBackend(), matrix, prefer_csr_alg2=False)
    with pytest.raises(TypeError, match="only FP32/FP64"):
        prepare_resident_csr(_FakeBackend(), _matrix(), dtype=np.int64, prefer_csr_alg2=False)


def test_version_drift_cannot_claim_csr_alg2() -> None:
    backend = SimpleNamespace(cp=SimpleNamespace(__version__="14.2.0"))
    selection, normal, transpose = gpu_sparse._try_alg2(
        backend, SimpleNamespace(dtype=np.dtype("float64")), object()
    )

    assert not selection.uses_csr_alg2
    assert selection.effective_label == DEFAULT_SPMV_LABEL
    assert "verified only for CuPy 14.1.1" in (selection.fallback_reason or "")
    assert normal is None
    assert transpose is None
