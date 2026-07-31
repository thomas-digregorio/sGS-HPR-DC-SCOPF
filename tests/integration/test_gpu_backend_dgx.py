"""Optional runtime checks executed in the pinned Stage 6 DGX environment."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from gpu_dcopf_hpr.gpu_backend import (
    GPUBackendUnavailable,
    create_gpu_backend,
    cupy_available,
)
from gpu_dcopf_hpr.gpu_sparse import prepare_resident_csr


def test_pinned_gpu_backend_uses_verified_alg2_and_matches_cpu() -> None:
    if not cupy_available():
        pytest.skip("CuPy/CUDA is not available")
    try:
        backend = create_gpu_backend()
    except GPUBackendUnavailable as error:
        if "requires CuPy 14.1.1" in str(error):
            pytest.skip(str(error))
        raise
    host_matrix = sparse.csr_matrix(
        np.array(
            [
                [2.0, 0.0, -1.0, 0.5],
                [0.0, 3.0, 4.0, 0.0],
                [-2.0, 1.0, 0.0, 5.0],
            ],
            dtype=np.float64,
        )
    )
    resident = prepare_resident_csr(backend, host_matrix)
    x_host = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y_host = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    x = backend.to_device(x_host, phase="test_vectors")
    y = backend.to_device(y_host, phase="test_vectors")

    normal = backend.to_host(resident.matvec(x), phase="test_results")
    transpose = backend.to_host(resident.matvec(y, transpose=True), phase="test_results")
    benchmark = resident.benchmark_matvec(x, y, warmup_calls=1, repetitions=3)
    diagnostics = backend.diagnostics
    memory = backend.memory_report()

    np.testing.assert_allclose(normal, host_matrix @ x_host, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(transpose, host_matrix.T @ y_host, rtol=1e-14, atol=1e-14)
    assert diagnostics.cupy_version == "14.1.1"
    assert diagnostics.fp64_supported
    assert diagnostics.fp64_itemsize_bytes == 8
    assert diagnostics.csr_index_bits == 32
    assert resident.kernel.uses_csr_alg2
    assert resident.kernel.probe_repeat_bitwise_equal
    assert "CUSPARSE_SPMV_CSR_ALG2" in resident.kernel_label
    assert benchmark.transpose_max_abs_difference <= 1e-14
    assert benchmark.normal_csr.repetitions == 3
    assert memory.total_device_bytes > 0
    assert backend.ledger.total_bytes("host_to_device") > 0
    assert backend.ledger.total_calls("device_to_host") >= 2
