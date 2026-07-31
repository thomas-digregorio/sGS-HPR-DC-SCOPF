"""Resident CuPy CSR operators and truthful Stage 6 SpMV benchmarking.

The low-level kernel path is deliberately narrow: it is enabled only for the
probed CuPy 14.1.1 binding, FP64 CSR matrices, 32-bit indices, the current CUDA
stream, and a successful deterministic smoke test.  Every other configuration
uses CuPy's public default-algorithm helper and labels that fallback explicitly.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from scipy import sparse

from .gpu_backend import CUDATiming, CuPyBackend, GPUBackendUnavailable

PINNED_CUPY_VERSION = "14.1.1"
CSR_ALG2_REQUESTED_LABEL = "cuSPARSE CUSPARSE_SPMV_CSR_ALG2"
CSR_ALG2_EFFECTIVE_LABEL = (
    "cuSPARSE CUSPARSE_SPMV_CSR_ALG2 (enum 3; pinned CuPy 14.1.1 low-level binding)"
)
DEFAULT_SPMV_LABEL = "cupyx.cusparse.spmv CUSPARSE_MV_ALG_DEFAULT"


def _import_cupyx_sparse() -> Any:
    try:
        return importlib.import_module("cupyx.scipy.sparse")
    except Exception as error:
        raise GPUBackendUnavailable(
            "CuPy is present, but cupyx.scipy.sparse is unavailable."
        ) from error


def _import_cupyx_cusparse() -> Any:
    try:
        return importlib.import_module("cupyx.cusparse")
    except Exception as error:
        raise GPUBackendUnavailable(
            "CuPy is present, but cupyx.cusparse is unavailable."
        ) from error


@dataclass(frozen=True, slots=True)
class SparseKernelSelection:
    """The requested and actually selected sparse-kernel policy."""

    requested_label: str
    effective_label: str
    uses_csr_alg2: bool
    fallback_reason: str | None
    probe_max_abs_error: float | None = None
    probe_repeat_bitwise_equal: bool | None = None

    def as_dict(self) -> dict[str, str | bool | float | None]:
        return {
            "requested_label": self.requested_label,
            "effective_label": self.effective_label,
            "uses_csr_alg2": self.uses_csr_alg2,
            "fallback_reason": self.fallback_reason,
            "probe_max_abs_error": self.probe_max_abs_error,
            "probe_repeat_bitwise_equal": self.probe_repeat_bitwise_equal,
        }


@dataclass(frozen=True, slots=True)
class SparseMatvecBenchmark:
    """CUDA-event timings for normal, transpose-flag, and explicit-transpose CSR."""

    normal_csr: CUDATiming
    transpose_flag: CUDATiming
    explicit_csr_transpose: CUDATiming
    transpose_max_abs_difference: float
    high_level_kernel_label: str = DEFAULT_SPMV_LABEL

    def as_dict(self) -> dict[str, Any]:
        return {
            "normal_csr": self.normal_csr.as_dict(),
            "transpose_flag": self.transpose_flag.as_dict(),
            "explicit_csr_transpose": self.explicit_csr_transpose.as_dict(),
            "transpose_max_abs_difference": self.transpose_max_abs_difference,
            "high_level_kernel_label": self.high_level_kernel_label,
        }


class _PinnedCuPy141CSRAlg2:
    """Persistent-descriptor FP64 CSR_ALG2 wrapper for one non-transposed CSR."""

    def __init__(self, backend: CuPyBackend, matrix: Any) -> None:
        cp = backend.cp
        if str(cp.__version__) != PINNED_CUPY_VERSION:
            raise RuntimeError(
                f"low-level CSR_ALG2 requires CuPy {PINNED_CUPY_VERSION}; found {cp.__version__}."
            )
        if str(matrix.dtype) != "float64":
            raise TypeError(f"low-level CSR_ALG2 requires FP64 data; received {matrix.dtype}.")
        if str(matrix.indices.dtype) != "int32" or str(matrix.indptr.dtype) != "int32":
            raise TypeError("low-level CSR_ALG2 requires int32 CSR indices and indptr.")
        if not matrix.has_canonical_format:
            raise ValueError("low-level CSR_ALG2 requires canonical CSR format.")

        cupyx_cusparse = _import_cupyx_cusparse()
        try:
            low_level = importlib.import_module("cupy_backends.cuda.libs.cusparse")
        except Exception as error:
            raise RuntimeError(
                "the pinned CuPy low-level cuSPARSE binding is unavailable."
            ) from error
        if int(getattr(low_level, "CUSPARSE_CSRMV_ALG2", -1)) != 3:
            raise RuntimeError("the pinned CSR_ALG2 enum guard did not equal the probed value 3.")
        if not hasattr(low_level, "dnVecSetValues"):
            raise RuntimeError(
                "the pinned binding lacks dnVecSetValues for persistent descriptors."
            )

        self.backend = backend
        self.matrix = matrix
        self._cusparse = low_level
        self._matrix_descriptor = cupyx_cusparse.SpMatDescriptor.create(matrix)
        self._x_placeholder = cp.empty(matrix.shape[1], dtype=cp.float64)
        self._y_placeholder = cp.empty(matrix.shape[0], dtype=cp.float64)
        self._x_descriptor = cupyx_cusparse.DnVecDescriptor.create(self._x_placeholder)
        self._y_descriptor = cupyx_cusparse.DnVecDescriptor.create(self._y_placeholder)
        self._alpha = np.array(1.0, dtype=np.float64)
        self._beta = np.array(0.0, dtype=np.float64)
        self._handle = cp.cuda.device.get_cusparse_handle()
        self._operation = low_level.CUSPARSE_OPERATION_NON_TRANSPOSE
        self._compute_type = cp.cuda.runtime.CUDA_R_64F
        self._algorithm = low_level.CUSPARSE_CSRMV_ALG2
        self._stream_pointer = int(cp.cuda.get_current_stream().ptr)
        buffer_bytes = int(
            low_level.spMV_bufferSize(
                self._handle,
                self._operation,
                int(self._alpha.ctypes.data),
                self._matrix_descriptor.desc,
                self._x_descriptor.desc,
                int(self._beta.ctypes.data),
                self._y_descriptor.desc,
                self._compute_type,
                self._algorithm,
            )
        )
        self.buffer_bytes = buffer_bytes
        self._buffer = cp.empty(max(buffer_bytes, 1), dtype=cp.int8)
        self._lock = Lock()

    def __call__(self, vector: Any, *, out: Any | None = None) -> Any:
        cp = self.backend.cp
        if not isinstance(vector, cp.ndarray):
            raise TypeError("CSR_ALG2 input must already be a CuPy array.")
        if vector.ndim != 1 or vector.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                "CSR_ALG2 input dimension mismatch: "
                f"expected ({self.matrix.shape[1]},), received {vector.shape}."
            )
        if str(vector.dtype) != "float64" or not vector.flags.c_contiguous:
            raise TypeError("CSR_ALG2 input must be contiguous FP64.")
        if out is None:
            out = cp.empty(self.matrix.shape[0], dtype=cp.float64)
        if not isinstance(out, cp.ndarray):
            raise TypeError("CSR_ALG2 output must already be a CuPy array.")
        if out.shape != (self.matrix.shape[0],):
            raise ValueError(
                f"CSR_ALG2 output must have shape ({self.matrix.shape[0]},); received {out.shape}."
            )
        if str(out.dtype) != "float64" or not out.flags.c_contiguous:
            raise TypeError("CSR_ALG2 output must be contiguous FP64.")
        if int(cp.cuda.get_current_stream().ptr) != self._stream_pointer:
            raise RuntimeError(
                "the persistent CSR_ALG2 descriptors are bound to their construction stream."
            )

        with self._lock:
            self._cusparse.dnVecSetValues(self._x_descriptor.desc, int(vector.data.ptr))
            self._cusparse.dnVecSetValues(self._y_descriptor.desc, int(out.data.ptr))
            self._cusparse.spMV(
                self._handle,
                self._operation,
                int(self._alpha.ctypes.data),
                self._matrix_descriptor.desc,
                self._x_descriptor.desc,
                int(self._beta.ctypes.data),
                self._y_descriptor.desc,
                self._compute_type,
                self._algorithm,
                int(self._buffer.data.ptr),
            )
        return out


@dataclass(slots=True)
class ResidentCSR:
    """One FP64 matrix and its explicit CSR transpose resident on the GPU."""

    backend: CuPyBackend
    matrix: Any
    transpose: Any
    kernel: SparseKernelSelection
    _normal_alg2: _PinnedCuPy141CSRAlg2 | None = None
    _transpose_alg2: _PinnedCuPy141CSRAlg2 | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.matrix.shape[0]), int(self.matrix.shape[1]))

    @property
    def nnz(self) -> int:
        return int(self.matrix.nnz)

    @property
    def index_dtype(self) -> str:
        return str(self.matrix.indices.dtype)

    @property
    def kernel_label(self) -> str:
        return self.kernel.effective_label

    def matvec(self, vector: Any, *, transpose: bool = False, out: Any | None = None) -> Any:
        """Apply resident ``A`` or explicit resident CSR ``A.T`` to a device vector."""

        operator = self.transpose if transpose else self.matrix
        low_level = self._transpose_alg2 if transpose else self._normal_alg2
        if low_level is not None:
            return low_level(vector, out=out)
        return _public_default_spmv(operator, vector, out=out, transa=False)

    def benchmark_matvec(
        self,
        normal_vector: Any,
        transpose_vector: Any,
        *,
        warmup_calls: int = 3,
        repetitions: int = 20,
    ) -> SparseMatvecBenchmark:
        """Benchmark public/default SpMV variants without claiming CSR_ALG2."""

        return benchmark_resident_csr(
            self,
            normal_vector,
            transpose_vector,
            warmup_calls=warmup_calls,
            repetitions=repetitions,
        )


def _public_default_spmv(
    matrix: Any,
    vector: Any,
    *,
    out: Any | None,
    transa: bool,
) -> Any:
    cupyx_cusparse = _import_cupyx_cusparse()
    return cupyx_cusparse.spmv(
        matrix,
        vector,
        y=out,
        alpha=1.0,
        beta=0.0,
        transa=transa,
    )


def _validated_host_csr(matrix: Any, *, dtype: np.dtype[Any]) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=dtype, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    if result.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional; received shape {result.shape}.")
    if not np.all(np.isfinite(result.data)):
        raise ValueError("matrix must contain only finite values.")
    int32_max = np.iinfo(np.int32).max
    if max(result.shape, default=0) > int32_max or result.nnz > int32_max:
        raise OverflowError(
            "the verified Stage 6 path requires matrix dimensions and nnz in int32."
        )
    if result.indices.size and int(result.indices.max()) > int32_max:
        raise OverflowError("CSR column indices exceed the verified int32 range.")
    return sparse.csr_matrix(
        (
            np.asarray(result.data, dtype=dtype),
            np.asarray(result.indices, dtype=np.int32),
            np.asarray(result.indptr, dtype=np.int32),
        ),
        shape=result.shape,
    )


def _fallback(reason: str) -> SparseKernelSelection:
    return SparseKernelSelection(
        requested_label=CSR_ALG2_REQUESTED_LABEL,
        effective_label=DEFAULT_SPMV_LABEL,
        uses_csr_alg2=False,
        fallback_reason=reason,
    )


def _try_alg2(
    backend: CuPyBackend,
    matrix: Any,
    transpose: Any,
) -> tuple[SparseKernelSelection, _PinnedCuPy141CSRAlg2 | None, _PinnedCuPy141CSRAlg2 | None]:
    cp = backend.cp
    if str(matrix.dtype) != "float64":
        return (
            _fallback(
                "the pinned low-level CSR_ALG2 wrapper is verified only for FP64; "
                f"{matrix.dtype} uses the public default-algorithm diagnostic path."
            ),
            None,
            None,
        )
    if str(cp.__version__) != PINNED_CUPY_VERSION:
        return (
            _fallback(
                f"low-level CSR_ALG2 is verified only for CuPy {PINNED_CUPY_VERSION}; "
                f"found {cp.__version__}."
            ),
            None,
            None,
        )
    if matrix.nnz == 0 or min(matrix.shape, default=0) == 0:
        return (
            _fallback("the low-level CSR_ALG2 smoke test requires a non-empty matrix."),
            None,
            None,
        )

    try:
        normal_alg2 = _PinnedCuPy141CSRAlg2(backend, matrix)
        transpose_alg2 = _PinnedCuPy141CSRAlg2(backend, transpose)
        normal_input = cp.ones(matrix.shape[1], dtype=cp.float64)
        transpose_input = cp.ones(matrix.shape[0], dtype=cp.float64)
        normal_first = normal_alg2(normal_input)
        normal_repeat = normal_alg2(normal_input)
        transpose_first = transpose_alg2(transpose_input)
        transpose_repeat = transpose_alg2(transpose_input)
        normal_default = _public_default_spmv(matrix, normal_input, out=None, transa=False)
        transpose_default = _public_default_spmv(matrix, transpose_input, out=None, transa=True)
        maximum_error = cp.maximum(
            cp.max(cp.abs(normal_first - normal_default)),
            cp.max(cp.abs(transpose_first - transpose_default)),
        )
        repeat_equal = cp.logical_and(
            cp.array_equal(normal_first, normal_repeat),
            cp.array_equal(transpose_first, transpose_repeat),
        )
        max_abs_error = backend.scalar_to_host(maximum_error, phase="sparse_kernel_probe")
        bitwise_equal = bool(backend.scalar_to_host(repeat_equal, phase="sparse_kernel_probe"))
        tolerance = 64.0 * np.finfo(np.float64).eps
        scale = max(
            1.0,
            backend.scalar_to_host(
                cp.maximum(cp.max(cp.abs(normal_default)), cp.max(cp.abs(transpose_default))),
                phase="sparse_kernel_probe",
            ),
        )
        if not np.isfinite(max_abs_error) or max_abs_error > tolerance * scale:
            return (
                _fallback(
                    "CSR_ALG2 smoke-test disagreement exceeded the FP64 tolerance: "
                    f"{max_abs_error:.6e} > {(tolerance * scale):.6e}."
                ),
                None,
                None,
            )
        if not bitwise_equal:
            return (
                _fallback("CSR_ALG2 repeated results were not bitwise deterministic."),
                None,
                None,
            )
        return (
            SparseKernelSelection(
                requested_label=CSR_ALG2_REQUESTED_LABEL,
                effective_label=CSR_ALG2_EFFECTIVE_LABEL,
                uses_csr_alg2=True,
                fallback_reason=None,
                probe_max_abs_error=max_abs_error,
                probe_repeat_bitwise_equal=True,
            ),
            normal_alg2,
            transpose_alg2,
        )
    except Exception as error:
        return (
            _fallback(
                f"CSR_ALG2 initialization or smoke test failed: {type(error).__name__}: {error}"
            ),
            None,
            None,
        )


def prepare_resident_csr(
    backend: CuPyBackend,
    matrix: Any,
    *,
    phase: str = "matrix_setup",
    prefer_csr_alg2: bool = True,
    dtype: Any | None = None,
) -> ResidentCSR:
    """Transfer CSR arrays once and construct resident ``A`` and ``A.T``.

    FP64 is the verified solver path. FP32 is accepted for a labeled numerical
    diagnostic and intentionally falls back to CuPy's public default algorithm.
    """

    cp = backend.cp
    device_dtype = cp.dtype(cp.float64 if dtype is None else dtype)
    if str(device_dtype) not in {"float32", "float64"}:
        raise TypeError(
            f"resident CSR supports only FP32/FP64 diagnostics; received {device_dtype}."
        )
    host = _validated_host_csr(matrix, dtype=np.dtype(str(device_dtype)))
    cupyx_sparse = _import_cupyx_sparse()
    data = backend.to_device(
        host.data,
        phase=phase,
        dtype=device_dtype,
        kind="sparse_data",
    )
    indices = backend.to_device(
        host.indices,
        phase=phase,
        dtype=cp.int32,
        kind="sparse_indices",
    )
    indptr = backend.to_device(
        host.indptr,
        phase=phase,
        dtype=cp.int32,
        kind="sparse_indptr",
    )
    resident = cupyx_sparse.csr_matrix((data, indices, indptr), shape=host.shape)
    resident.sum_duplicates()
    resident.sort_indices()
    resident_transpose = resident.T.tocsr()
    resident_transpose.sum_duplicates()
    resident_transpose.sort_indices()

    if prefer_csr_alg2:
        kernel, normal_alg2, transpose_alg2 = _try_alg2(backend, resident, resident_transpose)
    else:
        kernel = _fallback("CSR_ALG2 was disabled by the caller.")
        normal_alg2 = None
        transpose_alg2 = None
    return ResidentCSR(
        backend=backend,
        matrix=resident,
        transpose=resident_transpose,
        kernel=kernel,
        _normal_alg2=normal_alg2,
        _transpose_alg2=transpose_alg2,
    )


def benchmark_resident_csr(
    resident: ResidentCSR,
    normal_vector: Any,
    transpose_vector: Any,
    *,
    warmup_calls: int = 3,
    repetitions: int = 20,
) -> SparseMatvecBenchmark:
    """Benchmark CuPy's public default path for ``A``, ``A.T``, and explicit CSR ``A.T``."""

    cp = resident.backend.cp
    if normal_vector.shape != (resident.shape[1],):
        raise ValueError(
            f"normal_vector must have shape ({resident.shape[1]},); received {normal_vector.shape}."
        )
    if transpose_vector.shape != (resident.shape[0],):
        raise ValueError(
            "transpose_vector must have shape "
            f"({resident.shape[0]},); received {transpose_vector.shape}."
        )
    normal_out = cp.empty(resident.shape[0], dtype=resident.matrix.dtype)
    transpose_flag_out = cp.empty(resident.shape[1], dtype=resident.matrix.dtype)
    explicit_transpose_out = cp.empty(resident.shape[1], dtype=resident.matrix.dtype)

    def normal_call() -> Any:
        return _public_default_spmv(resident.matrix, normal_vector, out=normal_out, transa=False)

    def transpose_flag_call() -> Any:
        return _public_default_spmv(
            resident.matrix, transpose_vector, out=transpose_flag_out, transa=True
        )

    def explicit_transpose_call() -> Any:
        return _public_default_spmv(
            resident.transpose,
            transpose_vector,
            out=explicit_transpose_out,
            transa=False,
        )

    _, normal_timing = resident.backend.timed_call(
        normal_call, warmup_calls=warmup_calls, repetitions=repetitions
    )
    _, transpose_flag_timing = resident.backend.timed_call(
        transpose_flag_call, warmup_calls=warmup_calls, repetitions=repetitions
    )
    _, explicit_transpose_timing = resident.backend.timed_call(
        explicit_transpose_call, warmup_calls=warmup_calls, repetitions=repetitions
    )
    maximum_difference = resident.backend.scalar_to_host(
        cp.max(cp.abs(transpose_flag_out - explicit_transpose_out)),
        phase="sparse_benchmark_validation",
    )
    return SparseMatvecBenchmark(
        normal_csr=normal_timing,
        transpose_flag=transpose_flag_timing,
        explicit_csr_transpose=explicit_transpose_timing,
        transpose_max_abs_difference=maximum_difference,
    )


__all__ = [
    "CSR_ALG2_EFFECTIVE_LABEL",
    "CSR_ALG2_REQUESTED_LABEL",
    "DEFAULT_SPMV_LABEL",
    "PINNED_CUPY_VERSION",
    "ResidentCSR",
    "SparseKernelSelection",
    "SparseMatvecBenchmark",
    "benchmark_resident_csr",
    "prepare_resident_csr",
]
