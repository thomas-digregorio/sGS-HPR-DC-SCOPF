"""Optional CuPy backend utilities for the Stage 6 GPU implementation.

Importing this module never imports CuPy.  The optional dependency is loaded only
when :func:`create_gpu_backend` (or :func:`cupy_available`) is called, preserving
the CPU-only package and test paths established in Stages 1--5.
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, TypeVar

import numpy as np

TransferDirection = Literal["host_to_device", "device_to_host"]
TransferKind = Literal[
    "scalar",
    "vector",
    "matrix",
    "sparse_data",
    "sparse_indices",
    "sparse_indptr",
]

_Result = TypeVar("_Result")


class GPUBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested GPU backend cannot be used."""


@dataclass(frozen=True, slots=True)
class TransferRecord:
    """Aggregated host/device transfers for one phase, direction, and payload kind."""

    phase: str
    direction: TransferDirection
    kind: TransferKind
    calls: int
    bytes: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "phase": self.phase,
            "direction": self.direction,
            "kind": self.kind,
            "calls": self.calls,
            "bytes": self.bytes,
        }


class TransferLedger:
    """Record explicit CPU/GPU transfers without counting device-only operations."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, TransferDirection, TransferKind], list[int]] = defaultdict(
            lambda: [0, 0]
        )

    def record(
        self,
        *,
        phase: str,
        direction: TransferDirection,
        kind: TransferKind,
        bytes: int,
        calls: int = 1,
    ) -> None:
        """Add one explicit transfer or an already aggregated group of transfers."""

        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("phase must be a non-empty string.")
        if direction not in {"host_to_device", "device_to_host"}:
            raise ValueError(f"unsupported transfer direction: {direction!r}.")
        if kind not in {
            "scalar",
            "vector",
            "matrix",
            "sparse_data",
            "sparse_indices",
            "sparse_indptr",
        }:
            raise ValueError(f"unsupported transfer kind: {kind!r}.")
        if not isinstance(bytes, int) or bytes < 0:
            raise ValueError("bytes must be a non-negative integer.")
        if not isinstance(calls, int) or calls <= 0:
            raise ValueError("calls must be a positive integer.")

        entry = self._entries[(phase.strip(), direction, kind)]
        entry[0] += calls
        entry[1] += bytes

    @property
    def records(self) -> tuple[TransferRecord, ...]:
        """Return deterministic, immutable aggregates suitable for evidence files."""

        return tuple(
            TransferRecord(
                phase=phase,
                direction=direction,
                kind=kind,
                calls=values[0],
                bytes=values[1],
            )
            for (phase, direction, kind), values in sorted(self._entries.items())
        )

    def total_bytes(self, direction: TransferDirection | None = None) -> int:
        return sum(
            record.bytes
            for record in self.records
            if direction is None or record.direction == direction
        )

    def total_calls(self, direction: TransferDirection | None = None) -> int:
        return sum(
            record.calls
            for record in self.records
            if direction is None or record.direction == direction
        )

    def summary(self) -> dict[str, Any]:
        """Return both individual records and direction totals."""

        return {
            "records": [record.as_dict() for record in self.records],
            "totals": {
                "host_to_device": {
                    "calls": self.total_calls("host_to_device"),
                    "bytes": self.total_bytes("host_to_device"),
                },
                "device_to_host": {
                    "calls": self.total_calls("device_to_host"),
                    "bytes": self.total_bytes("device_to_host"),
                },
            },
        }


@dataclass(frozen=True, slots=True)
class DeviceDiagnostics:
    """FP64, sparse-index, CUDA, and physical-device facts for one backend."""

    cupy_version: str
    cuda_runtime_version: int
    cuda_driver_version: int
    device_id: int
    device_name: str
    compute_capability: tuple[int, int]
    integrated: bool
    multiprocessor_count: int
    total_global_memory_bytes: int
    fp64_itemsize_bytes: int
    fp64_supported: bool
    csr_index_dtype: str
    csr_indptr_dtype: str
    csr_index_bits: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "cupy_version": self.cupy_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "compute_capability": list(self.compute_capability),
            "integrated": self.integrated,
            "multiprocessor_count": self.multiprocessor_count,
            "total_global_memory_bytes": self.total_global_memory_bytes,
            "fp64_itemsize_bytes": self.fp64_itemsize_bytes,
            "fp64_supported": self.fp64_supported,
            "csr_index_dtype": self.csr_index_dtype,
            "csr_indptr_dtype": self.csr_indptr_dtype,
            "csr_index_bits": self.csr_index_bits,
        }


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """CUDA allocator and device-memory counters sampled after synchronization."""

    free_device_bytes: int
    total_device_bytes: int
    device_pool_used_bytes: int
    device_pool_total_bytes: int
    device_pool_free_blocks: int
    pinned_pool_free_blocks: int

    @property
    def runtime_used_bytes(self) -> int:
        return self.total_device_bytes - self.free_device_bytes

    def as_dict(self) -> dict[str, int]:
        return {
            "free_device_bytes": self.free_device_bytes,
            "total_device_bytes": self.total_device_bytes,
            "runtime_used_bytes": self.runtime_used_bytes,
            "device_pool_used_bytes": self.device_pool_used_bytes,
            "device_pool_total_bytes": self.device_pool_total_bytes,
            "device_pool_free_blocks": self.device_pool_free_blocks,
            "pinned_pool_free_blocks": self.pinned_pool_free_blocks,
        }


@dataclass(frozen=True, slots=True)
class CUDATiming:
    """A synchronized CUDA-event interval containing repeated calls."""

    elapsed_seconds: float
    repetitions: int
    warmup_calls: int

    @property
    def elapsed_milliseconds(self) -> float:
        return 1_000.0 * self.elapsed_seconds

    @property
    def mean_seconds(self) -> float:
        return self.elapsed_seconds / self.repetitions

    def as_dict(self) -> dict[str, float | int]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "elapsed_milliseconds": self.elapsed_milliseconds,
            "mean_seconds": self.mean_seconds,
            "repetitions": self.repetitions,
            "warmup_calls": self.warmup_calls,
        }


def _import_cupy() -> Any:
    try:
        return importlib.import_module("cupy")
    except Exception as error:  # CuPy can fail after import begins if CUDA is unusable.
        raise GPUBackendUnavailable(
            "The CuPy GPU backend is unavailable. Install the Stage 6 pinned CuPy "
            "environment and verify that a CUDA device is visible."
        ) from error


def cupy_available() -> bool:
    """Return whether CuPy imports and reports at least one CUDA device."""

    try:
        cp = _import_cupy()
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _payload_kind(array: Any) -> TransferKind:
    dimensions = int(array.ndim) if hasattr(array, "ndim") else int(np.asarray(array).ndim)
    if dimensions == 0:
        return "scalar"
    if dimensions == 1:
        return "vector"
    return "matrix"


class CuPyBackend:
    """Thin, auditable owner of CuPy, CUDA timing, and explicit transfers."""

    def __init__(
        self,
        cupy_module: Any,
        *,
        device_id: int = 0,
        ledger: TransferLedger | None = None,
    ) -> None:
        self.cp = cupy_module
        self.device_id = int(device_id)
        self.ledger = ledger if ledger is not None else TransferLedger()
        self._diagnostics: DeviceDiagnostics | None = None
        self._transfer_elapsed_seconds: dict[TransferDirection, float] = {
            "host_to_device": 0.0,
            "device_to_host": 0.0,
        }

    @property
    def xp(self) -> Any:
        """Array namespace alias used by the GPU solver without importing CuPy."""

        return self.cp

    @property
    def diagnostics(self) -> DeviceDiagnostics:
        if self._diagnostics is None:
            self._diagnostics = self._collect_diagnostics()
        return self._diagnostics

    def _collect_diagnostics(self) -> DeviceDiagnostics:
        cp = self.cp
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw_name = properties.get("name", "unknown")
        if isinstance(raw_name, bytes):
            device_name = raw_name.decode("utf-8", errors="replace")
        else:
            device_name = str(raw_name)

        try:
            cupyx_sparse = importlib.import_module("cupyx.scipy.sparse")
            # Device constructors avoid hiding diagnostic H2D transfers from the ledger.
            data = cp.ones(1, dtype=cp.float64)
            indices = cp.zeros(1, dtype=cp.int32)
            indptr = cp.arange(2, dtype=cp.int32)
            probe = cupyx_sparse.csr_matrix((data, indices, indptr), shape=(1, 1))
            csr_index_dtype = str(probe.indices.dtype)
            csr_indptr_dtype = str(probe.indptr.dtype)
            csr_index_bits = int(probe.indices.dtype.itemsize) * 8
        except Exception as error:
            raise GPUBackendUnavailable(
                "CuPy imported, but its CSR sparse backend could not create an FP64/int32 "
                "probe matrix."
            ) from error

        itemsize = int(cp.dtype(cp.float64).itemsize)
        major = int(properties.get("major", 0))
        minor = int(properties.get("minor", 0))
        return DeviceDiagnostics(
            cupy_version=str(cp.__version__),
            cuda_runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
            cuda_driver_version=int(cp.cuda.runtime.driverGetVersion()),
            device_id=self.device_id,
            device_name=device_name,
            compute_capability=(major, minor),
            integrated=bool(properties.get("integrated", False)),
            multiprocessor_count=int(properties.get("multiProcessorCount", 0)),
            total_global_memory_bytes=int(properties.get("totalGlobalMem", 0)),
            fp64_itemsize_bytes=itemsize,
            fp64_supported=itemsize == 8 and (major, minor) >= (1, 3),
            csr_index_dtype=csr_index_dtype,
            csr_indptr_dtype=csr_indptr_dtype,
            csr_index_bits=csr_index_bits,
        )

    def to_device(
        self,
        values: Any,
        *,
        phase: str,
        dtype: Any | None = None,
        kind: TransferKind | None = None,
    ) -> Any:
        """Move a host payload to the current CUDA device and record it once."""

        cp = self.cp
        if isinstance(values, cp.ndarray):
            return cp.asarray(values, dtype=dtype)
        self._synchronize_transfer_boundary()
        started = perf_counter()
        result = cp.asarray(values, dtype=dtype)
        self._synchronize_transfer_boundary()
        self._transfer_elapsed_seconds["host_to_device"] += perf_counter() - started
        self.ledger.record(
            phase=phase,
            direction="host_to_device",
            kind=kind or _payload_kind(result),
            bytes=int(result.nbytes),
        )
        return result

    def to_host(
        self,
        values: Any,
        *,
        phase: str,
        kind: TransferKind | None = None,
    ) -> np.ndarray:
        """Move one device payload to NumPy and record the explicit transfer."""

        self._synchronize_transfer_boundary()
        started = perf_counter()
        result = np.asarray(self.cp.asnumpy(values))
        self._synchronize_transfer_boundary()
        self._transfer_elapsed_seconds["device_to_host"] += perf_counter() - started
        self.ledger.record(
            phase=phase,
            direction="device_to_host",
            kind=kind or _payload_kind(result),
            bytes=int(result.nbytes),
        )
        return result

    def scalar_to_host(self, value: Any, *, phase: str) -> float:
        """Extract one device scalar without hiding it inside a vector transfer."""

        host = self.to_host(value, phase=phase, kind="scalar")
        if host.size != 1:
            raise ValueError(f"scalar_to_host requires one value; received shape {host.shape}.")
        return float(host.reshape(-1)[0])

    def synchronize(self) -> None:
        """Synchronize the current stream, defining an explicit timing boundary."""

        self.cp.cuda.get_current_stream().synchronize()

    def _synchronize_transfer_boundary(self) -> None:
        """Synchronize real CuPy transfers while permitting lightweight test doubles."""

        if hasattr(self.cp, "cuda"):
            self.synchronize()

    def transfer_timing_summary(self) -> dict[str, float]:
        """Return synchronized wall time spent in explicit transfer boundaries."""

        return {
            "host_to_device_seconds": self._transfer_elapsed_seconds["host_to_device"],
            "device_to_host_seconds": self._transfer_elapsed_seconds["device_to_host"],
        }

    def timed_call(
        self,
        operation: Callable[[], _Result],
        *,
        warmup_calls: int = 1,
        repetitions: int = 1,
    ) -> tuple[_Result, CUDATiming]:
        """Time repeated GPU work with CUDA events and a synchronized end event."""

        if not isinstance(warmup_calls, int) or warmup_calls < 0:
            raise ValueError("warmup_calls must be a non-negative integer.")
        if not isinstance(repetitions, int) or repetitions <= 0:
            raise ValueError("repetitions must be a positive integer.")

        result: _Result
        for _ in range(warmup_calls):
            result = operation()
        self.synchronize()

        start = self.cp.cuda.Event()
        stop = self.cp.cuda.Event()
        start.record()
        for _ in range(repetitions):
            result = operation()
        stop.record()
        stop.synchronize()
        elapsed_milliseconds = float(self.cp.cuda.get_elapsed_time(start, stop))
        return result, CUDATiming(
            elapsed_seconds=elapsed_milliseconds / 1_000.0,
            repetitions=repetitions,
            warmup_calls=warmup_calls,
        )

    def memory_report(self) -> MemoryReport:
        """Synchronize and sample runtime, device-pool, and pinned-pool counters."""

        self.synchronize()
        free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
        device_pool = self.cp.get_default_memory_pool()
        pinned_pool = self.cp.get_default_pinned_memory_pool()
        return MemoryReport(
            free_device_bytes=int(free_bytes),
            total_device_bytes=int(total_bytes),
            device_pool_used_bytes=int(device_pool.used_bytes()),
            device_pool_total_bytes=int(device_pool.total_bytes()),
            device_pool_free_blocks=int(device_pool.n_free_blocks()),
            pinned_pool_free_blocks=int(pinned_pool.n_free_blocks()),
        )


def create_gpu_backend(
    *,
    device_id: int = 0,
    required_cupy_version: str | None = "14.1.1",
    ledger: TransferLedger | None = None,
) -> CuPyBackend:
    """Create the explicitly requested CUDA backend or raise a clear error."""

    cp = _import_cupy()
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as error:
        raise GPUBackendUnavailable("CuPy imported, but CUDA device discovery failed.") from error
    if device_count <= 0:
        raise GPUBackendUnavailable("CuPy imported, but no CUDA devices are visible.")
    if not 0 <= device_id < device_count:
        raise GPUBackendUnavailable(
            f"CUDA device {device_id} was requested, but only {device_count} device(s) are visible."
        )
    if required_cupy_version is not None and str(cp.__version__) != required_cupy_version:
        raise GPUBackendUnavailable(
            f"Stage 6 requires CuPy {required_cupy_version}; found {cp.__version__}."
        )

    try:
        cp.cuda.Device(device_id).use()
        backend = CuPyBackend(cp, device_id=device_id, ledger=ledger)
        diagnostics = backend.diagnostics
    except GPUBackendUnavailable:
        raise
    except Exception as error:
        raise GPUBackendUnavailable(
            f"CUDA device {device_id} could not initialize the Stage 6 backend."
        ) from error
    if not diagnostics.fp64_supported:
        raise GPUBackendUnavailable(
            f"CUDA device {device_id} does not satisfy the required FP64 diagnostics."
        )
    if diagnostics.csr_index_bits != 32:
        raise GPUBackendUnavailable(
            "The verified Stage 6 sparse path requires resident 32-bit CSR indices; "
            f"CuPy created {diagnostics.csr_index_bits}-bit indices."
        )
    return backend


__all__ = [
    "CUDATiming",
    "CuPyBackend",
    "DeviceDiagnostics",
    "GPUBackendUnavailable",
    "MemoryReport",
    "TransferKind",
    "TransferLedger",
    "TransferRecord",
    "create_gpu_backend",
    "cupy_available",
]
