from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import gpu_dcopf_hpr.gpu_backend as gpu_backend
from gpu_dcopf_hpr.gpu_backend import (
    CuPyBackend,
    GPUBackendUnavailable,
    TransferLedger,
    create_gpu_backend,
)


class _FakeDeviceArray:
    def __init__(self, values: object, dtype: object | None = None) -> None:
        self.values = np.asarray(values, dtype=dtype)

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def nbytes(self) -> int:
        return self.values.nbytes


class _FakeCuPyTransfers:
    ndarray = _FakeDeviceArray

    @staticmethod
    def asarray(values: object, dtype: object | None = None) -> _FakeDeviceArray:
        if isinstance(values, _FakeDeviceArray):
            if dtype is None or values.values.dtype == np.dtype(dtype):
                return values
            values = values.values
        return _FakeDeviceArray(values, dtype=dtype)

    @staticmethod
    def asnumpy(values: _FakeDeviceArray) -> np.ndarray:
        return np.asarray(values.values)


def test_gpu_modules_import_without_loading_cupy() -> None:
    code = (
        "import sys; "
        "sys.path.insert(0, 'src'); "
        "import gpu_dcopf_hpr.gpu_backend; "
        "import gpu_dcopf_hpr.gpu_sparse; "
        "assert 'cupy' not in sys.modules; "
        "assert 'cupyx' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_cupy_raises_clear_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = gpu_backend.importlib.import_module

    def unavailable(name: str) -> object:
        if name == "cupy":
            raise ModuleNotFoundError("no cupy")
        return real_import(name)

    monkeypatch.setattr(gpu_backend.importlib, "import_module", unavailable)
    with pytest.raises(GPUBackendUnavailable, match="CuPy GPU backend is unavailable"):
        create_gpu_backend()
    assert not gpu_backend.cupy_available()


def test_version_mismatch_fails_before_device_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(
        __version__="14.2.0",
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda: 1),
        ),
    )
    monkeypatch.setattr(gpu_backend, "_import_cupy", lambda: fake)
    with pytest.raises(GPUBackendUnavailable, match="requires CuPy 14.1.1"):
        create_gpu_backend()


def test_transfer_ledger_aggregates_phase_calls_bytes_and_payload_kind() -> None:
    ledger = TransferLedger()
    ledger.record(
        phase="setup",
        direction="host_to_device",
        kind="vector",
        bytes=24,
    )
    ledger.record(
        phase="setup",
        direction="host_to_device",
        kind="vector",
        bytes=16,
        calls=2,
    )
    ledger.record(
        phase="stopping_check",
        direction="device_to_host",
        kind="scalar",
        bytes=8,
    )

    assert ledger.total_calls("host_to_device") == 3
    assert ledger.total_bytes("host_to_device") == 40
    assert ledger.total_calls("device_to_host") == 1
    assert ledger.records[0].phase == "setup"
    assert ledger.records[1].kind == "scalar"
    assert ledger.summary()["totals"]["device_to_host"] == {"calls": 1, "bytes": 8}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"phase": "", "direction": "host_to_device", "kind": "vector", "bytes": 8}, "phase"),
        (
            {"phase": "x", "direction": "host_to_device", "kind": "vector", "bytes": -1},
            "bytes",
        ),
        (
            {
                "phase": "x",
                "direction": "host_to_device",
                "kind": "vector",
                "bytes": 8,
                "calls": 0,
            },
            "calls",
        ),
    ],
)
def test_transfer_ledger_rejects_ambiguous_records(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TransferLedger().record(**kwargs)  # type: ignore[arg-type]


def test_backend_records_vector_and_scalar_transfers_separately() -> None:
    ledger = TransferLedger()
    backend = CuPyBackend(_FakeCuPyTransfers, ledger=ledger)

    device_vector = backend.to_device(
        np.array([1.0, 2.0, 3.0]), phase="initial_state", dtype=np.float64
    )
    same_vector = backend.to_device(device_vector, phase="initial_state")
    host_vector = backend.to_host(device_vector, phase="final_solution")
    scalar = backend.scalar_to_host(_FakeDeviceArray(np.array(4.5)), phase="stopping_check")

    assert same_vector is device_vector
    np.testing.assert_array_equal(host_vector, [1.0, 2.0, 3.0])
    assert scalar == 4.5
    observed = {
        (record.phase, record.direction, record.kind): (record.calls, record.bytes)
        for record in ledger.records
    }
    assert observed == {
        ("initial_state", "host_to_device", "vector"): (1, 24),
        ("final_solution", "device_to_host", "vector"): (1, 24),
        ("stopping_check", "device_to_host", "scalar"): (1, 8),
    }


def test_synchronized_cuda_event_timing_and_memory_report() -> None:
    state = {"operation_calls": 0, "stream_syncs": 0, "stop_syncs": 0}

    class FakeStream:
        def synchronize(self) -> None:
            state["stream_syncs"] += 1

    class FakeEvent:
        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            state["stop_syncs"] += 1

    class FakePool:
        def used_bytes(self) -> int:
            return 100

        def total_bytes(self) -> int:
            return 200

        def n_free_blocks(self) -> int:
            return 3

    class FakePinnedPool:
        def n_free_blocks(self) -> int:
            return 4

    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=FakeEvent,
            get_current_stream=lambda: FakeStream(),
            get_elapsed_time=lambda _start, _stop: 2.5,
            runtime=SimpleNamespace(memGetInfo=lambda: (700, 1_000)),
        ),
        get_default_memory_pool=lambda: FakePool(),
        get_default_pinned_memory_pool=lambda: FakePinnedPool(),
    )
    backend = CuPyBackend(fake_cp)

    def operation() -> int:
        state["operation_calls"] += 1
        return state["operation_calls"]

    result, timing = backend.timed_call(operation, warmup_calls=2, repetitions=5)
    memory = backend.memory_report()

    assert result == 7
    assert state == {"operation_calls": 7, "stream_syncs": 2, "stop_syncs": 1}
    assert timing.elapsed_seconds == 0.0025
    assert timing.mean_seconds == 0.0005
    assert memory.runtime_used_bytes == 300
    assert memory.as_dict()["pinned_pool_free_blocks"] == 4
