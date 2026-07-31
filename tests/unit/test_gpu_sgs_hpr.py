from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from scipy import sparse

import gpu_dcopf_hpr.gpu_sgs_hpr as gpu_module
from gpu_dcopf_hpr.canonical_lp import CanonicalLP
from gpu_dcopf_hpr.gpu_sgs_hpr import (
    GPUHPRState,
    gpu_sgs_hpr_step,
    gpu_sgs_metric_y_quadratic,
    gpu_sgs_restart_merit,
    prepare_gpu_sgs_hpr,
)
from gpu_dcopf_hpr.hpr_generic import HPRState
from gpu_dcopf_hpr.preconditioning import precondition_lp
from gpu_dcopf_hpr.sgs_hpr import prepare_sgs_hpr, sgs_hpr_step
from gpu_dcopf_hpr.stage5_control import sgs_metric_y_quadratic, sgs_restart_merit
from gpu_dcopf_hpr.structural_y1 import DCOPFEqualityStructure, prepare_structural_y1
from gpu_dcopf_hpr.toy_problems import analytic_toy_case


@dataclass
class FakeBackend:
    """NumPy stand-in that fails immediately on a step-time host transfer."""

    xp: Any = np
    device_transfers: list[tuple[str, str, tuple[int, ...]]] = field(default_factory=list)
    host_transfer_calls: int = 0

    def to_device(self, values: Any, *, phase: str, kind: str) -> np.ndarray:
        result = np.array(values, copy=True)
        self.device_transfers.append((phase, kind, result.shape))
        return result

    def to_host(self, values: Any, *, phase: str, kind: str) -> np.ndarray:
        del values, phase, kind
        self.host_transfer_calls += 1
        raise AssertionError("the GPU step attempted a full host transfer")

    def scalar_to_host(self, value: Any, *, phase: str, kind: str) -> float:
        del value, phase, kind
        self.host_transfer_calls += 1
        raise AssertionError("the GPU step attempted a scalar host transfer")


@dataclass(frozen=True)
class FakeResidentCSR:
    matrix: sparse.csr_matrix
    transpose: sparse.csr_matrix
    name: str

    def matvec(
        self,
        vector: np.ndarray,
        *,
        transpose: bool = False,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        operator = self.transpose if transpose else self.matrix
        result = np.asarray(operator @ vector).reshape(-1)
        if out is None:
            return result
        out[...] = result
        return out


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend()

    def prepare_fake(
        selected_backend: FakeBackend,
        matrix: sparse.spmatrix,
        *,
        phase: str,
        prefer_csr_alg2: bool,
        dtype: Any,
    ) -> FakeResidentCSR:
        assert selected_backend is backend
        assert phase == "preparation"
        csr = sparse.csr_matrix(matrix, dtype=dtype, copy=True)
        return FakeResidentCSR(csr, csr.T.tocsr(), "alg2" if prefer_csr_alg2 else "default")

    monkeypatch.setattr(gpu_module, "prepare_resident_csr", prepare_fake)
    return backend


def _scaled_toy_lp() -> CanonicalLP:
    return precondition_lp(
        analytic_toy_case().lp,
        ruiz_iterations=10,
        pock_chambolle=True,
        normalize=True,
    ).scaled_lp


def _device_state(state: HPRState, backend: FakeBackend, dtype: str = "float64") -> GPUHPRState:
    return GPUHPRState.from_host(state, backend, dtype=dtype)  # type: ignore[arg-type]


def _assert_step_matches_cpu(gpu_step: Any, cpu_step: Any, *, atol: float) -> None:
    np.testing.assert_allclose(gpu_step.y1_half, cpu_step.y1_half, rtol=0.0, atol=atol)
    for gpu_state, cpu_state in (
        (gpu_step.proximal, cpu_step.proximal),
        (gpu_step.reflected, cpu_step.reflected),
        (gpu_step.next_state, cpu_step.next_state),
    ):
        np.testing.assert_allclose(gpu_state.y, cpu_state.y, rtol=0.0, atol=atol)
        np.testing.assert_allclose(gpu_state.z, cpu_state.z, rtol=0.0, atol=atol)
        np.testing.assert_allclose(gpu_state.x, cpu_state.x, rtol=0.0, atol=atol)
    for field_name in (
        "first_equality_relative_residual",
        "second_equality_relative_residual",
        "first_equality_infinity_residual",
        "second_equality_infinity_residual",
        "z_x_identity_error",
    ):
        assert getattr(gpu_step, field_name) == pytest.approx(
            getattr(cpu_step, field_name), abs=atol
        )
    assert gpu_step.update_order == cpu_step.update_order


def test_scaled_direct_step_matches_cpu_and_reuses_device_buffers(
    fake_backend: FakeBackend,
) -> None:
    lp = _scaled_toy_lp()
    cpu_workspace = prepare_sgs_hpr(lp)
    assert cpu_workspace.spectral is not None
    gpu_workspace = prepare_gpu_sgs_hpr(
        lp,
        equality_mode="scaled_direct",
        inequality_lambda=cpu_workspace.spectral.lambda_used,
        backend=fake_backend,  # type: ignore[arg-type]
    )
    current = HPRState(y=[-0.3, 0.4], z=[0.2, -0.1], x=[0.7, 0.2])
    anchor = HPRState(y=[0.1, 0.2], z=[-0.4, 0.5], x=[0.3, 0.6])
    cpu_first = sgs_hpr_step(
        lp,
        current,
        anchor,
        cpu_workspace,
        iteration=3,
        sigma=0.7,
    )
    gpu_first = gpu_sgs_hpr_step(
        lp,
        _device_state(current, fake_backend),
        _device_state(anchor, fake_backend),
        gpu_workspace,
        iteration=3,
        sigma=0.7,
    )

    _assert_step_matches_cpu(gpu_first, cpu_first, atol=5e-14)
    assert gpu_workspace.equality_mode == "scaled_direct"
    assert gpu_workspace.structural_y1 is None
    assert gpu_workspace.equality_gram is not None
    assert gpu_workspace.equality_cholesky is not None
    assert gpu_workspace.A1 is gpu_workspace.A1_resident.matrix
    assert gpu_workspace.A1_transpose is gpu_workspace.A1_resident.transpose
    assert gpu_workspace.A2 is gpu_workspace.A2_resident.matrix
    assert gpu_workspace.A2_transpose is gpu_workspace.A2_resident.transpose
    assert fake_backend.host_transfer_calls == 0

    delta_y = np.asarray([0.15, -0.35], dtype=np.float64)
    delta_x = np.asarray([-0.2, 0.45], dtype=np.float64)
    assert gpu_sgs_metric_y_quadratic(gpu_workspace, delta_y.copy()) == pytest.approx(
        sgs_metric_y_quadratic(cpu_workspace, delta_y),
        abs=5e-14,
    )
    assert gpu_sgs_restart_merit(
        gpu_workspace,
        delta_x=delta_x.copy(),
        delta_y=delta_y.copy(),
        sigma=0.9,
    ) == pytest.approx(
        sgs_restart_merit(
            cpu_workspace,
            delta_x=delta_x,
            delta_y=delta_y,
            sigma=0.9,
        ),
        abs=5e-14,
    )

    proximal_y_id = id(gpu_first.proximal.y)
    next_x_id = id(gpu_first.next_state.x)
    cpu_second = sgs_hpr_step(
        lp,
        cpu_first.next_state,
        cpu_first.proximal,
        cpu_workspace,
        iteration=0,
        sigma=1.1,
    )
    # Both inputs alias reusable output buffers.  Dedicated input snapshots
    # preserve the intended restart transition before those outputs mutate.
    gpu_second = gpu_sgs_hpr_step(
        lp,
        gpu_first.next_state,
        gpu_first.proximal,
        gpu_workspace,
        iteration=0,
        sigma=1.1,
    )
    _assert_step_matches_cpu(gpu_second, cpu_second, atol=8e-14)
    assert id(gpu_second.proximal.y) == proximal_y_id
    assert id(gpu_second.next_state.x) == next_x_id
    assert fake_backend.host_transfer_calls == 0


def test_float32_diagnostic_keeps_state_and_step_device_native(
    fake_backend: FakeBackend,
) -> None:
    lp = _scaled_toy_lp()
    cpu_workspace = prepare_sgs_hpr(lp)
    assert cpu_workspace.spectral is not None
    workspace = prepare_gpu_sgs_hpr(
        lp,
        equality_mode="scaled_direct",
        inequality_lambda=cpu_workspace.spectral.lambda_used,
        backend=fake_backend,  # type: ignore[arg-type]
        dtype="float32",
    )
    state = HPRState(y=[0.1, -0.2], z=[0.3, -0.4], x=[0.5, 0.6])
    device_state = _device_state(state, fake_backend, dtype="float32")
    step = gpu_sgs_hpr_step(
        lp,
        device_state,
        device_state,
        workspace,
        iteration=0,
        sigma=1.0,
    )

    assert workspace.dtype_name == "float32"
    assert step.proximal.y.dtype == np.float32
    assert step.proximal.z.dtype == np.float32
    assert step.proximal.x.dtype == np.float32
    assert fake_backend.host_transfer_calls == 0


def _unscaled_structural_lp() -> tuple[CanonicalLP, DCOPFEqualityStructure]:
    descriptor = DCOPFEqualityStructure(
        periods=1,
        generator_count=1,
        renewable_count=0,
        interval_hours=1.0,
        charge_efficiencies=(),
        discharge_efficiencies=(),
    )
    lp = CanonicalLP(
        c=[0.2, -0.1, 0.3],
        A1=[[1.0, 0.0, 0.0]],
        b1=[0.5],
        A2=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        b2=[-0.25, 0.1],
        lower=[-1.0, -1.0, -1.0],
        upper=[1.0, 1.0, 1.0],
    )
    return lp, descriptor


def test_unscaled_structural_step_and_device_metrics_match_cpu(
    fake_backend: FakeBackend,
) -> None:
    lp, descriptor = _unscaled_structural_lp()
    structural = prepare_structural_y1(lp, descriptor)
    cpu_workspace = prepare_sgs_hpr(lp, structural_y1=structural)
    assert cpu_workspace.spectral is not None
    gpu_workspace = prepare_gpu_sgs_hpr(
        lp,
        equality_mode="unscaled_structural",
        structural_y1=structural,
        inequality_lambda=cpu_workspace.spectral.lambda_used,
        backend=fake_backend,  # type: ignore[arg-type]
    )
    current = HPRState(
        y=[-0.3, 0.4, 0.25],
        z=[0.2, -0.1, 0.05],
        x=[0.7, 0.2, -0.2],
    )
    anchor = HPRState(y=[0.1, 0.2, 0.3], z=[-0.4, 0.5, 0.2], x=[0.3, 0.6, 0.1])
    cpu_step = sgs_hpr_step(
        lp,
        current,
        anchor,
        cpu_workspace,
        iteration=2,
        sigma=0.8,
    )
    gpu_step = gpu_sgs_hpr_step(
        lp,
        _device_state(current, fake_backend),
        _device_state(anchor, fake_backend),
        gpu_workspace,
        iteration=2,
        sigma=0.8,
    )
    _assert_step_matches_cpu(gpu_step, cpu_step, atol=2e-14)
    assert gpu_workspace.equality_mode == "unscaled_structural"
    assert gpu_workspace.equality_gram is None
    assert gpu_workspace.equality_cholesky is None
    assert gpu_workspace.structural_y1 is structural

    delta_y = np.asarray([0.2, -0.3, 0.4], dtype=np.float64)
    delta_x = np.asarray([-0.1, 0.5, 0.25], dtype=np.float64)
    expected_quadratic = sgs_metric_y_quadratic(cpu_workspace, delta_y)
    expected_merit = sgs_restart_merit(
        cpu_workspace,
        delta_x=delta_x,
        delta_y=delta_y,
        sigma=1.3,
    )
    actual_quadratic = gpu_sgs_metric_y_quadratic(gpu_workspace, delta_y.copy())
    actual_merit = gpu_sgs_restart_merit(
        gpu_workspace,
        delta_x=delta_x.copy(),
        delta_y=delta_y.copy(),
        sigma=1.3,
    )
    assert actual_quadratic == pytest.approx(expected_quadratic, abs=2e-14)
    assert actual_merit == pytest.approx(expected_merit, abs=2e-14)
    assert fake_backend.host_transfer_calls == 0


def test_equality_modes_cannot_be_mixed(fake_backend: FakeBackend) -> None:
    lp, descriptor = _unscaled_structural_lp()
    structural = prepare_structural_y1(lp, descriptor)

    with pytest.raises(ValueError, match="scaled_direct cannot use"):
        prepare_gpu_sgs_hpr(
            lp,
            equality_mode="scaled_direct",
            structural_y1=structural,
            inequality_lambda=1.0,
            backend=fake_backend,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="requires a corrected"):
        prepare_gpu_sgs_hpr(
            lp,
            equality_mode="unscaled_structural",
            inequality_lambda=1.0,
            backend=fake_backend,  # type: ignore[arg-type]
        )
    second, second_descriptor = _unscaled_structural_lp()
    mismatched = prepare_structural_y1(second, second_descriptor)
    with pytest.raises(ValueError, match="same CanonicalLP instance"):
        prepare_gpu_sgs_hpr(
            lp,
            equality_mode="unscaled_structural",
            structural_y1=mismatched,
            inequality_lambda=1.0,
            backend=fake_backend,  # type: ignore[arg-type]
        )


def test_state_validation_rejects_wrong_precision(fake_backend: FakeBackend) -> None:
    lp = _scaled_toy_lp()
    cpu_workspace = prepare_sgs_hpr(lp)
    assert cpu_workspace.spectral is not None
    workspace = prepare_gpu_sgs_hpr(
        lp,
        inequality_lambda=cpu_workspace.spectral.lambda_used,
        backend=fake_backend,  # type: ignore[arg-type]
        dtype="float32",
    )
    wrong = GPUHPRState(
        y=np.zeros(lp.m, dtype=np.float64),
        z=np.zeros(lp.n, dtype=np.float64),
        x=np.zeros(lp.n, dtype=np.float64),
    )

    with pytest.raises(ValueError, match="device dtype float32"):
        gpu_sgs_hpr_step(lp, wrong, wrong, workspace, iteration=0, sigma=1.0)
