from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from scipy import sparse

import gpu_dcopf_hpr.gpu_sgs_hpr as gpu_module
from gpu_dcopf_hpr.canonical_lp import CanonicalLP
from gpu_dcopf_hpr.gpu_sgs_hpr import GPUHPRState, gpu_sgs_hpr_step, prepare_gpu_sgs_hpr
from gpu_dcopf_hpr.gpu_stage5_control import prepare_gpu_stage6_problem
from gpu_dcopf_hpr.hpr_generic import HPRState
from gpu_dcopf_hpr.preconditioning import LPPreconditioner, precondition_lp
from gpu_dcopf_hpr.sgs_hpr import prepare_sgs_hpr, sgs_hpr_step
from gpu_dcopf_hpr.stage5_control import (
    Stage5Control,
    sgs_restart_merit,
    solve_stage5_sgs_hpr,
)
from gpu_dcopf_hpr.stage7_scaled_y1 import (
    ScaledBlockArrowY1Solver,
    prepare_scaled_block_arrow_y1,
)
from gpu_dcopf_hpr.stage7_spectral import estimate_sparse_spectral_norm_squared
from gpu_dcopf_hpr.structural_y1 import DCOPFEqualityStructure, prepare_structural_y1


def _raw_problem() -> tuple[CanonicalLP, DCOPFEqualityStructure]:
    structure = DCOPFEqualityStructure(
        periods=3,
        generator_count=1,
        renewable_count=0,
        interval_hours=1.0,
        charge_efficiencies=(0.92,),
        discharge_efficiencies=(0.89,),
    )
    A1 = sparse.lil_matrix(
        (structure.expected_equalities, structure.expected_variables),
        dtype=np.float64,
    )
    discharge_offset = structure.periods
    charge_offset = discharge_offset + structure.periods
    for period in range(structure.periods):
        A1[period, period] = 1.0
        A1[period, discharge_offset + period] = 1.0
        A1[period, charge_offset + period] = -1.0
        A1[structure.periods, discharge_offset + period] = -1.0 / 0.89
        A1[structure.periods, charge_offset + period] = 0.92

    A2 = sparse.lil_matrix((4, structure.expected_variables), dtype=np.float64)
    A2[0, 0] = 1.0
    A2[0, 1] = -0.4
    A2[1, 3 * structure.periods] = 0.7
    A2[1, discharge_offset] = -1.2
    A2[2, charge_offset + 1] = 0.8
    A2[2, charge_offset + 2] = -0.3
    A2[3, 2] = -0.6
    A2[3, discharge_offset + 2] = 1.1
    return (
        CanonicalLP(
            c=np.linspace(-0.3, 0.7, structure.expected_variables),
            A1=A1.tocsr(),
            b1=np.asarray([0.2, -0.1, 0.3, 0.0]),
            A2=A2.tocsr(),
            b2=np.asarray([0.4, 0.2, -0.1, 0.5]),
            lower=np.full(structure.expected_variables, -2.0),
            upper=np.full(structure.expected_variables, 3.0),
        ),
        structure,
    )


def _prepared() -> tuple[CanonicalLP, LPPreconditioner, ScaledBlockArrowY1Solver]:
    raw, structure = _raw_problem()
    preconditioner = precondition_lp(
        raw,
        ruiz_iterations=3,
        pock_chambolle=True,
        normalize=True,
    )
    return raw, preconditioner, prepare_scaled_block_arrow_y1(preconditioner, structure)


@dataclass
class _FakeBackend:
    xp: Any = np
    transfers: list[tuple[str, str, tuple[int, ...]]] = field(default_factory=list)

    def to_device(self, values: Any, *, phase: str, kind: str) -> np.ndarray:
        result = np.array(values, copy=True)
        self.transfers.append((phase, kind, result.shape))
        return result


@dataclass(frozen=True)
class _FakeResidentCSR:
    matrix: sparse.csr_matrix
    transpose: sparse.csr_matrix

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
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    backend = _FakeBackend()

    def prepare_fake(
        selected_backend: _FakeBackend,
        matrix: sparse.spmatrix,
        *,
        phase: str,
        prefer_csr_alg2: bool,
        dtype: Any,
    ) -> _FakeResidentCSR:
        assert selected_backend is backend
        assert phase == "preparation"
        del prefer_csr_alg2
        csr = sparse.csr_matrix(matrix, dtype=dtype, copy=True)
        return _FakeResidentCSR(csr, csr.T.tocsr())

    monkeypatch.setattr(gpu_module, "prepare_resident_csr", prepare_fake)
    return backend


def _states(lp: CanonicalLP) -> tuple[HPRState, HPRState]:
    current = HPRState(
        y=np.linspace(-0.2, 0.3, lp.m),
        z=np.linspace(0.1, -0.15, lp.n),
        x=np.linspace(-0.4, 0.6, lp.n),
    )
    anchor = HPRState(
        y=np.linspace(0.25, -0.1, lp.m),
        z=np.linspace(-0.2, 0.2, lp.n),
        x=np.linspace(0.3, -0.3, lp.n),
    )
    return current, anchor


def _assert_step_equal(actual: Any, expected: Any) -> None:
    np.testing.assert_allclose(actual.y1_half, expected.y1_half, rtol=2e-13, atol=2e-13)
    for actual_state, expected_state in (
        (actual.proximal, expected.proximal),
        (actual.reflected, expected.reflected),
        (actual.next_state, expected.next_state),
    ):
        np.testing.assert_allclose(actual_state.y, expected_state.y, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(actual_state.z, expected_state.z, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(actual_state.x, expected_state.x, rtol=2e-13, atol=2e-13)


def test_cpu_scaled_structural_step_matches_direct_with_sparse_certificate() -> None:
    raw, preconditioner, solver = _prepared()
    lp = preconditioner.scaled_lp
    certificate = estimate_sparse_spectral_norm_squared(sparse.csr_matrix(lp.A2))
    direct = prepare_sgs_hpr(lp, spectral_certificate=certificate)
    structural = prepare_sgs_hpr(
        lp,
        scaled_structural_y1=solver,
        spectral_certificate=certificate,
    )
    current, anchor = _states(lp)

    expected = sgs_hpr_step(lp, current, anchor, direct, iteration=4, sigma=0.75)
    actual = sgs_hpr_step(lp, current, anchor, structural, iteration=4, sigma=0.75)

    _assert_step_equal(actual, expected)
    assert structural.equality_backend == "scaled_structural"
    assert structural.equality_gram is None
    assert structural.scaled_structural_y1 is solver
    assert structural.spectral is certificate

    delta_x = np.linspace(-0.1, 0.15, lp.n)
    delta_y = np.linspace(0.2, -0.25, lp.m)
    expected_merit = sgs_restart_merit(
        direct,
        delta_x=delta_x,
        delta_y=delta_y,
        sigma=0.9,
    )
    actual_merit = sgs_restart_merit(
        structural,
        delta_x=delta_x,
        delta_y=delta_y,
        sigma=0.9,
    )
    assert actual_merit == pytest.approx(expected_merit, rel=2e-13, abs=2e-13)

    result = solve_stage5_sgs_hpr(
        raw,
        preconditioner=preconditioner,
        scaled_structural_y1=solver,
        spectral_certificate=certificate,
        max_iterations=1,
        control=Stage5Control(restart=True),
    )
    assert result.workspace.equality_backend == "scaled_structural"
    assert result.workspace.spectral is certificate

    reused = solve_stage5_sgs_hpr(
        raw,
        preconditioner=preconditioner,
        scaled_structural_y1=solver,
        spectral_certificate=certificate,
        prepared_workspace=structural,
        max_iterations=1,
        control=Stage5Control(restart=True),
    )
    assert reused.workspace is structural
    assert reused.preparation_elapsed_seconds == 0.0
    np.testing.assert_array_equal(reused.solution.x, result.solution.x)
    np.testing.assert_array_equal(reused.solution.y, result.solution.y)
    np.testing.assert_array_equal(reused.solution.z, result.solution.z)


def test_certificate_rejects_same_pattern_with_one_changed_coefficient() -> None:
    _raw, preconditioner, _solver = _prepared()
    lp = preconditioner.scaled_lp
    certificate = estimate_sparse_spectral_norm_squared(sparse.csr_matrix(lp.A2))
    changed_A2 = sparse.csr_matrix(lp.A2).copy()
    changed_A2.data[0] = np.nextafter(changed_A2.data[0], np.inf)
    changed = CanonicalLP(
        c=lp.c,
        A1=lp.A1,
        b1=lp.b1,
        A2=changed_A2,
        b2=lp.b2,
        lower=lp.lower,
        upper=lp.upper,
    )

    assert changed_A2.shape == sparse.csr_matrix(lp.A2).shape
    assert changed_A2.nnz == sparse.csr_matrix(lp.A2).nnz
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_sgs_hpr(changed, spectral_certificate=certificate)


def test_cpu_scaled_structural_rejects_mismatched_or_mixed_descriptors() -> None:
    raw, preconditioner, solver = _prepared()
    _second_raw, second_preconditioner, second_solver = _prepared()
    raw_structural = prepare_structural_y1(raw, solver.structure)

    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_sgs_hpr(
            preconditioner.scaled_lp,
            structural_y1=raw_structural,
            scaled_structural_y1=solver,
        )
    with pytest.raises(ValueError, match="same scaled CanonicalLP"):
        prepare_sgs_hpr(
            preconditioner.scaled_lp,
            scaled_structural_y1=second_solver,
        )
    with pytest.raises(ValueError, match="exact supplied preconditioner"):
        solve_stage5_sgs_hpr(
            raw,
            preconditioner=preconditioner,
            scaled_structural_y1=second_solver,
            max_iterations=1,
        )
    with pytest.raises(ValueError, match="requires its exact preconditioner"):
        solve_stage5_sgs_hpr(
            second_preconditioner.scaled_lp,
            scaled_structural_y1=second_solver,
            max_iterations=1,
        )


def test_prepared_cpu_workspace_rejects_backend_source_and_certificate_mismatch() -> None:
    raw, preconditioner, solver = _prepared()
    _other_raw, other_preconditioner, other_solver = _prepared()
    lp = preconditioner.scaled_lp
    certificate = estimate_sparse_spectral_norm_squared(sparse.csr_matrix(lp.A2))
    workspace = prepare_sgs_hpr(
        lp,
        scaled_structural_y1=solver,
        spectral_certificate=certificate,
    )
    direct = prepare_sgs_hpr(lp, spectral_certificate=certificate)
    other_certificate = estimate_sparse_spectral_norm_squared(sparse.csr_matrix(lp.A2))
    other_workspace = prepare_sgs_hpr(
        other_preconditioner.scaled_lp,
        scaled_structural_y1=other_solver,
        spectral_certificate=estimate_sparse_spectral_norm_squared(
            sparse.csr_matrix(other_preconditioner.scaled_lp.A2)
        ),
    )

    common = {
        "preconditioner": preconditioner,
        "scaled_structural_y1": solver,
        "spectral_certificate": certificate,
        "max_iterations": 1,
    }
    with pytest.raises(ValueError, match="exact algorithm LP"):
        solve_stage5_sgs_hpr(raw, prepared_workspace=other_workspace, **common)
    with pytest.raises(ValueError, match="exact supplied scaled_structural_y1"):
        solve_stage5_sgs_hpr(raw, prepared_workspace=direct, **common)
    with pytest.raises(ValueError, match="identity/fingerprint mismatch"):
        solve_stage5_sgs_hpr(
            raw,
            preconditioner=preconditioner,
            scaled_structural_y1=solver,
            spectral_certificate=other_certificate,
            prepared_workspace=workspace,
            max_iterations=1,
        )


def test_gpu_scaled_structural_numpy_backend_matches_cpu(
    fake_backend: _FakeBackend,
) -> None:
    raw, preconditioner, solver = _prepared()
    lp = preconditioner.scaled_lp
    certificate = estimate_sparse_spectral_norm_squared(sparse.csr_matrix(lp.A2))
    cpu_workspace = prepare_sgs_hpr(
        lp,
        scaled_structural_y1=solver,
        spectral_certificate=certificate,
    )
    gpu_workspace = prepare_gpu_sgs_hpr(
        lp,
        equality_mode="scaled_structural",
        scaled_structural_y1=solver,
        inequality_lambda=certificate.lambda_used,
        backend=fake_backend,  # type: ignore[arg-type]
    )
    current, anchor = _states(lp)

    expected = sgs_hpr_step(lp, current, anchor, cpu_workspace, iteration=2, sigma=1.1)
    actual = gpu_sgs_hpr_step(
        lp,
        GPUHPRState.from_host(current, fake_backend),  # type: ignore[arg-type]
        GPUHPRState.from_host(anchor, fake_backend),  # type: ignore[arg-type]
        gpu_workspace,
        iteration=2,
        sigma=1.1,
    )

    _assert_step_equal(actual, expected)
    assert gpu_workspace.equality_mode == "scaled_structural"
    assert gpu_workspace.equality_gram is None
    assert gpu_workspace.equality_cholesky is None
    assert gpu_workspace.device_scaled_structural_y1 is not None

    problem = prepare_gpu_stage6_problem(
        raw,
        preconditioner,
        backend=fake_backend,  # type: ignore[arg-type]
        inequality_lambda=certificate.lambda_used,
        scaled_structural_y1=solver,
    )
    assert problem.workspace.equality_mode == "scaled_structural"


def test_gpu_scaled_structural_rejects_precision_mode_and_identity(
    fake_backend: _FakeBackend,
) -> None:
    _raw, preconditioner, solver = _prepared()
    _other_raw, _other_preconditioner, other_solver = _prepared()
    lp = preconditioner.scaled_lp

    with pytest.raises(ValueError, match="requires FP64"):
        prepare_gpu_sgs_hpr(
            lp,
            equality_mode="scaled_structural",
            scaled_structural_y1=solver,
            inequality_lambda=1.0,
            backend=fake_backend,  # type: ignore[arg-type]
            dtype="float32",
        )
    with pytest.raises(ValueError, match="requires a ScaledBlockArrowY1Solver"):
        prepare_gpu_sgs_hpr(
            lp,
            equality_mode="scaled_structural",
            inequality_lambda=1.0,
            backend=fake_backend,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="same exact scaled CanonicalLP"):
        prepare_gpu_sgs_hpr(
            lp,
            equality_mode="scaled_structural",
            scaled_structural_y1=other_solver,
            inequality_lambda=1.0,
            backend=fake_backend,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="scaled_direct cannot use"):
        prepare_gpu_sgs_hpr(
            lp,
            equality_mode="scaled_direct",
            scaled_structural_y1=solver,
            inequality_lambda=1.0,
            backend=fake_backend,  # type: ignore[arg-type]
        )
