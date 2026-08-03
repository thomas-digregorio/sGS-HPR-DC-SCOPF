from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg, sparse

from gpu_dcopf_hpr.canonical_lp import CanonicalLP
from gpu_dcopf_hpr.preconditioning import LPPreconditioner, precondition_lp
from gpu_dcopf_hpr.sgs_hpr import prepare_sgs_hpr
from gpu_dcopf_hpr.stage7_scaled_y1 import prepare_scaled_block_arrow_y1
from gpu_dcopf_hpr.structural_y1 import DCOPFEqualityStructure


def _eq55_lp(structure: DCOPFEqualityStructure) -> CanonicalLP:
    periods = structure.periods
    generators = structure.generator_count
    renewables = structure.renewable_count
    storage = structure.storage_count
    columns = structure.expected_variables
    rows = structure.expected_equalities
    generator_offset = 0
    renewable_offset = periods * generators
    discharge_offset = renewable_offset + periods * renewables
    charge_offset = discharge_offset + periods * storage

    A1 = sparse.lil_matrix((rows, columns), dtype=np.float64)
    for period in range(periods):
        A1[
            period,
            generator_offset + period * generators : generator_offset + (period + 1) * generators,
        ] = 1.0
        A1[
            period,
            renewable_offset + period * renewables : renewable_offset + (period + 1) * renewables,
        ] = 1.0
        A1[
            period,
            discharge_offset + period * storage : discharge_offset + (period + 1) * storage,
        ] = 1.0
        A1[
            period,
            charge_offset + period * storage : charge_offset + (period + 1) * storage,
        ] = -1.0

    charge = np.asarray(structure.charge_efficiencies)
    discharge = np.asarray(structure.discharge_efficiencies)
    for device in range(storage):
        for period in range(periods):
            A1[periods + device, discharge_offset + period * storage + device] = (
                -structure.interval_hours / discharge[device]
            )
            A1[periods + device, charge_offset + period * storage + device] = (
                structure.interval_hours * charge[device]
            )

    return CanonicalLP(
        c=np.linspace(-0.5, 0.5, columns),
        A1=A1.tocsr(),
        b1=np.zeros(rows),
        A2=sparse.csr_matrix((0, columns), dtype=np.float64),
        b2=np.empty(0),
        lower=np.full(columns, -2.0),
        upper=np.full(columns, 3.0),
    )


def _structure(*, storage: bool = True) -> DCOPFEqualityStructure:
    return DCOPFEqualityStructure(
        periods=5,
        generator_count=3,
        renewable_count=2,
        interval_hours=0.5,
        charge_efficiencies=(0.94, 0.81) if storage else (),
        discharge_efficiencies=(0.91, 0.73) if storage else (),
    )


def _prepared(structure: DCOPFEqualityStructure) -> LPPreconditioner:
    return precondition_lp(
        _eq55_lp(structure),
        ruiz_iterations=5,
        pock_chambolle=True,
        normalize=True,
    )


@pytest.mark.parametrize("storage", (False, True), ids=("no_storage", "two_storage"))
def test_scaled_block_arrow_matches_full_direct_solve(storage: bool) -> None:
    structure = _structure(storage=storage)
    preconditioner = _prepared(structure)
    solver = prepare_scaled_block_arrow_y1(preconditioner, structure)
    A1 = sparse.csr_matrix(preconditioner.scaled_lp.A1)
    full_gram = np.asarray((A1 @ A1.T).toarray(), dtype=np.float64)
    generator = np.random.default_rng(20260803)

    for scale in (1e-8, 1.0, 1e8):
        for _ in range(12):
            rhs = scale * generator.standard_normal(preconditioner.scaled_lp.m1)
            expected = np.linalg.solve(full_gram, rhs)
            actual = solver.solve(rhs)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12 * scale)
            residual_scale = max(1.0, float(np.linalg.norm(rhs, ord=np.inf)))
            assert np.linalg.norm(full_gram @ actual - rhs, ord=np.inf) / residual_scale < 2e-12

    assert solver.preconditioner is preconditioner
    assert solver.diagnostics.summary()["dense_equality_gram_materialized"] is False
    assert solver.diagnostics.summary()["dense_schur_shape"] == [
        structure.storage_count,
        structure.storage_count,
    ]


def _replace_scaled_a1(
    preconditioner: LPPreconditioner,
    A1: sparse.spmatrix,
) -> LPPreconditioner:
    scaled_lp = CanonicalLP(
        c=preconditioner.scaled_lp.c,
        A1=A1,
        b1=preconditioner.scaled_lp.b1,
        A2=preconditioner.scaled_lp.A2,
        b2=preconditioner.scaled_lp.b2,
        lower=preconditioner.scaled_lp.lower,
        upper=preconditioner.scaled_lp.upper,
    )
    return replace(preconditioner, scaled_lp=scaled_lp)


def test_prepare_explicitly_rejects_incompatible_scaled_sparsity() -> None:
    structure = _structure()
    preconditioner = _prepared(structure)
    perturbed = sparse.csr_matrix(preconditioner.scaled_lp.A1).tolil(copy=True)
    reserve_column = structure.periods * (
        structure.generator_count + structure.renewable_count + 2 * structure.storage_count
    )
    assert perturbed[0, reserve_column] == 0.0
    perturbed[0, reserve_column] = 0.125

    with pytest.raises(ValueError, match="incompatible sparsity"):
        prepare_scaled_block_arrow_y1(
            _replace_scaled_a1(preconditioner, perturbed.tocsr()),
            structure,
        )


def test_prepare_rejects_scaled_value_not_explained_by_denominators() -> None:
    structure = _structure()
    preconditioner = _prepared(structure)
    perturbed = sparse.csr_matrix(preconditioner.scaled_lp.A1).copy()
    perturbed.data[0] *= 1.01

    with pytest.raises(ValueError, match="diagonal transform"):
        prepare_scaled_block_arrow_y1(
            _replace_scaled_a1(preconditioner, perturbed),
            structure,
        )


def test_prepare_rejects_descriptor_for_a_different_raw_model() -> None:
    structure = _structure()
    preconditioner = _prepared(structure)
    wrong = DCOPFEqualityStructure(
        periods=structure.periods,
        generator_count=structure.generator_count + 1,
        renewable_count=structure.renewable_count,
        interval_hours=structure.interval_hours,
        charge_efficiencies=structure.charge_efficiencies,
        discharge_efficiencies=structure.discharge_efficiencies,
    )

    with pytest.raises(ValueError):
        prepare_scaled_block_arrow_y1(preconditioner, wrong)


def test_cpu_solver_rejects_bad_rhs() -> None:
    structure = _structure()
    solver = prepare_scaled_block_arrow_y1(_prepared(structure), structure)

    with pytest.raises(ValueError, match="shape"):
        solver.solve(np.zeros(structure.expected_equalities + 1))
    with pytest.raises(ValueError, match="finite"):
        solver.solve(np.full(structure.expected_equalities, np.inf))


def test_cpu_solver_duck_types_into_existing_sparse_workspace() -> None:
    structure = _structure()
    preconditioner = _prepared(structure)
    solver = prepare_scaled_block_arrow_y1(preconditioner, structure)
    workspace = prepare_sgs_hpr(
        preconditioner.scaled_lp,
        structural_y1=solver,  # type: ignore[arg-type]
    )

    assert solver.source_lp is preconditioner.scaled_lp
    assert workspace.source_lp is preconditioner.scaled_lp
    assert workspace.structural_y1 is solver
    assert workspace.equality_gram is None


def test_device_path_matches_cpu_with_numpy_backend_double() -> None:
    structure = _structure()
    solver = prepare_scaled_block_arrow_y1(_prepared(structure), structure)
    transfers: list[tuple[str, str]] = []

    class NumpyBackend:
        xp = np

        @staticmethod
        def to_device(
            values: object,
            *,
            phase: str,
            kind: str,
        ) -> np.ndarray:
            transfers.append((phase, kind))
            return np.asarray(values)

    device = solver.to_device(
        NumpyBackend(),  # type: ignore[arg-type]
        triangular_solve=linalg.solve_triangular,
    )
    rhs = np.linspace(-2.0, 3.0, structure.expected_equalities)

    np.testing.assert_allclose(device.solve(rhs), solver.solve(rhs), rtol=2e-15, atol=2e-15)
    out = np.empty_like(rhs)
    assert device.solve_into(rhs, out) is out
    np.testing.assert_allclose(out, solver.solve(rhs), rtol=2e-15, atol=2e-15)
    assert device.source_lp is solver.source_lp
    assert transfers == [
        ("stage7_scaled_equality_setup", "vector"),
        ("stage7_scaled_equality_setup", "matrix"),
        ("stage7_scaled_equality_setup", "matrix"),
    ]
    with pytest.raises(ValueError, match="shape"):
        device.solve(np.zeros(structure.expected_equalities + 1))
    with pytest.raises(ValueError, match="out"):
        device.solve_into(rhs, np.empty(structure.expected_equalities + 1))


def test_device_path_loads_cupyx_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    structure = _structure()
    solver = prepare_scaled_block_arrow_y1(_prepared(structure), structure)

    class NumpyBackend:
        xp = np

        @staticmethod
        def to_device(values: object, **_kwargs: object) -> np.ndarray:
            return np.asarray(values)

    requested: list[str] = []

    def fake_import(name: str) -> object:
        requested.append(name)
        return SimpleNamespace(solve_triangular=linalg.solve_triangular)

    monkeypatch.setattr(
        "gpu_dcopf_hpr.stage7_scaled_y1.importlib.import_module",
        fake_import,
    )
    device = solver.to_device(NumpyBackend())  # type: ignore[arg-type]

    assert requested == ["cupyx.scipy.linalg"]
    np.testing.assert_allclose(
        device.solve(np.ones(structure.expected_equalities)),
        solver.solve(np.ones(structure.expected_equalities)),
    )
