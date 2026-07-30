from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from gpu_dcopf_hpr.dcopf_model import (
    DCOPFModel,
    build_dcopf_model,
    load_dcopf_config,
)
from gpu_dcopf_hpr.network_data import load_matpower_case
from gpu_dcopf_hpr.sgs_hpr import solve_sgs_hpr
from gpu_dcopf_hpr.structural_y1 import (
    StructuralY1Solver,
    prepare_dcopf_structural_y1,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_PATH = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
CONFIG_PATHS = (
    PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json",
    PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json",
)


@dataclass(frozen=True)
class PreparedCase:
    model: DCOPFModel
    structural_y1: StructuralY1Solver


@pytest.fixture(scope="module", params=CONFIG_PATHS, ids=lambda path: path.stem)
def prepared_case(request: pytest.FixtureRequest) -> PreparedCase:
    network = load_matpower_case(NETWORK_PATH)
    config = load_dcopf_config(request.param, network)
    model = build_dcopf_model(network, config)
    return PreparedCase(
        model=model,
        structural_y1=prepare_dcopf_structural_y1(model),
    )


def test_real_dcopf_eq55_diagnostics_match_independent_coefficients(
    prepared_case: PreparedCase,
) -> None:
    model = prepared_case.model
    config = model.config
    solver = prepared_case.structural_y1
    diagnostics = solver.diagnostics
    periods = config.periods
    storage = config.storage
    interval = config.interval_hours

    expected_coupling = np.asarray(
        [
            -interval * (resource.charge_efficiency + 1.0 / resource.discharge_efficiency)
            for resource in storage
        ],
        dtype=np.float64,
    )
    expected_storage_diagonal = np.asarray(
        [
            periods
            * interval**2
            * (resource.charge_efficiency**2 + 1.0 / resource.discharge_efficiency**2)
            for resource in storage
        ],
        dtype=np.float64,
    )
    expected_balance_diagonal = float(
        len(model.generators) + len(config.renewables) + 2 * len(storage)
    )
    expected_alpha = (
        float(np.dot(np.square(expected_coupling), 1.0 / expected_storage_diagonal))
        if storage
        else 0.0
    )
    expected_schur_scalar = expected_balance_diagonal - periods * expected_alpha
    expected_nonzeros = periods * (
        len(model.generators) + len(config.renewables) + 4 * len(storage)
    )

    assert solver.source_lp is model.lp
    assert solver.structure.expected_variables == model.lp.n
    assert solver.structure.expected_equalities == model.lp.m1
    assert diagnostics.periods == periods
    assert diagnostics.generator_count == len(model.generators)
    assert diagnostics.renewable_count == len(config.renewables)
    assert diagnostics.storage_count == len(storage)
    assert diagnostics.equality_rows == model.lp.m1
    assert diagnostics.balance_diagonal == pytest.approx(
        expected_balance_diagonal,
        abs=1e-14,
    )
    np.testing.assert_allclose(
        diagnostics.coupling,
        expected_coupling,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        diagnostics.storage_diagonal,
        expected_storage_diagonal,
        rtol=0.0,
        atol=1e-14,
    )
    assert diagnostics.alpha == pytest.approx(expected_alpha, abs=1e-14)
    assert diagnostics.schur_scalar == pytest.approx(expected_schur_scalar, abs=1e-13)
    assert diagnostics.relative_schur_margin == pytest.approx(
        expected_schur_scalar / expected_balance_diagonal,
        abs=1e-14,
    )
    assert diagnostics.maximum_a1_pattern_error <= 1e-14
    assert diagnostics.expected_a1_nonzeros == expected_nonzeros
    assert diagnostics.expected_a1_nonzeros == model.lp.A1.nnz
    assert diagnostics.stored_float_count == 3 * len(storage)
    assert diagnostics.summary()["dense_gram_materialized"] is False
    assert diagnostics.summary()["explicit_kronecker_materialized"] is False


def test_real_dcopf_structural_rhs_solve_matches_dense_oracle(
    prepared_case: PreparedCase,
) -> None:
    lp = prepared_case.model.lp
    solver = prepared_case.structural_y1
    gram = np.asarray((lp.A1 @ lp.A1.T).toarray(), dtype=np.float64)
    generator = np.random.default_rng(20260730 + lp.m1)

    for rhs_scale in (1e-9, 1.0, 1e9):
        for _ in range(64):
            rhs = rhs_scale * generator.standard_normal(lp.m1)
            expected = np.linalg.solve(gram, rhs)
            actual = solver.solve(rhs)
            solution_scale = max(1.0, float(np.linalg.norm(expected, ord=np.inf)))
            rhs_norm_scale = max(1.0, float(np.linalg.norm(rhs, ord=np.inf)))

            assert float(np.linalg.norm(actual - expected, ord=np.inf)) / solution_scale <= 5e-12
            assert float(np.linalg.norm(gram @ actual - rhs, ord=np.inf)) / rhs_norm_scale <= 5e-12


def test_real_dcopf_direct_and_structural_trajectories_match_at_500_iterations(
    prepared_case: PreparedCase,
) -> None:
    model = prepared_case.model
    settings = {
        "sigma": 1.0,
        "tolerance": 5e-5,
        "kkt_tolerance": 0.02,
        "max_iterations": 500,
        "history_interval": 500,
    }
    direct = solve_sgs_hpr(model.lp, **settings)
    structural = solve_sgs_hpr(
        model.lp,
        structural_y1=prepared_case.structural_y1,
        **settings,
    )

    assert direct.iterations == structural.iterations == 500
    assert not direct.converged
    assert not structural.converged
    np.testing.assert_allclose(
        structural.solution.x,
        direct.solution.x,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        structural.solution.y,
        direct.solution.y,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        structural.solution.z,
        direct.solution.z,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        structural.current_state.x,
        direct.current_state.x,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        structural.current_state.y,
        direct.current_state.y,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        structural.current_state.z,
        direct.current_state.z,
        rtol=0.0,
        atol=1e-9,
    )
    assert model.objective(structural.solution.x) == pytest.approx(
        model.objective(direct.solution.x),
        rel=0.0,
        abs=1e-9,
    )
    assert structural.residuals.combined_norm == pytest.approx(
        direct.residuals.combined_norm,
        rel=0.0,
        abs=1e-9,
    )
    assert structural.residuals.normalized_combined_norm == pytest.approx(
        direct.residuals.normalized_combined_norm,
        rel=0.0,
        abs=1e-9,
    )
    np.testing.assert_allclose(
        structural.residuals.paper_raw_norms,
        direct.residuals.paper_raw_norms,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        structural.residuals.paper_normalized_norms,
        direct.residuals.paper_normalized_norms,
        rtol=0.0,
        atol=1e-9,
    )

    assert direct.workspace.equality_backend == "direct"
    assert direct.workspace.equality_gram is not None
    assert direct.workspace.equality_cholesky is not None
    assert structural.workspace.equality_backend == "structural"
    assert structural.workspace.structural_y1 is prepared_case.structural_y1
    assert structural.workspace.equality_gram is None
    assert structural.workspace.equality_cholesky is None
    assert structural.maximum_equality_solve_infinity_residual <= 5e-12
