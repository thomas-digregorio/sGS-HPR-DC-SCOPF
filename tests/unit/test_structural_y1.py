from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy import sparse

from gpu_dcopf_hpr.canonical_lp import CanonicalLP
from gpu_dcopf_hpr.sgs_hpr import prepare_sgs_hpr, solve_sgs_hpr
from gpu_dcopf_hpr.structural_y1 import (
    DCOPFEqualityStructure,
    prepare_structural_y1,
)


@dataclass(frozen=True)
class IndependentEq55Case:
    name: str
    periods: int
    generator_count: int
    renewable_count: int
    interval_hours: float
    charge_efficiencies: tuple[float, ...]
    discharge_efficiencies: tuple[float, ...]

    @property
    def storage_count(self) -> int:
        return len(self.charge_efficiencies)

    @property
    def variable_count(self) -> int:
        return self.periods * (
            3 * self.generator_count + self.renewable_count + 2 * self.storage_count
        )

    def descriptor(self) -> DCOPFEqualityStructure:
        return DCOPFEqualityStructure(
            periods=self.periods,
            generator_count=self.generator_count,
            renewable_count=self.renewable_count,
            interval_hours=self.interval_hours,
            charge_efficiencies=self.charge_efficiencies,
            discharge_efficiencies=self.discharge_efficiencies,
        )


CASES = (
    IndependentEq55Case(
        name="four_periods_no_storage",
        periods=4,
        generator_count=3,
        renewable_count=2,
        interval_hours=1.0,
        charge_efficiencies=(),
        discharge_efficiencies=(),
    ),
    IndependentEq55Case(
        name="five_periods_one_storage",
        periods=5,
        generator_count=4,
        renewable_count=1,
        interval_hours=0.25,
        charge_efficiencies=(0.95,),
        discharge_efficiencies=(0.90,),
    ),
    IndependentEq55Case(
        name="four_periods_three_storage",
        periods=4,
        generator_count=5,
        renewable_count=2,
        interval_hours=0.5,
        charge_efficiencies=(0.98, 0.83, 0.65),
        discharge_efficiencies=(0.96, 0.78, 0.51),
    ),
    IndependentEq55Case(
        name="extreme_valid_efficiencies",
        periods=3,
        generator_count=2,
        renewable_count=1,
        interval_hours=2.0,
        charge_efficiencies=(1.0, 0.05),
        discharge_efficiencies=(1.0, 0.04),
    ),
    IndependentEq55Case(
        name="many_ideal_storage_devices",
        periods=16,
        generator_count=1,
        renewable_count=0,
        interval_hours=1.0,
        charge_efficiencies=(1.0,) * 32,
        discharge_efficiencies=(1.0,) * 32,
    ),
)


def _independent_eq55_lp(case: IndependentEq55Case) -> CanonicalLP:
    """Build Equation (55) without using production indexing or model code."""

    periods = case.periods
    generators = case.generator_count
    renewables = case.renewable_count
    storage = case.storage_count
    columns = case.variable_count
    rows = periods + storage

    p_g = 0
    p_rg = p_g + periods * generators
    p_ess_dc = p_rg + periods * renewables
    p_ess_ch = p_ess_dc + periods * storage

    A1 = sparse.lil_matrix((rows, columns), dtype=np.float64)
    for period in range(periods):
        A1[period, p_g + period * generators : p_g + (period + 1) * generators] = 1.0
        A1[period, p_rg + period * renewables : p_rg + (period + 1) * renewables] = 1.0
        A1[period, p_ess_dc + period * storage : p_ess_dc + (period + 1) * storage] = 1.0
        A1[period, p_ess_ch + period * storage : p_ess_ch + (period + 1) * storage] = -1.0

    for device, (charge_efficiency, discharge_efficiency) in enumerate(
        zip(
            case.charge_efficiencies,
            case.discharge_efficiencies,
            strict=True,
        )
    ):
        for period in range(periods):
            A1[periods + device, p_ess_dc + period * storage + device] = (
                -case.interval_hours / discharge_efficiency
            )
            A1[periods + device, p_ess_ch + period * storage + device] = (
                case.interval_hours * charge_efficiency
            )

    return CanonicalLP(
        c=np.zeros(columns),
        A1=A1.tocsr(),
        b1=np.zeros(rows),
        A2=sparse.csr_matrix((0, columns), dtype=np.float64),
        b2=np.empty(0),
        lower=np.full(columns, -1.0),
        upper=np.full(columns, 1.0),
    )


def _with_a1(lp: CanonicalLP, A1: sparse.spmatrix) -> CanonicalLP:
    return CanonicalLP(
        c=lp.c,
        A1=A1,
        b1=lp.b1,
        A2=lp.A2,
        b2=lp.b2,
        lower=lp.lower,
        upper=lp.upper,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_structural_solve_matches_dense_oracle_for_randomized_scaled_rhs(
    case: IndependentEq55Case,
) -> None:
    lp = _independent_eq55_lp(case)
    solver = prepare_structural_y1(lp, case.descriptor())
    gram = np.asarray((lp.A1 @ lp.A1.T).toarray(), dtype=np.float64)
    generator = np.random.default_rng(20260730)

    assert solver.source_lp is lp
    for rhs_scale in (1e-10, 1.0, 1e10):
        for _ in range(32):
            rhs = rhs_scale * generator.standard_normal(lp.m1)
            expected = np.linalg.solve(gram, rhs)
            actual = solver.solve(rhs)

            solution_scale = max(1.0, float(np.linalg.norm(expected, ord=np.inf)))
            rhs_norm_scale = max(1.0, float(np.linalg.norm(rhs, ord=np.inf)))
            relative_error = float(np.linalg.norm(actual - expected, ord=np.inf)) / solution_scale
            scaled_residual = (
                float(np.linalg.norm(gram @ actual - rhs, ord=np.inf)) / rhs_norm_scale
            )

            assert relative_error <= 5e-12
            assert scaled_residual <= 5e-12


def test_structural_workspace_omits_dense_gram_and_cholesky() -> None:
    case = CASES[2]
    lp = _independent_eq55_lp(case)
    solver = prepare_structural_y1(lp, case.descriptor())
    workspace = prepare_sgs_hpr(lp, structural_y1=solver)

    assert workspace.source_lp is lp
    assert workspace.structural_y1 is solver
    assert workspace.equality_gram is None
    assert workspace.equality_cholesky is None
    assert workspace.equality.full_row_rank
    assert workspace.equality.positive_definite


def test_corrected_rank_one_sign_is_required() -> None:
    case = CASES[1]
    lp = _independent_eq55_lp(case)
    solver = prepare_structural_y1(lp, case.descriptor())
    rhs = np.linspace(-1.25, 2.0, lp.m1)
    actual = solver.solve(rhs)
    diagnostics = solver.diagnostics
    inverse_storage = solver.inverse_storage_diagonal
    balance_rhs = rhs[: case.periods]
    storage_rhs = rhs[case.periods :]
    reduced = (
        balance_rhs - np.dot(diagnostics.coupling * inverse_storage, storage_rhs)
    ) / diagnostics.balance_diagonal
    printed_denominator = 1.0 / diagnostics.alpha + case.periods / diagnostics.balance_diagonal
    printed_balance = reduced - (
        np.sum(reduced) / (diagnostics.balance_diagonal * printed_denominator)
    )
    printed_storage = inverse_storage * (
        storage_rhs - diagnostics.coupling * np.sum(printed_balance)
    )
    printed_solution = np.concatenate((printed_balance, printed_storage))
    gram = np.asarray((lp.A1 @ lp.A1.T).toarray(), dtype=np.float64)

    assert np.linalg.norm(gram @ actual - rhs) <= 5e-12
    assert np.linalg.norm(gram @ printed_solution - rhs) >= 1e-3


def test_complete_solver_accepts_structural_workspace_without_long_iteration() -> None:
    case = CASES[0]
    lp = _independent_eq55_lp(case)
    solver = prepare_structural_y1(lp, case.descriptor())

    result = solve_sgs_hpr(
        lp,
        structural_y1=solver,
        max_iterations=2,
        history_interval=1,
    )

    assert result.converged
    assert result.iterations == 1
    assert result.workspace.structural_y1 is solver
    assert result.workspace.equality_gram is None
    assert result.workspace.equality_cholesky is None
    np.testing.assert_array_equal(result.solution.x, np.zeros(lp.n))


@pytest.mark.parametrize(
    ("row", "column", "delta"),
    (
        (0, 0, 0.125),
        (3, 9, -0.25),
        (0, 21, 1.0),
    ),
    ids=("balance_coefficient", "terminal_coefficient", "unexpected_reserve_entry"),
)
def test_prepare_rejects_perturbed_a1(row: int, column: int, delta: float) -> None:
    case = IndependentEq55Case(
        name="perturbation_fixture",
        periods=3,
        generator_count=2,
        renewable_count=1,
        interval_hours=1.0,
        charge_efficiencies=(0.95, 0.80),
        discharge_efficiencies=(0.90, 0.75),
    )
    original = _independent_eq55_lp(case)
    perturbed = original.A1.tolil(copy=True)
    perturbed[row, column] += delta
    lp = _with_a1(original, perturbed.tocsr())

    with pytest.raises(ValueError):
        prepare_structural_y1(lp, case.descriptor())


def test_prepare_rejects_descriptor_that_does_not_describe_lp() -> None:
    case = CASES[1]
    lp = _independent_eq55_lp(case)
    incorrect = DCOPFEqualityStructure(
        periods=case.periods,
        generator_count=case.generator_count + 1,
        renewable_count=case.renewable_count,
        interval_hours=case.interval_hours,
        charge_efficiencies=case.charge_efficiencies,
        discharge_efficiencies=case.discharge_efficiencies,
    )

    with pytest.raises(ValueError):
        prepare_structural_y1(lp, incorrect)


@pytest.mark.parametrize(
    "overrides",
    (
        {"periods": 0},
        {"generator_count": -1},
        {"renewable_count": -1},
        {"interval_hours": 0.0},
        {"interval_hours": np.nan},
        {"charge_efficiencies": (0.95, 0.90)},
        {"charge_efficiencies": (0.0,)},
        {"discharge_efficiencies": (1.01,)},
        {"discharge_efficiencies": (np.nan,)},
    ),
    ids=(
        "zero_periods",
        "negative_generators",
        "negative_renewables",
        "zero_interval",
        "nonfinite_interval",
        "efficiency_length_mismatch",
        "zero_charge_efficiency",
        "discharge_efficiency_above_one",
        "nonfinite_discharge_efficiency",
    ),
)
def test_invalid_structure_descriptor_is_rejected(overrides: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "periods": 2,
        "generator_count": 3,
        "renewable_count": 1,
        "interval_hours": 1.0,
        "charge_efficiencies": (0.95,),
        "discharge_efficiencies": (0.90,),
    }
    arguments.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        DCOPFEqualityStructure(**arguments)


def test_structural_solver_rejects_bad_rhs() -> None:
    case = CASES[2]
    lp = _independent_eq55_lp(case)
    solver = prepare_structural_y1(lp, case.descriptor())

    with pytest.raises(ValueError):
        solver.solve(np.zeros(lp.m1 + 1))
    with pytest.raises(ValueError):
        solver.solve(np.full(lp.m1, np.inf))
