"""Independent HiGHS reference solve and Stage 1 comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog

from .canonical_lp import CanonicalLP
from .dcopf_model import DCOPFModel
from .hpr_generic import HPRState
from .residuals import ResidualEvaluation, evaluate_residuals


@dataclass(frozen=True, slots=True)
class HighsSolution:
    state: HPRState
    objective: float
    residuals: ResidualEvaluation
    status: int
    message: str
    iterations: int

    def summary(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "x": self.state.x.tolist(),
            "y": self.state.y.tolist(),
            "z": self.state.z.tolist(),
            "status": self.status,
            "message": self.message,
            "iterations": self.iterations,
            "residuals": self.residuals.summary(),
        }


def solve_with_highs(lp: CanonicalLP, *, tolerance: float = 5e-5) -> HighsSolution:
    """Solve the canonical LP with SciPy's independently maintained HiGHS backend."""

    A_eq = lp.A1 if lp.m1 else None
    b_eq = lp.b1 if lp.m1 else None
    A_ub = -lp.A2 if lp.m2 else None
    b_ub = -lp.b2 if lp.m2 else None
    bounds = list(zip(lp.lower.tolist(), lp.upper.tolist(), strict=True))
    result = linprog(
        lp.c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ds",
        options={
            "presolve": True,
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success:
        raise RuntimeError(f"HiGHS failed with status {result.status}: {result.message}")

    y1 = (
        np.asarray(result.eqlin.marginals, dtype=np.float64)
        if lp.m1
        else np.empty(0, dtype=np.float64)
    )
    y2 = (
        -np.asarray(result.ineqlin.marginals, dtype=np.float64)
        if lp.m2
        else np.empty(0, dtype=np.float64)
    )
    z = np.asarray(result.lower.marginals, dtype=np.float64) + np.asarray(
        result.upper.marginals,
        dtype=np.float64,
    )
    state = HPRState(y=np.concatenate((y1, y2)), z=z, x=result.x)
    residuals = evaluate_residuals(
        lp,
        x=state.x,
        y=state.y,
        z=state.z,
        tolerance=tolerance,
    )
    return HighsSolution(
        state=state,
        objective=float(result.fun),
        residuals=residuals,
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
    )


def maximum_primal_violation(lp: CanonicalLP, state: HPRState) -> float:
    equality = (
        float(np.max(np.abs(np.asarray(lp.A1 @ state.x).reshape(-1) - lp.b1))) if lp.m1 else 0.0
    )
    inequality = (
        float(np.max(np.maximum(lp.b2 - np.asarray(lp.A2 @ state.x).reshape(-1), 0.0)))
        if lp.m2
        else 0.0
    )
    lower = float(np.max(np.maximum(lp.lower - state.x, 0.0)))
    upper = float(np.max(np.maximum(state.x - lp.upper, 0.0)))
    return max(equality, inequality, lower, upper)


@dataclass(frozen=True, slots=True)
class ConstraintFamilyValidation:
    """Independent feasibility result for one physical constraint family."""

    family: str
    checks: int
    maximum_violation: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return bool(
            np.isfinite(self.maximum_violation) and self.maximum_violation <= self.tolerance
        )

    def summary(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "checks": self.checks,
            "maximum_violation": self.maximum_violation,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class DCOPFValidation:
    """Formula-based validation that does not trust the solver status or A2 alone."""

    families: tuple[ConstraintFamilyValidation, ...]
    variable_objective: float
    total_objective: float
    direct_total_objective: float
    objective_difference: float
    all_values_finite: bool
    maximum_ptdf_angle_flow_difference: float

    @property
    def passed(self) -> bool:
        return (
            self.all_values_finite
            and all(family.passed for family in self.families)
            and self.objective_difference <= 1e-8
            and self.maximum_ptdf_angle_flow_difference <= 1e-8
        )

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "all_values_finite": self.all_values_finite,
            "variable_objective": self.variable_objective,
            "objective_constant": self.total_objective - self.variable_objective,
            "total_objective": self.total_objective,
            "direct_total_objective": self.direct_total_objective,
            "objective_difference": self.objective_difference,
            "maximum_ptdf_angle_flow_difference": self.maximum_ptdf_angle_flow_difference,
            "families": [family.summary() for family in self.families],
        }


def _maximum(values: list[float]) -> float:
    return max(values, default=0.0)


def validate_dcopf_solution(
    model: DCOPFModel,
    x: np.ndarray,
    *,
    tolerance: float = 1e-7,
) -> DCOPFValidation:
    """Recalculate every printed physical constraint family from semantics."""

    vector = np.asarray(x, dtype=np.float64)
    if vector.shape != (model.lp.n,):
        raise ValueError(f"x has shape {vector.shape}; expected {(model.lp.n,)}.")
    blocks = model.unpack(vector)
    periods = model.config.periods
    families: list[ConstraintFamilyValidation] = []

    balance_violations: list[float] = []
    line_violations: list[float] = []
    ptdf_angle_differences: list[float] = []
    constrained_positions = {
        branch.branch_id: model.ptdf.branch_ids.index(branch.branch_id)
        for branch in model.constrained_branches
    }
    for period in range(periods):
        supply = (
            float(np.sum(blocks["p_g"][period]))
            + float(np.sum(blocks["p_rg"][period]))
            + float(np.sum(blocks["p_ess_dc"][period]))
            - float(np.sum(blocks["p_ess_ch"][period]))
        )
        balance_violations.append(abs(supply - float(np.sum(model.load_mw[period]))))
        injection = model.bus_injections(vector, period)
        ptdf_flows = model.ptdf.flows_from_injections(injection)
        _, angle_flows = model.ptdf.angles_and_flows(injection)
        ptdf_angle_differences.append(float(np.max(np.abs(ptdf_flows - angle_flows), initial=0.0)))
        for branch in model.constrained_branches:
            line_violations.append(
                max(
                    abs(float(ptdf_flows[constrained_positions[branch.branch_id]]))
                    - branch.rate_a_mw,
                    0.0,
                )
            )
    families.append(
        ConstraintFamilyValidation(
            "power_balance",
            len(balance_violations),
            _maximum(balance_violations),
            tolerance,
        )
    )
    families.append(
        ConstraintFamilyValidation(
            "line_flow",
            len(line_violations),
            _maximum(line_violations),
            tolerance,
        )
    )

    generator_bound_violations: list[float] = []
    reserve_bound_violations: list[float] = []
    headroom_violations: list[float] = []
    for period in range(periods):
        for column, (generator, ramp_up, ramp_down) in enumerate(
            zip(
                model.generators,
                model.config.generator_ramp_up_mw_per_hour,
                model.config.generator_ramp_down_mw_per_hour,
                strict=True,
            )
        ):
            pg = float(blocks["p_g"][period, column])
            reserve_up = float(blocks["r_up"][period, column])
            reserve_down = float(blocks["r_down"][period, column])
            generator_bound_violations.extend(
                [
                    max(generator.minimum_mw - pg, 0.0),
                    max(pg - generator.maximum_mw, 0.0),
                ]
            )
            reserve_bound_violations.extend(
                [
                    max(-reserve_up, 0.0),
                    max(
                        reserve_up - ramp_up * model.config.interval_hours,
                        0.0,
                    ),
                    max(-reserve_down, 0.0),
                    max(
                        reserve_down - ramp_down * model.config.interval_hours,
                        0.0,
                    ),
                ]
            )
            headroom_violations.extend(
                [
                    max(pg + reserve_up - generator.maximum_mw, 0.0),
                    max(generator.minimum_mw + reserve_down - pg, 0.0),
                ]
            )
    families.extend(
        [
            ConstraintFamilyValidation(
                "generator_bounds",
                len(generator_bound_violations),
                _maximum(generator_bound_violations),
                tolerance,
            ),
            ConstraintFamilyValidation(
                "reserve_bounds",
                len(reserve_bound_violations),
                _maximum(reserve_bound_violations),
                tolerance,
            ),
            ConstraintFamilyValidation(
                "reserve_headroom_footroom",
                len(headroom_violations),
                _maximum(headroom_violations),
                tolerance,
            ),
        ]
    )

    reserve_requirement_violations: list[float] = []
    for period in range(periods):
        reserve_requirement_violations.extend(
            [
                max(
                    model.config.reserve_up_mw[period] - float(np.sum(blocks["r_up"][period])),
                    0.0,
                ),
                max(
                    model.config.reserve_down_mw[period] - float(np.sum(blocks["r_down"][period])),
                    0.0,
                ),
            ]
        )
    families.append(
        ConstraintFamilyValidation(
            "reserve_requirements",
            len(reserve_requirement_violations),
            _maximum(reserve_requirement_violations),
            tolerance,
        )
    )

    ramp_violations: list[float] = []
    for period in range(1, periods):
        for column, (ramp_up, ramp_down) in enumerate(
            zip(
                model.config.generator_ramp_up_mw_per_hour,
                model.config.generator_ramp_down_mw_per_hour,
                strict=True,
            )
        ):
            change = float(blocks["p_g"][period, column] - blocks["p_g"][period - 1, column])
            ramp_violations.extend(
                [
                    max(
                        change - ramp_up * model.config.interval_hours,
                        0.0,
                    ),
                    max(
                        -ramp_down * model.config.interval_hours - change,
                        0.0,
                    ),
                ]
            )
    families.append(
        ConstraintFamilyValidation(
            "generator_ramping",
            len(ramp_violations),
            _maximum(ramp_violations),
            tolerance,
        )
    )

    renewable_violations: list[float] = []
    for column, renewable in enumerate(model.config.renewables):
        for period in range(periods):
            output = float(blocks["p_rg"][period, column])
            renewable_violations.extend(
                [
                    max(renewable.minimum_mw[period] - output, 0.0),
                    max(output - renewable.maximum_mw[period], 0.0),
                ]
            )
    families.append(
        ConstraintFamilyValidation(
            "renewable_bounds",
            len(renewable_violations),
            _maximum(renewable_violations),
            tolerance,
        )
    )

    storage_power_violations: list[float] = []
    storage_energy_violations: list[float] = []
    storage_terminal_violations: list[float] = []
    for column, storage in enumerate(model.config.storage):
        energy = storage.initial_energy_mwh
        for period in range(periods):
            discharge = float(blocks["p_ess_dc"][period, column])
            charge = float(blocks["p_ess_ch"][period, column])
            storage_power_violations.extend(
                [
                    max(-discharge, 0.0),
                    max(discharge - storage.maximum_discharge_mw, 0.0),
                    max(-charge, 0.0),
                    max(charge - storage.maximum_charge_mw, 0.0),
                ]
            )
            energy += model.config.interval_hours * (
                storage.charge_efficiency * charge - discharge / storage.discharge_efficiency
            )
            storage_energy_violations.extend(
                [
                    max(storage.minimum_energy_mwh - energy, 0.0),
                    max(energy - storage.maximum_energy_mwh, 0.0),
                ]
            )
        storage_terminal_violations.append(abs(energy - storage.initial_energy_mwh))
    families.extend(
        [
            ConstraintFamilyValidation(
                "storage_power_bounds",
                len(storage_power_violations),
                _maximum(storage_power_violations),
                tolerance,
            ),
            ConstraintFamilyValidation(
                "storage_energy_bounds",
                len(storage_energy_violations),
                _maximum(storage_energy_violations),
                tolerance,
            ),
            ConstraintFamilyValidation(
                "storage_terminal_energy",
                len(storage_terminal_violations),
                _maximum(storage_terminal_violations),
                tolerance,
            ),
        ]
    )

    equality_residual = np.asarray(model.lp.A1 @ vector).reshape(-1) - model.lp.b1
    inequality_slack = np.asarray(model.lp.A2 @ vector).reshape(-1) - model.lp.b2
    box_violation = np.maximum(model.lp.lower - vector, 0.0) + np.maximum(
        vector - model.lp.upper,
        0.0,
    )
    families.extend(
        [
            ConstraintFamilyValidation(
                "canonical_equalities",
                model.lp.m1,
                float(np.max(np.abs(equality_residual), initial=0.0)),
                tolerance,
            ),
            ConstraintFamilyValidation(
                "canonical_inequalities",
                model.lp.m2,
                float(np.max(np.maximum(-inequality_slack, 0.0), initial=0.0)),
                tolerance,
            ),
            ConstraintFamilyValidation(
                "canonical_box",
                model.lp.n,
                float(np.max(box_violation, initial=0.0)),
                tolerance,
            ),
        ]
    )

    direct_objective = 0.0
    for column, generator in enumerate(model.generators):
        slope, constant, omitted = generator.cost.paper_linear_terms()
        if omitted and any(abs(value) > 0.0 for value in omitted):
            raise ValueError("Independent validation does not accept a nonlinear generator cost.")
        direct_objective += sum(
            slope * float(blocks["p_g"][period, column]) + constant for period in range(periods)
        )
    for column, renewable in enumerate(model.config.renewables):
        direct_objective += sum(
            model.config.renewable_penalty_per_mwh
            * (renewable.maximum_mw[period] - float(blocks["p_rg"][period, column]))
            for period in range(periods)
        )
    for column, storage in enumerate(model.config.storage):
        direct_objective += sum(
            model.config.storage_loss_penalty_per_mwh
            * (
                float(blocks["p_ess_dc"][period, column])
                * (1.0 / storage.discharge_efficiency - 1.0)
                + float(blocks["p_ess_ch"][period, column]) * (1.0 - storage.charge_efficiency)
            )
            for period in range(periods)
        )

    variable_objective = model.objective(vector, include_constant=False)
    total_objective = model.objective(vector, include_constant=True)
    return DCOPFValidation(
        families=tuple(families),
        variable_objective=variable_objective,
        total_objective=total_objective,
        direct_total_objective=float(direct_objective),
        objective_difference=abs(total_objective - float(direct_objective)),
        all_values_finite=bool(np.all(np.isfinite(vector))),
        maximum_ptdf_angle_flow_difference=_maximum(ptdf_angle_differences),
    )
