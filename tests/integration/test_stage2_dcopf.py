from pathlib import Path

import numpy as np
import pytest

from gpu_dcopf_hpr.dcopf_model import (
    DCOPFConfig,
    build_dcopf_model,
    load_dcopf_config,
)
from gpu_dcopf_hpr.network_data import (
    Branch,
    Bus,
    Generator,
    GeneratorCost,
    NetworkCase,
    load_matpower_case,
)
from gpu_dcopf_hpr.validation import solve_with_highs, validate_dcopf_solution

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
CONFIGS = (
    PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json",
    PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json",
)


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda path: path.stem)
def test_highs_solution_passes_independent_physical_validation(config_path: Path) -> None:
    network = load_matpower_case(CASE)
    config = load_dcopf_config(config_path, network)
    model = build_dcopf_model(network, config)
    solution = solve_with_highs(model.lp, tolerance=1e-7)
    validation = validate_dcopf_solution(model, solution.state.x)

    assert solution.status == 0
    assert solution.residuals.conditions.all_satisfied
    assert validation.passed
    assert validation.maximum_ptdf_angle_flow_difference <= 1e-10
    assert validation.objective_difference <= 1e-10
    assert all(family.passed for family in validation.families)


def analytic_two_bus_case() -> tuple[NetworkCase, DCOPFConfig]:
    buses = (
        Bus(bus_id=205, bus_type=1, demand_mw=95.0, shunt_conductance_mw=5.0),
        Bus(bus_id=101, bus_type=3, demand_mw=0.0, shunt_conductance_mw=0.0),
    )
    generators = (
        Generator(
            index=0,
            bus_id=205,
            initial_output_mw=40.0,
            status=True,
            maximum_mw=200.0,
            minimum_mw=0.0,
            ramp_agc_mw_per_minute=0.0,
            ramp_10_mw=0.0,
            ramp_30_mw=0.0,
            cost=GeneratorCost(2, 0.0, 0.0, (30.0, 7.0)),
        ),
        Generator(
            index=1,
            bus_id=101,
            initial_output_mw=60.0,
            status=True,
            maximum_mw=200.0,
            minimum_mw=0.0,
            ramp_agc_mw_per_minute=0.0,
            ramp_10_mw=0.0,
            ramp_30_mw=0.0,
            cost=GeneratorCost(2, 0.0, 0.0, (10.0, 5.0)),
        ),
    )
    branches = (
        Branch(
            index=0,
            from_bus=101,
            to_bus=205,
            resistance_pu=0.0,
            reactance_pu=0.1,
            line_charging_pu=0.0,
            rate_a_mw=60.0,
            tap_ratio=0.0,
            phase_shift_degrees=0.0,
            status=True,
            angle_minimum_degrees=-360.0,
            angle_maximum_degrees=360.0,
        ),
    )
    network = NetworkCase(
        name="analytic_two_bus",
        base_mva=100.0,
        buses=buses,
        generators=generators,
        branches=branches,
        source_path="analytic",
        source_sha256="0" * 64,
    )
    config = DCOPFConfig(
        name="analytic",
        classification="analytic unit reference",
        synthetic_extension=False,
        periods=1,
        interval_hours=1.0,
        load_multipliers=(1.0,),
        reserve_up_mw=(0.0,),
        reserve_down_mw=(0.0,),
        generator_ramp_up_mw_per_hour=(0.0, 0.0),
        generator_ramp_down_mw_per_hour=(0.0, 0.0),
        renewable_penalty_per_mwh=0.0,
        storage_loss_penalty_per_mwh=0.0,
        renewables=(),
        storage=(),
        cost_mode="matpower_exact_linear",
        notes=(),
    )
    return network, config


def test_analytic_congested_network_has_expected_dispatch_flow_and_cost() -> None:
    network, config = analytic_two_bus_case()
    model = build_dcopf_model(network, config)
    solution = solve_with_highs(model.lp)
    validation = validate_dcopf_solution(model, solution.state.x)
    blocks = model.unpack(solution.state.x)
    injection = model.bus_injections(solution.state.x, 0)
    flows = model.ptdf.flows_from_injections(injection)

    np.testing.assert_allclose(blocks["p_g"], [[40.0, 60.0]], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(flows, [60.0], rtol=0.0, atol=1e-9)
    assert np.isclose(model.objective(solution.state.x), 1812.0, rtol=0.0, atol=1e-8)
    assert validation.passed
