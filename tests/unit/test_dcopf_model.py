from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from gpu_dcopf_hpr.dcopf_model import build_dcopf_model, load_dcopf_config
from gpu_dcopf_hpr.network_data import MATPOWERCaseError, load_matpower_case

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
BASE_CONFIG = PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json"
EXTENSION_CONFIG = PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json"


def test_base_case_dimensions_metadata_and_unconstrained_branch_semantics() -> None:
    network = load_matpower_case(CASE)
    config = load_dcopf_config(BASE_CONFIG, network)
    model = build_dcopf_model(network, config)

    assert sparse.isspmatrix_csr(model.lp.A1)
    assert sparse.isspmatrix_csr(model.lp.A2)
    assert model.expected_dimensions() == {"n": 15, "m1": 1, "m2": 16, "m": 17}
    assert model.dimension_summary() == {
        "periods": 1,
        "buses": 5,
        "generators": 5,
        "renewables": 0,
        "storage": 0,
        "active_topology_branches": 6,
        "thermally_constrained_branches": 2,
        "n": 15,
        "m1": 1,
        "m2": 16,
        "m": 17,
        "nnz_A1": 5,
        "nnz_A2": model.lp.A2.nnz,
    }
    assert len(model.ptdf.branch_ids) == 6
    assert [branch.branch_id for branch in model.constrained_branches] == [
        "branch_1",
        "branch_6",
    ]
    assert len(model.equality_rows) == model.lp.m1
    assert len(model.inequality_rows) == model.lp.m2
    assert {row.equation for row in model.inequality_rows} == {"2", "4", "5"}


def test_variable_index_is_bidirectional_block_then_period_major() -> None:
    network = load_matpower_case(CASE)
    config = load_dcopf_config(EXTENSION_CONFIG, network)
    model = build_dcopf_model(network, config)
    index = model.variables

    assert len(index) == 36
    assert index.index("p_g", 0, "gen_1") == 0
    assert index.index("p_g", 1, "gen_5") == 9
    assert index.index("p_rg", 0, "rg_bus_2") == 10
    assert index.index("p_rg", 1, "rg_bus_2") == 11
    assert index.index("p_ess_dc", 0, "ess_bus_3") == 12
    assert index.index("p_ess_ch", 1, "ess_bus_3") == 15
    assert index.index("r_up", 0, "gen_1") == 16
    assert index.index("r_down", 1, "gen_5") == 35
    for position, key in enumerate(index.keys):
        assert index.key(position) == key
        assert index.index(key.kind, key.period, key.device_id) == position


def test_synthetic_extension_exercises_every_paper_constraint_family() -> None:
    network = load_matpower_case(CASE)
    config = load_dcopf_config(EXTENSION_CONFIG, network)
    model = build_dcopf_model(network, config)

    assert model.config.synthetic_extension
    assert model.expected_dimensions() == {"n": 36, "m1": 3, "m2": 46, "m": 49}
    assert (model.lp.n, model.lp.m1, model.lp.m2, model.lp.m) == (36, 3, 46, 49)
    assert {row.equation for row in model.equality_rows} == {"1", "9"}
    assert {row.equation for row in model.inequality_rows} == {
        "2",
        "4",
        "5",
        "6",
        "8",
    }

    for row_index in range(0, 8, 2):
        np.testing.assert_allclose(
            model.lp.A2.getrow(row_index + 1).toarray(),
            -model.lp.A2.getrow(row_index).toarray(),
            rtol=0.0,
            atol=0.0,
        )
        branch = model.constrained_branches[(row_index // 2) % 2]
        assert np.isclose(
            model.lp.b2[row_index] + model.lp.b2[row_index + 1],
            -2.0 * branch.rate_a_mw,
        )


def test_quadratic_cost_is_rejected_instead_of_silently_linearized() -> None:
    network = load_matpower_case(CASE)
    config = load_dcopf_config(BASE_CONFIG, network)
    nonlinear_cost = replace(
        network.generators[0].cost,
        parameters=(0.1, 14.0, 0.0),
    )
    nonlinear_generator = replace(network.generators[0], cost=nonlinear_cost)
    nonlinear_network = replace(
        network,
        generators=(nonlinear_generator, *network.generators[1:]),
    )

    with pytest.raises(MATPOWERCaseError, match="higher-order"):
        build_dcopf_model(nonlinear_network, config)


def test_active_angle_limits_are_rejected_as_outside_the_paper_model() -> None:
    network = load_matpower_case(CASE)
    config = load_dcopf_config(BASE_CONFIG, network)
    limited_branch = replace(network.branches[0], angle_maximum_degrees=30.0)
    limited_network = replace(network, branches=(limited_branch, *network.branches[1:]))

    with pytest.raises(MATPOWERCaseError, match="angle limits"):
        build_dcopf_model(limited_network, config)
