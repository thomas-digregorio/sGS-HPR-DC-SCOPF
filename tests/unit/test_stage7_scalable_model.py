from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import gpu_dcopf_hpr.stage7_scalable_model as stage7_module
from gpu_dcopf_hpr.network_data import (
    Branch,
    Bus,
    Generator,
    GeneratorCost,
    NetworkCase,
    load_matpower_case,
)
from gpu_dcopf_hpr.stage7_scalable_model import (
    FROZEN_STAGE7_POLICY,
    PAPER_CASE_SPECS,
    TABLE_II_ROWS,
    CompactRowMetadata,
    Stage7CaseSpec,
    all_stage7_preflights,
    all_stage7_symbolic_ledgers,
    assert_stage7_reconstruction_contract,
    build_stage7_scalable_model,
    normalize_stage7_case,
    reconstruct_stage7_resources,
    stage7_preflight,
    validate_stage7_physical,
)

ROOT = Path(__file__).resolve().parents[2]


def _cost(slope: float) -> GeneratorCost:
    return GeneratorCost(model=2, startup=0.0, shutdown=0.0, parameters=(slope, 0.0))


def _generator(
    index: int,
    bus_id: int,
    *,
    status: bool,
    maximum_mw: float,
) -> Generator:
    return Generator(
        index=index,
        bus_id=bus_id,
        initial_output_mw=0.0,
        status=status,
        maximum_mw=maximum_mw,
        minimum_mw=0.0,
        ramp_agc_mw_per_minute=0.0,
        ramp_10_mw=0.0,
        ramp_30_mw=0.0,
        cost=_cost(10.0 + index),
    )


def _branch(
    index: int,
    from_bus: int,
    to_bus: int,
    *,
    rate_a_mw: float,
    phase_shift_degrees: float = 0.0,
    angle_minimum_degrees: float = -360.0,
    angle_maximum_degrees: float = 360.0,
) -> Branch:
    return Branch(
        index=index,
        from_bus=from_bus,
        to_bus=to_bus,
        resistance_pu=0.0,
        reactance_pu=0.1 + 0.01 * index,
        line_charging_pu=0.0,
        rate_a_mw=rate_a_mw,
        tap_ratio=0.0,
        phase_shift_degrees=phase_shift_degrees,
        status=True,
        angle_minimum_degrees=angle_minimum_degrees,
        angle_maximum_degrees=angle_maximum_degrees,
    )


def _tiny_case() -> tuple[NetworkCase, Stage7CaseSpec]:
    network = NetworkCase(
        name="tiny_stage7",
        base_mva=100.0,
        buses=(
            Bus(1, 3, 50.0, 0.0),
            Bus(2, 1, 60.0, 0.0),
            Bus(3, 1, 40.0, 0.0),
        ),
        generators=(
            _generator(0, 1, status=True, maximum_mw=200.0),
            _generator(1, 2, status=False, maximum_mw=100.0),
            _generator(2, 3, status=True, maximum_mw=150.0),
        ),
        branches=(
            _branch(0, 1, 2, rate_a_mw=1_000.0, phase_shift_degrees=3.0),
            _branch(
                1,
                2,
                3,
                rate_a_mw=0.0,
                angle_minimum_degrees=-30.0,
                angle_maximum_degrees=30.0,
            ),
            _branch(2, 3, 1, rate_a_mw=1_000.0),
        ),
        source_path="synthetic://tiny-stage7",
        source_sha256="1" * 64,
    )
    spec = Stage7CaseSpec("tiny_stage7", 3, 3, 3, 2, 1, (2,))
    return network, spec


def test_frozen_json_reconstruction_contract_matches_code() -> None:
    raw = json.loads(
        (ROOT / "configs" / "benchmarks" / "stage_7_small_medium.json").read_text(encoding="utf-8")
    )
    assert_stage7_reconstruction_contract(raw["reconstruction_protocol"])

    mutated = dict(raw["reconstruction_protocol"])
    mutated["ptdf_zero_atol"] = 1.0e-9
    with pytest.raises(ValueError, match="ptdf_zero_atol"):
        assert_stage7_reconstruction_contract(mutated)


def test_all_table_ii_preflights_match_dimensions_and_label_bounded_nnz() -> None:
    preflights = all_stage7_preflights()
    assert len(preflights) == len(TABLE_II_ROWS) == 18
    assert all(preflight.dimensions_match_table for preflight in preflights)
    assert all(
        preflight.dense_structural_nnz_upper_bound >= preflight.row.published_nnz
        for preflight in preflights
    )
    assert all(
        preflight.as_dict()["reconstructed_nnz_kind"]
        == "dense_structural_upper_bound_without_public_case_ptdf_factorization"
        for preflight in preflights
    )
    assert all(not preflight.as_dict()["full_lp_allocated"] for preflight in preflights)


def test_preflight_catches_signed_int32_nnz_boundary_before_stage8() -> None:
    t16 = stage7_preflight("case9241pegase", 16)
    t24 = stage7_preflight("case9241pegase", 24)
    t32 = stage7_preflight("case9241pegase", 32)
    assert t16.csr32_supported
    assert not t24.csr32_supported
    assert not t32.csr32_supported
    assert not t24.fits_dgx_planning_budget
    assert not t32.fits_dgx_planning_budget


def test_all_18_exact_symbolic_counts_without_full_horizon_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_expanded_builder(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"expanded LP builder called with {args!r}, {kwargs!r}")

    monkeypatch.setattr(stage7_module, "build_stage7_scalable_model", forbidden_expanded_builder)
    networks = {
        case_name: load_matpower_case(
            ROOT / "data" / "raw" / "matpower" / "stage7" / f"{case_name}.m"
        )
        for case_name in PAPER_CASE_SPECS
    }
    ledgers = all_stage7_symbolic_ledgers(networks)
    expected = [
        4_799_808,
        19_228_464,
        57_896_368,
        116_420_464,
        19_073_056,
        76_354_336,
        229_507_104,
        267_886_816,
        306_303_136,
        344_756_064,
        383_245_600,
        421_771_744,
        460_334_496,
        342_863_272,
        514_308_838,
        1_371_647_068,
        2_057_650_132,
        2_743_770_956,
    ]
    assert len(ledgers) == len(expected) == 18
    assert [ledger.reconstructed_nnz for ledger in ledgers] == expected
    assert all(not ledger.full_lp_allocated for ledger in ledgers)
    assert all(ledger.stage_8_large_allocation_locked for ledger in ledgers)
    assert all(not ledger.matches_paper for ledger in ledgers)
    assert all(
        ledger.count_kind == "exact_reconstruction_selected_bus_batched_sparse_ptdf_support"
        for ledger in ledgers
    )


@pytest.mark.parametrize(
    ("case_name", "offline", "zero_rate", "ignored_angles"),
    (
        ("case1354pegase", 0, 559, 0),
        ("case2868rte", 38, 1_527, 3_808),
    ),
)
def test_public_matpower_normalization_preserves_paper_population(
    case_name: str,
    offline: int,
    zero_rate: int,
    ignored_angles: int,
) -> None:
    path = ROOT / "data" / "raw" / "matpower" / "stage7" / f"{case_name}.m"
    normalized = normalize_stage7_case(load_matpower_case(path))
    spec = PAPER_CASE_SPECS[case_name]
    assert len(normalized.generator_ids) == spec.generators
    assert len(normalized.branch_ids) == spec.branches
    assert normalized.offline_generator_count == offline
    assert normalized.zero_rate_branch_count == zero_rate
    assert normalized.ignored_angle_limit_count == ignored_angles
    assert np.all(normalized.generator_lower_mw[~normalized.generator_online] == 0.0)
    assert np.all(normalized.generator_upper_mw[~normalized.generator_online] == 0.0)


def test_resource_reconstruction_is_deterministic_and_matches_frozen_totals() -> None:
    network, spec = _tiny_case()
    normalized = normalize_stage7_case(network, spec=spec)
    first = reconstruct_stage7_resources(normalized, 2)
    second = reconstruct_stage7_resources(normalized, 2)
    base_load = float(np.sum(network.demand_mw))
    assert first.policy_fingerprint == FROZEN_STAGE7_POLICY.fingerprint
    assert first.renewable_ids == second.renewable_ids
    np.testing.assert_array_equal(first.renewable_bus_positions, [0, 1])
    np.testing.assert_array_equal(first.storage_bus_positions, [0])
    assert np.sum(first.renewable_maximum_mw[0]) == pytest.approx(0.10 * base_load)
    assert np.sum(first.storage_maximum_charge_mw) == pytest.approx(0.01 * base_load)
    np.testing.assert_allclose(first.reserve_up_mw, 0.01 * base_load)
    np.testing.assert_allclose(first.reserve_down_mw, 0.01 * base_load)
    np.testing.assert_allclose(first.generator_ramp_up_mw, [20.0, 0.0, 15.0])


def test_vectorized_builder_compact_metadata_and_redundant_zero_rate_proof() -> None:
    network, spec = _tiny_case()
    model = build_stage7_scalable_model(network, 2, spec=spec)
    assert model.expected_dimensions() == {"n": 26, "m1": 3, "m2": 38, "m": 41}
    assert model.dimension_summary()["offline_generators_fixed_zero"] == 1
    assert isinstance(model.equality_rows, CompactRowMetadata)
    assert isinstance(model.inequality_rows, CompactRowMetadata)
    assert model.equality_rows.block_count == 2
    assert model.inequality_rows.block_count == 10
    assert model.inequality_rows[0].family == "line_flow"
    assert model.inequality_rows[-1].family == "storage_energy"

    offline_pg = [model.variables.index("p_g", period, "gen_2") for period in range(2)]
    assert np.all(model.lp.lower[offline_pg] == 0.0)
    assert np.all(model.lp.upper[offline_pg] == 0.0)
    zero_row = 1
    assert model.line_limits_mw[zero_row] > model.derived_line_limit_proof_mw[zero_row]
    assert np.isnan(model.derived_line_limit_proof_mw[0])


def test_batched_sparse_physical_validator_checks_equations_1_through_10() -> None:
    network, spec = _tiny_case()
    model = build_stage7_scalable_model(network, 2, spec=spec)
    x = np.zeros(model.lp.n, dtype=np.float64)
    for period in range(2):
        x[model.variables.index("p_g", period, "gen_1")] = 75.0
        x[model.variables.index("p_g", period, "gen_3")] = 75.0
        x[model.variables.index("r_up", period, "gen_1")] = 1.5
        x[model.variables.index("r_down", period, "gen_1")] = 1.5

    validation = validate_stage7_physical(model, x)
    assert validation.maximum_violation <= 1.0e-12
    assert validation.angle_vs_compressed_ptdf_flow_max_abs_mw <= 1.0e-10
    assert validation.sparse_factorization_reused
    assert validation.batched_periods

    unbalanced = x.copy()
    unbalanced[model.variables.index("p_g", 0, "gen_1")] += 2.0
    failed = validate_stage7_physical(model, unbalanced)
    assert failed.equation_1_power_balance_mw == pytest.approx(2.0)
    assert failed.maximum_violation >= 2.0
