from dataclasses import replace

import numpy as np
import pytest

from gpu_dcopf_hpr.network_data import Branch, Bus, MATPOWERCaseError, NetworkCase
from gpu_dcopf_hpr.ptdf import build_ptdf


def branch(
    index: int,
    from_bus: int,
    to_bus: int,
    reactance: float,
    *,
    rate: float = 100.0,
    tap: float = 0.0,
    shift: float = 0.0,
    status: bool = True,
) -> Branch:
    return Branch(
        index=index,
        from_bus=from_bus,
        to_bus=to_bus,
        resistance_pu=0.0,
        reactance_pu=reactance,
        line_charging_pu=0.0,
        rate_a_mw=rate,
        tap_ratio=tap,
        phase_shift_degrees=shift,
        status=status,
        angle_minimum_degrees=-360.0,
        angle_maximum_degrees=360.0,
    )


def network(
    buses: tuple[Bus, ...],
    branches: tuple[Branch, ...],
) -> NetworkCase:
    return NetworkCase(
        name="test",
        base_mva=100.0,
        buses=buses,
        generators=(),
        branches=branches,
        source_path="unit-test",
        source_sha256="0" * 64,
    )


def test_two_bus_ptdf_matches_direct_angle_flow_and_requires_balance() -> None:
    case = network(
        (Bus(1, 3, 0.0, 0.0), Bus(2, 1, 0.0, 0.0)),
        (branch(0, 1, 2, 0.1),),
    )
    ptdf = build_ptdf(case)

    np.testing.assert_allclose(ptdf.matrix, [[0.0, -1.0]], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(ptdf.flow_offset_mw, [0.0], rtol=0.0, atol=1e-14)
    injection = np.array([50.0, -50.0])
    ptdf_flow = ptdf.flows_from_injections(injection)
    angles, angle_flow = ptdf.angles_and_flows(injection)
    np.testing.assert_allclose(ptdf_flow, [50.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(angle_flow, ptdf_flow, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(angles, [0.0, -0.05], rtol=0.0, atol=1e-12)

    with pytest.raises(ValueError, match="balanced"):
        ptdf.flows_from_injections([50.0, -49.0])


def test_tap_and_phase_shift_are_preserved_as_an_affine_flow_offset() -> None:
    case = network(
        (
            Bus(30, 3, 0.0, 0.0),
            Bus(10, 1, 0.0, 0.0),
            Bus(20, 1, 90.0, 0.0),
        ),
        (
            branch(0, 30, 10, 0.1, tap=2.0, shift=10.0),
            branch(1, 10, 20, 0.2),
            branch(2, 30, 20, 0.2, rate=80.0),
        ),
    )
    ptdf = build_ptdf(case)
    expected = np.array(
        [
            [0.0, -2.0 / 3.0, -1.0 / 3.0],
            [0.0, 1.0 / 3.0, -1.0 / 3.0],
            [0.0, -1.0 / 3.0, -2.0 / 3.0],
        ]
    )
    offset_magnitude = 250.0 * np.pi / 27.0
    np.testing.assert_allclose(ptdf.matrix, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        ptdf.flow_offset_mw,
        [-offset_magnitude, -offset_magnitude, offset_magnitude],
        rtol=0.0,
        atol=1e-12,
    )

    injection = np.array([76.3667687001418, 0.0, 13.6332312998582 - 90.0])
    flows = ptdf.flows_from_injections(injection)
    _, angle_flows = ptdf.angles_and_flows(injection)
    np.testing.assert_allclose(flows, angle_flows, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        flows,
        [-3.6332312998582, -3.6332312998582, 80.0],
        rtol=0.0,
        atol=1e-10,
    )


def test_parallel_branches_share_flow_and_inactive_branch_is_ignored() -> None:
    case = network(
        (Bus(1, 3, 0.0, 0.0), Bus(2, 1, 0.0, 0.0)),
        (
            branch(0, 1, 2, 0.1),
            branch(1, 1, 2, 0.1),
            branch(2, 1, 2, 0.0, status=False),
        ),
    )
    ptdf = build_ptdf(case)

    assert ptdf.branch_ids == ("branch_1", "branch_2")
    np.testing.assert_allclose(
        ptdf.flows_from_injections([100.0, -100.0]),
        [50.0, 50.0],
        rtol=0.0,
        atol=1e-12,
    )


def test_disconnected_and_near_zero_reactance_cases_stop_clearly() -> None:
    disconnected = network(
        (
            Bus(1, 3, 0.0, 0.0),
            Bus(2, 1, 0.0, 0.0),
            Bus(3, 1, 0.0, 0.0),
        ),
        (branch(0, 1, 2, 0.1),),
    )
    with pytest.raises(MATPOWERCaseError, match="disconnected"):
        build_ptdf(disconnected)

    near_zero = network(
        (Bus(1, 3, 0.0, 0.0), Bus(2, 1, 0.0, 0.0)),
        (branch(0, 1, 2, 1e-12),),
    )
    with pytest.raises(MATPOWERCaseError, match="threshold"):
        build_ptdf(near_zero)


def test_reference_choice_does_not_change_balanced_physical_flows() -> None:
    case = network(
        (
            Bus(1, 3, 0.0, 0.0),
            Bus(2, 1, 0.0, 0.0),
            Bus(3, 1, 0.0, 0.0),
        ),
        (branch(0, 1, 2, 0.1), branch(1, 2, 3, 0.2)),
    )
    injection = np.array([70.0, -20.0, -50.0])

    default = build_ptdf(case)
    alternate = build_ptdf(case, reference_bus_id=2)
    np.testing.assert_allclose(
        default.flows_from_injections(injection),
        alternate.flows_from_injections(injection),
        rtol=0.0,
        atol=1e-12,
    )

    reversed_case = replace(
        case,
        branches=(replace(case.branches[0], from_bus=2, to_bus=1), case.branches[1]),
    )
    reversed_flows = build_ptdf(reversed_case).flows_from_injections(injection)
    original_flows = default.flows_from_injections(injection)
    np.testing.assert_allclose(reversed_flows[0], -original_flows[0], atol=1e-12)
