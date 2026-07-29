from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gpu_dcopf_hpr.network_data import (
    MATPOWERCaseError,
    NetworkCase,
    load_matpower_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE5 = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"


def test_public_matpower_case5_is_parsed_with_provenance_and_linear_costs() -> None:
    case = load_matpower_case(CASE5)

    assert case.name == "case5"
    assert case.base_mva == 100.0
    assert case.bus_ids == (1, 2, 3, 4, 5)
    assert case.reference_bus_id == 4
    assert len(case.buses) == 5
    assert len(case.active_generators) == 5
    assert len(case.active_branches) == 6
    assert np.sum(case.demand_mw) == 1000.0
    assert len(case.source_sha256) == 64

    linear_terms = [generator.cost.paper_linear_terms() for generator in case.generators]
    assert linear_terms == [
        (14.0, 0.0, ()),
        (15.0, 0.0, ()),
        (30.0, 0.0, ()),
        (40.0, 0.0, ()),
        (10.0, 0.0, ()),
    ]


def test_inactive_elements_are_retained_but_excluded_from_active_views() -> None:
    case = load_matpower_case(CASE5)
    modified = NetworkCase(
        name=case.name,
        base_mva=case.base_mva,
        buses=case.buses,
        generators=(replace(case.generators[0], status=False), *case.generators[1:]),
        branches=(replace(case.branches[0], status=False), *case.branches[1:]),
        source_path=case.source_path,
        source_sha256=case.source_sha256,
    )

    assert len(modified.generators) == 5
    assert len(modified.active_generators) == 4
    assert len(modified.branches) == 6
    assert len(modified.active_branches) == 5


def test_exactly_one_reference_bus_is_required() -> None:
    case = load_matpower_case(CASE5)
    buses = tuple(replace(bus, bus_type=1) for bus in case.buses)

    with pytest.raises(MATPOWERCaseError, match="Exactly one"):
        NetworkCase(
            name=case.name,
            base_mva=case.base_mva,
            buses=buses,
            generators=case.generators,
            branches=case.branches,
            source_path=case.source_path,
            source_sha256=case.source_sha256,
        )


def test_parser_does_not_evaluate_arbitrary_matlab(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.m"
    source = CASE5.read_text(encoding="utf-8")
    path.write_text(
        source.replace("mpc.baseMVA = 100;", "mpc.baseMVA = system('whoami');"),
        encoding="utf-8",
    )

    with pytest.raises(MATPOWERCaseError, match="baseMVA"):
        load_matpower_case(path)
