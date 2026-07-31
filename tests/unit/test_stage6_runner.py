from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from gpu_dcopf_hpr.hpr_generic import HPRState
from scripts.run_stage_6 import (
    DEFAULT_CONFIG,
    _audit_solver_transfers,
    _clean_json,
    _compare_arrays,
    _compare_states,
    _ledger_delta,
    _timing_delta,
    _validate_stage6_config,
)


def test_array_and_state_comparisons_apply_only_declared_tolerances() -> None:
    reference = np.array([1.0, -2.0, 3.0])
    within = reference + np.array([1e-13, -1e-13, 2e-13])
    comparison = _compare_arrays(
        within,
        reference,
        relative_tolerance=5e-13,
        absolute_tolerance=5e-13,
    )

    assert comparison["passed"]
    assert comparison["shape_matches"]
    assert comparison["finite"]

    candidate_state = HPRState(y=within, z=within, x=within)
    reference_state = HPRState(y=reference, z=reference, x=reference)
    state = _compare_states(
        candidate_state,
        reference_state,
        relative_tolerance=5e-13,
        absolute_tolerance=5e-13,
    )

    assert state["passed"]
    assert set(state["blocks"]) == {"x", "y", "z"}


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        ([1.0, 2.0], [1.0]),
        ([1.0, np.nan], [1.0, 2.0]),
        ([1.0, np.inf], [1.0, 2.0]),
    ],
)
def test_array_comparison_rejects_shape_and_nonfinite_values(
    candidate: list[float],
    reference: list[float],
) -> None:
    result = _compare_arrays(
        candidate,
        reference,
        relative_tolerance=None,
        absolute_tolerance=None,
    )

    assert not result["passed"]


def test_transfer_delta_and_residency_audit_are_phase_explicit() -> None:
    before = {
        "records": [
            {
                "phase": "matrix_setup",
                "direction": "host_to_device",
                "kind": "sparse_data",
                "calls": 1,
                "bytes": 32,
            }
        ]
    }
    after = {
        "records": [
            *before["records"],
            {
                "phase": "initial_state",
                "direction": "host_to_device",
                "kind": "vector",
                "calls": 3,
                "bytes": 96,
            },
            {
                "phase": "periodic_diagnostics",
                "direction": "device_to_host",
                "kind": "vector",
                "calls": 4,
                "bytes": 320,
            },
        ]
    }

    delta = _ledger_delta(before, after)
    audit = _audit_solver_transfers(delta)

    assert delta["totals"] == {
        "host_to_device": {"calls": 3, "bytes": 96},
        "device_to_host": {"calls": 4, "bytes": 320},
    }
    assert audit["passed"]

    after["records"].append(
        {
            "phase": "hidden_state_copy",
            "direction": "device_to_host",
            "kind": "vector",
            "calls": 1,
            "bytes": 1_024,
        }
    )
    rejected = _audit_solver_transfers(_ledger_delta(before, after))
    assert not rejected["passed"]
    assert rejected["unexpected_records"][0]["phase"] == "hidden_state_copy"


def test_monotone_transfer_summaries_are_required() -> None:
    assert _timing_delta(
        {"host_to_device_seconds": 1.0},
        {"host_to_device_seconds": 1.25},
    ) == {"host_to_device_seconds": 0.25}
    with pytest.raises(ValueError, match="cumulative and monotone"):
        _timing_delta(
            {"host_to_device_seconds": 1.0},
            {"host_to_device_seconds": 0.5},
        )


def test_versioned_stage6_config_preserves_stage7_and_precision_gates() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert _validate_stage6_config(config) == []

    invalid = copy.deepcopy(config)
    invalid["stage_boundary"]["stage_7_benchmarks_locked"] = False
    invalid["stage_boundary"]["gpu_speedup_claimed"] = True
    invalid["precision_study"]["mixed_precision_enabled"] = True
    errors = _validate_stage6_config(invalid)

    assert "Stage 7 benchmarks must remain locked" in errors
    assert "GPU speedup must remain unclaimed" in errors
    assert "mixed precision must remain disabled" in errors


def test_json_cleaner_preserves_nonfinite_failures_as_explicit_strings() -> None:
    result = _clean_json(
        {
            "nan": np.float64(np.nan),
            "positive": np.float64(np.inf),
            "negative": -np.inf,
        }
    )

    assert result == {
        "nan": "NaN",
        "positive": "Infinity",
        "negative": "-Infinity",
    }
    json.dumps(result, allow_nan=False)
