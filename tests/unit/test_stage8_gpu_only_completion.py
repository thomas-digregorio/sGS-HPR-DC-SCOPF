from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import scripts.check_stage_8_gpu_only_completion as checker
import scripts.run_stage_8 as stage8
import scripts.run_stage_8_gpu_only_completion as runner


def _config() -> dict:
    return json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _original_evidence() -> dict:
    return json.loads(
        (runner.PROJECT_ROOT / "results/raw/stage_8/stage_8_validation.json").read_text(
            encoding="utf-8"
        )
    )


def _base_evidence() -> dict:
    return json.loads(
        (runner.PROJECT_ROOT / "results/raw/stage_7/stage_7_validation.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_continuation_config_has_exact_scope_and_hash() -> None:
    config = _config()

    assert runner._validate_config(config) == []
    assert [row["key"] for row in config["requested_sequence"]] == list(runner.REQUESTED_KEYS)
    assert config["track_policy"]["required_solver_tracks"] == list(runner.REQUIRED_TRACKS)
    assert config["track_policy"]["explicitly_skipped_solver_tracks"] == list(runner.SKIPPED_TRACKS)
    canonical_lf = runner.DEFAULT_CONFIG.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical_lf).hexdigest() == runner.FROZEN_CONFIG_SHA256


def test_config_rejects_memory_track_and_stage9_drift() -> None:
    invalid = copy.deepcopy(_config())
    invalid["resource_policy"]["host_safety_fraction"] = 0.95
    invalid["track_policy"]["required_solver_tracks"].append("cpu_fp64_sgs_hpr")
    invalid["stage_boundary"]["stage_9_locked"] = False

    errors = runner._validate_config(invalid)

    assert "resource_policy drifted from the unchanged fail-closed contract" in errors
    assert "required solver tracks are not exactly HiGHS and GPU FP64 sGS-HPR" in errors
    assert "Stage 9 or scientific-claim boundary drifted" in errors


def test_resource_ledger_is_frozen_to_sequences_6_8() -> None:
    ledger = runner._resource_ledger(_original_evidence(), _base_evidence())
    indexed = {row["key"]: row for row in ledger}

    assert list(indexed) == list(runner.REQUESTED_KEYS)
    assert indexed["case9241pegase:T16"]["campaign_sequence"] == 6
    assert (
        indexed["case9241pegase:T16"]["resource_estimate"]["projected_unified_peak_bytes"]
        == 101_398_781_000
    )
    for key in runner.STATIC_BLOCK_KEYS:
        assert indexed[key]["static_preallocation_status"] == "BLOCKED"
        assert indexed[key]["resource_estimate"]["signed_int32_csr_supported"] is False
        assert indexed[key]["allocation_permitted_this_invocation"] is False


def test_static_sequence_7_8_cases_are_resolved_without_solver_or_allocation() -> None:
    ledger = runner._resource_ledger(_original_evidence(), _base_evidence())
    cases = [runner._static_case(row) for row in ledger if row["key"] in runner.STATIC_BLOCK_KEYS]
    evidence = {"cases": cases, "allocation_history": []}

    assert checker._static_cases_valid(evidence)
    assert all(case["status"] == "INDEX_BLOCKED" for case in cases)
    assert all(case["solver_tracks"] == {} for case in cases)
    assert all(case["full_lp_allocation_attempted"] is False for case in cases)


def test_t16_unchanged_unified_memory_gate_can_complete_as_memory_blocked() -> None:
    config = _config()
    ledger = runner._resource_ledger(_original_evidence(), _base_evidence())
    estimate = ledger[0]["resource_estimate"]
    observation = {
        "host": {"available_bytes": 120_076_537_856, "errors": []},
        "device": {"device_name": "NVIDIA GB10"},
        "device_total_bytes": 130_663_165_952,
        "device_free_bytes": 120_076_537_856,
        "errors": [],
        "passed": True,
    }
    gate = stage8._resource_gate(estimate, observation, config)
    t16 = {
        "key": runner.REQUESTED_KEYS[0],
        "status": "MEMORY_BLOCKED",
        "passed": False,
        "resolved": True,
        "full_lp_allocation_attempted": False,
        "resource_estimate": estimate,
        "stage8_resource_gate": gate,
        "solver_tracks": {},
        "required_solver_track_disposition": {
            name: "NOT_RUN_MEMORY_SAFETY_BLOCK" for name in runner.REQUIRED_TRACKS
        },
        "skipped_solver_tracks": runner._skip_records(resource_blocked=True),
        "failure": {"type": "MemorySafetyBlock", "full_lp_allocated": False},
    }
    static = [runner._static_case(row) for row in ledger[1:]]
    evidence = {
        "cases": [t16, *static],
        "stage_boundary": {"stage_9_locked": True, "stage_9_allocation_count": 0},
    }

    assert gate["passed"] is False
    assert gate["checks"]["within_host_safety_budget"] is False
    runner._update_status(evidence)
    assert evidence["status"] == "COMPLETE_WITH_RESOURCE_LIMITS"
    assert evidence["all_requested_rows_resolved"] is True
    assert evidence["stage_boundary"]["stage_9_locked"] is True


def test_allocation_audit_rejects_any_cpu_or_gurobi_call() -> None:
    evidence = {
        "allocation_history": [],
        "invocations": [
            {
                "allocated_keys": [],
                "cpu_hpr_called": False,
                "gurobi_called": False,
            }
        ],
        "stage_boundary": {
            "continuation_allocation_attempt_count": 0,
            "unique_allocated_keys": [],
        },
    }

    assert checker._allocation_valid(evidence)[0]
    evidence["invocations"][0]["cpu_hpr_called"] = True
    assert not checker._allocation_valid(evidence)[0]


def test_checker_can_label_external_checkpoint_paths() -> None:
    external = Path("/home/dgxsparktd/stage8-runs/evidence.json")

    assert checker._display_path(external) == external.as_posix()
