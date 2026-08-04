from __future__ import annotations

import copy
import json

import scripts.check_stage_8 as checker
import scripts.run_stage_8 as stage8


def _base_config() -> dict:
    return json.loads(
        (stage8.PROJECT_ROOT / "configs/benchmarks/stage_7_small_medium.json").read_text(
            encoding="utf-8"
        )
    )


def _base_evidence() -> dict:
    return json.loads(
        (stage8.PROJECT_ROOT / "results/raw/stage_7/stage_7_validation.json").read_text(
            encoding="utf-8"
        )
    )


def _passing_gate(projected: int = 100) -> dict:
    observation = {
        "host": {"available_bytes": 1_000, "errors": []},
        "device": {"device_name": "NVIDIA GB10"},
        "device_total_bytes": 1_000,
        "device_free_bytes": 1_000,
        "errors": [],
        "passed": True,
    }
    return {
        "observation": observation,
        "host_safety_budget_bytes": 800,
        "device_safety_budget_bytes": 800,
        "observed_device_total_bytes": 1_000,
        "observed_device_free_bytes": 1_000,
        "projected_unified_peak_bytes": projected,
        "checks": {
            "observation_available": True,
            "dimensions_match_table": True,
            "signed_int32_csr_supported": True,
            "within_host_safety_budget": True,
            "within_device_safety_budget": True,
        },
        "block_reasons": [],
        "passed": True,
        "evaluated_before_full_lp_allocation": True,
    }


def test_checker_recomputes_all_18_resource_rows() -> None:
    base = _base_evidence()
    evidence = {"resource_ledger": stage8._resource_ledger(base)}

    assert checker._resource_ledger_valid(evidence, base)
    evidence["resource_ledger"][6]["resource_estimate"]["projected_device_bytes"] += 1
    assert not checker._resource_ledger_valid(evidence, base)


def test_checker_rejects_out_of_order_or_multi_case_invocation() -> None:
    evidence = {
        "allocation_history": [
            {
                "key": "case2868rte:T64",
                "sequence": 2,
                "invocation_id": "i1",
                "preallocation_gate_passed": True,
            }
        ],
        "invocations": [
            {
                "id": "i1",
                "mode": "run_next",
                "approval_token_matched": True,
                "allocated_keys": ["case2868rte:T64"],
            }
        ],
        "stage_boundary": {
            "stage_8_allocation_attempt_count": 1,
            "unique_allocated_keys": ["case2868rte:T64"],
            "reconciliation_only_allocation_count": 0,
            "stage_9_allocation_count": 0,
        },
    }

    valid, _ = checker._allocation_order_valid(evidence)
    assert not valid
    evidence["allocation_history"][0]["key"] = "case2868rte:T48"
    evidence["allocation_history"][0]["sequence"] = 1
    evidence["invocations"][0]["allocated_keys"] = ["case2868rte:T48", "case2868rte:T64"]
    evidence["stage_boundary"]["unique_allocated_keys"] = ["case2868rte:T48"]
    valid, _ = checker._allocation_order_valid(evidence)
    assert not valid


def test_checker_accepts_honest_preallocation_block_and_rejects_allocation() -> None:
    gate = _passing_gate(projected=900)
    gate["checks"]["within_host_safety_budget"] = False
    gate["checks"]["within_device_safety_budget"] = False
    gate["block_reasons"] = ["within_host_safety_budget", "within_device_safety_budget"]
    gate["passed"] = False
    case = {
        "key": "case9241pegase:T24",
        "status": "MEMORY_BLOCKED",
        "passed": False,
        "full_lp_allocation_attempted": False,
        "stage8_resource_gate": gate,
        "failure": {
            "type": "MemorySafetyBlock",
            "full_lp_allocated": False,
        },
    }

    assert checker._blocked_case_valid(case)
    case["full_lp_allocation_attempted"] = True
    assert not checker._blocked_case_valid(case)


def test_checker_reuses_deep_stage7_numerical_and_timing_validation() -> None:
    evidence = _base_evidence()
    accepted = next(case for case in evidence["cases"] if case["key"] == "case2868rte:T16")
    case = copy.deepcopy(accepted)
    case["full_lp_allocation_attempted"] = True
    case["stage8_resource_gate"] = _passing_gate()

    assert checker._successful_case_valid(
        case,
        base_config=_base_config(),
        expected_nnz=76_354_336,
        gurobi_available=False,
    )
    case["solver_tracks"]["gpu_fp64_sgs_hpr"]["correctness"]["candidate"]["residuals"][
        "kkt_combined_norm"
    ] = 1.0
    assert not checker._successful_case_valid(
        case,
        base_config=_base_config(),
        expected_nnz=76_354_336,
        gurobi_available=False,
    )


def test_checker_campaign_rejects_bypassed_failed_predecessor() -> None:
    evidence = {
        "cases": [
            {
                "key": "case2868rte:T48",
                "status": "FAIL",
                "passed": False,
                "full_lp_allocation_attempted": True,
                "stage8_resource_gate": _passing_gate(),
                "failure": {"type": "RecordedSolverFailure"},
            },
            {"key": "case2868rte:T64", "status": "PASS"},
        ],
        "solver_availability": {"gurobi": {"available": False}},
    }

    valid, detail = checker._campaign_valid(evidence, _base_config(), _base_evidence())

    assert not valid
    assert "bypasses" in detail


def test_checker_rejects_plan_and_partial_as_stage8_acceptance() -> None:
    for status in ("PLANNED", "PARTIAL_PASS"):
        evidence = {
            "status": status,
            "all_passed": False,
            "completed_utc": "2026-08-03T00:00:00+00:00",
            "cases": [],
            "stage_boundary": {
                "stage_8_complete": False,
                "next_authorized_key": "case2868rte:T48",
                "passing_prefix_length": 0,
            },
        }

        valid, detail = checker._terminal_campaign_valid(evidence)

        assert not valid
        assert "nonterminal" in detail


def test_checker_accepts_honest_terminal_resource_stop_semantics() -> None:
    evidence = {
        "status": "COMPLETE_WITH_RESOURCE_LIMIT",
        "all_passed": False,
        "completed_utc": "2026-08-03T00:00:00+00:00",
        "cases": [
            {
                "key": "case2868rte:T48",
                "status": "MEMORY_BLOCKED",
                "passed": False,
            }
        ],
        "stage_boundary": {
            "stage_8_complete": True,
            "next_authorized_key": None,
            "passing_prefix_length": 0,
        },
    }

    valid, detail = checker._terminal_campaign_valid(evidence)

    assert valid, detail


def test_checker_main_writes_json_output(monkeypatch, tmp_path) -> None:
    result = {
        "checker_status": "PASS",
        "all_passed": True,
        "campaign_status": "STOPPED_ON_FAILURE",
        "checks": [],
    }
    monkeypatch.setattr(checker, "run_checks", lambda evidence, config: result)
    output = tmp_path / "stage_8_checks.json"

    exit_code = checker.main(
        [
            "--evidence",
            str(tmp_path / "evidence.json"),
            "--config",
            str(tmp_path / "config.json"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == result
