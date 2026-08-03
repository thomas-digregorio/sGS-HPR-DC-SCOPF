from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.run_stage_8 as stage8
from gpu_dcopf_hpr.stage7_scalable_model import all_stage7_preflights


def _config() -> dict:
    return json.loads(stage8.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _stage7_evidence() -> dict:
    return json.loads(
        (stage8.PROJECT_ROOT / "results/raw/stage_7/stage_7_validation.json").read_text(
            encoding="utf-8"
        )
    )


def _estimate(key: str) -> dict:
    exact = stage8._exact_nnz_by_key(_stage7_evidence())
    preflight = next(item for item in all_stage7_preflights() if stage8._key(item) == key)
    return stage8._resource_estimate(preflight, exact[key])


def test_frozen_config_preserves_stage7_contract_and_exact_order() -> None:
    config = _config()

    assert stage8._validate_stage8_config(config) == []
    assert tuple(config["campaign_order"]) == stage8.CAMPAIGN_ORDER
    assert tuple(config["reconciliation_only_rows"]) == stage8.RECONCILIATION_ONLY_ROWS
    canonical_lf = stage8.DEFAULT_CONFIG.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical_lf).hexdigest() == stage8.FROZEN_CONFIG_SHA256


def test_config_rejects_threshold_order_and_boundary_drift() -> None:
    invalid = copy.deepcopy(_config())
    invalid["resource_policy"]["host_safety_fraction"] = 0.95
    invalid["campaign_order"][0], invalid["campaign_order"][1] = (
        invalid["campaign_order"][1],
        invalid["campaign_order"][0],
    )
    invalid["stage_boundary"]["stage_9_locked"] = False

    errors = stage8._validate_stage8_config(invalid)

    assert "resource_policy drifted from the fail-closed memory contract" in errors
    assert "campaign_order differs from the approved Stage 8 sequence" in errors
    assert "stage_boundary drifted" in errors


def test_resource_ledger_covers_all_rows_and_never_authorizes_count_only_rows() -> None:
    ledger = stage8._resource_ledger(_stage7_evidence())
    indexed = {row["key"]: row for row in ledger}

    assert len(indexed) == 18
    assert {key for key, row in indexed.items() if row["disposition"] == "stage8_campaign"} == set(
        stage8.CAMPAIGN_ORDER
    )
    assert {
        key for key, row in indexed.items() if row["disposition"] == "reconciliation_only"
    } == set(stage8.RECONCILIATION_ONLY_ROWS)
    assert all(row["allocation_permitted_this_invocation"] is False for row in ledger)
    for key in stage8.STATIC_CSR32_BLOCKS:
        assert indexed[key]["static_preallocation_status"] == "BLOCKED"
        assert indexed[key]["resource_estimate"]["full_lp_allocated"] is False
        assert indexed[key]["resource_estimate"]["signed_int32_csr_supported"] is False


def test_resource_estimate_reports_every_requested_component() -> None:
    estimate = _estimate("case2868rte:T48")

    assert estimate["row_count"] == 493_583
    assert estimate["column_count"] == 113_856
    assert estimate["exact_reconstructed_nnz"] == 229_507_104
    assert estimate["csr_matrix_bytes"] > 0
    assert estimate["csr_transpose_bytes"] > 0
    assert estimate["iterate_and_workspace_vector_bytes"] > 0
    assert estimate["temporary_buffers_and_headroom_bytes"] > 0
    assert estimate["projected_unified_peak_bytes"] == (
        estimate["projected_host_assembly_peak_bytes"] + estimate["projected_device_bytes"]
    )


def test_unified_memory_gate_uses_current_free_memory_and_blocks_t16() -> None:
    config = _config()
    observation = {
        "host": {"available_bytes": 121_858_023_424, "errors": []},
        "device": {"device_name": "NVIDIA GB10"},
        "device_total_bytes": 130_663_165_952,
        "device_free_bytes": 110_625_198_080,
        "errors": [],
        "passed": True,
    }

    t48 = stage8._resource_gate(_estimate("case2868rte:T48"), observation, config)
    t16 = stage8._resource_gate(_estimate("case9241pegase:T16"), observation, config)

    assert t48["passed"]
    assert not t16["passed"]
    assert not t16["checks"]["within_device_safety_budget"]
    assert t16["observed_device_free_bytes"] == 110_625_198_080


def test_resource_gate_fails_closed_when_observation_is_missing() -> None:
    observation = {
        "host": {"available_bytes": None, "errors": ["psutil unavailable"]},
        "device": {},
        "device_total_bytes": None,
        "device_free_bytes": None,
        "errors": ["device unavailable"],
        "passed": False,
    }

    result = stage8._resource_gate(_estimate("case2868rte:T48"), observation, _config())

    assert not result["passed"]
    assert result["checks"]["observation_available"] is False
    assert result["checks"]["within_host_safety_budget"] is False
    assert result["checks"]["within_device_safety_budget"] is False


def test_next_key_requires_a_strict_passing_prefix_and_explicit_retry() -> None:
    evidence: dict = {"cases": []}
    assert stage8._next_key(evidence, retry_failed=False) == "case2868rte:T48"
    evidence["cases"].append({"key": "case2868rte:T48", "status": "PASS"})
    assert stage8._next_key(evidence, retry_failed=False) == "case2868rte:T64"
    evidence["cases"].append({"key": "case2868rte:T64", "status": "FAIL"})
    assert stage8._next_key(evidence, retry_failed=False) is None
    assert stage8._next_key(evidence, retry_failed=True) == "case2868rte:T64"


def test_interrupted_case_requires_explicit_retry_and_preserves_full_snapshot() -> None:
    interrupted = {
        "key": "case2868rte:T48",
        "case_name": "case2868rte",
        "periods": 48,
        "run_fingerprint": "fingerprint",
        "status": "RUNNING",
        "passed": False,
        "construction": {"wall_seconds": 12.5},
        "solver_tracks": {"highs": {"correctness": {"status": "SUCCESS"}}},
    }
    evidence = {"cases": [interrupted]}

    assert stage8._next_key(evidence, retry_failed=False) is None
    assert stage8._next_key(evidence, retry_failed=True) == "case2868rte:T48"
    replacement = stage8._fresh_retry(interrupted, "fingerprint")
    snapshot = replacement["retry_history"][-1]["prior_case"]
    assert snapshot["construction"] == interrupted["construction"]
    assert snapshot["solver_tracks"] == interrupted["solver_tracks"]
    assert replacement["status"] == "PENDING"


def test_incompatible_resume_refuses_to_overwrite_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "partial.json"
    checkpoint.write_text('{"run_fingerprint":"old"}', encoding="utf-8")

    with pytest.raises(stage8.Stage8ContractError, match="refusing overwrite"):
        stage8._compatible_partial(checkpoint, "new")


def test_resume_validation_rejects_immutable_ledger_mutation() -> None:
    config = _config()
    ledger = [{"key": "case2868rte:T48", "allocation_permitted_this_invocation": False}]
    checkpoint = {
        "schema_version": "1.0",
        "stage": 8,
        "run_fingerprint": "fingerprint",
        "configuration": config,
        "base_stage_7_contract": {"passed": True},
        "source_manifest": [{"path": "source.py"}],
        "resource_ledger": copy.deepcopy(ledger),
        "cases": [],
        "allocation_history": [],
        "invocations": [],
        "stage_boundary": {
            "stage_8_allocation_attempt_count": 0,
            "unique_allocated_keys": [],
            "reconciliation_only_allocation_count": 0,
            "stage_9_allocation_count": 0,
        },
    }
    arguments = {
        "fingerprint": "fingerprint",
        "config": config,
        "base_contract": {"passed": True},
        "sources": [{"path": "source.py"}],
        "expected_ledger": ledger,
        "base_config": json.loads(
            (stage8.PROJECT_ROOT / "configs/benchmarks/stage_7_small_medium.json").read_text(
                encoding="utf-8"
            )
        ),
        "solver_availability": {"gurobi": {"available": False}},
    }

    assert stage8._validate_resume_checkpoint(checkpoint, **arguments) == []
    checkpoint["resource_ledger"][0]["key"] = "case9241pegase:T32"
    assert "checkpoint resource ledger contains an immutable-field mutation" in (
        stage8._validate_resume_checkpoint(checkpoint, **arguments)
    )


def test_resume_validation_rejects_forged_shallow_pass_predecessor() -> None:
    config = _config()
    base_config = json.loads(
        (stage8.PROJECT_ROOT / "configs/benchmarks/stage_7_small_medium.json").read_text(
            encoding="utf-8"
        )
    )
    ledger = stage8._resource_ledger(_stage7_evidence())
    checkpoint = {
        "schema_version": "1.0",
        "stage": 8,
        "run_fingerprint": "fingerprint",
        "configuration": config,
        "base_stage_7_contract": {"passed": True},
        "source_manifest": [{"path": "source.py"}],
        "resource_ledger": copy.deepcopy(ledger),
        "cases": [
            {
                "key": "case2868rte:T48",
                "status": "PASS",
                "passed": True,
                "run_fingerprint": "fingerprint",
            }
        ],
        "allocation_history": [
            {
                "key": "case2868rte:T48",
                "sequence": 1,
                "retry": False,
                "invocation_id": "run-1",
                "preallocation_gate_passed": True,
            }
        ],
        "invocations": [
            {
                "id": "run-1",
                "mode": "run_next",
                "approval_token_matched": True,
                "retry_failed": False,
                "allocated_keys": ["case2868rte:T48"],
            }
        ],
        "stage_boundary": {
            "stage_8_allocation_attempt_count": 1,
            "unique_allocated_keys": ["case2868rte:T48"],
            "reconciliation_only_allocation_count": 0,
            "stage_9_allocation_count": 0,
        },
    }

    errors = stage8._validate_resume_checkpoint(
        checkpoint,
        fingerprint="fingerprint",
        config=config,
        base_contract={"passed": True},
        sources=[{"path": "source.py"}],
        expected_ledger=ledger,
        base_config=base_config,
        solver_availability={"gurobi": {"available": False}},
    )

    assert any("failed deep numerical/timing validation" in error for error in errors)


def test_no_resume_refuses_existing_artifacts_without_modifying_them(tmp_path: Path) -> None:
    partial = tmp_path / stage8.PARTIAL_NAME
    final = tmp_path / stage8.FINAL_NAME
    partial.write_text("immutable partial", encoding="utf-8")
    final.write_text("immutable final", encoding="utf-8")

    result = stage8.main(["--plan-only", "--no-resume", "--output-dir", str(tmp_path)])

    assert result == 1
    assert partial.read_text(encoding="utf-8") == "immutable partial"
    assert final.read_text(encoding="utf-8") == "immutable final"


def test_initialization_failure_is_written_as_honest_evidence(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = stage8.main(["--plan-only", "--config", str(missing), "--output-dir", str(tmp_path)])
    evidence = json.loads((tmp_path / stage8.FINAL_NAME).read_text(encoding="utf-8"))

    assert result == 1
    assert evidence["status"] == "FAIL"
    assert evidence["initialization_completed"] is False
    assert evidence["stage_boundary"]["stage_9_allocation_count"] == 0


def test_initialization_failure_never_overwrites_resumable_evidence(tmp_path: Path) -> None:
    partial = tmp_path / stage8.PARTIAL_NAME
    partial.write_text("preserve me", encoding="utf-8")

    result = stage8.main(
        [
            "--plan-only",
            "--config",
            str(tmp_path / "temporarily-missing.json"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 1
    assert partial.read_text(encoding="utf-8") == "preserve me"
    assert not (tmp_path / stage8.FINAL_NAME).exists()


def test_output_lock_refuses_a_concurrent_campaign(tmp_path: Path) -> None:
    with stage8._exclusive_output_lock(tmp_path):
        with pytest.raises(stage8.Stage8ConcurrentRunError):
            with stage8._exclusive_output_lock(tmp_path):
                pytest.fail("second lock unexpectedly acquired")


def test_interrupted_checkpoint_is_exposed_as_retry_required() -> None:
    evidence = {
        "cases": [{"key": "case2868rte:T48", "status": "RUNNING", "passed": False}],
        "stage_boundary": {},
    }

    stage8._update_campaign_status(evidence)

    assert evidence["status"] == "RETRY_REQUIRED"
    assert evidence["stage_boundary"]["stage_8_complete"] is False
    assert evidence["stage_boundary"]["next_authorized_key"] is None
    assert evidence["stage_boundary"]["retry_required_key"] == "case2868rte:T48"


def test_canonical_git_blob_identity_treats_crlf_as_equivalent(tmp_path: Path) -> None:
    tracked = stage8.PROJECT_ROOT / "configs/benchmarks/stage_7_small_medium.json"
    crlf = tmp_path / "stage_7_small_medium.json"
    crlf.write_bytes(tracked.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    identity = stage8.stage7._canonical_git_blob_identity(
        Path("configs/benchmarks/stage_7_small_medium.json"), crlf
    )

    assert identity["passed"] is True
    assert identity["filtered_worktree_git_blob"] == identity["head_git_blob"]
    assert identity["canonical_git_blob_sha256"] == stage8.FROZEN_STAGE7_CONFIG_SHA256


def test_direct_script_entrypoint_can_import_src_without_editable_install() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_stage_8.py", "--help"],
        cwd=stage8.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_source_manifest_pins_the_deep_resume_validator() -> None:
    paths = {
        row["path"] for row in stage8._source_manifest(stage8.DEFAULT_CONFIG) if row.get("path")
    }

    assert "scripts/check_stage_7.py" in paths


def test_parser_has_no_arbitrary_case_selector() -> None:
    args = stage8.parse_args(["--plan-only"])

    assert args.plan_only
    with pytest.raises(SystemExit):
        stage8.parse_args(["--case", "case9241pegase:T32"])
