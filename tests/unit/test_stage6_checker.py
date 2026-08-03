from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from scripts import check_stage_6


def _checks_by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(check["name"]): check for check in result["checks"]}


def _mutated_evidence(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    evidence = json.loads(check_stage_6.DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    evidence_path = tmp_path / "stage_6_validation.json"
    trajectory_name = evidence["evidence_files"]["trajectories_and_policy_events"]
    shutil.copy2(
        check_stage_6.DEFAULT_EVIDENCE.parent / trajectory_name,
        tmp_path / trajectory_name,
    )
    return evidence, evidence_path


def _write_evidence(evidence: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_stage6_source_hashes_come_from_frozen_commit_not_current_glob(
    monkeypatch: Any,
) -> None:
    original_glob = Path.glob
    package_root = (check_stage_6.PROJECT_ROOT / "src" / "gpu_dcopf_hpr").resolve()

    def reject_current_package_scan(path: Path, pattern: str):
        if path.resolve() == package_root:
            raise AssertionError(f"current Stage 6 package must not be globbed: {pattern}")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", reject_current_package_scan)

    result = check_stage_6.run_checks()
    source_check = _checks_by_name(result)[
        "embedded_configuration_and_input_hashes_match_versioned_sources"
    ]

    assert source_check["passed"] is True


def test_checker_rejects_mutated_frozen_stage6_source_hash(tmp_path: Path) -> None:
    evidence, evidence_path = _mutated_evidence(tmp_path)
    evidence["inputs"]["source_files"][0]["sha256"] = "0" * 64
    _write_evidence(evidence, evidence_path)

    result = check_stage_6.run_checks(evidence_path, check_stage_6.DEFAULT_CONFIG)
    source_check = _checks_by_name(result)[
        "embedded_configuration_and_input_hashes_match_versioned_sources"
    ]

    assert result["passed"] is False
    assert source_check["passed"] is False
