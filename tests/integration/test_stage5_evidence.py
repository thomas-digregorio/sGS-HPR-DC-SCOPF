from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_stage_5 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_EVIDENCE,
    run_checks,
)


def _checks_by_name(result: dict[str, object]) -> dict[str, dict[str, object]]:
    checks = result["checks"]
    assert isinstance(checks, list)
    return {str(check["name"]): check for check in checks if isinstance(check, dict)}


def test_preserved_stage5_evidence_passes_independent_checker() -> None:
    result = run_checks(DEFAULT_EVIDENCE, DEFAULT_CONFIG)
    by_name = _checks_by_name(result)

    assert result["stage"] == 5
    assert result["passed"] is True
    assert result["summary"] == {"passed": len(by_name)}
    assert {
        "embedded_configuration_matches_versioned_config",
        "primary_sources_and_hpr_lp_pin_preserved",
        "published_formula_transfer_boundary_is_explicit",
        "dense_and_sparse_component_fixtures_present",
        "component_roundtrips_and_algebraic_identities_within_tolerance",
        "exact_ten_ruiz_then_one_pock_chambolle_order",
        "both_t1_and_t2_dcopf_cases_present",
        "dcopf_original_space_stopping_kkt_objective_and_physics",
        "all_four_control_combinations_present",
        "preprocessing_ablation_covers_unscaled_norm_ruiz_and_full",
        "adaptive_without_restart_is_explicitly_nonpaper_and_nongating",
        "initial_sigma_sensitivity_coverage_and_acceptance",
        "trajectory_gzip_is_valid_and_covers_every_recorded_run",
        "trajectory_rows_match_json_summaries_and_sampling_grid",
        "policy_events_match_json_counts_values_and_100_iteration_cadence",
        "stage_six_gpu_and_dgx_boundaries_remain_closed",
    }.issubset(by_name)
    assert all(check["passed"] is True for check in by_name.values())


def test_checker_rejects_a_paper_claim_for_adaptive_without_restart(
    tmp_path,
) -> None:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    base = next(
        case for case in evidence["dcopf_ablation"]["cases"] if case["name"] == "case5_base_t1"
    )
    adaptive_only = next(
        run
        for run in base["control_ablation"]
        if run["control"]["adaptive_sigma"] and not run["control"]["restart"]
    )
    adaptive_only["paper_algorithm_claim"] = True
    adaptive_only["interpretation"] = "Claimed as the paper algorithm."

    evidence_path = tmp_path / "stage_5_validation.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trajectory_name = evidence["evidence_files"]["trajectories_and_policy_events"]
    shutil.copy2(
        DEFAULT_EVIDENCE.parent / trajectory_name,
        tmp_path / trajectory_name,
    )

    result = run_checks(evidence_path, DEFAULT_CONFIG)
    by_name = _checks_by_name(result)

    assert result["passed"] is False
    assert (
        by_name["adaptive_without_restart_is_explicitly_nonpaper_and_nongating"]["passed"] is False
    )
