"""Validate preserved Stage 3 evidence and the Stage 4 boundary."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sgs_hpr" / "stage_3_fixed_sigma.json"
DEFAULT_EVIDENCE = PROJECT_ROOT / "results" / "raw" / "stage_3" / "stage_3_validation.json"


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _trajectory_summary(path: Path) -> tuple[dict[str, int], bool]:
    last_iterations: dict[str, int] = {}
    complete_fields = True
    required = {
        "case",
        "iteration",
        "canonical_variable_objective",
        "objective_constant",
        "total_objective",
        "iteration_loop_elapsed_seconds",
        "paper_raw",
        "paper_normalized",
        "sigma",
        "restart_count",
    }
    if not path.is_file():
        return last_iterations, False
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            complete_fields = complete_fields and required.issubset(row)
            last_iterations[str(row["case"])] = int(row["iteration"])
    return last_iterations, complete_fields


def run_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    required_paths = [
        "configs/sgs_hpr/stage_3_fixed_sigma.json",
        "src/gpu_dcopf_hpr/sgs_hpr.py",
        "scripts/run_cpu_solver.py",
        "scripts/check_stage_3.py",
        "tests/unit/test_sgs_hpr.py",
        "tests/integration/test_stage3_sgs_hpr.py",
        "results/raw/stage_3/stage_3_validation.json",
        "results/raw/stage_3/sgs_hpr_trajectories.jsonl.gz",
        "logs/stage_3/commands_and_results.txt",
        "docs/stage_reports/stage_3_report.md",
        "docs/project_state.md",
    ]
    missing = [path for path in required_paths if not (PROJECT_ROOT / path).is_file()]
    add_check(
        checks,
        "required_stage_three_paths",
        not missing,
        "complete" if not missing else f"missing {missing}",
    )

    config = _load_json(DEFAULT_CONFIG)
    evidence = _load_json(DEFAULT_EVIDENCE)
    add_check(
        checks,
        "stage_three_validation_passed",
        evidence.get("all_passed") is True,
        str(evidence.get("all_passed", "unavailable")),
    )
    fixed_baseline = (
        config.get("sigma") == 1.0
        and config.get("paper_tolerance") == 5e-5
        and config.get("precision") == "FP64"
        and all(value is False for value in config.get("unsupported_features", {}).values())
    )
    add_check(
        checks,
        "fixed_sigma_cpu_fp64_baseline",
        fixed_baseline,
        "sigma=1, FP64, no restart/adaptation/scaling/structural/GPU path",
    )

    cases = evidence.get("cases", [])
    by_name = {case.get("name"): case for case in cases}
    expected_names = {
        "analytic_toy",
        "box_bound_active",
        "inequality_inactive",
        "planted_random",
        "case5_base_t1",
        "case5_synthetic_extension_t2",
    }
    add_check(
        checks,
        "all_reference_cases_present",
        set(by_name) == expected_names,
        f"cases={sorted(str(name) for name in by_name)}",
    )

    case_checks_pass = bool(cases) and all(
        case.get("passed") is True and all(case.get("checks", {}).values()) for case in cases
    )
    add_check(checks, "all_case_checks_pass", case_checks_pass, f"{len(cases)} cases")

    stopping_and_kkt = bool(cases) and all(
        case.get("checks", {}).get("paper_stopping_satisfied") is True
        and case.get("checks", {}).get("kkt_target_satisfied") is True
        for case in cases
    )
    add_check(
        checks,
        "equation_54_and_stated_kkt_targets",
        stopping_and_kkt,
        "all six cases satisfy both tests",
    )

    deterministic = bool(cases) and all(case.get("deterministic_repeat") is True for case in cases)
    add_check(
        checks,
        "full_repeated_runs_deterministic",
        deterministic,
        "iteration counts, states, and non-timing histories match exactly",
    )

    spectral_cases = [
        case for case in cases if case.get("sgs_hpr", {}).get("inequality_spectrum") is not None
    ]
    spectral_valid = bool(spectral_cases)
    for case in spectral_cases:
        spectral = case["sgs_hpr"]["inequality_spectrum"]
        spectral_valid = spectral_valid and (
            spectral["power_converged"] is True
            and spectral["maximum_estimate_difference"] <= 1e-10
            and spectral["lambda_used"] > spectral["dense_eigendecomposition"]
            and spectral["s2_minimum_eigenvalue"] > 0.0
        )
    add_check(
        checks,
        "spectral_estimators_cross_checked",
        spectral_valid,
        f"{len(spectral_cases)} inequality-bearing cases",
    )

    equality_valid = bool(cases) and all(
        case["sgs_hpr"]["equality_system"]["full_row_rank"] is True
        and case["sgs_hpr"]["equality_system"]["positive_definite"] is True
        and case["sgs_hpr"]["maximum_equality_solve_infinity_residual"]
        <= config.get("maximum_equality_infinity_residual", -1.0)
        for case in cases
    )
    add_check(
        checks,
        "direct_equality_solves_accurate",
        equality_valid,
        "rank, SPD, and infinity-norm residual checks passed",
    )

    dcopf_names = {"case5_base_t1", "case5_synthetic_extension_t2"}
    dcopf_cases = [by_name[name] for name in dcopf_names if name in by_name]
    dcopf_valid = len(dcopf_cases) == 2 and all(
        case.get("approximate_candidate_validation", {}).get("passed") is True
        and case.get("approximate_candidate_validation", {}).get("mode")
        == "approximate_first_order_candidate"
        and case["comparison"]["maximum_physical_violation"]
        <= config.get("dcopf_physical_tolerance", -1.0)
        and case["comparison"]["scaled_objective_gap"]
        <= config.get("dcopf_maximum_scaled_objective_gap", -1.0)
        for case in dcopf_cases
    )
    add_check(
        checks,
        "t1_t2_dcopf_objective_and_physics",
        dcopf_valid,
        "public T=1 and labeled synthetic T=2 candidates",
    )

    trajectory_path = DEFAULT_EVIDENCE.parent / str(evidence.get("trajectory_file", "missing"))
    last_iterations, trajectory_fields = _trajectory_summary(trajectory_path)
    expected_iterations = {str(case["name"]): int(case["sgs_hpr"]["iterations"]) for case in cases}
    add_check(
        checks,
        "sampled_trajectories_complete",
        trajectory_fields and last_iterations == expected_iterations,
        f"last_iterations={last_iterations}",
    )

    boundary = evidence.get("reproduction_boundary", {})
    boundary_valid = (
        boundary.get("dgx_spark_touched") is False
        and boundary.get("gpu_code_used") is False
        and boundary.get("structural_y1_used") is False
        and boundary.get("adaptive_sigma_used") is False
        and boundary.get("restart_used") is False
        and boundary.get("scaling_used") is False
        and not (PROJECT_ROOT / "src" / "gpu_dcopf_hpr" / "structural_y1.py").exists()
    )
    add_check(
        checks,
        "later_stage_boundaries_preserved",
        boundary_valid,
        "no Stage 4, 5, or 6 implementation or DGX execution",
    )

    totals: dict[str, int] = defaultdict(int)
    for check in checks:
        totals["passed" if check["passed"] else "failed"] += 1
    return {
        "stage": 3,
        "passed": totals["failed"] == 0,
        "summary": dict(totals),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_checks()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
