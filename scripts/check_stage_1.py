"""Validate the preserved Stage 1 evidence and stage boundary."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "raw" / "stage_1" / "reference_comparison.json"
DEFAULT_TRAJECTORY = PROJECT_ROOT / "results" / "raw" / "stage_1" / "hpr_trajectories.jsonl.gz"


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def run_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    required_paths = [
        "src/gpu_dcopf_hpr/canonical_lp.py",
        "src/gpu_dcopf_hpr/projections.py",
        "src/gpu_dcopf_hpr/residuals.py",
        "src/gpu_dcopf_hpr/hpr_generic.py",
        "src/gpu_dcopf_hpr/toy_problems.py",
        "src/gpu_dcopf_hpr/validation.py",
        "scripts/run_toy_lp.py",
        "tests/unit/test_canonical_lp.py",
        "tests/unit/test_projections.py",
        "tests/unit/test_residuals.py",
        "tests/unit/test_halpern.py",
        "tests/integration/test_hpr_reference.py",
        "docs/stage_reports/stage_1_report.md",
        "docs/project_state.md",
    ]
    missing = [path for path in required_paths if not (PROJECT_ROOT / path).is_file()]
    add_check(
        checks,
        "required_stage_one_paths",
        not missing,
        "complete" if not missing else f"missing {missing}",
    )

    summary: dict[str, Any] = {}
    if DEFAULT_SUMMARY.is_file():
        summary = json.loads(DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    add_check(
        checks,
        "reference_comparison_passed",
        summary.get("all_passed") is True,
        str(summary.get("all_passed", "unavailable")),
    )

    expected_cases = {
        "analytic_toy",
        "box_bound_active",
        "inequality_inactive",
        "planted_random",
    }
    actual_cases = {case.get("name") for case in summary.get("cases", [])}
    add_check(
        checks,
        "four_required_reference_cases",
        actual_cases == expected_cases,
        f"cases={sorted(str(name) for name in actual_cases)}",
    )

    analytic = next(
        (case for case in summary.get("cases", []) if case.get("name") == "analytic_toy"),
        {},
    )
    analytic_x = analytic.get("hpr", {}).get("x", [])
    analytic_objective = analytic.get("hpr", {}).get("objective")
    analytic_valid = (
        len(analytic_x) == 2
        and abs(analytic_x[0] - 0.4) <= 5e-4
        and abs(analytic_x[1] - 0.6) <= 5e-4
        and analytic_objective is not None
        and abs(analytic_objective - 1.4) <= 2e-4
    )
    add_check(
        checks,
        "analytic_toy_solution",
        analytic_valid,
        f"x={analytic_x}, objective={analytic_objective}",
    )

    trajectory_count = 0
    if DEFAULT_TRAJECTORY.is_file():
        with gzip.open(DEFAULT_TRAJECTORY, "rt", encoding="utf-8") as stream:
            trajectory_count = sum(1 for _ in stream)
    add_check(
        checks,
        "convergence_trajectories_preserved",
        trajectory_count > 0,
        f"rows={trajectory_count}",
    )

    premature_stage_two = [
        path
        for path in (
            "src/gpu_dcopf_hpr/network_data.py",
            "src/gpu_dcopf_hpr/ptdf.py",
            "src/gpu_dcopf_hpr/dcopf_model.py",
        )
        if (PROJECT_ROOT / path).exists()
    ]
    add_check(
        checks,
        "no_premature_stage_two_implementation",
        not premature_stage_two,
        "none present" if not premature_stage_two else f"present {premature_stage_two}",
    )
    return {
        "stage": 1,
        "passed": all(check["passed"] for check in checks),
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
