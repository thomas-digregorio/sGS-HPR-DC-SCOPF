"""Validate the preserved Stage 2 evidence and Stage 3 boundary."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = PROJECT_ROOT / "results" / "raw" / "stage_2" / "stage_2_validation.json"


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def metadata_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return sum(1 for _ in stream)


def run_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    required_paths = [
        "data/raw/matpower/case5.m",
        "data/raw/matpower/README.md",
        "configs/dcopf/case5_base_stage_2.json",
        "configs/dcopf/case5_synthetic_extension_stage_2.json",
        "src/gpu_dcopf_hpr/network_data.py",
        "src/gpu_dcopf_hpr/ptdf.py",
        "src/gpu_dcopf_hpr/dcopf_model.py",
        "scripts/build_dcopf.py",
        "tests/unit/test_network_data.py",
        "tests/unit/test_ptdf.py",
        "tests/unit/test_dcopf_model.py",
        "tests/integration/test_stage2_dcopf.py",
        "docs/stage_reports/stage_2_report.md",
        "docs/project_state.md",
    ]
    missing = [path for path in required_paths if not (PROJECT_ROOT / path).is_file()]
    add_check(
        checks,
        "required_stage_two_paths",
        not missing,
        "complete" if not missing else f"missing {missing}",
    )

    evidence = (
        json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
        if DEFAULT_EVIDENCE.is_file()
        else {}
    )
    add_check(
        checks,
        "stage_two_validation_passed",
        evidence.get("all_passed") is True,
        str(evidence.get("all_passed", "unavailable")),
    )
    case_names = {case.get("name") for case in evidence.get("cases", [])}
    expected_names = {"case5_base_t1", "case5_synthetic_extension_t2"}
    add_check(
        checks,
        "base_and_synthetic_extension_present",
        case_names == expected_names,
        f"cases={sorted(str(name) for name in case_names)}",
    )

    expected_dimensions = {
        "case5_base_t1": {"n": 15, "m1": 1, "m2": 16, "m": 17},
        "case5_synthetic_extension_t2": {"n": 36, "m1": 3, "m2": 46, "m": 49},
    }
    dimensions_valid = True
    metadata_valid = True
    validations_valid = True
    ptdf_valid = True
    for case in evidence.get("cases", []):
        name = case.get("name")
        dimensions_valid = dimensions_valid and (
            case.get("expected_dimensions") == expected_dimensions.get(name)
        )
        metadata_path = DEFAULT_EVIDENCE.parent / str(case.get("row_metadata_file", "missing"))
        metadata_valid = metadata_valid and (
            metadata_count(metadata_path)
            == case.get("row_metadata_records")
            == case["dimensions"]["m"]
        )
        validations_valid = validations_valid and (
            case.get("independent_validation", {}).get("passed") is True
        )
        ptdf_valid = ptdf_valid and (
            case.get("ptdf_angle_probes", {}).get(
                "maximum_flow_difference_mw",
                float("inf"),
            )
            <= 1e-10
        )
    add_check(checks, "dimension_formulas", dimensions_valid, str(expected_dimensions))
    add_check(checks, "row_metadata_complete", metadata_valid, "one record per A1/A2 row")
    add_check(
        checks,
        "independent_physical_validation",
        validations_valid,
        "all physical families passed",
    )
    add_check(
        checks,
        "ptdf_matches_angle_flows",
        ptdf_valid,
        "maximum difference at most 1e-10 MW",
    )
    source = evidence.get("network_source", {})
    source_valid = (
        source.get("upstream_release") == "8.1"
        and source.get("upstream_blob") == "b6370ab230ac5346023d23be20d973a81f09e12a"
        and len(str(source.get("sha256", ""))) == 64
    )
    add_check(checks, "public_case_provenance", source_valid, str(source))

    premature_stage_three = [
        path
        for path in (
            "src/gpu_dcopf_hpr/sgs_hpr.py",
            "src/gpu_dcopf_hpr/structural_y1.py",
        )
        if (PROJECT_ROOT / path).exists()
    ]
    add_check(
        checks,
        "no_premature_stage_three_implementation",
        not premature_stage_three,
        "none present" if not premature_stage_three else f"present {premature_stage_three}",
    )
    return {
        "stage": 2,
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
