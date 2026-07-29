"""Run deterministic Stage 0 acceptance checks.

This script checks evidence and structure only. Test-suite, formatter, and
dashboard build results are recorded separately in the stage report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PDF_HASH = "7e9791646401e11bfddf9ebed6bd94491ed0b592744581edd851ddbf5e20dba4"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dimensions(
    *,
    horizon: int,
    branches: int,
    generators: int,
    renewables: int,
    storage: int,
) -> tuple[int, int]:
    variables = horizon * (3 * generators + renewables + 2 * storage)
    constraints = (
        2 * horizon * branches
        + (4 * horizon - 2) * generators
        + (2 * horizon + 1) * storage
        + 3 * horizon
    )
    return constraints, variables


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})


def run_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    paper = PROJECT_ROOT / "references" / "AnEfficientGPU-basedHalpernAccelerating.pdf"
    paper_metadata = PROJECT_ROOT / "results" / "raw" / "stage_0" / "paper_metadata.json"
    specification = PROJECT_ROOT / "docs" / "paper_specification.md"

    add_check(checks, "source_pdf_exists", paper.is_file(), str(paper.relative_to(PROJECT_ROOT)))
    actual_hash = digest(paper) if paper.is_file() else ""
    add_check(
        checks,
        "source_pdf_hash",
        actual_hash == EXPECTED_PDF_HASH,
        actual_hash or "unavailable",
    )

    metadata: dict[str, Any] = {}
    if paper_metadata.is_file():
        metadata = json.loads(paper_metadata.read_text(encoding="utf-8"))
    add_check(
        checks,
        "source_pdf_page_count",
        metadata.get("page_count") == 17,
        str(metadata.get("page_count", "unavailable")),
    )

    specification_text = (
        specification.read_text(encoding="utf-8") if specification.is_file() else ""
    )
    missing_equations = [
        number for number in range(1, 56) if f"({number})" not in specification_text
    ]
    add_check(
        checks,
        "equations_1_to_55_indexed",
        not missing_equations,
        "complete" if not missing_equations else f"missing {missing_equations}",
    )
    missing_algorithms = [
        algorithm
        for algorithm in ("Algorithm 1", "Algorithm 2")
        if algorithm not in specification_text
    ]
    add_check(
        checks,
        "algorithms_indexed",
        not missing_algorithms,
        "complete" if not missing_algorithms else f"missing {missing_algorithms}",
    )

    dimension_cases = {
        "case1354_T4": {
            "inputs": dict(
                horizon=4,
                branches=1991,
                generators=260,
                renewables=136,
                storage=68,
            ),
            "expected": (20_192, 4_208),
        },
        "case2868_T16": {
            "inputs": dict(
                horizon=16,
                branches=3808,
                generators=600,
                renewables=286,
                storage=143,
            ),
            "expected": (163_823, 37_952),
        },
        "case9241_T6": {
            "inputs": dict(
                horizon=6,
                branches=16_049,
                generators=1445,
                renewables=920,
                storage=460,
            ),
            "expected": (230_376, 37_050),
        },
    }
    for name, case in dimension_cases.items():
        actual = dimensions(**case["inputs"])
        add_check(
            checks,
            f"dimension_{name}",
            actual == case["expected"],
            f"m={actual[0]}, n={actual[1]}",
        )

    required_paths = [
        "README.md",
        "pyproject.toml",
        "environment/environment_report.md",
        "environment/package_versions.txt",
        "environment/local_environment.json",
        "environment/dgx_spark_environment.json",
        "docs/paper_specification.md",
        "docs/mathematical_notes.md",
        "docs/reproduction_limits.md",
        "docs/decisions.md",
        "docs/project_state.md",
        "docs/stage_reports/stage_0_report.md",
        "scripts/inspect_environment.py",
        "dashboard/app/ReproductionDashboard.tsx",
    ]
    missing_paths = [path for path in required_paths if not (PROJECT_ROOT / path).exists()]
    add_check(
        checks,
        "required_stage_zero_paths",
        not missing_paths,
        "complete" if not missing_paths else f"missing {missing_paths}",
    )

    premature_solver_files = [
        PROJECT_ROOT / "src" / "gpu_dcopf_hpr" / name
        for name in (
            "canonical_lp.py",
            "projections.py",
            "residuals.py",
            "hpr_generic.py",
            "sgs_hpr.py",
            "gpu_backend.py",
        )
    ]
    present_solver_files = [
        str(path.relative_to(PROJECT_ROOT)) for path in premature_solver_files if path.exists()
    ]
    add_check(
        checks,
        "no_premature_solver_implementation",
        not present_solver_files,
        "none present" if not present_solver_files else f"present {present_solver_files}",
    )

    return {
        "stage": 0,
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
