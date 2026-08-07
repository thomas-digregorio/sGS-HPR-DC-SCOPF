"""Independently validate the Stage 9 report, index, tables, and PDF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import generate_stage_9_artifacts as generator  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "results/raw/stage_9/stage_9_checks.json"
INDEX_PATH = PROJECT_ROOT / "results/stage_9_result_index.json"
MARKDOWN_PATH = PROJECT_ROOT / "docs/final_reproduction_report.md"
TEX_PATH = PROJECT_ROOT / "docs/final_reproduction_report.tex"
PDF_PATH = PROJECT_ROOT / "output/pdf/final_reproduction_report.pdf"

EXPECTED_CHECK_TOTALS = [10, 6, 9, 12, 20, 23, 21, 19, 12, 13]
REQUIRED_SUBJECTS = [
    "Source paper summary",
    "Mathematical formulation",
    "Implemented algorithm",
    "Derivation verification",
    "Missing source information",
    "Experimental environment",
    "CPU implementation",
    "GPU implementation",
    "Validation design and results",
    "Benchmark results",
    "Timing decomposition",
    "Memory and resource results",
    "Differences from the paper",
    "Exact-reproduction classification",
    "Limitations",
    "Recommended next research step",
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _required_files() -> list[Path]:
    return [
        PROJECT_ROOT / "docs/stage_9_contract.md",
        PROJECT_ROOT / "docs/stage_9_source_notes.md",
        MARKDOWN_PATH,
        TEX_PATH,
        PROJECT_ROOT / "docs/reproducibility_checklist.md",
        PROJECT_ROOT / "docs/regeneration_commands.md",
        INDEX_PATH,
        PROJECT_ROOT / "results/tables/stage_9_stage_checks.csv",
        PROJECT_ROOT / "results/tables/stage_9_benchmarks.csv",
        PROJECT_ROOT / "results/tables/stage_9_structural_reconciliation.csv",
        PROJECT_ROOT / "results/tables/stage_9_resource_boundaries.csv",
        PROJECT_ROOT / "results/tables/stage_9_timing_decomposition.csv",
        PROJECT_ROOT / "results/plots/stage_9_solver_timings.svg",
        PROJECT_ROOT / "results/plots/stage_9_resource_boundaries.svg",
        PDF_PATH,
    ]


def _report_subjects_valid(markdown: str) -> tuple[bool, list[str]]:
    normalized = markdown.casefold()
    aliases = {
        "Source paper summary": ["source paper summary"],
        "Mathematical formulation": ["mathematical formulation"],
        "Implemented algorithm": ["implemented algorithm"],
        "Derivation verification": ["derivation verification"],
        "Missing source information": ["missing source information"],
        "Experimental environment": ["experimental environment"],
        "CPU implementation": ["cpu implementation"],
        "GPU implementation": ["gpu implementation"],
        "Validation design and results": ["validation design and results"],
        "Benchmark results": ["benchmark results"],
        "Timing decomposition": ["timing decomposition"],
        "Memory and resource results": ["memory and resource results"],
        "Differences from the paper": ["differences from the paper"],
        "Exact-reproduction classification": ["exact-reproduction classification"],
        "Limitations": ["limitations"],
        "Recommended next research step": ["recommended next research step"],
    }
    missing = [
        subject
        for subject in REQUIRED_SUBJECTS
        if not any(alias in normalized for alias in aliases[subject])
    ]
    return not missing, missing


def _pdf_valid(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "compiled PDF is missing"
    data = path.read_bytes()
    page_count = len(re.findall(rb"/Type\s*/Page\b", data))
    valid = data.startswith(b"%PDF-") and len(data) >= 100_000 and page_count >= 5
    return valid, f"bytes={len(data)}, page_objects={page_count}, sha256={_sha256(path)}"


def run_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    required = _required_files()
    missing = [path.relative_to(PROJECT_ROOT).as_posix() for path in required if not path.exists()]
    _add(checks, "required_artifacts", not missing, f"missing={missing}")

    index = _load(INDEX_PATH) if INDEX_PATH.exists() else {}
    classification = index.get("final_classification") or {}
    classification_valid = (
        classification.get("code") == "D"
        and classification.get("label") == "structural reproduction"
        and classification.get("exact_reproduction") is False
        and classification.get("paper_timing_reproduced") is False
        and classification.get("local_speedup_claimed") is False
    )
    _add(
        checks, "classification_D", classification_valid, json.dumps(classification, sort_keys=True)
    )

    paper = _load(generator.PAPER_METADATA_PATH)
    paper_valid = (
        paper.get("page_count") == 17
        and paper.get("byte_size") == 9_157_238
        and paper.get("sha256")
        == "7e9791646401e11bfddf9ebed6bd94491ed0b592744581edd851ddbf5e20dba4"
    )
    _add(checks, "paper_identity", paper_valid, json.dumps(paper, sort_keys=True))

    check_rows = _read_csv(PROJECT_ROOT / "results/tables/stage_9_stage_checks.csv")
    totals = [int(row["total_checks"]) for row in check_rows]
    stage_checks_valid = (
        totals == EXPECTED_CHECK_TOTALS
        and all(row["checker_result"] == "PASS" for row in check_rows)
        and check_rows[-2]["scientific_result"] == "FAIL"
        and check_rows[-1]["scientific_result"] == "COMPLETE_WITH_RESOURCE_LIMITS"
    )
    checker_passes = sum(row["checker_result"] == "PASS" for row in check_rows)
    _add(
        checks,
        "checker_ledger",
        stage_checks_valid,
        f"totals={totals}, checker_passes={checker_passes}",
    )

    stage8 = _load(generator.STAGE8_PATH)
    t6 = stage8["cases"][-1]
    cpu = t6["solver_tracks"]["cpu_fp64_sgs_hpr"]["correctness"]
    gpu = t6["solver_tracks"]["gpu_fp64_sgs_hpr"]["correctness"]
    stage8_valid = (
        stage8.get("status") == "STOPPED_ON_FAILURE"
        and stage8.get("all_passed") is False
        and stage8["stage_boundary"]["passing_prefix_length"] == 4
        and len(stage8["cases"]) == 5
        and t6["key"] == "case9241pegase:T6"
        and t6["status"] == "FAIL"
        and cpu["status"] == "TIME_LIMIT"
        and abs(float(cpu["wall_seconds"]) - 3600.092738645966) < 1e-9
        and gpu["status"] == "SUCCESS"
        and gpu["passed"] is True
    )
    passing_prefix = stage8.get("stage_boundary", {}).get("passing_prefix_length")
    _add(
        checks,
        "stage_8_terminal_semantics",
        stage8_valid,
        f"status={stage8.get('status')}, prefix={passing_prefix}",
    )

    continuation = _load(generator.CONTINUATION_PATH)
    continuation_valid = (
        continuation.get("status") == "COMPLETE_WITH_RESOURCE_LIMITS"
        and continuation.get("all_requested_rows_resolved") is True
        and continuation.get("allocation_history") == []
        and [case["status"] for case in continuation["cases"]]
        == ["MEMORY_BLOCKED", "INDEX_BLOCKED", "INDEX_BLOCKED"]
        and all(case["full_lp_allocation_attempted"] is False for case in continuation["cases"])
    )
    continuation_allocations = len(continuation.get("allocation_history", []))
    _add(
        checks,
        "continuation_semantics",
        continuation_valid,
        f"status={continuation.get('status')}, allocations={continuation_allocations}",
    )

    benchmarks = _read_csv(PROJECT_ROOT / "results/tables/stage_9_benchmarks.csv")
    allocated = [row for row in benchmarks if row["full_lp_allocated"] == "True"]
    benchmark_valid = (
        len(benchmarks) == 14
        and len(allocated) == 11
        and sum(row["gpu_status"] == "SUCCESS" for row in allocated) == 11
        and sum(row["cpu_status"] == "SUCCESS" for row in allocated) == 10
        and sum(row["row_status"] == "PASS" for row in allocated) == 10
        and benchmarks[-3]["row_status"] == "MEMORY_BLOCKED"
        and benchmarks[-2]["row_status"] == "INDEX_BLOCKED"
        and benchmarks[-1]["row_status"] == "INDEX_BLOCKED"
    )
    gpu_passes = sum(row["gpu_status"] == "SUCCESS" for row in allocated)
    _add(
        checks,
        "benchmark_aggregation",
        benchmark_valid,
        f"rows={len(benchmarks)}, allocated={len(allocated)}, gpu_pass={gpu_passes}",
    )

    structural = _read_csv(PROJECT_ROOT / "results/tables/stage_9_structural_reconciliation.csv")
    structural_valid = (
        len(structural) == 18
        and all(row["dimension_match"] == "True" for row in structural)
        and all(row["nnz_match"] == "False" for row in structural)
        and all(row["paper_time_comparable"] == "False" for row in structural)
    )
    dimension_matches = sum(row["dimension_match"] == "True" for row in structural)
    nnz_matches = sum(row["nnz_match"] == "True" for row in structural)
    _add(
        checks,
        "structural_reconciliation",
        structural_valid,
        f"rows={len(structural)}, dimension_matches={dimension_matches}, nnz_matches={nnz_matches}",
    )

    resources = _read_csv(PROJECT_ROOT / "results/tables/stage_9_resource_boundaries.csv")
    resource_map = {row["case_key"]: row for row in resources}
    t16_resource = resource_map.get("case9241pegase:T16", {})
    t24_resource = resource_map.get("case9241pegase:T24", {})
    t32_resource = resource_map.get("case9241pegase:T32", {})
    resource_valid = (
        len(resources) == 8
        and t16_resource.get("status") == "MEMORY_BLOCKED"
        and float(t16_resource["projected_unified_gib"])
        > min(float(t16_resource["host_budget_gib"]), float(t16_resource["device_budget_gib"]))
        and float(t16_resource["projected_unified_gib"]) < float(t16_resource["nominal_80pct_gib"])
        and t16_resource.get("block_reasons")
        == "failed:within_host_safety_budget;failed:within_device_safety_budget"
        and int(t24_resource["planning_nnz"]) > generator.INT32_MAX
        and int(t24_resource["exact_reconstructed_nnz"]) < generator.INT32_MAX
        and int(t32_resource["planning_nnz"]) > generator.INT32_MAX
        and int(t32_resource["exact_reconstructed_nnz"]) > generator.INT32_MAX
        and t24_resource["status"] == t32_resource["status"] == "INDEX_BLOCKED"
    )
    _add(
        checks,
        "resource_boundaries",
        resource_valid,
        (
            "T16 live-budget-blocked below nominal 80%; T24 policy-envelope-blocked with "
            "exact count below int32; T32 exact and envelope counts above int32"
        ),
    )

    markdown = MARKDOWN_PATH.read_text(encoding="utf-8") if MARKDOWN_PATH.exists() else ""
    subjects_valid, missing_subjects = _report_subjects_valid(markdown)
    _add(checks, "required_report_subjects", subjects_valid, f"missing={missing_subjects}")

    report_semantics_valid = all(
        phrase in markdown
        for phrase in [
            "Final classification: D - structural reproduction",
            "Stage 8: FAIL",
            "No speedup is claimed",
            "Stage 10: LOCKED",
            "T16",
            "2,531,600,260",
            "3,375,704,460",
            "2,057,650,132",
            "2,743,770,956",
            "preregistered project-specific A--E",
            "Code and data availability",
            "one censored correctness attempt",
        ]
    )
    _add(
        checks,
        "report_semantics",
        report_semantics_valid,
        "classification, failure, caveat, and resource phrases",
    )

    tex = TEX_PATH.read_text(encoding="utf-8") if TEX_PATH.exists() else ""
    tex_valid = all(
        token in tex
        for token in [
            r"\documentclass[10pt,journal]{IEEEtran}",
            r"\begin{abstract}",
            r"\appendices",
            r"\classresult",
            r"\begin{thebibliography}{99}",
            "CPU timeout",
            "Stage 10 remains locked",
            "Independent Researcher",
            "stage9-report-v2",
            "0.206",
            r"3.03\times10^{-16}",
        ]
    )
    _add(
        checks,
        "latex_scientific_structure",
        tex_valid,
        "IEEEtran, abstract, equations, figures, appendices, bibliography",
    )

    hash_errors: list[str] = []
    for group in ("evidence", "tables", "figures"):
        for item in index.get(group, []):
            path = PROJECT_ROOT / item["path"]
            actual = _sha256(path) if path.exists() else None
            if actual != item.get("sha256"):
                hash_errors.append(f"{item.get('path')}: {actual} != {item.get('sha256')}")
    _add(checks, "result_index_hashes", not hash_errors, f"errors={hash_errors}")

    coverage = index.get("coverage") or {}
    coverage_valid = coverage == {
        "allocated_benchmark_rows": 11,
        "dimension_matches": 18,
        "nnz_matches": 0,
        "resource_resolved_without_allocation": 3,
        "symbolic_table_ii_rows": 18,
        "validated_cpu_rows": 10,
        "validated_gpu_rows": 11,
    }
    _add(checks, "result_index_coverage", coverage_valid, json.dumps(coverage, sort_keys=True))

    pdf_ok, pdf_detail = _pdf_valid(PDF_PATH)
    _add(checks, "compiled_pdf", pdf_ok, pdf_detail)

    stage10 = index.get("stage_10") or {}
    project_state = (PROJECT_ROOT / "docs/project_state.md").read_text(encoding="utf-8")
    stage10_valid = (
        stage10 == {"n_minus_1_work_performed": False, "status": "LOCKED"}
        and "Stage 10" in markdown
        and "Stage 10" in tex
        and "Stage 10" in project_state
        and "locked" in project_state.casefold()
    )
    _add(checks, "stage_10_locked", stage10_valid, json.dumps(stage10, sort_keys=True))

    generation = subprocess.run(
        [
            sys.executable,
            str(generator.PROJECT_ROOT / "scripts/generate_stage_9_artifacts.py"),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _add(
        checks,
        "generated_outputs_current",
        generation.returncode == 0,
        (generation.stdout + generation.stderr).strip(),
    )

    all_passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "stage": 9,
        "checker_status": "PASS" if all_passed else "FAIL",
        "all_passed": all_passed,
        "classification": "D - structural reproduction",
        "stage_8_scientific_result": "FAIL",
        "stage_10_status": "LOCKED",
        "summary": {
            "passed": sum(check["passed"] for check in checks),
            "total": len(checks),
        },
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
