"""Generate Stage 9 tables, figures, and the machine-readable result index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
TABLE_ROOT = RESULTS_ROOT / "tables"
PLOT_ROOT = RESULTS_ROOT / "plots"
INDEX_PATH = RESULTS_ROOT / "stage_9_result_index.json"

STAGE_CHECK_PATHS = [
    PROJECT_ROOT / f"results/raw/stage_{stage}/stage_{stage}_checks.json" for stage in range(8)
]
STAGE_CHECK_PATHS.extend(
    [
        PROJECT_ROOT / "results/raw/stage_8/stage_8_checks.json",
        PROJECT_ROOT
        / "results/raw/stage_8/gpu_only_completion/stage_8_gpu_only_completion_checks.json",
    ]
)

STAGE7_PATH = PROJECT_ROOT / "results/raw/stage_7/stage_7_validation.json"
STAGE8_PATH = PROJECT_ROOT / "results/raw/stage_8/stage_8_validation.json"
CONTINUATION_PATH = (
    PROJECT_ROOT
    / "results/raw/stage_8/gpu_only_completion/stage_8_gpu_only_completion_validation.json"
)
STAGE6_PATH = PROJECT_ROOT / "results/raw/stage_6/stage_6_validation.json"
PAPER_METADATA_PATH = PROJECT_ROOT / "results/raw/stage_0/paper_metadata.json"

INT32_MAX = 2_147_483_647
GIB = 1024**3
REPORT_BASE_COMMIT = "c08c53b7ef5d2bde006728c76fb43fe621685e20"


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


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _overall_passed(checker: dict[str, Any]) -> bool:
    value = checker.get("passed", checker.get("all_passed"))
    return value is True


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _track_summary(track: dict[str, Any] | None) -> dict[str, Any]:
    if not track:
        return {
            "status": "NOT_RUN",
            "median_seconds": None,
            "iterations": None,
            "restart_count": None,
            "objective": None,
            "raw_kkt": None,
            "physical_violation": None,
            "objective_gap": None,
        }
    correctness = track.get("correctness") or {}
    candidate = correctness.get("candidate") or {}
    residuals = candidate.get("residuals") or {}
    physical = candidate.get("physical_validation") or {}
    statistics = track.get("statistics") or {}
    return {
        "status": correctness.get("status", track.get("timing_status", "UNKNOWN")),
        "median_seconds": statistics.get("median_seconds"),
        "iterations": correctness.get("iterations"),
        "restart_count": correctness.get("restart_count"),
        "objective": candidate.get("objective"),
        "raw_kkt": residuals.get("kkt_combined_norm"),
        "physical_violation": physical.get("maximum_violation"),
        "objective_gap": candidate.get("scaled_objective_gap_to_highs"),
    }


def _benchmark_row(stage: int, case: dict[str, Any]) -> dict[str, Any]:
    dimensions = (case.get("construction") or {}).get("dimensions") or {}
    structural = case.get("structural_reconciliation") or {}
    tracks = case.get("solver_tracks") or {}
    highs = _track_summary(tracks.get("highs"))
    cpu = _track_summary(tracks.get("cpu_fp64_sgs_hpr"))
    gpu = _track_summary(tracks.get("gpu_fp64_sgs_hpr"))
    paper_nnz = structural.get("published_nnz")
    actual_nnz = dimensions.get("nnz_A")
    nnz_difference_pct = None
    if isinstance(paper_nnz, int) and paper_nnz and isinstance(actual_nnz, int):
        nnz_difference_pct = 100.0 * (actual_nnz - paper_nnz) / paper_nnz
    return {
        "sequence": "",
        "stage": stage,
        "case_key": case["key"],
        "m": dimensions.get("m"),
        "n": dimensions.get("n"),
        "paper_nnz": paper_nnz,
        "reconstructed_nnz": actual_nnz,
        "nnz_difference_pct": nnz_difference_pct,
        "full_lp_allocated": case.get("full_lp_allocation_attempted", True),
        "row_status": case.get("status"),
        "highs_status": highs["status"],
        "highs_median_seconds": highs["median_seconds"],
        "cpu_status": cpu["status"],
        "cpu_median_seconds": cpu["median_seconds"],
        "gpu_status": gpu["status"],
        "gpu_median_seconds": gpu["median_seconds"],
        "gpu_iterations": gpu["iterations"],
        "gpu_restart_count": gpu["restart_count"],
        "gpu_objective": gpu["objective"],
        "gpu_raw_kkt": gpu["raw_kkt"],
        "gpu_physical_violation": gpu["physical_violation"],
        "gpu_objective_gap": gpu["objective_gap"],
    }


def _continuation_row(case: dict[str, Any]) -> dict[str, Any]:
    estimate = case.get("resource_estimate") or {}
    return {
        "sequence": case.get("sequence"),
        "stage": 8,
        "case_key": case["key"],
        "m": estimate.get("row_count"),
        "n": estimate.get("column_count"),
        "paper_nnz": estimate.get("paper_nnz"),
        "reconstructed_nnz": estimate.get("exact_reconstructed_nnz"),
        "nnz_difference_pct": (
            100.0
            * (estimate["exact_reconstructed_nnz"] - estimate["paper_nnz"])
            / estimate["paper_nnz"]
            if estimate.get("paper_nnz") and estimate.get("exact_reconstructed_nnz")
            else None
        ),
        "full_lp_allocated": case.get("full_lp_allocation_attempted"),
        "row_status": case.get("status"),
        "highs_status": "NOT_RUN",
        "highs_median_seconds": None,
        "cpu_status": "SKIPPED_BY_APPROVED_SCOPE",
        "cpu_median_seconds": None,
        "gpu_status": "NOT_RUN",
        "gpu_median_seconds": None,
        "gpu_iterations": None,
        "gpu_restart_count": None,
        "gpu_objective": None,
        "gpu_raw_kkt": None,
        "gpu_physical_violation": None,
        "gpu_objective_gap": None,
    }


def _stage_check_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in STAGE_CHECK_PATHS:
        checker = _load(path)
        checks = checker.get("checks") or []
        passed = sum(check.get("passed") is True for check in checks)
        is_continuation = "gpu_only_completion" in path.name
        stage = int(checker["stage"])
        if is_continuation:
            scientific_result = "COMPLETE_WITH_RESOURCE_LIMITS"
            interpretation = "Sequences 6--8 resolved without allocation"
        elif stage == 8:
            scientific_result = "FAIL"
            interpretation = "Protocol valid; required T6 CPU track timed out"
        else:
            scientific_result = "PASS"
            interpretation = "Stage acceptance passed"
        rows.append(
            {
                "scope": "Stage 8 continuation" if is_continuation else f"Stage {stage}",
                "scientific_result": scientific_result,
                "checker_result": "PASS" if _overall_passed(checker) else "FAIL",
                "passed_checks": passed,
                "total_checks": len(checks),
                "interpretation": interpretation,
                "artifact": _relative(path),
                "sha256": _sha256(path),
            }
        )
    return rows


def _structural_rows(stage7: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in stage7["symbolic_ledger"]:
        paper_nnz = int(item["published_nnz"])
        reproduced = int(item["actual_reconstruction_nnz"])
        rows.append(
            {
                "case_key": item["key"],
                "paper_m": item["published_m"],
                "reproduced_m": item["computed_m"],
                "paper_n": item["published_n"],
                "reproduced_n": item["computed_n"],
                "paper_nnz": paper_nnz,
                "reconstructed_nnz": reproduced,
                "nnz_difference": reproduced - paper_nnz,
                "nnz_difference_pct": 100.0 * (reproduced - paper_nnz) / paper_nnz,
                "dimension_match": item["dimensions_match_table"],
                "nnz_match": reproduced == paper_nnz,
                "paper_time_comparable": item["paper_time_comparable"],
            }
        )
    return rows


def _resource_rows(stage8: dict[str, Any], continuation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in stage8["cases"]:
        gate = case.get("stage8_resource_gate") or {}
        rows.append(
            {
                "sequence": next(
                    (
                        entry.get("sequence")
                        for entry in stage8.get("allocation_history", [])
                        if entry.get("key") == case["key"]
                    ),
                    None,
                ),
                "case_key": case["key"],
                "status": case["status"],
                "full_lp_allocated": case.get("full_lp_allocation_attempted"),
                "projected_unified_gib": gate.get("projected_unified_peak_bytes", 0) / GIB,
                "host_budget_gib": gate.get("host_safety_budget_bytes", 0) / GIB,
                "device_budget_gib": gate.get("device_safety_budget_bytes", 0) / GIB,
                "planning_nnz": None,
                "int32_limit": INT32_MAX,
                "block_reasons": ";".join(gate.get("block_reasons") or []),
            }
        )
    for case in continuation["cases"]:
        gate = case.get("stage8_resource_gate") or {}
        estimate = case.get("resource_estimate") or {}
        rows.append(
            {
                "sequence": case.get("sequence"),
                "case_key": case["key"],
                "status": case["status"],
                "full_lp_allocated": case.get("full_lp_allocation_attempted"),
                "projected_unified_gib": (
                    gate["projected_unified_peak_bytes"] / GIB
                    if gate.get("projected_unified_peak_bytes") is not None
                    else estimate.get("projected_unified_peak_bytes", 0) / GIB
                ),
                "host_budget_gib": (
                    gate["host_safety_budget_bytes"] / GIB
                    if gate.get("host_safety_budget_bytes") is not None
                    else None
                ),
                "device_budget_gib": (
                    gate["device_safety_budget_bytes"] / GIB
                    if gate.get("device_safety_budget_bytes") is not None
                    else None
                ),
                "planning_nnz": estimate.get("conservative_planning_nnz"),
                "int32_limit": INT32_MAX,
                "block_reasons": ";".join(gate.get("block_reasons") or []),
            }
        )
    return rows


def _timing_rows(stage6: dict[str, Any]) -> list[dict[str, Any]]:
    boundaries = (stage6.get("timing_boundaries") or {}).get("boundaries") or {}
    order = [
        "CUDA initialization",
        "CPU matrix construction and preprocessing",
        "first-run compilation and warm-up",
        "allocation",
        "host-to-device transfer",
        "GPU solver initialization",
        "iteration loop",
        "residual checks",
        "device-to-host transfer",
        "complete end-to-end wall time",
    ]
    return [
        {
            "boundary": name,
            "seconds": boundaries[name].get("seconds"),
            "status": boundaries[name].get("status"),
            "method": boundaries[name].get("method"),
        }
        for name in order
    ]


@dataclass(frozen=True)
class Series:
    name: str
    color: str
    marker: str
    values: list[float | None]


def _solver_timing_svg(rows: list[dict[str, Any]], path: Path) -> None:
    executed = [row for row in rows if row["stage"] in (7, 8) and row["sequence"] == ""]
    labels = [row["case_key"].replace("case", "").replace("pegase", "") for row in executed]
    series = [
        Series("HiGHS", "#2A6FBB", "circle", [row["highs_median_seconds"] for row in executed]),
        Series("CPU FP64", "#4B5563", "square", [row["cpu_median_seconds"] for row in executed]),
        Series("GPU FP64", "#E67E22", "triangle", [row["gpu_median_seconds"] for row in executed]),
    ]
    width, height = 1280, 720
    left, right, top, bottom = 108, 42, 70, 142
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.8, 4200.0

    def x_pos(index: int) -> float:
        return left + (index + 0.5) * plot_w / len(labels)

    def y_pos(value: float) -> float:
        fraction = (math.log10(value) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min))
        return top + plot_h * (1.0 - fraction)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1F2937}.small{font-size:15px}"
        ".axis{stroke:#374151;stroke-width:1.5}.grid{stroke:#D1D5DB;stroke-width:1}"
        ".title{font-size:24px;font-weight:700}.note{font-size:14px;fill:#4B5563}</style>",
        (
            '<text x="640" y="33" text-anchor="middle" class="title">'
            "Local validated solver-core time</text>"
        ),
    ]
    for tick in [1, 10, 100, 1000, 3600]:
        y = y_pos(float(tick))
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 14}" y="{y + 5:.2f}" text-anchor="end" class="small">{tick:g}</text>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            (
                f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" '
                f'y2="{top + plot_h}" class="axis"/>'
            ),
            f'<text x="28" y="{top + plot_h / 2}" transform="rotate(-90 28 {top + plot_h / 2})" '
            'text-anchor="middle" class="small">Median seconds (log scale)</text>',
        ]
    )
    offsets = [-10, 0, 10]
    for s_index, item in enumerate(series):
        for index, value in enumerate(item.values):
            x = x_pos(index) + offsets[s_index]
            if value is None:
                if item.name == "CPU FP64" and labels[index].endswith("T6"):
                    y = y_pos(3600.0)
                    parts.append(
                        f'<path d="M{x - 6:.2f},{y - 6:.2f} L{x + 6:.2f},{y + 6:.2f} '
                        f'M{x + 6:.2f},{y - 6:.2f} L{x - 6:.2f},{y + 6:.2f}" '
                        f'stroke="{item.color}" stroke-width="3"/>'
                    )
                    parts.append(
                        f'<text x="{x + 10:.2f}" y="{y - 9:.2f}" class="note">timeout</text>'
                    )
                continue
            y = y_pos(float(value))
            if item.marker == "circle":
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{item.color}"/>')
            elif item.marker == "square":
                parts.append(
                    f'<rect x="{x - 6:.2f}" y="{y - 6:.2f}" width="12" '
                    f'height="12" fill="{item.color}"/>'
                )
            else:
                parts.append(
                    f'<path d="M{x:.2f},{y - 7:.2f} L{x + 7:.2f},{y + 6:.2f} '
                    f'L{x - 7:.2f},{y + 6:.2f} Z" fill="{item.color}"/>'
                )
    for index, label in enumerate(labels):
        x = x_pos(index)
        escaped = html.escape(label)
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 25}" '
            f'transform="rotate(38 {x:.2f} {top + plot_h + 25})" '
            f'text-anchor="start" class="small">{escaped}</text>'
        )
    legend_x = 420
    for index, item in enumerate(series):
        x = legend_x + index * 170
        parts.append(f'<rect x="{x}" y="52" width="13" height="13" fill="{item.color}"/>')
        parts.append(f'<text x="{x + 20}" y="64" class="small">{item.name}</text>')
    parts.append(
        '<text x="640" y="700" text-anchor="middle" class="note">'
        "Timing boundaries differ by solver; the figure is descriptive and is not a "
        "controlled speedup claim.</text>"
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _resource_svg(resources: list[dict[str, Any]], path: Path) -> None:
    lookup = {row["case_key"]: row for row in resources}
    t16 = lookup["case9241pegase:T16"]
    t24 = lookup["case9241pegase:T24"]
    t32 = lookup["case9241pegase:T32"]
    width, height = 1280, 630
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1F2937}.title{font-size:24px;font-weight:700}"
        ".sub{font-size:18px;font-weight:700}.label{font-size:15px}.note{font-size:14px;fill:#4B5563}"
        ".grid{stroke:#D1D5DB;stroke-width:1}</style>",
        (
            '<text x="640" y="34" text-anchor="middle" class="title">'
            "Stage 8 fail-closed resource boundaries</text>"
        ),
        '<text x="315" y="74" text-anchor="middle" class="sub">T16 unified-memory gate</text>',
        (
            '<text x="960" y="74" text-anchor="middle" class="sub">'
            "T24/T32 signed-int32 CSR gate</text>"
        ),
    ]
    memory_values = [
        ("Projected", t16["projected_unified_gib"], "#E67E22"),
        ("Host budget", t16["host_budget_gib"], "#2A6FBB"),
        ("CUDA budget", t16["device_budget_gib"], "#4B5563"),
    ]
    max_memory = 105.0
    for index, (label, value, color) in enumerate(memory_values):
        y = 125 + index * 90
        bar = 430 * float(value) / max_memory
        parts.append(f'<text x="95" y="{y + 23}" text-anchor="end" class="label">{label}</text>')
        parts.append(f'<rect x="110" y="{y}" width="430" height="36" fill="#EEF2F7"/>')
        parts.append(f'<rect x="110" y="{y}" width="{bar:.2f}" height="36" fill="{color}"/>')
        parts.append(f'<text x="{120 + bar:.2f}" y="{y + 24}" class="label">{value:.3f} GiB</text>')
    index_values = [
        ("T24 planning nnz", int(t24["planning_nnz"]), "#E67E22"),
        ("T32 planning nnz", int(t32["planning_nnz"]), "#B45309"),
        ("signed-int32 max", INT32_MAX, "#2A6FBB"),
    ]
    max_nnz = 3.6e9
    for index, (label, value, color) in enumerate(index_values):
        y = 125 + index * 90
        bar = 430 * value / max_nnz
        parts.append(f'<text x="750" y="{y + 23}" text-anchor="end" class="label">{label}</text>')
        parts.append(f'<rect x="765" y="{y}" width="430" height="36" fill="#EEF2F7"/>')
        parts.append(f'<rect x="765" y="{y}" width="{bar:.2f}" height="36" fill="{color}"/>')
        parts.append(
            f'<text x="{775 + bar:.2f}" y="{y + 24}" class="label">{value / 1e9:.3f} billion</text>'
        )
    parts.extend(
        [
            (
                '<text x="315" y="454" text-anchor="middle" class="note">'
                "94.435 GiB exceeded both unchanged 80% live budgets.</text>"
            ),
            (
                '<text x="960" y="454" text-anchor="middle" class="note">'
                "Both planning envelopes exceed 2,147,483,647.</text>"
            ),
            '<rect x="72" y="500" width="1136" height="82" rx="8" fill="#F3F4F6"/>',
            (
                '<text x="640" y="531" text-anchor="middle" class="sub">'
                "Scientific interpretation</text>"
            ),
            (
                '<text x="640" y="558" text-anchor="middle" class="label">'
                "The guards stopped before LP construction or solver allocation; these are "
                "measured safety outcomes, not OOM crashes.</text>"
            ),
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def generate() -> dict[str, Any]:
    stage7 = _load(STAGE7_PATH)
    stage8 = _load(STAGE8_PATH)
    continuation = _load(CONTINUATION_PATH)
    stage6 = _load(STAGE6_PATH)
    paper = _load(PAPER_METADATA_PATH)

    check_rows = _stage_check_rows()
    benchmark_rows = [_benchmark_row(7, case) for case in stage7["cases"]]
    benchmark_rows.extend(_benchmark_row(8, case) for case in stage8["cases"])
    benchmark_rows.extend(_continuation_row(case) for case in continuation["cases"])
    structural_rows = _structural_rows(stage7)
    resource_rows = _resource_rows(stage8, continuation)
    timing_rows = _timing_rows(stage6)

    _write_csv(TABLE_ROOT / "stage_9_stage_checks.csv", list(check_rows[0]), check_rows)
    _write_csv(TABLE_ROOT / "stage_9_benchmarks.csv", list(benchmark_rows[0]), benchmark_rows)
    _write_csv(
        TABLE_ROOT / "stage_9_structural_reconciliation.csv",
        list(structural_rows[0]),
        structural_rows,
    )
    _write_csv(
        TABLE_ROOT / "stage_9_resource_boundaries.csv", list(resource_rows[0]), resource_rows
    )
    _write_csv(TABLE_ROOT / "stage_9_timing_decomposition.csv", list(timing_rows[0]), timing_rows)

    _solver_timing_svg(benchmark_rows, PLOT_ROOT / "stage_9_solver_timings.svg")
    _resource_svg(resource_rows, PLOT_ROOT / "stage_9_resource_boundaries.svg")

    evidence_paths = [
        PAPER_METADATA_PATH,
        STAGE6_PATH,
        STAGE7_PATH,
        STAGE8_PATH,
        CONTINUATION_PATH,
        *STAGE_CHECK_PATHS,
    ]
    evidence_index = [
        {"path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in evidence_paths
    ]
    generated_paths = [
        TABLE_ROOT / "stage_9_stage_checks.csv",
        TABLE_ROOT / "stage_9_benchmarks.csv",
        TABLE_ROOT / "stage_9_structural_reconciliation.csv",
        TABLE_ROOT / "stage_9_resource_boundaries.csv",
        TABLE_ROOT / "stage_9_timing_decomposition.csv",
        PLOT_ROOT / "stage_9_solver_timings.svg",
        PLOT_ROOT / "stage_9_resource_boundaries.svg",
    ]

    allocated_rows = [row for row in benchmark_rows if row["full_lp_allocated"]]
    validated_gpu = [row for row in allocated_rows if row["gpu_status"] == "SUCCESS"]
    index: dict[str, Any] = {
        "schema_version": 1,
        "stage": 9,
        "generated_utc": "2026-08-05T00:00:00Z",
        "generator": "scripts/generate_stage_9_artifacts.py",
        "source_git": {
            "stage_9_base_commit": REPORT_BASE_COMMIT,
            "branch": "main",
            "self_reference_note": (
                "The final Stage 9 commit is reported after commit creation; it cannot be "
                "embedded in its own tree without changing that commit."
            ),
        },
        "paper": {
            "title": (
                "An Efficient GPU-based Halpern Accelerating Algorithm for Large-scale "
                "DC Optimal Power Flow"
            ),
            "doi": "10.1109/TPWRS.2025.3635652",
            "local_pdf": {
                "path": "references/AnEfficientGPU-basedHalpernAccelerating.pdf",
                "bytes": paper["byte_size"],
                "pages": paper["page_count"],
                "sha256": paper["sha256"],
            },
        },
        "final_classification": {
            "code": "D",
            "label": "structural reproduction",
            "decision_rule": "docs/stage_9_contract.md",
            "exact_reproduction": False,
            "paper_timing_reproduced": False,
            "local_speedup_claimed": False,
        },
        "stage_decisions": check_rows,
        "stage_8": {
            "scientific_result": "FAIL",
            "campaign_status": stage8["status"],
            "passing_prefix_length": stage8["stage_boundary"]["passing_prefix_length"],
            "allocated_rows": len(stage8["cases"]),
            "terminal_failure": "case9241pegase:T6 CPU FP64 correctness TIME_LIMIT",
            "cpu_timeout_seconds": stage8["cases"][-1]["solver_tracks"]["cpu_fp64_sgs_hpr"][
                "correctness"
            ]["wall_seconds"],
            "checker_result": "PASS",
            "checker_checks": "12/12",
            "continuation": {
                "status": continuation["status"],
                "resolved_rows": [case["key"] for case in continuation["cases"]],
                "allocation_attempts": len(continuation["allocation_history"]),
                "checker_result": "PASS",
                "checker_checks": "13/13",
            },
        },
        "coverage": {
            "symbolic_table_ii_rows": len(structural_rows),
            "dimension_matches": sum(row["dimension_match"] for row in structural_rows),
            "nnz_matches": sum(row["nnz_match"] for row in structural_rows),
            "allocated_benchmark_rows": len(allocated_rows),
            "validated_gpu_rows": len(validated_gpu),
            "validated_cpu_rows": sum(row["cpu_status"] == "SUCCESS" for row in allocated_rows),
            "resource_resolved_without_allocation": sum(
                row["row_status"] in {"MEMORY_BLOCKED", "INDEX_BLOCKED"} for row in benchmark_rows
            ),
        },
        "environment": {
            "platform": stage8["environment"]["platform"],
            "machine": stage8["environment"]["machine"],
            "python": stage8["environment"]["python"],
            "numpy": stage8["environment"]["numpy"],
            "scipy": stage8["environment"]["scipy"],
            "cupy": stage8["environment"]["packages"]["cupy-cuda13x"],
            "gpu": (stage8.get("resource_observations") or [{}])[0].get("device"),
            "total_unified_memory_bytes": stage8["environment"]["host_memory"]["total_bytes"],
        },
        "frozen_gates": {
            "normalized_stopping_block_max": 5e-5,
            "raw_kkt_max": 0.01,
            "physical_violation_max": 0.01,
            "scaled_objective_gap_max": 2e-4,
            "per_solve_deadline_seconds": 3600,
            "correctness_runs": 1,
            "warmup_runs": 1,
            "measured_repetitions": 5,
            "host_memory_safety_fraction": 0.8,
            "device_memory_safety_fraction": 0.8,
            "csr_index_max": INT32_MAX,
        },
        "tables": [
            {"path": _relative(path), "sha256": _sha256(path)} for path in generated_paths[:5]
        ],
        "figures": [
            {"path": _relative(path), "sha256": _sha256(path)} for path in generated_paths[5:]
        ],
        "evidence": evidence_index,
        "stage_10": {"status": "LOCKED", "n_minus_1_work_performed": False},
    }
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if regeneration changes outputs")
    args = parser.parse_args(argv)

    tracked_outputs = [
        INDEX_PATH,
        TABLE_ROOT / "stage_9_stage_checks.csv",
        TABLE_ROOT / "stage_9_benchmarks.csv",
        TABLE_ROOT / "stage_9_structural_reconciliation.csv",
        TABLE_ROOT / "stage_9_resource_boundaries.csv",
        TABLE_ROOT / "stage_9_timing_decomposition.csv",
        PLOT_ROOT / "stage_9_solver_timings.svg",
        PLOT_ROOT / "stage_9_resource_boundaries.svg",
    ]
    before = {path: path.read_bytes() if path.exists() else None for path in tracked_outputs}
    index = generate()
    changed = [path for path in tracked_outputs if before[path] != path.read_bytes()]
    if args.check and changed:
        print("Stage 9 generated outputs are stale:")
        for path in changed:
            print(f"- {_relative(path)}")
        return 1
    print(
        json.dumps(
            {
                "classification": index["final_classification"],
                "generated": [_relative(path) for path in tracked_outputs],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
