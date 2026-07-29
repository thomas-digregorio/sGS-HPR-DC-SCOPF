"""Build, solve, and preserve the complete Stage 2 DCOPF validation evidence."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gpu_dcopf_hpr.dcopf_model import (  # noqa: E402
    DCOPFModel,
    build_dcopf_model,
    load_dcopf_config,
)
from gpu_dcopf_hpr.network_data import load_matpower_case, sha256_file  # noqa: E402
from gpu_dcopf_hpr.ptdf import build_ptdf  # noqa: E402
from gpu_dcopf_hpr.validation import (  # noqa: E402
    solve_with_highs,
    validate_dcopf_solution,
)

DEFAULT_NETWORK = PROJECT_ROOT / "data" / "raw" / "matpower" / "case5.m"
DEFAULT_CONFIGS = (
    PROJECT_ROOT / "configs" / "dcopf" / "case5_base_stage_2.json",
    PROJECT_ROOT / "configs" / "dcopf" / "case5_synthetic_extension_stage_2.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "raw" / "stage_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--config", type=Path, action="append", dest="configs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def ptdf_probe_summary(model: DCOPFModel) -> dict[str, Any]:
    """Compare every unit transfer column with a reduced angle solve."""

    reference = model.ptdf.reference_position
    errors: list[float] = []
    probes: list[dict[str, Any]] = []
    for bus_position, bus_id in enumerate(model.ptdf.bus_ids):
        if bus_position == reference:
            continue
        injection = np.zeros(len(model.ptdf.bus_ids), dtype=np.float64)
        injection[bus_position] = 1.0
        injection[reference] = -1.0
        ptdf_flow = model.ptdf.flows_from_injections(injection)
        _, angle_flow = model.ptdf.angles_and_flows(injection)
        error = float(np.max(np.abs(ptdf_flow - angle_flow), initial=0.0))
        errors.append(error)
        probes.append(
            {
                "injection_bus": bus_id,
                "withdrawal_bus": model.ptdf.reference_bus_id,
                "maximum_flow_difference_mw": error,
            }
        )
    return {
        "maximum_flow_difference_mw": max(errors, default=0.0),
        "probes": probes,
    }


def write_row_metadata(model: DCOPFModel, path: Path) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for block, rows in (
            ("A1", model.equality_rows),
            ("A2", model.inequality_rows),
        ):
            for row_index, metadata in enumerate(rows):
                record = {
                    "case": model.config.name,
                    "block": block,
                    "row": row_index,
                    **metadata.as_dict(),
                }
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
                stream.write("\n")
                count += 1
    return count


def case_summary(model: DCOPFModel, metadata_path: Path) -> dict[str, Any]:
    solution = solve_with_highs(model.lp, tolerance=1e-7)
    repeated = solve_with_highs(model.lp, tolerance=1e-7)
    validation = validate_dcopf_solution(model, solution.state.x)
    probe = ptdf_probe_summary(model)
    metadata_rows = write_row_metadata(model, metadata_path)
    expected_dimensions = model.expected_dimensions()
    actual_dimensions = {
        "n": model.lp.n,
        "m1": model.lp.m1,
        "m2": model.lp.m2,
        "m": model.lp.m,
    }
    deterministic = (
        np.array_equal(solution.state.x, repeated.state.x)
        and np.array_equal(solution.state.y, repeated.state.y)
        and np.array_equal(solution.state.z, repeated.state.z)
        and solution.iterations == repeated.iterations
    )

    period_results: list[dict[str, Any]] = []
    blocks = model.unpack(solution.state.x)
    for period in range(model.config.periods):
        injection = model.bus_injections(solution.state.x, period)
        flows = model.ptdf.flows_from_injections(injection)
        period_results.append(
            {
                "period": period,
                "load_mw": float(np.sum(model.load_mw[period])),
                "generation_mw": blocks["p_g"][period].tolist(),
                "renewable_mw": blocks["p_rg"][period].tolist(),
                "storage_discharge_mw": blocks["p_ess_dc"][period].tolist(),
                "storage_charge_mw": blocks["p_ess_ch"][period].tolist(),
                "reserve_up_mw": blocks["r_up"][period].tolist(),
                "reserve_down_mw": blocks["r_down"][period].tolist(),
                "bus_injection_mw": injection.tolist(),
                "branch_flow_mw": flows.tolist(),
            }
        )

    checks = {
        "highs_success": solution.status == 0,
        "formula_dimensions_match": actual_dimensions == expected_dimensions,
        "independent_physical_validation": validation.passed,
        "ptdf_matches_angle_flows": probe["maximum_flow_difference_mw"] <= 1e-10,
        "row_metadata_complete": metadata_rows == model.lp.m,
        "repeated_highs_solution_deterministic": deterministic,
        "synthetic_resources_labeled": (
            not (model.config.renewables or model.config.storage)
            or model.config.synthetic_extension
        ),
    }
    return {
        "name": model.config.name,
        "classification": model.config.classification,
        "synthetic_extension": model.config.synthetic_extension,
        "notes": list(model.config.notes),
        "dimensions": model.dimension_summary(),
        "expected_dimensions": expected_dimensions,
        "objective": {
            "canonical_variable_part": float(solution.objective),
            "constant": model.objective_constant,
            "total": model.objective(solution.state.x),
        },
        "highs": solution.summary(),
        "independent_validation": validation.summary(),
        "ptdf_angle_probes": probe,
        "row_metadata_file": metadata_path.name,
        "row_metadata_records": metadata_rows,
        "period_results": period_results,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    network_path = args.network.resolve()
    config_paths = (
        tuple(path.resolve() for path in args.configs)
        if args.configs
        else tuple(path.resolve() for path in DEFAULT_CONFIGS)
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    network = load_matpower_case(network_path)
    ptdf = build_ptdf(network)
    cases: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = load_dcopf_config(config_path, network)
        model = build_dcopf_model(network, config, ptdf=ptdf)
        metadata_path = output_dir / f"{config.name}_row_metadata.jsonl.gz"
        cases.append(case_summary(model, metadata_path))

    summary = {
        "stage": 2,
        "all_passed": all(case["passed"] for case in cases),
        "network_source": {
            "name": network.name,
            "local_path": str(network_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(network_path),
            "upstream_project": "MATPOWER/matpower",
            "upstream_release": "8.1",
            "upstream_blob": "b6370ab230ac5346023d23be20d973a81f09e12a",
            "upstream_url": "https://github.com/MATPOWER/matpower/blob/8.1/data/case5.m",
            "buses": len(network.buses),
            "active_generators": len(network.active_generators),
            "active_topology_branches": len(network.active_branches),
            "thermally_constrained_branches": sum(
                branch.rate_a_mw > 0.0 for branch in network.active_branches
            ),
            "reference_bus": network.reference_bus_id,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "highs_interface": "scipy.optimize.linprog(method='highs-ds')",
            "highspy_installed": importlib.util.find_spec("highspy") is not None,
            "platform": platform.platform(),
            "precision": "FP64",
        },
        "cases": cases,
    }
    output_path = output_dir / "stage_2_validation.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
