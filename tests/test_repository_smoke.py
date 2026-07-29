from pathlib import Path

import gpu_dcopf_hpr

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports() -> None:
    assert gpu_dcopf_hpr.__version__ == "0.0.0"


def test_stage_zero_required_paths_exist() -> None:
    required_paths = [
        PROJECT_ROOT / "references" / "AnEfficientGPU-basedHalpernAccelerating.pdf",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "environment",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "src" / "gpu_dcopf_hpr",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "artifacts",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_paths if not path.exists()]
    assert not missing, f"Missing Stage 0 repository paths: {missing}"
