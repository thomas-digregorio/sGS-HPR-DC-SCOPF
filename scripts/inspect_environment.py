"""Collect a read-only, machine-readable CPU/GPU software environment audit.

The utility deliberately uses only the Python standard library for collection.
Optional numerical packages are inspected when present, but nothing is
installed or modified.  Gurobi's license is probed without recording license
paths, credentials, server addresses, or the text of license errors.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import ctypes.util
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


def run_command(
    command: list[str],
    *,
    timeout: float = 10.0,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run a read-only command and return bounded output."""
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"available": False, "returncode": None, "stdout": "", "stderr": ""}
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "timed_out": True,
        }
    except OSError:
        return {"available": False, "returncode": None, "stdout": "", "stderr": ""}

    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout[:100_000].strip(),
        "stderr": completed.stderr[:20_000].strip(),
    }


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def first_distribution_version(names: Iterable[str]) -> tuple[str | None, str | None]:
    for name in names:
        try:
            return name, importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None, None


def package_record(module_name: str, distributions: Iterable[str]) -> dict[str, Any]:
    distribution, version = first_distribution_version(distributions)
    return {
        "available": module_available(module_name),
        "module": module_name,
        "distribution": distribution,
        "version": version,
    }


def cpu_model() -> str | None:
    system = platform.system()
    if system == "Windows":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except (OSError, ImportError):
            pass

    if system == "Linux":
        lscpu = shutil.which("lscpu")
        if lscpu:
            result = run_command([lscpu], timeout=5)
            if result.get("returncode") == 0:
                model_names = []
                for line in result.get("stdout", "").splitlines():
                    if line.lower().startswith("model name:"):
                        model_name = line.split(":", 1)[1].strip()
                        if model_name and model_name not in model_names:
                            model_names.append(model_name)
                if model_names:
                    return " + ".join(model_names)
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            preferred = ("model name", "hardware", "processor")
            fields: dict[str, str] = {}
            for line in cpuinfo.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                fields.setdefault(key.strip().lower(), value.strip())
            for key in preferred:
                value = fields.get(key)
                if value and not value.isdigit():
                    return value
        except OSError:
            pass

    value = platform.processor().strip()
    return value or None


def physical_core_count() -> int | None:
    if module_available("psutil"):
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                psutil = importlib.import_module("psutil")
                count = psutil.cpu_count(logical=False)
            if count:
                return int(count)
        except Exception:
            pass

    if platform.system() == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            package_id: str | None = None
            core_id: str | None = None
            pairs: set[tuple[str, str]] = set()
            for line in [*cpuinfo.splitlines(), ""]:
                if not line.strip():
                    if package_id is not None and core_id is not None:
                        pairs.add((package_id, core_id))
                    package_id = None
                    core_id = None
                elif ":" in line:
                    key, value = (part.strip() for part in line.split(":", 1))
                    if key == "physical id":
                        package_id = value
                    elif key == "core id":
                        core_id = value
            if pairs:
                return len(pairs)
        except OSError:
            pass
    return None


def memory_report() -> dict[str, Any]:
    if module_available("psutil"):
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                psutil = importlib.import_module("psutil")
                memory = psutil.virtual_memory()
            return {
                "total_bytes": int(memory.total),
                "total_gib": round(memory.total / 2**30, 3),
                "available_bytes": int(memory.available),
                "available_gib": round(memory.available / 2**30, 3),
                "source": "psutil.virtual_memory",
            }
        except Exception:
            pass

    if platform.system() == "Linux":
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
            total = values["MemTotal"]
            available = values.get("MemAvailable")
            return {
                "total_bytes": total,
                "total_gib": round(total / 2**30, 3),
                "available_bytes": available,
                "available_gib": (round(available / 2**30, 3) if available is not None else None),
                "source": "/proc/meminfo",
            }
        except (OSError, KeyError, ValueError):
            pass

    if platform.system() == "Windows":
        try:

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.length = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return {
                    "total_bytes": int(status.total_physical),
                    "total_gib": round(status.total_physical / 2**30, 3),
                    "available_bytes": int(status.available_physical),
                    "available_gib": round(status.available_physical / 2**30, 3),
                    "source": "GlobalMemoryStatusEx",
                }
        except (AttributeError, OSError):
            pass

    return {
        "total_bytes": None,
        "total_gib": None,
        "available_bytes": None,
        "available_gib": None,
        "source": "unavailable",
    }


def operating_system_distribution() -> dict[str, str] | None:
    if platform.system() != "Linux":
        return None
    try:
        values: dict[str, str] = {}
        for line in (
            Path("/etc/os-release").read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip("\"'")
        selected = {
            key.lower(): values[key]
            for key in ("ID", "NAME", "PRETTY_NAME", "VERSION_ID", "VERSION")
            if key in values
        }
        return selected or None
    except OSError:
        return None


def parse_optional_number(value: str, cast: type[int] | type[float]) -> int | float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return cast(cleaned)
    except ValueError:
        return None


def nvidia_smi_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "available": False,
            "executable": None,
            "driver_supported_cuda_version": None,
            "gpus": [],
        }

    summary = run_command([executable], timeout=15)
    summary_text = summary.get("stdout", "")
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", summary_text)

    base_fields = [
        "name",
        "driver_version",
        "memory.total",
        "memory.free",
        "memory.used",
    ]
    fields = [*base_fields, "compute_cap"]
    query = run_command(
        [
            executable,
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if query.get("returncode") != 0:
        fields = base_fields
        query = run_command(
            [
                executable,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )

    gpus: list[dict[str, Any]] = []
    if query.get("returncode") == 0:
        for row in csv.reader(io.StringIO(query.get("stdout", ""))):
            if len(row) != len(fields):
                continue
            values = dict(zip(fields, (item.strip() for item in row), strict=True))
            gpus.append(
                {
                    "name": values["name"],
                    "driver_version": values["driver_version"],
                    "compute_capability": values.get("compute_cap"),
                    "addressing_mode": None,
                    "memory_total_mib": parse_optional_number(values["memory.total"], float),
                    "memory_free_mib": parse_optional_number(values["memory.free"], float),
                    "memory_used_mib": parse_optional_number(values["memory.used"], float),
                }
            )

    addressing_query = run_command(
        [
            executable,
            "--query-gpu=addressing_mode",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if addressing_query.get("returncode") == 0:
        addressing_modes = [
            line.strip() for line in addressing_query.get("stdout", "").splitlines()
        ]
        for gpu, addressing_mode in zip(gpus, addressing_modes, strict=False):
            gpu["addressing_mode"] = (
                None if addressing_mode.upper() in {"N/A", "[N/A]", "NONE"} else addressing_mode
            )

    return {
        "available": True,
        "executable": executable,
        "driver_supported_cuda_version": (cuda_match.group(1) if cuda_match else None),
        "query_succeeded": query.get("returncode") == 0,
        "gpus": gpus,
        "note": (
            "The CUDA version shown by nvidia-smi is the maximum CUDA version "
            "supported by the installed driver, not proof of a CUDA runtime toolkit. "
            "An ATS or HMM addressing mode means system-allocated memory is GPU "
            "addressable; it does not make all system RAM freely allocatable."
        ),
    }


def nvcc_report() -> dict[str, Any]:
    executable = shutil.which("nvcc")
    if executable is None:
        return {"available": False, "executable": None, "version": None}
    result = run_command([executable, "--version"], timeout=10)
    combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    match = re.search(r"release\s+([0-9.]+)", combined)
    return {
        "available": result.get("returncode") == 0,
        "executable": executable,
        "version": match.group(1) if match else None,
        "version_output": combined.strip(),
    }


def cuda_runtime_report() -> dict[str, Any]:
    candidates: list[str] = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)

    if platform.system() == "Windows":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            candidates.extend(str(path) for path in Path(cuda_path, "bin").glob("cudart64_*.dll"))
        candidates.extend(["cudart64_130.dll", "cudart64_12.dll", "cudart64_110.dll"])
    else:
        candidates.extend(["libcudart.so", "libcudart.so.13", "libcudart.so.12"])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            library = ctypes.CDLL(candidate)
            value = ctypes.c_int()
            status = library.cudaRuntimeGetVersion(ctypes.byref(value))
            if status == 0:
                raw = int(value.value)
                major = raw // 1000
                minor = (raw % 1000) // 10
                return {
                    "available": True,
                    "version": f"{major}.{minor}",
                    "raw_version": raw,
                    "library": str(candidate),
                }
        except (AttributeError, OSError):
            continue

    return {
        "available": False,
        "version": None,
        "raw_version": None,
        "library": None,
    }


def torch_report(base: dict[str, Any]) -> dict[str, Any]:
    report = dict(base)
    report.update(
        {
            "import_succeeded": False,
            "cuda_available": None,
            "cuda_build_version": None,
            "cudnn_version": None,
            "devices": [],
        }
    )
    if not base["available"]:
        return report
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            torch = importlib.import_module("torch")
            report["import_succeeded"] = True
            report["cuda_build_version"] = getattr(torch.version, "cuda", None)
            report["cuda_available"] = bool(torch.cuda.is_available())
            if report["cuda_available"]:
                report["cudnn_version"] = torch.backends.cudnn.version()
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    capability = torch.cuda.get_device_capability(index)
                    report["devices"].append(
                        {
                            "index": index,
                            "name": properties.name,
                            "compute_capability": f"{capability[0]}.{capability[1]}",
                            "total_memory_bytes": int(properties.total_memory),
                            "total_memory_gib": round(properties.total_memory / 2**30, 3),
                        }
                    )
    except Exception as exc:
        report["import_error_type"] = type(exc).__name__
    return report


def cupy_report(base: dict[str, Any]) -> dict[str, Any]:
    report = dict(base)
    report.update(
        {
            "import_succeeded": False,
            "cuda_available": None,
            "cuda_runtime_version": None,
            "cuda_driver_version": None,
            "devices": [],
        }
    )
    if not base["available"]:
        return report
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cupy = importlib.import_module("cupy")
            report["import_succeeded"] = True
            runtime = cupy.cuda.runtime
            device_count = int(runtime.getDeviceCount())
            report["cuda_available"] = device_count > 0
            report["cuda_runtime_version"] = int(runtime.runtimeGetVersion())
            report["cuda_driver_version"] = int(runtime.driverGetVersion())
            for index in range(device_count):
                properties = runtime.getDeviceProperties(index)
                name = properties.get("name")
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                report["devices"].append(
                    {
                        "index": index,
                        "name": str(name),
                        "compute_capability": (
                            f"{properties.get('major')}.{properties.get('minor')}"
                        ),
                        "total_global_memory_bytes": int(properties.get("totalGlobalMem", 0)),
                    }
                )
    except Exception as exc:
        report["import_error_type"] = type(exc).__name__
        if report["import_succeeded"]:
            report["cuda_available"] = False
    return report


def numba_report(base: dict[str, Any]) -> dict[str, Any]:
    report = dict(base)
    report["cuda_available"] = None
    if not base["available"]:
        return report
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            numba_cuda = importlib.import_module("numba.cuda")
            report["cuda_available"] = bool(numba_cuda.is_available())
    except Exception as exc:
        report["cuda_probe_error_type"] = type(exc).__name__
    return report


def jax_report(base: dict[str, Any]) -> dict[str, Any]:
    report = dict(base)
    report.update({"import_succeeded": False, "default_backend": None, "devices": []})
    if not base["available"]:
        return report
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            jax = importlib.import_module("jax")
            report["import_succeeded"] = True
            report["default_backend"] = jax.default_backend()
            report["devices"] = [
                {
                    "platform": getattr(device, "platform", None),
                    "device_kind": getattr(device, "device_kind", None),
                }
                for device in jax.devices()
            ]
    except Exception as exc:
        report["import_error_type"] = type(exc).__name__
    return report


def gurobi_report(base: dict[str, Any]) -> dict[str, Any]:
    report = dict(base)
    report.update(
        {
            "import_succeeded": False,
            "api_version": None,
            "license_check": {
                "attempted": False,
                "licensed": False,
                "status": "not_installed",
                "error_code": None,
                "details_suppressed": True,
            },
            "secrets_recorded": False,
        }
    )
    if not base["available"]:
        return report

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            gp = importlib.import_module("gurobipy")
            report["import_succeeded"] = True
            report["api_version"] = ".".join(str(part) for part in gp.gurobi.version())
            report["license_check"]["attempted"] = True
            environment = gp.Env(empty=True)
            try:
                environment.setParam("OutputFlag", 0)
                environment.start()
                model = gp.Model(env=environment)
                model.dispose()
                report["license_check"].update(
                    {
                        "licensed": True,
                        "status": "valid_for_model_creation",
                    }
                )
            finally:
                environment.dispose()
    except Exception as exc:
        report["license_check"].update(
            {
                "licensed": False,
                "status": (
                    "license_unavailable_or_invalid"
                    if report["import_succeeded"]
                    else "import_failed"
                ),
                "error_code": getattr(exc, "errno", None),
                "error_type": type(exc).__name__,
            }
        )
    return report


def isolated_package_probe(
    probe_name: str,
    base: dict[str, Any],
    *,
    timeout: float = 45.0,
) -> dict[str, Any]:
    """Probe one native package in a child process.

    Numerical Python wheels can bundle mutually incompatible OpenMP runtimes.
    Keeping each optional import in its own process prevents an environment
    audit from crashing merely because two installed packages cannot coexist
    in one interpreter.
    """
    if not base["available"]:
        if probe_name == "torch":
            return torch_report(base)
        if probe_name == "cupy":
            return cupy_report(base)
        if probe_name == "numba":
            return numba_report(base)
        if probe_name == "jax":
            return jax_report(base)
        if probe_name == "gurobi":
            return gurobi_report(base)
        return dict(base)

    result = run_command(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-probe",
            probe_name,
        ],
        timeout=timeout,
    )
    if result.get("returncode") == 0:
        try:
            parsed = json.loads(result.get("stdout", ""))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    report = dict(base)
    report.update(
        {
            "probe_succeeded": False,
            "probe_status": ("timed_out" if result.get("timed_out") else "child_process_failed"),
            "probe_returncode": result.get("returncode"),
            "probe_error_output_suppressed": True,
        }
    )
    if probe_name == "torch":
        report.update(
            {
                "import_succeeded": False,
                "cuda_available": None,
                "cuda_support_status": "unknown_because_import_failed",
                "cuda_build_version": None,
                "cudnn_version": None,
                "devices": [],
            }
        )
    elif probe_name == "cupy":
        report.update(
            {
                "import_succeeded": False,
                "cuda_available": None,
                "cuda_runtime_version": None,
                "cuda_driver_version": None,
                "devices": [],
            }
        )
    elif probe_name == "numba":
        report["cuda_available"] = None
    elif probe_name == "jax":
        report.update({"import_succeeded": False, "default_backend": None, "devices": []})
    if probe_name == "gurobi":
        report.update(
            {
                "import_succeeded": None,
                "api_version": None,
                "license_check": {
                    "attempted": True,
                    "licensed": False,
                    "status": "probe_failed_or_timed_out",
                    "error_code": None,
                    "details_suppressed": True,
                },
                "secrets_recorded": False,
            }
        )
    return report


def git_report() -> dict[str, Any]:
    executable = shutil.which("git")
    if executable is None:
        return {
            "available": False,
            "executable": None,
            "version": None,
            "repository": {"inside_work_tree": False, "commit": None, "dirty": None},
        }

    version_result = run_command([executable, "--version"])
    match = re.search(r"git version\s+(.+)", version_result.get("stdout", ""))
    inside_result = run_command([executable, "rev-parse", "--is-inside-work-tree"], timeout=5)
    inside = (
        inside_result.get("returncode") == 0
        and inside_result.get("stdout", "").strip().lower() == "true"
    )
    repository: dict[str, Any] = {
        "inside_work_tree": inside,
        "commit": None,
        "dirty": None,
    }
    if inside:
        commit_result = run_command([executable, "rev-parse", "HEAD"], timeout=5)
        if commit_result.get("returncode") == 0:
            repository["commit"] = commit_result.get("stdout", "").strip()
        status_result = run_command(
            [executable, "status", "--porcelain", "--untracked-files=normal"],
            timeout=10,
        )
        if status_result.get("returncode") == 0:
            repository["dirty"] = bool(status_result.get("stdout", "").strip())

    return {
        "available": version_result.get("returncode") == 0,
        "executable": executable,
        "version": match.group(1).strip() if match else None,
        "repository": repository,
    }


def collect_package_bases() -> dict[str, dict[str, Any]]:
    return {
        "numpy": package_record("numpy", ["numpy"]),
        "scipy": package_record("scipy", ["scipy"]),
        "highspy": package_record("highspy", ["highspy"]),
        "torch": package_record("torch", ["torch"]),
        "cupy": package_record(
            "cupy",
            [
                "cupy",
                "cupy-cuda13x",
                "cupy-cuda12x",
                "cupy-cuda11x",
            ],
        ),
        "numba": package_record("numba", ["numba"]),
        "jax": package_record("jax", ["jax"]),
        "jaxlib": package_record("jaxlib", ["jaxlib"]),
        "triton": package_record("triton", ["triton"]),
        "gurobipy": package_record("gurobipy", ["gurobipy"]),
        "psutil": package_record("psutil", ["psutil"]),
        "pynvml": package_record("pynvml", ["nvidia-ml-py", "pynvml"]),
        "pandas": package_record("pandas", ["pandas"]),
        "polars": package_record("polars", ["polars"]),
        "matplotlib": package_record("matplotlib", ["matplotlib"]),
        "pytest": package_record("pytest", ["pytest"]),
        "ruff": package_record("ruff", ["ruff"]),
        "pip": package_record("pip", ["pip"]),
    }


def run_internal_probe(probe_name: str) -> dict[str, Any]:
    package_bases = collect_package_bases()
    if probe_name == "torch":
        return torch_report(package_bases["torch"])
    if probe_name == "cupy":
        return cupy_report(package_bases["cupy"])
    if probe_name == "numba":
        return numba_report(package_bases["numba"])
    if probe_name == "jax":
        return jax_report(package_bases["jax"])
    if probe_name == "gurobi":
        return gurobi_report(package_bases["gurobipy"])
    raise ValueError(f"Unsupported internal probe: {probe_name}")


def collect_environment(label: str | None = None) -> dict[str, Any]:
    package_bases = collect_package_bases()
    nvidia_smi = nvidia_smi_report()
    nvcc = nvcc_report()
    cudart = cuda_runtime_report()

    packages = dict(package_bases)
    packages["torch"] = isolated_package_probe("torch", package_bases["torch"])
    packages["cupy"] = isolated_package_probe("cupy", package_bases["cupy"])
    packages["numba"] = isolated_package_probe("numba", package_bases["numba"])
    packages["jax"] = isolated_package_probe("jax", package_bases["jax"])
    packages["gurobipy"] = isolated_package_probe("gurobi", package_bases["gurobipy"], timeout=30.0)

    cuda_runtime_candidates = []
    if cudart["version"]:
        cuda_runtime_candidates.append({"source": "libcudart", "version": cudart["version"]})
    if packages["torch"].get("cuda_build_version"):
        cuda_runtime_candidates.append(
            {
                "source": "PyTorch build",
                "version": packages["torch"]["cuda_build_version"],
            }
        )
    if packages["cupy"].get("cuda_runtime_version"):
        raw = int(packages["cupy"]["cuda_runtime_version"])
        cuda_runtime_candidates.append(
            {
                "source": "CuPy runtime API",
                "version": f"{raw // 1000}.{(raw % 1000) // 10}",
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "label": label,
        "host": {"hostname": platform.node()},
        "operating_system": {
            "system": platform.system(),
            "distribution": operating_system_distribution(),
            "release": platform.release(),
            "version": platform.version(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "python_process_bits": struct.calcsize("P") * 8,
            "platform": platform.platform(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": {
            "model": cpu_model(),
            "physical_cores": physical_core_count(),
            "logical_cores": os.cpu_count(),
        },
        "memory": memory_report(),
        "nvidia": {
            "nvidia_smi": nvidia_smi,
            "driver_version": (
                nvidia_smi["gpus"][0].get("driver_version") if nvidia_smi["gpus"] else None
            ),
            "gpu_memory_report": nvidia_smi["gpus"],
            "memory_interpretation": (
                "nvidia-smi device memory is reported verbatim. On an integrated "
                "CPU/GPU platform this may be backed by shared system memory; this "
                "script does not infer coherence or reservation semantics."
            ),
        },
        "cuda": {
            "driver_supported_version": nvidia_smi.get("driver_supported_cuda_version"),
            "runtime": cudart,
            "runtime_candidates": cuda_runtime_candidates,
            "nvcc": nvcc,
        },
        "packages": packages,
        "gurobi": packages["gurobipy"],
        "git": git_report(),
        "collection_policy": {
            "read_only": True,
            "packages_installed": False,
            "license_paths_or_credentials_recorded": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="Optional human-readable machine label.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to also write the JSON result.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent the emitted JSON.",
    )
    parser.add_argument(
        "--internal-probe",
        choices=("torch", "cupy", "numba", "jax", "gurobi"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.internal_probe:
        print(json.dumps(run_internal_probe(args.internal_probe), sort_keys=True))
        return 0
    report = collect_environment(args.label)
    indent = 2 if args.pretty else None
    rendered = json.dumps(report, indent=indent, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
