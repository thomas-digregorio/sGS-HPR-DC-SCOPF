"""Validated MATPOWER network data used by the Stage 2 CPU DCOPF model."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MATPOWERCaseError(ValueError):
    """Raised when a MATPOWER case cannot be represented safely."""


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise MATPOWERCaseError(f"{name} must be finite; received {result}.")
    return result


def _integer(value: float, *, name: str) -> int:
    result = int(round(float(value)))
    if not np.isclose(value, result, rtol=0.0, atol=1e-9):
        raise MATPOWERCaseError(f"{name} must be integer-valued; received {value}.")
    return result


@dataclass(frozen=True, slots=True)
class Bus:
    """The active-power fields needed from one MATPOWER bus row."""

    bus_id: int
    bus_type: int
    demand_mw: float
    shunt_conductance_mw: float

    @property
    def is_reference(self) -> bool:
        return self.bus_type == 3


@dataclass(frozen=True, slots=True)
class GeneratorCost:
    """One MATPOWER ``gencost`` row."""

    model: int
    startup: float
    shutdown: float
    parameters: tuple[float, ...]

    def paper_linear_terms(self) -> tuple[float, float, tuple[float, ...]]:
        """Return ``(slope, constant, omitted_higher_order_terms)``.

        The paper requires a linear objective. For a MATPOWER polynomial row,
        this method exposes the coefficient already attached to the linear
        term, plus its constant, while returning every higher-order coefficient
        so the caller can disclose that they were not represented.

        A piecewise-linear row is exactly representable only when it has one
        segment (two points).
        """

        if self.model == 2:
            if len(self.parameters) < 2:
                raise MATPOWERCaseError(
                    "A polynomial gencost row needs at least a linear and constant term."
                )
            return (
                float(self.parameters[-2]),
                float(self.parameters[-1]),
                tuple(float(value) for value in self.parameters[:-2]),
            )

        if self.model == 1:
            if len(self.parameters) != 4:
                raise MATPOWERCaseError(
                    "Only one-segment MATPOWER piecewise-linear costs are exactly "
                    "representable by the paper's linear objective."
                )
            x1, y1, x2, y2 = self.parameters
            if np.isclose(x1, x2):
                raise MATPOWERCaseError("Piecewise-linear cost points must have distinct x values.")
            slope = (y2 - y1) / (x2 - x1)
            return float(slope), float(y1 - slope * x1), ()

        raise MATPOWERCaseError(f"Unsupported MATPOWER gencost model {self.model}.")


@dataclass(frozen=True, slots=True)
class Generator:
    """The active-power fields needed from one MATPOWER generator row."""

    index: int
    bus_id: int
    initial_output_mw: float
    status: bool
    maximum_mw: float
    minimum_mw: float
    ramp_agc_mw_per_minute: float
    ramp_10_mw: float
    ramp_30_mw: float
    cost: GeneratorCost

    @property
    def generator_id(self) -> str:
        return f"gen_{self.index + 1}"


@dataclass(frozen=True, slots=True)
class Branch:
    """The DC fields needed from one MATPOWER branch row."""

    index: int
    from_bus: int
    to_bus: int
    resistance_pu: float
    reactance_pu: float
    line_charging_pu: float
    rate_a_mw: float
    tap_ratio: float
    phase_shift_degrees: float
    status: bool
    angle_minimum_degrees: float
    angle_maximum_degrees: float

    @property
    def branch_id(self) -> str:
        return f"branch_{self.index + 1}"

    @property
    def effective_tap_ratio(self) -> float:
        return 1.0 if np.isclose(self.tap_ratio, 0.0) else self.tap_ratio


@dataclass(frozen=True, slots=True)
class NetworkCase:
    """A validated, immutable subset of a MATPOWER case."""

    name: str
    base_mva: float
    buses: tuple[Bus, ...]
    generators: tuple[Generator, ...]
    branches: tuple[Branch, ...]
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.base_mva <= 0.0:
            raise MATPOWERCaseError("baseMVA must be positive.")
        if not self.buses:
            raise MATPOWERCaseError("The case must contain at least one bus.")

        bus_ids = [bus.bus_id for bus in self.buses]
        if len(set(bus_ids)) != len(bus_ids):
            raise MATPOWERCaseError("MATPOWER bus identifiers must be unique.")
        known_buses = set(bus_ids)
        for generator in self.generators:
            if generator.bus_id not in known_buses:
                raise MATPOWERCaseError(
                    f"Generator {generator.generator_id} references unknown bus {generator.bus_id}."
                )
            if generator.minimum_mw > generator.maximum_mw:
                raise MATPOWERCaseError(
                    f"Generator {generator.generator_id} has Pmin greater than Pmax."
                )
        for branch in self.branches:
            if branch.from_bus not in known_buses or branch.to_bus not in known_buses:
                raise MATPOWERCaseError(
                    f"Branch {branch.branch_id} references an unknown endpoint."
                )

        reference_buses = [bus.bus_id for bus in self.buses if bus.is_reference]
        if len(reference_buses) != 1:
            raise MATPOWERCaseError(
                f"Exactly one MATPOWER reference bus (type 3) is required; found {reference_buses}."
            )

    @property
    def reference_bus_id(self) -> int:
        return next(bus.bus_id for bus in self.buses if bus.is_reference)

    @property
    def bus_ids(self) -> tuple[int, ...]:
        return tuple(bus.bus_id for bus in self.buses)

    @property
    def active_generators(self) -> tuple[Generator, ...]:
        return tuple(generator for generator in self.generators if generator.status)

    @property
    def active_branches(self) -> tuple[Branch, ...]:
        return tuple(branch for branch in self.branches if branch.status)

    @property
    def demand_mw(self) -> np.ndarray:
        return np.asarray(
            [bus.demand_mw + bus.shunt_conductance_mw for bus in self.buses],
            dtype=np.float64,
        )

    def bus_position(self, bus_id: int) -> int:
        try:
            return self.bus_ids.index(bus_id)
        except ValueError as exc:
            raise KeyError(f"Unknown bus identifier {bus_id}.") from exc


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def _matrix(text: str, name: str) -> list[list[float]]:
    match = re.search(rf"\bmpc\.{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", text, re.DOTALL)
    if match is None:
        raise MATPOWERCaseError(f"Missing mpc.{name} matrix.")

    rows: list[list[float]] = []
    for row_number, raw_row in enumerate(match.group(1).split(";"), start=1):
        stripped = raw_row.strip()
        if not stripped:
            continue
        if "..." in stripped:
            raise MATPOWERCaseError(
                f"MATLAB continuation expressions are unsupported in mpc.{name} row {row_number}."
            )
        tokens = stripped.replace(",", " ").split()
        try:
            rows.append([float(token) for token in tokens])
        except ValueError as exc:
            raise MATPOWERCaseError(
                f"mpc.{name} row {row_number} contains a nonnumeric expression."
            ) from exc
    if not rows:
        raise MATPOWERCaseError(f"mpc.{name} must contain at least one row.")
    return rows


def _scalar(text: str, name: str) -> float:
    match = re.search(
        rf"\bmpc\.{re.escape(name)}\s*=\s*([-+0-9.eE]+)\s*;",
        text,
    )
    if match is None:
        raise MATPOWERCaseError(f"Missing numeric mpc.{name} assignment.")
    return _finite(float(match.group(1)), name=f"mpc.{name}")


def _case_name(text: str, path: Path) -> str:
    match = re.search(r"\bfunction\s+\w+\s*=\s*([A-Za-z]\w*)", text)
    return match.group(1) if match else path.stem


def _require_columns(row: list[float], minimum: int, *, matrix: str, row_number: int) -> None:
    if len(row) < minimum:
        raise MATPOWERCaseError(
            f"mpc.{matrix} row {row_number} needs at least {minimum} columns; received {len(row)}."
        )


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without modifying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matpower_case(path: str | Path) -> NetworkCase:
    """Parse the numeric core of a MATPOWER version-2 ``.m`` case.

    Supported tables are ``bus``, ``gen``, ``branch``, and ``gencost``.
    Arbitrary MATLAB code is deliberately not evaluated.
    """

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw_text = source.read_text(encoding="utf-8")
    text = _strip_comments(raw_text)

    bus_rows = _matrix(text, "bus")
    generator_rows = _matrix(text, "gen")
    branch_rows = _matrix(text, "branch")
    cost_rows = _matrix(text, "gencost")
    if len(generator_rows) != len(cost_rows):
        raise MATPOWERCaseError(
            "Stage 2 requires exactly one active-power gencost row per generator; "
            f"found {len(cost_rows)} costs for {len(generator_rows)} generators."
        )

    buses: list[Bus] = []
    for index, row in enumerate(bus_rows):
        _require_columns(row, 13, matrix="bus", row_number=index + 1)
        buses.append(
            Bus(
                bus_id=_integer(row[0], name=f"bus[{index}].bus_id"),
                bus_type=_integer(row[1], name=f"bus[{index}].type"),
                demand_mw=_finite(row[2], name=f"bus[{index}].Pd"),
                shunt_conductance_mw=_finite(row[4], name=f"bus[{index}].Gs"),
            )
        )

    costs: list[GeneratorCost] = []
    for index, row in enumerate(cost_rows):
        _require_columns(row, 4, matrix="gencost", row_number=index + 1)
        model = _integer(row[0], name=f"gencost[{index}].model")
        count = _integer(row[3], name=f"gencost[{index}].n")
        expected_parameters = count if model == 2 else 2 * count if model == 1 else -1
        if expected_parameters < 0:
            raise MATPOWERCaseError(f"Unsupported gencost model {model} in row {index + 1}.")
        if len(row[4:]) != expected_parameters:
            raise MATPOWERCaseError(
                f"gencost row {index + 1} declares {count} entries but has "
                f"{len(row[4:])} numeric parameters."
            )
        costs.append(
            GeneratorCost(
                model=model,
                startup=_finite(row[1], name=f"gencost[{index}].startup"),
                shutdown=_finite(row[2], name=f"gencost[{index}].shutdown"),
                parameters=tuple(
                    _finite(value, name=f"gencost[{index}].parameter") for value in row[4:]
                ),
            )
        )

    generators: list[Generator] = []
    for index, (row, cost) in enumerate(zip(generator_rows, costs, strict=True)):
        _require_columns(row, 10, matrix="gen", row_number=index + 1)
        generators.append(
            Generator(
                index=index,
                bus_id=_integer(row[0], name=f"gen[{index}].bus"),
                initial_output_mw=_finite(row[1], name=f"gen[{index}].Pg"),
                status=bool(_integer(row[7], name=f"gen[{index}].status")),
                maximum_mw=_finite(row[8], name=f"gen[{index}].Pmax"),
                minimum_mw=_finite(row[9], name=f"gen[{index}].Pmin"),
                ramp_agc_mw_per_minute=_finite(
                    row[16] if len(row) > 16 else 0.0,
                    name=f"gen[{index}].ramp_agc",
                ),
                ramp_10_mw=_finite(
                    row[17] if len(row) > 17 else 0.0,
                    name=f"gen[{index}].ramp_10",
                ),
                ramp_30_mw=_finite(
                    row[18] if len(row) > 18 else 0.0,
                    name=f"gen[{index}].ramp_30",
                ),
                cost=cost,
            )
        )

    branches: list[Branch] = []
    for index, row in enumerate(branch_rows):
        _require_columns(row, 11, matrix="branch", row_number=index + 1)
        branches.append(
            Branch(
                index=index,
                from_bus=_integer(row[0], name=f"branch[{index}].from_bus"),
                to_bus=_integer(row[1], name=f"branch[{index}].to_bus"),
                resistance_pu=_finite(row[2], name=f"branch[{index}].r"),
                reactance_pu=_finite(row[3], name=f"branch[{index}].x"),
                line_charging_pu=_finite(row[4], name=f"branch[{index}].b"),
                rate_a_mw=_finite(row[5], name=f"branch[{index}].rateA"),
                tap_ratio=_finite(row[8], name=f"branch[{index}].tap"),
                phase_shift_degrees=_finite(row[9], name=f"branch[{index}].shift"),
                status=bool(_integer(row[10], name=f"branch[{index}].status")),
                angle_minimum_degrees=_finite(
                    row[11] if len(row) > 11 else -360.0,
                    name=f"branch[{index}].angmin",
                ),
                angle_maximum_degrees=_finite(
                    row[12] if len(row) > 12 else 360.0,
                    name=f"branch[{index}].angmax",
                ),
            )
        )

    return NetworkCase(
        name=_case_name(text, source),
        base_mva=_scalar(text, "baseMVA"),
        buses=tuple(buses),
        generators=tuple(generators),
        branches=tuple(branches),
        source_path=str(source),
        source_sha256=sha256_file(source),
    )
