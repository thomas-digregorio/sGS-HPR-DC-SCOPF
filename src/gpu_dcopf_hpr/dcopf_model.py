"""Sparse CPU construction of the paper's multi-period DCOPF linear program."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from .canonical_lp import CanonicalLP
from .network_data import Branch, Generator, MATPOWERCaseError, NetworkCase
from .ptdf import PTDF, build_ptdf

VariableKind = Literal["p_g", "p_rg", "p_ess_dc", "p_ess_ch", "r_up", "r_down"]
FloatArray = NDArray[np.float64]


def _finite_tuple(
    values: ArrayLike,
    *,
    name: str,
    length: int,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have length {length}; received shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    if nonnegative and np.any(vector < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return tuple(float(value) for value in vector)


@dataclass(frozen=True, slots=True)
class RenewableSpec:
    """One explicitly configured renewable resource."""

    resource_id: str
    bus_id: int
    minimum_mw: tuple[float, ...]
    maximum_mw: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StorageSpec:
    """One explicitly configured storage resource."""

    resource_id: str
    bus_id: int
    initial_energy_mwh: float
    minimum_energy_mwh: float
    maximum_energy_mwh: float
    maximum_charge_mw: float
    maximum_discharge_mw: float
    charge_efficiency: float
    discharge_efficiency: float


@dataclass(frozen=True, slots=True)
class DCOPFConfig:
    """Versioned operating data that are not available in a MATPOWER case."""

    name: str
    classification: str
    synthetic_extension: bool
    periods: int
    interval_hours: float
    load_multipliers: tuple[float, ...]
    reserve_up_mw: tuple[float, ...]
    reserve_down_mw: tuple[float, ...]
    generator_ramp_up_mw_per_hour: tuple[float, ...]
    generator_ramp_down_mw_per_hour: tuple[float, ...]
    renewable_penalty_per_mwh: float
    storage_loss_penalty_per_mwh: float
    renewables: tuple[RenewableSpec, ...]
    storage: tuple[StorageSpec, ...]
    cost_mode: str
    notes: tuple[str, ...]


def load_dcopf_config(path: str | Path, network: NetworkCase) -> DCOPFConfig:
    """Load and validate an explicit Stage 2 JSON configuration."""

    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    periods = int(raw["periods"])
    if periods <= 0:
        raise ValueError("periods must be positive.")
    interval_hours = float(raw["interval_hours"])
    if not np.isfinite(interval_hours) or interval_hours <= 0.0:
        raise ValueError("interval_hours must be finite and positive.")

    generator_count = len(network.active_generators)
    load_multipliers = _finite_tuple(
        raw["load_multipliers"],
        name="load_multipliers",
        length=periods,
        nonnegative=True,
    )
    reserve_up = _finite_tuple(
        raw["reserve_up_mw"],
        name="reserve_up_mw",
        length=periods,
        nonnegative=True,
    )
    reserve_down = _finite_tuple(
        raw["reserve_down_mw"],
        name="reserve_down_mw",
        length=periods,
        nonnegative=True,
    )
    ramp_up = _finite_tuple(
        raw["generator_ramp_up_mw_per_hour"],
        name="generator_ramp_up_mw_per_hour",
        length=generator_count,
        nonnegative=True,
    )
    ramp_down = _finite_tuple(
        raw["generator_ramp_down_mw_per_hour"],
        name="generator_ramp_down_mw_per_hour",
        length=generator_count,
        nonnegative=True,
    )

    known_buses = set(network.bus_ids)
    renewables: list[RenewableSpec] = []
    renewable_ids: set[str] = set()
    for entry in raw.get("renewables", []):
        resource_id = str(entry["id"])
        bus_id = int(entry["bus_id"])
        if resource_id in renewable_ids:
            raise ValueError(f"Duplicate renewable identifier {resource_id}.")
        if bus_id not in known_buses:
            raise ValueError(f"Renewable {resource_id} references unknown bus {bus_id}.")
        minimum = _finite_tuple(
            entry["minimum_mw"],
            name=f"{resource_id}.minimum_mw",
            length=periods,
            nonnegative=True,
        )
        maximum = _finite_tuple(
            entry["maximum_mw"],
            name=f"{resource_id}.maximum_mw",
            length=periods,
            nonnegative=True,
        )
        if any(lower > upper for lower, upper in zip(minimum, maximum, strict=True)):
            raise ValueError(f"Renewable {resource_id} has a minimum above its maximum.")
        renewable_ids.add(resource_id)
        renewables.append(
            RenewableSpec(
                resource_id=resource_id,
                bus_id=bus_id,
                minimum_mw=minimum,
                maximum_mw=maximum,
            )
        )

    storage: list[StorageSpec] = []
    storage_ids: set[str] = set()
    for entry in raw.get("storage", []):
        resource_id = str(entry["id"])
        bus_id = int(entry["bus_id"])
        if resource_id in storage_ids:
            raise ValueError(f"Duplicate storage identifier {resource_id}.")
        if bus_id not in known_buses:
            raise ValueError(f"Storage {resource_id} references unknown bus {bus_id}.")
        initial = float(entry["initial_energy_mwh"])
        minimum = float(entry["minimum_energy_mwh"])
        maximum = float(entry["maximum_energy_mwh"])
        maximum_charge = float(entry["maximum_charge_mw"])
        maximum_discharge = float(entry["maximum_discharge_mw"])
        charge_efficiency = float(entry["charge_efficiency"])
        discharge_efficiency = float(entry["discharge_efficiency"])
        values = (
            initial,
            minimum,
            maximum,
            maximum_charge,
            maximum_discharge,
            charge_efficiency,
            discharge_efficiency,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Storage {resource_id} contains a nonfinite value.")
        if not minimum <= initial <= maximum:
            raise ValueError(f"Storage {resource_id} initial energy lies outside its bounds.")
        if maximum_charge < 0.0 or maximum_discharge < 0.0:
            raise ValueError(f"Storage {resource_id} power limits must be nonnegative.")
        if not 0.0 < charge_efficiency <= 1.0:
            raise ValueError(f"Storage {resource_id} charge efficiency must lie in (0, 1].")
        if not 0.0 < discharge_efficiency <= 1.0:
            raise ValueError(f"Storage {resource_id} discharge efficiency must lie in (0, 1].")
        storage_ids.add(resource_id)
        storage.append(
            StorageSpec(
                resource_id=resource_id,
                bus_id=bus_id,
                initial_energy_mwh=initial,
                minimum_energy_mwh=minimum,
                maximum_energy_mwh=maximum,
                maximum_charge_mw=maximum_charge,
                maximum_discharge_mw=maximum_discharge,
                charge_efficiency=charge_efficiency,
                discharge_efficiency=discharge_efficiency,
            )
        )

    renewable_penalty = float(raw.get("renewable_penalty_per_mwh", 0.0))
    storage_penalty = float(raw.get("storage_loss_penalty_per_mwh", 0.0))
    if (
        not np.isfinite(renewable_penalty)
        or not np.isfinite(storage_penalty)
        or renewable_penalty < 0.0
        or storage_penalty < 0.0
    ):
        raise ValueError("Renewable and storage penalties must be finite and nonnegative.")

    synthetic_extension = bool(raw["synthetic_extension"])
    if (renewables or storage) and not synthetic_extension:
        raise ValueError(
            "Configurations with renewable or storage additions must set synthetic_extension=true."
        )
    cost_mode = str(raw.get("cost_mode", "matpower_exact_linear"))
    if cost_mode != "matpower_exact_linear":
        raise ValueError(
            f"Stage 2 supports only exact MATPOWER linear costs; received cost_mode={cost_mode!r}."
        )

    return DCOPFConfig(
        name=str(raw["name"]),
        classification=str(raw["classification"]),
        synthetic_extension=synthetic_extension,
        periods=periods,
        interval_hours=interval_hours,
        load_multipliers=load_multipliers,
        reserve_up_mw=reserve_up,
        reserve_down_mw=reserve_down,
        generator_ramp_up_mw_per_hour=ramp_up,
        generator_ramp_down_mw_per_hour=ramp_down,
        renewable_penalty_per_mwh=renewable_penalty,
        storage_loss_penalty_per_mwh=storage_penalty,
        renewables=tuple(renewables),
        storage=tuple(storage),
        cost_mode=cost_mode,
        notes=tuple(str(note) for note in raw.get("notes", [])),
    )


@dataclass(frozen=True, slots=True)
class VariableKey:
    kind: VariableKind
    period: int
    device_id: str


class VariableIndex:
    """Deterministic bidirectional semantic-to-vector indexing."""

    _block_order: tuple[VariableKind, ...] = (
        "p_g",
        "p_rg",
        "p_ess_dc",
        "p_ess_ch",
        "r_up",
        "r_down",
    )

    def __init__(
        self,
        periods: int,
        generator_ids: tuple[str, ...],
        renewable_ids: tuple[str, ...],
        storage_ids: tuple[str, ...],
    ) -> None:
        devices: dict[VariableKind, tuple[str, ...]] = {
            "p_g": generator_ids,
            "p_rg": renewable_ids,
            "p_ess_dc": storage_ids,
            "p_ess_ch": storage_ids,
            "r_up": generator_ids,
            "r_down": generator_ids,
        }
        keys: list[VariableKey] = []
        for kind in self._block_order:
            for period in range(periods):
                keys.extend(
                    VariableKey(kind=kind, period=period, device_id=device_id)
                    for device_id in devices[kind]
                )
        self._keys = tuple(keys)
        mapping = {key: index for index, key in enumerate(self._keys)}
        if len(mapping) != len(self._keys):
            raise ValueError("Variable keys must be unique.")
        self._mapping = MappingProxyType(mapping)

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> tuple[VariableKey, ...]:
        return self._keys

    def index(self, kind: VariableKind, period: int, device_id: str) -> int:
        try:
            return self._mapping[VariableKey(kind, period, device_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown variable ({kind}, {period}, {device_id}).") from exc

    def key(self, index: int) -> VariableKey:
        return self._keys[index]


@dataclass(frozen=True, slots=True)
class ConstraintRow:
    """Traceability metadata for one scalar canonical row."""

    family: str
    period: int | None
    element_id: str
    side: str
    equation: str

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "family": self.family,
            "period": self.period,
            "element_id": self.element_id,
            "side": self.side,
            "equation": self.equation,
        }


@dataclass(frozen=True, slots=True)
class DCOPFModel:
    """The canonical LP plus the physical data needed for validation."""

    network: NetworkCase
    config: DCOPFConfig
    ptdf: PTDF
    variables: VariableIndex
    lp: CanonicalLP
    equality_rows: tuple[ConstraintRow, ...]
    inequality_rows: tuple[ConstraintRow, ...]
    constrained_branches: tuple[Branch, ...]
    load_mw: FloatArray
    objective_constant: float

    def __post_init__(self) -> None:
        load = np.asarray(self.load_mw, dtype=np.float64).copy()
        expected = (self.config.periods, len(self.network.buses))
        if load.shape != expected:
            raise ValueError(f"load_mw has shape {load.shape}; expected {expected}.")
        load.setflags(write=False)
        object.__setattr__(self, "load_mw", load)
        if len(self.equality_rows) != self.lp.m1:
            raise ValueError("Equality metadata count does not match A1.")
        if len(self.inequality_rows) != self.lp.m2:
            raise ValueError("Inequality metadata count does not match A2.")

    @property
    def generators(self) -> tuple[Generator, ...]:
        return self.network.active_generators

    def expected_dimensions(self) -> dict[str, int]:
        periods = self.config.periods
        generators = len(self.generators)
        constrained_lines = len(self.constrained_branches)
        renewables = len(self.config.renewables)
        storage = len(self.config.storage)
        n = periods * (3 * generators + renewables + 2 * storage)
        m1 = periods + storage
        m2 = (
            2 * periods * constrained_lines
            + 2 * periods * generators
            + 2 * periods
            + 2 * (periods - 1) * generators
            + 2 * periods * storage
        )
        return {"n": n, "m1": m1, "m2": m2, "m": m1 + m2}

    def dimension_summary(self) -> dict[str, int]:
        return {
            "periods": self.config.periods,
            "buses": len(self.network.buses),
            "generators": len(self.generators),
            "renewables": len(self.config.renewables),
            "storage": len(self.config.storage),
            "active_topology_branches": len(self.network.active_branches),
            "thermally_constrained_branches": len(self.constrained_branches),
            "n": self.lp.n,
            "m1": self.lp.m1,
            "m2": self.lp.m2,
            "m": self.lp.m,
            "nnz_A1": int(self.lp.A1.nnz),
            "nnz_A2": int(self.lp.A2.nnz),
        }

    def objective(self, x: ArrayLike, *, include_constant: bool = True) -> float:
        vector = np.asarray(x, dtype=np.float64)
        result = float(self.lp.c @ vector)
        return result + self.objective_constant if include_constant else result

    def unpack(self, x: ArrayLike) -> dict[str, FloatArray]:
        vector = np.asarray(x, dtype=np.float64)
        if vector.shape != (self.lp.n,):
            raise ValueError(f"x has shape {vector.shape}; expected {(self.lp.n,)}.")
        device_ids: dict[VariableKind, tuple[str, ...]] = {
            "p_g": tuple(generator.generator_id for generator in self.generators),
            "p_rg": tuple(resource.resource_id for resource in self.config.renewables),
            "p_ess_dc": tuple(resource.resource_id for resource in self.config.storage),
            "p_ess_ch": tuple(resource.resource_id for resource in self.config.storage),
            "r_up": tuple(generator.generator_id for generator in self.generators),
            "r_down": tuple(generator.generator_id for generator in self.generators),
        }
        result: dict[str, FloatArray] = {}
        for kind in VariableIndex._block_order:
            values = np.empty((self.config.periods, len(device_ids[kind])), dtype=np.float64)
            for period in range(self.config.periods):
                for column, device_id in enumerate(device_ids[kind]):
                    values[period, column] = vector[self.variables.index(kind, period, device_id)]
            result[kind] = values
        return result

    def bus_injections(self, x: ArrayLike, period: int) -> FloatArray:
        """Return the independently assembled balanced net bus injections."""

        blocks = self.unpack(x)
        injection = -self.load_mw[period].copy()
        for column, generator in enumerate(self.generators):
            injection[self.network.bus_position(generator.bus_id)] += blocks["p_g"][period, column]
        for column, renewable in enumerate(self.config.renewables):
            injection[self.network.bus_position(renewable.bus_id)] += blocks["p_rg"][period, column]
        for column, storage in enumerate(self.config.storage):
            position = self.network.bus_position(storage.bus_id)
            injection[position] += (
                blocks["p_ess_dc"][period, column] - blocks["p_ess_ch"][period, column]
            )
        return injection


def _add_coefficient(row: dict[int, float], index: int, value: float) -> None:
    if value:
        row[index] = row.get(index, 0.0) + float(value)


def _sparse_rows(
    rows: list[dict[int, float]],
    *,
    columns: int,
) -> sparse.csr_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows):
        for column_index, value in sorted(row.items()):
            if value:
                row_indices.append(row_index)
                column_indices.append(column_index)
                values.append(value)
    return sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), columns),
        dtype=np.float64,
    ).tocsr()


def build_dcopf_model(
    network: NetworkCase,
    config: DCOPFConfig,
    *,
    ptdf: PTDF | None = None,
) -> DCOPFModel:
    """Construct every paper constraint family as a sparse canonical LP."""

    shift_factors = build_ptdf(network) if ptdf is None else ptdf
    generators = network.active_generators
    active_angle_limits = [
        branch.branch_id
        for branch in network.active_branches
        if branch.angle_minimum_degrees > -360.0 or branch.angle_maximum_degrees < 360.0
    ]
    if active_angle_limits:
        raise MATPOWERCaseError(
            "Active MATPOWER branch-angle limits are outside the paper's printed model: "
            f"{active_angle_limits}."
        )
    constrained_branches = tuple(
        branch for branch in network.active_branches if 0.0 < branch.rate_a_mw < 1e10
    )
    if any(branch.rate_a_mw < 0.0 for branch in network.active_branches):
        raise MATPOWERCaseError("MATPOWER rateA values must be nonnegative.")
    if not constrained_branches:
        raise MATPOWERCaseError("At least one active branch must have a positive rateA limit.")

    variables = VariableIndex(
        config.periods,
        tuple(generator.generator_id for generator in generators),
        tuple(resource.resource_id for resource in config.renewables),
        tuple(resource.resource_id for resource in config.storage),
    )
    expected_n = config.periods * (
        3 * len(generators) + len(config.renewables) + 2 * len(config.storage)
    )
    if len(variables) != expected_n:
        raise AssertionError("Variable indexing does not satisfy the paper's dimension formula.")

    load_mw = np.vstack([network.demand_mw * multiplier for multiplier in config.load_multipliers])
    lower = np.zeros(len(variables), dtype=np.float64)
    upper = np.zeros(len(variables), dtype=np.float64)
    cost = np.zeros(len(variables), dtype=np.float64)
    objective_constant = 0.0

    for generator, ramp_up, ramp_down in zip(
        generators,
        config.generator_ramp_up_mw_per_hour,
        config.generator_ramp_down_mw_per_hour,
        strict=True,
    ):
        slope, constant, omitted = generator.cost.paper_linear_terms()
        if omitted and any(abs(value) > 0.0 for value in omitted):
            raise MATPOWERCaseError(
                f"{generator.generator_id} has nonzero higher-order cost coefficients {omitted}; "
                "Stage 2 will not silently linearize them."
            )
        objective_constant += config.periods * constant
        for period in range(config.periods):
            pg = variables.index("p_g", period, generator.generator_id)
            ru = variables.index("r_up", period, generator.generator_id)
            rd = variables.index("r_down", period, generator.generator_id)
            lower[pg], upper[pg], cost[pg] = generator.minimum_mw, generator.maximum_mw, slope
            lower[ru], upper[ru] = 0.0, ramp_up * config.interval_hours
            lower[rd], upper[rd] = 0.0, ramp_down * config.interval_hours

    for renewable in config.renewables:
        for period in range(config.periods):
            index = variables.index("p_rg", period, renewable.resource_id)
            lower[index] = renewable.minimum_mw[period]
            upper[index] = renewable.maximum_mw[period]
            cost[index] = -config.renewable_penalty_per_mwh
            objective_constant += config.renewable_penalty_per_mwh * renewable.maximum_mw[period]

    for storage in config.storage:
        for period in range(config.periods):
            discharge = variables.index("p_ess_dc", period, storage.resource_id)
            charge = variables.index("p_ess_ch", period, storage.resource_id)
            lower[discharge], upper[discharge] = 0.0, storage.maximum_discharge_mw
            lower[charge], upper[charge] = 0.0, storage.maximum_charge_mw
            cost[discharge] = config.storage_loss_penalty_per_mwh * (
                1.0 / storage.discharge_efficiency - 1.0
            )
            cost[charge] = config.storage_loss_penalty_per_mwh * (1.0 - storage.charge_efficiency)

    equality_rows: list[dict[int, float]] = []
    equality_rhs: list[float] = []
    equality_metadata: list[ConstraintRow] = []

    for period in range(config.periods):
        row: dict[int, float] = {}
        for generator in generators:
            _add_coefficient(
                row,
                variables.index("p_g", period, generator.generator_id),
                1.0,
            )
        for renewable in config.renewables:
            _add_coefficient(
                row,
                variables.index("p_rg", period, renewable.resource_id),
                1.0,
            )
        for storage in config.storage:
            _add_coefficient(
                row,
                variables.index("p_ess_dc", period, storage.resource_id),
                1.0,
            )
            _add_coefficient(
                row,
                variables.index("p_ess_ch", period, storage.resource_id),
                -1.0,
            )
        equality_rows.append(row)
        equality_rhs.append(float(np.sum(load_mw[period])))
        equality_metadata.append(ConstraintRow("power_balance", period, "system", "equality", "1"))

    for storage in config.storage:
        row = {}
        for period in range(config.periods):
            _add_coefficient(
                row,
                variables.index("p_ess_dc", period, storage.resource_id),
                -config.interval_hours / storage.discharge_efficiency,
            )
            _add_coefficient(
                row,
                variables.index("p_ess_ch", period, storage.resource_id),
                config.interval_hours * storage.charge_efficiency,
            )
        equality_rows.append(row)
        equality_rhs.append(0.0)
        equality_metadata.append(
            ConstraintRow(
                "storage_terminal_energy",
                None,
                storage.resource_id,
                "equality",
                "9",
            )
        )

    inequality_rows: list[dict[int, float]] = []
    inequality_rhs: list[float] = []
    inequality_metadata: list[ConstraintRow] = []

    ptdf_branch_positions = {
        branch_id: position for position, branch_id in enumerate(shift_factors.branch_ids)
    }
    for period in range(config.periods):
        for branch in constrained_branches:
            ptdf_row = ptdf_branch_positions[branch.branch_id]
            coefficients: dict[int, float] = {}
            for generator in generators:
                value = shift_factors.matrix[ptdf_row, network.bus_position(generator.bus_id)]
                _add_coefficient(
                    coefficients,
                    variables.index("p_g", period, generator.generator_id),
                    value,
                )
            for renewable in config.renewables:
                value = shift_factors.matrix[ptdf_row, network.bus_position(renewable.bus_id)]
                _add_coefficient(
                    coefficients,
                    variables.index("p_rg", period, renewable.resource_id),
                    value,
                )
            for storage in config.storage:
                value = shift_factors.matrix[ptdf_row, network.bus_position(storage.bus_id)]
                _add_coefficient(
                    coefficients,
                    variables.index("p_ess_dc", period, storage.resource_id),
                    value,
                )
                _add_coefficient(
                    coefficients,
                    variables.index("p_ess_ch", period, storage.resource_id),
                    -value,
                )

            fixed_flow = float(
                shift_factors.flow_offset_mw[ptdf_row]
                - shift_factors.matrix[ptdf_row] @ load_mw[period]
            )
            inequality_rows.append(coefficients)
            inequality_rhs.append(-branch.rate_a_mw - fixed_flow)
            inequality_metadata.append(
                ConstraintRow("line_flow", period, branch.branch_id, "lower", "2")
            )
            inequality_rows.append({index: -value for index, value in coefficients.items()})
            inequality_rhs.append(fixed_flow - branch.rate_a_mw)
            inequality_metadata.append(
                ConstraintRow("line_flow", period, branch.branch_id, "upper", "2")
            )

    for period in range(config.periods):
        for generator in generators:
            up = {
                variables.index("p_g", period, generator.generator_id): -1.0,
                variables.index("r_up", period, generator.generator_id): -1.0,
            }
            inequality_rows.append(up)
            inequality_rhs.append(-generator.maximum_mw)
            inequality_metadata.append(
                ConstraintRow(
                    "reserve_headroom",
                    period,
                    generator.generator_id,
                    "upper",
                    "4",
                )
            )
            down = {
                variables.index("p_g", period, generator.generator_id): 1.0,
                variables.index("r_down", period, generator.generator_id): -1.0,
            }
            inequality_rows.append(down)
            inequality_rhs.append(generator.minimum_mw)
            inequality_metadata.append(
                ConstraintRow(
                    "reserve_footroom",
                    period,
                    generator.generator_id,
                    "lower",
                    "4",
                )
            )

    for period in range(config.periods):
        up = {
            variables.index("r_up", period, generator.generator_id): 1.0 for generator in generators
        }
        inequality_rows.append(up)
        inequality_rhs.append(config.reserve_up_mw[period])
        inequality_metadata.append(
            ConstraintRow("reserve_requirement", period, "system", "up", "5")
        )
        down = {
            variables.index("r_down", period, generator.generator_id): 1.0
            for generator in generators
        }
        inequality_rows.append(down)
        inequality_rhs.append(config.reserve_down_mw[period])
        inequality_metadata.append(
            ConstraintRow("reserve_requirement", period, "system", "down", "5")
        )

    for period in range(1, config.periods):
        for generator, ramp_up, ramp_down in zip(
            generators,
            config.generator_ramp_up_mw_per_hour,
            config.generator_ramp_down_mw_per_hour,
            strict=True,
        ):
            increase = {
                variables.index("p_g", period, generator.generator_id): 1.0,
                variables.index("p_g", period - 1, generator.generator_id): -1.0,
            }
            inequality_rows.append(increase)
            inequality_rhs.append(-ramp_down * config.interval_hours)
            inequality_metadata.append(
                ConstraintRow("generator_ramp", period, generator.generator_id, "down", "6")
            )
            decrease = {index: -value for index, value in increase.items()}
            inequality_rows.append(decrease)
            inequality_rhs.append(-ramp_up * config.interval_hours)
            inequality_metadata.append(
                ConstraintRow("generator_ramp", period, generator.generator_id, "up", "6")
            )

    for period in range(config.periods):
        for storage in config.storage:
            cumulative: dict[int, float] = {}
            for earlier in range(period + 1):
                _add_coefficient(
                    cumulative,
                    variables.index("p_ess_dc", earlier, storage.resource_id),
                    -config.interval_hours / storage.discharge_efficiency,
                )
                _add_coefficient(
                    cumulative,
                    variables.index("p_ess_ch", earlier, storage.resource_id),
                    config.interval_hours * storage.charge_efficiency,
                )
            inequality_rows.append(cumulative)
            inequality_rhs.append(storage.minimum_energy_mwh - storage.initial_energy_mwh)
            inequality_metadata.append(
                ConstraintRow(
                    "storage_energy",
                    period,
                    storage.resource_id,
                    "lower",
                    "8",
                )
            )
            inequality_rows.append({index: -value for index, value in cumulative.items()})
            inequality_rhs.append(storage.initial_energy_mwh - storage.maximum_energy_mwh)
            inequality_metadata.append(
                ConstraintRow(
                    "storage_energy",
                    period,
                    storage.resource_id,
                    "upper",
                    "8",
                )
            )

    lp = CanonicalLP(
        c=cost,
        A1=_sparse_rows(equality_rows, columns=len(variables)),
        b1=equality_rhs,
        A2=_sparse_rows(inequality_rows, columns=len(variables)),
        b2=inequality_rhs,
        lower=lower,
        upper=upper,
    )
    model = DCOPFModel(
        network=network,
        config=config,
        ptdf=shift_factors,
        variables=variables,
        lp=lp,
        equality_rows=tuple(equality_metadata),
        inequality_rows=tuple(inequality_metadata),
        constrained_branches=constrained_branches,
        load_mw=load_mw,
        objective_constant=float(objective_constant),
    )
    expected = model.expected_dimensions()
    actual = {"n": lp.n, "m1": lp.m1, "m2": lp.m2, "m": lp.m}
    if actual != expected:
        raise AssertionError(f"DCOPF dimensions {actual} do not match formula {expected}.")
    return model
