"""Scalable, disclosed reconstruction of the paper's Stage 7 benchmarks.

This module is deliberately separate from the validated Stage 2 small-case
builder.  The public MATPOWER cases do not contain the renewable, storage, or
time-series inputs used by the paper, so the additions below are a frozen
*structural reconstruction*, not a claim to have recovered author data.

The implementation keeps the paper's printed row and variable counts, builds
only the PTDF columns needed by controllable devices, and represents row
metadata by a handful of blocks rather than one Python object per row.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu

from .canonical_lp import CanonicalLP
from .dcopf_model import ConstraintRow, VariableKey, VariableKind
from .network_data import MATPOWERCaseError, NetworkCase

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

_INT32_MAX = int(np.iinfo(np.int32).max)
_DGX_SPARK_MEMORY_BYTES = int(round(121.690 * (2**30)))


def _readonly_float(values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _readonly_bool(values: ArrayLike) -> BoolArray:
    result = np.asarray(values, dtype=np.bool_).copy()
    result.setflags(write=False)
    return result


def _readonly_int(values: ArrayLike) -> IntArray:
    result = np.asarray(values, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class Stage7CaseSpec:
    """Paper Table I counts and Table II horizons for one public case."""

    name: str
    buses: int
    branches: int
    generators: int
    renewables: int
    storage: int
    horizons: tuple[int, ...]


PAPER_CASE_SPECS = MappingProxyType(
    {
        "case1354pegase": Stage7CaseSpec(
            "case1354pegase", 1_354, 1_991, 260, 136, 68, (4, 16, 48, 96)
        ),
        "case2868rte": Stage7CaseSpec(
            "case2868rte", 2_868, 3_808, 600, 286, 143, (4, 16, 48, 56, 64, 72, 80, 88, 96)
        ),
        "case9241pegase": Stage7CaseSpec(
            "case9241pegase", 9_241, 16_049, 1_445, 920, 460, (4, 6, 16, 24, 32)
        ),
    }
)


@dataclass(frozen=True, slots=True)
class TableIIRow:
    """One dimension/nonzero record transcribed from the paper's Table II."""

    case_name: str
    periods: int
    published_m: int
    published_n: int
    published_nnz: int


TABLE_II_ROWS = (
    TableIIRow("case1354pegase", 4, 20_192, 4_208, 7_190_640),
    TableIIRow("case1354pegase", 16, 82_124, 16_832, 28_791_792),
    TableIIRow("case1354pegase", 48, 247_276, 50_496, 86_586_352),
    TableIIRow("case1354pegase", 96, 495_004, 100_992, 173_800_432),
    TableIIRow("case2868rte", 4, 40_163, 9_488, 30_111_616),
    TableIIRow("case2868rte", 16, 163_823, 37_952, 120_508_576),
    TableIIRow("case2868rte", 48, 493_583, 113_856, 295_998_240),
    TableIIRow("case2868rte", 56, 576_023, 132_832, 345_459_808),
    TableIIRow("case2868rte", 64, 658_463, 151_808, 394_957_984),
    TableIIRow("case2868rte", 72, 740_903, 170_784, 444_492_768),
    TableIIRow("case2868rte", 80, 823_343, 189_760, 494_064_160),
    TableIIRow("case2868rte", 88, 905_783, 208_736, 543_672_160),
    TableIIRow("case2868rte", 96, 988_223, 227_712, 593_316_768),
    TableIIRow("case9241pegase", 4, 152_774, 24_700, 373_238_888),
    TableIIRow("case9241pegase", 6, 230_376, 37_050, 559_872_262),
    TableIIRow("case9241pegase", 16, 618_386, 98_800, 1_493_149_532),
    TableIIRow("case9241pegase", 24, 928_794, 148_200, 2_239_903_828),
    TableIIRow("case9241pegase", 32, 1_239_202, 197_600, 2_986_775_884),
)

_TABLE_II_BY_KEY = MappingProxyType({(row.case_name, row.periods): row for row in TABLE_II_ROWS})


@dataclass(frozen=True, slots=True)
class Stage7ReconstructionPolicy:
    """A-priori choices frozen before any benchmark solve is observed."""

    policy_id: str = "stage7-structural-reconstruction-v1"
    resource_selection: str = "cycle_generator_buses_in_ascending_generator_row_order"
    storage_selection: str = "every_second_renewable_location_with_deterministic_cycle"
    load_profile: str = "flat_public_matpower_demand"
    load_definition: str = "MATPOWER_Pd_plus_Gs_as_active_withdrawal"
    renewable_profile: str = "flat_equal_share_of_10pct_base_load_nameplate"
    renewable_nameplate_distribution: str = "equal_across_reconstructed_renewable_rows"
    renewable_total_nameplate_fraction_of_base_load: float = 0.10
    renewable_minimum_fraction: float = 0.0
    renewable_availability_fraction: float = 1.0
    storage_total_power_fraction_of_base_load: float = 0.01
    storage_power_distribution: str = "equal_across_reconstructed_storage_rows"
    storage_duration_hours: float = 4.0
    storage_minimum_energy_fraction: float = 0.0
    storage_initial_fraction: float = 0.50
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    interval_hours: float = 1.0
    renewable_penalty_per_mwh: float = 1.0
    storage_loss_penalty_per_mwh: float = 1.0
    reserve_up_fraction_of_base_load: float = 0.01
    reserve_down_fraction_of_base_load: float = 0.01
    generator_ramp_fraction_of_pmax_per_hour: float = 0.10
    ramp_rule: str = "10pct_public_pmax_per_hour"
    seed: int = 20_260_803
    ptdf_zero_atol: float = 1.0e-12
    ptdf_rhs_chunk_columns: int = 128
    ptdf_reference_policy: str = "public_matpower_type_3_reference_bus"
    zero_rate_a_bound_proof: str = "outward_rounded_triangle_bound_over_complete_variable_box"
    inactive_branch_policy: str = "retain_zero_flow_rows_exclude_from_topology"
    gpu_planning_headroom: float = 1.25
    dgx_usable_fraction: float = 0.80
    host_assembly_peak_multiplier: float = 2.50

    def __post_init__(self) -> None:
        unit_interval = (
            self.renewable_total_nameplate_fraction_of_base_load,
            self.renewable_availability_fraction,
            self.storage_total_power_fraction_of_base_load,
            self.storage_initial_fraction,
            self.charge_efficiency,
            self.discharge_efficiency,
            self.reserve_up_fraction_of_base_load,
            self.reserve_down_fraction_of_base_load,
            self.generator_ramp_fraction_of_pmax_per_hour,
            self.dgx_usable_fraction,
        )
        if not all(0.0 < value <= 1.0 for value in unit_interval):
            raise ValueError("Stage 7 fractional policy values must lie in (0, 1].")
        if self.storage_duration_hours <= 0.0 or self.interval_hours <= 0.0:
            raise ValueError("Stage 7 time constants must be positive.")
        if self.renewable_minimum_fraction != 0.0 or self.storage_minimum_energy_fraction != 0.0:
            raise ValueError("Frozen Stage 7 RG/ESS minimum fractions must be zero.")
        if self.ptdf_zero_atol < 0.0 or self.ptdf_rhs_chunk_columns <= 0:
            raise ValueError("The PTDF tolerance/chunk policy is invalid.")
        if self.gpu_planning_headroom < 1.0 or self.host_assembly_peak_multiplier < 1.0:
            raise ValueError("Memory planning multipliers must be at least one.")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FROZEN_STAGE7_POLICY = Stage7ReconstructionPolicy()


def assert_stage7_reconstruction_contract(
    reconstruction_protocol: dict[str, object],
    *,
    policy: Stage7ReconstructionPolicy = FROZEN_STAGE7_POLICY,
) -> None:
    """Fail closed when the JSON benchmark contract drifts from the code.

    Callers pass the ``reconstruction_protocol`` object from
    ``configs/benchmarks/stage_7_small_medium.json``.  Exact string checks are
    intentional: these choices were frozen before outcomes and must not be
    changed silently in a later run.
    """

    expected: dict[str, object] = {
        "frozen_before_benchmark_runs": True,
        "timing_tuning_prohibited": True,
        "load_profile": "flat MATPOWER active demand in every period",
        "load_definition": "MATPOWER Pd + Gs treated as active withdrawal",
        "generator_policy": (
            "retain every MATPOWER generator row; originally offline rows are fixed at zero "
            "output and zero reserve"
        ),
        "generator_ramp_fraction_of_pmax_per_hour": (
            policy.generator_ramp_fraction_of_pmax_per_hour
        ),
        "renewable_placement": (
            "cycle through MATPOWER generator buses in ascending generator-row order"
        ),
        "renewable_total_nameplate_fraction_of_base_load": (
            policy.renewable_total_nameplate_fraction_of_base_load
        ),
        "renewable_nameplate_distribution": "equal across reconstructed renewable rows",
        "renewable_minimum_fraction": policy.renewable_minimum_fraction,
        "renewable_availability_fraction": policy.renewable_availability_fraction,
        "storage_placement": (
            "use every second reconstructed renewable location, cycling deterministically "
            "when required"
        ),
        "storage_total_power_fraction_of_base_load": (
            policy.storage_total_power_fraction_of_base_load
        ),
        "storage_power_distribution": "equal across reconstructed storage rows",
        "storage_duration_hours": policy.storage_duration_hours,
        "storage_minimum_energy_fraction": policy.storage_minimum_energy_fraction,
        "storage_initial_state_fraction": policy.storage_initial_fraction,
        "storage_charge_efficiency": policy.charge_efficiency,
        "storage_discharge_efficiency": policy.discharge_efficiency,
        "reserve_up_fraction_of_base_load": policy.reserve_up_fraction_of_base_load,
        "reserve_down_fraction_of_base_load": policy.reserve_down_fraction_of_base_load,
        "renewable_penalty_per_mwh": policy.renewable_penalty_per_mwh,
        "storage_loss_penalty_per_mwh": policy.storage_loss_penalty_per_mwh,
        "positive_rate_a_policy": "preserve the MATPOWER rateA value",
        "zero_rate_a_policy": (
            "retain the paper-required branch rows with a finite bound derived to be "
            "redundant over the variable box"
        ),
        "zero_rate_a_bound_proof": (
            "outward-rounded triangle bound over the complete variable box"
        ),
        "inactive_branch_policy": (
            "retain paper-count rows as zero-flow rows while excluding them from topology"
        ),
        "angle_limit_policy": (
            "ignore MATPOWER angle-difference limits because they are absent from the paper model"
        ),
        "objective_policy": (
            "use the linear and constant MATPOWER polynomial terms; reject nonzero "
            "higher-order terms"
        ),
        "interval_hours": policy.interval_hours,
        "ptdf_zero_atol": policy.ptdf_zero_atol,
        "ptdf_rhs_chunk_columns": policy.ptdf_rhs_chunk_columns,
        "ptdf_reference_policy": "use the public MATPOWER type-3 reference bus",
        "seed": policy.seed,
    }
    if reconstruction_protocol != expected:
        differing = sorted(
            key
            for key in set(reconstruction_protocol) | set(expected)
            if reconstruction_protocol.get(key) != expected.get(key)
        )
        raise ValueError(
            "Stage 7 reconstruction protocol differs from the frozen implementation; "
            f"differing keys={differing}."
        )


@dataclass(frozen=True, slots=True)
class Stage7Preflight:
    """Dimension, sparse-index, and memory check before matrix allocation."""

    row: TableIIRow
    m1: int
    m2: int
    computed_m: int
    computed_n: int
    dense_structural_nnz_upper_bound: int
    planning_nnz: int
    csr_one_orientation_bytes: int
    gpu_matrix_and_transpose_bytes: int
    gpu_vector_bytes: int
    gpu_planning_bytes: int
    host_assembly_peak_bytes: int
    csr32_supported: bool
    dimensions_match_table: bool
    fits_dgx_planning_budget: bool
    policy_fingerprint: str

    @property
    def gpu_planning_gib(self) -> float:
        return self.gpu_planning_bytes / (2**30)

    @property
    def host_assembly_peak_gib(self) -> float:
        return self.host_assembly_peak_bytes / (2**30)

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "case_name": self.row.case_name,
            "periods": self.row.periods,
            "published_m": self.row.published_m,
            "published_n": self.row.published_n,
            "published_nnz": self.row.published_nnz,
            "reconstructed_nnz_kind": (
                "dense_structural_upper_bound_without_public_case_ptdf_factorization"
            ),
            "full_lp_allocated": False,
            "stage_8_large_allocation_locked": True,
            "m1": self.m1,
            "m2": self.m2,
            "computed_m": self.computed_m,
            "computed_n": self.computed_n,
            "dense_structural_nnz_upper_bound": self.dense_structural_nnz_upper_bound,
            "planning_nnz": self.planning_nnz,
            "csr_one_orientation_bytes": self.csr_one_orientation_bytes,
            "gpu_matrix_and_transpose_bytes": self.gpu_matrix_and_transpose_bytes,
            "gpu_vector_bytes": self.gpu_vector_bytes,
            "gpu_planning_bytes": self.gpu_planning_bytes,
            "gpu_planning_gib": self.gpu_planning_gib,
            "host_assembly_peak_bytes": self.host_assembly_peak_bytes,
            "host_assembly_peak_gib": self.host_assembly_peak_gib,
            "csr32_supported": self.csr32_supported,
            "dimensions_match_table": self.dimensions_match_table,
            "fits_dgx_planning_budget": self.fits_dgx_planning_budget,
            "policy_fingerprint": self.policy_fingerprint,
        }


def _dimensions(spec: Stage7CaseSpec, periods: int) -> tuple[int, int, int, int]:
    generators = spec.generators
    renewables = spec.renewables
    storage = spec.storage
    branches = spec.branches
    n = periods * (3 * generators + renewables + 2 * storage)
    m1 = periods + storage
    m2 = (
        2 * periods * branches
        + 2 * periods * generators
        + 2 * periods
        + 2 * (periods - 1) * generators
        + 2 * periods * storage
    )
    return n, m1, m2, m1 + m2


def _dense_structural_nnz_upper_bound(spec: Stage7CaseSpec, periods: int) -> int:
    """Count all printed structural entries before PTDF numerical zeros."""

    generators = spec.generators
    renewables = spec.renewables
    storage = spec.storage
    devices = generators + renewables + 2 * storage
    line = 2 * periods * spec.branches * devices
    balance = periods * devices
    terminal_storage = 2 * periods * storage
    reserve_box = 4 * periods * generators
    reserve_system = 2 * periods * generators
    ramp = 4 * (periods - 1) * generators
    cumulative_storage = 2 * storage * periods * (periods + 1)
    return (
        line + balance + terminal_storage + reserve_box + reserve_system + ramp + cumulative_storage
    )


def stage7_preflight(
    case_name: str,
    periods: int,
    *,
    policy: Stage7ReconstructionPolicy = FROZEN_STAGE7_POLICY,
) -> Stage7Preflight:
    """Return a no-allocation preflight for one published Table II row.

    The GPU vector count follows the existing Stage 6 resident FP64 workspace:
    31 ``n``-vectors, 19 ``m1``-vectors, and 16 ``m2``-vectors.  Sparse storage
    includes both CSR orientations because the GPU backend keeps ``A`` and
    ``A.T`` resident.  Planning uses the larger of the paper's nnz and the
    reconstruction's dense-PTDF structural upper bound.
    """

    try:
        spec = PAPER_CASE_SPECS[case_name]
        row = _TABLE_II_BY_KEY[(case_name, periods)]
    except KeyError as exc:
        raise KeyError(f"No Table II row for ({case_name!r}, T={periods}).") from exc

    n, m1, m2, m = _dimensions(spec, periods)
    structural_nnz = _dense_structural_nnz_upper_bound(spec, periods)
    planning_nnz = max(row.published_nnz, structural_nnz)
    csr_bytes = planning_nnz * (8 + 4) + (m1 + m2 + 2) * 4
    gpu_sparse = 2 * csr_bytes
    gpu_vectors = 8 * (31 * n + 19 * m1 + 16 * m2)
    gpu_planning = int(math.ceil(policy.gpu_planning_headroom * (gpu_sparse + gpu_vectors)))
    host_peak = int(math.ceil(policy.host_assembly_peak_multiplier * csr_bytes))
    csr32_supported = planning_nnz <= _INT32_MAX
    budget = int(policy.dgx_usable_fraction * _DGX_SPARK_MEMORY_BYTES)
    return Stage7Preflight(
        row=row,
        m1=m1,
        m2=m2,
        computed_m=m,
        computed_n=n,
        dense_structural_nnz_upper_bound=structural_nnz,
        planning_nnz=planning_nnz,
        csr_one_orientation_bytes=csr_bytes,
        gpu_matrix_and_transpose_bytes=gpu_sparse,
        gpu_vector_bytes=gpu_vectors,
        gpu_planning_bytes=gpu_planning,
        host_assembly_peak_bytes=host_peak,
        csr32_supported=csr32_supported,
        dimensions_match_table=(m == row.published_m and n == row.published_n),
        fits_dgx_planning_budget=(csr32_supported and gpu_planning <= budget),
        policy_fingerprint=policy.fingerprint,
    )


def all_stage7_preflights(
    *, policy: Stage7ReconstructionPolicy = FROZEN_STAGE7_POLICY
) -> tuple[Stage7Preflight, ...]:
    """Return preflights for all 18 rows in the paper's Table II."""

    return tuple(
        stage7_preflight(row.case_name, row.periods, policy=policy) for row in TABLE_II_ROWS
    )


@dataclass(frozen=True, slots=True)
class Stage7NormalizedCase:
    """Public MATPOWER data normalized to the paper's Table I population."""

    network: NetworkCase
    spec: Stage7CaseSpec
    generator_ids: tuple[str, ...]
    generator_bus_positions: IntArray
    generator_online: BoolArray
    generator_lower_mw: FloatArray
    generator_upper_mw: FloatArray
    generator_cost_slope: FloatArray
    generator_cost_constant: FloatArray
    branch_ids: tuple[str, ...]
    branch_topology_active: BoolArray
    original_rate_a_mw: FloatArray
    zero_rate_mask: BoolArray
    ignored_angle_limit_mask: BoolArray

    def __post_init__(self) -> None:
        generators = self.spec.generators
        branches = self.spec.branches
        for name in (
            "generator_bus_positions",
            "generator_online",
            "generator_lower_mw",
            "generator_upper_mw",
            "generator_cost_slope",
            "generator_cost_constant",
        ):
            if np.asarray(getattr(self, name)).shape != (generators,):
                raise ValueError(f"{name} does not match the paper generator count.")
        for name in (
            "branch_topology_active",
            "original_rate_a_mw",
            "zero_rate_mask",
            "ignored_angle_limit_mask",
        ):
            if np.asarray(getattr(self, name)).shape != (branches,):
                raise ValueError(f"{name} does not match the paper branch count.")

    @property
    def offline_generator_count(self) -> int:
        return int(np.count_nonzero(~self.generator_online))

    @property
    def zero_rate_branch_count(self) -> int:
        return int(np.count_nonzero(self.zero_rate_mask))

    @property
    def ignored_angle_limit_count(self) -> int:
        return int(np.count_nonzero(self.ignored_angle_limit_mask))


def normalize_stage7_case(
    network: NetworkCase,
    *,
    spec: Stage7CaseSpec | None = None,
) -> Stage7NormalizedCase:
    """Normalize one public MATPOWER case without dropping paper rows.

    All MATPOWER generator rows remain in the variable population.  Offline
    generators receive exactly ``lower=upper=0``.  All branch rows remain in
    the line-constraint population; inactive topology rows (none in the three
    pinned public files) are represented later by a zero flow row.  MATPOWER
    angle limits are counted but ignored because no angle-limit family exists
    in the paper's printed LP.  A zero ``rateA`` is marked for a derived,
    box-redundant bound after the reconstructed device columns are known.
    """

    if spec is None:
        try:
            spec = PAPER_CASE_SPECS[network.name]
        except KeyError as exc:
            raise KeyError(f"No Stage 7 paper specification for {network.name!r}.") from exc
    actual = (len(network.buses), len(network.branches), len(network.generators))
    expected = (spec.buses, spec.branches, spec.generators)
    if actual != expected:
        raise MATPOWERCaseError(
            f"{network.name} public counts {actual} do not match paper counts {expected}."
        )

    bus_positions = {bus_id: position for position, bus_id in enumerate(network.bus_ids)}
    online = np.asarray([generator.status for generator in network.generators], dtype=np.bool_)
    lower = np.asarray(
        [generator.minimum_mw if generator.status else 0.0 for generator in network.generators],
        dtype=np.float64,
    )
    upper = np.asarray(
        [generator.maximum_mw if generator.status else 0.0 for generator in network.generators],
        dtype=np.float64,
    )
    slopes: list[float] = []
    constants: list[float] = []
    for generator in network.generators:
        slope, constant, omitted = generator.cost.paper_linear_terms()
        if any(abs(value) > 0.0 for value in omitted):
            raise MATPOWERCaseError(
                f"{generator.generator_id} has a nonlinear public cost; the Stage 7 "
                "reconstruction does not tune or silently linearize it."
            )
        slopes.append(slope)
        constants.append(constant)

    rates = np.asarray([branch.rate_a_mw for branch in network.branches], dtype=np.float64)
    if np.any(rates < 0.0):
        raise MATPOWERCaseError("MATPOWER rateA values must be nonnegative.")
    ignored_angles = np.asarray(
        [
            branch.angle_minimum_degrees > -360.0 or branch.angle_maximum_degrees < 360.0
            for branch in network.branches
        ],
        dtype=np.bool_,
    )
    return Stage7NormalizedCase(
        network=network,
        spec=spec,
        generator_ids=tuple(generator.generator_id for generator in network.generators),
        generator_bus_positions=_readonly_int(
            [bus_positions[generator.bus_id] for generator in network.generators]
        ),
        generator_online=_readonly_bool(online),
        generator_lower_mw=_readonly_float(lower),
        generator_upper_mw=_readonly_float(upper),
        generator_cost_slope=_readonly_float(slopes),
        generator_cost_constant=_readonly_float(constants),
        branch_ids=tuple(branch.branch_id for branch in network.branches),
        branch_topology_active=_readonly_bool([branch.status for branch in network.branches]),
        original_rate_a_mw=_readonly_float(rates),
        zero_rate_mask=_readonly_bool(np.isclose(rates, 0.0, rtol=0.0, atol=0.0)),
        ignored_angle_limit_mask=_readonly_bool(ignored_angles),
    )


@dataclass(frozen=True, slots=True)
class Stage7ResourceFleet:
    """Deterministic RG/ESS and time-series structural reconstruction."""

    policy_id: str
    policy_fingerprint: str
    periods: int
    renewable_ids: tuple[str, ...]
    renewable_bus_positions: IntArray
    renewable_minimum_mw: FloatArray
    renewable_maximum_mw: FloatArray
    storage_ids: tuple[str, ...]
    storage_bus_positions: IntArray
    storage_maximum_charge_mw: FloatArray
    storage_maximum_discharge_mw: FloatArray
    storage_minimum_energy_mwh: FloatArray
    storage_initial_energy_mwh: FloatArray
    storage_maximum_energy_mwh: FloatArray
    storage_charge_efficiency: FloatArray
    storage_discharge_efficiency: FloatArray
    load_multipliers: FloatArray
    reserve_up_mw: FloatArray
    reserve_down_mw: FloatArray
    generator_ramp_up_mw: FloatArray
    generator_ramp_down_mw: FloatArray
    interval_hours: float
    renewable_penalty_per_mwh: float
    storage_loss_penalty_per_mwh: float
    classification: str = "deterministic_structural_reconstruction_not_author_data"


def reconstruct_stage7_resources(
    normalized: Stage7NormalizedCase,
    periods: int,
    *,
    policy: Stage7ReconstructionPolicy = FROZEN_STAGE7_POLICY,
) -> Stage7ResourceFleet:
    """Create the frozen structural RG/ESS reconstruction for one horizon."""

    if periods <= 0:
        raise ValueError("periods must be positive.")
    base_load = float(np.sum(normalized.network.demand_mw))
    if not np.isfinite(base_load) or base_load <= 0.0:
        raise ValueError("The public MATPOWER base active load must be positive.")
    generator_positions = normalized.generator_bus_positions
    renewable_positions = np.asarray(
        [
            generator_positions[index % generator_positions.size]
            for index in range(normalized.spec.renewables)
        ],
        dtype=np.int64,
    )

    renewable_ids = tuple(
        f"rg_{rank + 1:04d}_bus_{normalized.network.bus_ids[position]}"
        for rank, position in enumerate(renewable_positions)
    )
    renewable_nameplate = (
        policy.renewable_total_nameplate_fraction_of_base_load
        * base_load
        / normalized.spec.renewables
    )
    renewable_maximum = np.full(
        (periods, normalized.spec.renewables),
        policy.renewable_availability_fraction * renewable_nameplate,
        dtype=np.float64,
    )
    renewable_minimum = policy.renewable_minimum_fraction * renewable_maximum

    every_second = renewable_positions[::2]
    storage_positions = np.asarray(
        [every_second[index % every_second.size] for index in range(normalized.spec.storage)],
        dtype=np.int64,
    )
    if storage_positions.size != normalized.spec.storage:
        raise ValueError("The even-ranked RG policy did not produce the paper ESS count.")
    storage_ids = tuple(
        f"ess_{rank + 1:04d}_bus_{normalized.network.bus_ids[position]}"
        for rank, position in enumerate(storage_positions)
    )
    storage_power = np.full(
        normalized.spec.storage,
        policy.storage_total_power_fraction_of_base_load * base_load / normalized.spec.storage,
        dtype=np.float64,
    )
    storage_energy = policy.storage_duration_hours * storage_power
    ramp = np.where(
        normalized.generator_online,
        policy.generator_ramp_fraction_of_pmax_per_hour
        * np.maximum(normalized.generator_upper_mw, 0.0),
        0.0,
    )
    reserve_up = policy.reserve_up_fraction_of_base_load * base_load
    reserve_down = policy.reserve_down_fraction_of_base_load * base_load

    return Stage7ResourceFleet(
        policy_id=policy.policy_id,
        policy_fingerprint=policy.fingerprint,
        periods=periods,
        renewable_ids=renewable_ids,
        renewable_bus_positions=_readonly_int(renewable_positions),
        renewable_minimum_mw=_readonly_float(renewable_minimum),
        renewable_maximum_mw=_readonly_float(renewable_maximum),
        storage_ids=storage_ids,
        storage_bus_positions=_readonly_int(storage_positions),
        storage_maximum_charge_mw=_readonly_float(storage_power),
        storage_maximum_discharge_mw=_readonly_float(storage_power),
        storage_minimum_energy_mwh=_readonly_float(
            policy.storage_minimum_energy_fraction * storage_energy
        ),
        storage_initial_energy_mwh=_readonly_float(
            policy.storage_initial_fraction * storage_energy
        ),
        storage_maximum_energy_mwh=_readonly_float(storage_energy),
        storage_charge_efficiency=_readonly_float(
            np.full(normalized.spec.storage, policy.charge_efficiency)
        ),
        storage_discharge_efficiency=_readonly_float(
            np.full(normalized.spec.storage, policy.discharge_efficiency)
        ),
        load_multipliers=_readonly_float(np.ones(periods)),
        reserve_up_mw=_readonly_float(np.full(periods, reserve_up)),
        reserve_down_mw=_readonly_float(np.full(periods, reserve_down)),
        generator_ramp_up_mw=_readonly_float(ramp),
        generator_ramp_down_mw=_readonly_float(ramp),
        interval_hours=policy.interval_hours,
        renewable_penalty_per_mwh=policy.renewable_penalty_per_mwh,
        storage_loss_penalty_per_mwh=policy.storage_loss_penalty_per_mwh,
    )


class Stage7VariableIndex:
    """Formula-based equivalent of :class:`VariableIndex` without key expansion."""

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
        self.periods = periods
        self._devices: dict[VariableKind, tuple[str, ...]] = {
            "p_g": generator_ids,
            "p_rg": renewable_ids,
            "p_ess_dc": storage_ids,
            "p_ess_ch": storage_ids,
            "r_up": generator_ids,
            "r_down": generator_ids,
        }
        self._positions: dict[VariableKind, MappingProxyType[str, int]] = {
            kind: MappingProxyType({device_id: index for index, device_id in enumerate(ids)})
            for kind, ids in self._devices.items()
        }
        self._offsets: dict[VariableKind, int] = {}
        offset = 0
        for kind in self._block_order:
            self._offsets[kind] = offset
            offset += periods * len(self._devices[kind])
        self._length = offset

    def __len__(self) -> int:
        return self._length

    def block_slice(self, kind: VariableKind) -> slice:
        start = self._offsets[kind]
        return slice(start, start + self.periods * len(self._devices[kind]))

    def index(self, kind: VariableKind, period: int, device_id: str) -> int:
        if not 0 <= period < self.periods:
            raise KeyError(f"Unknown period {period} for {kind}.")
        try:
            position = self._positions[kind][device_id]
        except KeyError as exc:
            raise KeyError(f"Unknown variable ({kind}, {period}, {device_id}).") from exc
        return self._offsets[kind] + period * len(self._devices[kind]) + position

    def key(self, index: int) -> VariableKey:
        if not 0 <= index < self._length:
            raise IndexError(index)
        for kind in self._block_order:
            block = self.block_slice(kind)
            if block.start <= index < block.stop:
                local = index - block.start
                count = len(self._devices[kind])
                period, position = divmod(local, count)
                return VariableKey(kind, period, self._devices[kind][position])
        raise AssertionError("Variable index did not map to a block.")


@dataclass(frozen=True, slots=True)
class RowFamilyBlock:
    """A compact period-major description of one consecutive row family."""

    start: int
    family: str
    side: str
    equation: str
    element_ids: tuple[str, ...]
    period_count: int
    period_offset: int = 0
    period_is_none: bool = False

    @property
    def stop(self) -> int:
        return self.start + self.period_count * len(self.element_ids)

    def row(self, absolute_index: int) -> ConstraintRow:
        if not self.start <= absolute_index < self.stop:
            raise IndexError(absolute_index)
        local = absolute_index - self.start
        period_index, element_index = divmod(local, len(self.element_ids))
        period = None if self.period_is_none else self.period_offset + period_index
        return ConstraintRow(
            self.family,
            period,
            self.element_ids[element_index],
            self.side,
            self.equation,
        )


class CompactRowMetadata(Sequence[ConstraintRow]):
    """Sequence-compatible row metadata whose storage is O(number of families)."""

    def __init__(self, blocks: Sequence[RowFamilyBlock]) -> None:
        self._blocks = tuple(blocks)
        expected = 0
        for block in self._blocks:
            if block.start != expected:
                raise ValueError("Row metadata blocks must be consecutive.")
            if not block.element_ids or block.period_count <= 0:
                raise ValueError("Row metadata blocks must describe at least one row.")
            expected = block.stop
        self._length = expected
        self._starts = tuple(block.start for block in self._blocks)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> ConstraintRow | tuple[ConstraintRow, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(self._length)))
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        block_position = bisect_right(self._starts, index) - 1
        return self._blocks[block_position].row(index)

    def __iter__(self) -> Iterator[ConstraintRow]:
        for block in self._blocks:
            for index in range(block.start, block.stop):
                yield block.row(index)

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def blocks(self) -> tuple[RowFamilyBlock, ...]:
        return self._blocks


@dataclass(slots=True)
class Stage7SparseDCFactorization:
    """One reusable sparse reduced-angle factorization for batched validation."""

    source_sha256: str
    bus_ids: tuple[int, ...]
    branch_ids: tuple[str, ...]
    reference_bus_id: int
    active_branch_positions: IntArray
    nonreference_positions: IntArray
    reduced_position_by_bus: IntArray
    branch_angle_reduced: sparse.csr_matrix
    phase_branch_flow_active_mw: FloatArray
    phase_bus_injection_mw: FloatArray
    reduced_bus_matrix_nnz: int
    _factor: object

    def shift_columns(self, bus_positions: IntArray, *, chunk_columns: int) -> FloatArray:
        """Return active-branch PTDF columns in bounded dense RHS batches."""

        positions = np.asarray(bus_positions, dtype=np.int64)
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= len(self.bus_ids)):
            raise ValueError("PTDF bus positions are outside the public case.")
        if chunk_columns <= 0:
            raise ValueError("chunk_columns must be positive.")
        reference = self.bus_ids.index(self.reference_bus_id)
        result = np.zeros((self.active_branch_positions.size, positions.size), dtype=np.float64)
        for start in range(0, positions.size, chunk_columns):
            selected = positions[start : start + chunk_columns]
            rhs = np.zeros((self.nonreference_positions.size, selected.size), dtype=np.float64)
            valid = selected != reference
            rhs[self.reduced_position_by_bus[selected[valid]], np.flatnonzero(valid)] = 1.0
            angles = self._factor.solve(rhs)
            result[:, start : start + selected.size] = self.branch_angle_reduced @ angles
        return result

    def shift_vector(self, bus_values: ArrayLike) -> FloatArray:
        """Apply the zero-reference PTDF to one bus vector."""

        values = np.asarray(bus_values, dtype=np.float64)
        if values.shape != (len(self.bus_ids),):
            raise ValueError(
                f"bus_values has shape {values.shape}; expected {(len(self.bus_ids),)}."
            )
        angles = self._factor.solve(values[self.nonreference_positions])
        return _readonly_float(self.branch_angle_reduced @ angles)

    def affine_flow_offset(self) -> FloatArray:
        """Return the active-branch affine phase-shift offset."""

        phase_angles = self._factor.solve(self.phase_bus_injection_mw[self.nonreference_positions])
        return _readonly_float(
            self.phase_branch_flow_active_mw - self.branch_angle_reduced @ phase_angles
        )

    def solve_angles_and_flows(
        self,
        injections_mw: ArrayLike,
        *,
        require_balanced: bool = True,
    ) -> tuple[FloatArray, FloatArray]:
        """Solve all periods in one sparse-factorization call.

        ``injections_mw`` is period-major with shape ``(T, N_B)``.  Returned
        flows retain every paper-count branch row; inactive topology rows are
        exactly zero.  No dense bus matrix is formed and the sparse LU is not
        recomputed between periods.
        """

        injections = np.asarray(injections_mw, dtype=np.float64)
        if injections.ndim != 2 or injections.shape[1] != len(self.bus_ids):
            raise ValueError(
                "injections_mw must have shape (periods, "
                f"{len(self.bus_ids)}); received {injections.shape}."
            )
        if not np.all(np.isfinite(injections)):
            raise ValueError("injections_mw must be finite.")
        totals = np.sum(injections, axis=1)
        tolerance = 1.0e-10 * (1.0 + np.linalg.norm(injections, ord=1, axis=1))
        if require_balanced and np.any(np.abs(totals) > tolerance):
            period = int(np.flatnonzero(np.abs(totals) > tolerance)[0])
            raise ValueError(f"Period {period} is not power-balanced; sum={totals[period]:.6g} MW.")
        right_hand_side = (
            injections[:, self.nonreference_positions]
            - self.phase_bus_injection_mw[self.nonreference_positions][None, :]
        ).T
        reduced_angles = self._factor.solve(right_hand_side)
        angles = np.zeros((injections.shape[0], len(self.bus_ids)), dtype=np.float64)
        angles[:, self.nonreference_positions] = reduced_angles.T
        active_flows = (
            self.branch_angle_reduced @ reduced_angles + self.phase_branch_flow_active_mw[:, None]
        ).T
        flows = np.zeros((injections.shape[0], len(self.branch_ids)), dtype=np.float64)
        flows[:, self.active_branch_positions] = active_flows
        return _readonly_float(angles), _readonly_float(flows)


def build_stage7_sparse_dc_factorization(
    normalized: Stage7NormalizedCase,
    *,
    minimum_reactance: float = 1.0e-9,
) -> Stage7SparseDCFactorization:
    """Build the reusable sparse network factorization without any horizon expansion."""

    network = normalized.network
    if minimum_reactance <= 0.0:
        raise ValueError("minimum_reactance must be positive.")
    active_rows = np.flatnonzero(normalized.branch_topology_active)
    if active_rows.size == 0:
        raise MATPOWERCaseError("The Stage 7 topology has no active branches.")
    bus_positions = {bus_id: position for position, bus_id in enumerate(network.bus_ids)}
    row_index: list[int] = []
    column_index: list[int] = []
    incidence_data: list[float] = []
    susceptance: list[float] = []
    phase_shift: list[float] = []
    for local_row, branch_position in enumerate(active_rows):
        branch = network.branches[int(branch_position)]
        if abs(branch.reactance_pu) < minimum_reactance:
            raise MATPOWERCaseError(f"{branch.branch_id} has |x| below {minimum_reactance:.3g}.")
        tap = branch.effective_tap_ratio
        if tap <= 0.0:
            raise MATPOWERCaseError(f"{branch.branch_id} has nonpositive tap {tap}.")
        row_index.extend((local_row, local_row))
        column_index.extend((bus_positions[branch.from_bus], bus_positions[branch.to_bus]))
        incidence_data.extend((1.0, -1.0))
        susceptance.append(network.base_mva / (branch.reactance_pu * tap))
        phase_shift.append(np.deg2rad(branch.phase_shift_degrees))
    incidence = sparse.csr_matrix(
        (incidence_data, (row_index, column_index)),
        shape=(active_rows.size, len(network.buses)),
        dtype=np.float64,
    )
    adjacency = (abs(incidence).T @ abs(incidence)).tocsr()
    component_count, _ = csgraph.connected_components(adjacency, directed=False)
    if component_count != 1:
        raise MATPOWERCaseError(
            f"The active Stage 7 network has {component_count} disconnected components."
        )
    susceptance_array = np.asarray(susceptance, dtype=np.float64)
    branch_angle = sparse.diags(susceptance_array, format="csr") @ incidence
    bus_matrix = (incidence.T @ branch_angle).tocsc()
    reference = network.bus_position(network.reference_bus_id)
    nonreference = np.flatnonzero(np.arange(len(network.buses)) != reference)
    reduced = bus_matrix[nonreference][:, nonreference].tocsc()
    try:
        factor = splu(reduced)
    except RuntimeError as exc:
        raise MATPOWERCaseError("The reduced Stage 7 DC bus matrix is singular.") from exc
    phase_branch = -susceptance_array * np.asarray(phase_shift, dtype=np.float64)
    phase_bus = np.asarray(incidence.T @ phase_branch, dtype=np.float64).reshape(-1)
    reduced_position = np.full(len(network.buses), -1, dtype=np.int64)
    reduced_position[nonreference] = np.arange(nonreference.size)
    return Stage7SparseDCFactorization(
        source_sha256=network.source_sha256,
        bus_ids=network.bus_ids,
        branch_ids=normalized.branch_ids,
        reference_bus_id=network.reference_bus_id,
        active_branch_positions=_readonly_int(active_rows),
        nonreference_positions=_readonly_int(nonreference),
        reduced_position_by_bus=_readonly_int(reduced_position),
        branch_angle_reduced=(branch_angle[:, nonreference]).tocsr(),
        phase_branch_flow_active_mw=_readonly_float(phase_branch),
        phase_bus_injection_mw=_readonly_float(phase_bus),
        reduced_bus_matrix_nnz=int(reduced.nnz),
        _factor=factor,
    )


@dataclass(frozen=True, slots=True)
class Stage7CompressedPTDF:
    """Sparse device-column PTDF and affine fixed-load branch flow."""

    bus_ids: tuple[int, ...]
    branch_ids: tuple[str, ...]
    reference_bus_id: int
    generator_matrix: sparse.csr_matrix
    renewable_matrix: sparse.csr_matrix
    storage_matrix: sparse.csr_matrix
    fixed_flow_mw: FloatArray
    flow_offset_mw: FloatArray
    load_shift_mw: FloatArray
    zero_atol: float
    reduced_bus_matrix_nnz: int

    @property
    def reference_position(self) -> int:
        return self.bus_ids.index(self.reference_bus_id)


def _sparsified_csr(values: NDArray[np.float64], *, zero_atol: float) -> sparse.csr_matrix:
    detached = np.asarray(values, dtype=np.float64).copy()
    if zero_atol:
        detached[np.abs(detached) <= zero_atol] = 0.0
    result = sparse.csr_matrix(detached, dtype=np.float64)
    result.eliminate_zeros()
    result.sort_indices()
    return result


def build_stage7_compressed_ptdf(
    normalized: Stage7NormalizedCase,
    fleet: Stage7ResourceFleet,
    *,
    policy: Stage7ReconstructionPolicy = FROZEN_STAGE7_POLICY,
    minimum_reactance: float = 1.0e-9,
    factorization: Stage7SparseDCFactorization | None = None,
) -> Stage7CompressedPTDF:
    """Factor the sparse bus matrix and solve only required device columns."""

    if fleet.policy_fingerprint != policy.fingerprint:
        raise ValueError("The resource fleet and PTDF policy fingerprints differ.")
    network = normalized.network
    factorization = (
        build_stage7_sparse_dc_factorization(normalized, minimum_reactance=minimum_reactance)
        if factorization is None
        else factorization
    )
    if factorization.source_sha256 != network.source_sha256:
        raise ValueError("The sparse DC factorization belongs to a different MATPOWER file.")
    active_rows = factorization.active_branch_positions

    requested = np.concatenate(
        (
            normalized.generator_bus_positions,
            fleet.renewable_bus_positions,
            fleet.storage_bus_positions,
        )
    )
    unique_positions = np.unique(requested)
    device_columns_active = factorization.shift_columns(
        _readonly_int(unique_positions), chunk_columns=policy.ptdf_rhs_chunk_columns
    )
    flow_offset_active = factorization.affine_flow_offset()
    load_shift_active = factorization.shift_vector(network.demand_mw)

    all_device_columns = np.zeros(
        (normalized.spec.branches, unique_positions.size), dtype=np.float64
    )
    all_device_columns[active_rows] = device_columns_active
    flow_offset = np.zeros(normalized.spec.branches, dtype=np.float64)
    load_shift = np.zeros(normalized.spec.branches, dtype=np.float64)
    flow_offset[active_rows] = flow_offset_active
    load_shift[active_rows] = load_shift_active
    fixed_flow = flow_offset[None, :] - fleet.load_multipliers[:, None] * load_shift[None, :]
    unique_lookup = {int(position): index for index, position in enumerate(unique_positions)}

    def columns(positions: IntArray) -> sparse.csr_matrix:
        indices = [unique_lookup[int(position)] for position in positions]
        return _sparsified_csr(all_device_columns[:, indices], zero_atol=policy.ptdf_zero_atol)

    return Stage7CompressedPTDF(
        bus_ids=network.bus_ids,
        branch_ids=normalized.branch_ids,
        reference_bus_id=network.reference_bus_id,
        generator_matrix=columns(normalized.generator_bus_positions),
        renewable_matrix=columns(fleet.renewable_bus_positions),
        storage_matrix=columns(fleet.storage_bus_positions),
        fixed_flow_mw=_readonly_float(fixed_flow),
        flow_offset_mw=_readonly_float(flow_offset),
        load_shift_mw=_readonly_float(load_shift),
        zero_atol=policy.ptdf_zero_atol,
        reduced_bus_matrix_nnz=factorization.reduced_bus_matrix_nnz,
    )


@dataclass(frozen=True, slots=True)
class Stage7NNZLedger:
    """Exact reconstructed nnz count obtained without horizon matrix allocation."""

    case_name: str
    periods: int
    published_nnz: int
    reconstructed_nnz: int
    difference_from_paper: int
    one_period_one_sided_line_nnz: int
    expanded_line_nnz: int
    nonline_nnz: int
    reconstructed_csr_bytes: int
    count_kind: str
    ptdf_zero_atol: float
    policy_fingerprint: str
    full_lp_allocated: bool = False
    stage_8_large_allocation_locked: bool = True

    @property
    def matches_paper(self) -> bool:
        return self.difference_from_paper == 0

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "case_name": self.case_name,
            "periods": self.periods,
            "published_nnz": self.published_nnz,
            "reconstructed_nnz": self.reconstructed_nnz,
            "difference_from_paper": self.difference_from_paper,
            "matches_paper": self.matches_paper,
            "one_period_one_sided_line_nnz": self.one_period_one_sided_line_nnz,
            "expanded_line_nnz": self.expanded_line_nnz,
            "nonline_nnz": self.nonline_nnz,
            "reconstructed_csr_bytes": self.reconstructed_csr_bytes,
            "count_kind": self.count_kind,
            "ptdf_zero_atol": self.ptdf_zero_atol,
            "policy_fingerprint": self.policy_fingerprint,
            "full_lp_allocated": self.full_lp_allocated,
            "stage_8_large_allocation_locked": self.stage_8_large_allocation_locked,
        }


def stage7_reconstructed_nnz_ledger(
    normalized: Stage7NormalizedCase,
    periods: int,
    *,
    fleet: Stage7ResourceFleet | None = None,
    ptdf: Stage7CompressedPTDF | None = None,
    factorization: Stage7SparseDCFactorization | None = None,
    policy: Stage7ReconstructionPolicy = FROZEN_STAGE7_POLICY,
) -> Stage7NNZLedger:
    """Count exact reconstructed support using only selected-bus PTDF columns.

    This function never constructs ``A1`` or ``A2``.  In particular, it is the
    Stage 7-safe path for every case9241 Table II row while Stage 8 large LP
    allocation remains locked.
    """

    row = _TABLE_II_BY_KEY.get((normalized.spec.name, periods))
    if row is None:
        raise KeyError(f"No Table II row for ({normalized.spec.name!r}, T={periods}).")
    fleet = (
        reconstruct_stage7_resources(normalized, periods, policy=policy) if fleet is None else fleet
    )
    if fleet.periods != periods:
        # Device placement does not depend on T, but the explicit check avoids
        # accidentally mixing fixed-flow arrays across execution paths.
        raise ValueError("fleet.periods must match the requested nnz ledger horizon.")
    if ptdf is None:
        factorization = (
            build_stage7_sparse_dc_factorization(normalized)
            if factorization is None
            else factorization
        )
        ptdf = build_stage7_compressed_ptdf(
            normalized,
            fleet,
            policy=policy,
            factorization=factorization,
        )
    line_base = int(
        ptdf.generator_matrix.nnz + ptdf.renewable_matrix.nnz + 2 * ptdf.storage_matrix.nnz
    )
    expanded_line = 2 * periods * line_base
    dense_total = _dense_structural_nnz_upper_bound(normalized.spec, periods)
    dense_line = (
        2
        * periods
        * normalized.spec.branches
        * (normalized.spec.generators + normalized.spec.renewables + 2 * normalized.spec.storage)
    )
    nonline = dense_total - dense_line
    reconstructed = expanded_line + nonline
    _, m1, m2, _ = _dimensions(normalized.spec, periods)
    csr_bytes = reconstructed * (8 + 4) + (m1 + m2 + 2) * 4
    return Stage7NNZLedger(
        case_name=normalized.spec.name,
        periods=periods,
        published_nnz=row.published_nnz,
        reconstructed_nnz=reconstructed,
        difference_from_paper=reconstructed - row.published_nnz,
        one_period_one_sided_line_nnz=line_base,
        expanded_line_nnz=expanded_line,
        nonline_nnz=nonline,
        reconstructed_csr_bytes=csr_bytes,
        count_kind="exact_reconstruction_selected_bus_batched_sparse_ptdf_support",
        ptdf_zero_atol=ptdf.zero_atol,
        policy_fingerprint=policy.fingerprint,
    )


def stage7_case_symbolic_ledgers(
    network: NetworkCase,
    *,
    policy: Stage7ReconstructionPolicy = FROZEN_STAGE7_POLICY,
) -> tuple[Stage7NNZLedger, ...]:
    """Count every Table II horizon for one case with one selected-bus solve."""

    normalized = normalize_stage7_case(network)
    representative_periods = normalized.spec.horizons[0]
    representative_fleet = reconstruct_stage7_resources(
        normalized, representative_periods, policy=policy
    )
    factorization = build_stage7_sparse_dc_factorization(normalized)
    representative_ptdf = build_stage7_compressed_ptdf(
        normalized,
        representative_fleet,
        policy=policy,
        factorization=factorization,
    )
    ledgers: list[Stage7NNZLedger] = []
    for periods in normalized.spec.horizons:
        fleet = reconstruct_stage7_resources(normalized, periods, policy=policy)
        # PTDF support is horizon-independent.  Reuse the selected-bus matrices
        # while constructing no expanded LP; only zero_atol and support matter.
        fixed = np.zeros((periods, normalized.spec.branches), dtype=np.float64)
        reusable_ptdf = Stage7CompressedPTDF(
            bus_ids=representative_ptdf.bus_ids,
            branch_ids=representative_ptdf.branch_ids,
            reference_bus_id=representative_ptdf.reference_bus_id,
            generator_matrix=representative_ptdf.generator_matrix,
            renewable_matrix=representative_ptdf.renewable_matrix,
            storage_matrix=representative_ptdf.storage_matrix,
            fixed_flow_mw=_readonly_float(fixed),
            flow_offset_mw=representative_ptdf.flow_offset_mw,
            load_shift_mw=representative_ptdf.load_shift_mw,
            zero_atol=representative_ptdf.zero_atol,
            reduced_bus_matrix_nnz=representative_ptdf.reduced_bus_matrix_nnz,
        )
        ledgers.append(
            stage7_reconstructed_nnz_ledger(
                normalized,
                periods,
                fleet=fleet,
                ptdf=reusable_ptdf,
                policy=policy,
            )
        )
    return tuple(ledgers)


def all_stage7_symbolic_ledgers(
    networks: Mapping[str, NetworkCase],
    *,
    policy: Stage7ReconstructionPolicy = FROZEN_STAGE7_POLICY,
) -> tuple[Stage7NNZLedger, ...]:
    """Return exact reconstructed nnz ledgers for all 18 rows, with no full LPs."""

    missing = sorted(set(PAPER_CASE_SPECS) - set(networks))
    if missing:
        raise KeyError(f"Missing public MATPOWER cases for symbolic ledger: {missing}.")
    result: list[Stage7NNZLedger] = []
    for case_name in PAPER_CASE_SPECS:
        result.extend(stage7_case_symbolic_ledgers(networks[case_name], policy=policy))
    return tuple(result)


def _outward_box_flow_bound(
    fixed_flow: FloatArray,
    coefficient_blocks: Sequence[sparse.csr_matrix],
    maximum_absolute_blocks: Sequence[FloatArray],
    rows: NDArray[np.int64],
) -> FloatArray:
    """Outward-rounded triangle bound for selected affine-flow rows."""

    result = np.zeros(rows.size, dtype=np.float64)
    for output, row in enumerate(rows):
        terms = [math.nextafter(abs(float(value)), math.inf) for value in fixed_flow[:, row]]
        for matrix, bounds in zip(coefficient_blocks, maximum_absolute_blocks, strict=True):
            csr_row = matrix.getrow(int(row))
            for value, column in zip(csr_row.data, csr_row.indices, strict=True):
                product = abs(float(value)) * float(bounds[column])
                terms.append(math.nextafter(product, math.inf))
        result[output] = math.nextafter(math.fsum(terms), math.inf)
    return result


def _derive_line_limits(
    normalized: Stage7NormalizedCase,
    fleet: Stage7ResourceFleet,
    ptdf: Stage7CompressedPTDF,
) -> tuple[FloatArray, FloatArray]:
    rates = np.asarray(normalized.original_rate_a_mw, dtype=np.float64).copy()
    proof = np.full(rates.size, np.nan, dtype=np.float64)
    zero_rows = np.flatnonzero(normalized.zero_rate_mask)
    if zero_rows.size:
        generator_maximum_absolute = np.maximum(
            np.abs(normalized.generator_lower_mw),
            np.abs(normalized.generator_upper_mw),
        )
        renewable_maximum_absolute = np.max(
            np.maximum(
                np.abs(fleet.renewable_minimum_mw),
                np.abs(fleet.renewable_maximum_mw),
            ),
            axis=0,
        )
        storage_maximum_absolute = np.maximum(
            fleet.storage_maximum_charge_mw,
            fleet.storage_maximum_discharge_mw,
        )
        bounds = _outward_box_flow_bound(
            ptdf.fixed_flow_mw,
            (ptdf.generator_matrix, ptdf.renewable_matrix, ptdf.storage_matrix),
            (
                generator_maximum_absolute,
                renewable_maximum_absolute,
                2.0 * storage_maximum_absolute,
            ),
            zero_rows,
        )
        # Storage has two opposite-signed variables; the factor of two above
        # bounds their combined absolute contribution.  nextafter makes the
        # installed limit strictly larger than the outward-rounded proof.
        proof[zero_rows] = bounds
        rates[zero_rows] = np.nextafter(bounds, np.inf)
    if np.any(rates <= 0.0) or not np.all(np.isfinite(rates)):
        raise ValueError("Every Stage 7 branch limit must be finite and positive.")
    return _readonly_float(rates), _readonly_float(proof)


def _zeros(rows: int, columns: int) -> sparse.csr_matrix:
    return sparse.csr_matrix((rows, columns), dtype=np.float64)


def _time_kron(periods: int, block: sparse.spmatrix) -> sparse.csr_matrix:
    return sparse.kron(sparse.eye(periods, format="csr"), block, format="csr")


def _block_row(blocks: Sequence[sparse.spmatrix], *, rows: int) -> sparse.csr_matrix:
    if any(block.shape[0] != rows for block in blocks):
        raise ValueError("Sparse block-row heights differ.")
    return sparse.hstack(blocks, format="csr", dtype=np.float64)


def _metadata(
    normalized: Stage7NormalizedCase,
    fleet: Stage7ResourceFleet,
) -> tuple[CompactRowMetadata, CompactRowMetadata]:
    periods = fleet.periods
    equality: list[RowFamilyBlock] = []
    start = 0
    equality.append(RowFamilyBlock(start, "power_balance", "equality", "1", ("system",), periods))
    start = equality[-1].stop
    equality.append(
        RowFamilyBlock(
            start,
            "storage_terminal_energy",
            "equality",
            "9",
            fleet.storage_ids,
            1,
            period_is_none=True,
        )
    )

    inequality: list[RowFamilyBlock] = []

    def add(
        family: str,
        side: str,
        equation: str,
        element_ids: tuple[str, ...],
        period_count: int,
        period_offset: int = 0,
    ) -> None:
        row_start = inequality[-1].stop if inequality else 0
        inequality.append(
            RowFamilyBlock(
                row_start,
                family,
                side,
                equation,
                element_ids,
                period_count,
                period_offset,
            )
        )

    add("line_flow", "lower", "2", normalized.branch_ids, periods)
    add("line_flow", "upper", "2", normalized.branch_ids, periods)
    add("reserve_headroom", "upper", "4", normalized.generator_ids, periods)
    add("reserve_footroom", "lower", "4", normalized.generator_ids, periods)
    add("reserve_requirement", "up", "5", ("system",), periods)
    add("reserve_requirement", "down", "5", ("system",), periods)
    if periods > 1:
        add("generator_ramp", "down", "6", normalized.generator_ids, periods - 1, 1)
        add("generator_ramp", "up", "6", normalized.generator_ids, periods - 1, 1)
    add("storage_energy", "lower", "8", fleet.storage_ids, periods)
    add("storage_energy", "upper", "8", fleet.storage_ids, periods)
    return CompactRowMetadata(equality), CompactRowMetadata(inequality)


@dataclass(frozen=True, slots=True)
class Stage7ScalableModel:
    """Canonical LP plus compact Stage 7 reconstruction provenance."""

    normalized: Stage7NormalizedCase
    fleet: Stage7ResourceFleet
    dc_factorization: Stage7SparseDCFactorization
    ptdf: Stage7CompressedPTDF
    variables: Stage7VariableIndex
    lp: CanonicalLP
    equality_rows: CompactRowMetadata
    inequality_rows: CompactRowMetadata
    line_limits_mw: FloatArray
    derived_line_limit_proof_mw: FloatArray
    load_mw: FloatArray
    objective_constant: float
    preflight: Stage7Preflight | None

    @property
    def network(self) -> NetworkCase:
        return self.normalized.network

    @property
    def generators(self) -> tuple:
        """All paper-count generator rows, including fixed-zero offline rows."""

        return self.network.generators

    def expected_dimensions(self) -> dict[str, int]:
        n, m1, m2, m = _dimensions(self.normalized.spec, self.fleet.periods)
        return {"n": n, "m1": m1, "m2": m2, "m": m}

    def dimension_summary(self) -> dict[str, int]:
        return {
            "periods": self.fleet.periods,
            "buses": self.normalized.spec.buses,
            "generators": self.normalized.spec.generators,
            "offline_generators_fixed_zero": self.normalized.offline_generator_count,
            "renewables": self.normalized.spec.renewables,
            "storage": self.normalized.spec.storage,
            "branches": self.normalized.spec.branches,
            "zero_rate_branches_derived": self.normalized.zero_rate_branch_count,
            "ignored_angle_limits": self.normalized.ignored_angle_limit_count,
            "n": self.lp.n,
            "m1": self.lp.m1,
            "m2": self.lp.m2,
            "m": self.lp.m,
            "nnz_A1": int(self.lp.A1.nnz),
            "nnz_A2": int(self.lp.A2.nnz),
            "nnz_A": int(self.lp.A1.nnz + self.lp.A2.nnz),
            "metadata_blocks": self.equality_rows.block_count + self.inequality_rows.block_count,
        }

    def objective(self, x: ArrayLike, *, include_constant: bool = True) -> float:
        vector = np.asarray(x, dtype=np.float64)
        if vector.shape != (self.lp.n,):
            raise ValueError(f"x has shape {vector.shape}; expected {(self.lp.n,)}.")
        value = float(self.lp.c @ vector)
        return value + self.objective_constant if include_constant else value

    def unpack(self, x: ArrayLike) -> dict[str, FloatArray]:
        vector = np.asarray(x, dtype=np.float64)
        if vector.shape != (self.lp.n,):
            raise ValueError(f"x has shape {vector.shape}; expected {(self.lp.n,)}.")
        result: dict[str, FloatArray] = {}
        counts: dict[VariableKind, int] = {
            "p_g": self.normalized.spec.generators,
            "p_rg": self.normalized.spec.renewables,
            "p_ess_dc": self.normalized.spec.storage,
            "p_ess_ch": self.normalized.spec.storage,
            "r_up": self.normalized.spec.generators,
            "r_down": self.normalized.spec.generators,
        }
        for kind in Stage7VariableIndex._block_order:
            result[kind] = vector[self.variables.block_slice(kind)].reshape(
                self.fleet.periods, counts[kind]
            )
        return result


@dataclass(frozen=True, slots=True)
class Stage7PhysicalValidation:
    """Independent, batched physical violations for the paper's Eqs. (1)-(10)."""

    equation_1_power_balance_mw: float
    equation_2_line_limit_mw: float
    equation_3_reserve_box_mw: float
    equation_4_headroom_footroom_mw: float
    equation_5_reserve_requirement_mw: float
    equation_6_generator_ramp_mw: float
    equation_7_renewable_box_mw: float
    equation_8_storage_energy_mwh: float
    equation_9_terminal_energy_mwh: float
    equation_10_storage_power_box_mw: float
    angle_vs_compressed_ptdf_flow_max_abs_mw: float
    maximum_violation: float
    sparse_factorization_reused: bool = True
    batched_periods: bool = True

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "equation_1_power_balance_mw": self.equation_1_power_balance_mw,
            "equation_2_line_limit_mw": self.equation_2_line_limit_mw,
            "equation_3_reserve_box_mw": self.equation_3_reserve_box_mw,
            "equation_4_headroom_footroom_mw": (self.equation_4_headroom_footroom_mw),
            "equation_5_reserve_requirement_mw": (self.equation_5_reserve_requirement_mw),
            "equation_6_generator_ramp_mw": self.equation_6_generator_ramp_mw,
            "equation_7_renewable_box_mw": self.equation_7_renewable_box_mw,
            "equation_8_storage_energy_mwh": self.equation_8_storage_energy_mwh,
            "equation_9_terminal_energy_mwh": self.equation_9_terminal_energy_mwh,
            "equation_10_storage_power_box_mw": (self.equation_10_storage_power_box_mw),
            "angle_vs_compressed_ptdf_flow_max_abs_mw": (
                self.angle_vs_compressed_ptdf_flow_max_abs_mw
            ),
            "maximum_violation": self.maximum_violation,
            "sparse_factorization_reused": self.sparse_factorization_reused,
            "batched_periods": self.batched_periods,
        }


def _maximum_positive(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    return max(0.0, float(np.max(array, initial=0.0)))


def _box_violation(values: FloatArray, lower: ArrayLike, upper: ArrayLike) -> float:
    return max(
        _maximum_positive(np.asarray(lower, dtype=np.float64) - values),
        _maximum_positive(values - np.asarray(upper, dtype=np.float64)),
    )


def validate_stage7_physical(
    model: Stage7ScalableModel,
    x: ArrayLike,
) -> Stage7PhysicalValidation:
    """Validate Eqs. (1)-(10) with one sparse batched angle solve.

    This path does not call ``dense_A()``, form a dense bus matrix, or refactor
    the network per period.  It is therefore suitable for the T=96 Stage 7
    execution boundary.
    """

    blocks = model.unpack(x)
    periods = model.fleet.periods
    buses = model.normalized.spec.buses

    def bus_device_map(positions: IntArray) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (
                np.ones(positions.size, dtype=np.float64),
                (positions, np.arange(positions.size)),
            ),
            shape=(buses, positions.size),
        )

    generator_map = bus_device_map(model.normalized.generator_bus_positions)
    renewable_map = bus_device_map(model.fleet.renewable_bus_positions)
    storage_map = bus_device_map(model.fleet.storage_bus_positions)
    injections = -np.asarray(model.load_mw, dtype=np.float64).copy()
    injections += np.asarray(generator_map @ blocks["p_g"].T).T
    injections += np.asarray(renewable_map @ blocks["p_rg"].T).T
    injections += np.asarray(storage_map @ (blocks["p_ess_dc"] - blocks["p_ess_ch"]).T).T
    balance = np.sum(injections, axis=1)
    _, angle_flows = model.dc_factorization.solve_angles_and_flows(
        injections, require_balanced=False
    )
    compressed_flows = (
        np.asarray(model.ptdf.generator_matrix @ blocks["p_g"].T).T
        + np.asarray(model.ptdf.renewable_matrix @ blocks["p_rg"].T).T
        + np.asarray(model.ptdf.storage_matrix @ (blocks["p_ess_dc"] - blocks["p_ess_ch"]).T).T
        + model.ptdf.fixed_flow_mw
    )
    equation_1 = float(np.max(np.abs(balance), initial=0.0))
    equation_2 = _maximum_positive(np.abs(angle_flows) - model.line_limits_mw[None, :])

    zero_generator = np.zeros_like(blocks["r_up"])
    ramp_up_box = np.tile(model.fleet.generator_ramp_up_mw, (periods, 1))
    ramp_down_box = np.tile(model.fleet.generator_ramp_down_mw, (periods, 1))
    equation_3 = max(
        _box_violation(blocks["r_up"], zero_generator, ramp_up_box),
        _box_violation(blocks["r_down"], zero_generator, ramp_down_box),
    )
    generator_minimum = model.normalized.generator_lower_mw[None, :]
    generator_maximum = model.normalized.generator_upper_mw[None, :]
    equation_4 = max(
        _maximum_positive(blocks["p_g"] + blocks["r_up"] - generator_maximum),
        _maximum_positive(generator_minimum - blocks["p_g"] + blocks["r_down"]),
    )
    equation_5 = max(
        _maximum_positive(model.fleet.reserve_up_mw - np.sum(blocks["r_up"], axis=1)),
        _maximum_positive(model.fleet.reserve_down_mw - np.sum(blocks["r_down"], axis=1)),
    )
    if periods > 1:
        generator_difference = np.diff(blocks["p_g"], axis=0)
        equation_6 = max(
            _maximum_positive(generator_difference - model.fleet.generator_ramp_up_mw[None, :]),
            _maximum_positive(-model.fleet.generator_ramp_down_mw[None, :] - generator_difference),
        )
    else:
        equation_6 = 0.0
    equation_7 = _box_violation(
        blocks["p_rg"],
        model.fleet.renewable_minimum_mw,
        model.fleet.renewable_maximum_mw,
    )
    energy_change = model.fleet.interval_hours * (
        blocks["p_ess_ch"] * model.fleet.storage_charge_efficiency[None, :]
        - blocks["p_ess_dc"] / model.fleet.storage_discharge_efficiency[None, :]
    )
    energy = model.fleet.storage_initial_energy_mwh[None, :] + np.cumsum(energy_change, axis=0)
    equation_8 = _box_violation(
        energy,
        model.fleet.storage_minimum_energy_mwh[None, :],
        model.fleet.storage_maximum_energy_mwh[None, :],
    )
    equation_9 = float(np.max(np.abs(np.sum(energy_change, axis=0)), initial=0.0))
    zero_storage = np.zeros_like(blocks["p_ess_dc"])
    equation_10 = max(
        _box_violation(
            blocks["p_ess_dc"],
            zero_storage,
            model.fleet.storage_maximum_discharge_mw[None, :],
        ),
        _box_violation(
            blocks["p_ess_ch"],
            zero_storage,
            model.fleet.storage_maximum_charge_mw[None, :],
        ),
    )
    angle_difference = float(np.max(np.abs(angle_flows - compressed_flows), initial=0.0))
    violations = (
        equation_1,
        equation_2,
        equation_3,
        equation_4,
        equation_5,
        equation_6,
        equation_7,
        equation_8,
        equation_9,
        equation_10,
    )
    return Stage7PhysicalValidation(
        equation_1_power_balance_mw=equation_1,
        equation_2_line_limit_mw=equation_2,
        equation_3_reserve_box_mw=equation_3,
        equation_4_headroom_footroom_mw=equation_4,
        equation_5_reserve_requirement_mw=equation_5,
        equation_6_generator_ramp_mw=equation_6,
        equation_7_renewable_box_mw=equation_7,
        equation_8_storage_energy_mwh=equation_8,
        equation_9_terminal_energy_mwh=equation_9,
        equation_10_storage_power_box_mw=equation_10,
        angle_vs_compressed_ptdf_flow_max_abs_mw=angle_difference,
        maximum_violation=max(violations),
    )


def build_stage7_scalable_model(
    network: NetworkCase,
    periods: int,
    *,
    spec: Stage7CaseSpec | None = None,
    policy: Stage7ReconstructionPolicy = FROZEN_STAGE7_POLICY,
    host_memory_budget_bytes: int | None = None,
) -> Stage7ScalableModel:
    """Build the reconstructed multi-period LP with vectorized sparse blocks."""

    normalized = normalize_stage7_case(network, spec=spec)
    fleet = reconstruct_stage7_resources(normalized, periods, policy=policy)
    preflight = None
    if (normalized.spec.name, periods) in _TABLE_II_BY_KEY:
        preflight = stage7_preflight(normalized.spec.name, periods, policy=policy)
        if not preflight.dimensions_match_table:
            raise AssertionError("Stage 7 dimensions do not match Table II.")
        if host_memory_budget_bytes is not None and (
            preflight.host_assembly_peak_bytes > host_memory_budget_bytes
        ):
            raise MemoryError(
                "Stage 7 host assembly preflight exceeds the supplied budget: "
                f"{preflight.host_assembly_peak_bytes} > {host_memory_budget_bytes} bytes."
            )
    elif host_memory_budget_bytes is not None and host_memory_budget_bytes <= 0:
        raise ValueError("host_memory_budget_bytes must be positive.")

    dc_factorization = build_stage7_sparse_dc_factorization(normalized)
    ptdf = build_stage7_compressed_ptdf(
        normalized, fleet, policy=policy, factorization=dc_factorization
    )
    line_limits, line_proof = _derive_line_limits(normalized, fleet, ptdf)
    variables = Stage7VariableIndex(
        periods,
        normalized.generator_ids,
        fleet.renewable_ids,
        fleet.storage_ids,
    )
    generators = normalized.spec.generators
    renewables = normalized.spec.renewables
    storage = normalized.spec.storage
    n = len(variables)

    lower = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    cost = np.zeros(n, dtype=np.float64)
    pg = variables.block_slice("p_g")
    prg = variables.block_slice("p_rg")
    pdc = variables.block_slice("p_ess_dc")
    pch = variables.block_slice("p_ess_ch")
    rup = variables.block_slice("r_up")
    rdown = variables.block_slice("r_down")
    lower[pg] = np.tile(normalized.generator_lower_mw, periods)
    upper[pg] = np.tile(normalized.generator_upper_mw, periods)
    cost[pg] = np.tile(normalized.generator_cost_slope, periods)
    lower[prg] = fleet.renewable_minimum_mw.reshape(-1)
    upper[prg] = fleet.renewable_maximum_mw.reshape(-1)
    cost[prg] = -fleet.renewable_penalty_per_mwh
    upper[pdc] = np.tile(fleet.storage_maximum_discharge_mw, periods)
    upper[pch] = np.tile(fleet.storage_maximum_charge_mw, periods)
    cost[pdc] = np.tile(
        fleet.storage_loss_penalty_per_mwh * (1.0 / fleet.storage_discharge_efficiency - 1.0),
        periods,
    )
    cost[pch] = np.tile(
        fleet.storage_loss_penalty_per_mwh * (1.0 - fleet.storage_charge_efficiency),
        periods,
    )
    upper[rup] = np.tile(fleet.generator_ramp_up_mw, periods)
    upper[rdown] = np.tile(fleet.generator_ramp_down_mw, periods)
    objective_constant = float(
        periods * np.sum(normalized.generator_cost_constant)
        + fleet.renewable_penalty_per_mwh * np.sum(fleet.renewable_maximum_mw)
    )

    # Equality rows: system balance followed by one terminal row per ESS.
    balance_pg = _time_kron(periods, sparse.csr_matrix(np.ones((1, generators))))
    balance_rg = _time_kron(periods, sparse.csr_matrix(np.ones((1, renewables))))
    balance_s = _time_kron(periods, sparse.csr_matrix(np.ones((1, storage))))
    balance = _block_row(
        (
            balance_pg,
            balance_rg,
            balance_s,
            -balance_s,
            _zeros(periods, periods * generators),
            _zeros(periods, periods * generators),
        ),
        rows=periods,
    )
    time_sum = sparse.csr_matrix(np.ones((1, periods)))
    terminal_dc = sparse.kron(
        time_sum,
        sparse.diags(-fleet.interval_hours / fleet.storage_discharge_efficiency),
        format="csr",
    )
    terminal_ch = sparse.kron(
        time_sum,
        sparse.diags(fleet.interval_hours * fleet.storage_charge_efficiency),
        format="csr",
    )
    terminal = _block_row(
        (
            _zeros(storage, periods * generators),
            _zeros(storage, periods * renewables),
            terminal_dc,
            terminal_ch,
            _zeros(storage, periods * generators),
            _zeros(storage, periods * generators),
        ),
        rows=storage,
    )
    A1 = sparse.vstack((balance, terminal), format="csr")
    load_mw = fleet.load_multipliers[:, None] * network.demand_mw[None, :]
    b1 = np.concatenate((np.sum(load_mw, axis=1), np.zeros(storage)))

    # Inequality rows are assembled a family at a time with Kronecker blocks.
    line_pg = _time_kron(periods, ptdf.generator_matrix)
    line_rg = _time_kron(periods, ptdf.renewable_matrix)
    line_s = _time_kron(periods, ptdf.storage_matrix)
    line_rows = periods * normalized.spec.branches
    line = _block_row(
        (
            line_pg,
            line_rg,
            line_s,
            -line_s,
            _zeros(line_rows, periods * generators),
            _zeros(line_rows, periods * generators),
        ),
        rows=line_rows,
    )

    generator_identity = sparse.eye(periods * generators, format="csr")
    variable_widths = (
        periods * generators,
        periods * renewables,
        periods * storage,
        periods * storage,
        periods * generators,
        periods * generators,
    )

    def generator_constraint(
        pg_sign: float, reserve_kind: Literal["up", "down"]
    ) -> sparse.csr_matrix:
        blocks = [_zeros(periods * generators, width) for width in variable_widths]
        blocks[0] = pg_sign * generator_identity
        blocks[4 if reserve_kind == "up" else 5] = -generator_identity
        return _block_row(blocks, rows=periods * generators)

    headroom = generator_constraint(-1.0, "up")
    footroom = generator_constraint(1.0, "down")
    reserve_sum = _time_kron(periods, sparse.csr_matrix(np.ones((1, generators))))

    def reserve_constraint(kind: Literal["up", "down"]) -> sparse.csr_matrix:
        blocks = [_zeros(periods, width) for width in variable_widths]
        blocks[4 if kind == "up" else 5] = reserve_sum
        return _block_row(blocks, rows=periods)

    reserve_up = reserve_constraint("up")
    reserve_down = reserve_constraint("down")

    if periods > 1:
        differences = sparse.diags(
            (-np.ones(periods - 1), np.ones(periods - 1)),
            (0, 1),
            shape=(periods - 1, periods),
            format="csr",
        )
        ramp_pg = sparse.kron(differences, sparse.eye(generators, format="csr"), format="csr")
        ramp_rows = (periods - 1) * generators
        ramp = _block_row(
            (ramp_pg, *(_zeros(ramp_rows, width) for width in variable_widths[1:])),
            rows=ramp_rows,
        )
    else:
        ramp_rows = 0
        ramp = _zeros(0, n)

    cumulative_time = sparse.csr_matrix(np.tril(np.ones((periods, periods))))
    energy_dc = sparse.kron(
        cumulative_time,
        sparse.diags(-fleet.interval_hours / fleet.storage_discharge_efficiency),
        format="csr",
    )
    energy_ch = sparse.kron(
        cumulative_time,
        sparse.diags(fleet.interval_hours * fleet.storage_charge_efficiency),
        format="csr",
    )
    energy_rows = periods * storage
    energy = _block_row(
        (
            _zeros(energy_rows, periods * generators),
            _zeros(energy_rows, periods * renewables),
            energy_dc,
            energy_ch,
            _zeros(energy_rows, periods * generators),
            _zeros(energy_rows, periods * generators),
        ),
        rows=energy_rows,
    )

    A2 = sparse.vstack(
        (
            line,
            -line,
            headroom,
            footroom,
            reserve_up,
            reserve_down,
            ramp,
            -ramp,
            energy,
            -energy,
        ),
        format="csr",
    )
    fixed_flow = ptdf.fixed_flow_mw
    b2 = np.concatenate(
        (
            (-line_limits[None, :] - fixed_flow).reshape(-1),
            (fixed_flow - line_limits[None, :]).reshape(-1),
            np.tile(-normalized.generator_upper_mw, periods),
            np.tile(normalized.generator_lower_mw, periods),
            fleet.reserve_up_mw,
            fleet.reserve_down_mw,
            np.tile(-fleet.generator_ramp_down_mw, max(periods - 1, 0)),
            np.tile(-fleet.generator_ramp_up_mw, max(periods - 1, 0)),
            np.tile(
                fleet.storage_minimum_energy_mwh - fleet.storage_initial_energy_mwh,
                periods,
            ),
            np.tile(
                fleet.storage_initial_energy_mwh - fleet.storage_maximum_energy_mwh,
                periods,
            ),
        )
    )
    lp = CanonicalLP(c=cost, A1=A1, b1=b1, A2=A2, b2=b2, lower=lower, upper=upper)
    equality_rows, inequality_rows = _metadata(normalized, fleet)
    expected = _dimensions(normalized.spec, periods)
    actual = (lp.n, lp.m1, lp.m2, lp.m)
    if actual != (expected[0], expected[1], expected[2], expected[3]):
        raise AssertionError(f"Stage 7 dimensions {actual} do not match {expected}.")
    if len(equality_rows) != lp.m1 or len(inequality_rows) != lp.m2:
        raise AssertionError("Compact Stage 7 metadata does not cover every LP row.")
    return Stage7ScalableModel(
        normalized=normalized,
        fleet=fleet,
        dc_factorization=dc_factorization,
        ptdf=ptdf,
        variables=variables,
        lp=lp,
        equality_rows=equality_rows,
        inequality_rows=inequality_rows,
        line_limits_mw=line_limits,
        derived_line_limit_proof_mw=line_proof,
        load_mw=_readonly_float(load_mw),
        objective_constant=objective_constant,
        preflight=preflight,
    )


__all__ = [
    "CompactRowMetadata",
    "FROZEN_STAGE7_POLICY",
    "PAPER_CASE_SPECS",
    "RowFamilyBlock",
    "Stage7CaseSpec",
    "Stage7CompressedPTDF",
    "Stage7NNZLedger",
    "Stage7NormalizedCase",
    "Stage7PhysicalValidation",
    "Stage7Preflight",
    "Stage7ReconstructionPolicy",
    "Stage7ResourceFleet",
    "Stage7ScalableModel",
    "Stage7SparseDCFactorization",
    "Stage7VariableIndex",
    "TABLE_II_ROWS",
    "TableIIRow",
    "all_stage7_preflights",
    "all_stage7_symbolic_ledgers",
    "assert_stage7_reconstruction_contract",
    "build_stage7_compressed_ptdf",
    "build_stage7_scalable_model",
    "build_stage7_sparse_dc_factorization",
    "normalize_stage7_case",
    "reconstruct_stage7_resources",
    "stage7_case_symbolic_ledgers",
    "stage7_preflight",
    "stage7_reconstructed_nnz_ledger",
    "validate_stage7_physical",
]
