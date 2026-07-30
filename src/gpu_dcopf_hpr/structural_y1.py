"""Matrix-free Proposition 5 equality solve for the implemented DCOPF model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from .canonical_lp import CanonicalLP

if TYPE_CHECKING:
    from .dcopf_model import DCOPFModel

FloatVector = NDArray[np.float64]


def _readonly_vector(values: ArrayLike, *, name: str, length: int) -> FloatVector:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have length {length}; received shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(vector, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class DCOPFEqualityStructure:
    """Dimensions and storage coefficients required by the implemented Equation (55)."""

    periods: int
    generator_count: int
    renewable_count: int
    interval_hours: float
    charge_efficiencies: ArrayLike
    discharge_efficiencies: ArrayLike

    def __post_init__(self) -> None:
        for name in ("periods", "generator_count", "renewable_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
        if self.periods <= 0:
            raise ValueError("periods must be positive.")
        if self.generator_count < 0:
            raise ValueError("generator_count must be nonnegative.")
        if self.renewable_count < 0:
            raise ValueError("renewable_count must be nonnegative.")
        if not np.isfinite(self.interval_hours) or self.interval_hours <= 0.0:
            raise ValueError("interval_hours must be a positive finite scalar.")

        charge = np.asarray(self.charge_efficiencies, dtype=np.float64)
        discharge = np.asarray(self.discharge_efficiencies, dtype=np.float64)
        if charge.ndim != 1 or discharge.ndim != 1 or charge.shape != discharge.shape:
            raise ValueError(
                "charge_efficiencies and discharge_efficiencies must be equally sized vectors."
            )
        charge = _readonly_vector(
            charge,
            name="charge_efficiencies",
            length=int(charge.size),
        )
        discharge = _readonly_vector(
            discharge,
            name="discharge_efficiencies",
            length=int(charge.size),
        )
        if np.any(charge <= 0.0) or np.any(charge > 1.0):
            raise ValueError("charge_efficiencies must lie in (0, 1].")
        if np.any(discharge <= 0.0) or np.any(discharge > 1.0):
            raise ValueError("discharge_efficiencies must lie in (0, 1].")
        if self.generator_count + self.renewable_count + 2 * charge.size <= 0:
            raise ValueError("the power-balance rows must contain at least one resource.")

        object.__setattr__(self, "interval_hours", float(self.interval_hours))
        object.__setattr__(self, "charge_efficiencies", charge)
        object.__setattr__(self, "discharge_efficiencies", discharge)

    @property
    def storage_count(self) -> int:
        return int(np.asarray(self.charge_efficiencies).size)

    @property
    def expected_variables(self) -> int:
        resources = 3 * self.generator_count + self.renewable_count + 2 * self.storage_count
        return self.periods * resources

    @property
    def expected_equalities(self) -> int:
        return self.periods + self.storage_count

    @classmethod
    def from_model(cls, model: DCOPFModel) -> DCOPFEqualityStructure:
        """Extract and independently check the semantic Equation (55) dimensions."""

        periods = model.config.periods
        storage = model.config.storage
        expected_families = ("power_balance",) * periods + ("storage_terminal_energy",) * len(
            storage
        )
        actual_families = tuple(row.family for row in model.equality_rows)
        if actual_families != expected_families:
            raise ValueError(
                "DCOPF equality rows must be ordered as power balance followed by "
                "terminal-storage rows."
            )
        if tuple(row.period for row in model.equality_rows[:periods]) != tuple(range(periods)):
            raise ValueError("Power-balance equality rows must be ordered by period.")
        if tuple(row.element_id for row in model.equality_rows[periods:]) != tuple(
            resource.resource_id for resource in storage
        ):
            raise ValueError("Terminal-storage equality rows must follow storage order.")

        return cls(
            periods=periods,
            generator_count=len(model.generators),
            renewable_count=len(model.config.renewables),
            interval_hours=model.config.interval_hours,
            charge_efficiencies=tuple(resource.charge_efficiency for resource in storage),
            discharge_efficiencies=tuple(resource.discharge_efficiency for resource in storage),
        )


@dataclass(frozen=True, slots=True)
class StructuralY1Diagnostics:
    """Auditable scalar and diagonal terms for the Proposition 5 solve."""

    periods: int
    generator_count: int
    renewable_count: int
    storage_count: int
    equality_rows: int
    balance_diagonal: float
    coupling: FloatVector
    storage_diagonal: FloatVector
    alpha: float
    schur_scalar: float
    relative_schur_margin: float
    maximum_a1_pattern_error: float
    expected_a1_nonzeros: int
    stored_float_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "periods": self.periods,
            "generator_count": self.generator_count,
            "renewable_count": self.renewable_count,
            "storage_count": self.storage_count,
            "equality_rows": self.equality_rows,
            "balance_diagonal": self.balance_diagonal,
            "coupling": self.coupling.tolist(),
            "storage_diagonal": self.storage_diagonal.tolist(),
            "alpha": self.alpha,
            "schur_scalar": self.schur_scalar,
            "relative_schur_margin": self.relative_schur_margin,
            "maximum_a1_pattern_error": self.maximum_a1_pattern_error,
            "expected_a1_nonzeros": self.expected_a1_nonzeros,
            "stored_float_count": self.stored_float_count,
            "solve_complexity": "O(T + N_ESS)",
            "dense_gram_materialized": False,
            "explicit_kronecker_materialized": False,
        }


@dataclass(frozen=True, slots=True)
class StructuralY1Solver:
    """Prepared matrix-free equality solver tied to one canonical LP instance."""

    source_lp: CanonicalLP
    structure: DCOPFEqualityStructure
    inverse_storage_diagonal: FloatVector
    diagnostics: StructuralY1Diagnostics

    @property
    def backend(self) -> str:
        return "structural"

    def solve(self, right_hand_side: ArrayLike) -> FloatVector:
        """Solve ``(A1 A1.T) y1 = rhs`` without a Gram matrix or factorization."""

        rhs = np.asarray(right_hand_side, dtype=np.float64)
        expected = self.diagnostics.equality_rows
        if rhs.shape != (expected,):
            raise ValueError(
                f"right_hand_side must have shape ({expected},); received {rhs.shape}."
            )
        if not np.all(np.isfinite(rhs)):
            raise ValueError("right_hand_side must contain only finite values.")

        periods = self.diagnostics.periods
        balance_diagonal = self.diagnostics.balance_diagonal
        if self.diagnostics.storage_count == 0:
            return np.asarray(rhs / balance_diagonal, dtype=np.float64)

        balance_rhs = rhs[:periods]
        storage_rhs = rhs[periods:]
        coupling = self.diagnostics.coupling
        weighted_storage_rhs = float(np.dot(coupling * self.inverse_storage_diagonal, storage_rhs))
        reduced_rhs = balance_rhs - weighted_storage_rhs

        # The mean/zero-mean split is the cancellation-resistant form of
        # (a I - alpha 11.T)^-1.  The correction sign is positive.
        reduced_mean = float(np.mean(reduced_rhs))
        balance_solution = (
            reduced_rhs - reduced_mean
        ) / balance_diagonal + reduced_mean / self.diagnostics.schur_scalar
        storage_solution = self.inverse_storage_diagonal * (
            storage_rhs - coupling * float(np.sum(balance_solution))
        )
        return np.concatenate((balance_solution, storage_solution))


def _row_pattern_error(
    matrix: sparse.csr_matrix,
    row: int,
    expected_indices: NDArray[np.int64],
    expected_values: FloatVector,
) -> float:
    start = matrix.indptr[row]
    stop = matrix.indptr[row + 1]
    actual_indices = matrix.indices[start:stop]
    actual_values = matrix.data[start:stop]
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            f"A1 row {row} is incompatible with the implemented Equation (55) column pattern."
        )
    if actual_values.size == 0:
        return 0.0
    error = float(np.max(np.abs(actual_values - expected_values)))
    scale = max(
        1.0,
        float(np.max(np.abs(actual_values))),
        float(np.max(np.abs(expected_values))),
    )
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    if error > tolerance:
        raise ValueError(
            f"A1 row {row} is incompatible with the implemented Equation (55) "
            f"coefficients; maximum error {error} exceeds {tolerance}."
        )
    return error


def _validate_a1_structure(
    lp: CanonicalLP,
    structure: DCOPFEqualityStructure,
) -> tuple[float, int]:
    periods = structure.periods
    generators = structure.generator_count
    renewables = structure.renewable_count
    storage = structure.storage_count
    expected_shape = (structure.expected_equalities, structure.expected_variables)
    if (lp.m1, lp.n) != expected_shape:
        raise ValueError(
            "CanonicalLP dimensions are incompatible with the implemented Equation (55): "
            f"expected {expected_shape}, received {(lp.m1, lp.n)}."
        )

    matrix = sparse.csr_matrix(lp.A1, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    generator_offset = 0
    renewable_offset = periods * generators
    discharge_offset = renewable_offset + periods * renewables
    charge_offset = discharge_offset + periods * storage
    maximum_error = 0.0

    for period in range(periods):
        indices = np.concatenate(
            (
                generator_offset + period * generators + np.arange(generators),
                renewable_offset + period * renewables + np.arange(renewables),
                discharge_offset + period * storage + np.arange(storage),
                charge_offset + period * storage + np.arange(storage),
            )
        ).astype(np.int64)
        values = np.concatenate(
            (
                np.ones(generators + renewables + storage, dtype=np.float64),
                -np.ones(storage, dtype=np.float64),
            )
        )
        maximum_error = max(
            maximum_error,
            _row_pattern_error(matrix, period, indices, values),
        )

    charge_efficiencies = np.asarray(structure.charge_efficiencies)
    discharge_efficiencies = np.asarray(structure.discharge_efficiencies)
    for storage_index in range(storage):
        indices = np.concatenate(
            (
                discharge_offset + np.arange(periods) * storage + storage_index,
                charge_offset + np.arange(periods) * storage + storage_index,
            )
        ).astype(np.int64)
        values = np.concatenate(
            (
                np.full(
                    periods,
                    -structure.interval_hours / discharge_efficiencies[storage_index],
                    dtype=np.float64,
                ),
                np.full(
                    periods,
                    structure.interval_hours * charge_efficiencies[storage_index],
                    dtype=np.float64,
                ),
            )
        )
        maximum_error = max(
            maximum_error,
            _row_pattern_error(matrix, periods + storage_index, indices, values),
        )

    expected_nonzeros = periods * (generators + renewables + 4 * storage)
    if matrix.nnz != expected_nonzeros:
        raise ValueError(
            "A1 has an incompatible number of nonzeros for the implemented Equation (55): "
            f"expected {expected_nonzeros}, received {matrix.nnz}."
        )
    return maximum_error, expected_nonzeros


def prepare_structural_y1(
    lp: CanonicalLP,
    structure: DCOPFEqualityStructure,
) -> StructuralY1Solver:
    """Validate Equation (55) and prepare the corrected Proposition 5 solver."""

    maximum_pattern_error, expected_nonzeros = _validate_a1_structure(lp, structure)
    periods = structure.periods
    storage_count = structure.storage_count
    charge = np.asarray(structure.charge_efficiencies, dtype=np.float64)
    inverse_discharge = 1.0 / np.asarray(
        structure.discharge_efficiencies,
        dtype=np.float64,
    )
    interval = structure.interval_hours

    coupling = -interval * (charge + inverse_discharge)
    storage_diagonal = periods * interval**2 * (np.square(charge) + np.square(inverse_discharge))
    if not np.all(np.isfinite(coupling)) or not np.all(np.isfinite(storage_diagonal)):
        raise ValueError("Storage coefficients overflowed FP64 during structural preparation.")
    if np.any(storage_diagonal <= 0.0):
        raise ValueError("The structural storage diagonal must be positive.")
    inverse_storage_diagonal = np.asarray(1.0 / storage_diagonal, dtype=np.float64)
    balance_diagonal = float(
        structure.generator_count + structure.renewable_count + 2 * storage_count
    )
    alpha = float(np.dot(np.square(coupling), inverse_storage_diagonal))

    # Algebraically this equals a - alpha*T, but the expression below avoids
    # cancellation when many ideal-efficiency storage devices are present.
    schur_scalar = float(
        structure.generator_count
        + structure.renewable_count
        + np.sum(
            np.square(charge - inverse_discharge)
            / (np.square(charge) + np.square(inverse_discharge))
        )
    )
    schur_tolerance = 128.0 * np.finfo(np.float64).eps * max(1.0, balance_diagonal)
    if not np.isfinite(schur_scalar) or schur_scalar <= schur_tolerance:
        raise ValueError(
            "The structural Schur complement is singular or numerically unsafe: "
            f"{schur_scalar} <= {schur_tolerance}."
        )

    coupling = np.asarray(coupling, dtype=np.float64)
    storage_diagonal = np.asarray(storage_diagonal, dtype=np.float64)
    for vector in (coupling, storage_diagonal, inverse_storage_diagonal):
        vector.setflags(write=False)
    diagnostics = StructuralY1Diagnostics(
        periods=periods,
        generator_count=structure.generator_count,
        renewable_count=structure.renewable_count,
        storage_count=storage_count,
        equality_rows=structure.expected_equalities,
        balance_diagonal=balance_diagonal,
        coupling=coupling,
        storage_diagonal=storage_diagonal,
        alpha=alpha,
        schur_scalar=schur_scalar,
        relative_schur_margin=schur_scalar / balance_diagonal,
        maximum_a1_pattern_error=maximum_pattern_error,
        expected_a1_nonzeros=expected_nonzeros,
        stored_float_count=3 * storage_count,
    )
    return StructuralY1Solver(
        source_lp=lp,
        structure=structure,
        inverse_storage_diagonal=inverse_storage_diagonal,
        diagnostics=diagnostics,
    )


def prepare_dcopf_structural_y1(model: DCOPFModel) -> StructuralY1Solver:
    """Prepare the structural equality solver directly from a built DCOPF model."""

    return prepare_structural_y1(
        model.lp,
        DCOPFEqualityStructure.from_model(model),
    )
