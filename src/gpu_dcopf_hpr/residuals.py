"""Equation (28) KKT mapping and the distinct Equation (54) stopping tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .canonical_lp import CanonicalLP
from .projections import project_box, project_dual_set

FloatVector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class KKTResidualBlocks:
    """The three raw blocks of the paper's Equation (28)."""

    dual_projection: FloatVector
    box: FloatVector
    stationarity: FloatVector


@dataclass(frozen=True, slots=True)
class PaperStoppingBlocks:
    """Equation (54) blocks, before or after their published normalizations."""

    primal_feasibility: FloatVector
    box: FloatVector
    stationarity: FloatVector


@dataclass(frozen=True, slots=True)
class StoppingConditions:
    primal_feasibility: bool
    box: bool
    stationarity: bool

    @property
    def all_satisfied(self) -> bool:
        return self.primal_feasibility and self.box and self.stationarity


@dataclass(frozen=True, slots=True)
class ResidualEvaluation:
    """Complete residual evidence for one candidate primal-dual state."""

    kkt: KKTResidualBlocks
    paper_raw: PaperStoppingBlocks
    paper_normalized: PaperStoppingBlocks
    combined_norm: float
    normalized_combined_norm: float
    conditions: StoppingConditions
    tolerance: float

    @property
    def paper_normalized_norms(self) -> tuple[float, float, float]:
        return (
            float(np.linalg.norm(self.paper_normalized.primal_feasibility)),
            float(np.linalg.norm(self.paper_normalized.box)),
            float(np.linalg.norm(self.paper_normalized.stationarity)),
        )

    @property
    def paper_raw_norms(self) -> tuple[float, float, float]:
        return (
            float(np.linalg.norm(self.paper_raw.primal_feasibility)),
            float(np.linalg.norm(self.paper_raw.box)),
            float(np.linalg.norm(self.paper_raw.stationarity)),
        )

    def summary(self) -> dict[str, Any]:
        normalized = self.paper_normalized_norms
        raw = self.paper_raw_norms
        return {
            "kkt_combined_norm": self.combined_norm,
            "paper_normalized_combined_norm": self.normalized_combined_norm,
            "paper_raw_norms": {
                "primal_feasibility": raw[0],
                "box": raw[1],
                "stationarity": raw[2],
            },
            "paper_normalized_norms": {
                "primal_feasibility": normalized[0],
                "box": normalized[1],
                "stationarity": normalized[2],
            },
            "paper_stopping": {
                "primal_feasibility": self.conditions.primal_feasibility,
                "box": self.conditions.box,
                "stationarity": self.conditions.stationarity,
                "all_satisfied": self.conditions.all_satisfied,
            },
            "tolerance": self.tolerance,
        }


def _state_vector(values: ArrayLike, *, name: str, expected: int) -> FloatVector:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},); received {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _combined_norm(blocks: tuple[FloatVector, FloatVector, FloatVector]) -> float:
    squared = sum(float(np.dot(block, block)) for block in blocks)
    return float(np.sqrt(squared))


def evaluate_residuals(
    lp: CanonicalLP,
    *,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    tolerance: float = 5e-5,
) -> ResidualEvaluation:
    """Evaluate Eq. (28) and every separately normalized Eq. (54) condition.

    Equation (28)'s first block depends on ``y`` and captures projected dual
    complementarity. Equation (54a) is only a primal-feasibility test. They are
    intentionally returned under different names.
    """

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite scalar.")

    x_vector = _state_vector(x, name="x", expected=lp.n)
    y_vector = _state_vector(y, name="y", expected=lp.m)
    z_vector = _state_vector(z, name="z", expected=lp.n)
    A = lp.A
    b = lp.b

    Ax = np.asarray(A @ x_vector, dtype=np.float64).reshape(-1)
    Aty = np.asarray(A.T @ y_vector, dtype=np.float64).reshape(-1)

    dual_argument = y_vector - Ax + b
    dual_projection = y_vector - project_dual_set(dual_argument, lp.m1)
    box = x_vector - project_box(x_vector - z_vector, lp.lower, lp.upper)
    stationarity = lp.c - Aty - z_vector
    kkt = KKTResidualBlocks(
        dual_projection=dual_projection,
        box=box,
        stationarity=stationarity,
    )

    primal_feasibility = project_dual_set(b - Ax, lp.m1)
    paper_raw = PaperStoppingBlocks(
        primal_feasibility=primal_feasibility,
        box=box.copy(),
        stationarity=stationarity.copy(),
    )
    primal_denominator = 1.0 + float(np.linalg.norm(b))
    box_denominator = 1.0 + float(np.linalg.norm(x_vector)) + float(np.linalg.norm(z_vector))
    stationarity_denominator = 1.0 + float(np.linalg.norm(lp.c))
    paper_normalized = PaperStoppingBlocks(
        primal_feasibility=primal_feasibility / primal_denominator,
        box=box / box_denominator,
        stationarity=stationarity / stationarity_denominator,
    )
    normalized_norms = (
        float(np.linalg.norm(paper_normalized.primal_feasibility)),
        float(np.linalg.norm(paper_normalized.box)),
        float(np.linalg.norm(paper_normalized.stationarity)),
    )
    conditions = StoppingConditions(
        primal_feasibility=normalized_norms[0] <= tolerance,
        box=normalized_norms[1] <= tolerance,
        stationarity=normalized_norms[2] <= tolerance,
    )

    return ResidualEvaluation(
        kkt=kkt,
        paper_raw=paper_raw,
        paper_normalized=paper_normalized,
        combined_norm=_combined_norm((dual_projection, box, stationarity)),
        normalized_combined_norm=_combined_norm(
            (
                paper_normalized.primal_feasibility,
                paper_normalized.box,
                paper_normalized.stationarity,
            )
        ),
        conditions=conditions,
        tolerance=float(tolerance),
    )
