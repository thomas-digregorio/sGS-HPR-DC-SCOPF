"""Deterministic tiny LPs used to validate Stage 1 mathematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .canonical_lp import CanonicalLP
from .hpr_generic import HPRState

FloatVector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    name: str
    description: str
    lp: CanonicalLP
    expected_state: HPRState
    expected_objective: float
    solution_tolerance: float
    provenance_seed: int | None = None


def analytic_toy_case() -> ReferenceCase:
    """The exact LP prescribed by the Stage 1 brief."""

    lp = CanonicalLP(
        c=[2.0, 1.0],
        A1=[[1.0, 1.0]],
        b1=[1.0],
        A2=[[1.0, -1.0]],
        b2=[-0.2],
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
    )
    return ReferenceCase(
        name="analytic_toy",
        description="Required analytic LP with one active inequality.",
        lp=lp,
        expected_state=HPRState(y=[1.5, 0.5], z=[0.0, 0.0], x=[0.4, 0.6]),
        expected_objective=1.4,
        solution_tolerance=5e-4,
    )


def box_bound_active_case() -> ReferenceCase:
    lp = CanonicalLP(
        c=[-1.0, 0.0],
        A1=[[1.0, 1.0]],
        b1=[1.0],
        A2=np.empty((0, 2)),
        b2=[],
        lower=[0.0, 0.0],
        upper=[0.75, 1.0],
    )
    return ReferenceCase(
        name="box_bound_active",
        description="Unique optimum on an upper box bound with no inequality rows.",
        lp=lp,
        expected_state=HPRState(y=[0.0], z=[-1.0, 0.0], x=[0.75, 0.25]),
        expected_objective=-0.75,
        solution_tolerance=5e-4,
    )


def inequality_inactive_case() -> ReferenceCase:
    lp = CanonicalLP(
        c=[1.0, 0.0],
        A1=[[1.0, 1.0]],
        b1=[1.0],
        A2=[[1.0, -1.0]],
        b2=[-0.8],
        lower=[0.25, 0.0],
        upper=[1.0, 1.0],
    )
    return ReferenceCase(
        name="inequality_inactive",
        description="The inequality has 0.3 slack and a zero multiplier at the optimum.",
        lp=lp,
        expected_state=HPRState(y=[0.0, 0.0], z=[1.0, 0.0], x=[0.25, 0.75]),
        expected_objective=0.25,
        solution_tolerance=5e-4,
    )


def planted_random_case() -> ReferenceCase:
    """A seeded, deliberately feasible LP with a planted unique KKT point."""

    lp = CanonicalLP(
        c=[
            -0.5941083934541853,
            0.5619109601872230,
            2.2175780215524390,
            3.3078757801039140,
            -6.0538910524445160,
        ],
        A1=[
            [
                -0.2716171072605280,
                0.0668914769230116,
                1.1219726675517807,
                -1.2101315987843864,
                0.3346055607163113,
            ],
            [
                0.5659211615521283,
                -0.1875439957990173,
                -0.8842674712258111,
                -1.1573972577179215,
                2.2890423358992193,
            ],
        ],
        b1=[1.0597848216157364, 0.3325174600787564],
        A2=[
            [
                0.2475233731877643,
                0.9563540395004890,
                0.8035256240140611,
                0.7039823988928422,
                -1.4070887641503578,
            ],
            [
                0.6976151710641401,
                -0.7164551304892209,
                -0.4625731222713524,
                -0.4298207085732225,
                -0.6734656623402371,
            ],
            [
                0.4475208539254938,
                -1.2410477181943390,
                0.6223379720807442,
                -2.2923157534787926,
                -1.0125630958240457,
            ],
            [
                -0.8236839061937683,
                1.6937129481436830,
                0.2559234155081145,
                -0.9192325203650720,
                0.9024773939498413,
            ],
        ],
        b2=[
            0.0955055657875411,
            -1.9470249783717597,
            -1.8814314899896645,
            2.3053422511285320,
        ],
        lower=[-1.0] * 5,
        upper=[1.0] * 5,
    )
    return ReferenceCase(
        name="planted_random",
        description=(
            "Seeded feasible LP with planted active multipliers, four inequality "
            "slacks, and a full-rank active system."
        ),
        lp=lp,
        expected_state=HPRState(
            y=[
                -0.2421934113292550,
                -2.1790290101297840,
                0.7,
                0.0,
                0.0,
                0.0,
            ],
            z=[0.4, -0.5, 0.0, 0.0, 0.0],
            x=[-1.0, 1.0, 0.2, -0.3, 0.4],
        ),
        expected_objective=-1.8143841970570846,
        solution_tolerance=5e-3,
        provenance_seed=20260729,
    )


def reference_cases() -> tuple[ReferenceCase, ...]:
    return (
        analytic_toy_case(),
        box_bound_active_case(),
        inequality_inactive_case(),
        planted_random_case(),
    )
