"""DC shift-factor construction with an independent angle-based check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import csgraph

from .network_data import MATPOWERCaseError, NetworkCase

FloatArray = NDArray[np.float64]


def _readonly(values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PTDF:
    """An affine DC flow map ``flow = matrix @ injection + offset``."""

    bus_ids: tuple[int, ...]
    branch_ids: tuple[str, ...]
    reference_bus_id: int
    matrix: FloatArray
    flow_offset_mw: FloatArray
    bus_susceptance_mw: FloatArray
    branch_angle_matrix_mw: FloatArray
    phase_shift_bus_injection_mw: FloatArray
    phase_shift_branch_flow_mw: FloatArray

    def __post_init__(self) -> None:
        matrix = _readonly(self.matrix)
        offset = _readonly(self.flow_offset_mw)
        bus_matrix = _readonly(self.bus_susceptance_mw)
        branch_matrix = _readonly(self.branch_angle_matrix_mw)
        bus_shift = _readonly(self.phase_shift_bus_injection_mw)
        branch_shift = _readonly(self.phase_shift_branch_flow_mw)
        n_bus = len(self.bus_ids)
        n_branch = len(self.branch_ids)
        if matrix.shape != (n_branch, n_bus):
            raise ValueError(f"PTDF matrix has shape {matrix.shape}; expected {(n_branch, n_bus)}.")
        if offset.shape != (n_branch,):
            raise ValueError("PTDF flow offset has the wrong dimension.")
        if bus_matrix.shape != (n_bus, n_bus):
            raise ValueError("Bus susceptance matrix has the wrong dimension.")
        if branch_matrix.shape != (n_branch, n_bus):
            raise ValueError("Branch angle matrix has the wrong dimension.")
        if bus_shift.shape != (n_bus,) or branch_shift.shape != (n_branch,):
            raise ValueError("Phase-shift vectors have the wrong dimension.")
        for name, values in (
            ("matrix", matrix),
            ("flow_offset_mw", offset),
            ("bus_susceptance_mw", bus_matrix),
            ("branch_angle_matrix_mw", branch_matrix),
            ("phase_shift_bus_injection_mw", bus_shift),
            ("phase_shift_branch_flow_mw", branch_shift),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains a nonfinite value.")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "flow_offset_mw", offset)
        object.__setattr__(self, "bus_susceptance_mw", bus_matrix)
        object.__setattr__(self, "branch_angle_matrix_mw", branch_matrix)
        object.__setattr__(self, "phase_shift_bus_injection_mw", bus_shift)
        object.__setattr__(self, "phase_shift_branch_flow_mw", branch_shift)

    @property
    def reference_position(self) -> int:
        return self.bus_ids.index(self.reference_bus_id)

    def _balanced_injection(self, injection_mw: ArrayLike) -> FloatArray:
        injection = np.asarray(injection_mw, dtype=np.float64)
        if injection.shape != (len(self.bus_ids),):
            raise ValueError(
                f"Injection vector has shape {injection.shape}; expected {(len(self.bus_ids),)}."
            )
        if not np.all(np.isfinite(injection)):
            raise ValueError("Injection vector must contain only finite values.")
        tolerance = 1e-10 * (1.0 + float(np.linalg.norm(injection, ord=1)))
        if abs(float(np.sum(injection))) > tolerance:
            raise ValueError(
                "DC branch flows require a balanced bus-injection vector; "
                f"sum={float(np.sum(injection)):.6g} MW."
            )
        return injection

    def flows_from_injections(self, injection_mw: ArrayLike) -> FloatArray:
        """Evaluate branch flows with the precomputed affine PTDF map."""

        injection = self._balanced_injection(injection_mw)
        return np.asarray(self.matrix @ injection + self.flow_offset_mw, dtype=np.float64)

    def angles_and_flows(self, injection_mw: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Solve reduced bus angles and compute flows without using the PTDF."""

        injection = self._balanced_injection(injection_mw)
        reference = self.reference_position
        nonreference = np.arange(len(self.bus_ids)) != reference
        reduced = self.bus_susceptance_mw[np.ix_(nonreference, nonreference)]
        right_hand_side = injection[nonreference] - self.phase_shift_bus_injection_mw[nonreference]
        angles = np.zeros(len(self.bus_ids), dtype=np.float64)
        angles[nonreference] = np.linalg.solve(reduced, right_hand_side)
        flows = self.branch_angle_matrix_mw @ angles + self.phase_shift_branch_flow_mw
        return angles, np.asarray(flows, dtype=np.float64)


def build_ptdf(
    network: NetworkCase,
    *,
    reference_bus_id: int | None = None,
    minimum_reactance: float = 1e-9,
) -> PTDF:
    """Build an affine PTDF for the active connected network.

    Inactive branches are excluded. Parallel branches and non-unity transformer
    taps are supported. A nonzero phase shift is represented by the affine
    flow offset. Disconnected networks and near-zero reactances are rejected.
    """

    branches = network.active_branches
    if not branches:
        raise MATPOWERCaseError("The active network has no branches.")
    if minimum_reactance <= 0.0:
        raise ValueError("minimum_reactance must be positive.")

    bus_ids = network.bus_ids
    positions = {bus_id: index for index, bus_id in enumerate(bus_ids)}
    reference_id = network.reference_bus_id if reference_bus_id is None else reference_bus_id
    if reference_id not in positions:
        raise MATPOWERCaseError(f"Reference bus {reference_id} is not in the network.")

    n_bus = len(bus_ids)
    n_branch = len(branches)
    incidence = np.zeros((n_branch, n_bus), dtype=np.float64)
    susceptance_mw = np.zeros(n_branch, dtype=np.float64)
    shift_radians = np.zeros(n_branch, dtype=np.float64)

    for row, branch in enumerate(branches):
        if abs(branch.reactance_pu) < minimum_reactance:
            raise MATPOWERCaseError(
                f"{branch.branch_id} has |x|={abs(branch.reactance_pu):.3g}, below "
                f"the supported threshold {minimum_reactance:.3g}."
            )
        tap = branch.effective_tap_ratio
        if tap <= 0.0:
            raise MATPOWERCaseError(
                f"{branch.branch_id} has nonpositive transformer tap ratio {tap}."
            )
        incidence[row, positions[branch.from_bus]] = 1.0
        incidence[row, positions[branch.to_bus]] = -1.0
        susceptance_mw[row] = network.base_mva / (branch.reactance_pu * tap)
        shift_radians[row] = np.deg2rad(branch.phase_shift_degrees)

    adjacency = sparse.csr_matrix(np.abs(incidence).T @ np.abs(incidence))
    component_count, _ = csgraph.connected_components(adjacency, directed=False)
    if component_count != 1:
        raise MATPOWERCaseError(
            f"The active network has {component_count} disconnected components; "
            "Stage 2 does not repair or combine islands."
        )

    branch_angle_matrix = susceptance_mw[:, None] * incidence
    bus_susceptance = incidence.T @ branch_angle_matrix
    phase_branch_flow = -susceptance_mw * shift_radians
    phase_bus_injection = incidence.T @ phase_branch_flow

    reference = positions[reference_id]
    nonreference = np.arange(n_bus) != reference
    reduced = bus_susceptance[np.ix_(nonreference, nonreference)]
    if np.linalg.matrix_rank(reduced) != n_bus - 1:
        raise MATPOWERCaseError("Reduced DC bus susceptance matrix is singular.")

    matrix = np.zeros((n_branch, n_bus), dtype=np.float64)
    matrix[:, nonreference] = np.linalg.solve(
        reduced.T,
        branch_angle_matrix[:, nonreference].T,
    ).T
    flow_offset = phase_branch_flow - matrix[:, nonreference] @ phase_bus_injection[nonreference]

    return PTDF(
        bus_ids=bus_ids,
        branch_ids=tuple(branch.branch_id for branch in branches),
        reference_bus_id=reference_id,
        matrix=matrix,
        flow_offset_mw=flow_offset,
        bus_susceptance_mw=bus_susceptance,
        branch_angle_matrix_mw=branch_angle_matrix,
        phase_shift_bus_injection_mw=phase_bus_injection,
        phase_shift_branch_flow_mw=phase_branch_flow,
    )
