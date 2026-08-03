"""Device-resident state and one-step implementation of Algorithm 2.

This module deliberately does not alter :mod:`gpu_dcopf_hpr.sgs_hpr`, which
remains the CPU correctness oracle.  CuPy is acquired lazily through the
Stage 6 backend adapter, so importing this module on a CPU-only machine is
safe.

The equality modes are intentionally explicit:

``scaled_direct``
    Factor the supplied (normally fully preconditioned) algorithm LP's dense
    equality Gram matrix on the device and use Cholesky triangular solves.

``unscaled_structural``
    Apply the corrected Stage 4 Equation (55) descriptor on the raw,
    unscaled LP.  A descriptor tied to that exact LP instance is required.

``scaled_structural``
    Apply the Stage 7 generalized block-arrow factors to the exact diagonally
    scaled LP in FP64. The prepared factors are uploaded once and reused.

The equality descriptors are mutually exclusive and tied to exact LP objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import numpy as np
from scipy import sparse

from .canonical_lp import CanonicalLP
from .gpu_backend import CuPyBackend, create_gpu_backend
from .gpu_sparse import ResidentCSR, prepare_resident_csr
from .hpr_generic import HPRState
from .sgs_hpr import ALGORITHM_2_UPDATE_ORDER, estimate_inequality_spectrum
from .stage7_scaled_y1 import (
    DeviceScaledBlockArrowY1Solver,
    ScaledBlockArrowY1Solver,
)
from .structural_y1 import StructuralY1Solver

GPUEqualityMode = Literal["scaled_direct", "unscaled_structural", "scaled_structural"]
GPUPrecision = Literal["float64", "float32"]


@dataclass(frozen=True, slots=True)
class GPUHPRState:
    """One device-resident HPR state in the paper's ``(y, z, x)`` order."""

    y: Any
    z: Any
    x: Any

    def detached_copy(self) -> GPUHPRState:
        """Return an independent device-to-device copy for anchors or checkpoints."""

        return GPUHPRState(y=self.y.copy(), z=self.z.copy(), x=self.x.copy())

    @classmethod
    def from_host(
        cls,
        state: HPRState,
        backend: CuPyBackend,
        *,
        phase: str = "initialization",
        dtype: GPUPrecision = "float64",
    ) -> GPUHPRState:
        """Copy a CPU state to the device through the audited transfer ledger."""

        if dtype not in {"float64", "float32"}:
            raise ValueError("dtype must be 'float64' or 'float32'.")
        host_dtype = np.float64 if dtype == "float64" else np.float32
        return cls(
            y=backend.to_device(np.asarray(state.y, dtype=host_dtype), phase=phase, kind="vector"),
            z=backend.to_device(np.asarray(state.z, dtype=host_dtype), phase=phase, kind="vector"),
            x=backend.to_device(np.asarray(state.x, dtype=host_dtype), phase=phase, kind="vector"),
        )


@dataclass(slots=True)
class _GPUVectorBuffers:
    """Reusable device buffers; returned step views live until the next call."""

    current_y: Any
    current_z: Any
    current_x: Any
    anchor_y: Any
    anchor_z: Any
    anchor_x: Any
    y1_half: Any
    proximal_y: Any
    proximal_z: Any
    proximal_x: Any
    reflected_y: Any
    reflected_z: Any
    reflected_x: Any
    next_y: Any
    next_z: Any
    next_x: Any
    n0: Any
    n1: Any
    n2: Any
    n3: Any
    n4: Any
    equality_n: Any
    m1_rhs: Any
    m1_residual: Any
    m1_triangular: Any
    m2_work: Any


@dataclass(slots=True)
class GPUSGSHPRWorkspace:
    """Resident Algorithm 2 operators, factors, constants, and scratch space."""

    source_lp: CanonicalLP
    backend: CuPyBackend
    dtype_name: GPUPrecision
    dtype: Any
    equality_mode: GPUEqualityMode
    A1_resident: ResidentCSR
    A2_resident: ResidentCSR
    A1: Any
    A1_transpose: Any
    A2: Any
    A2_transpose: Any
    c: Any
    b1: Any
    b2: Any
    lower: Any
    upper: Any
    inequality_lambda: float | None
    equality_gram: Any | None
    equality_cholesky: Any | None
    structural_y1: StructuralY1Solver | None
    scaled_structural_y1: ScaledBlockArrowY1Solver | None
    device_scaled_structural_y1: DeviceScaledBlockArrowY1Solver | None
    structural_coupling: Any | None
    structural_inverse_storage_diagonal: Any | None
    structural_weight: Any | None
    triangular_solve: Any | None
    zero_scalar: Any
    buffers: _GPUVectorBuffers

    @property
    def m1(self) -> int:
        return self.source_lp.m1

    @property
    def m2(self) -> int:
        return self.source_lp.m2

    @property
    def m(self) -> int:
        return self.source_lp.m

    @property
    def n(self) -> int:
        return self.source_lp.n


@dataclass(frozen=True, slots=True)
class GPUSGSHPRStep:
    """One Algorithm 2 step whose arrays and diagnostics remain on device.

    The arrays are views of reusable workspace buffers.  They are stable for
    inspection until the next call using the same workspace.  Inputs are
    copied device-to-device into dedicated buffers at the start of every call,
    so feeding ``next_state`` or a restart ``proximal`` back into the workspace
    is safe.
    """

    y1_half: Any
    proximal: GPUHPRState
    reflected: GPUHPRState
    next_state: GPUHPRState
    first_equality_relative_residual: Any
    second_equality_relative_residual: Any
    first_equality_infinity_residual: Any
    second_equality_infinity_residual: Any
    z_x_identity_error: Any
    update_order: tuple[str, ...] = ALGORITHM_2_UPDATE_ORDER


def _empty_buffers(
    xp: Any,
    *,
    m: int,
    m1: int,
    m2: int,
    n: int,
    dtype: Any,
) -> _GPUVectorBuffers:
    def zeros(length: int) -> Any:
        return xp.zeros(length, dtype=dtype)

    return _GPUVectorBuffers(
        current_y=zeros(m),
        current_z=zeros(n),
        current_x=zeros(n),
        anchor_y=zeros(m),
        anchor_z=zeros(n),
        anchor_x=zeros(n),
        y1_half=zeros(m1),
        proximal_y=zeros(m),
        proximal_z=zeros(n),
        proximal_x=zeros(n),
        reflected_y=zeros(m),
        reflected_z=zeros(n),
        reflected_x=zeros(n),
        next_y=zeros(m),
        next_z=zeros(n),
        next_x=zeros(n),
        n0=zeros(n),
        n1=zeros(n),
        n2=zeros(n),
        n3=zeros(n),
        n4=zeros(n),
        equality_n=zeros(n),
        m1_rhs=zeros(m1),
        m1_residual=zeros(m1),
        m1_triangular=zeros(m1),
        m2_work=zeros(m2),
    )


def _select_triangular_solver(xp: Any) -> Any:
    """Load the array-module-specific triangular solve only during prepare."""

    module_name = getattr(xp, "__name__", "")
    if module_name == "numpy":
        from scipy.linalg import solve_triangular

        return solve_triangular
    if module_name.startswith("cupy"):
        from cupyx.scipy.linalg import solve_triangular

        return solve_triangular
    return None


def _matvec_into(operator: ResidentCSR, vector: Any, out: Any, *, transpose: bool) -> Any:
    """Run one sparse matvec and retain the result in a reusable buffer."""

    result = operator.matvec(vector, transpose=transpose, out=out)
    if result is not out:
        out[...] = result
    return out


def _validate_state(workspace: GPUSGSHPRWorkspace, state: GPUHPRState, *, name: str) -> None:
    expected = (("y", workspace.m), ("z", workspace.n), ("x", workspace.n))
    for block, length in expected:
        value = getattr(state, block)
        shape = getattr(value, "shape", None)
        if shape != (length,):
            raise ValueError(f"{name}.{block} must have shape ({length},); received {shape}.")
        if getattr(value, "dtype", None) != workspace.dtype:
            raise ValueError(
                f"{name}.{block} must have device dtype {workspace.dtype_name}; "
                f"received {getattr(value, 'dtype', None)}."
            )


def prepare_gpu_sgs_hpr(
    lp: CanonicalLP,
    *,
    equality_mode: GPUEqualityMode = "scaled_direct",
    structural_y1: StructuralY1Solver | None = None,
    scaled_structural_y1: ScaledBlockArrowY1Solver | None = None,
    inequality_lambda: float | None = None,
    backend: CuPyBackend | None = None,
    dtype: GPUPrecision = "float64",
) -> GPUSGSHPRWorkspace:
    """Upload an LP and prepare exactly one GPU equality backend.

    ``scaled_direct`` treats ``lp`` as the Stage 5 algorithm LP after its full
    diagonal preconditioning and performs dense Cholesky solves on that scaled
    system. ``unscaled_structural`` requires the corrected raw Equation (55)
    descriptor. ``scaled_structural`` requires Stage 7 block-arrow factors
    tied to this exact scaled LP and is intentionally FP64-only.
    """

    allowed_modes = {"scaled_direct", "unscaled_structural", "scaled_structural"}
    if equality_mode not in allowed_modes:
        raise ValueError(
            "equality_mode must be 'scaled_direct', 'unscaled_structural', or 'scaled_structural'."
        )
    if dtype not in {"float64", "float32"}:
        raise ValueError("dtype must be 'float64' or 'float32'.")
    if structural_y1 is not None and scaled_structural_y1 is not None:
        raise ValueError("structural_y1 and scaled_structural_y1 are mutually exclusive.")
    if equality_mode == "scaled_direct":
        if structural_y1 is not None:
            raise ValueError("scaled_direct cannot use the raw StructuralY1Solver.")
        if scaled_structural_y1 is not None:
            raise ValueError("scaled_direct cannot use a ScaledBlockArrowY1Solver.")
    if equality_mode == "unscaled_structural":
        if scaled_structural_y1 is not None:
            raise ValueError("unscaled_structural cannot use a ScaledBlockArrowY1Solver.")
        if structural_y1 is None:
            raise ValueError("unscaled_structural requires a corrected StructuralY1Solver.")
        if structural_y1.source_lp is not lp:
            raise ValueError(
                "the unscaled structural descriptor must be prepared from the same "
                "CanonicalLP instance."
            )
    if equality_mode == "scaled_structural":
        if structural_y1 is not None:
            raise ValueError("scaled_structural cannot use the raw StructuralY1Solver.")
        if scaled_structural_y1 is None:
            raise ValueError("scaled_structural requires a ScaledBlockArrowY1Solver.")
        if dtype != "float64":
            raise ValueError("scaled_structural requires FP64 (dtype='float64').")
        if scaled_structural_y1.source_lp is not lp:
            raise ValueError(
                "the scaled structural descriptor must be prepared from the same exact "
                "scaled CanonicalLP instance."
            )

    if lp.m2:
        if inequality_lambda is None:
            inequality_lambda = estimate_inequality_spectrum(lp.A2).lambda_used
        if not isfinite(inequality_lambda) or inequality_lambda <= 0.0:
            raise ValueError("inequality_lambda must be a positive finite scalar.")
        inequality_lambda = float(inequality_lambda)
    elif inequality_lambda is not None:
        raise ValueError("inequality_lambda must be None when A2 has no rows.")

    selected_backend = create_gpu_backend() if backend is None else backend
    xp = selected_backend.xp
    device_dtype = xp.float64 if dtype == "float64" else xp.float32
    host_dtype = np.float64 if dtype == "float64" else np.float32
    A1_host = sparse.csr_matrix(lp.A1, dtype=host_dtype, copy=True)
    A2_host = sparse.csr_matrix(lp.A2, dtype=host_dtype, copy=True)
    A1_resident = prepare_resident_csr(
        selected_backend,
        A1_host,
        phase="preparation",
        prefer_csr_alg2=dtype == "float64",
        dtype=device_dtype,
    )
    A2_resident = prepare_resident_csr(
        selected_backend,
        A2_host,
        phase="preparation",
        prefer_csr_alg2=dtype == "float64",
        dtype=device_dtype,
    )

    def upload(values: Any) -> Any:
        host_values = np.asarray(values, dtype=host_dtype)
        transfer_kind = "matrix" if host_values.ndim == 2 else "vector"
        return selected_backend.to_device(
            host_values,
            phase="preparation",
            kind=transfer_kind,
        )

    c = upload(lp.c)
    b1 = upload(lp.b1)
    b2 = upload(lp.b2)
    lower = upload(lp.lower)
    upper = upload(lp.upper)

    equality_gram = None
    equality_cholesky = None
    triangular_solve = None
    structural_coupling = None
    structural_inverse = None
    structural_weight = None
    device_scaled_structural_y1 = None
    if equality_mode == "scaled_direct":
        if lp.m1:
            host_gram = np.asarray((A1_host @ A1_host.T).toarray(), dtype=host_dtype)
            host_gram = 0.5 * (host_gram + host_gram.T)
            equality_gram = upload(host_gram)
            factorization_error = getattr(xp.linalg, "LinAlgError", np.linalg.LinAlgError)
            try:
                equality_cholesky = xp.linalg.cholesky(equality_gram)
            except factorization_error as error:
                raise ValueError(
                    "the scaled direct equality Gram must be positive definite."
                ) from error
            triangular_solve = _select_triangular_solver(xp)
        else:
            equality_gram = xp.empty((0, 0), dtype=device_dtype)
            equality_cholesky = xp.empty((0, 0), dtype=device_dtype)
    elif equality_mode == "unscaled_structural":
        assert structural_y1 is not None
        structural_coupling = upload(structural_y1.diagnostics.coupling)
        structural_inverse = upload(structural_y1.inverse_storage_diagonal)
        structural_weight = structural_coupling * structural_inverse
    else:
        assert scaled_structural_y1 is not None
        device_scaled_structural_y1 = scaled_structural_y1.to_device(
            selected_backend,
            phase="preparation",
            triangular_solve=_select_triangular_solver(xp),
        )

    zero_scalar = xp.asarray(0.0, dtype=device_dtype)
    return GPUSGSHPRWorkspace(
        source_lp=lp,
        backend=selected_backend,
        dtype_name=dtype,
        dtype=device_dtype,
        equality_mode=equality_mode,
        A1_resident=A1_resident,
        A2_resident=A2_resident,
        A1=A1_resident.matrix,
        A1_transpose=A1_resident.transpose,
        A2=A2_resident.matrix,
        A2_transpose=A2_resident.transpose,
        c=c,
        b1=b1,
        b2=b2,
        lower=lower,
        upper=upper,
        inequality_lambda=inequality_lambda,
        equality_gram=equality_gram,
        equality_cholesky=equality_cholesky,
        structural_y1=structural_y1,
        scaled_structural_y1=scaled_structural_y1,
        device_scaled_structural_y1=device_scaled_structural_y1,
        structural_coupling=structural_coupling,
        structural_inverse_storage_diagonal=structural_inverse,
        structural_weight=structural_weight,
        triangular_solve=triangular_solve,
        zero_scalar=zero_scalar,
        buffers=_empty_buffers(
            xp,
            m=lp.m,
            m1=lp.m1,
            m2=lp.m2,
            n=lp.n,
            dtype=device_dtype,
        ),
    )


def _solve_scaled_direct(
    workspace: GPUSGSHPRWorkspace,
    rhs: Any,
    out: Any,
) -> tuple[Any, Any]:
    xp = workspace.backend.xp
    factor = workspace.equality_cholesky
    assert factor is not None
    triangular_solve = workspace.triangular_solve
    if triangular_solve is None:
        # This fallback is used only by non-CuPy test adapters.  The production
        # CuPy path above always selects cupyx.scipy.linalg.solve_triangular.
        first = xp.linalg.solve(factor, rhs)
        out[...] = xp.linalg.solve(factor.T, first)
    else:
        temporary = workspace.buffers.m1_triangular
        temporary[...] = rhs
        first = triangular_solve(
            factor,
            temporary,
            lower=True,
            check_finite=False,
            overwrite_b=True,
        )
        if first is not temporary:
            temporary[...] = first
        second = triangular_solve(
            factor.T,
            temporary,
            lower=False,
            check_finite=False,
            overwrite_b=True,
        )
        if second is not out:
            out[...] = second
    assert workspace.equality_gram is not None
    residual = workspace.buffers.m1_residual
    residual[...] = workspace.equality_gram @ out
    residual -= rhs
    relative = xp.linalg.norm(residual) / (1.0 + xp.linalg.norm(rhs))
    infinity = xp.linalg.norm(residual, ord=xp.inf)
    return relative, infinity


def _solve_unscaled_structural(
    workspace: GPUSGSHPRWorkspace,
    rhs: Any,
    out: Any,
) -> tuple[Any, Any]:
    xp = workspace.backend.xp
    descriptor = workspace.structural_y1
    assert descriptor is not None
    diagnostics = descriptor.diagnostics
    periods = diagnostics.periods
    if diagnostics.storage_count == 0:
        xp.divide(rhs, diagnostics.balance_diagonal, out=out)
    else:
        coupling = workspace.structural_coupling
        inverse = workspace.structural_inverse_storage_diagonal
        weight = workspace.structural_weight
        assert coupling is not None
        assert inverse is not None
        assert weight is not None
        balance_solution = out[:periods]
        storage_solution = out[periods:]
        weighted_storage_rhs = xp.dot(weight, rhs[periods:])
        xp.subtract(rhs[:periods], weighted_storage_rhs, out=balance_solution)
        reduced_mean = xp.mean(balance_solution)
        balance_solution -= reduced_mean
        balance_solution /= diagnostics.balance_diagonal
        balance_solution += reduced_mean / diagnostics.schur_scalar
        balance_sum = xp.sum(balance_solution)
        xp.multiply(coupling, balance_sum, out=storage_solution)
        xp.subtract(rhs[periods:], storage_solution, out=storage_solution)
        storage_solution *= inverse

    return _sparse_equality_residual(workspace, rhs, out)


def _sparse_equality_residual(
    workspace: GPUSGSHPRWorkspace,
    rhs: Any,
    out: Any,
) -> tuple[Any, Any]:
    """Evaluate ``A1 A1.T out - rhs`` through the resident sparse operator."""

    xp = workspace.backend.xp
    residual = workspace.buffers.m1_residual
    _matvec_into(
        workspace.A1_resident,
        out,
        workspace.buffers.equality_n,
        transpose=True,
    )
    _matvec_into(
        workspace.A1_resident,
        workspace.buffers.equality_n,
        residual,
        transpose=False,
    )
    residual -= rhs
    relative = xp.linalg.norm(residual) / (1.0 + xp.linalg.norm(rhs))
    infinity = xp.linalg.norm(residual, ord=xp.inf)
    return relative, infinity


def _solve_scaled_structural(
    workspace: GPUSGSHPRWorkspace,
    rhs: Any,
    out: Any,
) -> tuple[Any, Any]:
    solver = workspace.device_scaled_structural_y1
    assert solver is not None
    solver.solve_into(rhs, out)
    return _sparse_equality_residual(workspace, rhs, out)


def _solve_equality(
    workspace: GPUSGSHPRWorkspace,
    rhs: Any,
    out: Any,
) -> tuple[Any, Any]:
    if workspace.m1 == 0:
        return workspace.zero_scalar, workspace.zero_scalar
    if workspace.equality_mode == "scaled_direct":
        return _solve_scaled_direct(workspace, rhs, out)
    if workspace.equality_mode == "unscaled_structural":
        return _solve_unscaled_structural(workspace, rhs, out)
    return _solve_scaled_structural(workspace, rhs, out)


def gpu_sgs_hpr_step(
    lp: CanonicalLP,
    current: GPUHPRState,
    anchor: GPUHPRState,
    workspace: GPUSGSHPRWorkspace,
    *,
    iteration: int,
    sigma: float,
) -> GPUSGSHPRStep:
    """Perform one FP64 Algorithm 2 iteration without a device-to-host copy."""

    if workspace.source_lp is not lp:
        raise ValueError("workspace must be prepared from the same CanonicalLP instance.")
    _validate_state(workspace, current, name="current")
    _validate_state(workspace, anchor, name="anchor")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer.")
    if not isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite scalar.")

    xp = workspace.backend.xp
    buffers = workspace.buffers

    # Snapshot inputs with device-to-device assignments.  This makes feedback
    # safe even though output arrays are reusable workspace views.
    buffers.current_y[...] = current.y
    buffers.current_z[...] = current.z
    buffers.current_x[...] = current.x
    buffers.anchor_y[...] = anchor.y
    buffers.anchor_z[...] = anchor.z
    buffers.anchor_x[...] = anchor.x
    current_y = buffers.current_y
    current_z = buffers.current_z
    current_x = buffers.current_x
    y1_current = current_y[: workspace.m1]
    y2_current = current_y[workspace.m1 :]

    # A.T y, with A1/A2 and both transposes permanently resident as CSR.
    if workspace.m1:
        _matvec_into(workspace.A1_resident, y1_current, buffers.n0, transpose=True)
    else:
        buffers.n0.fill(0.0)
    if workspace.m2:
        _matvec_into(workspace.A2_resident, y2_current, buffers.n1, transpose=True)
        xp.add(buffers.n0, buffers.n1, out=buffers.n0)

    # z_bar and x_bar.
    xp.subtract(buffers.n0, workspace.c, out=buffers.n2)
    buffers.n2 *= sigma
    buffers.n2 += current_x
    xp.maximum(buffers.n2, workspace.lower, out=buffers.n3)
    xp.minimum(buffers.n3, workspace.upper, out=buffers.n3)
    xp.subtract(buffers.n3, buffers.n2, out=buffers.proximal_z)
    buffers.proximal_z /= sigma
    xp.add(buffers.n0, buffers.proximal_z, out=buffers.proximal_x)
    buffers.proximal_x -= workspace.c
    buffers.proximal_x *= sigma
    buffers.proximal_x += current_x
    xp.subtract(buffers.proximal_x, buffers.n3, out=buffers.n4)
    z_x_identity_error = xp.linalg.norm(buffers.n4, ord=xp.inf)

    # First equality sweep.
    if workspace.m2:
        _matvec_into(workspace.A2_resident, y2_current, buffers.n1, transpose=True)
    else:
        buffers.n1.fill(0.0)
    xp.add(buffers.n1, buffers.proximal_z, out=buffers.n4)
    buffers.n4 -= workspace.c
    buffers.n4 *= sigma
    buffers.n4 += buffers.proximal_x
    if workspace.m1:
        _matvec_into(
            workspace.A1_resident,
            buffers.n4,
            buffers.m1_rhs,
            transpose=False,
        )
        xp.subtract(workspace.b1, buffers.m1_rhs, out=buffers.m1_rhs)
        buffers.m1_rhs /= sigma
    first_relative, first_infinity = _solve_equality(
        workspace,
        buffers.m1_rhs,
        buffers.y1_half,
    )

    # Inequality projection.
    if workspace.m2:
        assert workspace.inequality_lambda is not None
        _matvec_into(
            workspace.A1_resident,
            buffers.y1_half,
            buffers.n0,
            transpose=True,
        )
        _matvec_into(workspace.A2_resident, y2_current, buffers.n1, transpose=True)
        xp.add(buffers.n0, buffers.n1, out=buffers.n4)
        buffers.n4 += buffers.proximal_z
        buffers.n4 -= workspace.c
        xp.divide(buffers.proximal_x, sigma, out=buffers.n0)
        buffers.n4 += buffers.n0
        _matvec_into(
            workspace.A2_resident,
            buffers.n4,
            buffers.m2_work,
            transpose=False,
        )
        xp.divide(workspace.b2, sigma, out=buffers.proximal_y[workspace.m1 :])
        buffers.proximal_y[workspace.m1 :] -= buffers.m2_work
        buffers.proximal_y[workspace.m1 :] /= workspace.inequality_lambda
        buffers.proximal_y[workspace.m1 :] += y2_current
        xp.maximum(
            buffers.proximal_y[workspace.m1 :],
            0.0,
            out=buffers.proximal_y[workspace.m1 :],
        )

    # Second equality sweep.
    y2_bar = buffers.proximal_y[workspace.m1 :]
    if workspace.m2:
        _matvec_into(workspace.A2_resident, y2_bar, buffers.n1, transpose=True)
    else:
        buffers.n1.fill(0.0)
    xp.add(buffers.n1, buffers.proximal_z, out=buffers.n4)
    buffers.n4 -= workspace.c
    buffers.n4 *= sigma
    buffers.n4 += buffers.proximal_x
    if workspace.m1:
        _matvec_into(
            workspace.A1_resident,
            buffers.n4,
            buffers.m1_rhs,
            transpose=False,
        )
        xp.subtract(workspace.b1, buffers.m1_rhs, out=buffers.m1_rhs)
        buffers.m1_rhs /= sigma
    second_relative, second_infinity = _solve_equality(
        workspace,
        buffers.m1_rhs,
        buffers.proximal_y[: workspace.m1],
    )

    # Reflection followed by the fixed-anchor Halpern weights.
    for proximal, old, reflected in (
        (buffers.proximal_y, current_y, buffers.reflected_y),
        (buffers.proximal_z, current_z, buffers.reflected_z),
        (buffers.proximal_x, current_x, buffers.reflected_x),
    ):
        xp.multiply(proximal, 2.0, out=reflected)
        reflected -= old
    anchor_weight = 1.0 / (iteration + 2.0)
    reflected_weight = (iteration + 1.0) / (iteration + 2.0)
    for initial, reflected, following, scratch in (
        (buffers.anchor_y, buffers.reflected_y, buffers.next_y, buffers.current_y),
        (buffers.anchor_z, buffers.reflected_z, buffers.next_z, buffers.current_z),
        (buffers.anchor_x, buffers.reflected_x, buffers.next_x, buffers.current_x),
    ):
        xp.multiply(initial, anchor_weight, out=scratch)
        xp.multiply(reflected, reflected_weight, out=following)
        following += scratch

    return GPUSGSHPRStep(
        y1_half=buffers.y1_half,
        proximal=GPUHPRState(
            y=buffers.proximal_y,
            z=buffers.proximal_z,
            x=buffers.proximal_x,
        ),
        reflected=GPUHPRState(
            y=buffers.reflected_y,
            z=buffers.reflected_z,
            x=buffers.reflected_x,
        ),
        next_state=GPUHPRState(y=buffers.next_y, z=buffers.next_z, x=buffers.next_x),
        first_equality_relative_residual=first_relative,
        second_equality_relative_residual=second_relative,
        first_equality_infinity_residual=first_infinity,
        second_equality_infinity_residual=second_infinity,
        z_x_identity_error=z_x_identity_error,
    )


def gpu_sgs_metric_y_quadratic(
    workspace: GPUSGSHPRWorkspace,
    delta_y: Any,
) -> Any:
    """Evaluate the exact Stage 5 sGS multiplier metric on the device."""

    if getattr(delta_y, "shape", None) != (workspace.m,):
        raise ValueError(
            "delta_y must match the workspace multiplier dimension; "
            f"received {getattr(delta_y, 'shape', None)}."
        )
    if getattr(delta_y, "dtype", None) != workspace.dtype:
        raise ValueError(f"delta_y must have device dtype {workspace.dtype_name}.")
    xp = workspace.backend.xp
    buffers = workspace.buffers
    dy1 = delta_y[: workspace.m1]
    dy2 = delta_y[workspace.m1 :]
    if workspace.m1:
        _matvec_into(workspace.A1_resident, dy1, buffers.n0, transpose=True)
    else:
        buffers.n0.fill(0.0)
    if workspace.m2:
        _matvec_into(workspace.A2_resident, dy2, buffers.n1, transpose=True)
        buffers.n0 += buffers.n1

    equality_term = workspace.zero_scalar
    if workspace.m1:
        _matvec_into(
            workspace.A1_resident,
            buffers.n0,
            buffers.m1_rhs,
            transpose=False,
        )
        _solve_equality(workspace, buffers.m1_rhs, buffers.m1_triangular)
        equality_term = xp.dot(buffers.m1_rhs, buffers.m1_triangular)

    inequality_term = workspace.zero_scalar
    if workspace.m2:
        assert workspace.inequality_lambda is not None
        inequality_term = workspace.inequality_lambda * xp.dot(dy2, dy2)
    return xp.maximum(equality_term + inequality_term, workspace.zero_scalar)


def gpu_sgs_restart_merit(
    workspace: GPUSGSHPRWorkspace,
    *,
    delta_x: Any,
    delta_y: Any,
    sigma: float,
) -> Any:
    """Evaluate the HPR-LP restart merit and retain the scalar on device."""

    if not isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite scalar.")
    if getattr(delta_x, "shape", None) != (workspace.n,):
        raise ValueError(
            "delta_x must match the workspace primal dimension; "
            f"received {getattr(delta_x, 'shape', None)}."
        )
    if getattr(delta_x, "dtype", None) != workspace.dtype:
        raise ValueError(f"delta_x must have device dtype {workspace.dtype_name}.")

    xp = workspace.backend.xp
    buffers = workspace.buffers
    h_quadratic = gpu_sgs_metric_y_quadratic(workspace, delta_y)
    cross = workspace.zero_scalar
    if workspace.m1:
        _matvec_into(
            workspace.A1_resident,
            delta_x,
            buffers.m1_rhs,
            transpose=False,
        )
        cross = cross + xp.dot(buffers.m1_rhs, delta_y[: workspace.m1])
    if workspace.m2:
        _matvec_into(
            workspace.A2_resident,
            delta_x,
            buffers.m2_work,
            transpose=False,
        )
        cross = cross + xp.dot(buffers.m2_work, delta_y[workspace.m1 :])
    squared = xp.dot(delta_x, delta_x) / sigma + 2.0 * cross + sigma * h_quadratic
    return xp.sqrt(xp.maximum(squared, workspace.zero_scalar))
