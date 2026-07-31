# Stage 6 report

> **Status:** PASS
>
> **Date:** July 31, 2026
>
> **Stage:** GPU port for NVIDIA DGX Spark

Stage 6 successfully ported the validated Stage 5 FP64 solver to the DGX
Spark. The GPU implementation matched the CPU oracle at the locked short
horizons, passed both full-policy correctness cases, preserved the restart and
adaptive-sigma schedules, and kept the iterative state resident on the GPU.

This is a correctness result, not a speedup result. Stage 7 remains the first
stage authorized to run the benchmark campaign.

## 1. What was tested

The GPU port implements the paper's Algorithm 2 with the Stage 5
preconditioner and control policy. The frozen run used:

- an all-zero initial state `(y, z, x)`;
- initial sigma equal to `1`;
- FP64 for every gating calculation;
- explicit resident CSR matrices for `A1`, `A2`, and their transposes;
- the scaled direct equality solve for the full correctness runs;
- the unscaled structural equality solve as an independently checked path;
- residual checks every iteration for correctness runs; and
- policy checks every 100 iterations.

The two acceptance cases were the public case5 instance at one time period
and the frozen synthetic extension at two time periods. The second case adds a
renewable unit and a storage unit; it is a structural extension, not a
paper-scale benchmark.

| Case | Variables | Constraints | Equality rows | Inequality rows | Nonzeros in A1 / A2 |
|---|---:|---:|---:|---:|---:|
| Public case5, T=1 | 15 | 17 | 1 | 16 | 5 / 46 |
| Synthetic extension, T=2 | 36 | 49 | 3 | 46 | 20 / 148 |

## 2. Recorded DGX environment

The run executed in this environment:

| Item | Recorded value |
|---|---|
| Host platform | Linux 6.17.0-1029-nvidia, aarch64, glibc 2.39 |
| Python | CPython 3.12.3 |
| Python executable | `/home/dgxsparktd/sGS-HPR-DC-SCOPF/.venv/bin/python` |
| GPU | NVIDIA GB10, device 0 |
| Compute capability | 12.1 |
| Multiprocessors | 48 |
| Integrated GPU | Yes |
| Total global memory | 130,663,165,952 bytes |
| CuPy | 14.1.1 |
| CuPy package | `cupy-cuda13x==14.1.1` |
| NumPy / SciPy | 2.3.5 / 1.16.3 |
| NVIDIA driver | 580.173.02 |
| CUDA driver API | 13.0, reported as `13000` |
| CuPy CUDA runtime API | 13.2, reported as `13020` |
| CUDA toolkit / nvcc | 13.0.3 / 13.0.88 |
| CSR index type | signed 32-bit integer for indices and row pointers |
| Required precision | FP64, 8 bytes per value |

All recorded device checks passed. At runner start, 120,396,972,032 device
bytes were free and the CUDA runtime reported 10,266,193,920 bytes in use.

The cuBLAS, cuSPARSE, and cuSOLVER version helper calls were unavailable
through the installed Python binding: the first two require a library handle,
and this cuSOLVER binding does not expose `getVersion`. This is a metadata
limitation, not a numerical fallback. The actual FP64 sparse-kernel path was
verified separately as described below.

The evidence was produced from branch `main` at commit
`07a4b91dc53347c51622148c45bb047347144164`. The tree contained the Stage 6
implementation changes at execution time. The compressed trajectory and
policy-event evidence has SHA-256 digest
`eb5f1fafe2fb7496726893a97457bee155fb503a8a71e0e90872ad484b0f9c3d`.

## 3. Residual and acceptance definitions

For a candidate `(y, z, x)`, the report uses three separately normalized
Equation (54) blocks:

```text
primal residual     = projection onto D of (b - A x)
box residual        = x - projection onto the box of (x - z)
stationarity        = c - transpose(A) y - z

normalized primal   = norm(primal residual) / (1 + norm(b))
normalized box      = norm(box residual) / (1 + norm(x) + norm(z))
normalized stationarity = norm(stationarity) / (1 + norm(c))
```

Each normalized block had to be at most `5e-5`. The additional raw KKT
measure had to be at most `0.01`; physical violations had to be at most
`0.01 MW/MWh`; and the scaled objective gap to HiGHS had to be at most
`2e-4`.

Equality-solve infinity residuals were limited to `5e-10`, and the projected
`z`/`x` identity error was limited to `1e-10`. No tolerance was changed after
the results were observed.

## 4. Sparse kernel fidelity

Every FP64 `A1` and `A2` operator, including the operators used by the
structural checks, recorded the following path:

| Field | Recorded value |
|---|---|
| Requested kernel | `cuSPARSE CUSPARSE_SPMV_CSR_ALG2` |
| Effective kernel | `cuSPARSE CUSPARSE_SPMV_CSR_ALG2 (enum 3; pinned CuPy 14.1.1 low-level binding)` |
| Uses CSR ALG2 | Yes |
| Fallback reason | None |
| Kernel probe maximum absolute error | 0 |
| Repeated probe bitwise equal | Yes |

Thus the gating FP64 solver did not use the generic high-level fallback. The
probe also confirmed the exact low-level path before the solver evidence was
accepted.

Normal and transpose products were compared with the CPU oracle. The largest
error over `A1`, `A2`, and both transposes was:

| Case | Maximum absolute error | Maximum relative error | Tolerance | Result |
|---|---:|---:|---:|---|
| Public case5, T=1 | 2.22045e-16 | 1.09033e-16 | 5e-13 | PASS |
| Synthetic extension, T=2 | 2.22045e-16 | 1.09329e-16 | 5e-13 | PASS |

The preregistered sparse diagnostic used 50 warm-up calls and 2,000 timed
repetitions. The transpose flag and the explicit CSR transpose agreed to zero
for T=1 and to `1.11022e-16` for T=2.

## 5. Fixed-horizon FP64 CPU/GPU parity

The GPU and CPU solvers started from identical states and ran for exactly 1,
10, and 100 iterations. The table reports the maximum error over `x`, `y`, and
`z` after recovery to the original LP coordinates.

| Case | Steps | Maximum relative error | Maximum absolute error | Locked tolerance | Result |
|---|---:|---:|---:|---|---|
| Public case5, T=1 | 1 | 0 | 0 | rel 2e-12, abs 2e-12 | PASS |
| Public case5, T=1 | 10 | 1.18907e-15 | 1.42109e-13 | rel 2e-10, abs 2e-11 | PASS |
| Public case5, T=1 | 100 | 1.13876e-14 | 5.68434e-13 | rel 1e-10 | PASS |
| Synthetic extension, T=2 | 1 | 1.01680e-17 | 4.44089e-16 | rel 2e-12, abs 2e-12 | PASS |
| Synthetic extension, T=2 | 10 | 5.01053e-16 | 6.06182e-14 | rel 2e-10, abs 2e-11 | PASS |
| Synthetic extension, T=2 | 100 | 2.29361e-15 | 1.27898e-12 | rel 1e-10 | PASS |

The scaled-coordinate comparisons also passed. Their worst maximum relative
errors at 100 steps were `1.01270e-15` for T=1 and `1.98057e-15` for T=2.
Every fixed-horizon transfer audit passed.

## 6. Equality paths and structural safeguards

Two equality modes were exercised:

| Mode | Coordinates | Role | Result |
|---|---|---|---|
| `scaled_direct` | Stage 5-scaled LP | Gating full solve | PASS |
| `unscaled_structural` | Original LP | Independent structural parity check | PASS |

The structural state, proximal point, reflected point, and first equality
sweep matched exactly in both cases. The largest structural diagnostic error
was zero for T=1 and `1.13687e-13` absolute (`1.15685e-13` relative) for T=2,
well inside the locked `2e-12` tolerances.

The runner also proved that the two modes cannot be accidentally mixed. Both
cases produced the expected `ValueError` for each invalid pairing:

- A raw structural solver attached to `scaled_direct` was rejected with:
  `scaled_direct cannot use the raw StructuralY1Solver; a scaled structural
  formula has not been derived.`
- A raw descriptor attached to a scaled LP was rejected with:
  `the unscaled structural descriptor must be prepared from the same
  CanonicalLP instance.`

This matters because diagonal scaling changes the Gram structure used by the
paper's Equation (55). Stage 6 does not claim a scaled structural formula.

## 7. Determinism and device residency

Each case was repeated three times for exactly 100 FP64 iterations. Repeats 2
and 3 had zero absolute and relative state error against repeat 1, identical
policy schedules, and bitwise-equal `x`, `y`, and `z`. Bitwise equality was
observed even though the gate required only relative error at most `1e-12`.

The transfer ledger allowed only named preparation, diagnostic, policy, and
finalization transfers. Every audit reported an empty unexpected-transfer
list. In particular, no full state was copied to the host inside the resident
iteration loop.

The 1,000-step resident timing diagnostic checked residuals every 100
iterations:

| Case | Loop GPU time | Residual-check time | Loop excluding checks | Device to host | Host to device | Audit |
|---|---:|---:|---:|---:|---:|---|
| Public case5, T=1 | 0.922267 s | 0.004024 s | 0.918243 s | 1,896 B / 37 calls | 376 B / 3 calls | PASS |
| Synthetic extension, T=2 | 0.906732 s | 0.004003 s | 0.902729 s | 3,080 B / 37 calls | 968 B / 3 calls | PASS |

The full correctness runs also passed their transfer audits. They recorded
33,704 device-to-host bytes in 425 calls for T=1 and 84,840 bytes in 1,059
calls for T=2. Those larger call counts are expected because correctness
residuals were checked every iteration. Each run performed only three initial
host-to-device state transfers: 376 bytes for T=1 and 968 bytes for T=2.

Across every Stage 6 phase, including cross-checks and diagnostics, the final
ledger recorded 229,760 device-to-host bytes in 2,760 calls and 30,884
host-to-device bytes in 169 calls.

## 8. Full-policy FP64 acceptance

Both GPU runs converged with the same iteration count and the same restart and
adaptive-sigma schedule as the CPU oracle.

| Metric | Public case5, T=1 | Synthetic extension, T=2 | Gate |
|---|---:|---:|---:|
| GPU iterations | 410 | 1,032 | at most 150,000 |
| GPU objective | 17,479.839088899 | 26,580.274984099 | cross-checked below |
| HiGHS objective | 17,479.896925381 | 26,580.003335526 | reference |
| Scaled objective gap | 3.30874e-6 | 1.02200e-5 | at most 2e-4 |
| Raw KKT | 0.00561846 | 0.00894842 | at most 0.01 |
| Normalized box residual | 1.01551e-16 | 1.87201e-16 | at most 5e-5 |
| Normalized primal residual | 3.69029e-6 | 2.90138e-6 | at most 5e-5 |
| Normalized stationarity residual | 1.15718e-5 | 8.95773e-6 | at most 5e-5 |
| Maximum physical violation | 0.00423091 | 0.00605917 | at most 0.01 |
| Equality-solve infinity residual | 1.11022e-16 | 1.66533e-16 | at most 5e-10 |
| Projected z/x identity error | 3.20575e-15 | 3.17801e-15 | at most 1e-10 |
| Restarts / policy events | 4 / 4 | 8 / 10 | match CPU schedule |
| Result | **PASS** | **PASS** | all gates |

The final CPU/GPU objective difference was `1.81899e-11` in both cases. The
relative differences were `1.04062e-15` for T=1 and `6.84338e-16` for T=2,
against the locked `1e-8` tolerance. The largest difference among the three
normalized residual blocks was `5.53461e-15` for T=1 and `2.34488e-16` for
T=2, against the locked `1e-8` absolute tolerance.

## 9. Timing disclosure

All explicit transfers and CUDA-event intervals synchronized before their
measurements ended. Host phases used a monotonic host clock. Residual checks
are nested within the iteration-loop time and must not be added to it.

| Boundary | Seconds | Method |
|---|---:|---|
| CUDA initialization | 0.404647 | Host clock around backend/device initialization, ending with device synchronization |
| CPU construction and preprocessing | 0.008021 | Host clock around DCOPF construction and Stage 5 preprocessing |
| First-run compilation and warm-up | 0.010213 | Host clock around the first completed one-step GPU run, with synchronized return |
| Allocation | included below | Not independently timed; included in synchronized GPU solver initialization |
| Host-to-device transfer | 0.004171 | Host clock around synchronized explicit transfers |
| GPU solver initialization | 0.074478 | Host clock around resident preparation, including allocation, ending with synchronization |
| Resident iteration loops | 1.828999 | CUDA events; sum of both 1,000-step diagnostics |
| Residual checks | 0.008027 | CUDA events; nested checks summed across both resident diagnostics |
| Device-to-host transfer | 0.024864 | Host clock around synchronized explicit transfers |
| Complete Stage 6 wall time | 6.716848 | Monotonic host clock around the runner |

For reference, the full-policy GPU loops took `0.567655 s` for T=1 and
`1.427785 s` for T=2, including per-iteration residual checks. The nested
residual work accounted for `0.163988 s` and `0.409864 s`, respectively.
These measurements describe boundaries only. They are not combined with CPU
end-to-end times and do not support a speedup claim.

## 10. Optional FP32 diagnostic

The FP32 study ran only after every global FP64 gate passed. It was
preregistered as non-gating, and mixed precision remained disabled.

Both cases completed 100 iterations with finite FP32 device state and clean
residency audits. Unlike the gating FP64 path, FP32 deliberately disabled
CSR ALG2 and used `cupyx.cusparse.spmv CUSPARSE_MV_ALG_DEFAULT`; the recorded
fallback reason was `CSR_ALG2 was disabled by the caller.`

| Case | Maximum relative error versus CPU FP64 | Maximum absolute error | Frozen FP64 100-step parity gate | FP32 diagnostic |
|---|---:|---:|---:|---|
| Public case5, T=1 | 2.04226e-5 | 2.25094e-4 | 1e-10 relative | Does not meet FP64 parity |
| Synthetic extension, T=2 | 9.28169e-7 | 4.02311e-4 | 1e-10 relative | Does not meet FP64 parity |

This is an important negative result: FP32 ran successfully, but neither case
reproduced the frozen FP64 100-step trajectory closely enough to satisfy the
FP64 correctness threshold. Because the study was explicitly non-gating,
this does not change the Stage 6 PASS. It does mean FP32 must not be presented
as an equivalent replacement without a separate accuracy study.

The observed FP32 loop times were `0.120669 s` for T=1 and `0.247479 s` for
T=2. No comparison with the FP64 timings is labeled a speedup: the precision,
kernel path, and study purpose differ.

## 11. What we learned

1. **The mathematical port is faithful in FP64.** Short-horizon state parity,
   full stopping diagnostics, objective values, and control schedules all
   agree with the independent CPU implementation by margins far tighter than
   the locked gates.
2. **Kernel selection must be proven, not inferred.** The high-level CuPy API
   does not by itself establish CSR ALG2. Pinning the low-level enum and
   running an exact probe made the FP64 kernel claim auditable.
3. **Scaling and structural algebra are different coordinate systems.** The
   explicit rejection tests prevent a plausible but mathematically invalid
   combination from silently producing results.
4. **Residency is about transfer intent, not merely small byte counts.** The
   phase-labeled ledger shows why each transfer occurred and proves that no
   unplanned full-state transfer entered the iteration loop.
5. **Reduced precision needs its own acceptance standard.** A finite FP32 run
   is not enough to establish equivalence with the validated FP64 trajectory.
6. **Timing boundaries are now reproducible.** Initialization, warm-up,
   allocation inclusion, transfers, loop work, diagnostics, and end-to-end
   time are disclosed separately, which prepares Stage 7 for fair benchmark
   comparisons.

## 12. Acceptance checklist

- [x] DGX software and hardware facts are captured in machine-readable evidence.
- [x] The CPU package retains the optional, lazy CuPy boundary.
- [x] The actual FP64 SpMV path is recorded and independently probed.
- [x] CSR and transpose-CSR products match the CPU oracle.
- [x] One-step, ten-step, and 100-step FP64 state comparisons pass.
- [x] Both compatible equality paths match their CPU oracles.
- [x] Incompatible scaled/structural pairings are rejected.
- [x] Transfer audits show no unplanned full-state loop transfers.
- [x] Three FP64 repetitions pass the determinism gate.
- [x] Public T=1 and synthetic T=2 full-policy runs pass every original-space gate.
- [x] Equality-solve and projected-state identity thresholds pass.
- [x] All ten timing boundaries are reported with their synchronization rules.
- [x] FP32 is labeled non-gating and its FP64-parity miss is disclosed.
- [x] Stage 7 benchmark work remains locked.

## 13. Stage boundary

**Stage 6 result: PASS.**

No GPU speedup is claimed. No paper timing reproduction is claimed. The
small/medium benchmark campaign, Table II comparisons, and all headline
performance conclusions remain locked to Stage 7 or later.
