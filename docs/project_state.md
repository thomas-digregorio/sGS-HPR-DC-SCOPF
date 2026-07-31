# Project state

Last updated: 2026-07-31

## Stage gate

- Completed stage: **Stage 6 - GPU port for DGX Spark**
- Gate result: **PASS**
- Previous gate: **Stage 5 approved**
- Current state: **stopped at the Stage 7 approval gate**
- Next proposed stage: **Stage 7 - small and medium benchmark reproduction**
- Required approval: `APPROVE STAGE 6 AND RUN STAGE 7`
- Dashboard: private, owner-only Sites deployment; its URL and project
  identifier are intentionally omitted from the public repository.

Stage 7 has not started. The Stage 6 runs use only the two frozen correctness
fixtures established in earlier stages; no paper-scale benchmark case was run.

## Stage 6 implementation

The GPU port adds:

- a backend boundary for arrays, sparse matrices, matrix-vector products,
  reductions, projections, synchronization, transfers, and memory reporting;
- resident LP matrices and explicit transposes, scaling factors, state vectors,
  Halpern anchors, reusable workspaces, and diagnostic buffers;
- the complete Stage 5 preprocessing, adaptive-penalty, restart, and
  original-space validation path on CuPy FP64 arrays;
- a low-level cuSPARSE matrix-vector implementation whose selected algorithm
  can be inspected instead of inferred from a high-level call;
- direct Cholesky equality solves for the scaled production path and a separate
  unscaled structural cross-check;
- synchronized phase timing and an explicit host/device transfer ledger;
- an FP64 correctness gate and a separate non-gating FP32 diagnostic.

General diagonal scaling changes the equality Gram matrix. The raw Equation
(55) structural descriptor is therefore never reused for the scaled production
path.

## FP64 CPU and GPU parity

The trusted CPU and DGX GPU paths used the same frozen LP, preprocessing,
initial state, control policy, and stopping thresholds.

| Case | CPU / GPU iterations | CPU / GPU restarts | GPU raw KKT | GPU objective | Maximum final relative state error |
|---|---:|---:|---:|---:|---:|
| Public case5, T=1 | 410 / 410 | 4 / 4 | 0.005618456235 | 17479.839088898956 | 2.62e-14 |
| Synthetic resource fixture, T=2 | 1,032 / 1,032 | 8 / 8 | 0.008948422297 | 26580.274984099353 | 6.90e-15 |

For context, the corresponding CPU raw KKT values were 0.005618456243 and
0.008948422297. The CPU objectives were 17479.839088898974 and
26580.274984099335. The differences are at FP64 rounding scale.

Both GPU candidates preserve every Stage 5 acceptance boundary in original
units:

- each separately normalized Equation (54) block is at most `5e-5`;
- raw KKT is at most `0.01`;
- maximum physical violation is at most `0.01 MW/MWh`;
- scaled objective gap is at most `2e-4`.

One-step and short-trajectory checks make the parity claim stronger than a
final-objective comparison. They compare resident sparse products, state
updates, policy decisions, and recovered residuals before the full stopping
point.

## Sparse execution evidence

The DGX run verifies that the low-level cuSPARSE path actually selects
`CUSPARSE_SPMV_CSR_ALG2`. The matrix and explicit transpose descriptors remain
resident, their work buffers are reused, and repeated probes are deterministic
on the frozen input.

The project also preserves a clearly labeled high-level CuPy sparse fallback.
That fallback uses the library default and is not described as CSR_ALG2. This
distinction prevents a requested algorithm name from being mistaken for an
observed runtime choice.

## Timing boundary

Stage 6 records the following synchronized phases separately:

1. initialization;
2. first compilation and warm-up;
3. allocation accounting inside synchronized solver initialization;
4. host-to-device transfer;
5. solver initialization;
6. iterations;
7. residual checks;
8. device-to-host transfer;
9. total end-to-end time.

It also preserves a fixed 1,000-iteration resident-loop diagnostic with
residual checks every 100 steps. That diagnostic isolates steady-state device
work; it is not a convergence run.

Stage 6 makes **no CPU/GPU speedup claim**. The T=1 and T=2 fixtures are small
correctness cases, the paper used an A100 rather than a DGX Spark GB10, and the
paper does not publish a compatible end-to-end timing boundary. Repeated fair
benchmarking belongs to Stage 7.

## Precision boundary

FP64 is the sole Stage 6 acceptance precision. FP32 is run only after the FP64
gate passes and is stored as a diagnostic. An FP32 outcome cannot replace,
weaken, or retroactively change any FP64 threshold. Mixed precision remains
outside the approved Stage 6 scope.

## Evidence and quality status

- `results/raw/stage_6/stage_6_validation.json`: machine-readable DGX,
  numerical, sparse-kernel, residency, precision, and timing evidence
- `results/raw/stage_6/stage_6_trajectories.jsonl.gz`: preserved trajectory and
  policy records
- `results/raw/stage_6/stage_6_checks.json`: independent evidence checks
- `configs/sgs_hpr/stage_6_gpu_dgx.json`: thresholds and timing boundaries
  frozen before the final interpretation
- `docs/stage_reports/stage_6_report.md`: human-readable acceptance report

## Environment

Stage 6 ran in a repository-local virtual environment on:

- Ubuntu 24.04.4 LTS, aarch64;
- NVIDIA DGX Spark with GB10 GPU, compute capability 12.1;
- NVIDIA driver 580.173.02;
- CUDA driver API 13.0, CuPy CUDA runtime API 13.2, and CUDA 13.0.3 toolkit;
- CPython 3.12.3;
- CuPy 14.1.1 for CUDA 13, NumPy 2.3.5, and SciPy 1.16.3;
- 121.690 GiB unified system memory with ATS addressing.

The public environment record excludes the SSH alias, network address,
credentials, and raw host inventory.

## Supported claim and limitations

Stage 6 supports this claim: the validated Stage 5 FP64 algorithm has a
resident CuPy/cuSPARSE DGX Spark implementation that follows the same path and
reaches the same stopping iterations on the two frozen correctness cases.

It does not establish:

- identity with the authors' unpublished CUDA source;
- equivalence between the DGX Spark GB10 and the paper's A100 platform;
- an exact paper timing reproduction;
- a CPU/GPU speedup;
- paper-scale numerical or performance behavior;
- suitability of FP32 as the acceptance precision.

The public case5 result remains a mathematical reproduction of the printed
model on an independently sourced network. The T=2 resource case remains a
labeled synthetic structural fixture. Neither is an exact paper instance.

## Next proposed stage

Stage 7 may begin only after the exact approval command:

```text
APPROVE STAGE 6 AND RUN STAGE 7
```

Stage 7 will pin public benchmark-case provenance, reconcile dimensions and
sparsity before solving, run small and medium cases, validate every result in
physical and optimization terms, and use repeated compatible timing boundaries
before discussing performance.
