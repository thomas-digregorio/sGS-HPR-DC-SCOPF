# Project state

Last updated: 2026-08-03

## Stage gate

- Completed stage: **Stage 7 - small and medium benchmark reproduction**
- Gate result: **PASS**
- Previous gate: **Stage 6 approved**
- Current state: **stopped at the Stage 8 approval gate**
- Next proposed stage: **Stage 8 - large paper-scale benchmarks**
- Required approval: `APPROVE STAGE 7 AND RUN STAGE 8`
- Dashboard: private, owner-only Sites deployment; its URL and project
  identifier are intentionally omitted from the public repository.

Stage 8 remains locked. Stage 7 allocated and solved exactly six cases and
recorded zero Stage 8 allocations.

## Stage 7 outcome

The campaign ran these structural reconstructions on the DGX Spark:

- `case1354pegase`: T=4, T=16, T=48, and T=96;
- `case2868rte`: T=4 and T=16.

For every case, SciPy HiGHS, CPU FP64 sGS-HPR, and GPU FP64 sGS-HPR passed the
frozen objective, stopping-residual, raw-KKT, and physical-validation gates.
Gurobi was not installed or licensed and remained optional under the frozen
protocol. The independent checker passed **19/19** checks.

The largest accepted values over all 18 required candidates were:

| Metric | Maximum | Gate |
|---|---:|---:|
| Normalized primal block | 2.01695e-8 | 5e-5 |
| Normalized stationarity block | 4.17967e-6 | 5e-5 |
| Normalized box block | 5.49412e-15 | 5e-5 |
| Raw KKT norm | 0.0096347433 | 0.01 |
| Physical violation | 0.0062210399 MW/MWh | 0.01 MW/MWh |
| Scaled objective gap to HiGHS | 4.28499e-8 | 2e-4 |

CPU and GPU sGS-HPR reached the same iteration and restart counts in all six
cases. The largest CPU/GPU objective difference was `6.06e-9`.

## Reproduction classification

This stage is a **structural reproduction**, not an exact paper-instance or
paper-timing reproduction.

The three public network files are pinned to MATPOWER 8.1 at commit
`1a828c7af590714499284e36ee9c81273388c594`. The authors' renewable/storage
locations, time series, physical modifications, and matrix-construction code
remain unavailable, so Stage 7 uses one disclosed deterministic protocol that
was frozen before execution and never tuned to the results.

All 18 Table II rows have exactly matching row and variable dimensions. Every
reconstructed sparse nonzero count differs from the paper by 8.136% to
36.659%. Consequently, zero rows are paper-time comparable. The 12 large rows
outside Stage 7 were reconciled by exact count-only analysis without allocating
their LPs.

## Timing and memory boundary

Each required track completed:

1. one correctness solve excluded from statistics;
2. one warm-up; and
3. at least five measured repetitions, extended to nine when the frozen
   variability rule fired.

The GPU solver-core medians were:

| Case | Median seconds |
|---|---:|
| case1354pegase, T=4 | 1.013 |
| case1354pegase, T=16 | 2.924 |
| case1354pegase, T=48 | 9.481 |
| case1354pegase, T=96 | 21.084 |
| case2868rte, T=4 | 3.939 |
| case2868rte, T=16 | 15.408 |

No speedup is claimed. HiGHS, CPU, GPU, complete-case, and paper boundaries
are kept separate. The sparse workloads also differ from Table II, and the DGX
Spark GB10 is not the paper's A100.

Memory planning, CUDA runtime snapshots, CuPy allocator state, cumulative
process high-water marks, and transfer ledgers are preserved per case. Every
transfer audit passed, and no full state moved to the host inside the resident
iteration loop. The backend does not expose a true isolated per-solve GPU
peak, so snapshots are not mislabeled as peak memory.

## Provenance correction

An initial attempt stopped during provenance preflight because configured
SHA-256 values represented Windows CRLF working-tree bytes while the DGX
checkout used canonical LF Git-blob bytes. It stopped before MATPOWER parsing,
symbolic counting, model allocation, or solver execution. That failed attempt
is preserved unchanged.

The portable correction defines identities over canonical LF Git-blob content
and verifies both the committed blobs and clean-filtered worktree. It changed
no model, solver, threshold, precision, scaling, or timing rule. The successful
campaign ran from clean detached commit
`ff6f762a00463e4769861f6aaf6f6fbbad6cc8af`.

After the run, integrity checks were tightened without changing accepted
evidence or thresholds. The checker now accepts accurately scoped Linux
`getrusage`-only cumulative-peak telemetry when `psutil` RSS is unavailable,
with strict provenance and scope tests. Source-manifest preflight now also
fails closed if a tracked Python source file is deleted. The accepted run
remains tied to `ff6f762`; no numerical rerun was required. The accepted
19/19 check is evaluated against that exact commit. Running the checker from a
later head with changed execution-source files fails source identity by design,
not because a numerical or timing gate changed. A clean detached `ff6f762`
recheck with the hardened checker passed 19/19; the full repository suite passed
269 tests with one expected local CuPy-version skip.

## Evidence

- `results/raw/stage_7/stage_7_validation.json`: complete machine-readable
  Stage 7 evidence
- `results/raw/stage_7/stage_7_checks.json`: independent 19-check result
- `results/raw/stage_7/attempts/`: immutable failed-preflight history
- `configs/benchmarks/stage_7_small_medium.json`: frozen protocol and gates
- `environment/dgx_stage7_requirements.txt`: frozen DGX package set
- `docs/stage_reports/stage_7_report.md`: detailed human-readable report

The successful environment was Linux/aarch64 with CPython 3.12.3, NVIDIA GB10
compute capability 12.1, CuPy 14.1.1, NumPy 2.3.5, and SciPy 1.16.3. The GPU
path used FP64, signed 32-bit CSR indices, the generalized scaled structural
equality solve, and verified `CUSPARSE_SPMV_CSR_ALG2` sparse products.

## Supported claim and limits

Stage 7 supports this claim: the frozen public-data structural reconstruction
passes independent LP, optimization, physical, CPU, and DGX GPU checks on the
six approved small/medium horizons, with repeated timing and explicit memory
and transfer evidence.

It does not establish:

- exact author-instance data, objectives, sparsity, or CUDA source;
- direct reproduction of Table II timing;
- Gurobi performance on the DGX;
- a true isolated per-solve device peak;
- behavior of Stage 8's locked large horizons;
- an N-1 SCOPF model; or
- platform equivalence between the GB10 and A100.

## Next proposed stage

Stage 8 may begin only after the exact approval command:

```text
APPROVE STAGE 7 AND RUN STAGE 8
```

Stage 8 remains locked until that command is received.
