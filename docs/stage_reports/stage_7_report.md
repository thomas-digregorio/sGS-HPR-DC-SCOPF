# Stage 7 report

> **Status:** PASS
>
> **Date:** August 3, 2026
>
> **Stage:** Small and medium benchmark reproduction
>
> **Classification:** Structural reproduction

Stage 7 completed the six predeclared small and medium DGX Spark runs. Every
required HiGHS, CPU FP64 sGS-HPR, and GPU FP64 sGS-HPR correctness and timing
track passed. The independent checker passed all 19 checks.

This is not an exact reproduction of the paper's numerical instances or A100
timings. The public network cases are pinned exactly, and all reported matrix
dimensions are reproduced exactly, but the authors' renewable and storage
placements, time series, and matrix-construction code remain unavailable. The
frozen reconstruction therefore has a different sparse nonzero pattern in all
18 Table II rows. That difference blocks direct comparison with every paper
time.

## 1. Acceptance decision

The thresholds were frozen before benchmark execution and were not relaxed:

| Acceptance item | Required value | Stage 7 result |
|---|---:|---|
| Each normalized paper stopping block | at most `5e-5` | PASS |
| Raw KKT norm | at most `0.01` | PASS |
| Maximum physical violation | at most `0.01 MW/MWh` | PASS |
| Scaled objective gap to HiGHS | at most `2e-4` | PASS |
| Per-solve deadline | at most `3,600 s` | PASS |
| Required solver tracks | HiGHS, CPU FP64, GPU FP64 | PASS |
| Executed Stage 7 cases | exactly six | PASS |
| Stage 8 allocations | zero | PASS |
| Independent evidence checks | all | PASS, `19/19` |

FP64 was the only accepted precision. Mixed precision was disabled. Gurobi was
optional only when installed and licensed; it was not installed on this DGX,
so its absence did not weaken any required gate.

## 2. Provenance and the fail-closed correction

### 2.1 Pinned public network files

The three unmodified network files come from MATPOWER 8.1 at resolved commit
`1a828c7af590714499284e36ee9c81273388c594`. Identity is defined over the
canonical LF Git-blob content so the same source has the same digest on Windows
and Linux.

| Case | Upstream Git blob | Canonical SHA-256 |
|---|---|---|
| `case1354pegase` | `d6ede376f35af472b45b93ae771209c483427c26` | `1b08b25a2f6c1d540d090009dfaff41ff2b05784a2d8d302a7ad695821557b89` |
| `case2868rte` | `0223116b52b3bd10786ccd61a808c440826aacdc` | `2b30e8943daf84ccb111cee30f19f4917afc9c3772cab3ce9eaf6193988a6861` |
| `case9241pegase` | `cc9816b188ef38725c1e7c5b04cb9555b6b8a78e` | `593a58ecddb5af509ff94410a6630f81021b48fa31da0694ff516acfa9ea5f3b` |

The recorded release DOI is `10.5281/zenodo.15871662`, and the files retain the
MATPOWER three-clause BSD license.

### 2.2 Immutable failed preflight

The first DGX attempt, at commit
`71706ef076114bb3c42480d367e735e0b367828c`, failed during provenance
preflight. The expected Git blob IDs were correct, but the configured SHA-256
values had been computed from a Windows CRLF working-tree representation. The
Linux checkout exposed the mismatch against canonical LF Git-blob bytes.

The runner stopped before MATPOWER parsing, symbolic-ledger construction,
model allocation, or solver execution. No benchmark result came from that
attempt. Its JSON evidence remains unchanged under
`results/raw/stage_7/attempts/stage7-71706ef07611-20260803T172803Z-preflight-fail/`.

The correction changed only the portable hash definition and its expected
digests. It did not change model construction, solver behavior, stopping
rules, precision, scaling, or timing policy. The successful run then executed
from clean detached commit `ff6f762a00463e4769861f6aaf6f6fbbad6cc8af`.

The final frozen identities were:

- configuration canonical SHA-256:
  `06a172463049c519ab14c446d8b9ab632cd91c8afa4b44264e284b3a4f59a062`;
- DGX requirements canonical SHA-256:
  `827065b5bfc2920492cfe653e922cd2d3b2b4289ade12b06d866bea83d32dacf`;
- reconstruction policy fingerprint:
  `e6911ef7e5ccab32a8392c917b892eeabbed3df16a44b1e342cd8ef664274dcf`;
- run fingerprint:
  `65051e4de81a55b2803d1ff885db182f75e9865ad3f1703dd4a7a335886d6624`.

### 2.3 Author additions remain unavailable

The paper-specific load profiles, renewable locations and profiles, storage
locations and parameters, reserve requirements, generator-ramp
modifications, matrix-construction code, and exact restart/penalty source were
not available. Stage 7 therefore used one frozen, transparent reconstruction:
flat public demand, deterministic resource placement, load-relative RG/ESS
and reserve magnitudes, fixed storage parameters, deterministic ramp proxies,
and a selected-bus PTDF zero threshold of `1e-12`.

None of these choices was tuned after seeing dimensions, nonzero counts,
convergence, or timing.

## 3. Exact structural ledger for all Table II rows

The row and variable counts match the paper exactly in all 18 rows. This is
possible because those dimensions follow from the published aggregate counts
and formulas. Sparse nonzero counts do not match because they also depend on
the unavailable resource locations and PTDF support.

`S7 run` means the full LP was allocated and solved in Stage 7. `S8 locked`
means only the exact symbolic count was evaluated; no full LP or solver state
was allocated. Cause `A` is defined below the table.

| Case | Scope | Paper m x n | Reproduced m x n | abs dm (%) | abs dn (%) | Paper nnz | Reproduced nnz | abs dnnz | dnnz (%) | Cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case1354pegase, T=4 | S7 run | 20,192 x 4,208 | 20,192 x 4,208 | 0 (0%) | 0 (0%) | 7,190,640 | 4,799,808 | 2,390,832 | 33.249% | A |
| case1354pegase, T=16 | S7 run | 82,124 x 16,832 | 82,124 x 16,832 | 0 (0%) | 0 (0%) | 28,791,792 | 19,228,464 | 9,563,328 | 33.215% | A |
| case1354pegase, T=48 | S7 run | 247,276 x 50,496 | 247,276 x 50,496 | 0 (0%) | 0 (0%) | 86,586,352 | 57,896,368 | 28,689,984 | 33.135% | A |
| case1354pegase, T=96 | S7 run | 495,004 x 100,992 | 495,004 x 100,992 | 0 (0%) | 0 (0%) | 173,800,432 | 116,420,464 | 57,379,968 | 33.015% | A |
| case2868rte, T=4 | S7 run | 40,163 x 9,488 | 40,163 x 9,488 | 0 (0%) | 0 (0%) | 30,111,616 | 19,073,056 | 11,038,560 | 36.659% | A |
| case2868rte, T=16 | S7 run | 163,823 x 37,952 | 163,823 x 37,952 | 0 (0%) | 0 (0%) | 120,508,576 | 76,354,336 | 44,154,240 | 36.640% | A |
| case2868rte, T=48 | S8 locked | 493,583 x 113,856 | 493,583 x 113,856 | 0 (0%) | 0 (0%) | 295,998,240 | 229,507,104 | 66,491,136 | 22.463% | A |
| case2868rte, T=56 | S8 locked | 576,023 x 132,832 | 576,023 x 132,832 | 0 (0%) | 0 (0%) | 345,459,808 | 267,886,816 | 77,572,992 | 22.455% | A |
| case2868rte, T=64 | S8 locked | 658,463 x 151,808 | 658,463 x 151,808 | 0 (0%) | 0 (0%) | 394,957,984 | 306,303,136 | 88,654,848 | 22.447% | A |
| case2868rte, T=72 | S8 locked | 740,903 x 170,784 | 740,903 x 170,784 | 0 (0%) | 0 (0%) | 444,492,768 | 344,756,064 | 99,736,704 | 22.438% | A |
| case2868rte, T=80 | S8 locked | 823,343 x 189,760 | 823,343 x 189,760 | 0 (0%) | 0 (0%) | 494,064,160 | 383,245,600 | 110,818,560 | 22.430% | A |
| case2868rte, T=88 | S8 locked | 905,783 x 208,736 | 905,783 x 208,736 | 0 (0%) | 0 (0%) | 543,672,160 | 421,771,744 | 121,900,416 | 22.422% | A |
| case2868rte, T=96 | S8 locked | 988,223 x 227,712 | 988,223 x 227,712 | 0 (0%) | 0 (0%) | 593,316,768 | 460,334,496 | 132,982,272 | 22.413% | A |
| case9241pegase, T=4 | S8 locked | 152,774 x 24,700 | 152,774 x 24,700 | 0 (0%) | 0 (0%) | 373,238,888 | 342,863,272 | 30,375,616 | 8.138% | A |
| case9241pegase, T=6 | S8 locked | 230,376 x 37,050 | 230,376 x 37,050 | 0 (0%) | 0 (0%) | 559,872,262 | 514,308,838 | 45,563,424 | 8.138% | A |
| case9241pegase, T=16 | S8 locked | 618,386 x 98,800 | 618,386 x 98,800 | 0 (0%) | 0 (0%) | 1,493,149,532 | 1,371,647,068 | 121,502,464 | 8.137% | A |
| case9241pegase, T=24 | S8 locked | 928,794 x 148,200 | 928,794 x 148,200 | 0 (0%) | 0 (0%) | 2,239,903,828 | 2,057,650,132 | 182,253,696 | 8.137% | A |
| case9241pegase, T=32 | S8 locked | 1,239,202 x 197,600 | 1,239,202 x 197,600 | 0 (0%) | 0 (0%) | 2,986,775,884 | 2,743,770,956 | 243,004,928 | 8.136% | A |

**Cause A:** The authors' renewable/storage placement and PTDF construction
are unavailable. The frozen deterministic selected-bus PTDF support uses a
`1e-12` zero threshold and was not tuned to Table II nonzeros or timing.

Every reproduced nonzero count is lower than the paper count. Therefore zero
of 18 rows is paper-time comparable. Matching dimensions alone is not enough
to make sparse workloads identical.

## 4. Executed campaign and solver availability

Exactly these six LPs were allocated and executed:

- `case1354pegase`: T=4, T=16, T=48, and T=96;
- `case2868rte`: T=4 and T=16.

No `case2868rte` horizon above T=16 and no `case9241pegase` horizon was
allocated. The evidence records `stage_8_allocation_count = 0`.

| Track | Availability | Role | Gating result |
|---|---|---|---|
| SciPy HiGHS dual simplex | Available, SciPy 1.16.3 | Independent LP reference | PASS |
| CPU FP64 sGS-HPR | Available | Structural CPU implementation | PASS |
| GPU FP64 sGS-HPR | Available, CuPy 14.1.1 | Resident DGX implementation | PASS |
| Gurobi | Not installed or licensed | Optional only if available | Not run; non-gating |

The CPU and GPU paths used the generalized scaled block-arrow equality solve.
No dense equality Gram matrix was materialized. The inequality spectral
certificate was also sparse-only: neither the dense matrix nor its normal
matrix was materialized. The GPU path verified FP64 and selected low-level
`CUSPARSE_SPMV_CSR_ALG2` for both equality and inequality sparse products.

## 5. Numerical correctness

### 5.1 Maxima over every required correctness candidate

The table below takes the maximum over all six cases and all three required
tracks, for 18 accepted candidates total.

| Metric | Observed maximum | Gate | Location | Result |
|---|---:|---:|---|---|
| Normalized primal block | 2.01695e-8 | 5e-5 | case2868rte, T=4, GPU | PASS |
| Normalized stationarity block | 4.17967e-6 | 5e-5 | case2868rte, T=4, GPU | PASS |
| Normalized box block | 5.49412e-15 | 5e-5 | case1354pegase, T=96, GPU | PASS |
| Raw KKT norm | 0.0096347433 | 0.01 | case2868rte, T=16, CPU | PASS |
| Maximum physical violation | 0.0062210399 | 0.01 | case1354pegase, T=4, GPU | PASS |
| Scaled objective gap to HiGHS | 4.28499e-8 | 2e-4 | case1354pegase, T=4, CPU | PASS |
| Equality violation | 0.0062210392 | 0.01 | case1354pegase, T=4, GPU | PASS |
| Positive inequality violation | 0.0016556756 | 0.01 | case2868rte, T=4, GPU | PASS |
| Box violation | 1.04045e-10 | 0.01 | case1354pegase, T=96, GPU | PASS |

The physical validator independently checked power balance, line limits,
reserve boxes and requirements, generator headroom/footroom and ramping,
renewable bounds, storage power and energy limits, terminal energy, and
angle-based flow against the compressed PTDF calculation.

### 5.2 Per-case reference and sGS-HPR results

CPU and GPU sGS-HPR reached the same iteration and restart counts in every
case. Their maximum objective difference was `6.06e-9`, at FP64 rounding
scale for these objective magnitudes.

| Case | HiGHS objective | CPU/GPU sGS-HPR objective | CPU/GPU iterations | CPU/GPU restarts | Worst HPR raw KKT | Worst HPR physical | HPR objective gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| case1354pegase, T=4 | 263,014.8120 | 263,014.8007 | 302 / 302 | 3 / 3 | 8.150e-3 | 6.221e-3 | 4.285e-8 |
| case1354pegase, T=16 | 1,052,059.248 | 1,052,059.232 | 304 / 304 | 3 / 3 | 7.312e-3 | 3.480e-3 | 1.527e-8 |
| case1354pegase, T=48 | 3,156,177.744 | 3,156,177.712 | 359 / 359 | 3 / 3 | 6.467e-3 | 2.838e-3 | 1.022e-8 |
| case1354pegase, T=96 | 6,312,355.488 | 6,312,355.490 | 404 / 404 | 4 / 4 | 7.677e-4 | 4.039e-4 | 3.383e-10 |
| case2868rte, T=4 | 283,774.6800 | 283,774.6696 | 416 / 416 | 4 / 4 | 6.293e-3 | 2.840e-3 | 3.652e-8 |
| case2868rte, T=16 | 1,135,098.720 | 1,135,098.703 | 454 / 454 | 4 / 4 | 9.635e-3 | 4.481e-3 | 1.504e-8 |

## 6. Timing protocol and results

For each available track, the runner first completed a correctness solve that
was excluded from statistics. It then completed one warm-up followed by at
least five measured repetitions. When the initial variability rule fired, the
runner preserved those samples and extended the set to nine repetitions.

The `First` column is the pre-timing correctness solve. `First measured` is
preserved separately from the summary statistics. `Measured` reports the
solver-core wall boundary named for each track. `Attempt median` includes the
small Python wrapper overhead around that core boundary. All values are seconds.

### 6.1 HiGHS timing

HiGHS core time is the SciPy `linprog` call, including its interface/model
setup and solve.

| Case | First | Warm-up | First measured | n | Measured median [min, max] | Std. dev. | IQR | Attempt median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case1354pegase, T=4 | 1.461 | 1.458 | 1.455 | 5 | 1.463 [1.453, 1.480] | 0.011 | 0.015 | 1.464 |
| case1354pegase, T=16 | 7.256 | 7.269 | 7.267 | 5 | 7.265 [7.258, 7.283] | 0.010 | 0.005 | 7.266 |
| case1354pegase, T=48 | 31.610 | 31.073 | 31.249 | 9 | 32.095 [31.249, 39.290] | 2.629 | 1.651 | 32.096 |
| case1354pegase, T=96 | 101.680 | 101.793 | 101.379 | 5 | 101.242 [100.940, 101.613] | 0.247 | 0.176 | 101.243 |
| case2868rte, T=4 | 5.370 | 5.293 | 5.348 | 5 | 5.366 [5.340, 5.375] | 0.016 | 0.026 | 5.367 |
| case2868rte, T=16 | 22.065 | 22.165 | 22.421 | 5 | 22.138 [22.082, 22.421] | 0.150 | 0.219 | 22.138 |

### 6.2 CPU FP64 sGS-HPR timing

The CPU core is a solve using one prepared sparse workspace. It includes the
iteration loop, stopping checks, and final original-coordinate residual
evaluation. One-time workspace setup is outside this boundary.

| Case | First | Warm-up | First measured | n | Measured median [min, max] | Std. dev. | IQR | Attempt median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case1354pegase, T=4 | 12.106 | 11.684 | 13.075 | 9 | 13.190 [11.836, 16.879] | 1.545 | 1.571 | 13.190 |
| case1354pegase, T=16 | 136.058 | 134.993 | 127.424 | 9 | 47.399 [46.635, 127.424] | 31.103 | 3.906 | 47.399 |
| case1354pegase, T=48 | 165.133 | 155.342 | 154.407 | 5 | 153.046 [152.641, 155.015] | 1.077 | 1.680 | 153.046 |
| case1354pegase, T=96 | 332.289 | 391.588 | 537.835 | 9 | 336.657 [330.488, 537.835] | 67.932 | 7.599 | 336.658 |
| case2868rte, T=4 | 60.617 | 60.179 | 60.350 | 5 | 60.350 [60.018, 60.807] | 0.300 | 0.253 | 60.350 |
| case2868rte, T=16 | 249.985 | 254.638 | 255.187 | 5 | 250.601 [249.273, 255.187] | 2.302 | 1.645 | 250.601 |

The long early values for case1354 T=16 and T=96 are preserved, not removed.
The variability rule extended both series to nine samples. The median and IQR
show the later stable cluster while the maximum and standard deviation keep
the early variability visible.

### 6.3 GPU FP64 sGS-HPR timing

The GPU core uses a prepared resident workspace and includes zero-state
upload, the synchronized iteration loop, every-iteration stopping checks, and
final state recovery and transfer.

| Case | First | Warm-up | First measured | n | Measured median [min, max] | Std. dev. | IQR | Attempt median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case1354pegase, T=4 | 1.167 | 1.015 | 0.995 | 5 | 1.013 [0.995, 1.092] | 0.037 | 0.010 | 1.014 |
| case1354pegase, T=16 | 3.047 | 2.897 | 2.983 | 5 | 2.924 [2.885, 2.996] | 0.052 | 0.097 | 2.925 |
| case1354pegase, T=48 | 9.515 | 9.503 | 9.477 | 5 | 9.481 [9.476, 9.531] | 0.028 | 0.048 | 9.481 |
| case1354pegase, T=96 | 20.985 | 20.986 | 21.289 | 5 | 21.084 [20.947, 21.380] | 0.184 | 0.274 | 21.085 |
| case2868rte, T=4 | 3.928 | 3.886 | 3.939 | 5 | 3.939 [3.846, 3.959] | 0.055 | 0.098 | 3.939 |
| case2868rte, T=16 | 15.399 | 15.399 | 15.394 | 5 | 15.408 [15.394, 15.413] | 0.007 | 0.006 | 15.408 |

### 6.4 Construction, preprocessing, and complete case wall time

The complete case wall time includes construction, preprocessing, workspace
preparation, correctness runs, warm-ups, all measured repetitions, validation,
and evidence writing. It must not be compared with a solver-core median.

| Case | Model construction | Preprocessing | CPU workspace | GPU workspace | Complete case wall |
|---|---:|---:|---:|---:|---:|
| case1354pegase, T=4 | 0.174 | 2.400 | 0.061 | 1.767 | 170.460 |
| case1354pegase, T=16 | 0.525 | 10.243 | 0.245 | 0.291 | 925.315 |
| case1354pegase, T=48 | 0.488 | 29.860 | 0.701 | 0.514 | 1,553.423 |
| case1354pegase, T=96 | 0.835 | 59.589 | 1.411 | 2.561 | 4,922.366 |
| case2868rte, T=4 | 0.494 | 10.123 | 0.241 | 0.072 | 500.691 |
| case2868rte, T=16 | 0.857 | 39.651 | 0.946 | 0.267 | 2,071.388 |

No speedup is computed. The local tracks have explicitly different inclusion
boundaries, the paper does not disclose a compatible boundary, the reconstructed
nonzero counts differ, and the DGX Spark GB10 is not the paper's A100.

## 7. Memory, residency, and transfers

The table reports the frozen planning estimate, CUDA runtime-used snapshots
before and after each GPU campaign, CuPy pool size after the campaign, and
the audited transfers for the complete GPU track. `Process peak` is the
cumulative process-lifetime high-water mark; it is not isolated per solve.

| Case | Planning GiB | Runtime used GiB, before -> after | Pool after GiB | Preparation H2D MiB | Track H2D / D2H MiB | Process peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| case1354pegase, T=4 | 0.242 | 13.525 -> 14.561 | 0.180 | 55.731 | 57.259 / 3.218 | 1.376 |
| case1354pegase, T=16 | 0.967 | 15.212 -> 16.449 | 0.826 | 223.188 | 229.372 / 12.531 | 4.572 |
| case1354pegase, T=48 | 2.908 | 18.515 -> 20.752 | 2.772 | 671.932 | 690.531 / 37.392 | 12.271 |
| case1354pegase, T=96 | 5.834 | 24.146 -> 27.789 | 6.253 | 1,351.024 | 1,388.247 / 74.663 | 23.896 |
| case2868rte, T=4 | 1.008 | 23.414 -> 23.338 | 6.253 | 220.013 | 223.172 / 6.540 | 23.896 |
| case2868rte, T=16 | 4.033 | 26.501 -> 26.045 | 6.253 | 880.381 | 893.184 / 25.849 | 23.896 |

CUDA/CuPy does not expose a true per-solve device high-water mark through this
backend. The report therefore does not relabel snapshots or the allocator pool
as peak memory. Unified-memory runtime use can also move independently of the
CuPy pool, which explains why some after snapshots are lower.

Every transfer audit passed. No full state was copied inside the resident
iteration loop. Device-to-host traffic during iteration was restricted to
scheduled diagnostics and policy scalars; full states moved only at named
initialization or finalization boundaries.

## 8. Environment and independent evidence check

The successful run used:

- Linux 6.17.0-1029-nvidia, aarch64;
- CPython 3.12.3;
- NVIDIA GB10, compute capability 12.1, 48 multiprocessors;
- 130,663,165,952 bytes of unified global/system memory;
- CUDA driver API 13.0 and CuPy CUDA runtime API 13.2;
- CuPy 14.1.1, NumPy 2.3.5, and SciPy 1.16.3;
- signed 32-bit CSR indices and FP64 values.

The independent checker passed `19/19` checks. It independently reloaded the
configuration and evidence, recomputed immutable identities and the run
fingerprint, verified the exact clean executed commit, checked all 18 ledger
rows and the six-case boundary, enforced required-solver semantics, inspected
all numerical and timing gates, audited GPU memory and transfers, and confirmed
zero Stage 8 allocations.

### Post-run integrity maintenance

The accepted benchmark remains tied to executed commit
`ff6f762a00463e4769861f6aaf6f6fbbad6cc8af`. After execution, the independent
checker was corrected to accept honest Linux `getrusage`-only cumulative-peak
telemetry when live `psutil` RSS is unavailable; strict source-provenance and
memory-scope tests accompany that correction. The runner's source-manifest
preflight was also hardened so deletion of a tracked Python source file fails
closed.

These post-run integrity changes did not alter the accepted JSON evidence,
model, numerical thresholds, timing thresholds, or measured samples. They
strengthen validation around the existing evidence, so no numerical rerun was
required and no later source revision is presented as the executed benchmark.
The accepted `19/19` check is evaluated against the exact accepted commit. A
checker invocation from a later current head, whose execution-source files
differ from `ff6f762`, fails the exact-source-identity check by design; that is
a provenance refusal, not a failed numerical or timing gate.

The hardened checker was therefore rerun against a clean detached `ff6f762`
checkout with the canonical evidence and again passed `19/19`, including exact
source identity. The full repository suite passed 269 tests; the sole skip was
the expected local CuPy-version compatibility skip.

Primary artifacts:

- `results/raw/stage_7/stage_7_validation.json` - complete machine-readable
  benchmark evidence;
- `results/raw/stage_7/stage_7_checks.json` - independent 19-check result;
- `configs/benchmarks/stage_7_small_medium.json` - frozen protocol and gates;
- `environment/dgx_stage7_requirements.txt` - frozen DGX packages;
- `results/raw/stage_7/attempts/` - immutable failed preflight history.

## 9. What this stage teaches

### Dimensions and sparsity are different claims

Published aggregate counts are enough to reproduce `m` and `n`, but not the
matrix's sparse support. Moving a renewable or storage device to another bus
changes which PTDF entries appear in its columns. Two LPs can therefore have
identical dimensions and materially different nonzero counts, memory traffic,
and runtime.

### Correctness must precede timing

A fast result is not useful if it solves a different or insufficiently
feasible problem. Each track first passed objective, normalized stopping, raw
KKT, and physical checks. Only then did its warm-up and measured repetitions
become reportable.

### A fail-closed preflight is useful evidence

The line-ending failure did not produce a benchmark number. That is the
desired behavior: input identity was uncertain, so execution stopped before
allocation. Preserving the failed attempt also shows exactly why the portable
hash definition changed.

### Count-only analysis respects the stage gate

The large Table II rows still receive exact dimensional and nonzero
reconciliation without allocating their LPs. This separates useful structural
analysis from authorization to run the expensive Stage 8 campaign.

## 10. Limitations and next gate

Stage 7 does not establish:

- access to the authors' complete benchmark inputs or CUDA source;
- equality with the paper's sparse matrices or objectives;
- a direct reproduction of any Table II time;
- a Gurobi comparison on this DGX;
- a per-solve device peak-memory measurement;
- behavior of the locked large horizons;
- an N-1 security-constrained formulation; or
- equivalence between the DGX Spark GB10 and the paper's A100-SXM4-80GB.

Stage 8 remains locked. It may begin only after review of this PASS report and
the exact approval command:

```text
APPROVE STAGE 7 AND RUN STAGE 8
```
