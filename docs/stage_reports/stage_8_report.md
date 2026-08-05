# Stage 8 report

> **Stage decision:** FAIL
>
> **Original campaign status:** `STOPPED_ON_FAILURE`
>
> **GPU-only sequence 6--8 continuation:** `COMPLETE_WITH_RESOURCE_LIMITS`
>
> **Updated:** August 5, 2026
>
> **Classification:** Structural reproduction

## 1. Executive result

Stage 8 reached an honest terminal state but did not pass its acceptance gate.
Four large structural reconstructions passed in strict campaign order:

- `case2868rte:T48`
- `case2868rte:T64`
- `case2868rte:T96`
- `case9241pegase:T4`

The fifth row, `case9241pegase:T6`, passed its live memory gate, HiGHS track,
and GPU FP64 sGS-HPR track. Its required CPU FP64 sGS-HPR correctness attempt
exceeded the frozen 3,600-second per-solve deadline. The attempt ended after
`3,600.092739 s` with `TIME_LIMIT`, before an accepted CPU candidate, warm-up,
or measured CPU repetition was produced.

The strict-prefix policy therefore stopped the campaign at T6. No retry was
attempted. T16, T24, and T32 were not executed, and Stage 9 recorded zero
allocations. The Stage 8 checker passed all 12 protocol and evidence checks,
which confirms that the terminal failure was preserved honestly; it does not
turn that failure into a Stage 8 acceptance pass. Stage 9 remains locked.

On August 5, the user separately authorized a sequence 6--8 continuation with
only HiGHS and GPU FP64 sGS-HPR eligible. CPU sGS-HPR and Gurobi were explicitly
skipped and were not represented as passes. The original T6 failure and its
evidence remained unchanged.

The continuation resolved all three requested rows without allocating an LP:

- T16's unchanged 94.435 GiB unified projection exceeded both live 80% safety
  budgets: 65.784 GiB from observed host-free pages and 65.496 GiB from
  observed CUDA-free memory. It was recorded as `MEMORY_BLOCKED` before model,
  HiGHS, or GPU allocation.
- T24 and T32 remained `INDEX_BLOCKED` because their conservative planning
  nonzero counts exceed the largest signed 32-bit CSR index. Both decisions
  were made without allocation.

Because no requested row passed its preallocation gate, the continuation
produced no new HiGHS or GPU correctness runs or timings. Its independent
checker passed 13 of 13 evidence checks. This safely completes the requested
classification sequence; it does not convert Stage 8 acceptance to PASS.

## 2. Evidence and provenance

| Item | Recorded value |
|---|---|
| Executed commit | `f1fffc2adcba197040578695ba11dd27b0d1981f` |
| Worktree state | Clean detached worktree |
| Run fingerprint | `4dcb61115fb60f49c5972839ac0c99585cf74de2d3d07b955e142bb4f0f8e7cd` |
| Campaign start | `2026-08-03T22:51:58.679362+00:00` |
| Campaign end | `2026-08-04T22:26:47.230493+00:00` |
| Stage 8 configuration SHA-256 | `ac61cf282fe3f146a1fbe5e2d1bff87b4ce36641d30f6feaaf3a64d48bd04284` |
| Final evidence SHA-256 | `f4197554d8f7e108a4ca2701cbb0b12a485d593febb76f56f431fb482af14254` |
| Checker output SHA-256 | `ff8f48af5c9f2866eb1a461776a3128e87100dc75bc99618671e63da5e94672b` |
| Executed source manifest | 27 of 27 entries passed |
| Allocation attempts | 5, all unique and in order |
| Retries | 0 |

The separate GPU-only continuation has the following provenance:

| Item | Recorded value |
|---|---|
| Executed commit | `1cf9da62e263a1fb8cc7e68e6cecc4958e602a22` |
| Worktree state | Clean detached worktree |
| Run fingerprint | `238231a3ac6f648c57bee8551bb4755b9d58a5d05be951bff085a22c6f5a70b0` |
| Continuation start | `2026-08-05T20:32:03.106308+00:00` |
| Continuation end | `2026-08-05T20:32:04.167362+00:00` |
| Continuation configuration SHA-256 | `76ff7cb76f70ff104d1691a152b57ebabaade3e26b784091b888c1d5918cc64c` |
| Final continuation evidence SHA-256 | `edf18f6cda959c47fe5d7c38370c5f88619ee08a8ee4f9dbb480756ff1d34f7b` |
| DGX checker output SHA-256 | `8cda412285568b2685abc58d6571132f7e83123f9158500d3efd1c1ec976fa65` |
| Executed source manifest | 34 of 34 entries passed |
| Continuation allocation attempts | 0 |

The accepted Stage 7 configuration, evidence, and DGX requirements were reused
without changing the algorithm, reconstruction, thresholds, or timing
protocol. Their canonical Git-blob SHA-256 values were:

| Frozen Stage 7 item | SHA-256 |
|---|---|
| Configuration | `06a172463049c519ab14c446d8b9ab632cd91c8afa4b44264e284b3a4f59a062` |
| Evidence | `180699f6b34228c3e1a69b158677b12dd3242582a7225e1cdb44aeaac29931ae` |
| DGX requirements | `827065b5bfc2920492cfe653e922cd2d3b2b4289ade12b06d866bea83d32dacf` |

The successful source and configuration preflights ran on Linux/aarch64 with
CPython 3.12.3, NumPy 2.3.5, SciPy 1.16.3, CuPy 14.1.1, CUDA driver API 13.0,
CUDA runtime API 13.2, and the NVIDIA GB10 at compute capability 12.1. FP64
and signed 32-bit CSR indices were required. Gurobi was not installed and was
optional under the frozen contract.

## 3. Frozen acceptance contract

| Gate | Required value |
|---|---:|
| Each normalized stopping block | at most `5e-5` |
| Raw KKT norm | at most `0.01` |
| Maximum physical violation | at most `0.01 MW/MWh` |
| Scaled objective gap to HiGHS | at most `2e-4` |
| Per-solve deadline | at most `3,600 s` |
| Tracks required for an allocated row | HiGHS, CPU FP64, and GPU FP64 |

Each successful timing track required one correctness solve, one warm-up, and
five measured repetitions. The campaign had to stop at the first numerical,
time, or resource failure. A later row could not be skipped to, and retrying a
failed row required an explicit retry invocation.

The separately frozen continuation changed only solver-track scope for
sequences 6--8: HiGHS and GPU FP64 were required if a row reached allocation;
CPU sGS-HPR and Gurobi were explicit non-gating skips. It did not change the
mathematical reconstruction, numerical thresholds, timing repetitions,
3,600-second deadline, 80% memory fractions, unified-memory accounting, or
signed-int32 sparse-index requirement.

## 4. Campaign outcome

| Seq. | Row | Full LP allocated? | Track result | Final row status |
|---:|---|---|---|---|
| 1 | `case2868rte:T48` | Yes | HiGHS PASS; CPU PASS; GPU PASS | `PASS` |
| 2 | `case2868rte:T64` | Yes | HiGHS PASS; CPU PASS; GPU PASS | `PASS` |
| 3 | `case2868rte:T96` | Yes | HiGHS PASS; CPU PASS; GPU PASS | `PASS` |
| 4 | `case9241pegase:T4` | Yes | HiGHS PASS; CPU PASS; GPU PASS | `PASS` |
| 5 | `case9241pegase:T6` | Yes | HiGHS PASS; CPU `TIME_LIMIT`; GPU PASS | `FAIL` |
| 6 | `case9241pegase:T16` | No | GPU-only continuation: HiGHS/GPU not run; live unified-memory gate failed | `MEMORY_BLOCKED` |
| 7 | `case9241pegase:T24` | No | GPU-only continuation: static signed-int32 CSR block | `INDEX_BLOCKED` |
| 8 | `case9241pegase:T32` | No | GPU-only continuation: static signed-int32 CSR block | `INDEX_BLOCKED` |

The original campaign's passing prefix length remains four. Its five allocated
keys exactly match the first five campaign rows. The separate continuation
made zero allocations while resolving sequences 6--8 at preallocation safety
boundaries. Reconciliation-only rows and Stage 9 still received zero
allocations, and the optional N-1 extension remained disabled.

T24 and T32 have fail-closed static resource entries because their conservative
planning envelopes exceed the largest signed 32-bit index value. Those ledger
entries are not claims that either full LP was allocated or executed.

## 5. Numerical results

### 5.1 Objectives and iteration counts

The table reports only candidates that completed validation. A dash means that
no accepted candidate exists for that required track.

| Row | HiGHS objective | CPU objective | CPU iter./restarts | GPU objective | GPU iter./restarts |
|---|---:|---:|---:|---:|---:|
| T48 | 3,405,296.160000 | 3,405,296.187709 | 499 / 4 | 3,405,296.187709 | 499 / 4 |
| T64 | 4,540,394.880000 | 4,540,394.884338 | 503 / 5 | 4,540,394.884338 | 503 / 5 |
| T96 | 6,810,592.320000 | 6,810,592.337710 | 503 / 5 | 6,810,592.337710 | 503 / 5 |
| T4 | 1,124,679.519623 | 1,124,679.506100 | 1,272 / 11 | 1,124,679.506128 | 1,272 / 11 |
| T6 | 1,687,019.279434 | - | - | 1,687,019.276585 | 1,541 / 12 |

T6's CPU timeout occurred during the correctness attempt's residual-evaluation
path. The evidence does not support inferring a CPU objective, convergence
result, or timing median from that interrupted attempt.

### 5.2 Maximum validated values

These maxima cover completed HiGHS, CPU, and GPU candidates only. T6's failed
CPU attempt is excluded because it produced no accepted candidate.

| Metric | Maximum observed | Location | Gate | Result |
|---|---:|---|---:|---|
| Normalized primal block | `3.010269e-9` | T4 CPU | `5e-5` | PASS |
| Normalized stationarity block | `9.659299e-6` | T96 GPU | `5e-5` | PASS |
| Normalized box block | `8.830567e-13` | T4 CPU | `5e-5` | PASS |
| Raw KKT norm | `0.0093260531` | T48 CPU | `0.01` | PASS |
| Maximum physical violation | `0.0063110950` | T4 CPU | `0.01` | PASS |
| Scaled objective gap to HiGHS | `1.202323e-8` | T4 CPU | `2e-4` | PASS |

The completed T6 GPU candidate also passed every numerical gate: raw KKT
`0.0071068783`, physical violation `0.0056598670`, and scaled objective gap
`1.688706e-9`. That GPU success does not satisfy the row contract without the
required CPU track.

## 6. Timing results

All values below are local solver-core seconds. Medians use five measured
repetitions after a separate correctness run and warm-up. The raw repetitions,
attempt-wall values, standard deviations, and IQRs remain in the machine-
readable evidence.

| Row | HiGHS median (range) | CPU median (range) | GPU median (range) |
|---|---:|---:|---:|
| T48 | 76.239 (70.892-77.949) | 804.863 (801.569-808.401) | 49.968 (49.927-49.998) |
| T64 | 108.232 (107.335-108.458) | 1,078.892 (1,077.122-1,135.954) | 68.383 (68.292-68.392) |
| T96 | 163.593 (158.374-164.829) | 1,621.905 (1,594.076-1,634.230) | 105.022 (105.001-105.174) |
| T4 | 142.479 (141.937-142.810) | 3,087.218 (3,070.085-3,096.832) | 193.632 (193.603-193.741) |
| T6 | 963.957 (960.497-966.056) | No median: correctness timed out at 3,600.093 | 357.544 (357.462-357.751) |

No Stage 8 speedup is claimed. The local CPU and GPU timing boundaries are not
identical, every reconstructed sparse workload differs from the paper, the
paper's inclusion boundary is under-specified, and the DGX Spark GB10 is not
the paper's A100. The local timing values therefore cannot be divided into, or
by, the paper's Table II values.

## 7. Memory and allocation safety

The GB10 exposes a unified physical memory pool. Before each allocation, the
runner required the conservative host-plus-device projection to fit within 80%
of both observed host-available memory and observed CUDA-free memory.

| Row | Unified projection GiB | Host budget GiB | Device budget GiB | Gate |
|---|---:|---:|---:|---|
| T48 | 24.114 | 82.657 | 82.367 | PASS |
| T64 | 32.169 | 80.811 | 80.522 | PASS |
| T96 | 48.303 | 80.447 | 80.157 | PASS |
| T4 | 23.606 | 79.659 | 79.255 | PASS |
| T6 | 35.410 | 79.413 | 79.124 | PASS |

The highest cumulative process-lifetime host high-water mark recorded in the
campaign was 95.375 GiB during T6. It is not an isolated per-solve peak. CUDA
and CuPy snapshots likewise bracket solves but are not mislabeled as true
per-solve GPU peaks. Every completed GPU candidate's transfer audit passed;
no full solver state moved to the host inside the resident iteration loop.

The original strict-prefix campaign never evaluated T16's live gate because it
stopped at T6. The separate GPU-only continuation did evaluate it. At that
instant the runner observed 82.231 GiB of host-free pages and 81.870 GiB of
CUDA-free memory. Applying the unchanged 80% fractions produced 65.784 GiB and
65.496 GiB budgets, respectively. The 94.435 GiB unified projection exceeded
both, so the runner stopped before constructing the LP.

T24 and T32 were resolved without allocation from the frozen count-only
ledger. Their conservative planning nonzero counts are 2,531,600,260 and
3,375,704,460, both above 2,147,483,647. No new HiGHS, CPU, GPU, or Gurobi
solver call was made for T16, T24, or T32.

## 8. Structural comparison boundary

Stage 8 remains a structural reproduction. Every reproduced Table II row has
the published row and variable dimensions, but every reproduced sparse nonzero
count differs from the paper. The five reached rows differ as follows:

| Row | Paper nnz | Reconstructed nnz | Difference |
|---|---:|---:|---:|
| T48 | 295,998,240 | 229,507,104 | -22.463% |
| T64 | 394,957,984 | 306,303,136 | -22.447% |
| T96 | 593,316,768 | 460,334,496 | -22.413% |
| T4 | 373,238,888 | 342,863,272 | -8.138% |
| T6 | 559,872,262 | 514,308,838 | -8.138% |

The public MATPOWER networks and deterministic reconstruction are pinned, but
the authors' resource placements, time series, physical modifications, exact
matrix construction, CUDA source, and compatible timing boundary remain
unavailable. These results do not reproduce the authors' hidden instances or
their A100 timings.

## 9. Independent checker

The independent Stage 8 check routine passed **12 of 12** checks. It verified:

- the frozen Stage 8 and accepted Stage 7 identities;
- all 18 preallocation estimates;
- the T24 and T32 signed-int32 ledger blocks without allocation;
- one-case-per-invocation strict-prefix execution;
- honest preservation of successful and failed terminal evidence;
- `STOPPED_ON_FAILURE` with a passing prefix of four;
- separate timing boundaries and no unsupported speedup;
- zero Stage 9 or N-1 allocation; and
- preservation of the single terminal failure record.

The checker result is `PASS` because the evidence obeyed the protocol. The
campaign result remains `STOPPED_ON_FAILURE`, and the Stage 8 acceptance result
remains `FAIL` because T6 did not complete its required CPU track.

The independent GPU-only continuation checker passed **13 of 13** checks. It
also replayed the original 12-check audit and verified that the original
configuration, evidence, checker output, T6 failure, and terminal status were
byte-for-byte preserved. It then independently recomputed the three resource
projections and the T16 live gate, verified the two static signed-int32 blocks,
confirmed zero continuation allocations, proved that CPU and Gurobi were never
called, and confirmed that Stage 9 remained locked.

An initial rehearsal from commit `79d10069089f29e9bca1649db58fdb8ab8c3488a`
reached the same zero-allocation resource decisions, but its checker wrapper
could not serialize an evidence path outside the repository. The wrapper was
fixed without changing any validation rule. The complete plan, continuation,
and checker were then rerun from fresh clean commit `1cf9da62...`; only that
rerun is the official continuation evidence.

The initial checker command-line wrapper contained a result-writer defect: its
final call referenced a missing `stage7_checker._atomic_write_json` helper.
The writer was repaired without changing the 12 validation checks, and the
official `stage_8_checks.json` was then produced. This serialization repair
does not alter the campaign evidence or the Stage 9 decision.

## 10. Evidence inventory

- `configs/benchmarks/stage_8_large.json`: frozen Stage 8 contract
- `results/raw/stage_8/stage_8_validation.json`: terminal campaign evidence
- `results/raw/stage_8/stage_8_checks.json`: independent 12-check result
- `configs/benchmarks/stage_8_gpu_only_completion.json`: frozen continuation
  contract
- `results/raw/stage_8/gpu_only_completion/stage_8_gpu_only_completion_validation.json`:
  terminal continuation evidence
- `results/raw/stage_8/gpu_only_completion/stage_8_gpu_only_completion_checks.json`:
  DGX 13-check result
- `results/raw/stage_8/gpu_only_completion/stage_8_gpu_only_completion_checks_local.json`:
  independent local replay of the same 13 checks
- `results/raw/stage_8/gpu_only_completion/stage_8_gpu_only_completion.partial.json`:
  final atomic checkpoint, identical to the terminal evidence
- `results/raw/stage_8/runs/stage8-f1fffc2adcba-20260803T225116Z/`: immutable
  invocation, PID, partial-checkpoint, and final-run archive
- `docs/stage_reports/stage_8_report.md`: this human-readable report

## 11. Final decision and next gate

Stage 8 demonstrated that four larger structural reconstructions pass the
frozen HiGHS, CPU FP64, GPU FP64, numerical, memory, timing, and provenance
protocol. It also demonstrated a successful T6 HiGHS and GPU track. It did not
demonstrate a complete T6 row because the required CPU correctness attempt hit
the frozen deadline.

Accordingly:

- campaign protocol and evidence integrity: **PASS**;
- all-cases numerical campaign: **FAIL**;
- Stage 8 acceptance: **FAIL**;
- Stage 9 authorization condition: **NOT SATISFIED**;
- Stage 9 state: **LOCKED**.

The GPU-only continuation additionally establishes:

- sequences 6--8 requested scope: **RESOLVED**;
- continuation allocations: **0**;
- new HiGHS/GPU timing evidence: **NONE (preallocation guards blocked all rows)**;
- continuation evidence integrity: **PASS, 13/13**;
- original Stage 8 result changed: **NO**;
- Stage 9 work started: **NO**.

No automatic retry is permitted. Any future T6 retry or protocol revision must
be separately authorized and must preserve this terminal evidence unchanged.
