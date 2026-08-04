# Project state

Last updated: 2026-08-04

## Stage gate

- Completed terminal campaign: **Stage 8 - large paper-scale benchmarks**
- Campaign status: **`STOPPED_ON_FAILURE`**
- Stage 8 acceptance: **FAIL**
- Independent protocol checker: **PASS, 12/12 checks**
- Current state: **Stage 9 locked**
- Automatic retry: **not permitted**
- Dashboard: private, owner-only Sites deployment; its URL and project
  identifier are intentionally omitted from the public repository.

The checker PASS and Stage 8 FAIL are intentionally separate decisions. The
checker confirms that the campaign obeyed its frozen order, provenance,
resource, timing, and failure-preservation rules. Stage 8 fails acceptance
because `case9241pegase:T6` did not complete its required CPU FP64 track.

## Stage 8 outcome

The campaign attempted five rows in strict order on the DGX Spark:

| Row | HiGHS | CPU FP64 sGS-HPR | GPU FP64 sGS-HPR | Final status |
|---|---|---|---|---|
| `case2868rte:T48` | PASS | PASS | PASS | PASS |
| `case2868rte:T64` | PASS | PASS | PASS | PASS |
| `case2868rte:T96` | PASS | PASS | PASS | PASS |
| `case9241pegase:T4` | PASS | PASS | PASS | PASS |
| `case9241pegase:T6` | PASS | `TIME_LIMIT` | PASS | FAIL |

T6's CPU correctness attempt exceeded the frozen 3,600-second limit after
`3,600.092739 s`. It produced no accepted CPU candidate, warm-up, or measured
CPU timing. No retry was attempted. T16, T24, and T32 were not executed, and
Stage 9 received zero allocations.

The terminal evidence records five unique allocation attempts, a passing
prefix of four, no retries, no reconciliation-only allocations, and no N-1
extension work.

## Successful timing tracks

Each completed timing track includes a separate correctness run, one warm-up,
and five measured repetitions. Solver-core medians are:

| Row | HiGHS median s | CPU median s | GPU median s |
|---|---:|---:|---:|
| T48 | 76.239 | 804.863 | 49.968 |
| T64 | 108.232 | 1,078.892 | 68.383 |
| T96 | 163.593 | 1,621.905 | 105.022 |
| T4 | 142.479 | 3,087.218 | 193.632 |
| T6 | 963.957 | Not available | 357.544 |

No speedup is claimed. Local CPU, GPU, complete-case, and paper timing
boundaries remain separate, the reconstructed sparse workloads differ from
Table II, and the DGX Spark GB10 is not the paper's A100.

## Numerical boundary

All completed candidates passed their frozen objective, stopping-residual,
raw-KKT, and physical-validation gates. The largest accepted values among the
completed Stage 8 candidates were:

| Metric | Maximum | Gate |
|---|---:|---:|
| Normalized primal block | `3.010269e-9` | `5e-5` |
| Normalized stationarity block | `9.659299e-6` | `5e-5` |
| Normalized box block | `8.830567e-13` | `5e-5` |
| Raw KKT norm | `0.0093260531` | `0.01` |
| Physical violation | `0.0063110950 MW/MWh` | `0.01 MW/MWh` |
| Scaled objective gap to HiGHS | `1.202323e-8` | `2e-4` |

These passing candidates do not override T6's missing required CPU result.

## Memory and execution boundary

All five reached rows passed the frozen live unified-memory gate before full LP
allocation. The projected peaks ranged from 23.606 to 48.303 GiB for the four
passing rows and were 35.410 GiB for T6. The corresponding live 80% host and
device budgets remained at least 79.124 GiB.

T16 has a static planning projection of 94.435 GiB, but its live gate was never
evaluated because the campaign stopped at T6. T24 and T32 have signed-int32
static block entries in the resource ledger; neither row was reached or
allocated. The maximum recorded host high-water mark was a cumulative
process-lifetime value, not an isolated per-solve peak.

## Reproduction classification

Stage 8 remains a **structural reproduction**, not an exact author-instance or
paper-timing reproduction. The pinned public MATPOWER networks and frozen
deterministic reconstruction reproduce every published row and variable
dimension, but every sparse nonzero count differs from the paper. The authors'
resource placements, time series, physical modifications, exact construction
code, CUDA implementation, and compatible timing boundary remain unavailable.

The successful large rows validate this repository's reconstruction on the
DGX Spark. They do not recover the hidden paper instances or reproduce the
paper's A100 timings.

## Provenance and checker

The campaign ran from clean detached commit
`f1fffc2adcba197040578695ba11dd27b0d1981f` with run fingerprint
`4dcb61115fb60f49c5972839ac0c99585cf74de2d3d07b955e142bb4f0f8e7cd`.
The 27-entry executed-source manifest, accepted Stage 7 identities, Stage 8
configuration, and frozen requirements all passed preflight.

The independent check routine passed 12/12 checks and preserved campaign
status `STOPPED_ON_FAILURE`. Its initial command-line wrapper had a final JSON-
writer reference defect. The writer was repaired without changing validation
logic, and the official checks JSON was then produced. This affected reporting
mechanics, not the checker result or Stage 8 acceptance decision.

## Evidence

- `results/raw/stage_8/stage_8_validation.json`: terminal Stage 8 evidence
- `results/raw/stage_8/stage_8_checks.json`: independent 12-check result
- `results/raw/stage_8/runs/stage8-f1fffc2adcba-20260803T225116Z/`: immutable
  run archive
- `configs/benchmarks/stage_8_large.json`: frozen Stage 8 contract
- `docs/stage_reports/stage_8_report.md`: detailed human-readable report

The final evidence SHA-256 is
`f4197554d8f7e108a4ca2701cbb0b12a485d593febb76f56f431fb482af14254`.
The checker-output SHA-256 is
`ff8f48af5c9f2866eb1a461776a3128e87100dc75bc99618671e63da5e94672b`.

## Next action

The user's conditional Stage 9 approval required Stage 8 to look good after
independent validation. That condition was not met because T6's CPU track hit
the frozen time limit. Stage 9 therefore remains locked.

No automatic retry is allowed. Any T6 retry, deadline change, or revised Stage
8 contract requires a new explicit decision and must preserve this failed
campaign unchanged.
