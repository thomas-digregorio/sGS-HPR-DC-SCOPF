# Project state

Last updated: 2026-08-05

## Stage gate

- Completed stage: **Stage 9 - final reproduction report**
- Final classification: **D - structural reproduction**
- Stage 9 checker: **PASS, 17/17 checks**
- Stages 0--7: **PASS**
- Stage 8 scientific acceptance: **FAIL**
- Stage 8 campaign: **`STOPPED_ON_FAILURE`**
- Stage 8 protocol checker: **PASS, 12/12 checks**
- GPU-only sequences 6--8 continuation:
  **`COMPLETE_WITH_RESOURCE_LIMITS`**
- Continuation checker: **PASS, 13/13 checks**
- Current gate: **Stage 10 locked**
- N-1 work performed: **No**

The final report is available as archival Markdown, LaTeX source, and a
compiled PDF. Every quantitative report table is generated from preserved
machine-readable evidence and indexed by SHA-256.

The Stage 9 dashboard source, rendered-output tests, and social preview pass
locally. Production publication is pending recovery of the persisted private
Sites project binding, which currently returns `project not found`; no
duplicate site was created.

## Final scientific result

The project reproduced the paper's mathematical and implementation structure:

- canonical multi-period DCOPF LP and sign conventions;
- Equation (28) KKT mapping and distinct Equation (54) stopping blocks;
- paper-order z, x, y1, y2, y1, reflection, and Halpern updates;
- corrected structural equality solve, checked against direct FP64 systems;
- reversible preconditioning and sourced control-policy reconstruction;
- FP64 CPU oracle and resident FP64 GPU implementation;
- observed cuSPARSE CSR ALG2 selection and transfer auditing; and
- original-space numerical and power-system validation.

All 18 reconstructed Table II rows match the publication's row and variable
dimensions. None matches the paper's sparse nonzero count. The authors'
modified numerical instances, placements, profiles, parameters, source code,
hardware, and exact timing boundary remain unavailable. Those facts exclude
exact and near-exact reproduction.

## Benchmark coverage

| Evidence class | Count | Outcome |
|---|---:|---|
| Symbolic Table II rows | 18 | 18 dimension matches; 0 nnz matches |
| Allocated Stage 7--8 rows | 11 | all 11 GPU FP64 candidates passed |
| Accepted CPU FP64 rows | 10 | T6 CPU correctness timed out |
| Fully passing Stage 7 rows | 6 | PASS |
| Fully passing Stage 8 rows | 4 | passing prefix before T6 failure |
| Resource-resolved rows without allocation | 3 | T16 memory; T24/T32 index |

Each accepted timing track used one correctness run, one warm-up, and five
measured repetitions. No local or paper speedup is claimed because the timing
boundaries, sparse workloads, and hardware are not controlled equivalents.

## Stage 8 terminal boundary

The strict campaign attempted five rows in order:

| Row | HiGHS | CPU FP64 sGS-HPR | GPU FP64 sGS-HPR | Final status |
|---|---|---|---|---|
| `case2868rte:T48` | PASS | PASS | PASS | PASS |
| `case2868rte:T64` | PASS | PASS | PASS | PASS |
| `case2868rte:T96` | PASS | PASS | PASS | PASS |
| `case9241pegase:T4` | PASS | PASS | PASS | PASS |
| `case9241pegase:T6` | PASS | `TIME_LIMIT` | PASS | FAIL |

T6's required CPU correctness attempt exceeded the frozen 3,600-second limit
after `3,600.092739 s`. It produced no accepted CPU candidate, warm-up, or
timing median. No retry was attempted.

The separately authorized GPU-only continuation then resolved:

- T16 as `MEMORY_BLOCKED`: 94.435 GiB projected versus 65.784 GiB host and
  65.496 GiB CUDA live budgets;
- T24 as `INDEX_BLOCKED`: 2,531,600,260 planning nnz; and
- T32 as `INDEX_BLOCKED`: 3,375,704,460 planning nnz.

T24 and T32 exceed the signed-int32 maximum 2,147,483,647. All three decisions
occurred before LP or solver allocation.

## Evidence and report artifacts

- `docs/final_reproduction_report.md`: archival report
- `docs/final_reproduction_report.tex`: scientific-paper source
- `output/pdf/final_reproduction_report.pdf`: compiled reader artifact
- `docs/reproducibility_checklist.md`: reproducibility audit
- `docs/regeneration_commands.md`: table, PDF, checker, test, and Git commands
- `results/stage_9_result_index.json`: machine-readable classification,
  coverage, environment, gates, and hashes
- `results/tables/stage_9_*.csv`: deterministic report tables
- `results/plots/stage_9_*.svg`: deterministic report figures
- `results/raw/stage_9/stage_9_checks.json`: independent Stage 9 result
- `docs/stage_reports/stage_9_report.md`: stage handoff

The original Stage 8 evidence SHA-256 remains
`f4197554d8f7e108a4ca2701cbb0b12a485d593febb76f56f431fb482af14254`.
The continuation evidence SHA-256 remains
`edf18f6cda959c47fe5d7c38370c5f88619ee08a8ee4f9dbb480756ff1d34f7b`.

## Next action

The paper reproduction is complete at Stage 9. The recommended next step is to
seek author-instance artifacts and reconcile exact sparse fingerprints on
small cases before attempting a stronger reproduction classification.

Stage 10 is an optional N-1 SCOPF research extension. It was not started and
requires separate explicit approval.
