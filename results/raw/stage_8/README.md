# Stage 8 evidence

Stage 8 reached an honest terminal runner state of `STOPPED_ON_FAILURE`.
The independent checker passed all 12 protocol and evidence-integrity checks,
but the campaign did not meet the gate for Stage 9 because
`case9241pegase:T6` failed its CPU correctness track at the frozen 3,600-second
time limit.

- Executed commit: `f1fffc2adcba197040578695ba11dd27b0d1981f`
- Final run: `runs/stage8-f1fffc2adcba-20260803T225116Z/`
- Passing campaign prefix: 4 cases
- Allocated cases: 5
- Terminal case: `case9241pegase:T6`
- T6 CPU result: `TIME_LIMIT` after `3600.092738645966` seconds
- T6 GPU result: `PASS`, five measured runs
- T6 HiGHS result: `PASS`, five measured runs
- Stage 9: locked

The root `stage_8_validation.json` and `stage_8_checks.json` are convenience
copies of the terminal evidence and its independent checker result.

| File | SHA-256 |
|---|---|
| `stage_8_validation.json` | `f4197554d8f7e108a4ca2701cbb0b12a485d593febb76f56f431fb482af14254` |
| `stage_8_checks.json` | `ff8f48af5c9f2866eb1a461776a3128e87100dc75bc99618671e63da5e94672b` |

The frozen checker CLI initially completed its checks but could not write its
output because the executed copy of `scripts/check_stage_8.py` called a missing
`check_stage_7._atomic_write_json` helper. The unchanged checker
`run_checks` function was therefore invoked directly; it returned
`checker_status: PASS`, `all_passed: true`, and 12 of 12 passing checks. No
source, threshold, or acceptance rule was changed.
The terminal publication commit fixes only that output-writer import and adds a
regression test; the archived campaign evidence remains tied to the clean
executed commit above.

## GPU-only sequences 6--8 continuation

On August 5, 2026, a separately frozen continuation resolved the three rows
that followed the original T6 stop. Only HiGHS and GPU FP64 sGS-HPR were
eligible if a row passed preallocation; CPU sGS-HPR and Gurobi were explicit
non-gating skips. The original terminal campaign above was not retried or
modified.

- Executed commit: `1cf9da62e263a1fb8cc7e68e6cecc4958e602a22`
- Continuation status: `COMPLETE_WITH_RESOURCE_LIMITS`
- T16: `MEMORY_BLOCKED` before allocation
- T24 and T32: `INDEX_BLOCKED` before allocation
- Full LP allocations: 0
- New HiGHS/GPU timings: none
- Independent continuation checker: PASS, 13/13
- Stage 9: locked, zero allocations

| File | SHA-256 |
|---|---|
| `gpu_only_completion/stage_8_gpu_only_completion_validation.json` | `edf18f6cda959c47fe5d7c38370c5f88619ee08a8ee4f9dbb480756ff1d34f7b` |
| `gpu_only_completion/stage_8_gpu_only_completion.partial.json` | `edf18f6cda959c47fe5d7c38370c5f88619ee08a8ee4f9dbb480756ff1d34f7b` |
| `gpu_only_completion/stage_8_gpu_only_completion_checks.json` | `8cda412285568b2685abc58d6571132f7e83123f9158500d3efd1c1ec976fa65` |
| `gpu_only_completion/stage_8_gpu_only_completion_checks_local.json` | `a3365f9b17ebefd538f0da8e1521c445f6c1527334f2d25e27958f29c18821ec` |

The T16 gate used the unchanged 94.435 GiB unified projection and 80% safety
fractions. Its observed host and CUDA-free budgets were 65.784 GiB and 65.496
GiB. T24 and T32 exceeded the signed-int32 conservative planning limit. All
three decisions were recorded before LP allocation.
