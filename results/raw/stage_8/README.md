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
