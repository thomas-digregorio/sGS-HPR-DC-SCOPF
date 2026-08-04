# Terminal Stage 8 DGX campaign

- Executed commit: `f1fffc2adcba197040578695ba11dd27b0d1981f`
- DGX device: NVIDIA GB10
- Started: 2026-08-03 22:51:58 UTC
- Completed: 2026-08-04 22:26:47 UTC
- Runner result: `STOPPED_ON_FAILURE`
- Checker result: `PASS` (12 of 12 protocol and evidence checks)
- Passing campaign prefix: 4 of 8 planned keys
- Full allocation attempts: 5
- Stage 9 allocations: 0

The terminal case was `case9241pegase:T6`. Its HiGHS and GPU tracks passed,
but the CPU correctness run reached the frozen 3,600-second time limit. The
runner consequently omitted CPU warm-up and measured timing, recorded the case
as `FAIL`, stopped before T16, and kept Stage 9 locked. No retry was attempted.

The final and partial validation files are byte-identical because the runner
atomically checkpointed terminal evidence before writing the final copy.
`invocation-06.log` and `invocation-07.log` are empty because structured
progress and failures are stored in JSON. `invocation-05.log` preserves a
pre-run command-line quoting error; it caused no allocation and was superseded
by the successful uniquely numbered invocation 06.

| File | SHA-256 |
|---|---|
| `stage_8_validation.json` | `f4197554d8f7e108a4ca2701cbb0b12a485d593febb76f56f431fb482af14254` |
| `stage_8_validation.partial.json` | `f4197554d8f7e108a4ca2701cbb0b12a485d593febb76f56f431fb482af14254` |
| `stage_8_checks.json` | `ff8f48af5c9f2866eb1a461776a3128e87100dc75bc99618671e63da5e94672b` |
| `invocation-05.log` | `3bb8026b045ba7f8603e7d2f1aa41b6174f77a513d3f6ad42fa5807a059047ec` |
| `invocation-06.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `invocation-07.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
