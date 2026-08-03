# Stage 7 preflight attempt

- Executed commit: `71706ef076114bb3c42480d367e735e0b367828c`
- DGX run directory: `stage7-71706ef07611-20260803T172803Z`
- Final status: `FAIL` during provenance preflight
- Benchmark allocation: none

The three expected Git blob IDs matched, but the configured SHA-256 values had
been computed from the Windows CRLF working-tree representation. The DGX Linux
checkout used the canonical LF Git-blob bytes, so the runner stopped before
MATPOWER loading, symbolic-ledger construction, model allocation, or any solver
execution.

The JSON files are retained unchanged as evidence of that fail-closed stop.
`run.log` is empty because the runner records structured failures in the JSON
evidence.
