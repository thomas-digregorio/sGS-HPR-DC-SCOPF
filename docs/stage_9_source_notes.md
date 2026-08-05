# Stage 9 source notes

These notes define the report spine and make the origin of each major claim
reviewable before prose is typeset.

## Primary conclusion

The final classification is **D - structural reproduction**. The canonical
mathematics, paper-order algorithm, corrected structural equality solve, CPU
oracle, resident FP64 GPU implementation, and public-network dimensions were
reproduced and independently checked. The authors' numerical instances,
sparse supports, code, A100 hardware, and timing boundary were not available,
and the frozen large-scale campaign retained an honest Stage 8 failure.

## Claim map

| Claim family | Authoritative source |
|---|---|
| Paper identity and PDF hash | `results/raw/stage_0/paper_metadata.json` |
| Equations, dimensions, Table II | `docs/paper_specification.md` plus supplied PDF |
| Ambiguities and missing inputs | `docs/reproduction_limits.md` |
| Toy LP and residual mechanics | Stage 1 evidence and report |
| DCOPF construction and physical checks | Stage 2 evidence and report |
| CPU paper-order solver | Stage 3 evidence and report |
| Equality-solve correction | Stage 4 evidence and report |
| Scaling and policy reconstruction | Stage 5 evidence and report |
| GPU kernel, residency, FP64 parity | Stage 6 evidence and report |
| Public-network structural campaign | `stage_7_validation.json` |
| Large-scale timings and T6 timeout | `stage_8_validation.json` |
| T16/T24/T32 resource decisions | GPU-only continuation JSON |
| Final derived tables and hashes | `results/stage_9_result_index.json` |

## Figure contracts

### Figure 1 - local validated solver-core time

- Question: how did locally measured HiGHS, CPU FP64, and GPU FP64 solver-core
  time vary across allocated Stage 7--8 reconstructions?
- Data: five-repeat medians after a correctness run and warm-up.
- Encoding: case/horizon on x, seconds on logarithmic y, distinct color and
  marker by track; T6 CPU timeout is an explicit cross at 3,600 seconds.
- Required caveat: boundaries differ by solver and are not a controlled
  paper-speedup comparison.

### Figure 2 - fail-closed resource boundaries

- Question: why were the final three authorized Stage 8 rows not allocated?
- Data: T16 projected unified memory and two live 80% budgets; T24/T32
  conservative planning nnz and signed-int32 maximum.
- Encoding: horizontal magnitude bars; measured/derived blockers highlighted.
- Required caveat: these are preallocation decisions, not out-of-memory
  crashes and not solver failures.

## Stage 9 exclusions

- No new LP construction or solver run.
- No post-hoc threshold change.
- No conversion of checker PASS into Stage 8 PASS.
- No performance ratio against the paper.
- No Stage 10 contingency or N-1 work.
