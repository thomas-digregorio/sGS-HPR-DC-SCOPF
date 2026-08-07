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

The internal A--E decision history remains in supporting evidence but is not
part of the concise reader-facing narrative. The revised paper identifies its
author as an independent researcher, provides a code-and-data statement, cites
the external numerical software and public-case sources, and is released as
Git tag `reproduction-paper-v6`.

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
- Data: medians after a correctness run and warm-up, with five measured
  repeats unless the frozen variability rule escalated the track to nine.
- Encoding: one shared logarithmic y-axis, case/horizon on x, and a small
  horizontal offset for each method within every reconstruction. Circles,
  open squares, and triangles distinguish HiGHS, CPU FP64, and GPU FP64
  without relying on color alone. Markers are medians with observed
  minimum--maximum whiskers; T6 CPU is one censored correctness attempt at
  3,600 seconds rather than a median.
- Required caveat: boundaries differ by solver and are not a controlled
  paper-speedup comparison.

### Figure 2 - fail-closed resource boundaries

- Question: why were the final three authorized Stage 8 rows not allocated?
- Data: T16 projected unified memory, two live 80% budgets, and the nominal
  128-GB-derived reference; T24/T32 conservative planning counts, exact
  reconstructed counts, and the signed-int32 maximum.
- Encoding: horizontal magnitude bars; measured/derived blockers highlighted.
- Required caveat: these are preallocation decisions, not out-of-memory
  crashes and not solver failures. T16 exceeds the live budgets but is below
  the nominal 80% reference. T24's planning count exceeds signed int32 while
  its exact reconstructed count does not; T32 exceeds the limit under both
  counts.

## Algebra contract

- The structural solve is stated as a lemma with a Sherman--Morrison proof and
  a mean/zero-mean interpretation.
- The source-paper sign gives median relative error 0.206 on the analytic
  family; the corrected sign gives maximum relative error 3.03e-16.
- Conditioning evidence spans condition numbers 1 through 25,616.42 and is
  reported separately from the algebraic proof.

## Stage 9 exclusions

- No new LP construction or solver run.
- No post-hoc threshold change.
- No conversion of checker PASS into Stage 8 PASS.
- No performance ratio against the paper.
- No Stage 10 contingency or N-1 work.
