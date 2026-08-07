# Reproducibility checklist

## Source identity

- [x] Supplied 17-page paper preserved and fingerprinted.
- [x] Paper SHA-256 recorded as
  `7e9791646401e11bfddf9ebed6bd94491ed0b592744581edd851ddbf5e20dba4`.
- [x] Public MATPOWER inputs pinned with content hashes.
- [x] Stage 7 and Stage 8 executed-source manifests preserved.
- [x] Strict Stage 8 and GPU-only continuation evidence preserved unchanged.

## Mathematical reproduction

- [x] Canonical LP signs, row families, variable order, and dimensions recorded.
- [x] Equation (28) KKT mapping implemented separately from Equation (54)
  stopping blocks.
- [x] Box and nonnegative projections checked on analytic examples.
- [x] Paper-order z, x, y1, y2, y1, reflection, and Halpern updates tested.
- [x] Structural equality solve checked against direct FP64 solves.
- [x] Printed rank-one inverse sign discrepancy disclosed and corrected.
- [x] Structural solve stated as a lemma with a mean/zero-mean derivation;
  printed-sign median error 0.206 and corrected-sign maximum error 3.03e-16
  reported with conditioning evidence.
- [x] Spectral proximal safeguard checked independently.

## Power-system model

- [x] PTDF construction checked against an independent angle solve.
- [x] Transformer phase-shift affine offsets retained.
- [x] Offline devices and zero thermal ratings handled explicitly.
- [x] Public base-network and synthetic storage fixtures kept separate.
- [x] Original-space canonical and physical violations checked for every
  accepted benchmark candidate.
- [x] Base-case DCOPF kept distinct from optional N-1 SCOPF.

## CPU and GPU implementations

- [x] Readable FP64 CPU oracle implemented first.
- [x] Ten Ruiz passes, Pock-Chambolle alpha one, and vector normalization
  preserved.
- [x] Adaptive penalty and restart provenance labeled as a sourced HPR-LP
  reconstruction rather than unpublished-author identity.
- [x] GPU FP64 trajectory matched the CPU oracle at 1, 10, and 100 iterations.
- [x] Low-level cuSPARSE CSR ALG2 path observed and probed.
- [x] Resident-loop transfer audit found no full-state host copy.
- [x] FP32 retained as a non-gating diagnostic.

## Benchmark protocol

- [x] Every accepted track completed one correctness run, one warm-up, and five
  measured repetitions, with nine repeats for tracks escalated by the frozen
  variability rule.
- [x] Frozen normalized, raw KKT, physical, objective, and time gates retained.
- [x] Stage 7 reports six fully validated public-network rows.
- [x] Stage 8 reports four fully passing rows and a T6 CPU timeout.
- [x] T16 memory block and T24/T32 index blocks occurred before allocation.
- [x] T16 is labeled as a live-budget decision below the nominal 80% memory
  reference; T24 planning and exact counts are distinguished; T32 exceeds the
  signed-int32 limit under both counts.
- [x] No unsupported CPU/GPU or paper speedup is claimed.

## Reporting and audit

- [x] The reader-facing paper states the structural-reproduction boundary
  directly without exposing internal stage or A--E workflow vocabulary.
- [x] Internal decision history remains preserved in machine-readable
  evidence rather than repeated in the scientific narrative.
- [x] The paper includes author affiliation/contact, scholarly references,
  exact solver settings, deterministic instance construction, timing
  boundaries, limitations, and code/data availability.
- [x] Solver timing uses one shared logarithmic axis with distinct marker
  shapes and horizontal method offsets; it reports median, minimum, maximum,
  IQR, standard deviation, repeat count, and T6 censoring semantics.
- [x] Machine-readable result index includes evidence hashes.
- [x] Every final table has a regeneration command.
- [x] LaTeX source and archival Markdown report agree on material conclusions.
- [x] Stage 9 independent checker passes, 17/17.
- [x] Full Python tests and Ruff pass: 308 tests passed, one local
  CuPy-version skip, and 112 files satisfy Ruff formatting.
- [x] Revised PDF is rendered page-by-page and visually inspected with no
  clipped or overfull content.
- [x] Revised report is committed, pushed, and tagged `reproduction-paper-v6`; the
  completion handoff records the resulting commit because a commit cannot
  embed its own identifier.

## Locked follow-up

- [x] Stage 10 remains locked.
- [x] No contingency model, LODF screen, or N-1 constraint generation was run
  in Stage 9.
