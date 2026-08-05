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
  measured repetitions.
- [x] Frozen normalized, raw KKT, physical, objective, and time gates retained.
- [x] Stage 7 reports six fully validated public-network rows.
- [x] Stage 8 reports four fully passing rows and a T6 CPU timeout.
- [x] T16 memory block and T24/T32 index blocks occurred before allocation.
- [x] No unsupported CPU/GPU or paper speedup is claimed.

## Reporting and audit

- [x] Final classification uses the frozen A--E vocabulary.
- [x] Machine-readable result index includes evidence hashes.
- [x] Every final table has a regeneration command.
- [x] LaTeX source and archival Markdown report agree on material conclusions.
- [x] Stage 9 independent checker passes, 17/17.
- [x] Full Python tests, Ruff, dashboard build/test/lint pass: 307 tests passed,
  one local CuPy-version skip, 112 files satisfy Ruff formatting, and both
  dashboard rendering tests pass.
- [x] Final six-page PDF is rendered page-by-page and visually inspected with
  no clipped or overfull content.
- [x] Final Git worktree is clean after commit and push; the completion handoff
  records the resulting commit because a commit cannot embed its own identifier.

## Locked follow-up

- [x] Stage 10 remains locked.
- [x] No contingency model, LODF screen, or N-1 constraint generation was run
  in Stage 9.
