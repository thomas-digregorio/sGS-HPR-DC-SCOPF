# Stage 9 report

> **Status:** PASS
>
> **Date:** August 7, 2026
>
> **Final classification:** D - structural reproduction
>
> **Independent checker:** PASS, 17/17
>
> **Next stage:** Stage 10 locked

## Executive result

Stage 9 completed the requested final reproduction report as an IEEE-style
LaTeX scientific paper with detailed appendices. The archival Markdown report
contains all 16 workflow-required subjects. Deterministic scripts generate the
supporting CSV tables, SVG figures, evidence hashes, and machine-readable
result index directly from preserved Stage 0--8 JSON.

The revised release `reproduction-paper-v6` removes file-hash detail from the
reader-facing narrative, moves the CPU, GPU, and reference-solver definitions to
an appendix, presents the acceptance/timing design as a table, explains the
equality-solve lemma in plain language, and places the benchmark results before
the discussion. The conditioning and timing-dispersion tables now sit in the
Results section, and tighter float, display, and table spacing produces a denser
layout. It retains the instance recipe, combined three-method timing display,
scholarly references, and code/data statement.

The report concludes that the project is a **structural reproduction**. The
mathematics, corrected structural linear algebra, CPU oracle, resident FP64 GPU
path, public-network dimensions, validation gates, and benchmark protocol were
reproduced. The authors' modified numerical instances, sparse support, source
code, A100 environment, and exact timing boundary were not.

## Evidence synthesis

| Item | Result |
|---|---|
| Stage checker scopes | 10 |
| Stage 0--7 checker total | 120/120 PASS |
| Stage 8 protocol checker | 12/12 PASS; scientific stage FAIL |
| GPU-only continuation checker | 13/13 PASS |
| Symbolic Table II rows | 18 |
| Dimension matches | 18 |
| Sparse nnz matches | 0 |
| Allocated benchmark rows | 11 |
| Validated GPU rows | 11 |
| Validated CPU rows | 10 |
| Resource-resolved rows without allocation | 3 |

Stage 8 remains `STOPPED_ON_FAILURE` because `case9241pegase:T6` did
not complete its required CPU FP64 correctness track within 3,600 seconds.
The later continuation preserves that failure and records T16 as memory-blocked
and T24/T32 as signed-int32 index-blocked before allocation.

The resource labels retain their exact semantics. T16's 94.435 GiB projection
exceeded the sampled 65.784/65.496 GiB live budgets but not the 97.352 GiB
nominal 80% reference. T24's conservative 2,531,600,260 planning count exceeds
signed int32 while its exact 2,057,650,132 reconstruction does not. T32 exceeds
the limit under both its planning and exact reconstructed counts.

## Created artifacts

- `docs/final_reproduction_report.tex`
- `docs/final_reproduction_report.md`
- `output/pdf/final_reproduction_report.pdf`
- `docs/reproducibility_checklist.md`
- `docs/regeneration_commands.md`
- `docs/stage_9_contract.md`
- `docs/stage_9_source_notes.md`
- `results/stage_9_result_index.json`
- `results/tables/stage_9_*.csv`
- `results/plots/stage_9_*.svg`
- `scripts/generate_stage_9_artifacts.py`
- `scripts/check_stage_9.py`
- `results/raw/stage_9/stage_9_checks.json`
- `dashboard/public/og-stage9.png`

## Quality gates

- The LaTeX source compiles with `pdflatex`.
- The compiled PDF is rendered to page images and visually inspected.
- High-impact counts, timings, timeouts, memory budgets, and index limits are
  independently reconciled against immutable JSON.
- All Stage 9 generated outputs reproduce byte-for-byte.
- Python tests pass 308/308, with one expected local CuPy-version skip; Ruff
  check and format validation pass across 112 files; dashboard build, two
  rendered-HTML tests, and lint pass.
- The dashboard marks Stage 9 complete and Stage 10 locked.

The dashboard's production build is ready, but the persisted private Sites
project ID returned `project not found` during publication. No replacement or
duplicate site was created; the validated dashboard source remains committed
for recovery of the existing binding.

## Stage boundary

Stage 9 is the terminal paper-reproduction stage. No benchmark was rerun, no
threshold was weakened, and no N-1 work was started. Further work requires a
new authorization and must not revise this final evidence boundary.
