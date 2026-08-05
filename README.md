# GPU DCOPF sGS-HPR reproduction

This repository is a staged reproduction of:

> Qi Wang et al., "An Efficient GPU-based Halpern Accelerating Algorithm for
> Large-scale DC Optimal Power Flow."

The long-term target is a validated Python implementation running on an NVIDIA
DGX Spark. Every equation, assumption, data dependency, and numerical claim is
turned into a testable research artifact before performance optimization.

## What this project is reproducing

The paper formulates a multi-period linear DC optimal power flow problem with:

- system-wide power balance;
- base-case shift-factor transmission limits;
- upward and downward spinning reserve;
- generator ramping;
- renewable output;
- energy storage;
- a linear objective;
- an sGS-decomposed Halpern Peaceman-Rachford algorithm.

The published model is not an explicit N-1 SCOPF formulation. It has no
contingency index, LODF rows, or post-outage flow limits. N-1 screening and
adaptive constraint generation are reserved for an optional research extension
after the paper reproduction is complete.

## Stage-gated workflow

Only one stage is executed at a time:

| Stage | Outcome |
|---:|---|
| 0 | Paper specification, repository, and environment audit |
| 1 | Generic toy LP and mathematical component tests |
| 2 | CPU DCOPF model construction |
| 3 | CPU sGS-HPR reference implementation |
| 4 | Paper-specific structural equality solve |
| 5 | Preconditioning, restart, and penalty management |
| 6 | GPU port for DGX Spark |
| 7 | Small and medium benchmark reproduction |
| 8 | Large paper-scale benchmarks |
| 9 | Final reproduction report |
| 10 | Optional N-1 SCOPF research extension |

Every stage ends with tests, a report, preserved evidence, and an acceptance
decision. Stage 8 reached a terminal `STOPPED_ON_FAILURE` state. Its evidence
protocol passed independent review, but Stage 8 itself failed because the T6
CPU correctness track exceeded the frozen per-solve deadline. Stage 9 remains
locked. A separately authorized GPU-only continuation later resolved sequences
6--8 at unchanged preallocation safety boundaries without modifying or
retrying that terminal campaign.

## Current status

Stage 8 is terminal with **FAIL** and campaign status
`STOPPED_ON_FAILURE`. Four large structural rows passed in order:
case2868rte at T=48, 64, and 96, followed by case9241pegase at T=4. The fifth
row, case9241pegase T=6, passed HiGHS and GPU FP64 sGS-HPR but its required CPU
FP64 correctness attempt hit the frozen 3,600-second limit. It has no accepted
CPU candidate or CPU timing median.

The independent Stage 8 checker passed **12/12** protocol and evidence checks.
That checker PASS verifies the honest terminal record; it does not convert the
T6 failure into a Stage 8 acceptance pass. Five unique rows were allocated,
the passing prefix is four, and no retry occurred.

The later GPU-only sequence 6--8 continuation passed its independent **13/13**
evidence audit. T16 was `MEMORY_BLOCKED` before allocation because its 94.435
GiB unified projection exceeded both live 80% budgets. T24 and T32 were
`INDEX_BLOCKED` before allocation because their planning nonzero counts exceed
signed-int32 CSR capacity. The continuation made zero allocations, ran no new
HiGHS or GPU solves, and explicitly skipped CPU sGS-HPR and Gurobi. Stage 9
still records zero allocations and remains locked.

All Table II rows still have exact row and variable dimensions while every
reconstructed sparse nonzero count differs from the paper. The result remains
a structural reproduction, not an exact instance or paper-timing
reproduction. No speedup is claimed.

Validated capabilities now include:

- a validated dense/sparse canonical LP representation;
- box, nonnegative-orthant, and mixed dual-set projections;
- separate implementations of Eq. (28) and Eq. (54);
- the required analytic LP with \(x=(0.4,0.6)\) and objective \(1.4\);
- a correctness-oriented FP64 implementation of generic Algorithm 1;
- independent SciPy HiGHS comparisons on four deterministic LPs;
- a safe numeric MATPOWER version-2 parser;
- affine PTDF construction validated against angle-based DC flow;
- deterministic variable indexing and traceable sparse DCOPF rows;
- a public one-period case5 model and a labeled two-period synthetic extension;
- independent physical validation of every printed constraint family;
- a fixed-sigma FP64 CPU implementation of the paper's Algorithm 2;
- accurate direct solves for both equality-multiplier sweeps;
- exact validation of the implemented Equation (55) row and column structure;
- a corrected, matrix-free Proposition 5 structural equality backend;
- direct-versus-structural randomized, trajectory, convergence, HiGHS, and
  physical cross-checks;
- cancellation-resistant treatment of near-degenerate storage structure;
- measured synthetic structural speedups and a disclosed empirical complexity
  study;
- a reversible sparse preprocessing pipeline with 10 simultaneous Ruiz
  iterations, one Pock-Chambolle step at alpha 1, and post-diagonal
  normalization of the complete `b` and `c` vectors;
- exact scaled/original state, LP-data, residual, and objective identities;
- a sourced HPR-LP restart policy with a forced first restart and policy checks
  every 100 iterations;
- an adaptive penalty update evaluated in the general sGS metric;
- fixed/adaptive and restart/no-restart control ablations;
- unscaled, normalization-only, Ruiz, and full-preconditioning ablations;
- three-way spectral cross-checks for the projected inequality update;
- Equation (54) checks on every intermediate iterate;
- HiGHS, KKT, objective, and physical comparisons on six cases;
- deterministic repeated trajectories with explicit timing boundaries;
- a NumPy/CuPy backend boundary with explicit transfer accounting;
- resident FP64 GPU matrices, transposes, scaling data, solver state, and
  reusable workspaces;
- a verified low-level cuSPARSE path that actually selects
  `CUSPARSE_SPMV_CSR_ALG2` on the DGX Spark GB10;
- sparse-operation, one-step, short-trajectory, and final-solution CPU/GPU
  cross-checks;
- synchronized initialization, warm-up, transfer, iteration, residual-check,
  recovery, and end-to-end timing fields, plus explicit allocation accounting;
- an FP64-first correctness gate followed by a separate non-gating FP32
  diagnostic;
- canonical Git-blob provenance for MATPOWER 8.1 case1354pegase,
  case2868rte, and case9241pegase;
- an exact 18-row symbolic dimension and sparse-nonzero ledger with Stage 8
  allocation locks;
- a generalized scaled block-arrow equality solve that avoids a dense Gram
  matrix on the Stage 7 cases;
- sparse-only spectral certification without materializing a normal matrix;
- repeated HiGHS, CPU FP64, and GPU FP64 timing with first-run, warm-up,
  measured-sample, variability, memory, and transfer evidence; and
- independent enforcement of the Stage 8 strict-prefix allocation order,
  unified-memory preflight, terminal-failure preservation, and locked Stage 9
  boundary.

See `docs/project_state.md` for the authoritative gate state and
`docs/stage_reports/stage_8_report.md` for the terminal Stage 8 evidence.

## Environment snapshot

The Stage 5 CPU reference ran locally with Python 3.13.5, NumPy 2.4.1, SciPy
1.16.3, SciPy's bundled HiGHS dual-simplex interface, pytest, and Ruff.

Stages 6 through 8 ran in an isolated virtual environment on the target DGX
Spark:

- Ubuntu 24.04.4 LTS, aarch64;
- NVIDIA GB10, compute capability 12.1;
- NVIDIA driver 580.173.02, CUDA driver API 13.0, CuPy CUDA runtime API 13.2,
  and CUDA 13.0.3 toolkit;
- 121.690 GiB system memory with ATS addressing;
- CPython 3.12.3;
- CuPy 14.1.1 for CUDA 13, NumPy 2.3.5, and SciPy 1.16.3.

Stage 7 executed from clean detached commit
`ff6f762a00463e4769861f6aaf6f6fbbad6cc8af`. Its configuration and package
requirements were verified over canonical LF Git-blob content before model
allocation. The required sparse GPU path selected
`CUSPARSE_SPMV_CSR_ALG2` for FP64 products.

Stage 8 executed from clean detached commit
`f1fffc2adcba197040578695ba11dd27b0d1981f`. Its 27-entry source manifest,
frozen configuration, inherited Stage 7 identities, and package requirements
passed preflight before each strict-prefix allocation.

The GPU-only sequence 6--8 continuation executed from clean detached commit
`1cf9da62e263a1fb8cc7e68e6cecc4958e602a22`. Its 34-entry source manifest and
13-check audit passed. All three rows stopped before allocation at the frozen
memory or sparse-index guards.

Raw machine inventories and access details remain local and are intentionally
excluded from the public repository. See `environment/README.md` for the
regeneration and privacy policy.

## Reproduction fidelity

Exact paper data are not yet known to be available. In particular, the
manuscript does not fully identify device placements, time series, several
physical parameters, the exact adaptive-penalty and restart implementation,
numerical precision, or timing boundaries. Stage 5 therefore pins the
published HPR-LP v0.1.0 policy as a sourced reconstruction and does not claim
that it is identical to the authors' unpublished DCOPF code. Stages 6 through
8 also run on a DGX Spark GB10 rather than the paper's A100.

Stage 7 pins the public MATPOWER 8.1 networks and freezes one transparent
reconstruction for the missing author additions. It reproduces every published
row and variable dimension, but none of the 18 sparse nonzero counts. Its
timing ledger defines this project's measurement boundaries; it does not make
those boundaries or sparse workloads identical to the unpublished paper
experiment.

Stage 8 preserves that classification. Its four passing large rows validate
the frozen structural reconstruction at larger scale; the T6 CPU time limit
prevents a complete Stage 8 pass. The later zero-allocation GPU-only
continuation does not change that decision, so Stage 9 remains locked.

Until the required inputs are recovered, this project will not claim exact
numerical reproduction. Stage 7 is explicitly labeled a structural
reproduction. Results continue to be classified as exact, mathematical,
structural, or approximate according to the evidence actually available.

## Repository map

```text
gpu-dcopf-hpr/
|-- references/         source paper
|-- docs/               paper specification, decisions, limits, and reports
|-- environment/        public package snapshot and local-audit guidance
|-- src/                production Python package
|-- tests/              unit, integration, and regression tests
|-- scripts/            inspection, extraction, and run entry points
|-- configs/            versioned experiment configurations
|-- data/               provenance-controlled inputs and generated instances
|-- results/            raw and processed numerical evidence
|-- logs/               human-readable stage logs
|-- artifacts/          exported research artifacts
`-- dashboard/          stage and task control room
```

Production solver logic belongs in `src/gpu_dcopf_hpr/`, not only in notebooks.
Raw validation outputs are preserved under `results/raw/stage_N/`.

## Stage 7 and Stage 8 checks

From this directory:

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/ruff.exe format --check .
./.venv/Scripts/python.exe scripts/check_stage_7.py
./.venv/Scripts/python.exe scripts/check_stage_8.py
```

The Stage 7 experiment itself runs on the DGX Spark from the pinned
repository-local environment:

```text
python scripts/run_stage_7.py --no-resume --device-id 0 --output-dir <outside-worktree-run-directory>
```

The dashboard has its own production build and rendered-output test:

```powershell
cd dashboard
npm run build
node --test tests/rendered-html.test.mjs
```

The checkers validate preserved evidence without starting a new experiment.
The Stage 8 check routine passed 12/12 checks while preserving campaign status
`STOPPED_ON_FAILURE`. Its initial command-line wrapper had a final JSON-writer
reference defect; the writer was repaired without changing validation logic,
and the official check result was preserved with the evidence. No retry or
later-stage allocation is implied by running a checker.

The accepted Stage 7 evidence remains tied to executed commit `ff6f762`. Later
integrity maintenance taught the checker to accept accurately scoped Linux
`getrusage`-only peak telemetry when `psutil` RSS is unavailable, and hardened
source preflight so a deleted tracked Python file fails closed. Those changes
altered neither the accepted evidence nor any numerical or timing threshold;
no benchmark rerun was required. The accepted 19/19 check is evaluated against
the exact accepted commit; a later head with changed execution-source files is
expected to fail source identity rather than inherit the old benchmark claim.
A clean detached `ff6f762` recheck passed 19/19 with the hardened checker.

## Research rules in one minute

1. Build the correct CPU method before GPU optimization.
2. Use FP64 for the correctness baseline.
3. Compare against HiGHS and validate physical constraints.
4. Do not infer missing algorithms or invent missing experimental data.
5. Do not compare incompatible timing boundaries.
6. Preserve the fixed-sigma baseline when adaptive features are added.
7. Keep the optional N-1 extension separate from paper-reproduction results.
