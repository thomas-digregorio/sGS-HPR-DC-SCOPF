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
decision. Stage 5 is approved and Stage 6 is complete. To start Stage 7 after
reviewing the GPU evidence, send exactly:

```text
APPROVE STAGE 6 AND RUN STAGE 7
```

Casual phrases such as "continue" or "looks good" are not stage approval.

## Current status

Stage 6 is complete and the project is stopped at the Stage 7 gate. The frozen
FP64 CPU and DGX GPU paths reached identical stopping iterations on both
correctness cases:

| Case | CPU / GPU iterations | GPU raw KKT | GPU objective | Maximum final relative state error |
|---|---:|---:|---:|---:|
| Public case5, T=1 | 410 / 410 | 0.005618456235 | 17479.839088898956 | 2.62e-14 |
| Synthetic resource fixture, T=2 | 1,032 / 1,032 | 0.008948422297 | 26580.274984099353 | 6.90e-15 |

These are implementation-parity results, not performance benchmarks. Stage 6
makes no CPU/GPU speedup claim.

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
  diagnostic.

See `docs/project_state.md` for the authoritative gate state and
`docs/stage_reports/stage_6_report.md` for acceptance evidence.

## Environment snapshot

The Stage 5 CPU reference ran locally with Python 3.13.5, NumPy 2.4.1, SciPy
1.16.3, SciPy's bundled HiGHS dual-simplex interface, pytest, and Ruff.

Stage 6 ran in an isolated virtual environment on the target DGX Spark:

- Ubuntu 24.04.4 LTS, aarch64;
- NVIDIA GB10, compute capability 12.1;
- NVIDIA driver 580.173.02, CUDA driver API 13.0, CuPy CUDA runtime API 13.2,
  and CUDA 13.0.3 toolkit;
- 121.690 GiB system memory with ATS addressing;
- CPython 3.12.3;
- CuPy 14.1.1 for CUDA 13, NumPy 2.3.5, and SciPy 1.16.3.

Raw machine inventories and access details remain local and are intentionally
excluded from the public repository. See `environment/README.md` for the
regeneration and privacy policy.

## Reproduction fidelity

Exact paper data are not yet known to be available. In particular, the
manuscript does not fully identify device placements, time series, several
physical parameters, the exact adaptive-penalty and restart implementation,
numerical precision, or timing boundaries. Stage 5 therefore pins the
published HPR-LP v0.1.0 policy as a sourced reconstruction and does not claim
that it is identical to the authors' unpublished DCOPF code. Stage 6 also runs
on a DGX Spark GB10 rather than the paper's A100. Its timing ledger defines this
project's measurement boundary; it does not retroactively make that boundary
identical to the unpublished paper experiment.

Until the required inputs are recovered, this project will not claim exact
numerical reproduction. Results will be labeled as exact, mathematical,
structural, or approximate benchmark reconstruction according to the evidence
actually available.

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

## Stage 6 checks

From this directory:

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/ruff.exe format --check .
./.venv/Scripts/python.exe scripts/check_stage_6.py
```

The Stage 6 experiment itself runs on the DGX Spark from the repository-local
environment:

```text
python scripts/run_stage_6.py
```

The dashboard has its own production build and rendered-output test:

```powershell
cd dashboard
npm run build
node --test tests/rendered-html.test.mjs
```

The checker validates the preserved Stage 6 evidence without starting Stage 7.
The experiment command records DGX data and should be run only in the prepared
CUDA 13 environment.

## Research rules in one minute

1. Build the correct CPU method before GPU optimization.
2. Use FP64 for the correctness baseline.
3. Compare against HiGHS and validate physical constraints.
4. Do not infer missing algorithms or invent missing experimental data.
5. Do not compare incompatible timing boundaries.
6. Preserve the fixed-sigma baseline when adaptive features are added.
7. Keep the optional N-1 extension separate from paper-reproduction results.
