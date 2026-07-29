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
decision. To start Stage 3 after reviewing Stage 2, send exactly:

```text
APPROVE STAGE 2 AND RUN STAGE 3
```

Casual phrases such as "continue" or "looks good" are not stage approval.

## Current status

Stage 2 is complete and the project is stopped at the Stage 3 gate.

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
- independent physical validation of every printed constraint family.

See `docs/project_state.md` for the authoritative gate state and
`docs/stage_reports/stage_2_report.md` for acceptance evidence.

## Environment snapshot

Stage 2 ran locally with Python 3.13.5, NumPy 2.4.1, SciPy 1.16.3, SciPy's
bundled HiGHS dual-simplex interface, pytest, and Ruff. A project-local virtual
environment provides the quality tooling while inheriting the audited
scientific packages. Standalone `highspy` is not installed.

The target DGX Spark was audited and remained unchanged during Stage 2:

- Ubuntu 24.04.4 LTS, aarch64;
- NVIDIA GB10, compute capability 12.1;
- NVIDIA driver 580.173.02 and loadable CUDA runtime 13.0;
- 121.690 GiB system memory with ATS addressing;
- CPython 3.12.3;
- no `nvcc` and no installed scientific Python stack.

Stage 2 intentionally did not install packages or run solver code on the DGX.
Raw machine inventories and access details remain local and are intentionally
excluded from the public repository. See `environment/README.md` for the
regeneration and privacy policy.

## Reproduction fidelity

Exact paper data are not yet known to be available. In particular, the
manuscript does not fully identify device placements, time series, several
physical parameters, adaptive penalty and restart formulas, numerical
precision, or timing boundaries.

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

## Stage 2 checks

From this directory:

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m ruff format --check .
./.venv/Scripts/python.exe scripts/build_dcopf.py
./.venv/Scripts/python.exe scripts/check_stage_2.py --output results/raw/stage_2/stage_2_checks.json
```

The dashboard has its own production build and rendered-output test:

```powershell
cd dashboard
npm run build
node --test tests/rendered-html.test.mjs
```

These commands reproduce the local Stage 2 evidence; they do not begin Stage 3.

## Research rules in one minute

1. Build the correct CPU method before GPU optimization.
2. Use FP64 for the correctness baseline.
3. Compare against HiGHS and validate physical constraints.
4. Do not infer missing algorithms or invent missing experimental data.
5. Do not compare incompatible timing boundaries.
6. Preserve the fixed-sigma baseline when adaptive features are added.
7. Keep the optional N-1 extension separate from paper-reproduction results.
