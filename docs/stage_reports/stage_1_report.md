# Stage 1 report

> **Status:** PASS  
> **Date:** July 29, 2026  
> **Stage:** Generic toy LP and mathematical component tests

## 1. Objective

Implement and independently validate the mathematical building blocks of the
paper's generic Algorithm 1 on tiny FP64 linear programs before introducing
power-system structure or GPU concerns.

This stage intentionally excluded MATPOWER parsing, PTDF construction, DCOPF
matrix construction, paper-specific sGS sweeps, DGX execution, and GPU code.

## 2. Work performed

Implemented and validated the following canonical linear program:

```text
minimize    cᵀx

subject to  A₁x = b₁
            A₂x ≥ b₂
            l ≤ x ≤ u
```

Additional work:

- Accepted dense and SciPy sparse matrix blocks while preserving the
  distinction between equality and inequality rows.
- Implemented projections onto a box, the nonnegative orthant, and the mixed
  dual set. Equality multipliers are unrestricted; inequality multipliers are
  nonnegative.
- Implemented the complete Eq. (28) residual mapping and the separate Eq. (54)
  feasibility and stopping blocks.
- Implemented pure reflection and fixed-anchor Halpern updates.
- Implemented generic Algorithm 1 using a verified spectral proximal operator.
- Added the required analytic LP and three reference LPs: box-bound active,
  inequality inactive, and seeded planted random.
- Added an independent SciPy HiGHS dual-simplex adapter. It converts
  `A₂x ≥ b₂` to HiGHS' `≤` convention and converts multiplier signs back to
  the paper's convention.
- Compared HPR with analytic KKT states and independent HiGHS solutions.
- Repeated every reference run and required exact deterministic equality of
  final states, iteration counts, and residual trajectories.
- Preserved full convergence trajectories as compressed JSONL and comparison
  summaries as JSON.

## 3. Mathematical decisions

### Exact projection in the y-subproblem

Algorithm 1 permits any positive-semidefinite proximal operator `T₁` satisfying:

```text
AAᵀ + T₁ ≻ 0
```

Stage 1 selects:

```text
T₁ = τI − AAᵀ
τ  = λmax(AAᵀ) + δ, where δ > 0
```

The implementation explicitly computes and checks the eigenvalues. Therefore:

```text
AAᵀ + T₁ = τI ≻ 0
```

This makes the constrained y-subproblem an ordinary Euclidean projection onto
the mixed dual set. It avoids the incorrect shortcut of solving a correlated
unconstrained system and then clipping only its inequality coordinates.

### Residual separation

These two related expressions have different purposes:

| Source | Expression | What it measures |
|---|---|---|
| Eq. (28), first block | `y − Π_D(y − Ax + b)` | Multiplier complementarity through a projected residual |
| Eq. (54a) | `Π_D(b − Ax)` | Primal feasibility only |

The API, tests, and evidence retain both rather than treating them as
interchangeable.

### Acceptance thresholds

The paper's three normalized Eq. (54) limits remain `5 × 10⁻⁵`. For the four
unit-scaled Stage 1 LPs, these additional validation targets were recorded:

| Measure | Required value |
|---|---:|
| Raw combined Eq. (28) norm | at most `2.5 × 10⁻⁴` |
| Maximum primal violation | at most `2.5 × 10⁻⁴` |
| Scaled objective gap to HiGHS | at most `2 × 10⁻⁴` |
| Inequality multipliers | at least `−1 × 10⁻¹²` |

These additional thresholds are Stage 1 validation choices; they are not
attributed to the paper.

### Random-case construction

The random case uses seed `20260729`, but feasibility is not left to chance. A
primal point, active constraints, dual multipliers, and bound normals were
planted first; `b` and `c` were then derived. Its active system has full rank,
making the known optimum a strong independent oracle.

## 4. Files created or modified

Principal Stage 1 artifacts:

- **Core implementation:** `canonical_lp.py`, `projections.py`, `residuals.py`,
  `hpr_generic.py`, `toy_problems.py`, and `validation.py` under
  `src/gpu_dcopf_hpr/`
- **Unit tests:** canonical LP, projections, residuals, and Halpern tests under
  `tests/unit/`
- **Integration tests:** `tests/integration/test_hpr_reference.py`
- **Configuration:** `configs/toy_lp/stage_1.json`
- **Reproduction scripts:** `scripts/run_toy_lp.py` and
  `scripts/check_stage_1.py`
- **Evidence:** comparison JSON, compressed trajectories, and Stage 1 checks
  under `results/raw/stage_1/`
- **Logs:** `logs/stage_1/`
- **Documentation:** decision log, mathematical notes, project state, this
  report, and the README
- **Dashboard:** Stage 1 data plus a rendered-output test

No Stage 2 source file was created.

## 5. Commands executed

Representative reproducible commands:

```powershell
python -m venv --system-site-packages .venv
./.venv/Scripts/python.exe -m pip install "ruff>=0.11"
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m ruff format --check .
python scripts/run_toy_lp.py
python scripts/check_stage_1.py --output results/raw/stage_1/stage_1_checks.json
npm run build
node --test tests/rendered-html.test.mjs
```

The first sandboxed Ruff download attempt was blocked by network isolation. An
approved retry installed Ruff only in the project-local virtual environment.

## 6. Test results

| Check | Result |
|---|---|
| Python test suite | **PASS** — 42 tests |
| Ruff lint | **PASS** — no findings |
| Ruff formatting | **PASS** — all Python files formatted |
| Four HiGHS reference solves | **PASS** |
| Four HPR reference solves | **PASS** |
| Exact repeated-run determinism | **PASS** |
| Stage 1 evidence checker | **PASS** |
| Dashboard production build | **PASS** |
| Dashboard rendered-output tests | **PASS** |

Static typing is not configured and was not made a Stage 1 blocker.

## 7. Numerical validation results

### Solver comparison

| Case | HPR iterations | HPR objective | HiGHS objective |
|---|---:|---:|---:|
| Analytic toy | 17,887 | 1.3998413703 | 1.4000000000 |
| Box-bound active | 2,501 | -0.7500000000 | -0.7500000000 |
| Inequality inactive | 4 | 0.2500000000 | 0.2500000000 |
| Planted random | 54,592 | -1.8149109627 | -1.8143841971 |

### Accuracy and feasibility

| Case | Scaled objective gap | Max primal violation | Eq. (28) norm |
|---|---:|---:|---:|
| Analytic toy | 6.61e-05 | 9.15e-05 | 1.24e-04 |
| Box-bound active | 0 | 1.00e-04 | 1.41e-04 |
| Inequality inactive | 0 | 2.50e-09 | 4.68e-09 |
| Planted random | 1.87e-04 | 1.98e-04 | 2.46e-04 |

All cases satisfy every configured target, all values are finite, and every
inequality multiplier is nonnegative.

The analytic LP has this exact solution:

| Quantity | Exact value |
|---|---:|
| Primal state, `x` | `(0.4, 0.6)` |
| Equality and inequality multipliers, `y` | `(1.5, 0.5)` |
| Bound multipliers, `z` | `(0, 0)` |
| Objective, `cᵀx` | `1.4` |

HiGHS recovers the solution to floating-point precision. HPR returns
`x = (0.3999329, 0.5999756)` at the first configured stopping point.

> **Why is the HPR objective slightly lower?**  
> The active inequality still has a small permitted violation at the stopping
> point. That is why every objective comparison is reported together with its
> feasibility measurement.

The residual trajectories are intentionally not required to decrease
monotonically. Final convergence and exact repeatability are the acceptance
properties.

## 8. Discrepancies from the paper

No new manuscript contradiction was required to implement generic Algorithm 1.
The stage did expose three details that must remain explicit:

1. Algorithm 1 does not prescribe a unique `T₁`; the Stage 1 spectral choice is
   a documented implementation decision.
2. Eq. (54a) is not the first block of Eq. (28), despite their related
   projections.
3. The paper's `5 × 10⁻⁵` threshold applies to three separately normalized
   blocks, not one raw combined norm.

Stage 1 iteration counts are correctness evidence only. They are not comparable
with the paper's optimized GPU timings or sGS iteration counts.

## 9. Unresolved questions

- How should the later CPU sGS implementation reconcile the paper's
  Proposition 5 inverse-sign discrepancy?
- Which exact generator bounds belong in the DCOPF box versus inequality block?
- Which public network release and PTDF conventions should be pinned for Stage
  2?
- What precision and spectral-estimation method did the paper's GPU code use?
- Can exact author benchmark data and source code be obtained?

These questions do not block the generic mathematical acceptance achieved here.

## 10. Acceptance-criteria checklist

- [x] Canonical LP representation validates dense and sparse dimensions, finite
  values, and bounds.
- [x] Box, nonnegative, and mixed dual-set projection tests pass.
- [x] Eq. (28) and Eq. (54) residual signs and normalizations are validated.
- [x] The required toy LP matches its analytic solution within the justified
  first-order tolerance.
- [x] Generic HPR matches independent HiGHS solutions on all four required case
  types.
- [x] Deterministic tests pass and full convergence logs are preserved.

## 11. Result

**PASS.** Stage 1 meets every acceptance criterion. The project is stopped at
the Stage 2 gate, and no CPU DCOPF construction has begun.

## 12. Approval command

To authorize the next stage, send exactly:

```text
APPROVE STAGE 1 AND RUN STAGE 2
```
