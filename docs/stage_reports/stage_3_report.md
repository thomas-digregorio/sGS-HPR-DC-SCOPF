# Stage 3 report

> **Status:** PASS
>
> **Date:** July 29, 2026
>
> **Stage:** CPU sGS-HPR reference implementation

## 1. Objective

Implement the paper's Algorithm 2 on the local CPU using conservative FP64
linear algebra, then compare it with HiGHS on the four Stage 1 LPs and both
Stage 2 DCOPF models.

This stage establishes a readable correctness oracle. It does not use the
paper's structural Proposition 5 shortcut, scaling, adaptive penalty, restart,
GPU code, or the DGX Spark.

## 2. Work performed

- Implemented the printed update sequence:

  ```text
  z_bar
  x_bar
  y1_half
  y2_bar
  y1_bar
  reflection
  fixed-anchor Halpern update
  ```

- Kept the current, intermediate, reflected, and fixed-anchor states distinct.
- Verified the Equations (33)-(34) projection identity on seeded random states.
- Built and checked `A1 A1^T`, then reused one Cholesky factor for both equality
  sweeps.
- Implemented Equation (50) without materializing `S2`.
- Cross-checked the required largest eigenvalue with dense eigendecomposition,
  sparse `eigsh`, and deterministic power iteration.
- Evaluated all three Equation (54) tests every iteration and stored a sampled
  trajectory plus the exact stopping iterate.
- Added separate preparation, iteration-loop, and total CPU timing fields.
- Added a distinct approximate-candidate DCOPF validator. It preserves and
  tests the original balance error while using a temporary reference-slack
  adjustment only for independent flow checks.
- Ran every case twice and compared iteration counts, final states, and
  non-timing trajectory values exactly.

## 3. Mathematical decisions

### Trusted equality oracle

Both equality sweeps solve:

```text
(A1 A1^T) y1 = right-hand side
```

Stage 3 checks full row rank, raw-matrix symmetry, positive definiteness, and
the solve residual. The largest observed infinity-norm linear-system residual
was `2.28e-13`, below the `2e-12` acceptance target.

The structural inverse in Proposition 5 remains locked because the manuscript
contains unresolved sign inconsistencies. It will be tested against this
direct oracle in Stage 4.

### Conservative inequality step

The implementation uses:

```text
lambda = largest eigenvalue of A2 A2^T
```

The dense result is the correctness authority on these small matrices. Sparse
`eigsh` and seeded power iteration are independent cross-checks. The value used
by the solver is the largest estimate plus a positive FP64 margin.

| DCOPF case | Dense estimate | Maximum estimator disagreement | Lambda used |
|---|---:|---:|---:|
| Public case5, T=1 | 6.541815375383 | 1.78e-15 | 6.541815376037 |
| Synthetic extension, T=2 | 11.825678984060 | 7.11e-15 | 11.825678985242 |

### Stopping versus validation

The paper's stopping rule remains `5e-5` for each separately normalized
Equation (54) block. Additional acceptance targets are explicit:

| Case type | Raw Eq. (28) target | Scaled objective gap | Raw physical target |
|---|---:|---:|---:|
| Unit-scale toy LP | 2.5e-4 | 5e-4 | 2.5e-4 primal violation |
| MW-scale DCOPF | 2e-2 | 2e-4 | 0.01 MW/MWh |

These do not replace or reinterpret the paper's stopping tolerance.

### Exact checks, sparse evidence

Equation (54) residuals are not monotone. Testing only every 250 iterations can
miss an earlier valid iterate. The solver therefore checks every iteration but
persists iteration 1, every 250th iteration, and the exact stopping iteration.

## 4. Files created or modified

### Solver and validation

- `src/gpu_dcopf_hpr/sgs_hpr.py`
- `src/gpu_dcopf_hpr/validation.py`
- `src/gpu_dcopf_hpr/__init__.py`
- `configs/sgs_hpr/stage_3_fixed_sigma.json`

### Tests and execution

- `tests/unit/test_sgs_hpr.py`
- `tests/integration/test_stage3_sgs_hpr.py`
- `tests/integration/test_stage2_dcopf.py`
- `scripts/run_cpu_solver.py`
- `scripts/check_stage_3.py`

### Evidence and documentation

- `results/raw/stage_3/stage_3_validation.json`
- `results/raw/stage_3/sgs_hpr_trajectories.jsonl.gz`
- `results/raw/stage_3/stage_3_checks.json`
- `logs/stage_3/commands_and_results.txt`
- `docs/mathematical_notes.md`
- `docs/decisions.md`
- `docs/project_state.md`
- `docs/stage_reports/stage_3_report.md`
- `README.md`
- dashboard stage data, copy, and rendered-output test

## 5. Commands executed

The principal reproducible commands were:

```powershell
./.venv/Scripts/python.exe scripts/run_cpu_solver.py
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/ruff.exe format --check .
./.venv/Scripts/python.exe scripts/check_stage_3.py --output results/raw/stage_3/stage_3_checks.json
```

Dashboard validation:

```powershell
cd dashboard
npm run build
node --test tests/rendered-html.test.mjs
```

The manuscript pages containing Algorithm 2 and Equations (31)-(54) were also
rendered and visually checked before implementation. Temporary page renders
were not retained as project evidence.

## 6. Test results

| Check | Result |
|---|---|
| Stage 3 solver and candidate-validator tests | PASS |
| Full Python suite | **76 passed** |
| Ruff lint | PASS |
| Ruff formatting | PASS |
| Stage 3 evidence checker | PASS |
| Dashboard production build | PASS |
| Dashboard rendered-output tests | PASS |

No NaN or infinite iterate was observed. All recorded inequality multipliers
were nonnegative within FP64 tolerance.

## 7. Numerical validation results

### Toy LPs

| Case | Iterations | sGS-HPR objective | HiGHS objective | Scaled gap | Raw Eq. (28) |
|---|---:|---:|---:|---:|---:|
| Analytic active inequality | 17,606 | 1.3998416601 | 1.4000000000 | 1.13e-4 | 1.24e-4 |
| Box bound active | 2,501 | -0.7500000000 | -0.7500000000 | 0 | 1.41e-4 |
| Inequality inactive | 4 | 0.2500000000 | 0.2500000000 | 0 | 3.14e-16 |
| Seeded planted random | 32,615 | -1.8149005548 | -1.8143841971 | 2.85e-4 | 2.47e-4 |

The analytic candidate's slightly lower objective is explained by its small
finite-tolerance feasibility error; it is not better than the exact optimum.

### DCOPF

| Case | Iterations | sGS-HPR total objective | HiGHS total objective | Scaled gap | Max physical violation |
|---|---:|---:|---:|---:|---:|
| Public case5, T=1 | 108,134 | 17,479.7077242630 | 17,479.8969253810 | 1.08e-5 | 0.004980 MW |
| Synthetic RG/storage, T=2 | 74,933 | 26,579.0043157485 | 26,580.0033355255 | 3.76e-5 | 0.007577 MW/MWh |

The final raw Equation (28) norms were `0.00577` and `0.01249`, both below the
stated MW-scale target of `0.02`. Each separately normalized Equation (54)
block passed `5e-5`.

The synthetic objective includes a `10,000` constant. Machine-readable
evidence labels the canonical variable part, constant, and total separately so
trajectory and summary values cannot be confused.

## 8. Discrepancies from the paper

- The paper states a cold start but does not print the initial vectors. Stage 3
  uses and records all-zero `w0`.
- The eigenvalue estimator and safety margin are not specified. Stage 3 uses
  three cross-checks and a disclosed positive margin.
- The adaptive penalty and restart formulas are not printed. Stage 3 uses
  fixed `sigma = 1` and zero restarts.
- The paper's exact DCOPF inputs remain unavailable. Results are a mathematical
  reproduction on a public base case and a structural validation on a labeled
  synthetic extension.
- The paper's CPU/GPU timing boundary is unclear. Stage 3 reports a named local
  CPU decomposition and makes no paper speedup claim.

## 9. Unresolved questions

1. Which sign convention is correct in the Proposition 5 rank-one inverse?
2. Should the equality-row count follow Equation (55) or the conflicting
   Appendix A expression?
3. What exact cold-start vectors, precision, spectral estimator, reduction
   order, adaptive penalty, and restart rule did the authors use?
4. Can the authors' device placements, time series, reserve requirements, and
   physical parameters be recovered?
5. Which preprocessing and transfer costs are included in the published times?

## 10. Acceptance-criteria checklist

- [x] Algorithm 2 update order is implemented and tested.
- [x] Current, intermediate, reflected, and fixed-anchor states remain distinct.
- [x] Equations (33)-(34) agree numerically on seeded random vectors.
- [x] Both equality sweeps use an accurate direct solve.
- [x] `A1` rank, Gram symmetry, and positive definiteness are checked.
- [x] Equation (50) uses the correct sign and nonnegative projection.
- [x] Dense, sparse, and power spectral estimates agree.
- [x] Equation (54) is evaluated exactly on the intermediate state.
- [x] Fixed-sigma CPU sGS-HPR converges on all four toy LPs.
- [x] T=1 and T=2 DCOPF candidates agree with HiGHS within stated tolerances.
- [x] Raw Eq. (28), physical constraints, and objective gaps pass.
- [x] Repeated complete runs are deterministic apart from timing.
- [x] No GPU code, DGX execution, structural shortcut, scaling, adaptation, or
      restart was introduced.

## 11. Stage result

**PASS**

Stage 3 satisfies every acceptance criterion. The direct CPU implementation is
now the oracle against which the paper-specific structural equality solve can
be evaluated.

## 12. Exact approval command for Stage 4

```text
APPROVE STAGE 3 AND RUN STAGE 4
```

Stage 4 has not begun.
