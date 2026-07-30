# Stage 4 report

> **Status:** PASS
>
> **Date:** July 30, 2026
>
> **Stage:** Paper-specific structural equality solve

## 1. Objective

Replace the generic equality-multiplier solve in both Algorithm 2 sweeps with
the paper-specific diagonal and rank-one method, while retaining the Stage 3
Cholesky path as an independent oracle.

Stage 4 remains local, CPU-only, fixed-sigma, no-restart, and FP64. It does not
implement Stage 5 scaling or penalty management, and it does not run Stage 6
GPU code on the DGX Spark.

## 2. Work performed

- Rendered and visually inspected manuscript pages 7 and 12.
- Derived the exact equality structure from the implemented variable order.
- Added a descriptor that refuses incompatible dimensions, row patterns,
  column order, coefficients, efficiencies, and unsafe Schur complements.
- Implemented a matrix-free structural solve using scalar terms and
  one-dimensional storage vectors only.
- Added explicit direct and structural equality backends to Algorithm 2.
- Preserved the direct dense Gram plus Cholesky backend as the default oracle.
- Tested 336 deterministic right-hand sides spanning:

  - no storage;
  - one storage device;
  - multiple storage devices;
  - one and many periods;
  - extreme valid efficiencies;
  - many ideal-efficiency storage devices;
  - three right-hand-side scales.

- Compared 5,000-iteration direct and structural trajectories.
- Ran both backends to convergence on the public T=1 and synthetic T=2 DCOPF
  models, including HiGHS and physical validation.
- Repeated each structural convergence run and compared all non-timing fields
  exactly.
- Measured direct versus structural performance and the empirical complexity
  trend on synthetic Equation (55) systems.

## 3. Mathematical decisions

### Implemented equality matrix

The decision-variable order is:

```text
(generation, renewable, storage discharge, storage charge,
 reserve up, reserve down)
```

Entries are period-major inside each block. The first T equality rows are
power balance. They use coefficients:

```text
generation          +1
renewable           +1
storage discharge   +1
storage charge      -1
reserve variables    0
```

The next S rows impose terminal storage energy. Storage device s uses:

```text
discharge coefficient   -h / eta_dc[s]
charge coefficient      +h * eta_ch[s]
```

Therefore:

```text
A1 has shape:
  (T + S) by T * (3G + R + 2S)

A1 A1^T =
  [ a I_T       1_T d^T ]
  [ d 1_T^T     D2      ]

a = G + R + 2S

d[s] = -h * (eta_ch[s] + 1 / eta_dc[s])

D2[s,s] =
  T * h^2 * (eta_ch[s]^2 + 1 / eta_dc[s]^2)
```

The repeated cross block is handled as a vector broadcast. No Kronecker
matrix is constructed.

### Corrected Proposition 5 sign

For right-hand side `(r11, r12)`, define:

```text
alpha = d^T D2^-1 d

r_hat =
  r11 - 1_T * (d^T D2^-1 r12)

gamma = a - T * alpha
```

Elimination produces a **minus** rank-one system:

```text
(a I_T - alpha 1_T 1_T^T) y11 = r_hat
```

Its correction sign is positive. The stable implementation splits `r_hat`
into mean and zero-mean parts:

```text
mu = mean(r_hat)

y11 =
  (r_hat - mu * 1_T) / a
  + (mu / gamma) * 1_T

y12 =
  D2^-1 * (r12 - d * sum(y11))
```

The manuscript's Equations (39), (44), and (45) print the sign pattern for a
plus rank-one update. That printed expression failed the direct oracle. The
corrected expression matched it to FP64 accuracy.

### Numerical stability

The implementation computes `gamma` with a positive-sum identity:

```text
gamma =
  G + R
  + sum over storage s of
      (eta_ch[s] - 1 / eta_dc[s])^2
      / (eta_ch[s]^2 + 1 / eta_dc[s]^2)
```

This avoids subtracting nearly equal values when many ideal storage devices
are present. A nonpositive or scale-small `gamma` is rejected. With no
storage, the solve reduces to division by `G + R`.

The structural formula assumes the unscaled Equation (55) row and column
families, one constant interval length, and the validated block order. A
reordered, scaled, or time-varying formulation must provide an adapted,
separately validated descriptor.

### Complexity boundary

The prepared inverse kernel performs linear work in `T + S`. The paper's
Corollary 1 also includes construction of `R1` through sparse `A1` products,
giving its stated work proxy:

```text
T * (G + R + S)
```

Two full evidence runs showed that the very short solve-only NumPy timing was
distorted by dispatch, allocation, cache, and memory-bandwidth effects. Those
non-gating failures are preserved in the log:

| Run | Slope | R-squared | Normalized spread |
|---|---:|---:|---:|
| Initial solve-only diagnostic | 0.636 | 0.819 | 4.889 |
| Increased-batch solve-only diagnostic | 0.552 | 0.889 | 4.860 |

Decision D-0033 keeps the unchanged empirical thresholds on the
paper-relevant RHS-plus-solve boundary. The solve-only wall-time fit remains a
reported, non-gating diagnostic. No empirical timing is presented as a formal
complexity proof.

## 4. Files created or modified

### Structural implementation

- `src/gpu_dcopf_hpr/structural_y1.py`
- `src/gpu_dcopf_hpr/sgs_hpr.py`
- `src/gpu_dcopf_hpr/__init__.py`
- `configs/sgs_hpr/stage_4_structural.json`

### Tests and evidence tools

- `tests/unit/test_structural_y1.py`
- `tests/integration/test_stage4_structural_y1.py`
- `scripts/run_stage_4.py`
- `scripts/check_stage_4.py`

### Evidence and logs

- `results/raw/stage_4/stage_4_validation.json`
- `results/raw/stage_4/structural_crosschecks.jsonl.gz`
- `results/raw/stage_4/solver_trajectories.jsonl.gz`
- `results/raw/stage_4/stage_4_checks.json`
- `logs/stage_4/commands_and_results.txt`

### Documentation and project state

- `docs/paper_specification.md`
- `docs/mathematical_notes.md`
- `docs/decisions.md`
- `docs/project_state.md`
- `docs/stage_reports/stage_4_report.md`
- `README.md`
- dashboard Stage 4 data, copy, and rendered-output checks

## 5. Commands executed

Principal reproducible commands:

```powershell
./.venv/Scripts/python.exe scripts/run_stage_4.py
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m ruff format --check .
./.venv/Scripts/python.exe scripts/check_stage_4.py --output results/raw/stage_4/stage_4_checks.json
```

Dashboard checks:

```powershell
cd dashboard
npm run build
node --test tests/rendered-html.test.mjs
```

The complete timing history, including two non-gating solve-only fit failures,
is preserved in `logs/stage_4/commands_and_results.txt`.

## 6. Test results

| Check | Result |
|---|---|
| Structural unit tests | **22 passed** |
| Real-model Stage 4 integration tests | **6 passed** |
| Full Python suite | **104 passed** |
| Ruff lint | PASS |
| Ruff formatting | PASS |
| Stage 4 evidence runner | PASS |
| Stage 4 independent checker | PASS |
| Dashboard production build | PASS |
| Dashboard rendered-output tests | PASS |

The final evidence runner passed all three sections:

```text
structural cross-checks       PASS
full solver cross-checks      PASS
performance and complexity   PASS
```

## 7. Numerical validation results

### Structural solve against the direct oracle

Each case used 48 deterministic right-hand sides. Absolute errors reflect
right-hand-side scales up to one million, so acceptance uses relative and
normalized metrics.

| Case | Rows | Condition number | Max absolute error | Max relative error | Max normalized residual | Max component error |
|---|---:|---:|---:|---:|---:|---:|
| No storage, T=1 | 1 | 1.00 | 2.91e-11 | 2.22e-16 | 1.67e-16 | 2.91e-11 |
| No storage, T=17 | 17 | 1.00 | 0 | 0 | 0 | 0 |
| Ideal storage, T=1 | 2 | 86.01 | 3.73e-9 | 2.18e-16 | 6.66e-16 | 3.73e-9 |
| One storage, T=2 | 3 | 3.58 | 1.30e-10 | 3.03e-16 | 3.21e-16 | 1.16e-10 |
| Extreme efficiency, T=32 | 33 | 25,616.42 | 1.14e-9 | 3.00e-15 | 2.93e-14 | 1.14e-9 |
| Heterogeneous storage, T=5 | 9 | 548.59 | 3.69e-10 | 1.15e-15 | 1.05e-15 | 3.11e-10 |
| Many ideal devices, T=16 | 48 | 292.03 | 2.75e-8 | 1.53e-14 | 6.44e-15 | 4.42e-9 |

Across the sign-resolution fixture:

```text
corrected maximum relative error       3.03e-16
printed-sign median relative error      0.206
```

### Complete solver comparison

| Case | Backend | Iterations | Total objective | Raw KKT norm | Max physical violation | Total runtime |
|---|---|---:|---:|---:|---:|---:|
| Public case5, T=1 | Direct | 108,134 | 17,479.7077242630 | 0.005771 | 0.004980 MW | 30.88 s |
| Public case5, T=1 | Structural | 108,134 | 17,479.7077242625 | 0.005771 | 0.004980 MW | 30.10 s |
| Synthetic extension, T=2 | Direct | 74,933 | 26,579.0043157485 | 0.012494 | 0.007577 MW/MWh | 21.49 s |
| Synthetic extension, T=2 | Structural | 74,933 | 26,579.0043157498 | 0.012494 | 0.007577 MW/MWh | 23.79 s |

The direct and structural paths stopped on exactly the same iteration in both
cases. Their scaled objective differences were `2.83e-14` and `4.90e-14`;
their final relative solution differences were `2.69e-14` and `3.63e-14`.

Across the fixed 5,000-iteration trajectories, the largest observed
direct-versus-structural differences were:

```text
relative x difference          7.78e-15
relative y difference          1.98e-14
relative z difference          3.85e-13
scaled objective difference    5.34e-15
combined residual difference   6.61e-12
```

The tiny T=1 and T=2 fixtures do not show a uniform runtime advantage because
Python overhead dominates. No speedup claim is made from those two runs.

### Synthetic performance and complexity

At the largest direct comparison, `T=1024` and `m1=1026`:

```text
solve-only direct / structural speedup       19.68x
RHS-plus-solve direct / structural speedup    7.61x
```

The final empirical fits were:

| Boundary | Slope | R-squared | Normalized spread | Role |
|---|---:|---:|---:|---|
| Structural solve only | 0.943 | 0.972 | 1.830 | Non-gating diagnostic |
| RHS formation plus structural solve | 1.158 | 0.995 | 1.723 | Complexity gate |

The gating fit satisfies the predeclared slope range `0.6` to `1.4`, minimum
R-squared `0.85`, and maximum normalized spread `4.0`. It is local empirical
support for the paper's theoretical claim, not a formal proof and not a
reproduction of the authors' GPU timing.

## 8. Discrepancies from the paper

1. Equation (43) has a minus rank-one Schur complement, but Equations (39),
   (44), and (45) print the inverse signs for a plus update. Direct numerical
   evidence supports Equation (43) and the corrected positive correction.
2. Appendix A replaces the terminal row's all-period vector with an identity
   matrix and claims `T(1 + S)` equality rows. Equation (55), Table II, and the
   implemented model support `T + S`.
3. The authors do not publish the exact benchmark inputs, CPU/GPU timing
   boundary, numerical precision, or reduction order needed for an exact
   performance reproduction.
4. The public T=1 result remains a mathematical reproduction and the T=2
   resource case remains a labeled synthetic structural fixture.

## 9. Unresolved questions

1. What exact power-system cases, profiles, reserve rules, renewable
   placements, storage placements, and device parameters produced the paper's
   tables?
2. What precision and GPU reduction order did the authors use?
3. How should the structural descriptor be generalized if later scaling or
   row and column reordering changes the raw Equation (55) form?
4. Which exact restart and adaptive-penalty rules were used? These remain
   locked until Stage 5 approval.
5. No Stage 6 GPU or DGX conclusion can be drawn from the local CPU evidence.

The Proposition 5 sign is no longer unresolved; Stage 4 resolved it against
the implemented matrix and direct oracle.

## 10. Acceptance-criteria checklist

- [x] Implemented `A1` matches the Equation (55) block structure.
- [x] Incompatible structures stop with explicit errors.
- [x] Structural solves match direct solves for no, one, and multiple storage
      devices, many periods, extreme efficiencies, and randomized right-hand
      sides.
- [x] No dense equality Gram, Cholesky factor, inverse, or explicit Kronecker
      matrix is used by the structural backend.
- [x] Complete direct and structural solver trajectories and final outputs
      agree.
- [x] HiGHS, Equation (54), raw KKT, objective, and physical targets remain
      satisfied.
- [x] A synthetic performance advantage was measured.
- [x] The paper-relevant empirical complexity trend passed unchanged limits.
- [x] Solve-only timing limitations and both earlier diagnostic failures are
      preserved.
- [x] Numerical stability limitations are documented and stress-tested.
- [x] Stage 5 scaling, restart, and adaptive sigma remain unimplemented.
- [x] Stage 6 GPU code and DGX execution remain unimplemented.

## 11. PASS or FAIL

**PASS**

Stage 4 satisfies every required acceptance criterion. The project is stopped
at the Stage 5 gate.

## 12. Exact command required to approve the next stage

```text
APPROVE STAGE 4 AND RUN STAGE 5
```
