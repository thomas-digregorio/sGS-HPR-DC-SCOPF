# Stage 2 report

> **Status:** PASS  
> **Date:** July 29, 2026  
> **Stage:** CPU DCOPF model construction

## 1. Objective

Construct and independently validate the paper's multi-period DCOPF linear
program on CPU before implementing the paper-specific sGS-HPR iteration.

This stage covered network data, DC shift factors, deterministic indexing,
sparse canonical matrices, HiGHS reference solutions, and direct physical
validation. It did not implement Algorithm 2, run on the DGX Spark, or add N-1
contingency constraints.

## 2. Work performed

- Added a safe numeric parser for MATPOWER version-2 bus, generator, branch,
  and generation-cost tables. It does not evaluate arbitrary MATLAB code.
- Pinned the unmodified MATPOWER 8.1 `case5.m` public network.
- Built an affine PTDF representation that supports transformer taps and phase
  shifts:

  ```text
  branch flow = H × balanced bus injection + phase-shift offset
  ```

- Validated PTDF flows against independent reduced-angle DC power-flow solves.
- Added explicit failures for disconnected networks, near-zero reactance,
  nonpositive taps, active angle limits, and nonlinear costs.
- Added deterministic bidirectional variable indexing in this block order:

  ```text
  conventional generation
  renewable generation
  storage discharge
  storage charge
  upward reserve
  downward reserve
  ```

- Constructed sparse `A1`, `A2`, `b1`, `b2`, `c`, lower, and upper arrays for
  every constraint family printed in Equations (1) through (10).
- Attached equation, period, physical element, and side metadata to every row.
- Solved a public one-period base model with SciPy HiGHS.
- Solved a separately labeled two-period synthetic extension that exercises
  reserve, ramp, renewable, storage-energy, and terminal-energy constraints.
- Recomputed every physical constraint directly from the returned dispatch
  instead of relying only on solver status or matrix residuals.

## 3. Mathematical decisions

### Topology branches versus constrained lines

MATPOWER case5 has six active branches. Four have `RATE_A=0`, which MATPOWER
uses to mean "unlimited." Those branches remain in the network physics, but
they do not receive invented thermal limits.

Therefore:

| Quantity | Count |
|---|---:|
| Active branches used in PTDF construction | 6 |
| Branches with finite positive thermal limits | 2 |
| Eq. (2) rows per period | 4 |

### Affine phase-shift treatment

A transformer phase shift can create a nonzero branch flow even when the
variable injection is zero. Stage 2 retains this as an affine offset rather
than silently dropping it. A three-bus analytic test checks both a non-unity
tap and a 10-degree phase shift.

### Conventional generator bounds

Conventional output limits are stored in the variable box and Eq. (4)
headroom/footroom rows are retained. The rows are redundant when reserves are
zero, but this split preserves finite generator limits and matches the compact
model's treatment of variable bounds.

### Exact linear public costs

All five case5 generation costs use MATPOWER polynomial model 2 with exactly
two coefficients: a linear slope and a constant. No quadratic coefficient was
dropped. Other nonlinear or multisegment costs stop with a clear error.

### Synthetic extension boundary

The public base case has no renewable or storage devices. The second case adds
one renewable device and one storage device using an explicit JSON
configuration. Its locations and time-series values are synthetic validation
data, not author data.

The synthetic 10 MW generator ramp limits force a nonzero storage trajectory,
so the energy equations are tested with actual charging and discharging.

## 4. Files created or modified

- **Network and physics:** `network_data.py` and `ptdf.py`
- **Sparse model construction:** `dcopf_model.py`
- **Independent validation:** extended `validation.py`
- **Public input:** pinned `data/raw/matpower/case5.m` plus provenance
- **Configurations:** public base and synthetic-extension JSON files under
  `configs/dcopf/`
- **Scripts:** `build_dcopf.py` and `check_stage_2.py`
- **Tests:** MATPOWER parsing, PTDF physics, sparse-model construction, and
  HiGHS integration tests
- **Evidence:** Stage 2 validation JSON plus compressed row-metadata files
- **Documentation:** decision log, mathematical notes, project state, README,
  and this report
- **Dashboard:** Stage 2 task state, findings, evidence links, and Stage 3 gate

No Stage 3 source file was created.

## 5. Commands executed

Representative reproducible commands:

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m ruff format --check .
./.venv/Scripts/python.exe scripts/build_dcopf.py
./.venv/Scripts/python.exe scripts/check_stage_2.py --output results/raw/stage_2/stage_2_checks.json
npm run build
node --test tests/rendered-html.test.mjs
```

No package installation or DGX Spark change was required.

## 6. Test results

| Check | Result |
|---|---|
| Python test suite | **PASS** - 59 tests |
| Ruff lint | **PASS** - no findings |
| Ruff formatting | **PASS** - all Python files formatted |
| Public MATPOWER parser regression | **PASS** |
| PTDF versus angle-flow comparisons | **PASS** |
| Transformer tap and phase-shift analytic test | **PASS** |
| Disconnected and near-zero-reactance failures | **PASS** |
| Sparse dimension and metadata checks | **PASS** |
| Public case5 HiGHS solve | **PASS** |
| Synthetic-extension HiGHS solve | **PASS** |
| Independent physical validation | **PASS** |
| Exact repeated HiGHS results | **PASS** |
| Stage 2 evidence checker | **PASS** |
| Dashboard production build and rendered-output test | **PASS** |

Static typing remains unconfigured and was not made a Stage 2 blocker.

## 7. Numerical validation results

### Public network provenance

| Item | Value |
|---|---|
| Upstream | `MATPOWER/matpower` |
| Release | `8.1` |
| File | `data/case5.m` |
| Upstream blob | `b6370ab230ac5346023d23be20d973a81f09e12a` |
| Checked-in SHA-256 | `ed9072089d080debde2aa029ed8a95491c311b08ff4f45ad7bf06882d86fe279` |
| Reference bus | 4 |

### Model and solver summary

| Case | Dimensions `(n, m1, m2)` | Total objective | HiGHS iterations | Largest physical violation |
|---|---:|---:|---:|---:|
| Public case5, T=1 | `(15, 1, 16)` | 17,479.8969253810 | 2 | 1.14e-13 |
| Synthetic extension, T=2 | `(36, 3, 46)` | 26,580.0033355255 | 15 | 1.14e-13 |

The public-case generation dispatch is:

| Generator | Output (MW) |
|---|---:|
| 1 | 40.000000 |
| 2 | 170.000000 |
| 3 | 323.494846 |
| 4 | 0.000000 |
| 5 | 466.505154 |

The sixth branch reaches its lower thermal limit at `-240 MW`. All other
reported physical and canonical violations are within FP64 solver tolerance.

### Nonzero synthetic storage trajectory

| Period | Charge (MW) | Discharge (MW) |
|---:|---:|---:|
| 1 | 32.519335 | 0.000000 |
| 2 | 0.000000 | 27.804031 |

The charge efficiency is `0.95` and the discharge efficiency is `0.90`.
The energy gained in period 1 equals the energy removed in period 2, so the
terminal energy returns to its initial value.

### Independent PTDF and physical checks

- Maximum unit-transfer PTDF versus angle-flow difference:
  `2.23e-16 MW`
- Maximum solved-point PTDF versus angle-flow difference:
  `8.53e-14 MW`
- Maximum physical constraint violation across both cases:
  `1.14e-13`
- Synthetic objective reconstructed independently from device formulas:
  agreement within `3.64e-12`
- Row metadata records:
  `17` for the base model and `49` for the synthetic extension

## 8. Discrepancies from the paper

1. The paper does not identify its exact MATPOWER release or preprocessing.
   Stage 2 uses a pinned public case5 input and makes no author-instance claim.
2. The paper does not publish the reserve, ramp, renewable, storage, or
   time-series data needed for its reported cases. The two-period extension is
   explicitly synthetic.
3. MATPOWER distinguishes active topology branches from thermally constrained
   branches through `RATE_A`; the paper does not describe this preprocessing.
4. Active MATPOWER branch-angle limits are outside Equations (1) through (10)
   and are rejected rather than silently added.
5. The equality-row count remains `T + N_ESS`, consistent with Equation (55)
   and the paper's dimension tables rather than Appendix A's conflicting text.

## 9. Unresolved questions

- Which exact case files, commits, reference buses, and branch filters did the
  authors use?
- What reserve requirements and generator ramp values were used?
- Where were renewable and storage devices placed?
- Which load, renewable, and storage time series generated the paper tables?
- Did the authors retain transformer phase-shift offsets in their shift-factor
  construction?
- Can exact author preprocessing code or benchmark data be obtained?

These questions prevent an exact paper-instance claim but do not block the
validated mathematical model.

## 10. Acceptance-criteria checklist

- [x] A public MATPOWER-compatible network is parsed with pinned provenance.
- [x] Buses, loads, active branches, generators, linear costs, taps, status,
  and reference bus are represented.
- [x] PTDF flows match independent angle-based flows.
- [x] Balanced injections, reference choice, islands, inactive and parallel
  branches, and near-zero reactance are tested.
- [x] Variable indexing is deterministic and bidirectional.
- [x] Every printed constraint family is built as sparse canonical rows.
- [x] Model dimensions match the derived formulas.
- [x] HiGHS solves both the public base and synthetic extension.
- [x] Every physical constraint family is recalculated independently.
- [x] Synthetic data are clearly labeled.
- [x] Every matrix row is traceable to a physical element and paper equation.

## 11. Result

**PASS.** Stage 2 meets every acceptance criterion. The project is stopped at
the Stage 3 gate. No CPU sGS-HPR iteration has begun.

## 12. Approval command

To authorize the next stage, send exactly:

```text
APPROVE STAGE 2 AND RUN STAGE 3
```
