# Project state

Last updated: 2026-07-29

## Stage gate

- Completed stage: **Stage 2 - CPU DCOPF model construction**
- Gate result: **PASS**
- Current state: **stopped at the Stage 3 approval gate**
- Next proposed stage: **Stage 3 - CPU sGS-HPR reference implementation**
- Required approval: `APPROVE STAGE 2 AND RUN STAGE 3`
- Dashboard: private, owner-only Sites deployment; its URL and project
  identifier are intentionally omitted from the public repository.

No Stage 3 sGS-HPR iteration or structural equality solver has been created.

## Current implementation status

The FP64 CPU package now provides:

- the complete Stage 1 canonical LP, projection, residual, and generic HPR
  reference layer;
- a safe numeric MATPOWER version-2 parser for bus, generator, branch, and
  generation-cost tables;
- explicit source hashing and preserved external bus and equipment ordering;
- an affine PTDF model with reference-bus reduction, transformer taps, phase
  shifts, inactive branches, and parallel branches;
- deterministic bidirectional variable indexing in the paper's block order;
- sparse construction of every printed DCOPF constraint family;
- per-row physical metadata tied to the original paper equation;
- objective constants separated from the canonical variable coefficient;
- a HiGHS reference solve and formula-based physical validation.

## Validated capabilities

- MATPOWER 8.1 `case5.m` is pinned by release, upstream blob, and SHA-256.
- The six active case5 branches all remain in the PTDF topology; only the two
  with finite positive `RATE_A` contribute line-limit rows.
- PTDF flows match reduced-angle DC flows within `2.23e-16 MW` on unit-transfer
  probes and within `8.53e-14 MW` on solved operating points.
- Tests cover arbitrary bus identifiers, reference-bus changes, disconnected
  systems, inactive and parallel branches, near-zero reactance rejection,
  transformer taps, and affine phase-shift offsets.
- The one-period public base model has dimensions
  `n=15, m1=1, m2=16, m=17`.
- The two-period synthetic extension has dimensions
  `n=36, m1=3, m2=46, m=49`.
- Every equality and inequality row has traceable metadata.
- HiGHS solves both configurations deterministically.
- Direct physical validation passes for balance, line flows, generator bounds,
  reserve bounds and totals, ramping, renewable limits, storage power and
  energy, terminal energy, and objective reconstruction.
- Test suite: 59 passed.
- Ruff lint and formatting checks pass.

## Numerical validation summary

| Case | Classification | Total objective | HiGHS iterations | Largest physical violation |
|---|---|---:|---:|---:|
| case5 base, T=1 | public mathematical reproduction | 17,479.8969253810 | 2 | 1.14e-13 |
| case5 synthetic extension, T=2 | deterministic coverage fixture | 26,580.0033355255 | 15 | 1.14e-13 |

The public base dispatch is:

```text
generation = (40.0000, 170.0000, 323.494846, 0.0000, 466.505154) MW
```

The synthetic extension forces the storage device to:

```text
period 1: charge    32.519335 MW
period 2: discharge 27.804031 MW
```

With charge efficiency `0.95` and discharge efficiency `0.90`, the terminal
energy returns exactly to its initial value within FP64 solver tolerance.

## Environment status

Stage 2 ran locally on:

- Windows 11;
- Python 3.13.5;
- NumPy 2.4.1;
- SciPy 1.16.3 with `linprog(method="highs-ds")`;
- FP64 arrays;
- pytest and Ruff 0.16.0 in the project-local virtual environment.

The DGX Spark remains audited and reachable but unchanged. Stage 2 installed
nothing and ran no solver there.

## Known limitations

- The MATPOWER parser accepts numeric version-2 tables; it deliberately does
  not execute arbitrary MATLAB expressions.
- Disconnected networks, near-zero active-branch reactance, nonpositive taps,
  nonlinear generation costs, and active branch-angle limits stop with clear
  errors.
- The paper's exact network release, time series, reserve rules, ramp data,
  renewable placements, and storage placements remain unavailable.
- The two-period resource case is synthetic validation data, not an author
  benchmark.
- Stage 2 constructs and solves the LP with HiGHS only. It does not yet run
  Algorithm 2.

## Unresolved paper ambiguities

1. Equation (55) and Table II imply `m1 = T + N_ESS`, while Appendix A prints
   `T(1 + N_ESS)`.
2. Equation (43) and Equations (39), (44), and (45) disagree on rank-one
   inverse signs.
3. Exact author preprocessing for reference buses, transformers, inactive
   branches, and thermal ratings is unavailable.
4. Adaptive penalty and restart formulas remain incomplete.
5. Precision, eigenvalue estimation, sparse storage, initialization, and
   timing boundaries remain incomplete.

## Reproduction classification

The public case5 result is a mathematical reproduction of the paper's printed
model on an independently sourced network. The two-period result is a
structural validation fixture. Neither is an exact paper-instance or timing
reproduction.

## Next proposed stage

Stage 3 will implement fixed-penalty CPU Algorithm 2:

1. preserve the printed z, x, y1-half, y2, y1 update order;
2. use trusted direct solves for both equality-multiplier sweeps;
3. implement the projected inequality-multiplier update;
4. retain the fixed Halpern anchor and Eq. (54) stopping rules;
5. compare the complete CPU iteration with HiGHS on the Stage 2 models.

The Proposition 5 structural inverse remains locked until Stage 4.
