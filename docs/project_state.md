# Project state

Last updated: 2026-07-29

## Stage gate

- Completed stage: **Stage 3 - CPU sGS-HPR reference implementation**
- Gate result: **PASS**
- Current state: **stopped at the Stage 4 approval gate**
- Next proposed stage: **Stage 4 - paper-specific structural equality solve**
- Required approval: `APPROVE STAGE 3 AND RUN STAGE 4`
- Dashboard: private, owner-only Sites deployment; its URL and project
  identifier are intentionally omitted from the public repository.

No Stage 4 structural equality solver has been created.

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
- a HiGHS reference solve and formula-based physical validation;
- the paper's complete Algorithm 2 update sequence in FP64 on CPU;
- trusted direct solves for both equality-multiplier sweeps;
- dense, sparse, and power-iteration spectral cross-checks;
- exact per-iteration Equation (54) checks with sparse trajectory storage;
- separate strict-solution and approximate-candidate DCOPF validators.

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
- CPU sGS-HPR converges on all four deterministic Stage 1 LPs.
- CPU sGS-HPR converges on the public T=1 and synthetic T=2 DCOPF models.
- The DCOPF scaled objective gaps to HiGHS are `1.08e-5` and `3.76e-5`.
- The largest candidate physical violations are `0.004980 MW` and
  `0.007577 MW/MWh`, below the stated `0.01` target.
- Both direct equality sweeps remain below `2.28e-13` infinity-norm residual.
- Repeated full runs reproduce iteration counts, states, and non-timing
  trajectory fields exactly.
- Test suite: 76 passed.
- Ruff lint and formatting checks pass.

## Numerical validation summary

| Case | Iterations | sGS-HPR total objective | HiGHS total objective | Raw Eq. (28) | Max physical violation |
|---|---:|---:|---:|---:|---:|
| case5 base, T=1 | 108,134 | 17,479.7077242630 | 17,479.8969253810 | 0.00577 | 0.004980 MW |
| synthetic extension, T=2 | 74,933 | 26,579.0043157485 | 26,580.0033355255 | 0.01249 | 0.007577 MW/MWh |

Both cases satisfy all three separately normalized Equation (54) tests at
`5e-5`, the raw DCOPF Equation (28) target of `0.02`, the physical target of
`0.01 MW/MWh`, and the scaled objective-gap target of `2e-4`.

## Environment status

Stage 3 ran locally on:

- Windows 11;
- Python 3.13.5;
- NumPy 2.4.1;
- SciPy 1.16.3 with `linprog(method="highs-ds")` and direct Cholesky;
- FP64 arrays;
- pytest and Ruff 0.16.0 in the project-local virtual environment.

The DGX Spark remains audited and reachable but unchanged. Stage 3 installed
nothing and ran no solver there. GPU execution remains locked until Stage 6.

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
- Stage 3 uses dense correctness cross-checks for the small spectral problems;
  this is not yet a large-case spectral strategy.
- The all-zero cold-start vectors are a disclosed local choice because the
  paper does not print them.
- The fixed-sigma method is a correctness baseline and requires tens of
  thousands of iterations on these unscaled cases.
- Approximate first-order candidates are not relabeled as strict Stage 2 power
  flows; their balance errors remain explicit.

## Unresolved paper ambiguities

1. Equation (55) and Table II imply `m1 = T + N_ESS`, while Appendix A prints
   `T(1 + N_ESS)`.
2. Equation (43) and Equations (39), (44), and (45) disagree on rank-one
   inverse signs.
3. Exact author preprocessing for reference buses, transformers, inactive
   branches, and thermal ratings is unavailable.
4. Adaptive penalty and restart formulas remain incomplete.
5. The paper's precision, eigenvalue estimator, initialization vectors, and
   timing boundaries remain incomplete. Stage 3 records its own choices rather
   than attributing them to the authors.

## Reproduction classification

The public case5 result is a mathematical reproduction of the paper's printed
model on an independently sourced network. The two-period result is a
structural validation fixture. Neither is an exact paper-instance or timing
reproduction.

## Next proposed stage

Stage 4 will replace the direct equality sweep with the paper's structural
formula:

1. derive the implemented `A1` block structure in the exact variable order;
2. implement diagonal and low-rank operations without large dense matrices;
3. cross-check many right-hand sides against the Stage 3 Cholesky oracle;
4. compare complete direct and structural solver trajectories;
5. measure, but not overclaim, the observed complexity trend.

The direct Stage 3 path remains available as the correctness reference.
