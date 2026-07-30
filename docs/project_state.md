# Project state

Last updated: 2026-07-30

## Stage gate

- Completed stage: **Stage 5 - preconditioning, restart, and penalty
  management**
- Gate result: **PASS**
- Current state: **stopped at the Stage 6 approval gate**
- Next proposed stage: **Stage 6 - GPU port for DGX Spark**
- Required approval: `APPROVE STAGE 5 AND RUN STAGE 6`
- Dashboard: private, owner-only Sites deployment; its URL and project
  identifier are intentionally omitted from the public repository.

Stage 6 has not started. No GPU solver code was executed and the DGX Spark was
not changed during Stage 5.

## Stage 5 implementation

The FP64 CPU package now adds:

- 10 simultaneous Ruiz row/column infinity-norm equilibration steps;
- one simultaneous Pock-Chambolle row/column L1 step with alpha 1;
- post-diagonal normalization using the complete scaled `b` and `c` vectors;
- cumulative positive factors and exact forward/inverse LP and state maps;
- original-objective recovery through the product of the two norm factors;
- original-space Equation (54), KKT, HiGHS, and physical checks;
- the HPR-LP restart criteria with a forced first restart;
- policy checkpoints at the DCOPF manuscript's 100-iteration cadence;
- adaptive sigma from the HPR-LP movement ratio in the general sGS metric;
- explicit movement and infeasibility-ratio guards;
- auditable restart reasons, sigma decisions, and sparse trajectories;
- all four fixed/adaptive and restart/no-restart combinations;
- unscaled, normalization-only, Ruiz, and full-preconditioning ablations;
- initial-sigma sensitivity runs at `0.1` and `10`.

The reconstruction pins HPR-LP v0.1.0 commit
`1941fbcfbf2dae14e4a439b22f0ea1e1c05f4a29`. A later HPR-LP commit was
inspected for source drift, but its additional penalty heuristics were not
adopted.

## Numerical validation

The principal full-policy results use 10 Ruiz steps, one Pock-Chambolle step,
norm normalization, adaptive sigma, and restart:

| Case | Iterations | Restarts | Final sigma | Raw KKT | Objective gap | Max physical violation |
|---|---:|---:|---:|---:|---:|---:|
| Public case5, T=1 | 410 | 4 | 8.9852 | 0.005618 | 3.31e-6 | 0.004231 MW |
| Synthetic extension, T=2 | 1,032 | 8 | 2.2433 | 0.008948 | 1.02e-5 | 0.006059 MW/MWh |

Both cases satisfy:

- every separately normalized Equation (54) block at `5e-5`;
- the Stage 5 raw KKT target of `0.01`;
- the physical target of `0.01 MW/MWh`;
- the scaled objective-gap target of `2e-4`;
- equality-solve infinity residual below `1.67e-16`;
- the `z`/projected-`x` identity below `3.21e-15`.

On the public T=1 case, fixed sigma without restart converged in 123,328
iterations, fixed sigma with restart in 524, and adaptive sigma with restart
in 410. The adaptive-without-restart diagnostic ran its declared 5,000
iterations and did not converge. It is a non-paper, non-gating component
ablation.

Both adaptive initial-sigma sensitivity runs converged: 477 iterations from
`sigma=0.1` and 532 iterations from `sigma=10`.

## Preprocessing validation

Dense analytic and sparse planted-random fixtures passed:

- exact 10-step and one-step counts;
- positive finite row and column factors;
- unchanged sparse nonzero count;
- LP-data and state round-trips;
- primal and stationarity transform identities;
- variable-objective recovery;
- complete-vector norm-factor checks.

The largest reported component discrepancy was `2.45e-16`, below the
`5e-12` identity tolerance and `5e-13` round-trip tolerance.

The fixed 5,000-iteration public-case preprocessing comparison remains
diagnostic:

| Preprocessing | Raw KKT after 5,000 iterations |
|---|---:|
| Unscaled Stage 4 structural baseline | 0.1311 |
| Norm normalization only | 0.9208 |
| 10 Ruiz plus norm | 0.7216 |
| 10 Ruiz plus Pock-Chambolle plus norm | 0.2475 |

These equal-horizon runs isolate transformations; they are not convergence
claims.

## Evidence and quality status

- `results/raw/stage_5/stage_5_validation.json`: PASS
- `results/raw/stage_5/stage_5_trajectories.jsonl.gz`: preserved histories and
  policy events
- `results/raw/stage_5/stage_5_checks.json`: 23 / 23 independent checks passed
- Stage 5 unit and integration tests: 39 passed
- Full Python suite: 143 passed
- Ruff lint and formatting checks: PASS

## Environment

Stage 5 ran locally on:

- Windows 11;
- Python 3.13.5;
- NumPy 2.4.1;
- SciPy 1.16.3 with `linprog(method="highs-ds")`;
- NumPy/SciPy CPU FP64;
- pytest and Ruff 0.16.0 in the project-local environment.

The DGX Spark remains audited and reachable but unchanged. GPU execution is
still locked behind Stage 6 approval.

## Supported claim and limitations

Stage 5 is a **sourced HPR-LP transfer** with the DCOPF paper's preprocessing
sequence and 100-iteration cadence. It is not a claim of byte-for-byte identity
with the authors' unpublished DCOPF implementation.

Current limitations include:

- exact author restart and adaptive-penalty code is unavailable;
- the inspected current HPR-LP source contains later heuristics that are
  intentionally outside this reconstruction;
- adaptive without restart is a controlled fixed-horizon ablation, not a
  paper algorithm;
- general diagonal scaling invalidates the raw Equation (55) structural
  descriptor, so scaled runs use direct equality solves;
- paper benchmark inputs and construction scripts remain unavailable;
- all Stage 5 results are local CPU timings, not DGX or GPU measurements.

## Reproduction classification

The public case5 result remains a mathematical reproduction of the printed
model on an independently sourced network. The T=2 resource case remains a
labeled synthetic structural fixture. Neither is an exact paper-instance or
paper-timing reproduction.

## Next proposed stage

Stage 6 may begin only after the exact approval command:

```text
APPROVE STAGE 5 AND RUN STAGE 6
```

Stage 6 will establish the DGX software environment, port the already validated
FP64 kernels, verify CPU/GPU agreement, and define honest device timing
boundaries. It must not weaken any Stage 5 original-space acceptance check.
