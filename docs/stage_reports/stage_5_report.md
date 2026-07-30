# Stage 5 report

> **Status:** PASS
>
> **Date:** July 30, 2026
>
> **Stage:** Preconditioning, restart, and penalty management

## 1. Objective

Add the paper's preprocessing sequence and a reproducible restart/adaptive
penalty policy to the validated CPU sGS-HPR solver, while preserving every
Stage 3 and Stage 4 correctness reference.

Stage 5 remains local, CPU-only, FP64 work. It does not install software or
execute solver code on the DGX Spark, and it does not begin the Stage 6 GPU
port.

## 2. Source boundary

The DCOPF manuscript specifies:

- 10 Ruiz iterations;
- Pock-Chambolle preconditioning with alpha 1;
- normalization by `1 + norm(b)` and `1 + norm(c)`;
- initial sigma 1;
- a restart check every 100 iterations;
- an adaptive sigma rule similar to HPR-LP;
- Equation (54) tolerance `5e-5`.

It does not print the restart criterion, restart state transition, or adaptive
sigma formula. Stage 5 therefore uses a disclosed, pinned reconstruction:

| Part | Source used |
|---|---|
| Preprocessing order | DCOPF manuscript, checked against PDLP and HPR-LP |
| Pock-Chambolle formula | Pock-Chambolle source and PDLP |
| Restart | HPR-LP Eqs. (10)-(12) and v0.1.0 source behavior |
| Adaptive sigma | published HPR-LP Eqs. (15)-(18) |
| Check interval | DCOPF manuscript value of 100 |

The HPR-LP reconstruction is pinned to v0.1.0 commit
`1941fbcfbf2dae14e4a439b22f0ea1e1c05f4a29`.

The current HPR-LP commit inspected for drift,
`0f8f1501bcc7013b53fec6822e8da91929a39d2e`, contains later sigma heuristics.
Those additions were not adopted. This report claims a sourced HPR-LP transfer,
not identity with unpublished author code.

## 3. Reversible preprocessing

### Order of operations

Each Ruiz iteration computes row and column infinity norms from the same
current matrix. It divides by their square roots and accumulates the positive
denominators. Stage 5 performs exactly 10 such iterations.

It then performs one simultaneous Pock-Chambolle step with alpha 1. This step
uses row and column L1 sums instead of infinity norms. Zero rows and columns
receive a neutral denominator of one.

Only after both diagonal stages does it compute the two normalization factors
from the complete diagonally scaled vectors:

```text
B = 1 + norm(diagonally scaled b)
C = 1 + norm(diagonally scaled c)
```

### Scaled data

Let `R` and `D` be diagonal matrices containing the cumulative row and column
denominators. The transformed LP is:

```text
A_scaled = R^-1 A D^-1
b_scaled = R^-1 b / B
c_scaled = D^-1 c / C
lower_scaled = D lower / B
upper_scaled = D upper / B
```

The positive diagonal operations preserve sparse ordering and nonzero count.

### Exact recovery

The solver state returns to original coordinates through:

```text
x = B D^-1 x_scaled
y = C R^-1 y_scaled
z = C D z_scaled
```

The variable objective identity is:

```text
original objective = B C * scaled objective
```

Stage 5 evaluates stopping, raw KKT, objective, and physical checks on the
recovered original state. Scaled residuals are retained as diagnostics but
never replace original-space acceptance.

Two component fixtures passed every round-trip and identity check. The largest
reported discrepancy was `2.45e-16`, compared with declared tolerances of
`5e-13` for round-trips and `5e-12` for transform identities.

## 4. General sGS metric

The adaptive policy and restart merit require the multiplier movement in the
actual sGS metric, not an ordinary Euclidean norm.

Split multiplier movement into equality and inequality blocks. Define:

```text
v  = A1^T Δy1 + A2^T Δy2
r1 = A1 v

Qy(Δy) =
  r1 dot solve(A1 A1^T, r1)
  + lambda * (Δy2 dot Δy2)
```

`Qy` is the squared multiplier movement in the metric used by Algorithm 2.
The equality term uses the verified equality solve, and the inequality term
uses the safeguarded spectral value from Stage 3.

The restart merit is:

```text
merit^2 =
  norm(Δx)^2 / sigma
  + 2 (A Δx) dot Δy
  + sigma Qy(Δy)
```

Unit tests compare both expressions with explicitly assembled dense
quadratics.

## 5. Restart and adaptive sigma

Policy checks occur at iterations 100, 200, 300, and so on. Following the
pinned HPR-LP v0.1.0 source, the first checkpoint forces a restart.

Later checkpoints apply three sourced criteria:

| Reason | Rule |
|---|---|
| Sufficient decay | current merit is at most `0.2 * reference merit` |
| Necessary decay without local progress | current merit is at most `0.6 * reference merit` and is worse than the previous checkpoint |
| Long inner loop | inner iterations are at least `0.2 * total iterations` |

An accepted restart makes the proximal point the new anchor and current state,
then resets the inner Halpern counter and merit history.

The adaptive update is:

```text
delta_x = norm(x_candidate - x_reference)
delta_y = sqrt(Qy(y_candidate - y_reference))
sigma_new = delta_x / delta_y
```

Both movements must be strictly inside `(1e-16, 1e12)`. The normalized
dual/primal infeasibility ratio must be strictly inside `(1e-8, 1e8)`. A guard
failure resets sigma to 1. Every checkpoint logs the merit, restart reason,
state count, old sigma, new sigma, and guard decision.

## 6. Ablation design

### Four control combinations

All control combinations use full preprocessing on the public T=1 case:

| Adaptive sigma | Restart | Horizon | Purpose |
|---|---|---|---|
| No | No | Run to acceptance | Fixed Stage 5 baseline |
| Yes | No | Fixed 5,000 | Isolate adaptation |
| No | Yes | Run to acceptance | Isolate restart |
| Yes | Yes | Run to acceptance | Full sourced policy |

Adaptive without restart is not presented as a paper algorithm. It is a
controlled, fixed-horizon component ablation. Its non-convergence is visible
and non-gating.

### Four preprocessing combinations

All preprocessing comparisons use fixed sigma, no restart, and exactly 5,000
iterations:

1. unscaled Stage 4 structural baseline;
2. norm normalization only;
3. 10 Ruiz steps plus norm normalization;
4. 10 Ruiz steps, one Pock-Chambolle step, and norm normalization.

General diagonal scaling changes the raw Equation (55) Gram structure.
Therefore the unscaled variant retains the structural backend, while the three
transformed variants use the direct equality backend. The code refuses an
invalid scaled/structural pairing.

## 7. Control-ablation results

Public case5, T=1:

| Control | Iterations | Converged | Restarts | Final sigma | Raw KKT | Max physical violation |
|---|---:|---|---:|---:|---:|---:|
| Fixed, no restart | 123,328 | Yes | 0 | 1.0000 | 0.009991 | 0.008919 MW |
| Adaptive, no restart | 5,000 fixed | No | 0 | 0.5520 | 10.3101 | 1.9685 MW |
| Fixed, restart | 524 | Yes | 5 | 1.0000 | 0.009084 | 0.001224 MW |
| Adaptive, restart | 410 | Yes | 4 | 8.9852 | 0.005618 | 0.004231 MW |

The adaptive-without-restart run completed all 50 expected policy checks at
100-iteration spacing. Its row passes the declared diagnostic requirements,
not the convergence requirements.

The full-policy T=1 restart reasons contained:

```text
forced first       1
long inner loop    3
sufficient decay   2
```

More than one reason can be true at the same checkpoint.

## 8. Preprocessing-ablation results

Public case5, T=1, after the common 5,000-iteration horizon:

| Preprocessing | Raw KKT | Objective gap | Max physical violation |
|---|---:|---:|---:|
| Unscaled structural | 0.1311 | 3.42e-4 | 0.1107 MW |
| Norm only | 0.9208 | 3.86e-3 | 0.8350 MW |
| 10 Ruiz plus norm | 0.7216 | 2.81e-3 | 0.5898 MW |
| Full preprocessing | 0.2475 | 8.79e-4 | 0.2211 MW |

None of these equal-horizon rows is labeled converged. The table isolates
conditioning choices; it is not a performance ranking or acceptance gate.

## 9. Full-policy validation

The acceptance runs use full preprocessing, adaptive sigma, and restart:

| Metric | Public case5, T=1 | Synthetic extension, T=2 |
|---|---:|---:|
| Dimensions `(n, m1, m2)` | `(15, 1, 16)` | `(36, 3, 46)` |
| Iterations | 410 | 1,032 |
| Restarts | 4 | 8 |
| Policy events | 4 | 10 |
| Sigma range | 0.5232 to 58.1205 | 0.0680 to 10.5908 |
| Final sigma | 8.9852 | 2.2433 |
| Total objective | 17,479.839089 | 26,580.274984 |
| Scaled objective gap to HiGHS | 3.31e-6 | 1.02e-5 |
| Raw KKT norm | 0.005618 | 0.008948 |
| Combined normalized Eq. (54) norm | 1.21e-5 | 9.42e-6 |
| Max physical violation | 0.004231 MW | 0.006059 MW/MWh |
| Local wall time | 0.143 s | 0.558 s |

Both cases pass:

- every separately normalized Equation (54) block at `5e-5`;
- raw KKT at the Stage 5 target `0.01`;
- maximum physical violation at `0.01 MW/MWh`;
- scaled objective gap at `2e-4`;
- equality-solve and projected-state identity checks.

The largest equality-solve infinity residual was `1.67e-16`. The largest
`z`/projected-`x` identity error was `3.21e-15`.

Initial-sigma sensitivity also passed on T=1:

| Initial sigma | Iterations | Final sigma | Restarts | Raw KKT |
|---:|---:|---:|---:|---:|
| 0.1 | 477 | 57.1042 | 4 | 0.007105 |
| 10 | 532 | 30.2961 | 5 | 0.009913 |

## 10. Verification

The reproducible evidence command is:

```powershell
./.venv/Scripts/python.exe scripts/run_stage_5.py
```

The independent evidence check is:

```powershell
./.venv/Scripts/python.exe scripts/check_stage_5.py
```

It writes:

- `results/raw/stage_5/stage_5_validation.json`;
- `results/raw/stage_5/stage_5_trajectories.jsonl.gz`.
- `results/raw/stage_5/stage_5_checks.json`.

Quality checks:

| Check | Result |
|---|---|
| Stage 5 unit and integration tests | **39 passed** |
| Full Python test suite | **143 passed** |
| Independent evidence checker | **23 / 23 passed** |
| Stage 5 evidence | **PASS** |
| Ruff lint | PASS |
| Ruff formatting | PASS |

The evidence records Python 3.13.5, NumPy 2.4.1, SciPy 1.16.3, Windows 11,
CPU FP64, and SciPy HiGHS dual simplex.

## 11. Limitations and acceptance

Known limits remain explicit:

1. The exact author DCOPF restart and adaptive-sigma code is unavailable.
2. The reconstruction is sourced from HPR-LP and uses the DCOPF interval; it
   is not claimed to be author-identical.
3. Adaptive without restart is a non-paper fixed-horizon ablation and did not
   converge; this is non-gating.
4. A scaled Equation (55) structural backend has not been derived. Scaled runs
   use direct equality solves.
5. Paper benchmark input files and construction scripts remain unavailable.
6. Stage 5 contains no GPU execution, DGX result, or GPU timing claim.

Acceptance checklist:

- [x] Exactly 10 simultaneous Ruiz steps are implemented and logged.
- [x] One Pock-Chambolle alpha-1 step is implemented and logged.
- [x] `b` and `c` normalization follows diagonal scaling and uses full vectors.
- [x] LP, state, residual, and objective recovery identities pass.
- [x] Original-space stopping, KKT, objective, and physical checks pass.
- [x] The first restart is forced and later policy checks follow a 100 cadence.
- [x] Adaptive sigma uses the general sGS metric and explicit guards.
- [x] Four control and four preprocessing combinations are preserved.
- [x] Initial-sigma sensitivity at 0.1 and 10 converges.
- [x] Unsupported author-specific details remain labeled.
- [x] Stage 6 GPU work has not started.

**Stage 5 result: PASS.**

## 12. Exact command required for Stage 6

The project is stopped. To approve the DGX GPU port, send exactly:

```text
APPROVE STAGE 5 AND RUN STAGE 6
```
