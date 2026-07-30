# Mathematical notes

## Purpose

These notes explain the implementation logic behind the paper specification.
They are deliberately more tutorial-like than `paper_specification.md`. The
specification records what the manuscript says; this file records how we will
reason about it, test it, and avoid common sign and dimension mistakes.

No solver is implemented in Stage 0.

## 1. The canonical LP is the contract

The entire reproduction will use one canonical form:

\[
\begin{aligned}
\min_x\quad &c^Tx\\
\text{s.t.}\quad&A_1x=b_1,\\
&A_2x\ge b_2,\\
&l\le x\le u.
\end{aligned}
\]

This is not just notation. It fixes three implementation decisions:

1. equality rows and inequality rows remain separate;
2. every inequality is oriented as "greater than or equal to";
3. simple bounds stay in the box \(C=[l,u]\), not in \(A_2\).

The dual inequality multiplier is consequently nonnegative:
\(y_2\in\mathbb R_+^{m_2}\). A row accidentally written in the opposite
direction changes the sign of its multiplier and breaks the projected update.

### Sign-check example

The upward reserve headroom rule is

\[
r_i^u(t)\le \bar P_{G,i}-p_{G,i}(t).
\]

Moving terms and orienting the row as \(A_2x\ge b_2\) gives

\[
-p_{G,i}(t)-r_i^u(t)\ge-\bar P_{G,i}.
\]

A useful Stage 2 test will evaluate both expressions at the same random
physical point and assert that their feasibility booleans agree.

## 2. Variable order and dimensions

The implementation-facing variable order is

\[
x=(p_G,p_{RG},p_{ESS}^{dc},p_{ESS}^{ch},r^u,r^d).
\]

Per period, the block widths are

\[
N_G,\quad N_{RG},\quad N_{ESS},\quad N_{ESS},\quad N_G,\quad N_G.
\]

Therefore

\[
n=T(3N_G+N_{RG}+2N_{ESS}).
\]

The equality rows are:

- \(T\) system-balance equations;
- \(N_{ESS}\) terminal-energy equations.

Hence \(m_1=T+N_{ESS}\). This is important because Appendix A's prose gives a
different count, while Eq. (55) and all published table dimensions support
\(T+N_{ESS}\).

The inequality-row count is

\[
m_2=
2TN_L
+2TN_G
+2T
+2(T-1)N_G
+2TN_{ESS}.
\]

The first and last terms often dominate. The line-flow block is also what makes
the matrix-vector products the central GPU workload.

## 3. Storage is modeled without an energy-state variable

The paper does not add a state-of-charge variable for every period. Instead,
the energy at time \(t\) is an affine cumulative sum:

\[
E_i(t)=E_i(0)+\sum_{\tau=1}^{t}
\left(\eta_i^{ch}p_i^{ch}(\tau)
-p_i^{dc}(\tau)/\eta_i^{dc}\right)\Delta t.
\]

This choice has two consequences:

- energy-bound rows are lower-triangular in time;
- the terminal-energy equality couples every period for one storage device.

It also explains the diagonal-plus-low-rank structure of \(A_1A_1^T\).

The published LP permits simultaneous charging and discharging because it has
no binary complementarity constraint. The loss penalty may discourage that
behavior, but it does not make it impossible. We will reproduce the LP as
written rather than silently add a mixed-integer restriction.

## 4. What the three KKT residual blocks mean

The paper's residual map is

\[
\mathcal R(w)=
\begin{pmatrix}
y-\Pi_D(y-Ax+b)\\
x-\Pi_C(x-z)\\
c-A^Ty-z
\end{pmatrix}.
\]

Each block answers a different question:

1. **Primal feasibility:** are equality rows satisfied and are inequality
   violations absent?
2. **Box/complementarity:** is \(z\) consistent with the normal cone of the
   variable bounds?
3. **Stationarity:** do cost and constraint gradients balance?

An objective value alone is not enough. A vector can have an attractive
objective and still violate line limits or stationarity.

The stopping test in Eq. (54) normalizes each block separately. We will record
the raw and normalized values so that a later change of tolerance cannot hide
a regression.

## 5. Algorithm 2 state anatomy

Four states must stay distinct:

- \(w^k\): current iterate;
- \(\bar w^{k+1}\): proximal/intermediate iterate;
- \(\hat w^{k+1}=2\bar w^{k+1}-w^k\): reflected iterate;
- \(w^0\): fixed Halpern anchor.

The update order is:

1. update \(\bar z\);
2. update \(\bar x\);
3. solve the first equality-multiplier sweep
   \(\bar y_1^{k+1/2}\);
4. project the inequality multiplier \(\bar y_2^{k+1}\);
5. solve the second equality-multiplier sweep
   \(\bar y_1^{k+1}\);
6. reflect;
7. combine the fixed anchor and reflected point.

The two \(y_1\) solves are not duplicates. The first uses \(y_2^k\); the second
uses the new \(\bar y_2^{k+1}\). Omitting the second sweep changes the method.

## 6. The box projection collapses two paper updates

Equations (33) and (34) imply

\[
\bar x^{k+1}=\Pi_C\left[x^k+\sigma(A^Ty^k-c)\right].
\]

Then

\[
\bar z^{k+1}
=\frac{\bar x^{k+1}-x^k-\sigma(A^Ty^k-c)}{\sigma}.
\]

This identity is valuable for both correctness and performance:

- Stage 3 will verify it numerically on small random vectors;
- Stage 6 can fuse or reuse buffers only after the identity passes on CPU.

## 7. Why the structural equality solve needs an oracle

The first implementation will solve

\[
A_1A_1^Ty_1=R_1
\]

with a trusted direct CPU solver. Only after that is validated will we replace
it with Proposition 5's diagonal and rank-one operations.

That ordering matters because the manuscript contains a sign inconsistency:
Eq. (43) is a minus rank-one update, while Eqs. (39), (44), and (45) print the
inverse pattern for a plus update. A direct solve will tell us which expression
matches the actual matrix produced by Eq. (55).

## 8. Why the \(y_2\) step can remain matrix-free

The paper chooses

\[
\mathcal S_2=\lambda I-A_2A_2^T,\qquad
\lambda=\lambda_{\max}(A_2A_2^T)=\|A_2\|_2^2.
\]

This makes the quadratic block

\[
A_2A_2^T+\mathcal S_2=\lambda I
\]

and yields an inexpensive nonnegative projection. We should not materialize
\(\mathcal S_2\). We need only a defensible value of \(\lambda\), SpMV, and
transpose-SpMV operations.

For small cases, dense eigenvalues, sparse eigensolvers, and deterministic
power iteration will be cross-checked. A conservative overestimate preserves
positive semidefiniteness; an underestimate may invalidate the method.

## 9. What "GPU speedup" will mean here

The paper's published comparisons use an A100 for GPU methods and a separate
Intel workstation for Gurobi. We will preserve those numbers as reported
results, but a new DGX Spark result will use explicit timing boundaries:

- one-time CUDA initialization;
- matrix construction and preprocessing;
- host-to-device transfer;
- solver initialization;
- iterations and periodic residual checks;
- device-to-host transfer;
- complete end-to-end time.

Only like-for-like boundaries, precisions, stopping tests, and warm-up states
will be divided to form a speedup.

## 10. Reproduction ladder

The project distinguishes:

- **exact reproduction:** same full inputs, construction, solver rules, and
  comparable environment;
- **mathematical reproduction:** same stated model and algorithm on available
  data;
- **structural reproduction:** same sparsity and scaling structure with
  transparent reconstructed inputs;
- **approximate benchmark reconstruction:** a disclosed attempt to approach
  reported dimensions or timings with incomplete inputs.

Stage 0 does not assign a final class. The current evidence is insufficient for
an exact numerical claim.

## 11. What Stage 1 validated

Stage 1 implements generic Algorithm 1, not the paper-specific sGS Algorithm 2.
That distinction is deliberate. The generic implementation tests the common
mathematical spine:

1. project the primal candidate onto the box;
2. recover the associated \(z\) normal-cone variable;
3. solve the complete dual \(y\) subproblem;
4. reflect the proximal point;
5. apply the fixed-anchor Halpern average.

Power-system matrices, the two \(y_1\) sweeps, and the special \(y_2\) update
remain locked. If the generic method cannot solve a two-variable LP with known
KKT multipliers, adding DCOPF structure would only make the error harder to
find.

## 12. Why the Stage 1 y projection is exact

For a general positive-definite metric

\[
H=AA^T+\mathcal T_1,
\]

the constrained \(y\) subproblem is a metric projection. Solving the
unconstrained system and clipping negative inequality multipliers is generally
wrong when \(H\) has off-diagonal coupling.

Stage 1 therefore selects

\[
\mathcal T_1=\tau I-AA^T,\qquad
\tau>\lambda_{\max}(AA^T).
\]

Then \(H=\tau I\), so the metric is merely a positive scalar times the
Euclidean metric. Projection onto
\(D=\mathbb R^{m_1}\times\mathbb R_+^{m_2}\) becomes exactly:

\[
\bar y_1=g_1/\tau,\qquad
\bar y_2=\max(g_2/\tau,0).
\]

The implementation computes the eigenvalue and a positive margin on these tiny
matrices, constructs \(\mathcal T_1\) explicitly, and checks its eigenvalues.
This is a correctness device, not the later large-scale preconditioner.

## 13. Feasibility stopping is not the full KKT mapping

The paper's Eq. (54a) uses

\[
\Pi_D(b-Ax),
\]

which measures equality error and positive inequality violation. Eq. (28)
instead uses

\[
y-\Pi_D(y-Ax+b),
\]

whose inequality block also tests multiplier complementarity. Stage 1 returns
and tests both. On the analytic LP, moving from \(x_1=0.4\) to \(x_1=0.3\)
produces opposite-signed inequality components in the two expressions; that
hand calculation catches a common implementation sign error.

## 14. Why an objective can look slightly better than the optimum

The HPR reference stops at a finite residual tolerance. Its analytic-toy
solution has objective \(1.399841\), slightly below the exact feasible optimum
\(1.4\), because its active inequality is still violated by about
\(9.2\times10^{-5}\). This is not a superior solution.

This is a useful optimization lesson: objective comparisons are meaningful
only after feasibility is checked at the same tolerance. HiGHS and the analytic
KKT point remain the independent references, while the HPR trajectory shows
how a first-order method approaches them.

## 15. What Stage 2 validated

Stage 2 turns the paper's symbolic DCOPF into sparse CPU matrices without
introducing the paper-specific solver. The separation is deliberate:

1. parse and validate a public network;
2. build the DC network physics and PTDF;
3. map physical variables and rows into the canonical LP;
4. solve with independent HiGHS;
5. recompute every physical constraint directly from the returned dispatch.

This makes the model itself an independently tested input to Stage 3.

## 16. A PTDF can be affine, not only linear

For balanced bus injections `p`, Stage 2 evaluates branch flow as:

```text
flow = H p + f_shift
```

The constant `f_shift` is zero for ordinary lines but can be nonzero for a
phase-shifting transformer. Transformer tap ratios also change the branch
susceptance used to construct `H`.

The test suite includes a three-bus loop with a non-unity tap and a 10-degree
phase shift. PTDF flows match a separate reduced-angle solve, including the
phase-driven loop flow.

## 17. Six topology branches do not imply six thermal constraints

MATPOWER case5 contains six active branches, but four have `RATE_A=0`.
MATPOWER interprets zero as "unlimited," not "out of service." Those four
branches still shape the bus-susceptance matrix and every PTDF column.

Only the two branches with positive finite limits contribute Eq. (2) rows.
This distinction prevents either of two common mistakes:

- deleting an electrically active branch because it has no thermal rating;
- interpreting a zero rating as a zero-flow constraint.

## 18. Why the public and synthetic cases remain separate

The one-period base case uses the unmodified public network and its exact
linear costs. It has no renewable or storage devices and does not invent
reserve requirements.

The two-period extension is explicitly synthetic. Its 10 MW generator ramp
limits force storage to charge in period 1 and discharge in period 2. That
nonzero trajectory tests the cumulative energy rows and terminal equality,
while its label prevents confusion with the authors' unavailable inputs.

## 19. Stage 3 implements the printed Algorithm 2 order

The CPU reference keeps four states distinct:

- `w0`: the fixed all-zero Halpern anchor;
- `wk`: the current iterate;
- `w_bar`: the intermediate proximal point used for stopping;
- `w_hat = 2 w_bar - wk`: the reflected point.

For `y = (y1, y2)` and fixed `sigma = 1`, one iteration performs:

```text
1. z_bar
2. x_bar
3. y1_half       first direct equality solve
4. y2_bar        nonnegative projected update
5. y1_bar        second direct equality solve
6. w_hat         reflection
7. w_next        fixed-anchor Halpern average
```

The first two equations are evaluated as:

```text
q     = x + sigma (A^T y - c)
z_bar = (project_box(q) - q) / sigma
x_bar = x + sigma (A^T y + z_bar - c)
```

The implementation independently checks that `x_bar = project_box(q)`. This
coding order mirrors Equations (33) and (34) instead of merely producing an
algebraically equivalent final vector.

## 20. Why there are two equality solves

Let `G1 = A1 A1^T`. For an inequality multiplier vector `v`, define:

```text
rhs1(v) =
  [b1 - A1 {x_bar + sigma (A2^T v + z_bar - c)}] / sigma
```

Algorithm 2 first solves:

```text
G1 y1_half = rhs1(y2_current)
```

It then updates `y2`, and solves again:

```text
G1 y1_bar = rhs1(y2_bar)
```

The second solve is not redundant: the projected inequality multiplier has
changed. Stage 3 verifies that `A1` has full row rank, checks the raw Gram
matrix's symmetry, confirms positive definiteness, factors it with Cholesky,
and records both absolute infinity-norm and relative solve residuals.

## 21. The projected y2 step and spectral safeguard

With:

```text
lambda = largest eigenvalue of A2 A2^T = ||A2||_2^2
Ry = x_bar / sigma
     + A1^T y1_half
     + A2^T y2_current
     + z_bar - c
```

Equation (50) becomes:

```text
y2_bar = project_nonnegative(
  y2_current + [b2 / sigma - A2 Ry] / lambda
)
```

The code never constructs `S2 = lambda I - A2 A2^T`. On the six Stage 3
correctness cases, dense eigendecomposition, sparse `eigsh`, and deterministic
power iteration agree. The value actually used is the largest estimate plus a
small positive FP64 margin, so `S2` remains positive semidefinite even at the
top eigenvalue.

## 22. Paper stopping and validation targets are different tests

The paper stops when each separately normalized Equation (54) block is at most
`5e-5`. Stage 3 evaluates those blocks on `w_bar` every iteration.

The project also reports the raw Equation (28) KKT mapping. For unit-scale toy
LPs the target remains `2.5e-4`. For the MW-scaled DCOPF cases, the stated raw
combined target is `0.02`; the physical candidate threshold is `0.01 MW/MWh`;
and the scaled objective gap to HiGHS must be no more than `2e-4`.

These are additional validation thresholds, not alternative paper settings.
In particular, the normalized paper tolerance should not be read as a raw
`5e-5 MW` requirement.

## 23. How approximate DCOPF candidates are checked physically

The strict Stage 2 validator requires power balance to near machine precision
before evaluating branch flows. A finite-tolerance first-order candidate
cannot satisfy that precondition at its first valid Equation (54) iterate.

Stage 3 therefore uses a separately labeled candidate validator:

1. compute and retain the candidate's original power-balance error;
2. test that error against `0.01 MW`;
3. leave the candidate decision vector unchanged;
4. for PTDF-versus-angle flow comparison only, absorb the measured imbalance
   at the reference bus;
5. independently test every other physical family on the original vector.

This does not convert the iterate into a strict power-flow solution. It makes
the reference-slack convention explicit while keeping the actual imbalance in
the acceptance record.
