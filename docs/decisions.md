# Decision log

This is an append-only research decision log. A decision can be superseded, but
its original rationale should remain reviewable.

## D-0001 - Treat the paper PDF as the primary specification

- **Status:** accepted
- **Stage:** 0
- **Decision:** Preserve the supplied PDF unchanged, record its hash, and map
  implementation choices to printed equations or clearly named external
  references.
- **Reason:** Exact wording and sign conventions matter more than assumptions
  based on general DCOPF or PDLP practice.

## D-0002 - Enforce explicit stage gates

- **Status:** accepted
- **Stage:** 0
- **Decision:** Execute one stage at a time. Start Stage \(N+1\) only after the
  user sends `APPROVE STAGE N AND RUN STAGE N+1`.
- **Reason:** Each optimization layer needs an independently validated baseline.
  A later speed improvement must not conceal an earlier mathematical error.

## D-0003 - Reproduce the published base-case DCOPF first

- **Status:** accepted
- **Stage:** 0
- **Decision:** Do not add line-outage contingencies, LODF rows, or adaptive
  constraint generation to Stages 0-9.
- **Reason:** Eqs. (1)-(10) contain base-case shift-factor line limits but no
  contingency index. The optional N-1 work is new research and remains Stage 10.

## D-0004 - Use one canonical inequality orientation

- **Status:** accepted
- **Stage:** 0
- **Decision:** Store all non-box inequalities as \(A_2x\ge b_2\), with
  \(y_2\ge0\).
- **Reason:** This is the convention in Eqs. (15)-(16) and determines the sign
  of the projected multiplier update.

## D-0005 - Keep simple limits in the box projection

- **Status:** accepted with one open detail
- **Stage:** 0
- **Decision:** Equations (3), (7), and (10) belong to \(C=[l,u]\), not \(A_2\).
  Conventional-generator bounds will be tested both as explicit box bounds and
  as bounds implied by nonnegative reserve plus Eq. (4) before the Stage 2 split
  is finalized.
- **Reason:** The compact-form prose puts variable bounds in \(C\), but the
  manuscript does not print a standalone conventional-generation box equation.

## D-0006 - Adopt the Eq. (55) equality-row structure

- **Status:** accepted
- **Stage:** 0
- **Decision:** Use \(m_1=T+N_{ESS}\): \(T\) power-balance rows and
  \(N_{ESS}\) terminal-energy rows.
- **Reason:** Eq. (9), Eq. (55), Section II-C, and every Table II dimension
  support this count. Appendix A's claim of \(T(1+N_{ESS})\) is inconsistent
  and is recorded as a likely manuscript error.

## D-0007 - Treat Proposition 5 as untrusted until cross-checked

- **Status:** accepted
- **Stage:** 0
- **Decision:** Implement a direct \(A_1A_1^T\) solve first. In Stage 4, compare
  every structural result with that oracle.
- **Reason:** Eq. (43) and Eqs. (39)/(44)/(45) have inconsistent rank-one
  inverse signs.

## D-0008 - Build a CPU FP64 reference before GPU work

- **Status:** accepted
- **Stage:** 0
- **Decision:** Stages 1-5 use correctness-oriented CPU linear algebra and FP64.
  The GPU backend begins only in Stage 6.
- **Reason:** It isolates mathematical errors from device kernels, sparse-format
  behavior, asynchronous execution, and reduced-precision effects.

## D-0009 - Use HiGHS as the mandatory LP reference

- **Status:** accepted
- **Stage:** 0
- **Decision:** Use SciPy HiGHS or `highspy` as the default independent solver.
  Use Gurobi only when installed and licensed.
- **Reason:** The project must remain open-source functional and must not depend
  on proprietary licensing.

## D-0010 - Start with fixed \(\sigma=1\)

- **Status:** accepted
- **Stage:** 0
- **Decision:** The first full CPU method uses \(\sigma=1\), no restart, and no
  inferred adaptive update.
- **Reason:** The paper supplies the initial value but not the adaptive formula
  or restart rule.

## D-0011 - Keep the dashboard evidence-driven

- **Status:** accepted
- **Stage:** 0
- **Decision:** Dashboard stage/task state is versioned with the project and
  updated only from completed checks and reports. Browser storage is not an
  authoritative source.
- **Reason:** Research state should survive browser sessions and remain tied to
  the evidence that justified each transition.

## D-0012 - Withhold the reproduction classification

- **Status:** accepted
- **Stage:** 0
- **Decision:** Do not call the current work exact, near-exact, or benchmark
  equivalent.
- **Reason:** Device placement, time-series inputs, several physical parameters,
  adaptive rules, precision, and timing boundaries are not fully specified.

## D-0013 - Audit the actual DGX Spark before selecting GPU packages

- **Status:** accepted
- **Stage:** 0
- **Decision:** Base Python, CuPy, and CUDA package choices on the detected
  aarch64/CUDA/driver environment. Do not copy package pins from the paper's
  A100 system.
- **Reason:** The reproduction target differs materially from the paper's A100
  platform, and Python GPU wheel support is platform-sensitive.

## D-0014 - Diagonalize the generic HPR y metric

- **Status:** accepted
- **Stage:** 1
- **Decision:** For the correctness reference, construct
  \(\mathcal T_1=\tau I-AA^T\), where
  \(\tau=\lambda_{\max}(AA^T)+\delta\) and \(\delta>0\) is a recorded numerical
  margin. Verify both \(\mathcal T_1\succeq0\) and
  \(AA^T+\mathcal T_1=\tau I\succ0\).
- **Reason:** Projecting an unconstrained solve onto \(D\) is generally wrong
  for a correlated metric. The scalar metric makes the Euclidean projection
  exact and keeps Algorithm 1 readable.

## D-0015 - Keep Eq. (28) and Eq. (54a) separate

- **Status:** accepted
- **Stage:** 1
- **Decision:** Return the full Eq. (28) projected KKT mapping and the three
  separately normalized Eq. (54) stopping blocks under distinct names.
- **Reason:** Eq. (28)'s first block depends on \(y\) and includes projected
  complementarity. Eq. (54a) checks only primal feasibility; conflating them
  would hide a sign or complementarity error.

## D-0016 - Use scale-aware Stage 1 acceptance thresholds

- **Status:** accepted
- **Stage:** 1
- **Decision:** Retain the paper's \(5\times10^{-5}\) tolerance for each
  normalized Eq. (54) block. On the four unit-scaled reference LPs, also
  require raw combined Eq. (28) norm at most \(2.5\times10^{-4}\), maximum
  primal violation at most \(2.5\times10^{-4}\), and scaled objective gap to
  HiGHS at most \(2\times10^{-4}\).
- **Reason:** The paper's stopping tests are normalized separately; imposing
  \(5\times10^{-5}\) on an unnormalized combined norm would be a different,
  scale-dependent rule. The additional thresholds are explicit Stage 1
  validation targets, not claimed paper settings.

## D-0017 - Plant feasibility and KKT structure in the random LP

- **Status:** accepted
- **Stage:** 1
- **Decision:** Use seed `20260729` to construct and then preserve a
  five-variable LP with a known feasible point, selected active constraints,
  nonnegative inequality multipliers, and a full-rank active system.
- **Reason:** Arbitrary random LPs can be infeasible, unbounded, or have
  nonunique solutions. A planted KKT point gives an independent oracle while
  still exercising nonsymmetric, nontrivial data.

## D-0018 - Keep Stage 1 local and CPU-only

- **Status:** accepted
- **Stage:** 1
- **Decision:** Run generic dense FP64 validation on the local workstation with
  NumPy, SciPy HiGHS, pytest, and Ruff. Do not install the DGX numerical stack
  or run GPU code in this stage.
- **Reason:** Stage 1 isolates mathematical correctness. DGX environment
  construction and GPU residence are later gated work.

## D-0019 - Pin MATPOWER 8.1 case5 for Stage 2

- **Status:** accepted
- **Stage:** 2
- **Decision:** Use the unmodified `case5.m` file from the immutable MATPOWER
  8.1 release as the first public network input. Record both its upstream Git
  blob and the checked-in SHA-256 digest.
- **Reason:** It is small, connected, and its five generation-cost rows are
  genuinely linear. This avoids silently modifying a quadratic objective while
  keeping the input independent of the paper authors' unavailable instances.

## D-0020 - Preserve affine transformer phase-shift flows

- **Status:** accepted
- **Stage:** 2
- **Decision:** Represent DC branch flow as `flow = H @ injection + offset`.
  Transformer tap ratios modify branch susceptance, and nonzero phase shifts
  contribute the affine offset.
- **Reason:** A PTDF matrix alone describes sensitivities. Dropping the phase
  offset can produce internally consistent but physically wrong line flows.

## D-0021 - Distinguish topology branches from thermal rows

- **Status:** accepted
- **Stage:** 2
- **Decision:** Retain every active branch in the DC topology. Add Eq. (2)
  rows only when MATPOWER `RATE_A` is positive and below its unlimited sentinel.
  Put conventional output limits in the box while retaining Eq. (4) reserve
  coupling.
- **Reason:** In MATPOWER, `RATE_A=0` denotes no thermal limit; it does not
  remove the branch. Finite box bounds are also required by the canonical LP
  and safely preserve generator limits when reserve requirements are zero.

## D-0022 - Separate the public base case from synthetic device coverage

- **Status:** accepted
- **Stage:** 2
- **Decision:** Solve an unmodified one-period public case with no renewable or
  storage devices, then use a separate, explicit two-period synthetic extension
  to exercise reserve, ramp, renewable, storage-energy, and terminal-energy
  rows.
- **Reason:** The paper does not publish the required device placements or time
  series. A labeled validation fixture tests the mathematics without implying
  that invented data came from the authors.
