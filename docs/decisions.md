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

## D-0023 - Use direct Cholesky solves as the Stage 3 equality oracle

- **Status:** accepted
- **Stage:** 3
- **Decision:** Factor the verified FP64 matrix `A1 A1^T` once and use the
  factor for both printed equality-multiplier sweeps.
- **Reason:** This is a trusted correctness baseline for Algorithm 2. The
  manuscript's disputed Proposition 5 signs remain locked until Stage 4 can
  compare the structural formula against this oracle.

## D-0024 - Cross-check lambda and use a conservative overestimate

- **Status:** accepted
- **Stage:** 3
- **Decision:** On correctness-scale cases, compute the largest eigenvalue of
  `A2 A2^T` with dense symmetric eigendecomposition, sparse `eigsh`, and seeded
  power iteration. Use the largest estimate plus a positive FP64 margin.
- **Reason:** An underestimate can invalidate the positive-semidefinite
  proximal block. The dense result is affordable and authoritative at this
  stage; the other two methods expose estimator mistakes before larger cases.

## D-0025 - Check Equation (54) every iteration and store sparse history

- **Status:** accepted
- **Stage:** 3
- **Decision:** Evaluate all three Equation (54) tests on the intermediate
  state every iteration. Persist iteration 1, every 250th iteration, and the
  exact stopping iteration.
- **Reason:** The residual trajectory oscillates, so checking only at storage
  intervals changes the first accepted iterate. Sparse storage preserves the
  trajectory without turning the evidence file into an unnecessarily large
  artifact.

## D-0026 - Validate first-order DCOPF candidates without weakening the strict oracle

- **Status:** accepted
- **Stage:** 3
- **Decision:** Keep the Stage 2 strict physical validator unchanged. Add a
  separately named approximate-candidate validator with a `0.01 MW/MWh`
  physical threshold. Report the original balance error, and adjust only the
  temporary flow-check injection at the reference bus.
- **Reason:** An Equation (54)-tolerant first-order iterate is not balanced to
  the `1e-9` level required by the strict PTDF oracle. The separate path tests,
  rather than hides, that imbalance while still allowing independent PTDF and
  angle-flow comparison under the model's reference-slack convention.

## D-0027 - State separate raw KKT and objective targets for DCOPF

- **Status:** accepted
- **Stage:** 3
- **Decision:** In addition to the paper's separately normalized `5e-5`
  stopping tests, require raw Equation (28) combined norm at most `0.02`,
  scaled objective gap to HiGHS at most `2e-4`, and raw physical violation at
  most `0.01 MW/MWh`.
- **Reason:** Raw residual magnitudes scale with the MW right-hand sides and
  cost vector. These explicit validation targets prevent a normalized stop
  from being mistaken for an unscaled `5e-5` KKT guarantee.

## D-0028 - Decompose CPU timing boundaries

- **Status:** accepted
- **Stage:** 3
- **Decision:** Record preparation, iteration-loop, and total elapsed time
  separately. Preparation includes sparse conversion, spectral cross-checks,
  and the Cholesky factorization.
- **Reason:** The paper does not define its timing boundary. A named
  decomposition prevents preprocessing costs from disappearing into an
  ambiguous "solver time" and prepares the project for fair CPU/GPU timing.

## D-0029 - Resolve Proposition 5 from the Schur complement

- **Status:** accepted; resolves D-0007
- **Stage:** 4
- **Decision:** Retain Eq. (43)'s minus rank-one Schur complement and use the
  corresponding positive Sherman-Morrison correction. Do not implement the
  negative correction and plus denominator printed in Eqs. (39), (44), and
  (45).
- **Reason:** The implemented Eq. (55) matrix produces the minus update
  exactly. The corrected expression matches direct Cholesky solves to FP64
  accuracy on deterministic no-, one-, and multiple-storage cases. The printed
  expression leaves a material linear-system residual.

## D-0030 - Require an exact structural compatibility check

- **Status:** accepted
- **Stage:** 4
- **Decision:** Enable the structural backend only through a descriptor that
  validates dimensions, block order, sparse row patterns, coefficients,
  storage diagonals, and a positive Schur complement against the same
  `CanonicalLP` instance.
- **Reason:** Proposition 5 is valid because of the implemented Eq. (55)
  structure, not for an arbitrary equality matrix. Refusing an incompatible
  LP prevents a fast but mathematically unrelated solve.

## D-0031 - Use the mean and zero-mean Schur decomposition

- **Status:** accepted
- **Stage:** 4
- **Decision:** Evaluate
  \((aI-\alpha\mathbf1\mathbf1^T)^{-1}\) by splitting the reduced right-hand
  side into its mean and zero-mean components. Compute the all-ones eigenvalue
  with the cancellation-resistant identity
  \[
  a-\alpha T=N_G+N_{RG}+
  \sum_s\frac{(\eta_s^{ch}-1/\eta_s^{dc})^2}
  {(\eta_s^{ch})^2+(1/\eta_s^{dc})^2}.
  \]
- **Reason:** This is algebraically equivalent to the corrected
  Sherman-Morrison formula but avoids subtracting nearly equal terms when many
  ideal-efficiency storage devices are present.

## D-0032 - Separate small-fixture timing from structural scaling evidence

- **Status:** accepted
- **Stage:** 4
- **Decision:** Report direct and structural runtimes on the two validation
  DCOPF cases without requiring a speedup. Measure the performance advantage
  on warmed synthetic Eq. (55) systems large enough that Python call overhead
  does not dominate, and label the result as local empirical evidence.
- **Reason:** A precomputed library Cholesky call can be faster than several
  NumPy vector operations on a three-row system. That fact does not contradict
  the structural method's linear storage and arithmetic scaling.

## D-0033 - Gate complexity on the paper-relevant equality-update boundary

- **Status:** accepted after two disclosed timing-protocol failures
- **Stage:** 4
- **Decision:** Use the measured right-hand-side formation plus structural
  solve as the empirical complexity gate. Preserve the solve-only log-log fit
  as a non-gating diagnostic, while retaining its exact \(O(T+N_{ESS})\)
  operation count.
- **Reason:** Corollary 1 explicitly combines the Proposition 5 formulas with
  the expression for \(R_1\), so sparse \(A_1\) products belong to the claimed
  \(O(T(N_G+N_{RG}+N_{ESS}))\) boundary. Two full runs showed that the required
  boundary passed the unchanged slope, \(R^2\), and normalized-spread limits,
  while the much shorter NumPy solve-only kernel was distorted by fixed
  dispatch, allocation, cache, and memory-bandwidth effects. The failed
  solve-only fits remain in the Stage 4 log and report; they are not relabeled
  as proof of linear wall time.

## D-0034 - Pin the published HPR-LP policy boundary

- **Status:** accepted
- **Stage:** 5
- **Decision:** Reconstruct restart behavior from HPR-LP v0.1.0 at commit
  `1941fbcfbf2dae14e4a439b22f0ea1e1c05f4a29` and reconstruct adaptive sigma
  from the published HPR-LP Eqs. (15)-(18). Audit the current HPR-LP commit
  separately for source drift, but do not adopt its later extra heuristics.
- **Reason:** The DCOPF manuscript delegates these details and does not publish
  its implementation. A fixed historical source plus published equations is
  reproducible; silently following a moving repository is not.

## D-0035 - Use simultaneous diagonal steps in the printed order

- **Status:** accepted
- **Stage:** 5
- **Decision:** Apply 10 simultaneous Ruiz row/column infinity-norm steps, one
  simultaneous Pock-Chambolle row/column L1 step with alpha 1, and then
  normalize the complete diagonally scaled `b` and `c` vectors by
  `1 + norm(vector)`. Give zero rows and columns a neutral denominator of one.
- **Reason:** This order matches the manuscript's numerical settings and the
  sourced implementations. Computing a column denominator after mutating the
  rows would be a different sequential algorithm.

## D-0036 - Make preprocessing exactly reversible

- **Status:** accepted
- **Stage:** 5
- **Decision:** Retain cumulative row, column, `b`, and `c` factors; expose
  forward and inverse LP/state maps; and make all stopping, KKT, objective, and
  physical acceptance decisions in recovered original coordinates.
- **Reason:** Scaled residuals are useful solver diagnostics but are not
  physical MW/MWh statements. Exact recovery identities prevent favorable
  scaled conditioning from hiding an incorrect change of problem.

## D-0037 - Preserve the forced first restart and 100-iteration cadence

- **Status:** accepted
- **Stage:** 5
- **Decision:** Force the first restart at iteration 100 as in the pinned
  HPR-LP v0.1.0 source. At later 100-iteration checkpoints, evaluate the
  HPR-LP sufficient-decay, necessary-decay/no-progress, and long-inner-loop
  rules with parameters `0.2`, `0.6`, and `0.2`. Restart from the proximal
  state and reset the inner Halpern counter and merit reference.
- **Reason:** The manuscript prints the cadence but not the criterion or state
  transition. The pinned source supplies a testable rule, including an
  easy-to-miss first-check special case.

## D-0038 - Evaluate adaptive sigma in the general sGS metric

- **Status:** accepted
- **Stage:** 5
- **Decision:** Generalize the published HPR-LP movement ratio by evaluating
  multiplier movement in `A A^T + T1`, using the verified equality solve and
  the safeguarded inequality spectral value. Retain the published movement and
  infeasibility-ratio guards and reset sigma to 1 when a guard fails.
- **Reason:** The diagonal HPR-LP special case is not the metric used by
  Algorithm 2. The general quadratic is the sourced mathematical transfer
  appropriate to sGS-HPR.

## D-0039 - Keep the raw structural baseline after scaling is added

- **Status:** accepted
- **Stage:** 5
- **Decision:** Preserve the unscaled Stage 4 structural backend as an
  ablation. Use the direct equality backend after normalization, Ruiz, or
  Pock-Chambolle scaling, and reject attempts to reuse the raw Equation (55)
  descriptor on a generally scaled matrix.
- **Reason:** General diagonal scaling changes the Gram structure on which the
  Stage 4 formula depends. Refusal is safer than presenting an unproved scaled
  structural solve as exact.

## D-0040 - Treat fixed-horizon component isolation as non-gating

- **Status:** accepted
- **Stage:** 5
- **Decision:** Run all four preprocessing variants for the same 5,000
  iterations and run adaptive-without-restart for the same fixed horizon.
  Require finite deterministic completion, correct event cadence, and
  preserved identities, but do not require convergence of those diagnostic
  runs. Require convergence for fixed/no-restart, fixed/restart,
  adaptive/restart, and the two adaptive initial-sigma sensitivity runs.
- **Reason:** A fixed horizon makes component effects comparable. In
  particular, adaptive-without-restart is a deliberately non-paper control;
  its non-convergence should remain visible without invalidating the sourced
  coupled policy.

## D-0041 - Stop Stage 5 before GPU execution

- **Status:** accepted
- **Stage:** 5
- **Decision:** Keep all Stage 5 implementation and evidence in local CPU FP64.
  Do not install a numerical stack or execute solver code on the DGX Spark
  until Stage 6 receives explicit approval.
- **Reason:** The preprocessing and control logic must pass independent CPU
  checks before device residence, sparse GPU kernels, and timing boundaries
  are introduced.

## D-0042 - Keep CuPy optional outside GPU entry points

- **Status:** accepted
- **Stage:** 6
- **Decision:** Import CuPy lazily through the GPU backend. The package root,
  CPU implementation, and CPU test suite must remain importable when CuPy is
  absent. A DGX GPU entry point must fail with an actionable backend error if
  the pinned CuPy runtime is unavailable.
- **Reason:** CuPy is required to execute Stage 6 on the Spark but is not a
  mathematical dependency of the validated CPU oracle. Eager import would
  unnecessarily break local development and blur simulated tests with DGX
  evidence.

## D-0043 - Expose two incompatible GPU equality modes

- **Status:** accepted
- **Stage:** 6
- **Decision:** Use `scaled_direct` for the fully preconditioned Stage 5 LP and
  solve its equality Gram system by device Cholesky. Use
  `unscaled_structural` only for the raw Equation (55) LP with the corrected
  Stage 4 descriptor prepared from that exact LP instance. Reject every mixed
  scaled/structural pairing.
- **Reason:** General diagonal scaling changes the Gram structure. A separate
  scaled structural formula has not been derived, so silently applying the raw
  formula would sacrifice correctness for an unsupported optimization.

## D-0044 - Disclose ALG2 selection and permit a named correctness fallback

- **Status:** accepted
- **Stage:** 6
- **Decision:** Request cuSPARSE `CUSPARSE_SPMV_CSR_ALG2` for resident CSR
  products. If the supported CuPy interface cannot select it audibly, use the
  declared CuPy cuSPARSE generic default and record the requested path,
  selected path, fallback flag, reason, and library versions.
- **Reason:** The paper identifies ALG2, but a library wrapper can hide the
  low-level algorithm selector. The fallback preserves a testable GPU
  mathematical port while explicitly reducing kernel-level fidelity and
  prohibiting an ALG2 performance claim.

## D-0045 - Evaluate acceptance residuals in original coordinates on device

- **Status:** accepted
- **Stage:** 6
- **Decision:** Keep residual vectors and reductions on the GPU and recover the
  preconditioned state into original coordinates on device before applying
  Equation (54) and the raw KKT test. Transfer only scheduled scalar norms and
  flags during the loop; transfer full state only at named validation or
  finalization boundaries.
- **Reason:** Host residual evaluation would hide repeated transfer costs and
  defeat device residency. Scaled-space residuals alone cannot certify the
  original MW/MWh problem.

## D-0046 - Separate correctness and resident timing residual cadence

- **Status:** accepted
- **Stage:** 6
- **Decision:** Check Equation (54) every iteration for the correctness run.
  Use a separate fixed 1,000-iteration resident timing diagnostic with checks
  every 100 iterations. Synchronize timed GPU boundaries and report residual
  time separately from the iteration loop.
- **Reason:** Every-iteration checking preserves the validated first accepted
  iterate, while the separate resident diagnostic reveals loop behavior
  without presenting sparse checking as the correctness algorithm.

## D-0047 - Lock FP64 thresholds before observing GPU results

- **Status:** accepted
- **Stage:** 6
- **Decision:** Use the thresholds in
  `configs/sgs_hpr/stage_6_gpu_dgx.json` without post-result relaxation. FP64
  is required and gating. FP32 may run only after FP64 passes, is non-gating,
  and cannot replace FP64 evidence. Mixed precision remains disabled.
- **Reason:** Predeclared thresholds prevent accelerator discrepancies from
  being normalized after the fact. Reduced precision is an informative study,
  not a substitute for the CPU FP64 correctness contract.

## D-0048 - Record timing components without claiming Stage 7 speedup

- **Status:** accepted
- **Stage:** 6
- **Decision:** Record CUDA initialization, CPU construction and preprocessing,
  compilation and warm-up, allocation, host-to-device transfer, GPU solver
  initialization, iteration loop, residual checks, device-to-host transfer,
  and complete end-to-end wall time as named boundaries. Do not form a speedup
  from unlike boundaries, and keep benchmark campaigns locked to Stage 7.
- **Reason:** The manuscript does not define enough timing detail for an exact
  comparison. A decomposed Stage 6 measurement supports later controlled
  benchmarks without turning a correctness port into a performance claim.
