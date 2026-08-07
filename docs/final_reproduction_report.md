# Evidence-Bounded Structural Reproduction of a GPU sGS-HPR Solver

## Large-Scale Multi-Period DC Optimal Power Flow on NVIDIA DGX Spark

Thomas DiGregorio

Independent Researcher

Revised August 7, 2026

## Abstract

We independently reconstruct a published GPU symmetric Gauss--Seidel
Halpern-accelerated proximal-reflection method for multi-period DC optimal
power flow without the authors' code or instances. The canonical LP,
iteration, stopping maps, structural equality solve, CPU oracle, and resident
DGX Spark GPU implementation are validated. A sign inconsistency in the
printed rank-one inverse is corrected: maximum relative error falls to
3.03e-16 from median error 0.206. Public MATPOWER models reproduce all 18
reported dimensions but none of the nonzero counts. The GPU validates 11
allocated cases and the CPU 10; larger cases meet time, memory, or
sparse-index limits. These results establish structural---not exact or
performance---reproduction; no A100-equivalent speedup is claimed.

**Keywords:** DC optimal power flow, Halpern iteration, proximal reflection,
symmetric Gauss--Seidel, GPU sparse linear algebra, reproducibility, DGX
Spark.

## 1. Introduction

The source paper proposes a GPU sGS-HPR method for large-scale multi-period
DC optimal power flow and reports results on an A100-SXM4-80GB. Reproduction
requires separating the mathematical map, numerical instance,
implementation, validation criteria, timer boundary, and hardware.

The mathematical path and an FP64 GPU implementation are independently
validated, including a correction to the structural equality solve. Public
models reproduce the published dimensions but not the sparse supports, and
the local hardware and timers differ from the source study. Here,
**structural reproduction** means that the algorithm, implementation
structure, public-network scaling, and validation behavior are reproduced
without claiming author-instance identity or controlled paper timing.

The main contributions are:

- a self-contained DCOPF and sGS-HPR implementation contract with independent
  residual, physical, direct-solve, and CPU/GPU trajectory checks;
- a formal correction and stable implementation of the structural equality
  inverse;
- an 18-case structural ledger, an 11-case benchmark record, and repeated
  timing distributions; and
- live-memory and sparse-index boundary evidence with a public, versioned
  code/data release.

The displayed source model has no outage or contingency index. The study is
therefore limited to base-case multi-period DCOPF; N-1 contingency analysis is
outside its scope.

## 2. Methods

### 2.1 Source paper summary

The source is *An Efficient GPU-based Halpern Accelerating Algorithm for
Large-scale DC Optimal Power Flow* by Wang and coauthors, DOI
10.1109/TPWRS.2025.3635652.

The paper applies HPR to the dual of a box-constrained LP, splits equality and
inequality multipliers with an sGS sweep, and uses CSR sparse products on an
A100. Its implementation description specifies ten Ruiz passes, a
Pock--Chambolle step, initial sigma one, a 100-iteration policy cadence, and a
cuSPARSE CSR ALG2 path. Exact author construction and policy source were not
released.

### 2.2 Public data and reconstruction boundary

Public base networks are MATPOWER 8.1 case1354pegase, case2868rte, and
case9241pegase. The missing
author inputs are renewable and storage locations, time series, reserve
profiles, modified ramps, storage data, matrix construction, exact control
code, and timer boundary. Appendix A states every deterministic replacement.

### 2.3 Mathematical formulation

The canonical primal is

$$
\begin{aligned}
\min_{x\in\mathbb{R}^n}\quad &c^\mathsf{T}x,\\
\text{subject to}\quad &A_1x=b_1,\\
&A_2x\ge b_2,\\
&\ell\le x\le u.
\end{aligned}
$$

With $A=[A_1;A_2]$, $b=[b_1;b_2]$, box $C=[\ell,u]$, and
$D=\mathbb{R}^{m_1}\times\mathbb{R}^{m_2}_+$, the dual is

$$
\min_{y,z}-b^\mathsf{T}y+\delta_D(y)+\delta_C^*(-z)
\quad\text{subject to}\quad A^\mathsf{T}y+z=c.
$$

The decision order is

$$
x=(p_G,p_{RG},p_{ESS}^{dc},p_{ESS}^{ch},r^u,r^d).
$$

For $T$ periods, $N_L$ lines, $N_G$ generators, $N_{RG}$ renewables, and
$N_{ESS}$ storage devices,

$$
n=T(3N_G+N_{RG}+2N_{ESS}),
$$

$$
m_1=T+N_{ESS},\qquad
m_2=2TN_L+(4T-2)N_G+2TN_{ESS}+2T.
$$

The full KKT map is

$$
\mathcal{R}(y,z,x)=
\begin{pmatrix}
y-\Pi_D(y-Ax+b)\\
x-\Pi_C(x-z)\\
c-A^\mathsf{T}y-z
\end{pmatrix}.
$$

The paper's normalized stopping blocks are retained separately because its
first stopping block is not the full first block of this KKT map.

### 2.4 Implemented algorithm

The FP64 solver follows the paper order:

1. Compute the box-projected primal point.
2. Solve the first equality multiplier system using the old inequality
   multiplier.
3. Apply the spectral proximal step to the inequality multiplier.
4. Solve the second equality system using the new inequality multiplier.
5. Reflect the proximal point.
6. Apply the fixed-anchor Halpern update.

The principal identities are

$$
\bar x^{k+1}=\Pi_C[x^k+\sigma(A^\mathsf{T}y^k-c)]
$$

and

$$
w^{k+1}=\frac{1}{k+2}w^0+\frac{k+1}{k+2}(2\bar w^{k+1}-w^k).
$$

Production preprocessing uses ten Ruiz passes, one simultaneous
Pock--Chambolle diagonal step with alpha one, and complete-vector b/c
normalization. The restart and adaptive-sigma rules are transferred from the
pinned HPR-LP implementation and are disclosed in Appendix C. Solver
implementations and measurement boundaries are documented in Appendix B.

### 2.5 Experimental environment

| Item | Reproduction | Paper |
|---|---|---|
| GPU | NVIDIA GB10, integrated | NVIDIA A100-SXM4-80GB |
| Memory | 130,663,165,952 bytes, 121.690 GiB unified | 80 GB GPU |
| Host | Linux/aarch64, Ubuntu 24.04.4 | not specified |
| Python stack | Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3, CuPy 14.1.1 | not specified |
| Sparse path | FP64 CSR, signed-int32 indices, observed CSR ALG2 | CSR ALG2; precision/index width unstated |
| Gurobi | unavailable and optional locally | 12.0 on separate Intel workstation |

### 2.6 Acceptance and timing protocol

HiGHS supplies an objective reference, the CPU implementation is an
algorithmic oracle, and the GPU implementation is the target path. The frozen
decision rules are:

| Element | Frozen design |
|---|---|
| Numerical acceptance | Normalized stopping blocks no greater than 5e-5; raw KKT no greater than 0.01; scaled objective gap to HiGHS no greater than 2e-4. |
| Physical acceptance | Maximum original-unit violation no greater than 0.01 MW/MWh. |
| Row acceptance | HiGHS, CPU FP64, and GPU FP64 must each pass; a preallocation resource stop cannot be reclassified as a solve. |
| Correctness and warm-up | One untimed correctness solve followed by one warm-up. |
| Measured repeats | Five measured solves; relative range above 0.2 triggers four additional repeats without deleting earlier samples. |
| Reported statistics | Median, minimum, maximum, standard deviation, and IQR. |
| Time limit | 3,600 seconds per solve; a limit event is censored and has no median. |
| Clock interpretation | Solver-specific boundaries are reported separately and are not divided into cross-solver speedups. |

## 3. Verification

### 3.1 Structural equality correction

Each sGS-HPR iteration solves a system involving $A_1A_1^\mathsf{T}$. Its first
$T$ rows are hourly power balances; its remaining rows enforce terminal
storage energy. The repeated temporal structure makes the Gram matrix diagonal
except for one direction shared by every period. Lemma 1 uses that fact to
avoid forming or factorizing a dense matrix.

**Lemma 1 (stable inverse of the reduced equality block).** Let $D_2$ be a
positive diagonal matrix, $a>0$, $d$ a vector,
$\alpha=d^\mathsf{T}D_2^{-1}d$, and

$$
\begin{bmatrix}
aI_T & \mathbf{1}d^\mathsf{T}\\
d\mathbf{1}^\mathsf{T} & D_2
\end{bmatrix}
\begin{bmatrix}y_{11}\\y_{12}\end{bmatrix}
=
\begin{bmatrix}r_{11}\\r_{12}\end{bmatrix}.
$$

Define

$$
\widehat r=r_{11}-\mathbf{1}d^\mathsf{T}D_2^{-1}r_{12},
\qquad
\gamma=a-T\alpha,
\qquad
\mu=T^{-1}\mathbf{1}^\mathsf{T}\widehat r.
$$

If $\gamma>0$, the solution is

$$
y_{11}=\frac{\widehat r-\mu\mathbf{1}}{a}
+\frac{\mu}{\gamma}\mathbf{1},
\qquad
y_{12}=D_2^{-1}\left(r_{12}-d\mathbf{1}^\mathsf{T}y_{11}\right).
$$

In plain language, $\mathbf{1}$ is the common-across-time direction.
$\widehat r$ splits into an average and hour-to-hour deviations. The storage
coupling does not act on the deviations, so they are divided by $a$. It acts
only on the average, which is divided by $\gamma$. The storage multipliers then
follow from a diagonal back-substitution. The dense-looking solve therefore
reduces to vector sums, diagonal divisions, and scalar operations.

Eliminating $y_{12}$ gives the minus rank-one system
$(aI_T-\alpha\mathbf{1}\mathbf{1}^\mathsf{T})y_{11}=\widehat r$. The source
paper prints the inverse pattern for a plus rank-one update, which gives the
wrong sign for the shared average component.

The sign printed in the source paper produced a median relative error of
0.206 on the analytic test family. The corrected expression reduced the
maximum relative error to $3.03\times10^{-16}$. Across broader conditioning
fixtures, direct-solve relative error remained at most $1.53\times10^{-14}$
and equality residual at most $2.93\times10^{-14}$, including a condition
number of 25,616.42.

### 3.2 Numerical and trajectory validation

Validation combined four independent views: normalized paper stopping blocks,
the complete KKT map, original-unit power-system limits, and objective
agreement with HiGHS. GPU trajectory parity was checked against the CPU oracle
at iterations 1, 10, and 100 before timing measurements. Every accepted
row passed all frozen thresholds; no threshold was relaxed after observing a
result.

## 4. Results

### 4.1 Benchmark results

Eleven public-network reconstructions were fully allocated. Ten passed every
solver and validation requirement. The remaining allocated row,
case9241pegase:T6, passed HiGHS and GPU validation but failed the strict row
because its single CPU correctness solve reached the 3,600-second limit.

| Network / horizon | Rows $m$ | Variables $n$ | Reconstructed nnz | Row result | GPU iterations |
|---|---:|---:|---:|---|---:|
| case1354pegase / T4 | 20,192 | 4,208 | 4,799,808 | PASS | 302 |
| case1354pegase / T16 | 82,124 | 16,832 | 19,228,464 | PASS | 304 |
| case1354pegase / T48 | 247,276 | 50,496 | 57,896,368 | PASS | 359 |
| case1354pegase / T96 | 495,004 | 100,992 | 116,420,464 | PASS | 404 |
| case2868rte / T4 | 40,163 | 9,488 | 19,073,056 | PASS | 416 |
| case2868rte / T16 | 163,823 | 37,952 | 76,354,336 | PASS | 454 |
| case2868rte / T48 | 493,583 | 113,856 | 229,507,104 | PASS | 499 |
| case2868rte / T64 | 658,463 | 151,808 | 306,303,136 | PASS | 503 |
| case2868rte / T96 | 988,223 | 227,712 | 460,334,496 | PASS | 503 |
| case9241pegase / T4 | 152,774 | 24,700 | 342,863,272 | PASS | 1,272 |
| case9241pegase / T6 | 230,376 | 37,050 | 514,308,838 | FAIL: CPU time limit | 1,541 |

The reconstructed support is systematically smaller than the dimensions
reported in the paper: about 33.0--33.2% lower for case1354pegase,
36.6--36.7% lower for case2868rte through T16, 22.4--22.5% lower for its
larger horizons, and 8.14% lower for case9241pegase. These are structural
reconstructions, not replicas of the unavailable author matrices.

### 4.2 Timing decomposition and uncertainty

The timing figure overlays all three methods on one shared logarithmic axis,
using distinct marker shapes and small horizontal offsets within each
reconstruction. The shared scale makes magnitude and scaling patterns visible,
but each clock still covers a different solver boundary. Markers show medians;
whiskers show the observed minimum and maximum. The CSV also retains IQR,
standard deviation, and repeat count. Representative median [minimum, maximum]
values in seconds are:

| Network / horizon | HiGHS | CPU FP64 | GPU FP64 |
|---|---:|---:|---:|
| case1354pegase / T4 | 1.463 [1.453, 1.480] | 13.190 [11.836, 16.879] | 1.013 [0.995, 1.092] |
| case2868rte / T48 | 76.239 [70.892, 77.949] | 804.863 [801.569, 808.401] | 49.968 [49.927, 49.998] |
| case9241pegase / T4 | 142.479 [141.937, 142.810] | 3,087.218 [3,070.085, 3,096.832] | 193.632 [193.603, 193.741] |
| case9241pegase / T6 | 963.957 [960.497, 966.056] | 3,600.093 censored | 357.544 [357.462, 357.751] |

No speedup is claimed. The values support local operational comparison only;
they do not reproduce the paper's A100 boundary or establish a controlled
CPU-to-GPU acceleration ratio.

### 4.3 Memory and resource results

The DGX Spark reports 130,663,165,952 bytes, or 121.690 GiB, of unified
memory. The nominal 80% reference is therefore 97.352 GiB, but allocation was
governed by the smaller live host and device budgets sampled immediately
before each row.

For T16, the projected requirement was 94.435 GiB: below the nominal 80%
reference but above the contemporaneous 65.784 GiB host and 65.496 GiB device
budgets. T16 is therefore a live-availability safety-gate decision, not proof
that a 128 GB machine cannot hold the problem and not an out-of-memory crash.

For T24, the conservative planning count was 2,531,600,260 nonzeros, above the
signed-int32 CSR limit of 2,147,483,647. The exact reconstructed count later
derived without allocation was 2,057,650,132, below that limit. T24 is thus a
deliberate policy-envelope block, not a demonstrated materialized-index
overflow. For T32, both the planning count of 3,375,704,460 and the exact
count of 2,743,770,956 exceed signed int32, so the index boundary is also
structurally present in the reconstruction. None of these rows was allocated.

## 5. Discussion

### 5.1 Reproduction boundary

The source paper used private modified cases, unpublished sparse supports and
profiles, an NVIDIA A100-SXM4-80GB system, and a timing boundary that cannot be
reconstructed from the article alone. This study instead uses deterministic
public MATPOWER 8.1 cases, explicit synthetic intertemporal fixtures, a DGX
Spark GB10, and separately disclosed solver clocks. The local CPU oracle is a
validation reference, not the paper's Gurobi workstation baseline.

### 5.2 Limitations and next step

- Public cases do not reveal the authors' placements, profiles, ramp and
  reserve inputs, storage parameters, or matrix sparsity choices.
- The local timing tracks have different inclusion boundaries and cannot be
  divided into defensible speedups.
- T6 CPU supplies one censored correctness attempt rather than a timing median.
- Resource stops are preallocation safety decisions; T16 and T24 must not be
  paraphrased as physical hardware failures.
- GPU parity and residual acceptance establish implementation consistency,
  not uniqueness of the optimizer or identity with the authors' trajectories.

The next scientific step is to obtain an authorized author-instance package
or a fully specified equivalent fixture, freeze its exact sparse structure and
timer boundary, and run a controlled A100-versus-GB10 study. That would test
whether the remaining gap is attributable to inputs, implementation, hardware,
or timing definitions.

## 6. Conclusion

This study reconstructs the path from canonical DCOPF to a resident FP64 GPU
sGS-HPR solver and corrects a quantitatively material rank-one sign error.
CPU/GPU trajectories and original-unit constraints agree across the accepted
public cases, but every reconstructed sparse support differs from the source
paper and one required CPU solve is censored at its time limit. The evidence
therefore establishes structural reproduction, not author-instance identity
or controlled performance reproduction.

## Appendix A. Deterministic public-instance recipe

The public fixtures are generated from pinned MATPOWER 8.1 snapshots using
seed 20260803. Loads and available generation are tiled across the requested
horizon; the deterministic temporal perturbation is zero-mean over the
horizon; storage fixtures use explicitly versioned placement, efficiency,
power, and energy parameters; offline equipment and zero thermal ratings are
handled before the canonical row assembly. Variable and row order are frozen
in the versioned configuration.

## Appendix B. Solver implementations and measurement boundaries

**HiGHS reference.** SciPy/HiGHS supplies the independent objective through
`linprog(method="highs-ds")` with presolve, 1e-9 primal and dual feasibility
tolerances, and a 3,600-second limit. Its clock includes interface/model setup
and the solve.

**CPU oracle.** The CPU sGS-HPR implementation uses SciPy sparse operators in
FP64, a zero initial state, both equality sweeps, original-coordinate recovery,
and independent residual and physical checks. Its clock begins after model,
preconditioner, and workspace preparation and includes the iterative solve and
scheduled residual work.

**GPU implementation.** The target path uses CuPy 14.1.1 arrays, FP64 CSR
matrices, resident state, and reused workspaces. The selected
`CUSPARSE_SPMV_CSR_ALG2` path is verified by normal and transpose probes, and a
transfer ledger rejects unexpected full-state copies inside the loop. Its clock
uses the same post-preparation iterative/residual boundary as the CPU path.

Initialization, construction, preprocessing, compilation, allocation,
transfers, recovery, and complete runner wall time are separately recorded.

## Appendix C. Preprocessing and controls

Ten Ruiz passes alternate row and column infinity-norm equilibration. One
simultaneous Pock--Chambolle diagonal step with alpha one follows, then the
complete vectors b and c are normalized. The adaptive penalty uses the
primal-to-dual movement ratio with bounded multiplicative updates; restart
tests the normalized fixed-point progress and resets the Halpern anchor only
when the sourced HPR-LP condition is met. Exact formulas, guards, and defaults
are included in the tagged release.

## Appendix D. Code and data availability

Source, public case snapshots, configurations, immutable raw evidence,
generated tables and figures, and the paper sources are available at
https://github.com/thomas-digregorio/sGS-HPR-DC-SCOPF. The revised report is
identified by release tag reproduction-paper-v5.

Regeneration is deterministic and does not rerun DGX allocations. The tagged
release includes the command inventory, machine-readable evidence index,
structural and benchmark ledgers, and timing summaries.

## Appendix E. Missing source information

Exact reproduction requires the authors' modified cases, placements, temporal
profiles, reserve and ramp inputs, storage parameters, sparse matrix assembly,
PTDF policy, numerical precision, control-code revision, and timer definition.
Unknown values were neither guessed nor fitted to published timings.

## References

1. Q. Wang et al., “An efficient GPU-based Halpern accelerating algorithm for large-scale DC optimal power flow,” IEEE Transactions on Power Systems, vol. 41, no. 3, 2026. DOI 10.1109/TPWRS.2025.3635652.
2. K. Chen et al., “HPR-LP: An implementation of an HPR method for solving linear programming,” Mathematical Programming Computation, vol. 18, pp. 183–210, 2026. DOI 10.1007/s12532-025-00292-0.
3. R. D. Zimmerman and C. E. Murillo-Sánchez, MATPOWER 8.1, 2025. DOI 10.5281/zenodo.15871662.
4. C. Josz et al., “AC power flow data in MATPOWER and QCQP format,” arXiv:1603.01533, 2016.
5. S. Fliscounakis et al., “Contingency ranking with respect to overloads in very large power systems,” IEEE Transactions on Power Systems, vol. 28, no. 4, 2013. DOI 10.1109/TPWRS.2013.2251015.
6. D. Ruiz, “A scaling algorithm to equilibrate both rows and columns norms in matrices,” RAL-TR-2001-034, 2001.
7. T. Pock and A. Chambolle, “Diagonal preconditioning for first order primal-dual algorithms in convex optimization,” ICCV, 2011. DOI 10.1109/ICCV.2011.6126441.
8. P. Virtanen et al., “SciPy 1.0,” Nature Methods, vol. 17, pp. 261–272, 2020. DOI 10.1038/s41592-019-0686-2.
9. Q. Huangfu and J. A. J. Hall, “Parallelizing the dual revised simplex method,” Mathematical Programming Computation, vol. 10, pp. 119–142, 2018.
10. R. Okuta et al., “CuPy: A NumPy-compatible library for NVIDIA GPU calculations,” NeurIPS Workshop on Machine Learning Systems, 2017.
11. NVIDIA, cuSPARSE Library Documentation, 2026.
12. NVIDIA, DGX Spark User Guide: Hardware Overview, 2026.
