# Reproduction limits and evidence gaps

## Status

This register separates what the manuscript actually specifies from what would
have to be recovered, derived, or transparently reconstructed. It was prepared
for Stage 0 on 2026-07-29 and updated after the Stage 7 structural benchmark
campaign on 2026-08-03.

**Current conclusion:** an exact numerical recreation of the paper's benchmark
tables cannot be claimed from the available information. Stage 7 therefore
reports a clearly labeled structural reproduction using pinned public networks
and a frozen deterministic protocol for the missing author additions.

Stage 7 establishes six-case HiGHS, CPU FP64, and GPU FP64 correctness plus
repeated timing, memory, and transfer evidence on this project's DGX Spark. It
matches all 18 published row and variable dimensions, but every reproduced
sparse nonzero count differs. That result strengthens this implementation
without recovering the authors' missing inputs, source, precision choice, or
timing boundary.

The supplied source is a 17-page accepted/prepublished manuscript. A current
institutional record identifies the final article as IEEE Transactions on Power
Systems, volume 41, issue 3, pages 2187-2204, May 2026, DOI
10.1109/TPWRS.2025.3635652, and provides an accepted-manuscript copy:
[PolyU institutional record](https://ira.lib.polyu.edu.hk/handle/10397/117738?mode=full).
The local PDF remains the primary mathematical specification for this project.

## Classification vocabulary

- **Explicitly specified:** printed directly in the paper.
- **Derivable:** follows uniquely from printed equations or tables.
- **Referenced elsewhere:** delegated to a cited work or public project.
- **Ambiguous:** more than one defensible interpretation remains.
- **Missing:** needed for the intended claim but not supplied or linked.

These categories describe evidence availability, not the importance of an item.

## 1. What is explicitly specified

| Detail | Paper evidence | Reproduction consequence |
|---|---|---|
| Optimization model | Eqs. (1)-(16) | The stated multi-period LP can be implemented. |
| Algorithm order | Algorithms 1-2 and Eqs. (17)-(50) | A correctness-oriented CPU method can be built without inventing update order. |
| Variable counts | Table I plus \(n=T(3N_G+N_{RG}+2N_{ESS})\) | Published variable dimensions are reproducible. |
| Constraint families | Eqs. (1)-(10) | Published constraint dimensions are derivable. |
| Test-system aggregate counts | Table I | Bus, branch, conventional-generator, RG, and ESS counts are known. |
| Matrix dimensions and nnz | Table II | Reconstructed matrices can be compared structurally. |
| Reported timings | Table II | Values can be preserved as paper results, not treated as independently verified measurements. |
| GPU hardware | NVIDIA A100-SXM4-80GB | The paper platform is known and differs from the DGX Spark target. |
| CUDA and Gurobi versions | CUDA 12.3; Gurobi 12.0.0 | Software headlines are known, but not full dependency or driver builds. |
| CPU comparison machine | Intel i9-14900HX, 24 cores, 96 GB; Gurobi cap 80 GB | Hardware is known, but GPU and CPU runs are not from one controlled platform. |
| Sparse/kernel choices | CSR; cuSPARSE `CUSPARSE_SPMV_CSR_ALG2`; 256 threads per block | The reported low-level direction can be benchmarked later. |
| Preconditioning headline | 10 Ruiz iterations, Pock-Chambolle diagonal preconditioning, \(\alpha=1\) | The sequence is known; implementation details still require cited sources. |
| \(b,c\) normalization | Divide by \(\|b\|+1\) and \(\|c\|+1\) | The manuscript supports norm-based, not elementwise, normalization. |
| Initial penalty | Cold start, \(\sigma=1\) | A fixed-\(\sigma\) baseline is supported. |
| Restart frequency | Check every 100 iterations | Frequency is known; criterion and state update are not. |
| Termination | Eq. (54), \(\epsilon=5\times10^{-5}\) | The three stopping blocks can be reproduced. |
| Time limit | 3,600 seconds | Failure/timeout classification can match the paper's cap. |

## 2. What is derivable

| Detail | Derivation | Validation |
|---|---|---|
| Variable order | Eq. (55) and objective/constraint blocks imply \((p_G,p_{RG},p^{dc},p^{ch},r^u,r^d)\). | Recorded in `paper_specification.md`. |
| Equality rows | Eq. (1) contributes \(T\); Eq. (9) contributes \(N_{ESS}\). | \(m_1=T+N_{ESS}\). |
| Inequality rows | Split Eqs. (2), (4)-(6), and (8) into \(\ge\) rows. | \(m_2=2TN_L+(4T-2)N_G+2TN_{ESS}+2T\). |
| Total rows | Add \(m_1+m_2\). | \(m=2TN_L+(4T-2)N_G+(2T+1)N_{ESS}+3T\). |
| Objective coefficient blocks | Expand Eqs. (12)-(14) and remove constants. | Absolute objective comparisons must restore dropped constants. |
| Storage-row time structure | Eq. (8) is cumulative; Eq. (9) couples the complete horizon. | Determines lower-triangular inequality blocks and low-rank equality coupling. |
| Box projection | Eqs. (33)-(34). | \(\bar x=\Pi_C[x+\sigma(A^Ty-c)]\). |
| \(y_2\) diagonalization | Eqs. (47)-(50). | \(\lambda=\|A_2\|_2^2\) makes \(A_2A_2^T+\mathcal S_2=\lambda I\). |

The derived dimension formulas exactly reproduce case1354 T4, case2868 T16,
and case9241 T6 from Table II. That validates row counting, not the unpublished
numerical inputs or matrix entries.

## 3. Details referenced elsewhere

### Public base-network cases

The paper names three cases but gives no repository, release, commit, or file
hash. Current MATPOWER source files exist for:

- [case1354pegase](https://github.com/MATPOWER/matpower/blob/master/data/case1354pegase.m)
- [case2868rte](https://github.com/MATPOWER/matpower/blob/master/data/case2868rte.m)
- [case9241pegase](https://github.com/MATPOWER/matpower/blob/master/data/case9241pegase.m)

These are viable public base cases for a later provenance-controlled
reconstruction. They are not proof of which MATPOWER release or modifications
the authors used. Stage 7 must pin a commit, record hashes, and reconcile every
count before using them as benchmark inputs.

### HPR-LP restart and penalty strategy

The paper says its adaptive penalty update is similar to HPR-LP [43] and refers
to HPR-LP and PDLP for restart details. HPR-LP is now publicly documented and
implemented:

- [official HPR-LP project page](https://www.polyu.edu.hk/ama/ior/HPR-LP.html)
- [PolyU-IOR/HPR-LP source repository](https://github.com/PolyU-IOR/HPR-LP)

This makes the cited strategy inspectable in Stage 5. It does not automatically
prove that the DCOPF implementation used the same revision, defaults, or
state-update details. The exact HPR-LP commit used by Wang et al. is not stated.

PDLP is also available in the
[Google OR-Tools repository](https://github.com/google/or-tools), but the
paper does not identify an OR-Tools revision or state which PDLP restart
variant was transferred to sGS-HPR.

### Pock-Chambolle preconditioning

The paper cites Pock and Chambolle [59] for diagonal preconditioning. The exact
formula must be taken from that source and reconciled with the implemented
matrix orientation in Stage 5. The name of the method alone is insufficient.

## 4. Ambiguous or internally inconsistent details

| Detail | Ambiguity | Required resolution |
|---|---|---|
| Equality-row count | Eq. (55), Eq. (9), and Table II imply \(T+N_{ESS}\); Appendix A claims \(T(1+N_{ESS})\). | Follow the table-consistent Eq. (55) structure and retain the discrepancy as an erratum. |
| Proposition 5 signs | Eq. (43) is a minus rank-one update, while Eqs. (39)/(44)/(45) print a plus-update inverse pattern. | Compare every structural solve with a direct solve before acceptance. |
| Conventional-generator box | Eq. (4) plus nonnegative reserve implies generator bounds, while compact prose says all variable bounds belong to \(C\). | Test explicit and implied-bound splits; document the selected projection. |
| "Security-constrained" label | The model has base-case line limits but no outage or contingency index. | Reproduce it as base-case DCOPF; keep N-1 constraints out of Stages 0-9. |
| Simultaneous ESS charge/discharge | No binary or complementarity constraint prohibits it. | Preserve the LP as written; do not add a mixed-integer restriction. |
| Objective comparison | Eq. (15) omits constants from Eqs. (12)-(13). | Restore constants when comparing absolute objective values. |
| Five "independent" runs | Deterministic SpMV is stated, but no random source is described. | Treat the reported average objective error as incompletely specified. |
| Gurobi baseline | Fastest of primal simplex, dual simplex, and barrier is reported on separate hardware. | Preserve as a paper result; use a separate controlled baseline for new speedup claims. |

## 5. Missing information required for exact numerical reproduction

### Network and device construction

| Required detail | Status | Why it matters |
|---|---|---|
| Exact case source, release, commit, and hashes | Missing | MATPOWER cases can change across releases and may have been modified. |
| Renewable-generator bus locations | Missing | Changes PTDF columns, congestion, objective, and nnz patterns. |
| ESS bus locations | Missing | Changes both line-flow and storage-coupling matrix entries. |
| Conventional generator filtering or aggregation | Missing | Affects \(N_G\), costs, bounds, and ramp rows. |
| Inactive branch treatment | Missing | Alters \(N_L\), connectivity, PTDF, and nnz. |
| Inactive bus/generator treatment | Missing | Alters balance, costs, and case dimensions. |
| Transformer taps and phase-shifter treatment | Missing | Alters DC susceptance and shift factors. |
| Islands and reference/slack bus | Missing | Determines PTDF construction and numerical rank. |
| PTDF sign convention and sparsification threshold | Missing | Changes matrix signs and reported nnz. |
| Base-MVA and unit-conversion procedure | Not fully specified | Needed to reproduce physical bounds and objective scale. |

### Time-series and physical parameters

| Required detail | Status | Why it matters |
|---|---|---|
| Load profiles for every horizon | Missing | Defines \(b_1\), line-flow offsets, and congestion. |
| Renewable availability profiles | Missing | Defines time-varying renewable bounds and curtailment. |
| Reserve requirement series \(SRU(t),SRD(t)\) | Missing | Defines system reserve rows. |
| Generator ramp rates used in experiments | Missing | Defines reserve bounds and inter-period constraints. |
| ESS initial energy, lower/upper capacity | Missing | Defines cumulative-energy right-hand sides. |
| ESS charge/discharge power limits | Missing | Defines box \(C\). |
| ESS efficiencies | Missing | Defines equality, energy rows, and loss costs. |
| Interval duration \(\Delta t\) | Symbolic only | Scales ramping and storage equations. |
| Renewable and ESS penalty coefficients | Missing | Defines objective tradeoffs. |
| Generation-cost modifications | Missing | Public case costs may not match experimental costs. |
| Random seeds or deterministic placement rules | Missing | Prevents regeneration of added RG/ESS fleets. |

### Algorithm and numerical implementation

| Required detail | Status | Why it matters |
|---|---|---|
| Adaptive-\(\sigma\) formula and thresholds | Missing from this paper; referenced elsewhere | Materially changes iteration count and Table IV timing. |
| Restart criterion, merit function, and restart state | Missing from this paper; referenced elsewhere | Materially changes convergence path. |
| \(\lambda_{\max}(A_2A_2^T)\) estimation method | Missing | An underestimate may invalidate \(\mathcal S_2\succeq0\); estimator cost affects timing. |
| Safety factor for spectral estimate | Missing | Affects both validity and step size. |
| Floating-point precision | Missing | FP32, FP64, or mixed precision can alter residuals and throughput. |
| Reduction order and norm implementation | Missing | Affects reproducibility near stopping thresholds. |
| Sparse transpose storage and format details | Missing | Affects memory, kernel choice, and timing. |
| Full dependency and driver versions | Missing | CUDA 12.3 alone is not a reproducible software environment. |
| Exact initialization vectors | "Cold start" stated, values not printed | Needed for trajectory-level equality. |
| Complete solver settings | Missing | Competitor defaults and preprocessing can change results. |

### Timing boundary

The paper does not say whether reported solving time includes:

- network and matrix construction;
- PTDF computation;
- Ruiz scaling and Pock-Chambolle preprocessing;
- spectral estimation;
- CUDA context initialization;
- JIT or kernel compilation;
- device allocation;
- host-to-device transfer;
- residual checks and synchronization;
- device-to-host transfer;
- warm-up runs.

It also does not report repetition-level times, variance, or the exact
aggregation used for each Table II cell. Consequently, paper timing values can
be quoted, but a new DGX Spark run cannot be labeled a timing reproduction
unless a compatible boundary is established.

## 6. Source-code availability decision

The manuscript and its institutional article record do not link a
paper-specific sGS-HPR/DCOPF source repository. The institutional record and
the now-published IEEE citation were rechecked together with targeted exact-title,
algorithm-name, and public-code searches on 2026-08-03; no author data or
paper-specific implementation was located. This is an
evidence-of-search statement, not proof that private or newly released code
does not exist.

Related HPR-LP source code is public and useful for investigating the cited
restart and penalty rules, but it is a general Julia LP solver, not the
paper-specific DCOPF data-construction and CUDA implementation.

## 7. Defensible reproduction classifications

### Exact reproduction

Requires the authors' complete numerical inputs, construction choices,
algorithmic rules, precision, code revision, and a defensibly comparable
environment. **Not currently possible from available evidence.**

### Mathematical reproduction

Implements the printed LP, Algorithm 2, closed-form updates, and Eq. (54);
validates against independent LP solutions on available data. **Completed as
the primary target for Stages 1-6.**

### Structural reproduction

Uses pinned public base cases plus a published, deterministic protocol for
placing and parameterizing RGs/ESSs and time series; compares dimensions,
sparsity, correctness, and scaling without claiming identical instances.
**This is the accepted Stage 7 and Stage 8 classification.** Stage 8's
terminal CPU time limit does not upgrade or invalidate the classification; it
limits how far the validated structural campaign reached.

### Approximate benchmark reconstruction

Attempts to approach published dimensions or timing trends with disclosed
differences. It must never be presented as an exact timing or data
reproduction.

## 8. What Stages 6 through 8 resolve for this project

Stage 6 makes several project-side choices observable and reproducible:

- FP64 is the correctness baseline;
- matrices, transposes, scaling data, state, and workspaces remain resident on
  the DGX Spark GPU;
- the low-level sparse path actually selects
  `CUSPARSE_SPMV_CSR_ALG2` on the NVIDIA GB10;
- initialization, warm-up, allocation, transfers, solver initialization,
  iterations, residual checks, recovery, and end-to-end time are recorded as
  separate fields;
- the frozen T1 and T2 cases reach the same CPU and GPU stopping iterations;
- FP32 is a non-gating diagnostic.

These decisions define and validate this implementation. They do not show that
the authors used FP64, the same transfer policy, identical preprocessing code,
or the same timing boundary. They also do not make the DGX Spark GB10 a
performance substitute for the paper's A100.

No Stage 6 speedup is claimed. The two small fixtures are designed to expose
correctness errors, not to support a throughput conclusion.

Stage 7 additionally makes the benchmark boundary observable:

- the public MATPOWER 8.1 case files, commit, Git blobs, and canonical hashes
  are pinned;
- one deterministic protocol supplies the missing resource placement and
  time-series inputs without tuning;
- all 18 Table II rows have exact `m` and `n` reconciliation;
- every sparse nonzero count differs by 8.136% to 36.659%, so no paper time is
  directly comparable;
- exactly six small/medium cases pass HiGHS, CPU FP64, GPU FP64, raw-KKT,
  physical, objective, memory, and transfer gates;
- at the end of Stage 7, the other 12 rows were count-only and had received
  zero Stage 8 allocations; and
- first runs, warm-ups, measured repetitions, variability, and complete-case
  walls are reported separately.

These observations still do not identify the authors' hidden additions or
make the DGX Spark GB10 equivalent to the paper's A100.

Stage 8 subsequently allocated five strict-prefix large rows. T48, T64, T96,
and case9241pegase T4 passed all required tracks. Case9241pegase T6 passed
HiGHS and GPU FP64 but its required CPU correctness attempt exceeded the
frozen 3,600-second limit. The campaign stopped there without retry. T16, T24,
and T32 were not executed. This establishes a larger validated prefix for the
structural reconstruction, not a complete large-case pass.

## 9. Locked follow-up plan

1. Preserve the terminal Stage 8 evidence and T6 CPU time limit unchanged.
2. Keep Stage 9 locked because the conditional Stage 9 gate was not satisfied.
3. Do not retry T6 or revise its deadline without a new explicit decision and
   a separately identified campaign.
4. Continue to preserve the structural classification and sparse-count
   differences in any authorized follow-up.
5. Keep first-run, warm-up, solver-core, transfer, and end-to-end boundaries
   separate.
6. Do not compare DGX timing directly with Table II unless sparse workload and
   inclusion boundaries are shown to be compatible.
7. Continue to preserve missing author items as missing rather than tuning
   synthetic values to match the paper's times.

## 10. Stage 7 public-data result

Stage 7 pins the unmodified network files from MATPOWER 8.1 at resolved commit
`1a828c7af590714499284e36ee9c81273388c594`. This resolves the base-network
source and version, but none of the missing author additions listed above.

The public cases also differ from the counts implied by Table II if interpreted
with ordinary MATPOWER operational semantics: some branch `rateA` values are
zero, case2868 contains offline generator rows, and case2868 includes branch
angle-difference limits that are absent from the printed DCOPF. Stage 7 therefore
uses a separate, frozen structural-reconstruction protocol. It preserves the
original files, records each transformation, and prohibits tuning toward the
paper's timings. A nonzero sparse-count difference blocks a direct comparison
with the corresponding paper time even when the row and variable dimensions
match exactly.

The completed ledger confirms that boundary: all 18 dimension pairs match,
while all 18 nonzero counts differ. Stage 7 executed only case1354pegase at
T=4, 16, 48, and 96 and case2868rte at T=4 and 16. The remaining rows were
evaluated count-only, without full model allocation or a solver run.

All six required HiGHS, CPU FP64 sGS-HPR, and GPU FP64 sGS-HPR tracks passed.
The largest raw KKT norm was `0.0096347433`, the largest physical violation was
`0.0062210399 MW/MWh`, and the largest scaled objective gap to HiGHS was
`4.28499e-8`. The independent checker passed 19/19 checks. These results
validate this structural reconstruction; they do not close any missing-author
item listed above.

The accepted run remains tied to commit `ff6f762`. Post-run checker maintenance
accepts honestly scoped Linux `getrusage`-only cumulative peaks when `psutil`
RSS is unavailable, and source preflight now fails closed on deletion of a
tracked Python file. Strict provenance and scope tests accompany both changes.
They changed no accepted evidence or threshold and required no numerical
rerun. The accepted 19/19 check is evaluated against `ff6f762`; checking a
later head with changed execution-source files fails source identity by design,
not a numerical or timing gate.

## 11. Stage 8 terminal result

Stage 8 ran from clean detached commit
`f1fffc2adcba197040578695ba11dd27b0d1981f` and preserved the Stage 7
construction, precision, numerical thresholds, and timing protocol. The first
four rows passed all required tracks. The fifth row, case9241pegase T6, passed
its memory gate, HiGHS reference, and GPU FP64 track, but its CPU FP64
correctness attempt reached `TIME_LIMIT` after `3,600.092739 s`.

That outcome has several important limits:

- the completed T6 GPU candidate is valid, but T6 as a required three-track row
  is not a pass;
- the interrupted CPU attempt supplies neither an accepted objective nor a
  timing median;
- T16's 94.435 GiB planning projection was not tested against a live budget;
- T24 and T32 retain static signed-int32 ledger blocks but were not executed;
- no Stage 8 speedup is supported; and
- Stage 9 remains locked.

The independent checker passed 12/12 protocol and evidence checks while
retaining campaign status `STOPPED_ON_FAILURE`. This distinction is essential:
checker PASS means the failure was recorded honestly, not that Stage 8 met its
scientific acceptance gate.
