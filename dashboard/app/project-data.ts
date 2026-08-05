export type StageStatus = "complete" | "active" | "stopped" | "locked" | "optional";
export type TaskStatus = "complete" | "working" | "queued" | "failed" | "locked";

export type ResearchTask = {
  id: string;
  title: string;
  status: TaskStatus;
};

export type ResearchStage = {
  id: number;
  title: string;
  purpose: string;
  status: StageStatus;
  tasks: ResearchTask[];
};

export type StageEightRowStatus =
  | "queued"
  | "running"
  | "passed"
  | "locked"
  | "memory-blocked"
  | "index-blocked"
  | "failed";

export type StageEightCorrectness = "pending" | "passed" | "not-run" | "failed";

export type StageEightCampaignRow = {
  order: number;
  caseName: string;
  horizon: string;
  role: "frozen" | "conditional";
  estimatedDeviceGiB: string;
  estimatedHostGiB: string;
  estimatedCombinedGiB: string;
  reconstructedNnz: string;
  status: StageEightRowStatus;
  correctness: StageEightCorrectness;
  highsMedianSeconds: string | null;
  cpuMedianSeconds: string | null;
  gpuMedianSeconds: string | null;
  note: string;
};

export type EvidenceAvailability = "available" | "pending";

export type EvidenceArtifact = {
  label: string;
  name: string;
  availability: EvidenceAvailability;
};

const stageEightTasks: ResearchTask[] = [
  {
    id: "8.1",
    title: "Freeze ordered campaign and preallocation resource estimates",
    status: "complete",
  },
  {
    id: "8.2",
    title: "Run each safe scale until the first frozen failure boundary",
    status: "failed",
  },
  {
    id: "8.3",
    title: "Preserve numerical, timeout, memory, and index failures",
    status: "complete",
  },
  {
    id: "8.4",
    title: "Resolve sequences 6--8 under the separately frozen GPU-only scope",
    status: "complete",
  },
  { id: "8.5", title: "Interpret timings only across fair boundaries", status: "complete" },
];

/**
 * Stage 8 has one intentionally centralized terminal evidence model. The component
 * derives all counters and tables from this immutable campaign summary.
 */
export const stageEightDashboard = {
  stageStatus: "stopped" as StageStatus,
  phaseLabel: "STAGE_9_COMPLETE",
  headline: "Stage 9 concludes: structural reproduction.",
  summary:
    "The final scientific paper reconciles every stage without hiding the Stage 8 failure: 18 dimension matches, zero sparse-nnz matches, 11 validated GPU rows, one CPU timeout, and three fail-closed resource outcomes.",
  gate: {
    command: "STAGE 10 LOCKED",
    state: "held" as "held" | "available",
    intro:
      "Stage 9 is the completed reporting boundary. The optional N-1 SCOPF extension is a separate research program and was not started.",
    note: "The final classification is D - structural reproduction. Stage 10 requires separate explicit approval.",
  },
  acceptance: {
    result: "fail" as "running" | "pass" | "fail",
    checkerPassed: 13 as number | null,
    checkerTotal: 13 as number | null,
    note:
      "13 / 13 checks validate the continuation; the original terminal audit remains 12 / 12 PASS and Stage 8 acceptance remains FAIL.",
  },
  guard: {
    usableFraction: "80%",
    formula:
      "min(0.8 x live host available, 0.8 x live CUDA free) >= host assembly peak + GPU planning footprint",
    liveGuardBudgetGiB: "65.50" as string | null,
    note:
      "At the continuation preflight, T16 projected 94.435 GiB. The host budget was 65.784 GiB and the smaller CUDA-free budget was 65.496 GiB, so allocation was denied.",
  },
  tasks: stageEightTasks,
  rows: [
    {
      order: 1,
      caseName: "case2868rte",
      horizon: "T48",
      role: "frozen",
      estimatedDeviceGiB: "12.110",
      estimatedHostGiB: "12.004",
      estimatedCombinedGiB: "24.114",
      reconstructedNnz: "229,507,104",
      status: "passed",
      correctness: "passed",
      highsMedianSeconds: "76.238645",
      cpuMedianSeconds: "804.863493",
      gpuMedianSeconds: "49.968351",
      note: "All required solver tracks passed.",
    },
    {
      order: 2,
      caseName: "case2868rte",
      horizon: "T64",
      role: "frozen",
      estimatedDeviceGiB: "16.155",
      estimatedHostGiB: "16.013",
      estimatedCombinedGiB: "32.169",
      reconstructedNnz: "306,303,136",
      status: "passed",
      correctness: "passed",
      highsMedianSeconds: "108.232479",
      cpuMedianSeconds: "1078.891699",
      gpuMedianSeconds: "68.383103",
      note: "All required solver tracks passed.",
    },
    {
      order: 3,
      caseName: "case2868rte",
      horizon: "T96",
      role: "frozen",
      estimatedDeviceGiB: "24.258",
      estimatedHostGiB: "24.045",
      estimatedCombinedGiB: "48.303",
      reconstructedNnz: "460,334,496",
      status: "passed",
      correctness: "passed",
      highsMedianSeconds: "163.592526",
      cpuMedianSeconds: "1621.905045",
      gpuMedianSeconds: "105.022444",
      note: "All required solver tracks passed.",
    },
    {
      order: 4,
      caseName: "case9241pegase",
      horizon: "T4",
      role: "frozen",
      estimatedDeviceGiB: "11.818",
      estimatedHostGiB: "11.788",
      estimatedCombinedGiB: "23.606",
      reconstructedNnz: "342,863,272",
      status: "passed",
      correctness: "passed",
      highsMedianSeconds: "142.478889",
      cpuMedianSeconds: "3087.217721",
      gpuMedianSeconds: "193.631896",
      note: "All required solver tracks passed.",
    },
    {
      order: 5,
      caseName: "case9241pegase",
      horizon: "T6",
      role: "frozen",
      estimatedDeviceGiB: "17.727",
      estimatedHostGiB: "17.682",
      estimatedCombinedGiB: "35.410",
      reconstructedNnz: "514,308,838",
      status: "failed",
      correctness: "failed",
      highsMedianSeconds: "963.956952",
      cpuMedianSeconds: null,
      gpuMedianSeconds: "357.543837",
      note:
        "CPU FP64 correctness returned SolveTimeLimit after 3,600.093 s during its included final original-space residual evaluation; CPU warm-up and measurements were not run. HiGHS and GPU FP64 passed.",
    },
    {
      order: 6,
      caseName: "case9241pegase",
      horizon: "T16",
      role: "frozen",
      estimatedDeviceGiB: "47.278",
      estimatedHostGiB: "47.157",
      estimatedCombinedGiB: "94.435",
      reconstructedNnz: "1,371,647,068",
      status: "memory-blocked",
      correctness: "not-run",
      highsMedianSeconds: null,
      cpuMedianSeconds: null,
      gpuMedianSeconds: null,
      note:
        "GPU-only continuation: the unchanged 94.435 GiB projection exceeded both live 80% budgets. No LP or solver track was allocated.",
    },
    {
      order: 7,
      caseName: "case9241pegase",
      horizon: "T24",
      role: "conditional",
      estimatedDeviceGiB: "70.922",
      estimatedHostGiB: "70.741",
      estimatedCombinedGiB: "141.663",
      reconstructedNnz: "2,057,650,132",
      status: "index-blocked",
      correctness: "not-run",
      highsMedianSeconds: null,
      cpuMedianSeconds: null,
      gpuMedianSeconds: null,
      note:
        "GPU-only continuation: the 2,531,600,260 planning nnz exceed signed-int32 CSR capacity. No allocation was attempted.",
    },
    {
      order: 8,
      caseName: "case9241pegase",
      horizon: "T32",
      role: "conditional",
      estimatedDeviceGiB: "94.569",
      estimatedHostGiB: "94.328",
      estimatedCombinedGiB: "188.897",
      reconstructedNnz: "2,743,770,956",
      status: "index-blocked",
      correctness: "not-run",
      highsMedianSeconds: null,
      cpuMedianSeconds: null,
      gpuMedianSeconds: null,
      note:
        "GPU-only continuation: the 3,375,704,460 planning nnz exceed signed-int32 CSR capacity. No allocation was attempted.",
    },
  ] satisfies StageEightCampaignRow[],
  evidence: [
    {
      label: "Final scientific report",
      name: "docs/final_reproduction_report.tex",
      availability: "available",
    },
    {
      label: "Archival report",
      name: "docs/final_reproduction_report.md",
      availability: "available",
    },
    {
      label: "Machine-readable result index",
      name: "results/stage_9_result_index.json",
      availability: "available",
    },
    {
      label: "Reproducibility checklist",
      name: "docs/reproducibility_checklist.md",
      availability: "available",
    },
    {
      label: "Frozen Stage 8 protocol",
      name: "configs/benchmarks/stage_8_large.json",
      availability: "available",
    },
    {
      label: "Stage 8 report",
      name: "docs/stage_reports/stage_8_report.md",
      availability: "available",
    },
    {
      label: "Terminal Stage 8 evidence",
      name: "results/raw/stage_8/stage_8_validation.json",
      availability: "available",
    },
    {
      label: "Independent 12-check audit",
      name: "results/raw/stage_8/stage_8_checks.json",
      availability: "available",
    },
    {
      label: "GPU-only continuation protocol",
      name: "configs/benchmarks/stage_8_gpu_only_completion.json",
      availability: "available",
    },
    {
      label: "GPU-only continuation evidence",
      name:
        "results/raw/stage_8/gpu_only_completion/stage_8_gpu_only_completion_validation.json",
      availability: "available",
    },
    {
      label: "Independent 13-check continuation audit",
      name:
        "results/raw/stage_8/gpu_only_completion/stage_8_gpu_only_completion_checks.json",
      availability: "available",
    },
  ] satisfies EvidenceArtifact[],
};

const locked = (id: string, title: string): ResearchTask => ({
  id,
  title,
  status: "locked",
});

export const stages: ResearchStage[] = [
  {
    id: 0,
    title: "Paper specification, repository, and environment audit",
    purpose: "Establish the evidence base before a single solver update is written.",
    status: "complete",
    tasks: [
      { id: "0.1", title: "Verify and fingerprint the source paper", status: "complete" },
      { id: "0.2", title: "Extract equations, algorithms, and dimensions", status: "complete" },
      { id: "0.3", title: "Classify exact-reproduction gaps", status: "complete" },
      { id: "0.4", title: "Audit local and DGX Spark environments", status: "complete" },
      { id: "0.5", title: "Create project documentation and scaffold", status: "complete" },
      { id: "0.6", title: "Run Stage 0 acceptance checks", status: "complete" },
    ],
  },
  {
    id: 1,
    title: "Generic toy LP and mathematical component tests",
    purpose: "Validate projections, residuals, and Halpern mechanics on problems we can solve by hand.",
    status: "complete",
    tasks: [
      { id: "1.1", title: "Canonical LP data structure", status: "complete" },
      { id: "1.2", title: "Box and nonnegative projections", status: "complete" },
      { id: "1.3", title: "KKT residual mapping", status: "complete" },
      { id: "1.4", title: "Analytic toy LP", status: "complete" },
      { id: "1.5", title: "Generic HPR reference", status: "complete" },
      { id: "1.6", title: "Halpern and reflection tests", status: "complete" },
      { id: "1.7", title: "Independent reference comparison", status: "complete" },
      { id: "1.8", title: "Stage checks", status: "complete" },
    ],
  },
  {
    id: 2,
    title: "CPU DCOPF model construction",
    purpose: "Translate network physics into the paper's canonical LP without introducing N-1 constraints.",
    status: "complete",
    tasks: [
      { id: "2.1", title: "Network data loading", status: "complete" },
      { id: "2.2", title: "PTDF and shift-factor construction", status: "complete" },
      { id: "2.3", title: "Deterministic variable indexing", status: "complete" },
      {
        id: "2.4",
        title: "Build A1, A2, right-hand sides, cost, and bounds",
        status: "complete",
      },
      { id: "2.5", title: "Small base model", status: "complete" },
      { id: "2.6", title: "Dimension checks", status: "complete" },
      { id: "2.7", title: "Solve with HiGHS", status: "complete" },
      { id: "2.8", title: "Independent physical validation", status: "complete" },
    ],
  },
  {
    id: 3,
    title: "CPU sGS-HPR reference implementation",
    purpose: "Implement Algorithm 2 conservatively before applying any GPU-specific shortcuts.",
    status: "complete",
    tasks: [
      { id: "3.1", title: "Paper update order", status: "complete" },
      { id: "3.2", title: "z and x update identity", status: "complete" },
      { id: "3.3", title: "Trusted direct y1 solve", status: "complete" },
      { id: "3.4", title: "Projected y2 update", status: "complete" },
      { id: "3.5", title: "Residuals and stopping", status: "complete" },
      { id: "3.6", title: "Fixed sigma baseline", status: "complete" },
      { id: "3.7", title: "Comparison against HiGHS", status: "complete" },
    ],
  },
  {
    id: 4,
    title: "Paper-specific structural equality solve",
    purpose: "Replace the dense-looking y1 solve with the diagonal and low-rank formula in Proposition 5.",
    status: "complete",
    tasks: [
      { id: "4.1", title: "Confirm A1 block structure", status: "complete" },
      { id: "4.2", title: "Implement Proposition 5", status: "complete" },
      { id: "4.3", title: "Cross-check against direct solves", status: "complete" },
      { id: "4.4", title: "Full-solver trajectory cross-check", status: "complete" },
      { id: "4.5", title: "Empirical complexity check", status: "complete" },
      { id: "4.6", title: "Stage 4 acceptance approved", status: "complete" },
    ],
  },
  {
    id: 5,
    title: "Preconditioning, restart, and penalty management",
    purpose: "Add paper-supported acceleration features one at a time and preserve the fixed-sigma baseline.",
    status: "complete",
    tasks: [
      { id: "5.1", title: "Ten reversible Ruiz scaling passes", status: "complete" },
      {
        id: "5.2",
        title: "Pock-Chambolle diagonal preconditioning with alpha one",
        status: "complete",
      },
      { id: "5.3", title: "Full-vector b and c normalization", status: "complete" },
      { id: "5.4", title: "Sourced adaptive penalty policy", status: "complete" },
      { id: "5.5", title: "Sourced restart policy at 100-step checks", status: "complete" },
      {
        id: "5.6",
        title: "Controls, preprocessing ablations, and acceptance approval",
        status: "complete",
      },
    ],
  },
  {
    id: 6,
    title: "GPU port for DGX Spark",
    purpose:
      "Run the validated method with resident FP64 GPU data, prove CPU parity, and disclose the complete timing boundary.",
    status: "complete",
    tasks: [
      { id: "6.1", title: "Device backend and transfer ledger", status: "complete" },
      { id: "6.2", title: "Resident LP data, state, matrices, and transposes", status: "complete" },
      { id: "6.3", title: "DGX sparse-format and transpose benchmarks", status: "complete" },
      { id: "6.4", title: "Sparse and vector operation parity", status: "complete" },
      { id: "6.5", title: "Reusable workspaces and allocation accounting", status: "complete" },
      { id: "6.6", title: "Synchronized phase and iteration timing", status: "complete" },
      { id: "6.7", title: "CPU and GPU trajectory cross-checks", status: "complete" },
      { id: "6.8", title: "FP64 gate and non-gating FP32 diagnostic", status: "complete" },
    ],
  },
  {
    id: 7,
    title: "Small and medium benchmark reproduction",
    purpose: "Reconstruct public cases transparently and validate every run before discussing speed.",
    status: "complete",
    tasks: [
      { id: "7.1", title: "Benchmark data provenance", status: "complete" },
      { id: "7.2", title: "Paper data availability decision", status: "complete" },
      { id: "7.3", title: "Dimension reproduction", status: "complete" },
      { id: "7.4", title: "Small and medium runs", status: "complete" },
      { id: "7.5", title: "Numerical and physical validation", status: "complete" },
      { id: "7.6", title: "Repeated fair timing", status: "complete" },
    ],
  },
  {
    id: 8,
    title: "Large paper-scale benchmarks",
    purpose: "Scale incrementally, estimate memory first, and preserve every failure as data.",
    status: stageEightDashboard.stageStatus,
    tasks: stageEightDashboard.tasks,
  },
  {
    id: 9,
    title: "Final reproduction report",
    purpose: "State exactly what was reproduced, what was reconstructed, and what remains unknown.",
    status: "complete",
    tasks: [
      { id: "9.1", title: "Synthesize mathematical and implementation record", status: "complete" },
      { id: "9.2", title: "Classify reproduction fidelity", status: "complete" },
      { id: "9.3", title: "Create reproducibility checklist and command index", status: "complete" },
      { id: "9.4", title: "Publish machine-readable result index", status: "complete" },
    ],
  },
  {
    id: 10,
    title: "Optional N-1 SCOPF research extension",
    purpose: "A separate post-reproduction track for contingencies, screening, and constraint generation.",
    status: "optional",
    tasks: [
      locked("10.1", "LODF-based contingency model"),
      locked("10.2", "Batched GPU screening"),
      locked("10.3", "Adaptive constraint generation"),
      locked("10.4", "Persistent warm starts"),
      locked("10.5", "Complete N-1 verification"),
    ],
  },
];

export const stageSevenTimings = [
  {
    caseName: "case1354pegase",
    horizon: "T4",
    highsMedian: "1.463376",
    cpuMedian: "13.190169",
    gpuMedian: "1.013252",
  },
  {
    caseName: "case1354pegase",
    horizon: "T16",
    highsMedian: "7.265354",
    cpuMedian: "47.398903",
    gpuMedian: "2.924268",
  },
  {
    caseName: "case1354pegase",
    horizon: "T48",
    highsMedian: "32.095478",
    cpuMedian: "153.046368",
    gpuMedian: "9.480950",
  },
  {
    caseName: "case1354pegase",
    horizon: "T96",
    highsMedian: "101.242296",
    cpuMedian: "336.657478",
    gpuMedian: "21.084461",
  },
  {
    caseName: "case2868rte",
    horizon: "T4",
    highsMedian: "5.366418",
    cpuMedian: "60.349654",
    gpuMedian: "3.938934",
  },
  {
    caseName: "case2868rte",
    horizon: "T16",
    highsMedian: "22.138002",
    cpuMedian: "250.601316",
    gpuMedian: "15.408042",
  },
] as const;

export const learningNotes = [
  {
    label: "Research design",
    title: "Why the CPU solver comes first",
    body:
      "A fast wrong answer is still wrong. A small CPU implementation gives us a readable oracle for every later structural and GPU optimization.",
  },
  {
    label: "Numerics",
    title: "Why FP64 is the baseline",
    body:
      "KKT residuals combine feasibility, bounds, and stationarity. FP64 keeps precision changes from being confused with algorithmic mistakes.",
  },
  {
    label: "Power systems",
    title: "DCOPF is not yet N-1 SCOPF",
    body:
      "The paper's published model uses base-case shift-factor limits. Contingency constraints are intentionally reserved for a separate extension.",
  },
  {
    label: "Method",
    title: "What sGS-HPR is buying us",
    body:
      "The symmetric Gauss-Seidel split isolates equality and inequality updates, while Halpern anchoring gives an O(1/k) KKT-residual guarantee.",
  },
  {
    label: "Stage 1 finding",
    title: "Feasibility before objective",
    body:
      "A first-order iterate can show an objective slightly better than the true optimum while still violating a constraint. Always read objective and feasibility together.",
  },
  {
    label: "Residuals",
    title: "Equation 28 is not Equation 54a",
    body:
      "The KKT mapping also tests multiplier complementarity; the paper's first stopping block checks primal feasibility only. Both are now computed and preserved.",
  },
  {
    label: "Stage 3 finding",
    title: "Check every iteration, store sparsely",
    body:
      "Equation 54 residuals oscillate. The solver therefore tests the stopping rule every iteration while writing only periodic trajectory samples plus the exact stopping point.",
  },
  {
    label: "Linear algebra",
    title: "Three estimates, one safe lambda",
    body:
      "Dense eigendecomposition is the small-case authority. Sparse eigsh and seeded power iteration cross-check it, then a positive margin keeps the projected y2 step valid.",
  },
  {
    label: "Physical validation",
    title: "An iterate is not an exact power flow",
    body:
      "First-order candidates retain a small balance error. That error is tested explicitly; only a temporary reference-bus adjustment is used to compare PTDF and angle flows.",
  },
  {
    label: "Stage 4 finding",
    title: "The Schur-complement sign decides the inverse",
    body:
      "Equation 43 gives a minus rank-one update, so its inverse needs a positive correction. The corrected formula matches Cholesky to FP64 accuracy; the paper's printed sign does not.",
  },
  {
    label: "Performance evidence",
    title: "Measure the boundary the theorem counts",
    body:
      "The paper's complexity claim includes forming the right-hand side with sparse A1 products. That full boundary scaled near-linearly; the shorter solve-only timing remains a disclosed diagnostic.",
  },
  {
    label: "Stage 5 preprocessing",
    title: "Scale for the solver, judge in original units",
    body:
      "Ruiz and Pock-Chambolle scaling make rows and columns numerically comparable. Every candidate is mapped back before KKT, objective, and power-system checks, so easier arithmetic cannot weaken the scientific acceptance test.",
  },
  {
    label: "Stage 5 controls",
    title: "Ablation separates the causes",
    body:
      "Fixed or adaptive penalty and restart on or off form four controlled runs. Comparing them shows which mechanism changes behavior instead of crediting the whole acceleration bundle at once.",
  },
  {
    label: "Stage 6 parity",
    title: "Match the path, not only the destination",
    body:
      "The FP64 CPU and GPU runs stop on the same iterations for both frozen cases. Matching intermediate states and policy events distinguishes an equivalent algorithmic path from two methods that merely finish near the same point.",
  },
  {
    label: "Sparse kernels",
    title: "A requested algorithm name is not evidence",
    body:
      "The DGX check reaches the low-level cuSPARSE descriptor and records that CSR_ALG2 was actually selected. A high-level sparse call that silently chooses its default would not support the paper-specific kernel claim.",
  },
  {
    label: "GPU residency",
    title: "Move results, not the iteration state",
    body:
      "The LP matrices, transposes, scaling, state vectors, and reusable workspaces stay on the GPU. Explicit host transfers are limited to setup, recorded diagnostics, and final recovery so transfer cost cannot hide inside an iteration claim.",
  },
  {
    label: "Precision study",
    title: "FP32 is a diagnostic after FP64",
    body:
      "FP64 first proves that the port preserves the CPU method at tight tolerances. The FP32 run is recorded separately and cannot weaken or replace that correctness gate.",
  },
  {
    label: "Stage 7 validation",
    title: "Correctness makes timing reportable",
    body:
      "Each HiGHS, CPU FP64, and GPU FP64 track first passed objective, normalized residual, raw KKT, and physical checks. Only then did its warm-up and measured repetitions become reportable evidence.",
  },
  {
    label: "Stage 7 structure",
    title: "Matching dimensions does not match sparse work",
    body:
      "All 18 Table II rows reproduce the published row and variable counts, but every reconstructed nonzero count differs. Sparse support changes memory traffic and runtime, so the paper's timings remain context rather than a direct comparison.",
  },
  {
    label: "Stage 8 safety",
    title: "Unified memory must be budgeted once",
    body:
      "DGX Spark lets the CPU and GPU share one memory pool. Adding the host assembly peak to the GPU planning footprint, then comparing that sum with the smaller live budget, prevents the same physical memory from being promised twice.",
  },
  {
    label: "Source boundary",
    title: "A sourced proxy is not author identity",
    body:
      "The paper names HPR-LP-style penalty and restart management but does not publish its exact DCOPF policy code. This stage transfers the published HPR-LP equations, uses the paper's 100-step interval, and labels that limit explicitly.",
  },
  {
    label: "Stage 2 finding",
    title: "A PTDF may need an offset",
    body:
      "Transformer phase shifts create affine branch flows. The implementation keeps that offset and checks it against an independent angle solve instead of treating every flow map as purely linear.",
  },
  {
    label: "Network modeling",
    title: "Topology is not the same as a limit",
    body:
      "A MATPOWER branch with RATE_A equal to zero is still electrically active. It belongs in the PTDF even though it contributes no thermal-limit row.",
  },
  {
    label: "Validation design",
    title: "Why the storage case is synthetic",
    body:
      "The paper does not publish device placements or time series. A separate labeled fixture forces real charge and discharge while keeping the public base network unmodified.",
  },
];

export const artifacts = [
  { label: "Primary source", name: "references/AnEfficientGPU-basedHalpernAccelerating.pdf" },
  { label: "Specification", name: "docs/paper_specification.md" },
  { label: "Pinned network bundle", name: "data/raw/matpower/stage7/README.md" },
  { label: "Stage evidence", name: "docs/stage_reports/stage_7_report.md" },
  { label: "Validation results", name: "results/raw/stage_7/stage_7_validation.json" },
  {
    label: "Independent checker",
    name: "results/raw/stage_7/stage_7_checks.json",
  },
  {
    label: "Evidence ledger",
    name: "results/raw/stage_7/README.md",
  },
  {
    label: "Frozen benchmark protocol",
    name: "configs/benchmarks/stage_7_small_medium.json",
  },
  {
    label: "DGX package pins",
    name: "environment/dgx_stage7_requirements.txt",
  },
];
