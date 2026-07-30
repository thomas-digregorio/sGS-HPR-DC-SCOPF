export type StageStatus = "complete" | "active" | "locked" | "optional";
export type TaskStatus = "complete" | "working" | "queued" | "locked";

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
      { id: "5.6", title: "Four-way controls and preprocessing ablations", status: "complete" },
    ],
  },
  {
    id: 6,
    title: "GPU port for DGX Spark",
    purpose: "Move the validated method to resident FP64 GPU arrays and measure the complete timing boundary.",
    status: "locked",
    tasks: [
      locked("6.1", "Backend abstraction"),
      locked("6.2", "GPU data residency"),
      locked("6.3", "Sparse-format benchmarks"),
      locked("6.4", "Profile-guided vector kernels"),
      locked("6.5", "Buffer reuse"),
      locked("6.6", "Synchronized GPU timing"),
      locked("6.7", "CPU/GPU numerical cross-check"),
      locked("6.8", "FP64-first precision study"),
    ],
  },
  {
    id: 7,
    title: "Small and medium benchmark reproduction",
    purpose: "Reconstruct public cases transparently and validate every run before discussing speed.",
    status: "locked",
    tasks: [
      locked("7.1", "Benchmark data provenance"),
      locked("7.2", "Paper data availability decision"),
      locked("7.3", "Dimension reproduction"),
      locked("7.4", "Small and medium runs"),
      locked("7.5", "Numerical and physical validation"),
      locked("7.6", "Repeated fair timing"),
    ],
  },
  {
    id: 8,
    title: "Large paper-scale benchmarks",
    purpose: "Scale incrementally, estimate memory first, and preserve every failure as data.",
    status: "locked",
    tasks: [
      locked("8.1", "Memory and resource estimate"),
      locked("8.2", "Incremental scale-up"),
      locked("8.3", "Failure preservation"),
      locked("8.4", "Table II reconstruction"),
      locked("8.5", "Careful speedup interpretation"),
    ],
  },
  {
    id: 9,
    title: "Final reproduction report",
    purpose: "State exactly what was reproduced, what was reconstructed, and what remains unknown.",
    status: "locked",
    tasks: [
      locked("9.1", "Synthesize mathematical and implementation record"),
      locked("9.2", "Classify reproduction fidelity"),
      locked("9.3", "Create reproducibility checklist and command index"),
      locked("9.4", "Publish machine-readable result index"),
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
  { label: "Public network", name: "data/raw/matpower/case5.m" },
  { label: "Stage evidence", name: "docs/stage_reports/stage_5_report.md" },
  { label: "Validation results", name: "results/raw/stage_5/stage_5_validation.json" },
  {
    label: "Trajectories and policy events",
    name: "results/raw/stage_5/stage_5_trajectories.jsonl.gz",
  },
  {
    label: "Independent checker",
    name: "results/raw/stage_5/stage_5_checks.json",
  },
  {
    label: "Solver settings",
    name: "configs/sgs_hpr/stage_5_preconditioning_controls.json",
  },
  {
    label: "Base row map",
    name: "results/raw/stage_2/case5_base_t1_row_metadata.jsonl.gz",
  },
  {
    label: "Synthetic row map",
    name: "results/raw/stage_2/case5_synthetic_extension_t2_row_metadata.jsonl.gz",
  },
];
