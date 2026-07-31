"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { artifacts, learningNotes, stages, type StageStatus } from "./project-data";

type Filter = "all" | StageStatus;

const filterLabels: Array<{ key: Filter; label: string }> = [
  { key: "all", label: "All stages" },
  { key: "active", label: "Current" },
  { key: "complete", label: "Passed" },
  { key: "locked", label: "Locked" },
  { key: "optional", label: "Optional" },
];

const statusLabels: Record<StageStatus, string> = {
  complete: "Passed",
  active: "In progress",
  locked: "Locked",
  optional: "Optional",
};

export function ReproductionDashboard() {
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");

  const coreStages = stages.filter((stage) => stage.id <= 9);
  const activeStage = coreStages.find((stage) => stage.status === "active");
  const latestPassedStage = [...coreStages]
    .reverse()
    .find((stage) => stage.status === "complete");
  const currentStage = activeStage ?? latestPassedStage ?? stages[0];
  const nextLockedStage = coreStages.find(
    (stage) => stage.id > currentStage.id && stage.status === "locked",
  );
  const passedCoreStages = coreStages.filter((stage) => stage.status === "complete").length;
  const currentDone = currentStage.tasks.filter((task) => task.status === "complete").length;
  const currentProgress = Math.round((currentDone / currentStage.tasks.length) * 100);
  const stageFocusLabel = activeStage
    ? `Now running / Stage ${currentStage.id}`
    : nextLockedStage
      ? `Stage ${currentStage.id} passed / Stage ${nextLockedStage.id} awaits approval`
      : `Stage ${currentStage.id} passed`;

  const filteredStages = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return stages.filter((stage) => {
      const matchesFilter = filter === "all" || stage.status === filter;
      const matchesQuery =
        !normalized ||
        stage.title.toLowerCase().includes(normalized) ||
        stage.purpose.toLowerCase().includes(normalized) ||
        stage.tasks.some(
          (task) =>
            task.id.includes(normalized) || task.title.toLowerCase().includes(normalized),
        );
      return matchesFilter && matchesQuery;
    });
  }, [filter, query]);

  const orbitStyle = {
    "--progress": `${currentProgress}%`,
  } as CSSProperties;

  return (
    <main className="dashboard-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">HPR reproduction lab</span>
        </div>
        <nav className="topnav" aria-label="Dashboard sections">
          <a href="#roadmap">Roadmap</a>
          <a href="#evidence">Evidence</a>
          <a href="#learn">Learn</a>
        </nav>
      </header>

      <section className="hero-wrap" aria-labelledby="hero-title">
        <div className="hero">
          <div>
            <p className="eyebrow">DGX Spark research program / stage-gated</p>
            <h1 id="hero-title">From paper to verified GPU solver.</h1>
            <p className="hero-copy">
              Reproducing Wang et al.&apos;s GPU-based sGS-HPR method for large-scale
              DC optimal power flow - with every equation, assumption, test, and timing
              boundary preserved as evidence.
            </p>
            <div className="hero-actions">
              <button
                className="primary-action"
                onClick={() => document.querySelector("#roadmap")?.scrollIntoView()}
              >
                Inspect the research plan
              </button>
              <button
                className="secondary-action"
                onClick={() => document.querySelector("#learn")?.scrollIntoView()}
              >
                Why this workflow?
              </button>
            </div>
          </div>
          <aside className="hero-aside" aria-label="Current stage progress">
            <div className="progress-orbit" style={orbitStyle}>
              <div className="progress-core">
                <span className="progress-value">{currentProgress}%</span>
                <span className="progress-caption">of current stage tasks complete</span>
              </div>
            </div>
            <div className="hero-stage">
              <div className="hero-stage-label">{stageFocusLabel}</div>
              <p className="hero-stage-title">{currentStage.title}</p>
            </div>
          </aside>
        </div>

        <div className="metric-strip" aria-label="Project summary">
          <div className="metric-card">
            <div className="metric-label">Core stages passed</div>
            <div className="metric-value">{passedCoreStages} / {coreStages.length}</div>
            <div className="metric-note">Stage gates prevent unsupported shortcuts.</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">FP64 CPU / GPU iterations</div>
            <div className="metric-value">410 / 410</div>
            <div className="metric-note">
              Public T1 parity; the labeled T2 fixture also matched at 1,032 / 1,032.
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Largest final state difference</div>
            <div className="metric-value">2.62 × 10⁻¹⁴</div>
            <div className="metric-note">
              Maximum relative x, y, or z error across the final FP64 T1 state.
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">DGX sparse kernel</div>
            <div className="metric-value">CSR_ALG2</div>
            <div className="metric-note">
              True cuSPARSE algorithm selection verified on the NVIDIA GB10.
            </div>
          </div>
        </div>
      </section>

      <div className="main-grid">
        <section id="roadmap" aria-labelledby="roadmap-title">
          <div className="section-heading">
            <div>
              <h2 id="roadmap-title">Research roadmap</h2>
              <p>Open a stage to see its atomic work units and current gate state.</p>
            </div>
            <input
              className="search-box"
              aria-label="Search stages and tasks"
              placeholder="Search tasks..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>

          <div className="filter-row" aria-label="Filter stages">
            {filterLabels.map((option) => (
              <button
                key={option.key}
                className="filter-button"
                aria-pressed={filter === option.key}
                onClick={() => setFilter(option.key)}
              >
                {option.label}
              </button>
            ))}
          </div>

          <div className="stage-stack">
            {filteredStages.map((stage) => (
              <details
                className="stage-card"
                data-status={stage.status}
                key={stage.id}
                open={stage.status === "active" || stage.id === currentStage.id}
              >
                <summary className="stage-summary">
                  <span className="stage-number">S{String(stage.id).padStart(2, "0")}</span>
                  <span>
                    <span className="stage-title">{stage.title}</span>
                    <span className="stage-purpose">{stage.purpose}</span>
                  </span>
                  <span className="summary-tail">
                    <span className={`status-badge ${stage.status}`}>
                      {statusLabels[stage.status]}
                    </span>
                    <span className="stage-chevron" aria-hidden="true">+</span>
                  </span>
                </summary>
                <div className="stage-body">
                  <ul className="task-list">
                    {stage.tasks.map((task) => (
                      <li className="task-row" key={task.id}>
                        <span className={`task-signal ${task.status}`} aria-hidden="true" />
                        <span className="task-code">{task.id}</span>
                        <span className="task-title">{task.title}</span>
                        <span className="task-state">{task.status}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </details>
            ))}
            {filteredStages.length === 0 && (
              <div className="empty-state">No stage or task matches that view.</div>
            )}
          </div>
        </section>

        <aside className="side-rail">
          <section className="rail-card dark" aria-labelledby="gate-title">
            <h2 className="rail-title" id="gate-title">Approval gate</h2>
            <p className="rail-copy">
              Stage 5 is approved and Stage 6 passed its FP64 parity and original-space
              correctness gates. Stage 7 remains locked until you send the exact
              approval phrase.
            </p>
            <div className="gate-command">APPROVE STAGE 6 AND RUN STAGE 7</div>
          </section>

          <section className="rail-card" aria-labelledby="stage-six-title">
            <h2 className="rail-title" id="stage-six-title">Stage 6 FP64 parity brief</h2>
            <div className="machine-list">
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">Public case5 / T1</span>
                  <span className="machine-state">410 / 410</span>
                </div>
                <div className="machine-detail">
                  GPU KKT 0.005618456235; objective 17479.839088898956; maximum
                  final relative state error 2.62e-14.
                </div>
              </div>
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">Synthetic resource fixture / T2</span>
                  <span className="machine-state">1,032 / 1,032</span>
                </div>
                <div className="machine-detail">
                  GPU KKT 0.008948422297; objective 26580.274984099353; maximum
                  final relative state error 6.90e-15.
                </div>
              </div>
            </div>
            <div className="ablation-panel" aria-label="DGX implementation evidence">
              <div className="ablation-heading">DGX GB10 implementation boundary</div>
              <div className="ablation-grid">
                <div className="ablation-cell">
                  <span>Resident sparse operators</span>
                  <strong>A and A transpose</strong>
                </div>
                <div className="ablation-cell">
                  <span>Actual cuSPARSE selection</span>
                  <strong>CSR_ALG2</strong>
                </div>
                <div className="ablation-cell">
                  <span>Correctness baseline</span>
                  <strong>FP64 PASS</strong>
                </div>
                <div className="ablation-cell">
                  <span>Reduced precision</span>
                  <strong>FP32 diagnostic</strong>
                </div>
              </div>
              <p className="ablation-note">
                These frozen fixtures establish implementation parity. They are not a
                repeated performance benchmark, so Stage 6 makes no speedup claim.
              </p>
            </div>
          </section>

          <section className="rail-card source-boundary" aria-labelledby="source-boundary-title">
            <div className="chip">Claim boundary</div>
            <h2 className="rail-title" id="source-boundary-title">Parity before performance</h2>
            <p className="rail-copy">
              The paper used an A100 and did not publish a complete timing boundary.
              Stage 6 runs on a DGX Spark GB10 and records initialization, warm-up,
              allocation, transfers, iterations, residual checks, and recovery. Fair
              repeated benchmark comparisons belong to Stage 7.
            </p>
          </section>

          <section className="rail-card" aria-labelledby="rules-title">
            <h2 className="rail-title" id="rules-title">Non-negotiables</h2>
            <ul className="rule-list">
              <li>CPU correctness before GPU optimization.</li>
              <li>FP64 first; reduced precision is an experiment.</li>
              <li>No invented data or unstated N-1 constraints.</li>
              <li>No speedup claim from Stage 6 correctness fixtures.</li>
            </ul>
          </section>

          <section className="rail-card" aria-labelledby="machines-title">
            <h2 className="rail-title" id="machines-title">Lab link</h2>
            <div className="machine-list">
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">Research workstation</span>
                  <span className="machine-state">CPU reference</span>
                </div>
                <div className="machine-detail">
                  The trusted FP64 path supplies frozen states, residuals, objectives,
                  and policy events for the DGX cross-check.
                </div>
              </div>
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">DGX Spark</span>
                  <span className="machine-state">GPU verified</span>
                </div>
                <div className="machine-detail">
                  NVIDIA GB10; FP64 CPU/GPU parity and true cuSPARSE CSR_ALG2 selection
                  verified on the device.
                </div>
              </div>
            </div>
          </section>

          <section className="rail-card" id="evidence" aria-labelledby="evidence-title">
            <h2 className="rail-title" id="evidence-title">Evidence register</h2>
            <p className="rail-copy">Versioned artifacts make each decision reviewable.</p>
            <div className="evidence-grid">
              {artifacts.map((artifact) => (
                <div className="artifact" key={artifact.name}>
                  <div className="artifact-label">{artifact.label}</div>
                  <div className="artifact-name">{artifact.name}</div>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>

      <section className="learning-section" id="learn" aria-labelledby="learn-title">
        <div className="learning-inner">
          <h2 id="learn-title">Learning layer</h2>
          <p className="learning-intro">
            The dashboard records more than completion. These are the design ideas that
            make the reproduction trustworthy and the later GPU results interpretable.
          </p>
          <div className="learning-grid">
            {learningNotes.map((note) => (
              <article className="lesson-card" key={note.title}>
                <div className="chip">{note.label}</div>
                <h3>{note.title}</h3>
                <p>{note.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <footer className="footer">
        <span><strong>HPR Reproduction Lab</strong> / evidence before acceleration</span>
        <span>Current scope: published DCOPF model, not the optional N-1 extension</span>
      </footer>
    </main>
  );
}
