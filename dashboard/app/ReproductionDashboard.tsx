"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import {
  artifacts,
  learningNotes,
  stages,
  stageSevenTimings,
  type StageStatus,
} from "./project-data";

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
            <h1 id="hero-title">Six benchmark runs, structurally verified.</h1>
            <p className="hero-copy">
              Stage 7 completed the predeclared small and medium DGX Spark campaign
              across HiGHS, CPU FP64 sGS-HPR, and GPU FP64 sGS-HPR. Every correctness
              gate passed; structural differences keep the paper&apos;s timings contextual.
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
            <div className="metric-label">Accepted benchmark rows</div>
            <div className="metric-value">6 / 6</div>
            <div className="metric-note">All three required solver tracks passed.</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Independent checker</div>
            <div className="metric-value">19 / 19</div>
            <div className="metric-note">Every evidence and boundary check passed.</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Reconstructed nnz matches</div>
            <div className="metric-value">0 / 18</div>
            <div className="metric-note">
              Dimensions match, but all 18 sparse nonzero counts differ from the paper.
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Longest measured solve</div>
            <div className="metric-value">537.835392 s</div>
            <div className="metric-note">
              Preserved CPU FP64 sample; it is not a solver-core median.
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
              Stage 7 passed its six-case correctness and timing campaign plus the
              independent 19-check audit. Stage 8 large runs remain locked until you
              send the exact approval phrase.
            </p>
            <div className="gate-command">APPROVE STAGE 7 AND RUN STAGE 8</div>
          </section>

          <section className="rail-card" aria-labelledby="stage-seven-title">
            <h2 className="rail-title" id="stage-seven-title">Stage 7 structural benchmark brief</h2>
            <p className="rail-copy">
              Six predeclared rows completed required HiGHS, CPU FP64, and GPU FP64
              correctness solves before any measured repetitions were accepted.
            </p>
            <div className="machine-list">
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">Accepted campaign</span>
                  <span className="machine-state">6 / 6</span>
                </div>
                <div className="machine-detail">
                  Four case1354pegase horizons and two case2868rte horizons passed
                  every solver, precision, objective, residual, and physical gate.
                </div>
              </div>
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">Structural ledger</span>
                  <span className="machine-state">18 / 18 differ</span>
                </div>
                <div className="machine-detail">
                  Published row and variable counts are reproduced exactly, while all
                  reconstructed nonzero counts differ from the paper.
                </div>
              </div>
            </div>
            <div className="ablation-panel" aria-label="Stage 7 numerical maxima">
              <div className="ablation-heading">Maximum across accepted candidates</div>
              <div className="ablation-grid">
                <div className="ablation-cell">
                  <span>Normalized stopping block</span>
                  <strong>4.1797236508e-6</strong>
                </div>
                <div className="ablation-cell">
                  <span>Raw KKT norm</span>
                  <strong>0.0096347433</strong>
                </div>
                <div className="ablation-cell">
                  <span>Physical violation</span>
                  <strong>0.00622103995</strong>
                </div>
                <div className="ablation-cell">
                  <span>Objective gap</span>
                  <strong>4.2849939e-8</strong>
                </div>
              </div>
            </div>
            <div className="timing-panel" aria-labelledby="timing-title">
              <div className="ablation-heading" id="timing-title">
                Solver-core median seconds
              </div>
              <div className="timing-table-wrap">
                <table className="timing-table">
                  <thead>
                    <tr>
                      <th scope="col">Benchmark</th>
                      <th scope="col">HiGHS</th>
                      <th scope="col">CPU FP64</th>
                      <th scope="col">GPU FP64</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stageSevenTimings.map((timing) => (
                      <tr key={timing.caseName + "-" + timing.horizon}>
                        <th scope="row">
                          <span>{timing.caseName}</span>
                          <small>{timing.horizon}</small>
                        </th>
                        <td>{timing.highsMedian}</td>
                        <td>{timing.cpuMedian}</td>
                        <td>{timing.gpuMedian}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="ablation-note">
                These are separate local medians with named inclusion boundaries. No
                timing ratio or speedup is computed, and paper timings are context only.
              </p>
            </div>
          </section>

          <section className="rail-card source-boundary" aria-labelledby="source-boundary-title">
            <div className="chip">Claim boundary</div>
            <h2 className="rail-title" id="source-boundary-title">Structure before comparison</h2>
            <p className="rail-copy">
              The authors&apos; resource placements, time series, and matrix-construction
              code remain unavailable. Exact dimensions therefore coexist with different
              sparse support in all 18 rows. The paper&apos;s timings are context only, not
              directly comparable values for the DGX Spark reconstruction.
            </p>
          </section>

          <section className="rail-card" aria-labelledby="rules-title">
            <h2 className="rail-title" id="rules-title">Non-negotiables</h2>
            <ul className="rule-list">
              <li>CPU correctness before GPU optimization.</li>
              <li>FP64 first; reduced precision is an experiment.</li>
              <li>No invented data or unstated N-1 constraints.</li>
              <li>No direct timing comparison across different sparse workloads.</li>
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
                  The FP64 path passed all six Stage 7 benchmark rows and supplies the
                  original-coordinate correctness reference for each GPU cross-check.
                </div>
              </div>
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">DGX Spark</span>
                  <span className="machine-state">Stage 7 PASS</span>
                </div>
                <div className="machine-detail">
                  NVIDIA GB10; six resident FP64 GPU tracks passed the numerical,
                  physical, memory, transfer, and timing evidence gates.
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
