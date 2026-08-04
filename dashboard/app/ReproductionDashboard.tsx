"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import {
  artifacts,
  learningNotes,
  stageEightDashboard,
  stages,
  type StageStatus,
} from "./project-data";

type Filter = "all" | StageStatus;

const filterLabels: Array<{ key: Filter; label: string }> = [
  { key: "all", label: "All stages" },
  { key: "active", label: "Running" },
  { key: "stopped", label: "Stopped" },
  { key: "complete", label: "Passed" },
  { key: "locked", label: "Locked" },
  { key: "optional", label: "Optional" },
];

const statusLabels: Record<StageStatus, string> = {
  complete: "Passed",
  active: "In progress",
  stopped: "Stopped",
  locked: "Locked",
  optional: "Optional",
};

const stageEightStatusLabels = {
  queued: "Queued",
  running: "Running",
  passed: "Passed",
  locked: "Locked / not run",
  "memory-blocked": "Memory-blocked",
  "index-blocked": "Index-blocked",
  failed: "Failed",
} as const;

const correctnessLabels = {
  pending: "Pending",
  passed: "Passed",
  "not-run": "Not run",
  failed: "Failed",
} as const;

const timingValue = (value: string | null) => value ?? "—";

export function ReproductionDashboard() {
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");

  const coreStages = stages.filter((stage) => stage.id <= 9);
  const activeStage = coreStages.find((stage) => stage.status === "active");
  const stoppedStage = coreStages.find((stage) => stage.status === "stopped");
  const latestPassedStage = [...coreStages]
    .reverse()
    .find((stage) => stage.status === "complete");
  const currentStage = activeStage ?? stoppedStage ?? latestPassedStage ?? stages[0];
  const nextLockedStage = coreStages.find(
    (stage) => stage.id > currentStage.id && stage.status === "locked",
  );
  const currentDone = currentStage.tasks.filter((task) => task.status === "complete").length;
  const currentProgress = Math.round((currentDone / currentStage.tasks.length) * 100);
  const stageEightPassedRows = stageEightDashboard.rows.filter(
    (row) => row.status === "passed",
  );
  const stageEightFailedRows = stageEightDashboard.rows.filter(
    (row) => row.status === "failed",
  );
  const stageEightAttemptedRows = stageEightDashboard.rows.filter(
    (row) => row.status === "passed" || row.status === "failed",
  );
  const stageEightLockedRows = stageEightDashboard.rows.filter(
    (row) => row.status === "locked",
  );
  const stageEightRecordedRows = stageEightDashboard.rows.filter(
    (row) => !["queued", "running", "locked"].includes(row.status),
  );
  const checkerScore =
    stageEightDashboard.acceptance.checkerPassed === null ||
    stageEightDashboard.acceptance.checkerTotal === null
      ? "Pending"
      : `${stageEightDashboard.acceptance.checkerPassed} / ${stageEightDashboard.acceptance.checkerTotal}`;
  const stageFocusLabel = activeStage
    ? `Now running / Stage ${currentStage.id}`
    : stoppedStage
      ? `Stage ${currentStage.id} stopped / Stage 9 locked`
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
            <h1 id="hero-title">{stageEightDashboard.headline}</h1>
            <p className="hero-copy">{stageEightDashboard.summary}</p>
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
            <div className="metric-label">Correctness rows passed</div>
            <div className="metric-value">
              {stageEightPassedRows.length} / {stageEightAttemptedRows.length}
            </div>
            <div className="metric-note">
              T6 failed its CPU correctness gate; HiGHS and GPU FP64 passed.
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Attempted campaign rows</div>
            <div className="metric-value">
              {stageEightRecordedRows.length} / {stageEightDashboard.rows.length}
            </div>
            <div className="metric-note">
              {stageEightLockedRows.length} later rows remain locked and were not executed.
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Live unified-memory budget</div>
            <div className="metric-value">
              {stageEightDashboard.guard.liveGuardBudgetGiB ?? "Pending"}
            </div>
            <div className="metric-note">
              {stageEightDashboard.guard.liveGuardBudgetGiB ? "GiB at the final DGX preflight." : "Captured immediately before allocation."}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Independent checker</div>
            <div className="metric-value">{checkerScore}</div>
            <div className="metric-note">{stageEightDashboard.acceptance.note}</div>
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
                open={
                  stage.status === "active" ||
                  stage.status === "stopped" ||
                  stage.id === currentStage.id
                }
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
            <p className="rail-copy">{stageEightDashboard.gate.intro}</p>
            <div className={`gate-state ${stageEightDashboard.gate.state}`}>
              {stageEightDashboard.acceptance.result === "fail"
                ? "Locked after terminal failure"
                : stageEightDashboard.gate.state === "held"
                  ? "Held until acceptance"
                  : "Available"}
            </div>
            <div className="gate-command">{stageEightDashboard.gate.command}</div>
            <p className="gate-note">{stageEightDashboard.gate.note}</p>
          </section>

          <section className="rail-card stage-eight-card" aria-labelledby="stage-eight-title">
            <div className="chip">Frozen order / failure-preserving</div>
            <h2 className="rail-title" id="stage-eight-title">Stage 8 large-case campaign</h2>
            <p className="rail-copy">
              One new scale is admitted at a time. A failed correctness gate, timeout,
              live-memory block, or sparse-index block is a recorded result and stops
              unsafe allocation.
            </p>
            <div className="machine-list">
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">Campaign evidence</span>
                  <span className="machine-state">
                    {stageEightRecordedRows.length} / {stageEightDashboard.rows.length}
                  </span>
                </div>
                <div className="machine-detail">
                  {stageEightPassedRows.length} rows passed, {stageEightFailedRows.length} failed,
                  and {stageEightLockedRows.length} later rows were never executed.
                </div>
              </div>
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">Protocol state</span>
                  <span className="machine-state">{stageEightDashboard.phaseLabel}</span>
                </div>
                <div className="machine-detail">
                  Stage 9 stays locked. Count-only horizons T56, T72, T80, and T88 do not
                  enter this allocation campaign.
                </div>
              </div>
            </div>
            <div className="guard-panel" aria-label="DGX Spark unified-memory guard">
              <div className="ablation-heading">Unified-memory guard / smaller live budget wins</div>
              <p className="guard-formula">{stageEightDashboard.guard.formula}</p>
              <p className="ablation-note">{stageEightDashboard.guard.note}</p>
            </div>
            <div className="timing-panel" aria-labelledby="campaign-title">
              <div className="ablation-heading" id="campaign-title">
                Ordered preallocation ledger / GiB
              </div>
              <div className="timing-table-wrap">
                <table className="timing-table campaign-table">
                  <thead>
                    <tr>
                      <th scope="col">Order / benchmark</th>
                      <th scope="col">Host</th>
                      <th scope="col">GPU</th>
                      <th scope="col">Combined</th>
                      <th scope="col">Outcome</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stageEightDashboard.rows.map((row) => (
                      <tr key={`${row.order}-${row.caseName}-${row.horizon}`}>
                        <th scope="row">
                          <span>{row.order}. {row.caseName}</span>
                          <small>{row.horizon} / {row.role}</small>
                          <small>nnz {row.reconstructedNnz}</small>
                        </th>
                        <td>{row.estimatedHostGiB}</td>
                        <td>{row.estimatedDeviceGiB}</td>
                        <td>{row.estimatedCombinedGiB}</td>
                        <td>
                          <span className={`row-state ${row.status}`}>
                            {stageEightStatusLabels[row.status]}
                          </span>
                          {!["queued", "running", "passed"].includes(row.status) && (
                            <small className="row-note">{row.note}</small>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="ablation-note">
                Combined planning footprint adds the host assembly peak and GPU plan.
                T16, T24, and T32 remain planning-only estimates: the frozen campaign
                stopped at T6 before any of those rows were evaluated live or allocated.
              </p>
            </div>

            <div className="timing-panel" aria-labelledby="stage-eight-timing-title">
              <div className="ablation-heading" id="stage-eight-timing-title">
                Correctness and solver-core median seconds
              </div>
              <div className="timing-table-wrap">
                <table className="timing-table result-table">
                  <thead>
                    <tr>
                      <th scope="col">Benchmark</th>
                      <th scope="col">Check</th>
                      <th scope="col">HiGHS</th>
                      <th scope="col">CPU</th>
                      <th scope="col">GPU</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stageEightDashboard.rows.map((row) => (
                      <tr key={`result-${row.order}-${row.caseName}-${row.horizon}`}>
                        <th scope="row">
                          <span>{row.caseName}</span>
                          <small>{row.horizon}</small>
                        </th>
                        <td>{correctnessLabels[row.correctness]}</td>
                        <td>{timingValue(row.highsMedianSeconds)}</td>
                        <td>{timingValue(row.cpuMedianSeconds)}</td>
                        <td>{timingValue(row.gpuMedianSeconds)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="ablation-note">
                T6 has no CPU median because its first CPU correctness solve returned
                SolveTimeLimit at 3,600 seconds during the included final original-space
                residual evaluation. Its passing HiGHS and GPU medians are retained, but
                they do not make the overall row pass. Local and paper timings stay
                separate; no formal speedup is computed across unmatched sparse workloads.
              </p>
            </div>
          </section>

          <section className="rail-card source-boundary" aria-labelledby="source-boundary-title">
            <div className="chip">Claim boundary</div>
            <h2 className="rail-title" id="source-boundary-title">Local evidence stays local</h2>
            <p className="rail-copy">
              The reconstructed matrices do not share the paper&apos;s nonzero counts.
              Table II therefore presents local timings beside paper context without
              converting that difference into an algorithmic or paper speedup claim.
            </p>
          </section>

          <section className="rail-card" aria-labelledby="rules-title">
            <h2 className="rail-title" id="rules-title">Non-negotiables</h2>
            <ul className="rule-list">
              <li>CPU correctness before GPU optimization.</li>
              <li>FP64 first; reduced precision is an experiment.</li>
              <li>No invented data or unstated N-1 constraints.</li>
              <li>Unified host and GPU plans must fit the smaller live budget.</li>
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
                  Stage 7&apos;s accepted FP64 path remains the original-coordinate
                  correctness reference for every safe Stage 8 GPU allocation.
                </div>
              </div>
              <div className="machine">
                <div className="machine-head">
                  <span className="machine-name">DGX Spark</span>
                  <span className="machine-state">{stageEightDashboard.phaseLabel}</span>
                </div>
                <div className="machine-detail">
                  NVIDIA GB10; T6 cleared the memory guard, then stopped on the independent
                  CPU correctness time limit. No later Stage 8 allocation was attempted.
                </div>
              </div>
            </div>
          </section>

          <section className="rail-card" id="evidence" aria-labelledby="evidence-title">
            <h2 className="rail-title" id="evidence-title">Evidence register</h2>
            <p className="rail-copy">Versioned artifacts make each decision reviewable.</p>
            <div className="evidence-grid">
              {stageEightDashboard.evidence.map((artifact) => (
                <div className="artifact" key={artifact.name} data-availability={artifact.availability}>
                  <div className="artifact-label">
                    {artifact.label} / {artifact.availability}
                  </div>
                  <div className="artifact-name">{artifact.name}</div>
                </div>
              ))}
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
