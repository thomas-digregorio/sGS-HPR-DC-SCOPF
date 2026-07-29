# Stage 0 report

Date: 2026-07-29  
Stage: 0 — Paper specification, repository, and environment audit  
Result: **PASS**

## 1. Objective

Establish an evidence-backed specification and implementation boundary before
writing solver code: preserve and read the paper, extract its mathematics and
algorithm, identify exact-reproduction gaps, audit the local workstation and
DGX Spark, scaffold the repository, and lock later work behind an explicit
approval gate.

## 2. Work performed

- Copied the supplied PDF unchanged into `references/`.
- Rendered and visually inspected all 17 pages and extracted searchable text.
- Recorded PDF size, page count, and SHA-256 in machine-readable evidence.
- Indexed Equations (1)-(55), Algorithms 1-2, Algorithm 2 update order,
  closed-form updates, stopping tests, dimensions, complexity claims, and
  Table II results.
- Independently derived the variable and constraint counts from constraint
  families.
- Classified evidence as explicit, derivable, referenced, ambiguous, or
  missing.
- Investigated public base-network cases and related HPR-LP/PDLP source
  availability without treating them as the authors' unpublished instance.
- Inspected the local Windows workstation and DGX Spark host through read-only
  probes. No packages were installed and no remote temporary files remain.
- Created the requested repository structure, documentation, inspection
  scripts, raw evidence, logs, and smoke tests.
- Built a private stage-tracking dashboard with roadmap filtering, task search,
  evidence cards, machine status, learning notes, and the exact approval gate.
- Published the dashboard as an owner-only deployment. Its URL and project
  identifier are intentionally omitted from the public repository.
- Confirmed that no solver implementation was created prematurely.

## 3. Mathematical decisions

### Canonical model boundary

The reproduced paper model is a multi-period base-case DCOPF LP. Despite the
paper's security-constrained terminology, it contains no outage or contingency
index. N-1 SCOPF remains a separate optional extension after the paper
reproduction.

The variable count is

`n = T(3N_G + N_RG + 2N_ESS)`.

The equality and inequality row counts used for implementation planning are

`m1 = T + N_ESS`,

`m2 = 2TN_L + (4T - 2)N_G + 2TN_ESS + 2T`,

and therefore

`m = 2TN_L + (4T - 2)N_G + (2T + 1)N_ESS + 3T`.

The planned canonical variable order is
`(p_G, p_RG, p_dc, p_ch, r_u, r_d)`, with every sign and row-family mapping
recorded in `docs/paper_specification.md`.

### Algorithm boundary

Algorithm 2 is documented as the paper's sGS-HPR scheme, including the
predictor/corrector equality updates, projected inequality update, primal
update, Halpern anchoring, and Eq. (54) stopping tests. Later implementation
will begin in FP64 and will validate each optimized formula against a direct
CPU reference.

No adaptive penalty, restart, structural inverse, GPU kernel, or synthetic
benchmark rule will be guessed merely to match a published timing.

## 4. Files created or modified

Principal artifacts:

- `README.md`
- `pyproject.toml`
- `references/AnEfficientGPU-basedHalpernAccelerating.pdf`
- `docs/paper_specification.md`
- `docs/mathematical_notes.md`
- `docs/reproduction_limits.md`
- `docs/decisions.md`
- `docs/project_state.md`
- `docs/stage_reports/stage_0_report.md`
- `environment/README.md`
- `environment/package_versions.txt`
- local-only machine inventories, excluded from the public repository
- `scripts/extract_paper_spec.py`
- `scripts/inspect_environment.py`
- `scripts/check_stage_0.py`
- `tests/test_repository_smoke.py`
- `results/raw/stage_0/paper_metadata.json`
- `results/raw/stage_0/stage_0_checks.json`
- `logs/stage_0/`
- `dashboard/`

The scaffold also includes `src/`, `configs/`, `data/`, `results/`, `logs/`,
`artifacts/`, and `notebooks/` placeholders for later gated stages.

## 5. Commands executed

Representative reproducible commands:

```text
python scripts/extract_paper_spec.py --pdf references/AnEfficientGPU-basedHalpernAccelerating.pdf --metadata-out results/raw/stage_0/paper_metadata.json
python scripts/inspect_environment.py --label local-workstation --output environment/local_environment.json --pretty
ssh <dgx-host> "python3 inspect_environment.py --label dgx-spark --output dgx_spark_environment.json --pretty"
python -m pytest -q
python -m py_compile scripts/check_stage_0.py scripts/extract_paper_spec.py scripts/inspect_environment.py src/gpu_dcopf_hpr/__init__.py
python scripts/check_stage_0.py --output results/raw/stage_0/stage_0_checks.json
npm run build
npm run lint
node --test tests/rendered-html.test.mjs
```

PDF rendering used the bundled Poppler `pdftoppm` utility. Environment and SSH
commands were read-only except for short-lived remote evidence files, which
were removed and verified absent.

## 6. Test results

| Check | Result |
|---|---|
| Repository smoke tests (`pytest -q`) | PASS — 2 passed |
| Python bytecode compilation | PASS |
| Deterministic Stage 0 acceptance script | PASS — 10/10 checks |
| Environment JSON parsing | PASS |
| Dashboard production build | PASS |
| Dashboard ESLint | PASS — no findings |
| Dashboard server-render tests | PASS — 2 passed |
| Ruff | Not run — not installed in the audited Stage 0 environment |
| mypy | Not run — not configured |

An early acceptance preflight correctly failed while this report,
`project_state.md`, and the gap register were still being written. The final
machine-readable result supersedes that expected intermediate failure.

## 7. Numerical validation results

No optimization solve belongs in Stage 0. Numerical validation was therefore
limited to independently recomputed structural dimensions:

| Case | Inputs from Table II | Recomputed `(m, n)` | Published `(m, n)` | Result |
|---|---:|---:|---:|---|
| case1354pegase T4 | `N_L=1991, N_G=260, N_RG=136, N_ESS=68` | `(20,192, 4,208)` | `(20,192, 4,208)` | exact match |
| case2868rte T16 | `N_L=3808, N_G=600, N_RG=286, N_ESS=143` | `(163,823, 37,952)` | `(163,823, 37,952)` | exact match |
| case9241pegase T6 | `N_L=16049, N_G=1445, N_RG=920, N_ESS=460` | `(230,376, 37,050)` | `(230,376, 37,050)` | exact match |

These matches validate the row and variable counting. They do not validate
unavailable numerical inputs, matrix entries, objectives, trajectories, or
timings.

## 8. Discrepancies from the paper

1. Eq. (55), the modeled constraints, and Table II imply
   `m1 = T + N_ESS`; Appendix A prints `T(1 + N_ESS)`.
2. Eq. (43) uses a minus rank-one update, whereas the inverse expressions in
   Eqs. (39), (44), and (45) use a plus-update pattern. Direct-solve comparison
   is mandatory before accepting Proposition 5 code.
3. The paper calls the model security-constrained, but the printed LP has only
   base-case line limits and no contingency set.
4. Absolute objective comparisons require restoring constants omitted by the
   compact objective.
5. The paper reports five independent runs but does not identify a random
   source that would make otherwise deterministic runs independent.

## 9. Unresolved questions

- Which exact case files, releases, edits, filters, slack buses, transformer
  conventions, and PTDF construction were used?
- Where were renewable generators and ESS devices placed?
- What load, renewable, reserve, ramping, ESS, and cost inputs generated the
  experiments?
- What are the complete adaptive-penalty and restart formulas and state
  transitions?
- How was the spectral bound estimated and protected from underestimation?
- Which floating-point precision, sparse transpose representation, reduction
  order, and initialization vectors were used?
- Which preprocessing, transfer, warm-up, synchronization, and result-transfer
  costs are included in Table II time?
- Can the authors provide the paper-specific source and benchmark instances?

Until these are resolved, an exact numerical or timing reproduction claim is
unsupported.

## 10. Acceptance-criteria checklist

- [x] The complete paper was successfully read and visually inspected.
- [x] The mathematical model was correctly summarized.
- [x] Algorithm 2 was documented.
- [x] The closed-form updates were documented.
- [x] The stopping criteria were documented.
- [x] Table dimensions were independently recalculated.
- [x] Missing reproduction information was listed.
- [x] Local and DGX Spark environment reports were generated.
- [x] The requested repository structure exists.
- [x] No unsupported exact-reproduction claim was made.

## 11. Result

**PASS.** Stage 0 meets all ten acceptance criteria. The project is stopped at
the Stage 1 gate, and no Stage 1 solver work has begun.

## 12. Approval command

To authorize the next stage, send exactly:

```text
APPROVE STAGE 0 AND RUN STAGE 1
```
