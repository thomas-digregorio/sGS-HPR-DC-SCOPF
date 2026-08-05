# Stage 9 reporting contract

Date frozen: 2026-08-05

## Scope

Stage 9 is a reporting and audit stage. It may read, reconcile, summarize, and
validate accepted Stage 0--8 evidence. It may not allocate a new optimization
problem, rerun a benchmark, change a numerical threshold, reinterpret a
resource guard, or begin the optional N-1 extension.

## Evidence hierarchy

When two sources differ, use the first applicable source in this order:

1. the supplied paper PDF and its Stage 0 SHA-256 record;
2. immutable machine-readable Stage 0--8 validation and checker JSON;
3. frozen benchmark configuration and source manifests;
4. stage reports and mathematical notes;
5. the dashboard, which is a presentation layer only.

Derived values must be recomputable from the machine-readable evidence. A
checker PASS means that the recorded protocol was followed; it does not change
the scientific acceptance result of the stage being checked.

## Final classification

The five allowed labels are:

- A: exact reproduction;
- B: near-exact reproduction;
- C: mathematical reproduction;
- D: structural reproduction;
- E: partial reproduction.

The preregistered Stage 9 decision rule selects **D: structural
reproduction** when all of the following hold:

- the canonical LP, sGS-HPR updates, stopping tests, and structural equality
  solve have independent mathematical and numerical checks;
- a resident FP64 GPU path matches the CPU oracle and proves the requested
  sparse-kernel path;
- public-network reconstructions match the paper's row and variable counts;
- the authors' numerical instances, sparse supports, source code, hardware,
  and timing boundary are not reproduced exactly; and
- the final benchmark campaign contains an honestly preserved acceptance
  failure or resource boundary.

All five conditions are satisfied by the frozen evidence. The classification
must not be promoted by a protocol-checker PASS or demoted merely because the
largest rows were safely blocked before allocation.

## Report acceptance gates

Stage 9 is complete only if all of the following are true:

1. `docs/final_reproduction_report.md` contains the 16 required subjects.
2. A LaTeX source and compiled, visually inspected PDF present the same final
   classification and material results.
3. A reproducibility checklist, regeneration command index, and
   machine-readable result index exist.
4. Every quantitative Stage 7--8 result in the report is reconciled against
   the original JSON; high-impact values are independently recomputed.
5. Timing figures state that local solver times do not establish a controlled
   speedup against the paper.
6. Stage 8 remains FAIL, its original checker remains PASS 12/12, and the
   GPU-only continuation remains `COMPLETE_WITH_RESOURCE_LIMITS` with checker
   PASS 13/13.
7. Stage 10 remains locked and no N-1 work is performed.
8. Python tests, Stage 9 checks, formatting/lint checks, and dashboard checks
   pass, or any unavailable check is explicitly recorded.

## Self-reference boundary

A Git commit cannot contain a truthful statement of its own not-yet-created
commit identifier. The report therefore records the evidence commits and the
command used to verify final repository state. The actual final commit and
clean `git status --short` result are reported after the commit is created.
