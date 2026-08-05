# Stage 9 regeneration and test commands

Run commands from the repository root. `python` denotes the project Python
environment with the packages in `pyproject.toml`; the DGX-specific package
pins are in `environment/dgx_stage7_requirements.txt`.

## Regenerate every Stage 9 table, figure, and index

```powershell
python scripts/generate_stage_9_artifacts.py
```

This one command deterministically regenerates:

- `results/tables/stage_9_stage_checks.csv`;
- `results/tables/stage_9_benchmarks.csv`;
- `results/tables/stage_9_structural_reconciliation.csv`;
- `results/tables/stage_9_resource_boundaries.csv`;
- `results/tables/stage_9_timing_decomposition.csv`;
- `results/plots/stage_9_solver_timings.svg`;
- `results/plots/stage_9_resource_boundaries.svg`; and
- `results/stage_9_result_index.json`.

Verify that checked-in generated outputs are current:

```powershell
python scripts/generate_stage_9_artifacts.py --check
```

## Compile the LaTeX paper

```powershell
New-Item -ItemType Directory -Force tmp/pdfs/stage9 | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error `
  -output-directory=tmp/pdfs/stage9 docs/final_reproduction_report.tex
pdflatex -interaction=nonstopmode -halt-on-error `
  -output-directory=tmp/pdfs/stage9 docs/final_reproduction_report.tex
Copy-Item tmp/pdfs/stage9/final_reproduction_report.pdf `
  output/pdf/final_reproduction_report.pdf
```

Two direct `pdflatex` passes are the portable command for the audited Windows
environment. `latexmk` may be substituted when its Perl runtime is installed.

## Render and inspect the PDF

```powershell
pdfinfo output/pdf/final_reproduction_report.pdf
pdftotext -layout output/pdf/final_reproduction_report.pdf `
  tmp/pdfs/stage9/final_reproduction_report.txt
pdftoppm -png -r 150 output/pdf/final_reproduction_report.pdf `
  tmp/pdfs/stage9/render/page
```

The page PNGs must be inspected for clipping, overlap, unreadable tables,
broken math, and orphaned headings.

## Run the independent Stage 9 checker

```powershell
python scripts/check_stage_9.py `
  --output results/raw/stage_9/stage_9_checks.json
```

## Run all tests and static checks

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m py_compile scripts/generate_stage_9_artifacts.py `
  scripts/check_stage_9.py
```

## Validate the dashboard

```powershell
Set-Location dashboard
npm test
npm run lint
Set-Location ..
```

## Re-run earlier independent evidence checkers

These commands audit preserved evidence and do not rerun the benchmark
campaigns:

```powershell
python scripts/check_stage_0.py `
  --output results/raw/stage_0/stage_0_checks.json
python scripts/check_stage_1.py `
  --output results/raw/stage_1/stage_1_checks.json
python scripts/check_stage_2.py `
  --output results/raw/stage_2/stage_2_checks.json
python scripts/check_stage_3.py `
  --output results/raw/stage_3/stage_3_checks.json
python scripts/check_stage_4.py `
  --output results/raw/stage_4/stage_4_checks.json
python scripts/check_stage_5.py `
  --output results/raw/stage_5/stage_5_checks.json
python scripts/check_stage_6.py `
  --output results/raw/stage_6/stage_6_checks.json
python scripts/check_stage_7.py `
  --output results/raw/stage_7/stage_7_checks.json
python scripts/check_stage_8.py `
  --output results/raw/stage_8/stage_8_checks.json
python scripts/check_stage_8_gpu_only_completion.py `
  --output results/raw/stage_8/gpu_only_completion/stage_8_gpu_only_completion_checks_local.json
```

Do not run `run_stage_7.py`, `run_stage_8.py`, or
`run_stage_8_gpu_only_completion.py` merely to rebuild this report. Those
commands allocate benchmark workloads; the accepted evidence is intentionally
immutable.

## Final Git-state audit

```powershell
git diff --check
git status --short
git rev-parse HEAD
git log -1 --format=fuller
```

The final `git status --short` command should produce no output after the
Stage 9 commit. Stage 10 must remain locked.
