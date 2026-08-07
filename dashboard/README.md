# HPR reproduction dashboard

This companion dashboard visualizes the research stages, atomic tasks, gate
state, evidence files, machine status, and short learning notes for the
GPU DCOPF sGS-HPR reproduction.

Current gate: Stage 8 acceptance remains FAIL after the preserved T6 CPU
timeout. A separately authorized GPU-only continuation resolved T16, T24, and
T32 at unchanged preallocation guards with zero allocations. The Stage 9
structural-reproduction report is complete; Stage 10 remains locked.

The dashboard is evidence-driven. Its source data live in
`app/project-data.ts` and are updated when a stage report changes the
authoritative state in the research repository. Browser storage is not used as
the project record.

## Local use

```powershell
npm install
npm run dev
```

The development server prints its local URL. Run the production build and
rendered-output test with:

```powershell
npm test
```

The site uses the bundled Sites-compatible vinext worker structure.

## Interactive paper

Open `/paper` to read and annotate the concise scientific report. Select any
passage, enter a requested revision, and save it. Annotations are scoped to the
signed-in reader (or a stable anonymous browser session), persisted in the
private Sites D1 database, and can be reopened, marked applied, edited, or
deleted. `Copy for Codex` and `Download Markdown` produce a structured revision
brief without changing the paper source automatically.

Generate a database migration after changing `db/schema.ts`:

```powershell
npm run db:generate
```
