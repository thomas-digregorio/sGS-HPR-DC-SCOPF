# HPR reproduction dashboard

This companion dashboard visualizes the research stages, atomic tasks, gate
state, evidence files, machine status, and short learning notes for the
GPU DCOPF sGS-HPR reproduction.

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

