# Environment evidence

The public repository retains package versions and the inspection script, but
not raw workstation or DGX Spark inventories. Those local JSON and Markdown
reports can contain hostnames, filesystem paths, network addresses, and other
machine-specific details.

Regenerate a local inventory with:

```text
python scripts/inspect_environment.py --label <machine-label> --output <local-output.json> --pretty
```

Do not commit credentials, access tokens, SSH configuration, or raw network
identifiers with regenerated evidence.

## Stage 6 reproducible environment

`package_versions.txt` preserves the original Stage 0 audit and a second
Stage 6 snapshot. The earlier `NOT INSTALLED` entries are historical findings,
not the current DGX state.

Stage 6 uses a repository-local Python 3.12 virtual environment on the DGX
Spark. Its exact Python package pins are in
`dgx_stage6_requirements.txt`. The public record includes the GPU model,
driver, CUDA, toolkit, Python, and package versions needed to interpret the
run, but continues to exclude the SSH alias, network address, credentials, and
raw host inventory.
