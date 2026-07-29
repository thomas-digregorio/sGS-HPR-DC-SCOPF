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
