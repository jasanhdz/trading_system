# Aegis

This sandbox owns the Aegis inference API, its operational configuration, and
the Aegis research and training workflows that existed before the architecture
migration.

Run the API through PM2 with:

```bash
pm2 start sandbox/aegis_baseline/deploy/ecosystem.config.js
```

Run its tests from this directory so relative experiment paths resolve inside
the sandbox:

```bash
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python -m pytest tests
```

New independent work belongs in a sibling `sandbox/<experiment>/` directory.
It may import only from `common/`; it must not import another experiment or
this sandbox's private modules.
