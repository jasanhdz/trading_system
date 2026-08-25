from __future__ import annotations

import ast
from pathlib import Path


SANDBOX = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORT_PREFIXES = (
    "aegis.",
    "binance",
    "ccxt",
    "requests",
    "websocket",
)


def test_runtime_has_no_production_exchange_or_network_imports() -> None:
    for path in sorted((SANDBOX / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not [
            name for name in imported
            if name == "aegis" or name.startswith(FORBIDDEN_IMPORT_PREFIXES)
        ], path


def test_all_w11_code_and_outputs_are_inside_disposable_sandbox() -> None:
    repository = SANDBOX.parents[1]
    assert SANDBOX == repository / "sandbox" / "aegis_ephemeral_regime_w11"
    assert (SANDBOX / "w11_ephemeral_regime_verdict.json").is_file()
    assert (SANDBOX / "w11_ephemeral_regime_result.md").is_file()
