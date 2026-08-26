from __future__ import annotations

import ast
from pathlib import Path


SANDBOX = Path(__file__).resolve().parents[1]
FORBIDDEN = ("aegis", "binance", "ccxt", "requests", "websocket")


def test_sandbox_runtime_has_no_production_or_network_imports() -> None:
    for path in sorted((SANDBOX / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not [name for name in imports if name.split(".")[0] in FORBIDDEN], path


def test_w12_is_separate_from_w11_and_production() -> None:
    repository = SANDBOX.parents[1]
    assert SANDBOX == repository / "sandbox" / "aegis_ideal_entry_reverse_engineering_w12"
    assert "w11" not in SANDBOX.name
    assert not any("binance-futures-bot-ts" in str(path) for path in (SANDBOX / "src").rglob("*.py"))
