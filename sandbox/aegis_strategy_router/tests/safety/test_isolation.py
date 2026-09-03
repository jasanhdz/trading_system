from pathlib import Path
import ast

import aegis.research.live_entry_multitimeframe as existing_features

from aegis_strategy_router.safety.import_audit import audit_source_tree


def test_phase1_source_has_no_network_or_financial_capability() -> None:
    source = Path(__file__).resolve().parents[2] / "src"
    result = audit_source_tree(source)
    assert result.files_scanned > 0
    assert result.safe, result.violations


def test_production_does_not_import_sandbox() -> None:
    repository = Path(__file__).resolve().parents[4]
    production_roots = [repository / "src", repository / "binance-futures-bot-ts" / "src"]
    offenders = []
    for root in production_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".ts", ".tsx", ".js"}:
                continue
            if "aegis_strategy_router" in path.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(path))
    assert offenders == []


def test_reused_research_adapter_has_only_data_science_imports() -> None:
    path = Path(existing_features.__file__).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert roots <= {"__future__", "math", "typing", "numpy", "pandas"}
