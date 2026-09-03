"""AST-based proof that Phase 1 source has no network or financial imports."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_MODULE_ROOTS = frozenset({
    "aiohttp", "binance", "ccxt", "httpx", "requests", "socket", "urllib",
    "websocket", "websockets",
})
FORBIDDEN_CALL_NAMES = frozenset({
    "cancel_all_orders", "cancel_order", "change_leverage", "create_order",
    "futures_account", "futures_create_order", "futures_position_information",
    "new_order", "place_order", "set_leverage",
})
FORBIDDEN_IMPORT_PREFIXES = (
    "aegis.execution",
    "aegis.position",
    "aegis.runtime",
    "aegis.trading",
)
ALLOWED_EXTERNAL_AEGIS_IMPORTS = frozenset({"aegis.research.live_entry_multitimeframe"})


@dataclass(frozen=True, slots=True)
class SafetyAuditResult:
    files_scanned: int
    violations: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return not self.violations


def audit_source_tree(root: Path) -> SafetyAuditResult:
    violations: list[str] = []
    files = sorted(root.rglob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _audit_import(alias.name, path, violations)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                _audit_import(module, path, violations)
            elif isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in FORBIDDEN_CALL_NAMES:
                    violations.append(f"{path}:{node.lineno}: forbidden financial call {name}")
    return SafetyAuditResult(len(files), tuple(sorted(violations)))


def _audit_import(module: str, path: Path, violations: list[str]) -> None:
    root = module.split(".", 1)[0]
    if root in FORBIDDEN_MODULE_ROOTS:
        violations.append(f"{path}: forbidden network/exchange import {module}")
    if module.startswith(FORBIDDEN_IMPORT_PREFIXES):
        violations.append(f"{path}: forbidden production import {module}")
    if root == "aegis" and module not in ALLOWED_EXTERNAL_AEGIS_IMPORTS:
        violations.append(f"{path}: undeclared external Aegis import {module}")


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""

