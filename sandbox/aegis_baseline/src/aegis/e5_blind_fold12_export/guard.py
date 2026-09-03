"""Clean-room path enforcement for E5 Phase 1A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import BlindExportError


@dataclass(frozen=True)
class CleanRoomGuard:
    combined_source: Path
    sealed_manifest: Path
    semi_blind_roots: tuple[Path, ...]
    lockbox_roots: tuple[Path, ...]

    def validate_export_source(self, requested: Path) -> Path:
        resolved = requested.resolve(strict=True)
        if resolved != self.combined_source.resolve(strict=True):
            raise BlindExportError("E5_BLIND_EXPORT_SOURCE_AUTHORITY_BLOCKED", "source path is not authoritative")
        self._deny_prohibited(resolved)
        return resolved

    def validate_downstream_path(self, requested: Path) -> Path:
        resolved = requested.resolve(strict=False)
        self._deny_prohibited(resolved)
        combined = self.combined_source.resolve(strict=False)
        if resolved == combined or combined.parent == resolved or combined.parent in resolved.parents:
            raise BlindExportError("E5_PHASE1_COMBINED_SOURCE_ACCESS_PROHIBITED", "combined source access denied")
        if resolved != self.sealed_manifest.resolve(strict=False):
            raise BlindExportError(
                "E5_PHASE1_COMBINED_SOURCE_ACCESS_PROHIBITED",
                "downstream path is not sealed manifest",
            )
        return resolved

    def _deny_prohibited(self, resolved: Path) -> None:
        for root in self.semi_blind_roots:
            denied = root.resolve(strict=False)
            if resolved == denied or denied in resolved.parents:
                raise BlindExportError("SEMIBLIND_ACCESS_ATTEMPT", "semi-blind path denied")
        for root in self.lockbox_roots:
            denied = root.resolve(strict=False)
            if resolved == denied or denied in resolved.parents:
                raise BlindExportError("LOCKBOX_ACCESS_ATTEMPT", "lockbox path denied")
