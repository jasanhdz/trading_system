from __future__ import annotations


def entropy_for_lineage(lineage: str) -> float:
    return {"champion": 0.09, "bc": 0.12, "fresh": 0.16}.get(lineage, 0.10)
