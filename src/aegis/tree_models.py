"""Inspectable JSON tree ensembles with deterministic dependency-free evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .utils import Sha256HashProvider


class TreeModelError(ValueError):
    pass


class EnsembleAggregation(str, Enum):
    AVERAGE = "AVERAGE"
    ADDITIVE = "ADDITIVE"
    ADDITIVE_LOGIT = "ADDITIVE_LOGIT"


@dataclass(frozen=True)
class TreeNode:
    feature_index: int
    threshold: float
    left: int
    right: int
    value: float
    is_leaf: bool
    missing_go_to_left: bool = False


@dataclass(frozen=True)
class DecisionTree:
    nodes: tuple[TreeNode, ...]

    def evaluate(self, row: Sequence[float]) -> float:
        index = 0
        for _ in range(len(self.nodes) + 1):
            node = self.nodes[index]
            if node.is_leaf:
                if not math.isfinite(node.value):
                    raise TreeModelError("tree leaf is non-finite")
                return node.value
            if not 0 <= node.feature_index < len(row):
                raise TreeModelError("tree feature index is outside input")
            value = float(row[node.feature_index])
            index = node.left if (math.isnan(value) and node.missing_go_to_left) or (not math.isnan(value) and value <= node.threshold) else node.right
            if not 0 <= index < len(self.nodes):
                raise TreeModelError("tree child index is invalid")
        raise TreeModelError("tree traversal contains a cycle")


@dataclass(frozen=True)
class TreeEnsemble:
    ensemble_id: str
    schema_version: str
    feature_names: tuple[str, ...]
    aggregation: EnsembleAggregation
    base_value: float
    trees: tuple[DecisionTree, ...]
    content_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-tree-ensemble-v1" or not self.ensemble_id:
            raise TreeModelError("unsupported tree ensemble contract")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names) or not self.trees:
            raise TreeModelError("tree ensemble dimensions are invalid")
        if not math.isfinite(self.base_value):
            raise TreeModelError("tree ensemble base value is invalid")

    def evaluate(self, row: Sequence[float]) -> float:
        if len(row) != len(self.feature_names):
            raise TreeModelError("tree input dimension mismatch")
        values = tuple(tree.evaluate(row) for tree in self.trees)
        if self.aggregation is EnsembleAggregation.AVERAGE:
            return math.fsum(values) / len(values)
        raw = self.base_value + math.fsum(values)
        if self.aggregation is EnsembleAggregation.ADDITIVE:
            return raw
        if raw >= 0:
            return 1.0 / (1.0 + math.exp(-min(raw, 700.0)))
        exp = math.exp(max(raw, -700.0))
        return exp / (1.0 + exp)

    def unsigned_payload(self) -> Mapping[str, Any]:
        return {
            "ensemble_id": self.ensemble_id, "schema_version": self.schema_version,
            "feature_names": list(self.feature_names), "aggregation": self.aggregation.value,
            "base_value": self.base_value,
            "trees": [[{
                "feature_index": node.feature_index, "threshold": node.threshold,
                "left": node.left, "right": node.right, "value": node.value,
                "is_leaf": node.is_leaf, "missing_go_to_left": node.missing_go_to_left,
            } for node in tree.nodes] for tree in self.trees],
        }

    def to_payload(self) -> Mapping[str, Any]:
        payload = dict(self.unsigned_payload())
        payload["content_hash"] = Sha256HashProvider().digest_value(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TreeEnsemble":
        unsigned = dict(payload); claimed = str(unsigned.pop("content_hash", ""))
        if claimed != Sha256HashProvider().digest_value(unsigned):
            raise TreeModelError("tree ensemble content hash mismatch")
        try:
            trees = tuple(DecisionTree(tuple(TreeNode(
                feature_index=int(node["feature_index"]), threshold=float(node["threshold"]),
                left=int(node["left"]), right=int(node["right"]), value=float(node["value"]),
                is_leaf=bool(node["is_leaf"]), missing_go_to_left=bool(node.get("missing_go_to_left", False)),
            ) for node in tree)) for tree in payload["trees"])
            return cls(
                ensemble_id=str(payload["ensemble_id"]), schema_version=str(payload["schema_version"]),
                feature_names=tuple(str(name) for name in payload["feature_names"]),
                aggregation=EnsembleAggregation(str(payload["aggregation"])),
                base_value=float(payload["base_value"]), trees=trees, content_hash=claimed,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TreeModelError("tree ensemble payload is invalid") from exc
