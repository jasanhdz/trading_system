"""Inference adapter for externally supplied causal MarketSnapshot values."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Mapping

from aegis.domain import MarketSnapshot
from aegis.live_decision import CurrentBrainDecisionService, CurrentBrainError, compatibility_response


def predict_from_snapshot(
    service: CurrentBrainDecisionService,
    snapshot: MarketSnapshot,
    symbol: str,
    request_trace_id: str,
) -> Mapping[str, Any]:
    """Evaluate the current brain without asking its REST SnapshotProvider for data."""
    with service._metrics_lock:  # noqa: SLF001 - adapter shares service metrics
        service.request_count += 1
    try:
        batch = _batch_from_snapshot(service, snapshot)
        return compatibility_response(batch, symbol, request_trace_id)
    except Exception:
        with service._metrics_lock:  # noqa: SLF001 - adapter shares service metrics
            service.error_count += 1
        raise


def _batch_from_snapshot(
    service: CurrentBrainDecisionService,
    snapshot: MarketSnapshot,
) -> Mapping[str, Any]:
    now = time.monotonic()
    closed_at = snapshot.closed_at.isoformat().replace("+00:00", "Z")

    # Reuse the exact existing Current Brain cache semantics. Closed 5m bars are
    # immutable, so a batch with the same market_timestamp is equivalent even
    # when multiple symbols request it during the same cycle.
    with service._lock:  # noqa: SLF001 - adapter intentionally shares service cache
        if service._cache is not None and service._cache[1].get("market_timestamp") == closed_at:  # noqa: SLF001
            service._cache = (now, service._cache[1])  # noqa: SLF001
            return service._cache[1]  # noqa: SLF001

        batch = service.engine.evaluate(snapshot)
        if service.research_observer is not None:
            try:
                overlay = service.research_observer.observe_batch(batch)
            except Exception as exc:
                with service._metrics_lock:  # noqa: SLF001 - adapter shares service metrics
                    service.research_error_count += 1
                mode = str(
                    getattr(
                        service.research_observer.mode,
                        "value",
                        service.research_observer.mode,
                    )
                )
                if mode == "LIVE":
                    raise CurrentBrainError(
                        "AEGIS_ENTRY_QUALITY_V2_LIVE_EVALUATION_FAILED"
                    ) from exc
                overlay = {}
            batch = {**batch, "_entry_quality_v2": overlay}

        if service.hybrid_live_selector is not None:
            batch = {
                **batch,
                "_hybrid_directional_live": service.hybrid_live_selector.apply(batch),
            }
        if service.v17_challenger_config is not None:
            batch = {
                **batch,
                "_v17_execution_challenger": service.v17_challenger_config.health(),
            }

        service._cache = (now, batch)  # noqa: SLF001
        with service._metrics_lock:  # noqa: SLF001 - adapter shares service metrics
            service.last_inference_at = datetime.now(timezone.utc)
        return batch
