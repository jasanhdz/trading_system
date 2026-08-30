from pathlib import Path

path = Path("src/aegis/live_api.py")
text = path.read_text()

old_import = "from aegis.config import CANONICAL_SYMBOLS\nfrom aegis.live_decision import (\n"
new_import = (
    "from aegis.config import CANONICAL_SYMBOLS\n"
    "from aegis.context_inference import predict_from_snapshot\n"
    "from aegis.market_context import MarketContextError, market_snapshot_from_context\n"
    "from aegis.live_decision import (\n"
)
assert old_import in text, "expected Aegis import boundary not found"
text = text.replace(old_import, new_import, 1)

old_request = (
    "class PredictRequest(BaseModel):\n"
    "    model_config = ConfigDict(extra=\"forbid\")\n"
    "    symbol: str = Field(min_length=1, max_length=20)\n"
)
new_request = old_request + "    market_context: dict[str, Any] | None = None\n"
assert old_request in text, "PredictRequest contract not found"
text = text.replace(old_request, new_request, 1)

old_predict = '''        try:\n            result = dict(runtime.predict(symbol, trace_id()))\n        except CurrentBrainError as exc:\n            raise HTTPException(status_code=503, detail=str(exc)) from exc\n        except Exception as exc:\n            raise HTTPException(\n                status_code=503, detail="AEGIS_CURRENT_BRAIN_INFERENCE_FAILED"\n            ) from exc\n'''
new_predict = '''        try:\n            request_trace_id = trace_id()\n            if request.market_context is not None:\n                snapshot = market_snapshot_from_context(\n                    request.market_context,\n                    expected_symbol=symbol,\n                )\n                result = dict(\n                    predict_from_snapshot(runtime, snapshot, symbol, request_trace_id)\n                )\n                result.setdefault("metadata", {})["market_context"] = {\n                    "version": request.market_context.get("version"),\n                    "source": "TYPESCRIPT_SHARED_WEBSOCKET",\n                    "closed_at": snapshot.closed_at.isoformat().replace("+00:00", "Z"),\n                    "rest_snapshot_provider_used": False,\n                }\n            else:\n                # Compatibility/recovery path. The normal TypeScript bot supplies\n                # market_context so Python does not refetch the 11 canonical series.\n                result = dict(runtime.predict(symbol, request_trace_id))\n                result.setdefault("metadata", {})["market_context"] = {\n                    "source": "PYTHON_PUBLIC_REST_RECOVERY",\n                    "rest_snapshot_provider_used": True,\n                }\n        except MarketContextError as exc:\n            raise HTTPException(status_code=422, detail=str(exc)) from exc\n        except CurrentBrainError as exc:\n            raise HTTPException(status_code=503, detail=str(exc)) from exc\n        except Exception as exc:\n            raise HTTPException(\n                status_code=503, detail="AEGIS_CURRENT_BRAIN_INFERENCE_FAILED"\n            ) from exc\n'''
assert old_predict in text, "predict execution boundary not found"
text = text.replace(old_predict, new_predict, 1)

assert "market_context: dict[str, Any] | None = None" in text
assert "predict_from_snapshot(runtime, snapshot, symbol, request_trace_id)" in text
assert '"rest_snapshot_provider_used": False' in text
assert "runtime.predict(symbol, request_trace_id)" in text

path.write_text(text)
