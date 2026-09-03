"""Deterministic artifact and natural-language report generation for W11."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn

from .experiment import ExperimentResult, FRAME_NAMES

ARTIFACT_FILES = {name: f"{name}.csv" for name in FRAME_NAMES}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if not isinstance(value, (str, dict, list, tuple)):
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    return value


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_clean(dict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    ordered = frame.copy()
    for column in ordered.columns:
        if ordered[column].dtype == "object":
            ordered[column] = ordered[column].map(
                lambda value: json.dumps(_clean(value), sort_keys=True, separators=(",", ":"))
                if isinstance(value, dict) else value
            )
    ordered.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _fmt(value: Any, suffix: str = "") -> str:
    return "unavailable" if value is None else f"{value:.3f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"


def _report(result: ExperimentResult) -> str:
    summary = result.summary
    prospective = summary["prospective"]
    gates = summary["gates"]
    expirations = summary["expiration_counts"]
    half_life = summary["half_life"]
    questions = [
        "¿Existe edge efímero?", "¿En qué ventanas aparece?", "¿LONG, SHORT o ambos?",
        "¿Qué horizonte funciona mejor?", "¿Cuál es el edge bruto?", "¿Cuál es el edge neto?",
        "¿Cuánto dura aproximadamente?", "¿Cuál es el edge half-life?",
        "¿Qué ocurre después de que el modelo envejece?",
        "¿Regime similarity predice correctamente la muerte del edge?",
        "¿Expiration Guardian mejora el resultado?", "¿Cuántas instancias habrían sido creadas?",
        "¿Cuántas expiraron por TTL?", "¿Cuántas por regime drift?",
        "¿Cuántas por edge decay?", "¿Cuántos trades/candidatos genera?",
        "¿Qué porcentaje del tiempo permanece SKIP?", "¿Supera costos?",
        "¿Es suficientemente robusto para justificar Shadow?",
        "¿Justifica alguna integración futura con E4/TS?",
    ]
    answers = [
        f"Se detectaron candidatos locales en validación, pero no edge prospectivo: {summary['verdict']} y {_fmt(prospective['net14_mean_bps'], ' bps')} netos por trade.",
        f"Economía por ventana: {prospective['economics_by_window']}. La mejor es la de mayor net14, aunque siga negativa.",
        f"Hubo {prospective['long_trades']} LONG y {prospective['short_trades']} SHORT. Economía por lado: {prospective['economics_by_side']}.",
        f"Economía por horizonte: {prospective['economics_by_horizon']}. El mejor es el de mayor net14, sin reinterpretarlo como ganador.",
        f"El gross medio orientado fue {_fmt(prospective['gross_mean_bps'], ' bps')}.",
        f"El neto medio fue {_fmt(prospective['net14_mean_bps'], ' bps')} a 14 bps, {_fmt(prospective['net20_mean_bps'], ' bps')} a 20 bps y {_fmt(prospective['net30_mean_bps'], ' bps')} a 30 bps.",
        f"La vida económica se observa por buckets hasta 48h; medianas netas: {half_life['median_age_bucket_net14_bps']}.",
        f"El sistema no tuvo edge inicial positivo que pueda partirse por la mitad. Entre instancias aisladas con inicio positivo, la mediana observada fue {_fmt(half_life['median_observed_hours'], 'h')} en {half_life['observed_instances']}; {half_life['censored_instances']} quedaron censuradas.",
        f"La sensibilidad TTL fue {summary['ttl_sensitivity']}; es descriptiva y no tuvo autoridad de selección.",
        f"Spearman similarity/net fue {_fmt(prospective['similarity_net_spearman'])}; el gate {'pasó' if gates['similarity_edge_relationship'] else 'falló'}.",
        f"El Guardian cambió expectancy por trade en {_fmt(summary['guardian_delta_net14_bps'], ' bps')} y redujo drawdown acumulado en {_fmt(summary['guardian_drawdown_reduction_bps'], ' bps')}. El gate {'pasó por reducción de exposición/drawdown' if gates['guardian_improvement'] else 'falló'}; no rescató la economía.",
        f"Se crearon {summary['prospective_instances_created']} instancias prospectivas ({summary['selected_instances']} incluyendo validación).",
        f"Expiraron {expirations['TTL']} por TTL; {summary['instances_censored_at_partition_end']} quedaron censuradas al final del periodo.",
        f"Expiraron {expirations['REGIME_DRIFT']} por regime drift.",
        f"Expiraron {expirations['EDGE_DECAY']} por edge decay.",
        f"Se ejecutaron {prospective['trade_count']} trades; se evaluaron {summary['prospective_candidate_evaluations']} candidatos y {summary['prospective_eligible_candidates']} pasaron gates.",
        f"El sistema permaneció SKIP en {_fmt(100.0 * prospective['skip_fraction_all_market_slots'] if prospective['skip_fraction_all_market_slots'] is not None else None, '%')} de snapshots-símbolo prospectivos.",
        f"Los gates de 14/20 bps {'pasaron' if gates['baseline_net_positive'] and gates['stress_net_positive'] else 'no pasaron'}; CI 95% diario [{_fmt(prospective['day_bootstrap_ci_lower'])}, {_fmt(prospective['day_bootstrap_ci_upper'])}] bps. Baselines: {summary['baselines']}.",
        f"{'Sí' if summary['merits_phase_two'] else 'No'} justifica avanzar a una segunda fase controlada; grado {summary['grade']}.",
        f"{'Solo justifica una propuesta futura, no implementación' if summary['merits_phase_two'] else 'No justifica integración futura'} con E4/TS. Este estudio no concede autoridad de producción.",
    ]
    lines = [
        "# W11 Ephemeral Regime Result", "", f"Generado con datos hasta `{result.data_end.isoformat()}`.", "",
        f"## Veredicto\n\n**{summary['grade']} - {summary['verdict']}**", "", "## Preguntas Requeridas", "",
    ]
    for index, (question, answer) in enumerate(zip(questions, answers, strict=True), 1):
        lines.extend([f"### {index}. {question}", "", answer, ""])
    lines.extend([
        "## Limitations", "",
        "This is an offline candle study. Funding, spread, queue position, intrabar ordering, and measured slippage are unavailable. The 2023 historical period is not evidence that the same relationship exists now. Repeated symbols share market shocks, and the configured external holdouts remained sealed.", "",
        "## Authority", "",
        "This result has no production, Shadow, E4, promotion, or TypeScript authority.", "",
    ])
    if summary["merits_phase_two"]:
        lines.extend([
            "## Phase Two Architecture Proposal", "",
            "A phase-two design should preserve immutable model IDs and causal creation, run a read-only shadow scorer behind an append-only attribution/expiration journal, monitor delayed outcomes separately, and require a new governed holdout before any promotion discussion.", "",
        ])
    return "\n".join(lines)


def write_results(
    result: ExperimentResult,
    config: Mapping[str, Any],
    sandbox_dir: str | Path,
    *,
    repository_dir: str | Path,
) -> dict[str, Any]:
    root = Path(sandbox_dir).resolve()
    repository = Path(repository_dir).resolve()
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for name, filename in ARTIFACT_FILES.items():
        _csv(artifacts / filename, getattr(result, name))
    _json(artifacts / "summary.json", result.summary)

    verdict = {
        "experiment_id": config["experiment_id"], "grade": result.summary["grade"],
        "verdict": result.summary["verdict"], "all_success_gates_passed": all(result.summary["gates"].values()),
        "gates": result.summary["gates"], "no_production_authority": True,
    }
    _json(root / "w11_ephemeral_regime_verdict.json", verdict)
    (root / "w11_ephemeral_regime_result.md").write_text(_report(result), encoding="utf-8")

    config_path = root / "config" / "w11_frozen.json"
    source_manifest = repository / config["source"]["manifest"]
    source_hashes: dict[str, str] = {"manifest": sha256_file(source_manifest)}
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    for symbol in config["source"]["symbols"]:
        source_hashes[symbol] = source_payload["symbols"][symbol]["parquet_sha256"]
    code_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted((root / "src").rglob("*.py"))
    }
    output_paths = [artifacts / filename for filename in ARTIFACT_FILES.values()] + [
        artifacts / "summary.json", root / "w11_ephemeral_regime_verdict.json",
        root / "w11_ephemeral_regime_result.md",
    ]
    manifest = {
        "experiment_id": config["experiment_id"], "schema_version": config["schema_version"],
        "generated_at": result.data_end.isoformat(), "data_start": result.data_start.isoformat(),
        "data_end": result.data_end.isoformat(), "seed": int(config["seed"]),
        "config_sha256": sha256_file(config_path), "source_sha256": source_hashes,
        "code_sha256": code_hashes,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "output_sha256": {path.relative_to(root).as_posix(): sha256_file(path) for path in output_paths},
        "external_holdouts_accessed": False,
    }
    _json(artifacts / "manifest.json", manifest)
    return manifest
