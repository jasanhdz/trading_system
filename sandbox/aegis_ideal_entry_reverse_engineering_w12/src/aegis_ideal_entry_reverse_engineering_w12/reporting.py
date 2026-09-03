"""Deterministic W12 artifacts, manifest, verdict, and 24-question report."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import SANDBOX, sha256_file
from .experiment import ExperimentResult


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
    if value is pd.NaT or (not isinstance(value, (str, dict, list, tuple)) and isinstance(pd.isna(value), (bool, np.bool_)) and pd.isna(value)):
        return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(dict(value)), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    for column in output.columns:
        if output[column].dtype == "object":
            output[column] = output[column].map(
                lambda value: json.dumps(_clean(value), sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list)) else value
            )
    output.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _fmt(value: Any, suffix: str = "") -> str:
    return "no disponible" if value is None else f"{value:.3f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"


def build_report(result: ExperimentResult) -> str:
    summary = result.summary
    metrics = summary["prospective_metrics"]
    economics = summary["prospective_economics"]
    counts = summary["counts"]
    top = metrics["top"]
    features = result.feature_analysis.head(8)[["feature", "standardized_median_difference", "rank_biserial", "decile_monotonicity", "mutual_information"]].to_dict(orient="records")
    labels = result.label_analysis[result.label_analysis["partition"].eq("PROSPECTIVE")]
    teacher_rates = {
        key.upper(): float(labels[f"teacher_{key}_good_rate"].mean()) if len(labels) else None
        for key in "abcde"
    }
    side_stability = result.stability[result.stability["dimension"].eq("predicted_side")].to_dict(orient="records")
    symbol_stability = result.stability[result.stability["dimension"].eq("symbol")]
    month_stability = result.stability[result.stability["dimension"].eq("month")]
    controls = result.negative_controls.to_dict(orient="records")
    answers = [
        f"El modelo tuvo PR AUC {_fmt(metrics['pr_auc'])} frente a prevalencia {_fmt(metrics['prevalence'])}; veredicto {summary['verdict']}.",
        f"Las mayores diferencias discovery, sin autoridad de selección posterior, fueron: {features}.",
        "Se incluyeron perfiles T-60/T-30/T-15/T-5/T-1; su evidencia está en feature_analysis y no se añadió ninguna secuencia post hoc.",
        f"Resultados por lado predicho: {side_stability or 'sin operaciones'}.",
        f"Prevalencia prospectiva media por teacher: {teacher_rates}; consistencia completa en label_analysis.csv.",
        f"Majority fue el label primario preregistrado; strict y weighted se conservaron como diagnóstico y no reemplazaron el resultado.",
        f"La formulación seleccionada fue {summary['selected_candidate']['formulation']} ({summary['selected_candidate']['name']}).",
        f"El mejor horizonte validado y congelado fue {summary['selected_candidate']['horizon_minutes']} minutos.",
        f"Existían {counts['prospective_ideal_zones']} best-entry zones prospectivas entre todos los teachers/sides/horizontes.",
        f"El sistema permaneció SKIP en {_fmt(100 * economics['skip_fraction'] if economics['skip_fraction'] is not None else None, '%')} del universo del candidato.",
        f"Precision top 1/2/5/10%: { {key: top[key]['precision'] for key in ('1','2','5','10')} }.",
        f"Gross medio del top 2% congelado: {_fmt(economics['gross_mean_bps'], ' bps')}.",
        f"Neto a 14 bps: {_fmt(economics['net14_mean_bps'], ' bps')}.",
        f"Neto a 20 bps: {_fmt(economics['net20_mean_bps'], ' bps')}.",
        f"Prospective fue abierto una vez tras selección en validation; produjo {counts['prospective_opportunities']} señales.",
        "No se ejecutó un walk-forward rolling. La evidencia disponible es un único recorrido discovery→validation→prospective con meses prospectivos reportados por separado; por tanto, este diagnóstico no demuestra supervivencia walk-forward.",
        f"Se operaron {counts['prospective_symbols']} símbolos; máxima concentración y detalle: {symbol_stability.sort_values('trades', ascending=False).head(5).to_dict(orient='records') if len(symbol_stability) else []}.",
        f"Resultados mensuales: {month_stability.to_dict(orient='records')}.",
        f"Baselines completos: {[row for row in controls if row['kind'] == 'BASELINE']}.",
        f"Controles negativos: {[row for row in controls if row['kind'].startswith('NEGATIVE')]}; gate {'pasó' if summary['negative_controls_passed'] else 'falló'}.",
        f"El subconjunto top 2% {'fue' if economics['net14_mean_bps'] is not None and economics['net14_mean_bps'] > 0 else 'no fue'} rentable después de 14 bps.",
        f"W12.1 {'está justificado' if summary['merits_w12_1'] else 'no está justificado'} por los gates preregistrados.",
        "Shadow no está justificado: incluso grado A solo autorizaría una segunda fase gobernada, nunca integración automática.",
        f"Decisión final: {summary['verdict']}; la línea {'avanza solo a nueva evidencia independiente' if summary['merits_w12_1'] else 'se cierra con esta evidencia'}.",
    ]
    questions = [
        "¿Las entradas ideales presentan características detectables antes de ocurrir?", "¿Qué features las distinguen?",
        "¿Existen secuencias temporales recurrentes antes de ellas?", "¿LONG y SHORT tienen ADN diferente?",
        "¿Qué teacher produce labels más consistentes?", "¿Consensus labels funcionan mejor?",
        "¿Es mejor clasificación o quality score?", "¿Qué horizonte funciona mejor?",
        "¿Cuántas oportunidades A+ existen?", "¿Qué porcentaje del mercado es SKIP?",
        "¿Cuál es precision@top1/2/5/10%?", "¿Cuál es gross bps?", "¿Cuál es net bps con 14 bps?",
        "¿Cuál es net bps con 20 bps?", "¿Sobrevive fuera de muestra?", "¿Sobrevive walk-forward?",
        "¿Depende de un símbolo?", "¿Depende de un periodo?", "¿Supera baselines?",
        "¿Supera negative controls?", "¿Existe realmente un subconjunto A+ económicamente rentable?",
        "¿Existe suficiente evidencia para justificar un W12.1?", "¿Existe suficiente evidencia para considerar Shadow?",
        "¿O debemos cerrar esta línea?",
    ]
    lines = [
        "# W12 Ideal Entry Reverse Engineering Result", "", "## Veredicto", "",
        f"**{summary['grade']} - {summary['verdict']}**", "",
        f"Leakage audit: **{summary['leakage_audit']}**. External holdouts accessed: **false**.", "",
        "## Preguntas Finales", "",
    ]
    for index, (question, answer) in enumerate(zip(questions, answers, strict=True), 1):
        lines.extend([f"### {index}. {question}", "", answer, ""])
    lines.extend([
        "## Inferencia", "",
        f"Bootstrap UTC-day de {summary['bootstrap']['draws']} draws: CI 95% [{_fmt(summary['bootstrap']['ci_lower'])}, {_fmt(summary['bootstrap']['ci_upper'])}] bps net14; P(mean>0)={_fmt(summary['bootstrap']['probability_positive'])}.", "",
        "## Limitaciones", "",
        "Los labels usan OHLC 1m y resolución adverse-first, pero no BBO, queue position, funding ni fills observados. MFE es movimiento disponible; la economía primaria usa una política fija +30/-20/neither-horizon. Los datos 2022-2023 no prueban vigencia actual.", "",
        "## Autoridad", "",
        "Este resultado no autoriza E4, TypeScript, Shadow, producción, órdenes ni despliegue.", "",
    ])
    return "\n".join(lines)


def write_results(result: ExperimentResult, config: Mapping[str, Any], repository: Path) -> dict[str, Any]:
    artifacts = SANDBOX / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    frames = {
        "feature_analysis.csv": result.feature_analysis,
        "label_analysis.csv": result.label_analysis,
        "candidate_metrics.csv": result.candidate_metrics,
        "prospective_predictions.csv": result.prospective_predictions,
        "prospective_trades.csv": result.prospective_trades,
        "economic_metrics.csv": result.economic_metrics,
        "negative_controls.csv": result.negative_controls,
        "stability.csv": result.stability,
    }
    for filename, frame in frames.items():
        write_csv(artifacts / filename, frame)
    write_json(artifacts / "summary.json", result.summary)
    model_path = artifacts / "selected_model.joblib"
    joblib.dump(result.selected_candidate, model_path, compress=3)
    report_path = SANDBOX / "w12_ideal_entry_result.md"
    report_path.write_text(build_report(result), encoding="utf-8")
    verdict = {
        "experiment_id": config["experiment_id"], "grade": result.summary["grade"],
        "verdict": result.summary["verdict"], "gates": result.summary["gates"],
        "leakage_audit": result.summary["leakage_audit"],
        "external_holdouts_accessed": False, "no_production_authority": True,
        "merits_w12_1": result.summary["merits_w12_1"], "merits_shadow": False,
    }
    verdict_path = SANDBOX / "w12_ideal_entry_verdict.json"
    write_json(verdict_path, verdict)
    config_path = SANDBOX / "config" / "w12_frozen.json"
    output_paths = [artifacts / filename for filename in frames] + [artifacts / "summary.json", model_path, report_path, verdict_path]
    manifest = {
        "schema_version": "aegis-w12-result-manifest-v1", "experiment_id": config["experiment_id"],
        "seed": int(config["seed"]), "preregistration_checkpoint": "0484906",
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": result.cache_manifest["input_manifest_sha256"],
        "source_sha256": result.cache_manifest["source_sha256"],
        "feature_schema_sha256": result.cache_manifest["feature_schema_sha256"],
        "code_sha256": {path.relative_to(SANDBOX).as_posix(): sha256_file(path) for path in sorted((SANDBOX / "src").rglob("*.py"))},
        "output_sha256": {path.relative_to(SANDBOX).as_posix(): sha256_file(path) for path in output_paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "external_holdouts_accessed": False,
    }
    manifests = SANDBOX / "manifests"
    manifests.mkdir(exist_ok=True)
    write_json(manifests / "w12_result_manifest.json", manifest)
    return manifest
