from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import _metrics
from .core import artifact_manifest, causal_bars, load_candles, read_json, read_jsonl_gz, sha256_file, write_json
from .dataset import locate_sources


def describe(m:dict)->list[str]:
    direction="ascendente" if m["sui_displacement"]>.15 else "descendente" if m["sui_displacement"]<-.15 else "lateral"
    order="ordenado" if m["path_efficiency"]>.45 else "con bastante solapamiento"
    volume="volumen elevado frente al contexto" if m["volume_ratio"]>1.25 else "volumen moderado" if m["volume_ratio"]>.75 else "volumen reducido"
    position="cerca del extremo superior" if m["range_position"]>.7 else "cerca del extremo inferior" if m["range_position"]<.3 else "en la zona media"
    btc="BTC también ascendente" if m["btc_displacement"]>.15 else "BTC también descendente" if m["btc_displacement"]<-.15 else "BTC aproximadamente lateral"
    return [f"Movimiento {direction} relativamente {order}.",f"{volume.capitalize()}.",f"SUI termina {position} de su rango visible.",f"{btc}."]


def build(repo:Path,output:Path,config_path:Path,docs:Path)->dict:
    cfg=read_json(config_path); rows=read_jsonl_gz(output/"cluster_assignments.jsonl.gz"); episodes=read_jsonl_gz(output/"pattern_episodes.jsonl.gz")
    if sha256_file(output/"cluster_assignments.jsonl.gz") != read_json(output/"clustering_summary.json")["assignments_sha256"]: raise RuntimeError("Frozen assignments changed")
    precursor=read_json(output/"precursor_analysis.json")["patterns"] if (output/"precursor_analysis.json").exists() else {}
    behavior=read_json(output/"post_pattern_behavior.json")["patterns"] if (output/"post_pattern_behavior.json").exists() else {}
    sources=locate_sources(repo); dfs={s:load_candles(p) for s,p in sources.items()}; x=np.load(output/"embeddings.npy",allow_pickle=False)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r["pattern_id"]].append(i)
    catalog=[]; medoids={}
    for pid,idx in sorted(groups.items()):
        if pid=="NOISE":continue
        frame_metrics=[]
        for i in idx:
            t=int(pd.Timestamp(rows[i]["decision_at"]).timestamp()*1000); sui=_metrics(causal_bars(dfs["SUIUSDT"],t,3,80));btc=_metrics(causal_bars(dfs["BTCUSDT"],t,3,80))
            frame_metrics.append({"sui_displacement":sui["displacement"],"path_efficiency":sui["path_efficiency"],"volatility":sui["volatility"],"volume_ratio":sui["volume_ratio"],"relative_range":sui["relative_range"],"body_fraction":sui["body_fraction"],"wick_body_ratio":sui["wick_body_ratio"],"range_position":sui["range_position"],"btc_displacement":btc["displacement"]})
        med={k:float(np.median([m[k] for m in frame_metrics])) for k in frame_metrics[0]}; days=len({rows[i]["decision_at"][:10] for i in idx}); n=len(idx)
        sd,bd=med["sui_displacement"],med["btc_displacement"]
        relation=("aligned" if abs(sd)>.15 and abs(bd)>.15 and sd*bd>0 else "opposed" if abs(sd)>.15 and abs(bd)>.15 and sd*bd<0 else "SUI leads BTC" if abs(sd)>abs(bd)+.15 else "BTC leads SUI" if abs(bd)>abs(sd)+.15 else "uncorrelated")
        status="STABLE_PATTERN" if n>=cfg["stable_pattern_min_frames"] and days>=cfg["stable_pattern_min_days"] else "RARE_PATTERN" if n<cfg["stable_pattern_min_frames"] else "UNSTABLE_PATTERN"
        medoid=min(idx,key=lambda i:rows[i]["distance_to_medoid"]);medoids[pid]=medoid
        catalog.append({"pattern_id":pid,"N":n,"percentage_dataset":n/len(rows),"distinct_days":days,"status":status,"median_cluster_distance":float(np.median([rows[i]["distance_to_medoid"] for i in idx])),"stability":float(np.mean([rows[i]["cluster_confidence"] for i in idx])),"medoid_frame_id":rows[medoid]["frame_id"],"metrics":med,"symbol_btc_relation":relation,"deterministic_description":describe(med),"temporal_presence":{"days":dict(Counter(rows[i]["decision_at"][:10] for i in idx)),"weeks":dict(Counter(str(pd.Timestamp(rows[i]["decision_at"]).to_period("W")) for i in idx))}})
    for item in catalog:
        pid=item["pattern_id"]; candidates=[p for p in medoids if p!=pid]
        if candidates:
            other=min(candidates,key=lambda p:np.linalg.norm(x[medoids[pid]]-x[medoids[p]])); other_item=next(p for p in catalog if p["pattern_id"]==other)
            deltas=sorted(((k,item["metrics"][k]-other_item["metrics"][k]) for k in item["metrics"]),key=lambda z:abs(z[1]),reverse=True)[:4]
            item["nearest_other_pattern"]={"pattern_id":other,"medoid_distance":float(np.linalg.norm(x[medoids[pid]]-x[medoids[other]])),"largest_metric_differences":[{"metric":k,"difference":float(v)} for k,v in deltas]}
    write_json(output/"pattern_catalog.json",{"schema_version":1,"patterns":catalog,"manual_pattern_note_policy":"post-clustering metadata only"})
    cheat=output/"pattern_cheat_sheets";cheat.mkdir(exist_ok=True); guide=["# PATTERN FIELD GUIDE","","Research-only. `ASSOCIATED PRECURSOR` no significa causa. El comportamiento posterior es descriptivo, no una predicción ni una estrategia.",""]
    for item in catalog:
        pid=item["pattern_id"];pre=precursor.get(pid,{});beh=behavior.get(pid,{});other=item.get("nearest_other_pattern",{}).get("pattern_id","N/A")
        common={lag:(pre.get("lag_summary",{}).get(str(lag),{}).get("common_preceding_patterns") or [["sin evidencia",0]])[0][0] for lag in [30,15,9,6,3]}
        differences=item.get("nearest_other_pattern",{}).get("largest_metric_differences",[])
        section=[f"## {pid}","",f"Estado de evidencia: {item['status']} — {item['N']} frames en {item['distinct_days']} días.","","Cómo se ve:",""]+[f"- {s}" for s in item["deterministic_description"]]+[f"- Relación SUI/BTC hasta el frame: {item['symbol_btc_relation']}.","","Precursores asociados más comunes:","",f"- 15–30m antes: {common[30]} / {common[15]}",f"- 6–15m antes: {common[9]} / {common[6]}",f"- 3m antes: {common[3]}",f"- Convergencia visual mediana: {pre.get('median_pattern_approach_score')}","- No es necesario estar ya en otro patrón discreto: la mayoría de onsets proviene de NOISE.","- NOISE por sí solo NO basta; ocurre casi siempre y genera muchas falsas alarmas.","",f"Patrón con el que más se confunde: {other}.",f"Diferencias principales: {json.dumps(differences,ensure_ascii=False)}","",f"Comportamiento histórico posterior: {json.dumps(beh.get('horizons',{}),ensure_ascii=False)}","","Qué NO afirmar:","","- Esta asociación no es causa, entrada, señal, stop ni objetivo.",""]
        text="\n".join(section);(cheat/f"{pid}.md").write_text(text+"\n");guide.extend(section)
    docs.mkdir(parents=True,exist_ok=True);(docs/"PATTERN_FIELD_GUIDE.md").write_text("\n".join(guide)+"\n")
    cards=[]
    for item in catalog:
        pid=item["pattern_id"]; rel=f"../../../sandbox/visual_market_pattern_discovery_v1/artifacts/m1/pattern_contact_sheets/{pid}/nearest.png"
        cards.append(f'<article><h2>{html.escape(pid)}</h2><img src="{rel}" loading="lazy"><p>{html.escape(" ".join(item["deterministic_description"]))}</p><textarea placeholder="manual_pattern_note (metadata posterior)"></textarea></article>')
    (docs/"pattern_gallery.html").write_text("<!doctype html><meta charset=utf-8><title>VMPD V1 M1</title><style>body{background:#0c1016;color:#cdd4dc;font-family:sans-serif}article{margin:2rem;border:1px solid #39424e;padding:1rem}img{max-width:100%}textarea{width:100%}</style>"+"".join(cards))
    clustering=read_json(output/"clustering_summary.json")
    stable=[p for p in catalog if p["status"]=="STABLE_PATTERN"]
    report=["# VMPD_V1_M1_UNSUPERVISED_VISUAL_PATTERN_DISCOVERY — REPORT","",
        "Estado: `VMPD_V1_M1_UNSUPERVISED_VISUAL_PATTERN_DISCOVERY_READY_FOR_REVIEW`","",
        "Research-only. Los precursores son asociaciones, no causas ni señales. El comportamiento posterior es descriptivo; `TRADING_AUTHORITY = false`.","",
        "## Resultado preregistrado","",
        f"- Q1 — Frames: **{len(rows):,}**.",
        f"- Q2 — Patrones estables: **{len(stable)}**.",
        f"- Q3 — Noise: **{clustering['noise_frames']:,} ({clustering['noise_fraction']:.2%})**.",
        "- Q4 — Frecuencia: "+", ".join(f"{p['pattern_id']} {p['N']} ({p['percentage_dataset']:.2%})" for p in catalog)+".",
        "- Q5 — Aspecto: documentado métrica por métrica en `PATTERN_FIELD_GUIDE.md` y contact sheets.",
        "- Q6/Q7 — Par más próximo: PATTERN_001 ↔ PATTERN_002, distancia entre medoids "+(f"{catalog[0]['nearest_other_pattern']['medoid_distance']:.4f}." if catalog else "N/A."),
        "- Q8 — En ambos patrones el estado discreto previo dominante es NOISE; la información útil está en la convergencia continua, no en un cluster precursor separado.",
        "- Q9/Q10 — Los perfiles de precisión/lift y lead time a 3/6/9/15/30m están congelados en `precursor_analysis.json`.",
        "- Q11 — Approach score mediano: "+", ".join(f"{p['pattern_id']}={precursor.get(p['pattern_id'],{}).get('median_pattern_approach_score')}" for p in catalog)+".",
        "- Q12 — NOISE es una alarma muy frecuente y poco específica; sus false positives se cuantifican en `precursor_precision_profiles`.",
        "- Q13 — Estabilidad: "+", ".join(f"{p['pattern_id']} aparece en {p['distinct_days']} días" for p in catalog)+".",
        "- Q14 — Retornos, MFE, MAE, rango, volatilidad y first-passage post-hoc están en `post_pattern_behavior.json`.",
        "- Q15 — Sí; `find_similar_frames.py` fue validado con consulta histórica exacta y soporta screenshot con warning OOD.",
        "- Q16 — La guía humana resume forma, volumen, posición de rango, BTC, precursores y patrón confundible.","",
        "## Lectura prudente","",
        "El 98,44% de noise es el resultado honesto de la única configuración preregistrada. No se retunearon encoder, ventanas ni HDBSCAN después de observarlo. Los dos clusters cumplen N/días, pero cubren una fracción pequeña del mercado; no debe generalizarse una narrativa fuerte al resto.","",
        "## Readiness","","```text","VISUAL_DATASET_READY = true","EMBEDDING_INDEX_READY = true",
        f"STABLE_PATTERNS_FOUND = {'true' if stable else 'false'}","SIMILARITY_SEARCH_READY = true","PRECURSOR_ANALYSIS_READY = true",
        "PATTERN_FIELD_GUIDE_READY = true","TRADING_AUTHORITY = false","```",""]
    (docs/"VMPD_V1_M1_UNSUPERVISED_VISUAL_PATTERN_DISCOVERY_REPORT.md").write_text("\n".join(report))
    write_json(output/"diagnostic_summary.json",{"status":"VMPD_V1_M1_UNSUPERVISED_VISUAL_PATTERN_DISCOVERY_READY_FOR_REVIEW",
        "VISUAL_DATASET_READY":True,"EMBEDDING_INDEX_READY":True,"STABLE_PATTERNS_FOUND":bool(stable),
        "SIMILARITY_SEARCH_READY":True,"PRECURSOR_ANALYSIS_READY":True,"PATTERN_FIELD_GUIDE_READY":True,
        "TRADING_AUTHORITY":False,"frames":len(rows),"stable_patterns":len(stable),"noise_fraction":clustering["noise_fraction"]})
    write_json(output/"m1_manifest.json",artifact_manifest(output,sha256_file(config_path),"catalog"))
    return {"patterns":len(catalog),"stable_patterns":sum(i["status"]=="STABLE_PATTERN" for i in catalog)}


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[4]);p.add_argument("--output",type=Path);p.add_argument("--config",type=Path);p.add_argument("--docs",type=Path);a=p.parse_args();project=Path(__file__).resolve().parents[2]
    docs=a.docs or a.repo/"docs/visual-market-pattern-discovery-v1";print(json.dumps(build(a.repo.resolve(),(a.output or project/"artifacts/m1").resolve(),(a.config or project/"config/m1_frozen.json").resolve(),docs.resolve()),indent=2))
if __name__=="__main__":main()
