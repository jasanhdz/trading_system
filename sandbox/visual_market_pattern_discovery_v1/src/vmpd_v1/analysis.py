from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (artifact_manifest, causal_bars, load_candles, read_json,
                   read_jsonl_gz, sha256_file, write_json)
from .dataset import locate_sources


def _at_or_before(rows: list[dict], when: pd.Timestamp) -> dict | None:
    times=np.array([pd.Timestamp(r["decision_at"]).value for r in rows],dtype=np.int64)
    pos=int(np.searchsorted(times,when.value,side="right")-1)
    return rows[pos] if pos >= 0 else None


def _forward(df: pd.DataFrame, t_ms: int, horizon: int, pre_volatility: float | None = None) -> dict:
    now=df.loc[df.close_time_ms<=t_ms,"close"]
    future=df[(df.open_time_ms>t_ms)&(df.close_time_ms<=t_ms+horizon*60_000)]
    if now.empty or future.empty: return {}
    p=float(now.iloc[-1]); close=float(future.close.iloc[-1]); highs=future.high.to_numpy(float); lows=future.low.to_numpy(float)
    volatility=float(np.std(np.diff(np.log(np.r_[p,future.close.to_numpy(float)]))))
    return {"forward_return":close/p-1,"mfe":float(highs.max()/p-1),"mae":float(lows.min()/p-1),
            "future_realized_range":float(highs.max()/lows.min()-1),"volatility":volatility,
            "volatility_expansion":volatility/max(pre_volatility or 0.0,1e-12)}


def _first_passage(df: pd.DataFrame,t_ms:int,up_bps:int,down_bps:int,horizon:int=120)->str:
    now=df.loc[df.close_time_ms<=t_ms,"close"]
    if now.empty:return "UNAVAILABLE"
    p=float(now.iloc[-1]); future=df[(df.open_time_ms>t_ms)&(df.close_time_ms<=t_ms+horizon*60_000)]
    up=p*(1+up_bps/10000); down=p*(1+down_bps/10000)
    for _,bar in future.iterrows():
        hit_up=bar.high>=up; hit_down=bar.low<=down
        if hit_up and hit_down:return "ADVERSE_FIRST"
        if hit_up:return "UP_FIRST"
        if hit_down:return "DOWN_FIRST"
    return "NEITHER"


def _metrics(bars: pd.DataFrame)->dict:
    c=bars.close.to_numpy(float); o=bars.open.to_numpy(float); h=bars.high.to_numpy(float); l=bars.low.to_numpy(float); v=bars.volume.to_numpy(float)
    total=max(float(h.max()-l.min()),1e-12); moves=np.abs(np.diff(c))
    body=np.abs(c-o); candle_range=np.maximum(h-l,1e-12); wick=(h-np.maximum(o,c))+(np.minimum(o,c)-l)
    return {"displacement":float((c[-1]-o[0])/total),"path_efficiency":float(abs(c[-1]-o[0])/max(moves.sum(),1e-12)),
            "volatility":float(np.std(np.diff(np.log(c)))),"volume_ratio":float(v[-10:].mean()/max(np.median(v),1e-12)),
            "relative_range":float(np.mean(candle_range[-10:])/total),"body_fraction":float(np.median(body/candle_range)),
            "wick_body_ratio":float(np.median(wick/np.maximum(body,1e-12))),"range_position":float((c[-1]-l.min())/total)}


def analyze(repo:Path,output:Path,config_path:Path)->dict:
    cfg=read_json(config_path); assignments=read_jsonl_gz(output/"cluster_assignments.jsonl.gz"); episodes=read_jsonl_gz(output/"pattern_episodes.jsonl.gz")
    frozen=read_json(output/"clustering_summary.json")["assignments_sha256"]
    if sha256_file(output/"cluster_assignments.jsonl.gz") != frozen: raise RuntimeError("Frozen cluster assignments integrity failure")
    x=np.load(output/"embeddings.npy",allow_pickle=False); by_time={r["decision_at"]:r for r in assignments}; times=[pd.Timestamp(r["decision_at"]) for r in assignments]
    times_ns=np.array([t.value for t in times],dtype=np.int64)
    row_index={r["frame_id"]:i for i,r in enumerate(assignments)}
    by_pid=defaultdict(list)
    for i,r in enumerate(assignments): by_pid[r["pattern_id"]].append(i)
    transitions={}
    for lag in cfg["precursor_lags_minutes"]:
        counts=Counter(); antecedent=Counter(); target=Counter()
        offset=lag//3
        for pos,r in enumerate(assignments):
            if pos < offset: continue
            prev=assignments[pos-offset]
            a,b=prev["pattern_id"],r["pattern_id"]; counts[(a,b)]+=1; antecedent[a]+=1; target[b]+=1
        total=sum(counts.values()); transitions[str(lag)] = [
            {"from":a,"to":b,"support_count":n,"conditional_probability":n/antecedent[a],"base_probability":target[b]/total,
             "lift":(n/antecedent[a])/(target[b]/total) if target[b] else None}
            for (a,b),n in sorted(counts.items())]
    write_json(output/"pattern_transition_matrix.json",{"lags_minutes":cfg["precursor_lags_minutes"],"transitions":transitions})
    precursor={}; sequences=defaultdict(Counter)
    for ep in episodes:
        target=ep["pattern_id"]; onset=pd.Timestamp(ep["started_at"])
        med_candidates=by_pid[target]
        med=min(med_candidates,key=lambda i: assignments[i].get("distance_to_medoid") or 1e9)
        item={"episode_id":ep["episode_id"],"onset":ep["started_at"],"lags":{}}
        seq=[]
        for lag in sorted(cfg["precursor_lags_minutes"],reverse=True):
            target_ns=(onset-pd.Timedelta(minutes=lag)).value
            pos=int(np.searchsorted(times_ns,target_ns,side="right")-1)
            prev=assignments[pos] if pos>=0 else None
            if prev:
                pi=row_index[prev["frame_id"]]; dist=float(np.linalg.norm(x[pi]-x[med])); item["lags"][str(lag)]={"pattern_id":prev["pattern_id"],"embedding_distance_to_target_medoid":dist}; seq.append(prev["pattern_id"])
        d=[v["embedding_distance_to_target_medoid"] for _,v in sorted(item["lags"].items(),key=lambda z:-int(z[0]))]
        item["pattern_approach_score"]=(float((d[0]-d[-1])/max(d[0],1e-12)) if len(d)>1 else None)
        precursor.setdefault(target,[]).append(item); sequences[target][tuple(seq[-cfg["sequence_max_states"]:])]+=1
    precursor_output={}
    for pid,items in precursor.items():
        eligible=len(items)>=cfg["precursor_min_independent_onsets"]
        lag_summary={}
        for lag in cfg["precursor_lags_minutes"]:
            vals=[i["lags"][str(lag)] for i in items if str(lag) in i["lags"]]
            lag_summary[str(lag)]={"N":len(vals),"median_embedding_distance":float(np.median([v["embedding_distance_to_target_medoid"] for v in vals])) if vals else None,
                                     "common_preceding_patterns":Counter(v["pattern_id"] for v in vals).most_common(5)}
        precision_profiles={}
        for lag in cfg["precursor_lags_minutes"]:
            steps=lag//3; profiles=[]; target_windows=0
            for i in range(len(assignments)-steps):
                if any(assignments[j]["pattern_id"]==pid for j in range(i+1,i+steps+1)): target_windows+=1
            for state in sorted({r["pattern_id"] for r in assignments}):
                positions=[i for i,r in enumerate(assignments[:-steps]) if r["pattern_id"]==state]
                leads=[]
                for i in positions:
                    matches=[j for j in range(1,steps+1) if assignments[i+j]["pattern_id"]==pid]
                    if matches: leads.append(matches[0]*3)
                follows=len(leads); occurrences=len(positions); precision=follows/occurrences if occurrences else 0.0
                base=target_windows/max(len(assignments)-steps,1)
                profiles.append({"precursor_pattern":state,"occurrences_precursor":occurrences,"target_follows":follows,
                    "target_does_not_follow":occurrences-follows,"precision":precision,"recall":follows/max(target_windows,1),
                    "lift":precision/base if base else None,"median_lead_time_minutes":float(np.median(leads)) if leads else None})
            precision_profiles[str(lag)]=profiles
        precursor_output[pid]={"independent_onsets":len(items),"eligible":eligible,"lag_summary":lag_summary,
            "median_pattern_approach_score":float(np.median([i["pattern_approach_score"] for i in items if i["pattern_approach_score"] is not None])) if items else None,
            "precursor_precision_profiles":precision_profiles,
            "frequent_sequences":[{"states":list(k),"support":v} for k,v in sequences[pid].most_common() if v>=cfg["sequence_min_support"]][:20],"occurrences":items}
    write_json(output/"precursor_analysis.json",{"language_guard":"ASSOCIATED PRECURSOR; NOT CAUSE OR TRADING SIGNAL","patterns":precursor_output})
    sources=locate_sources(repo); dfs={s:load_candles(p) for s,p in sources.items()}; behavior=defaultdict(list)
    for ep in episodes:
        t=int(pd.Timestamp(ep["started_at"]).timestamp()*1000); record={"episode_id":ep["episode_id"],"started_at":ep["started_at"],"horizons":{},"first_passage":{}}
        past=dfs["SUIUSDT"].loc[dfs["SUIUSDT"].close_time_ms<=t].tail(31).close.to_numpy(float)
        pre_vol=float(np.std(np.diff(np.log(past)))) if len(past)>1 else 0.0
        pre_direction=float(past[-1]/past[-16]-1) if len(past)>=16 else 0.0
        record["pre_onset_15m_return"]=pre_direction; record["pre_onset_volatility"]=pre_vol
        for h in cfg["forward_horizons_minutes"]:
            record["horizons"][str(h)]={"SUIUSDT":_forward(dfs["SUIUSDT"],t,h,pre_vol),"BTCUSDT":_forward(dfs["BTCUSDT"],t,h)}
        for up,down in cfg["first_passage_bps"]:
            record["first_passage"][f"+{up}_before_{down}"]=_first_passage(dfs["SUIUSDT"],t,up,down)
            record["first_passage"][f"{abs(down)}down_before_{up}up"]=_first_passage(dfs["SUIUSDT"],t,abs(down),-up)
        behavior[ep["pattern_id"]].append(record)
    summaries={}
    for pid,items in behavior.items():
        summaries[pid]={"N":len(items),"horizons":{},"first_passage_rates":{}}
        for h in cfg["forward_horizons_minutes"]:
            values=[i["horizons"][str(h)]["SUIUSDT"] for i in items if i["horizons"][str(h)]["SUIUSDT"]]
            summaries[pid]["horizons"][str(h)]={k:{"mean":float(np.mean([v[k] for v in values])),"median":float(np.median([v[k] for v in values]))} for k in ["forward_return","mfe","mae","future_realized_range","volatility","volatility_expansion"]} if values else {}
            if values:
                pairs=[(i["pre_onset_15m_return"],i["horizons"][str(h)]["SUIUSDT"]["forward_return"]) for i in items if i["horizons"][str(h)]["SUIUSDT"]]
                summaries[pid]["horizons"][str(h)]["continuation_frequency"]=sum(a*b>0 for a,b in pairs)/len(pairs)
                summaries[pid]["horizons"][str(h)]["reversal_frequency"]=sum(a*b<0 for a,b in pairs)/len(pairs)
        keys=items[0]["first_passage"] if items else {}
        for key in keys:summaries[pid]["first_passage_rates"][key]={k:v/len(items) for k,v in Counter(i["first_passage"][key] for i in items).items()}
    write_json(output/"post_pattern_behavior.json",{"label":"POST-HOC HISTORICAL BEHAVIOR; NOT A PREDICTION","patterns":summaries,"occurrences":behavior})
    write_json(output/"m1_manifest.json",artifact_manifest(output,sha256_file(config_path),"precursors_and_posthoc_behavior"))
    return {"patterns":len(precursor_output),"episodes":len(episodes),"eligible_patterns":sum(v["eligible"] for v in precursor_output.values())}


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[4]);p.add_argument("--output",type=Path);p.add_argument("--config",type=Path);a=p.parse_args();project=Path(__file__).resolve().parents[2]
    print(json.dumps(analyze(a.repo.resolve(),(a.output or project/"artifacts/m1").resolve(),(a.config or project/"config/m1_frozen.json").resolve()),indent=2))
if __name__=="__main__":main()
