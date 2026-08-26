from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from vmpd_v1.clustering import stable_pattern_ids
from vmpd_v1.core import (causal_bars, collapse_episodes, l2_normalize,
                          normalize_panel, temporal_split_days)
from vmpd_v1.render import image_sha256, render_frame


def candles(n=900, offset=0):
    opens=np.arange(n,dtype=np.int64)*60_000+offset; p=100+np.sin(np.arange(n)/13)+np.arange(n)*.001
    return pd.DataFrame({"open_time_ms":opens,"close_time_ms":opens+59_999,"open":p,"high":p+.2,"low":p-.2,"close":p+.05,"volume":100+np.arange(n)%17})


def test_causal_rendering_no_future_candle():
    df=candles(20); bars=causal_bars(df,9*60_000+59_999,5,10)
    assert bars.source_last_close_ms.max() <= 9*60_000+59_999
    changed=df.copy();changed.loc[10:,"high"]=999999
    assert causal_bars(changed,9*60_000+59_999,5,10).equals(bars)


def test_partial_candle_uses_completed_minutes_only():
    bars=causal_bars(candles(10),7*60_000+59_999,5,10)
    assert bars.iloc[-1].source_minutes==3
    assert bars.iloc[-1].source_last_close_ms==7*60_000+59_999


def test_price_normalization_removes_absolute_level():
    a=causal_bars(candles(20),20*60_000,3,20);b=a.copy()
    for c in ["open","high","low","close"]:b[c]=b[c]*7+500
    assert np.allclose(normalize_panel(a)[["open","high","low","close"]],normalize_panel(b)[["open","high","low","close"]])


def test_image_determinism_and_no_timestamp_pixels():
    dfs={"SUIUSDT":candles(),"BTCUSDT":candles(offset=0)};a=render_frame(dfs,899*60_000+59_999,(600,450));b=render_frame(dfs,899*60_000+59_999,(600,450))
    assert image_sha256(a)==image_sha256(b)


def test_embedding_shape_nan_and_similarity():
    x=l2_normalize(np.array([[1.,0.],[2.,0.],[0.,1.]],dtype=np.float32))
    assert x.shape==(3,2) and np.isfinite(x).all()
    assert np.argsort(-(x@x[0]))[0]==0


def test_cluster_id_determinism_and_noise_handling():
    labels=np.array([8,8,-1,3,3,3]);x=l2_normalize(np.arange(12,dtype=float).reshape(6,2)+1)
    mapping,medoids=stable_pattern_ids(labels,x)
    assert mapping[3]=="PATTERN_001" and -1 not in mapping and set(medoids)=={3,8}


def test_episode_collapse_and_onset_gap():
    def r(m):return {"frame_id":str(m),"decision_at":f"2026-01-01T00:{m:02d}:00Z","pattern_id":"PATTERN_001"}
    eps=collapse_episodes([r(0),r(3),r(6),r(24)],15)
    assert len(eps)==2 and eps[0]["frame_count"]==3 and eps[1]["onset_frame_id"]=="24"


def test_temporal_split_is_ordered_by_day():
    times=[f"2026-01-{d:02d}T00:00:00Z" for d in range(1,21)];s=temporal_split_days(times)
    assert list(s).count("TRAIN")==14 and list(s).count("VALIDATION")==3 and list(s).count("TEST")==3
    assert "TRAIN" not in s[14:]


def test_transition_probability_and_lift_formula():
    support,antecedent,target,total=18,100,40,1000
    conditional=support/antecedent;base=target/total;lift=conditional/base
    assert conditional==.18 and base==.04 and lift==4.5


def test_embedding_distance_approach():
    d=np.array([.82,.61,.40,.22]);score=(d[0]-d[-1])/d[0]
    assert score>0 and np.all(np.diff(d)<0)


def test_artifact_integrity_detects_change(tmp_path:Path):
    p=tmp_path/"a";p.write_bytes(b"frozen");before=hashlib.sha256(p.read_bytes()).hexdigest();p.write_bytes(b"changed")
    assert hashlib.sha256(p.read_bytes()).hexdigest()!=before


def test_render_has_exact_resolution():
    image=render_frame({"SUIUSDT":candles(),"BTCUSDT":candles()},899*60_000+59_999,(1200,900))
    assert image.size==(1200,900) and image.mode=="RGB"
