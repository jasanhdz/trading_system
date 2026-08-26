from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import read_json, read_jsonl_gz, write_json
from .embeddings import encode_image, load_encoder
from .render import layout_distance


def query(output:Path,config_path:Path,frame_id:str|None,image_path:Path|None,top_k:int,device:str)->dict:
    cfg=read_json(config_path); index=read_json(output/"embedding_index.json"); assignments=read_jsonl_gz(output/"cluster_assignments.jsonl.gz")
    x=np.load(output/"embeddings.npy",allow_pickle=False); frame_rows=index["frames"]
    warning=None
    if frame_id:
        lookup={r["frame_id"]:i for i,r in enumerate(frame_rows)}
        if frame_id not in lookup:raise KeyError(f"Unknown frame_id: {frame_id}")
        q=x[lookup[frame_id]:lookup[frame_id]+1]
    else:
        if image_path is None:raise ValueError("Provide --frame-id or --image")
        external=Image.open(image_path).convert("RGB"); canonical=Image.open(output/frame_rows[0]["image_path"]).convert("RGB")
        aspect_delta=abs(external.width/external.height-canonical.width/canonical.height)/(canonical.width/canonical.height)
        ld=layout_distance(external.resize(canonical.size),canonical)
        if aspect_delta>.02 or ld>.75:warning="OUT_OF_DISTRIBUTION_LAYOUT_WARNING"
        model,preprocess,meta=load_encoder(cfg,device);q=encode_image(model,preprocess,external,device);q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
        if meta["weights_sha256"]!=index["model"]["weights_sha256"]:raise RuntimeError("Query encoder weights do not match frozen embedding index")
    similarity=(x@q[0]).astype(float); order=np.argsort(-similarity)
    if frame_id:order=order[similarity[order]<.999999][:top_k]
    else:order=order[:top_k]
    neighbors=[{"rank":rank,"frame_id":frame_rows[i]["frame_id"],"timestamp":frame_rows[i]["decision_at"],"pattern_id":assignments[i]["pattern_id"],"similarity_score":float(similarity[i]),"image_path":frame_rows[i]["image_path"]} for rank,i in enumerate(order,1)]
    closest=neighbors[0]["pattern_id"] if neighbors else None
    catalog=read_json(output/"pattern_catalog.json") if (output/"pattern_catalog.json").exists() else {"patterns":[]}
    item=next((p for p in catalog["patterns"] if p["pattern_id"]==closest),{})
    precursor=read_json(output/"precursor_analysis.json").get("patterns",{}).get(closest,{}) if (output/"precursor_analysis.json").exists() else {}
    behavior=read_json(output/"post_pattern_behavior.json").get("patterns",{}).get(closest,{}) if (output/"post_pattern_behavior.json").exists() else {}
    return {"warning":warning,"closest_visual_pattern":closest,"similarity_confidence":neighbors[0]["similarity_score"] if neighbors else None,
            "nearest_historical_examples":neighbors,"common_visual_traits":item.get("deterministic_description",[]),"typical_precursors":precursor.get("lag_summary",{}),
            "typical_subsequent_behavior":{"label":"POST-HOC HISTORICAL BEHAVIOR",**behavior},"what_would_make_this_not_the_same_pattern":f"Traits closer to {item.get('nearest_other_pattern',{}).get('pattern_id','another pattern')} or low similarity."}


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument("--frame-id");g.add_argument("--image",type=Path);p.add_argument("--top-k",type=int,default=50);p.add_argument("--output",type=Path);p.add_argument("--config",type=Path);p.add_argument("--device",default="cpu");p.add_argument("--report",type=Path);a=p.parse_args();project=Path(__file__).resolve().parents[2]
    result=query((a.output or project/"artifacts/m1").resolve(),(a.config or project/"config/m1_frozen.json").resolve(),a.frame_id,a.image,a.top_k,a.device)
    if a.report:write_json(a.report,result)
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()

