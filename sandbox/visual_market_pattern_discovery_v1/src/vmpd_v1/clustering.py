from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import HDBSCAN

from .core import (artifact_manifest, collapse_episodes, read_json, sha256_file,
                   write_json, write_jsonl_gz)


def stable_pattern_ids(labels: np.ndarray, embeddings: np.ndarray) -> tuple[dict[int,str], dict[int,int]]:
    """Assign IDs deterministically by descending cluster size, then medoid row."""
    info=[]; medoids={}
    for label in sorted(set(labels) - {-1}):
        idx=np.flatnonzero(labels==label); center=embeddings[idx].mean(axis=0)
        medoid=int(idx[np.argmin(np.linalg.norm(embeddings[idx]-center,axis=1))]); medoids[int(label)]=medoid
        info.append((-len(idx),medoid,int(label)))
    mapping={label:f"PATTERN_{rank:03d}" for rank,(_,_,label) in enumerate(sorted(info),1)}
    return mapping,medoids


def make_sheet(output: Path, pid: str, selected: list[int], frames: list[dict], title: str) -> None:
    thumb=(300,225); cols=5; rows=max(1,int(np.ceil(len(selected)/cols)))
    sheet=Image.new("RGB",(cols*thumb[0],rows*(thumb[1]+26)),(12,16,22)); draw=ImageDraw.Draw(sheet)
    for slot,idx in enumerate(selected):
        image=Image.open(output/frames[idx]["image_path"]).convert("RGB").resize(thumb)
        x=(slot%cols)*thumb[0]; y=(slot//cols)*(thumb[1]+26); sheet.paste(image,(x,y))
        draw.text((x+5,y+thumb[1]+4),frames[idx]["frame_id"],fill=(205,212,220))
    target=output/"pattern_contact_sheets"/pid; target.mkdir(parents=True,exist_ok=True)
    sheet.save(target/f"{title}.png",format="PNG")


def discover(output: Path, config_path: Path) -> dict:
    cfg=read_json(config_path); index=read_json(output/"embedding_index.json"); frames=index["frames"]
    x=np.load(output/"embeddings.npy",allow_pickle=False)
    ccfg=cfg["clustering"]
    model=HDBSCAN(min_cluster_size=ccfg["min_cluster_size"],min_samples=ccfg["min_samples"],metric="euclidean",cluster_selection_method=ccfg["cluster_selection_method"],store_centers="medoid",copy=True)
    labels=model.fit_predict(x); probs=getattr(model,"probabilities_",np.ones(len(x)))
    mapping,medoids=stable_pattern_ids(labels,x)
    rows=[]
    for i,(label,prob) in enumerate(zip(labels,probs)):
        pid="NOISE" if label == -1 else mapping[int(label)]
        medoid=None if label == -1 else medoids[int(label)]
        dist=None if medoid is None else float(np.linalg.norm(x[i]-x[medoid]))
        rows.append({**frames[i],"pattern_id":pid,"raw_cluster_label":int(label),"cluster_confidence":float(prob),"distance_to_medoid":dist})
    write_jsonl_gz(output/"cluster_assignments.jsonl.gz",rows)
    episodes=collapse_episodes(rows,cfg["episode_gap_minutes"]); write_jsonl_gz(output/"pattern_episodes.jsonl.gz",episodes)
    sheet_root=output/"pattern_contact_sheets"; sheet_root.mkdir(exist_ok=True)
    for label,pid in mapping.items():
        idx=np.flatnonzero(labels==label); med=medoids[label]; d=np.linalg.norm(x[idx]-x[med],axis=1)
        near=list(idx[np.argsort(d)[:10]])
        boundary=list(idx[np.argsort(probs[idx])[:10]])
        diverse=[int(idx[np.argmax(np.min(np.linalg.norm(x[idx,None,:]-x[np.array(chosen)][None,:,:],axis=2),axis=1))]) for chosen in [[med]]]
        chosen=[med]
        while len(chosen)<min(10,len(idx)):
            distances=np.min(np.linalg.norm(x[idx,None,:]-x[np.array(chosen)][None,:,:],axis=2),axis=1)
            chosen.append(int(idx[np.argmax(distances)]))
        make_sheet(output,pid,[med],frames,"medoid"); make_sheet(output,pid,near,frames,"nearest")
        make_sheet(output,pid,chosen,frames,"diverse"); make_sheet(output,pid,boundary,frames,"boundary")
    coords_method="UMAP"
    try:
        import umap
        coords=umap.UMAP(n_neighbors=cfg["umap"]["n_neighbors"],min_dist=cfg["umap"]["min_dist"],metric=cfg["umap"]["metric"],random_state=cfg["umap"]["seed"],transform_seed=cfg["umap"]["seed"]).fit_transform(x)
    except ImportError:
        from sklearn.decomposition import PCA
        coords=PCA(n_components=2,random_state=cfg["seed"]).fit_transform(x); coords_method="PCA_FALLBACK_UMAP_NOT_INSTALLED"
    np.save(output/"umap_coordinates.npy",coords.astype(np.float32),allow_pickle=False)
    sim=output/"similarity_index";sim.mkdir(exist_ok=True)
    write_json(sim/"index.json",{"type":"exact_cosine_on_l2_normalized_numpy","embeddings":"../embeddings.npy","metadata":"../embedding_index.json","embedding_sha256":sha256_file(output/"embeddings.npy")})
    result={"frames":len(rows),"clusters":len(mapping),"noise_frames":int((labels==-1).sum()),"noise_fraction":float((labels==-1).mean()),"pattern_mapping":mapping,"visualization_method":coords_method,"assignments_sha256":sha256_file(output/"cluster_assignments.jsonl.gz")}
    write_json(output/"clustering_summary.json",result); write_json(output/"m1_manifest.json",artifact_manifest(output,sha256_file(config_path),"patterns_frozen"))
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path); p.add_argument("--config",type=Path); a=p.parse_args(); project=Path(__file__).resolve().parents[2]
    print(json.dumps(discover((a.output or project/"artifacts/m1").resolve(),(a.config or project/"config/m1_frozen.json").resolve()),indent=2))
if __name__=="__main__": main()
