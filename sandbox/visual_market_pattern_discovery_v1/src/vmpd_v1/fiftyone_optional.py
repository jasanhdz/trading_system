from __future__ import annotations

from pathlib import Path

from .core import read_json, read_jsonl_gz


def create_dataset(output:Path,name:str="VMPD_V1_M1"):
    try:import fiftyone as fo
    except ImportError as exc:raise RuntimeError("FiftyOne is optional: pip install '.[fiftyone]'") from exc
    index=read_json(output/"embedding_index.json");assignments=read_jsonl_gz(output/"cluster_assignments.jsonl.gz")
    if name in fo.list_datasets():fo.delete_dataset(name)
    dataset=fo.Dataset(name,persistent=True)
    samples=[]
    for meta,a in zip(index["frames"],assignments):
        sample=fo.Sample(filepath=str((output/meta["image_path"]).resolve()),frame_id=meta["frame_id"],decision_at=meta["decision_at"],pattern_id=a["pattern_id"],episode_id=None,raw_image=str((output/meta["image_path"]).resolve()),structural_image=None,manual_note="",cluster_confidence=a["cluster_confidence"])
        sample["embedding"]=None;samples.append(sample)
    dataset.add_samples(samples);dataset.set_values("embedding",__import__("numpy").load(output/"embeddings.npy").tolist());return dataset

