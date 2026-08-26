from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .core import artifact_manifest, l2_normalize, read_json, read_jsonl_gz, sha256_file, write_json


def _weight_hash(model: torch.nn.Module) -> str:
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_encoder(cfg: dict, device: str = "cpu") -> tuple[torch.nn.Module, Callable, dict]:
    """Select exactly one pretrained visual encoder; never silently use random weights."""
    selected = cfg.get("encoder_default", {})
    family = selected.get("family", "open_clip")
    if family == "dinov3":
        repo, weights = selected.get("repo"), selected.get("weights")
        if not repo or not weights or not Path(repo).exists() or not Path(weights).exists():
            raise RuntimeError("Frozen DINOv3 requires existing local repo and approved weights")
        model = torch.hub.load(repo, selected["model"], source="local", weights=weights)
        from torchvision.transforms import v2
        preprocess = v2.Compose([v2.Resize((224,224), antialias=True), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize([.485,.456,.406],[.229,.224,.225])])
    elif family == "dinov2":
        repo, weights = selected.get("repo"), selected.get("weights")
        if not repo or not weights or not Path(repo).exists() or not Path(weights).exists():
            raise RuntimeError("Frozen DINOv2 requires existing local repo and approved weights")
        model = torch.hub.load(repo, selected["model"], source="local", pretrained=False)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        from torchvision.transforms import v2
        preprocess = v2.Compose([v2.Resize((224,224), antialias=True), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize([.485,.456,.406],[.229,.224,.225])])
    elif family == "open_clip":
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("OpenCLIP fallback selected; install optional dependency open-clip-torch") from exc
        model, _, preprocess = open_clip.create_model_and_transforms(selected["model"], pretrained=selected["pretrained"])
    else:
        raise ValueError(f"Unsupported encoder family: {family}")
    model = model.eval().to(device)
    meta = {**selected, "weights_sha256": _weight_hash(model), "torch_version": torch.__version__,
            "device": device, "preprocessing": repr(preprocess)}
    return model, preprocess, meta


def encode_image(model: torch.nn.Module, preprocess: Callable, image: Image.Image, device: str) -> np.ndarray:
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        if hasattr(model, "encode_image"):
            value = model.encode_image(tensor)
        else:
            value = model(tensor)
        if isinstance(value, dict):
            value = value.get("x_norm_clstoken", next(iter(value.values())))
    return value.detach().float().cpu().numpy().reshape(1, -1)


def compute(output: Path, config_path: Path, batch_size: int = 32, device: str = "cpu") -> dict:
    cfg = read_json(config_path); frames = read_jsonl_gz(output / "frames_manifest.jsonl.gz")
    model, preprocess, model_meta = load_encoder(cfg, device)
    batches, result = [], []
    with torch.inference_mode():
        for i, frame in enumerate(frames):
            batches.append(preprocess(Image.open(output / frame["image_path"]).convert("RGB")))
            if len(batches) == batch_size or i + 1 == len(frames):
                tensor = torch.stack(batches).to(device)
                value = model.encode_image(tensor) if hasattr(model, "encode_image") else model(tensor)
                if isinstance(value, dict): value = value.get("x_norm_clstoken", next(iter(value.values())))
                result.append(value.detach().float().cpu().numpy()); batches.clear()
            if (i + 1) % 1000 == 0: print(json.dumps({"embedded": i+1, "total": len(frames)}), flush=True)
    embeddings = l2_normalize(np.concatenate(result).astype(np.float32))
    if not np.isfinite(embeddings).all(): raise ValueError("Embedding contains NaN/Inf")
    np.save(output / "embeddings.npy", embeddings, allow_pickle=False)
    model_meta["embedding_dimension"] = int(embeddings.shape[1]); model_meta["image_count"] = len(frames)
    model_meta["embeddings_sha256"] = sha256_file(output / "embeddings.npy")
    write_json(output / "embedding_index.json", {"schema_version": 1, "model": model_meta,
               "frames": [{"row": i, "frame_id": f["frame_id"], "decision_at": f["decision_at"],
                           "image_path": f["image_path"], "source_hash": f["source_hash"]} for i, f in enumerate(frames)]})
    write_json(output / "m1_manifest.json", artifact_manifest(output, sha256_file(config_path), "embeddings"))
    return model_meta


def main() -> None:
    p=argparse.ArgumentParser(description="Compute frozen pretrained visual embeddings")
    p.add_argument("--output",type=Path); p.add_argument("--config",type=Path); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args(); project=Path(__file__).resolve().parents[2]
    print(json.dumps(compute((a.output or project/"artifacts/m1").resolve(), (a.config or project/"config/m1_frozen.json").resolve(), a.batch_size,a.device),indent=2))

if __name__ == "__main__": main()

