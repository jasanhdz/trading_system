"""Deterministic compute audit and representative W12 backend benchmark."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import psutil


SANDBOX = Path(__file__).resolve().parents[2]
SEED = 20260826


def _library(name: str) -> dict[str, Any]:
    available = importlib.util.find_spec(name) is not None
    version = None
    if available:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
        except Exception as exc:  # audit must survive optional library import failures
            return {"available": True, "version": None, "import_error": str(exc)}
    return {"available": available, "version": version}


def _rocm_smi() -> str | None:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def compute_environment() -> dict[str, Any]:
    logical = os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    memory = psutil.virtual_memory()
    libraries = {
        name: _library(name)
        for name in ("numpy", "pandas", "pyarrow", "sklearn", "joblib", "psutil", "torch", "xgboost", "lightgbm", "polars")
    }
    gpus: list[dict[str, Any]] = []
    torch_state: dict[str, Any] = {"available": False, "device_count": 0}
    if libraries["torch"]["available"]:
        import torch

        torch_state = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "version": torch.__version__,
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_version": torch.version.cuda,
        }
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "vram_bytes": int(properties.total_memory),
            })
    reserve = 2
    configured_workers = max(1, min(4, physical - reserve))
    return {
        "schema_version": "aegis-w12-compute-environment-v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": {
            "processor": platform.processor(),
            "physical_cores": physical,
            "logical_cores": logical,
        },
        "memory": {
            "total_bytes": int(memory.total),
            "available_bytes_at_audit": int(memory.available),
        },
        "gpus": gpus,
        "rocm_available": bool(_rocm_smi()) and bool(torch_state.get("hip_version")),
        "rocm_smi_detected": bool(_rocm_smi()),
        "pytorch": torch_state,
        "libraries": libraries,
        "parallelism": {
            "workers": configured_workers,
            "threads_per_worker": 1,
            "physical_cores_reserved": reserve,
            "blas_thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            "selected_backend": "CPU_VECTORIZED_PROCESS_POOL",
            "gpu_reason": "Frozen sklearn models have no same-method ROCm backend; GPU is not forced.",
        },
    }


def _label_kernel(task: tuple[int, int, int]) -> tuple[float, float, str]:
    seed, rows, horizon = task
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.00035, size=(rows, horizon))
    path = np.cumprod(1.0 + returns, axis=1)
    favorable = (path.max(axis=1) - 1.0) * 10_000.0
    adverse = (1.0 - path.min(axis=1)) * 10_000.0
    digest = hashlib.sha256(np.column_stack((favorable, adverse)).tobytes()).hexdigest()
    return float(favorable.mean()), float(adverse.mean()), digest


def benchmark(environment: dict[str, Any]) -> dict[str, Any]:
    workers = int(environment["parallelism"]["workers"])
    tasks = [(SEED + index, 25_000, 60) for index in range(8)]
    started = time.perf_counter()
    single = [_label_kernel(task) for task in tasks]
    single_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parallel = list(pool.map(_label_kernel, tasks))
    parallel_seconds = time.perf_counter() - started
    if single != parallel:
        raise RuntimeError("parallel representative workload changed deterministic outputs")

    gpu_result: dict[str, Any] = {
        "attempted": False,
        "eligible_for_pipeline": False,
        "reason": "No identical sklearn model backend is available on ROCm; vector label benchmark cannot authorize a methodology change.",
    }
    if environment["pytorch"].get("available"):
        try:
            import torch

            sample = np.random.default_rng(SEED).normal(0.0, 0.00035, size=(25_000, 60))
            started = time.perf_counter()
            tensor = torch.as_tensor(sample, dtype=torch.float64, device="cuda:0")
            path = torch.cumprod(1.0 + tensor, dim=1)
            torch.cuda.synchronize(0)
            gpu_seconds = time.perf_counter() - started
            gpu_result = {
                "attempted": True,
                "device": 0,
                "workload_seconds": gpu_seconds,
                "eligible_for_pipeline": False,
                "reason": "Micro-kernel is not the complete hash-identical pandas/sklearn pipeline; CPU remains the auditable backend.",
            }
        except Exception as exc:
            gpu_result = {
                "attempted": True,
                "device": 0,
                "eligible_for_pipeline": False,
                "error": str(exc),
                "reason": "ROCm enumerates the devices but the representative kernel failed; GPU is disabled fail-closed.",
            }
    return {
        "schema_version": "aegis-w12-performance-benchmark-v1",
        "seed": SEED,
        "workload": "8 independent synthetic 25000x60 future-path label reductions",
        "single_worker": {"workers": 1, "seconds": single_seconds},
        "cpu_parallel": {
            "workers": workers,
            "seconds": parallel_seconds,
            "speedup": single_seconds / parallel_seconds if parallel_seconds else None,
            "deterministic_match": True,
        },
        "gpu": gpu_result,
        "selected": "CPU_VECTORIZED_PROCESS_POOL",
        "selection_reason": "Same outputs, bounded workers, and direct compatibility with frozen NumPy/pandas/sklearn methods.",
    }


def write_compute_reports(output_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    target = output_dir or SANDBOX / "artifacts"
    target.mkdir(parents=True, exist_ok=True)
    environment = compute_environment()
    performance = benchmark(environment)
    for name, payload in (
        ("compute_environment.json", environment),
        ("performance_benchmark.json", performance),
    ):
        (target / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
    return environment, performance


if __name__ == "__main__":
    write_compute_reports()
