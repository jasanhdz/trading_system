#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = ("conservative", "edge", "ultra")


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--target-idle-pct", type=float, default=0.84)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sampler-power", type=float, default=0.65)
    parser.add_argument("--output-dir", default="aegis_alpha/logs/coliseum")
    parser.add_argument("--window-steps", type=int, default=4032)
    parser.add_argument("--recent-windows", type=int, default=4)
    parser.add_argument("--random-windows", type=int, default=4)
    parser.add_argument("--regime-windows-per-regime", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4667)
    args = parser.parse_args()

    dataset_paths: dict[str, Path] = {}
    model_paths: dict[str, Path] = {}
    for variant in VARIANTS:
        dataset_path = Path(f"aegis_alpha/data/processed/bc_{variant}_dataset.npz")
        model_path = Path(f"aegis_alpha/models/bc/aegis_bc_{variant}.zip")
        dataset_paths[variant] = dataset_path
        model_paths[variant] = model_path

        build_cmd = [
            sys.executable,
            "-m",
            "aegis_alpha.tools.build_bc_dataset",
            "--config",
            args.config,
            "--variant",
            variant,
            "--output",
            str(dataset_path),
            "--target-idle-pct",
            str(args.target_idle_pct),
        ]
        if args.max_samples is not None:
            build_cmd.extend(["--max-samples", str(args.max_samples)])
        _run(build_cmd)

        train_cmd = [
            sys.executable,
            "-m",
            "aegis_alpha.bc.train_bc",
            "--dataset",
            str(dataset_path),
            "--output",
            str(model_path),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--sampler-power",
            str(args.sampler_power),
        ]
        if args.max_samples is not None:
            train_cmd.extend(["--max-samples", str(args.max_samples)])
        if args.device:
            train_cmd.extend(["--device", args.device])
        _run(train_cmd)

    eval_cmd = [
        sys.executable,
        "-m",
        "aegis_alpha.tools.evaluate_bc_walkforward",
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
        "--window-steps",
        str(args.window_steps),
        "--recent-windows",
        str(args.recent_windows),
        "--random-windows",
        str(args.random_windows),
        "--regime-windows-per-regime",
        str(args.regime_windows_per_regime),
        "--seed",
        str(args.seed),
        "--models",
    ]
    eval_cmd.extend(f"{variant}={model_paths[variant]}" for variant in VARIANTS)
    _run(eval_cmd)

    print("Datasets:")
    for variant, path in dataset_paths.items():
        print(f"  {variant}: {path}")
    print("Models:")
    for variant, path in model_paths.items():
        print(f"  {variant}: {path}")


if __name__ == "__main__":
    main()
