"""Ablation: compare stateful vs reset-every-window training at one L.

Isolates the effect of hidden-state persistence from BPTT window length.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config                           # noqa: E402
from src.train import run_training      # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=15,
                   help="Sequence length to ablate (default: 15, our best L).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = config.OUTPUT_DIR / f"L_{args.L}_nostate"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_training(args.L, out_dir, device, reset_every_window=True)
    print(
        f"[L={args.L} reset-every-window] "
        f"best_epoch={summary['best_epoch']} "
        f"train={summary['train_mse']:.4e} "
        f"test={summary['test_mse']:.4e} "
        f"({summary['runtime_sec']:.1f}s)"
    )


if __name__ == "__main__":
    main()
