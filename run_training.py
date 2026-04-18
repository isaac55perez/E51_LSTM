"""Entry point: train the LSTM across all sequence-length values."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", type=int, default=None,
        help="Run a single L value (e.g. --only 7). Omit to run the full sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    L_list = [args.only] if args.only is not None else config.L_VALUES
    summaries = []
    for L in L_list:
        out_dir = config.OUTPUT_DIR / f"L_{L}"
        summary = run_training(L, out_dir, device)
        summaries.append(summary)
        print(
            f"[L={L}] best_epoch={summary['best_epoch']} "
            f"train={summary['train_mse']:.4e} "
            f"test={summary['test_mse']:.4e} "
            f"({summary['runtime_sec']:.1f}s)"
        )
    print("done.")


if __name__ == "__main__":
    main()
