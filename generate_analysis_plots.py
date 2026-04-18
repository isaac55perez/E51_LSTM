"""Generate the extra analysis plots referenced from the README:

- per_frequency_mse.png         grouped bar chart of test MSE per (L, f_i)
- reset_ablation.png            stateful vs reset-every-window bars at L=1 and L=15
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config                                                   # noqa: E402


def plot_per_frequency(per_f_path: Path, out_path: Path) -> None:
    data = json.loads(per_f_path.read_text())
    Ls = sorted(int(k) for k in data.keys())
    freqs = config.FREQS
    width = 0.18
    x = np.arange(len(Ls))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for j, f in enumerate(freqs):
        vals = [data[str(L)][str(f)] for L in Ls]
        ax.bar(x + (j - 1.5) * width, vals, width=width,
               label=f"f = {f} Hz", color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels([f"L = {L}" for L in Ls])
    ax.set_ylabel("test MSE")
    ax.set_title("Per-frequency test MSE across sequence lengths")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.5, color="k", linestyle="--", lw=0.8,
               label="Var(sin) = 0.5 (mean predictor)")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {out_path}")


def _summary(L: int, nostate: bool) -> dict:
    folder = f"L_{L}_nostate" if nostate else f"L_{L}"
    p = config.OUTPUT_DIR / folder / "summary.json"
    return json.loads(p.read_text())


def plot_reset_ablation(out_path: Path) -> None:
    Ls = [1, 15]
    stateful_test = [_summary(L, False)["test_mse"] for L in Ls]
    nostate_test = [_summary(L, True)["test_mse"] for L in Ls]
    stateful_train = [_summary(L, False)["train_mse"] for L in Ls]
    nostate_train = [_summary(L, True)["train_mse"] for L in Ls]

    x = np.arange(len(Ls))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - 1.5 * width, stateful_train, width,
           label="stateful — train", color="tab:blue", alpha=0.55)
    ax.bar(x - 0.5 * width, stateful_test, width,
           label="stateful — test", color="tab:blue")
    ax.bar(x + 0.5 * width, nostate_train, width,
           label="reset/window — train", color="tab:red", alpha=0.55)
    ax.bar(x + 1.5 * width, nostate_test, width,
           label="reset/window — test", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L = {L}" for L in Ls])
    ax.set_ylabel("MSE (best epoch)")
    ax.set_title("Effect of resetting (h, c) at every window boundary")
    ax.axhline(0.5, color="k", linestyle="--", lw=0.8, label="Var(sin)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    plot_per_frequency(
        config.OUTPUT_DIR / "per_frequency_mse.json",
        config.OUTPUT_DIR / "per_frequency_mse.png",
    )
    plot_reset_ablation(config.OUTPUT_DIR / "reset_ablation.png")


if __name__ == "__main__":
    main()
