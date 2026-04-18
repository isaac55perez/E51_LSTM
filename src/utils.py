"""Plotting and JSON helpers shared across scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def save_metrics(metrics: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def load_metrics(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def plot_curves(metrics: list[dict], out_path: Path, title: str) -> None:
    """Train/test MSE per epoch (log-y)."""
    epochs = [m["epoch"] for m in metrics]
    tr = [m["train_mse"] for m in metrics]
    te = [m["test_mse"] for m in metrics]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, tr, label="train MSE", marker="o", ms=3)
    ax.plot(epochs, te, label="test MSE", marker="s", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_triple_compare(
    t: np.ndarray,
    noisy: np.ndarray,
    clean: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """PDF §5.2 graph 1: noisy + clean + LSTM output in one panel."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, noisy, color="0.65", lw=0.6, label="noisy S")
    ax.plot(t, clean, color="tab:green", lw=1.6, label="clean target")
    ax.plot(t, pred, color="tab:red", lw=1.1, label="LSTM output")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_frequency_grid(
    t: np.ndarray,
    noisy: np.ndarray,
    cleans: np.ndarray,
    preds: np.ndarray,
    freqs: Iterable[int],
    out_path: Path,
    title: str,
) -> None:
    """PDF §5.2 graph 2: 4 sub-plots, one per frequency."""
    freqs = list(freqs)
    fig, axes = plt.subplots(len(freqs), 1, figsize=(9, 2.2 * len(freqs)), sharex=True)
    for ax, f, clean, pred in zip(axes, freqs, cleans, preds):
        ax.plot(t, noisy, color="0.7", lw=0.5, label="noisy S")
        ax.plot(t, clean, color="tab:green", lw=1.4, label=f"target f={f} Hz")
        ax.plot(t, pred, color="tab:red", lw=1.0, label="LSTM")
        ax.set_ylabel(f"{f} Hz")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_L_comparison(rows: list[dict], out_path: Path) -> None:
    """MSE (train & test) as a function of sequence length L."""
    rows = sorted(rows, key=lambda r: r["L"])
    Ls = [r["L"] for r in rows]
    train = [r["train_mse"] for r in rows]
    test = [r["test_mse"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ls, train, marker="o", label="train MSE")
    ax.plot(Ls, test, marker="s", label="test MSE")
    ax.set_xlabel("sequence length L")
    ax.set_ylabel("MSE (best epoch)")
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.set_title("MSE vs sequence length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
