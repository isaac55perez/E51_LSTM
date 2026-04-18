"""Signal generation and dataset assembly (PDF §2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import config


def _noisy_sinus(f: float, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A_i * sin(2π f t + φ_i) + ε(t) — deviation from a literal read of PDF §2.2.

    Interpretation: A_i ~ U(0.8, 1.2) and φ_i ~ U(0, 2π) are drawn ONCE per
    realization (per frequency), and additive Gaussian noise ε(t) ~ N(0, σ²)
    gives the per-timestep variability the PDF refers to as "noise".  This
    preserves a recoverable periodic structure in S[t] so the LSTM actually
    has something to extract.
    """
    n = t.shape[0]
    A = rng.uniform(config.AMP_RANGE[0], config.AMP_RANGE[1])
    phi = rng.uniform(config.PHASE_RANGE[0], config.PHASE_RANGE[1])
    eps = rng.normal(0.0, config.NOISE_STD, size=n)
    return A * np.sin(2 * np.pi * f * t + phi) + eps


def generate_signals(seed: int = config.SEED):
    """Build the per-selector signals described in the PDF.

    Returns
    -------
    t        : (N,) time axis
    S        : (N,) mixed noisy signal
    targets  : (4, N) clean target sinusoids per frequency
    """
    rng = np.random.default_rng(seed)
    t = np.arange(config.N_SAMPLES, dtype=np.float64) / config.FS
    noisy_per_freq = np.stack(
        [_noisy_sinus(f, t, rng) for f in config.FREQS], axis=0
    )                                               # (4, N)
    S = 0.25 * noisy_per_freq.sum(axis=0)           # (N,)
    targets = np.stack(
        [np.sin(2 * np.pi * f * t) for f in config.FREQS], axis=0
    )                                               # (4, N)
    return t.astype(np.float32), S.astype(np.float32), targets.astype(np.float32)


def build_dataset(seed: int = config.SEED):
    """Tile the PDF's 40 000-row table as 4 selector blocks.

    Returns dict with:
      inputs  : (4, N, 5)  — [S[t], one_hot(4)] per selector block
      targets : (4, N)     — Target_i[t]
      t       : (N,)
    """
    t, S, targets = generate_signals(seed)
    N = S.shape[0]
    eye = np.eye(config.N_SELECTORS, dtype=np.float32)
    inputs = np.empty((config.N_SELECTORS, N, config.INPUT_SIZE), dtype=np.float32)
    for i in range(config.N_SELECTORS):
        inputs[i, :, 0] = S
        inputs[i, :, 1:] = eye[i]                   # broadcast one-hot over time
    return {"inputs": inputs, "targets": targets, "t": t}


def load_or_build(cache_path: Path | None = None, seed: int = config.SEED):
    """Build the dataset or load it from `.npz` cache (idempotent w.r.t. seed)."""
    if cache_path is None:
        cache_path = config.DATA_DIR / f"dataset_seed{seed}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {"inputs": data["inputs"], "targets": data["targets"], "t": data["t"]}
    ds = build_dataset(seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **ds)
    return ds


def temporal_split(ds):
    """Split each selector block 80/20 along time (PDF §2.4 override)."""
    cut = config.TRAIN_STEPS
    train = {
        "inputs":  ds["inputs"][:, :cut, :],
        "targets": ds["targets"][:, :cut],
        "t":       ds["t"][:cut],
    }
    test = {
        "inputs":  ds["inputs"][:, cut:, :],
        "targets": ds["targets"][:, cut:],
        "t":       ds["t"][cut:],
    }
    return train, test
