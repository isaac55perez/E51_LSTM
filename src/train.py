"""Stateful training loop parameterised by sequence length L."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

import config
from src import data as data_mod
from src.model import FrequencyExtractorLSTM
from src.utils import plot_curves, save_metrics


def _as_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def _window_iter(n_steps: int, L: int):
    """Yield non-overlapping [start, end) windows covering the full range."""
    start = 0
    while start < n_steps:
        end = min(start + L, n_steps)
        yield start, end
        start = end


def _run_pass(
    model: FrequencyExtractorLSTM,
    inputs: torch.Tensor,     # (B=4, T, 5)
    targets: torch.Tensor,    # (B=4, T)
    L: int,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    criterion: nn.Module,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    reset_every_window: bool = False,
) -> tuple[float, np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
    """Stream through time in L-length windows, carrying state.

    Returns (mean MSE, predictions (B, T), final state) so the caller can
    chain passes (e.g. train-block warm-up → test-block evaluation).
    """
    training = optimizer is not None
    model.train(training)
    B, T, _ = inputs.shape
    # Reset at the start of each selector block — unless a state is provided.
    if initial_state is None:
        h, c = model.init_state(batch_size=B, device=device)
    else:
        h, c = initial_state
        h = h.detach()
        c = c.detach()
    preds_full = torch.empty(B, T, device=device)
    total_sq_err = 0.0
    for s, e in _window_iter(T, L):
        x = inputs[:, s:e, :]                 # (B, L, 5)
        y = targets[:, s:e].unsqueeze(-1)     # (B, L, 1)
        if reset_every_window:
            h, c = model.init_state(batch_size=B, device=device)
        if training:
            optimizer.zero_grad()
        out, (h, c) = model(x, (h, c))        # (B, L, 1)
        loss = criterion(out, y)
        if training:
            loss.backward()
            optimizer.step()
        # Truncate BPTT: detach so graph doesn't grow but state persists.
        h = h.detach()
        c = c.detach()
        preds_full[:, s:e] = out.squeeze(-1).detach()
        total_sq_err += loss.item() * (e - s) * B
    mean_mse = total_sq_err / (B * T)
    return mean_mse, preds_full.cpu().numpy(), (h, c)


def run_training(
    L: int,
    out_dir: Path,
    device: torch.device,
    reset_every_window: bool = False,
) -> dict:
    """Train a fresh model for one L value, save artefacts under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = data_mod.load_or_build()
    train_ds, test_ds = data_mod.temporal_split(ds)

    x_train = _as_tensor(train_ds["inputs"], device)
    y_train = _as_tensor(train_ds["targets"], device)
    x_test = _as_tensor(test_ds["inputs"], device)
    y_test = _as_tensor(test_ds["targets"], device)

    model = FrequencyExtractorLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.MSELoss()

    metrics: list[dict] = []
    best_test = float("inf")
    best_epoch = -1
    bad = 0
    t_start = time.time()

    label = f"L={L}{' (reset-each-window)' if reset_every_window else ''}"
    pbar = tqdm(range(1, config.EPOCHS + 1), desc=label)
    for epoch in pbar:
        train_mse, _, end_state = _run_pass(
            model, x_train, y_train, L, device, optimizer, criterion,
            reset_every_window=reset_every_window,
        )
        with torch.no_grad():
            # Warm up on train block so the test pass starts from a realistic
            # state, then evaluate on the test block. If we're ablating state
            # persistence, the warm-up is meaningless — the test pass also
            # resets at every window.
            _, _, warm_state = _run_pass(
                model, x_train, y_train, L, device, None, criterion,
                reset_every_window=reset_every_window,
            )
            test_mse, _, _ = _run_pass(
                model, x_test, y_test, L, device, None, criterion,
                initial_state=None if reset_every_window else warm_state,
                reset_every_window=reset_every_window,
            )
        metrics.append({"epoch": epoch, "train_mse": train_mse, "test_mse": test_mse})
        pbar.set_postfix(train=f"{train_mse:.4e}", test=f"{test_mse:.4e}")

        improved = test_mse < best_test - 1e-6
        if improved:
            best_test = test_mse
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "L": L,
                    "test_mse": test_mse,
                    "train_mse": train_mse,
                },
                out_dir / "best_model.pt",
            )
        else:
            bad += 1
            if bad >= config.PATIENCE:
                break

    runtime = time.time() - t_start
    save_metrics(metrics, out_dir / "metrics.json")
    plot_curves(metrics, out_dir / "curves.png", f"L = {L}")

    summary = {
        "L": L,
        "best_epoch": best_epoch,
        "train_mse": metrics[best_epoch - 1]["train_mse"],
        "test_mse": best_test,
        "epochs_run": len(metrics),
        "runtime_sec": runtime,
        "reset_every_window": reset_every_window,
    }
    (out_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2)
    )
    return summary
