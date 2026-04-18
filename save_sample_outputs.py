"""Generate per-L, per-frequency prediction plots from the saved checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config                                                  # noqa: E402
from src import data as data_mod                               # noqa: E402
from src.model import FrequencyExtractorLSTM                   # noqa: E402
from src.utils import (                                        # noqa: E402
    plot_per_frequency_grid,
    plot_triple_compare,
)


def _reconstruct(
    model: FrequencyExtractorLSTM,
    inputs: np.ndarray,
    L: int,
    device,
    warmup_inputs: np.ndarray | None = None,
):
    """Stream inputs through the model, stateful across windows.

    If `warmup_inputs` is provided, the model is first streamed over that
    prefix so the hidden state is realistic before prediction begins.
    """
    model.eval()
    B, T, _ = inputs.shape
    state = model.init_state(B, device)
    with torch.no_grad():
        if warmup_inputs is not None:
            xw = torch.from_numpy(np.ascontiguousarray(warmup_inputs)).to(device)
            start = 0
            while start < xw.shape[1]:
                end = min(start + L, xw.shape[1])
                _, state = model(xw[:, start:end, :], state)
                start = end
        x = torch.from_numpy(np.ascontiguousarray(inputs)).to(device)
        preds = torch.empty(B, T, device=device)
        start = 0
        while start < T:
            end = min(start + L, T)
            out, state = model(x[:, start:end, :], state)
            preds[:, start:end] = out.squeeze(-1)
            start = end
    return preds.cpu().numpy()


def process_L(L: int, device: torch.device) -> None:
    out_dir = config.OUTPUT_DIR / f"L_{L}"
    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"[L={L}] skip — no checkpoint at {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = FrequencyExtractorLSTM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    ds = data_mod.load_or_build()
    train_ds, test_ds = data_mod.temporal_split(ds)
    preds = _reconstruct(                                           # (4, T)
        model, test_ds["inputs"], L, device,
        warmup_inputs=train_ds["inputs"],
    )
    targets = test_ds["targets"]                                   # (4, T)
    noisy = test_ds["inputs"][0, :, 0]                             # shared S[t]
    t = test_ds["t"]

    window = slice(0, min(config.PRED_PLOT_WINDOW, t.shape[0]))

    for i, f in enumerate(config.FREQS):
        plot_triple_compare(
            t=t[window],
            noisy=noisy[window],
            clean=targets[i, window],
            pred=preds[i, window],
            out_path=out_dir / f"triple_compare_f{f}.png",
            title=f"L={L}  f={f} Hz  (noisy, clean, LSTM)",
        )
        plot_triple_compare(
            t=t[window],
            noisy=noisy[window],
            clean=targets[i, window],
            pred=preds[i, window],
            out_path=out_dir / f"pred_vs_target_f{f}.png",
            title=f"L={L}  prediction vs target  (f={f} Hz)",
        )

    plot_per_frequency_grid(
        t=t[window],
        noisy=noisy[window],
        cleans=targets[:, window],
        preds=preds[:, window],
        freqs=config.FREQS,
        out_path=out_dir / "all_frequencies.png",
        title=f"L={L}  —  all 4 target frequencies",
    )
    print(f"[L={L}] saved plots → {out_dir}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for L in config.L_VALUES:
        process_L(L, device)


if __name__ == "__main__":
    main()
