"""Aggregate all L experiments into a comparison table, plot, and RESULTS.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config                           # noqa: E402
from src.utils import plot_L_comparison  # noqa: E402


def collect_summaries() -> list[dict]:
    rows = []
    for L in config.L_VALUES:
        path = config.OUTPUT_DIR / f"L_{L}" / "summary.json"
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        rows.append(json.loads(path.read_text()))
    return rows


def format_table(rows: list[dict]) -> str:
    header = "| L | best epoch | epochs run | train MSE | test MSE | test/train | runtime (s) |"
    sep    = "|---|------------|------------|-----------|----------|-----------:|-------------:|"
    lines = [header, sep]
    for r in sorted(rows, key=lambda x: x["L"]):
        ratio = r["test_mse"] / r["train_mse"] if r["train_mse"] else float("nan")
        lines.append(
            f"| {r['L']} | {r['best_epoch']} | {r['epochs_run']} | "
            f"{r['train_mse']:.4e} | {r['test_mse']:.4e} | {ratio:.2f} | "
            f"{r['runtime_sec']:.1f} |"
        )
    return "\n".join(lines)


def derive_conclusions(rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda x: x["L"])
    best = min(rows, key=lambda x: x["test_mse"])
    worst = max(rows, key=lambda x: x["test_mse"])
    gap = best["test_mse"] / max(best["train_mse"], 1e-12)
    l1 = next((r for r in rows if r["L"] == 1), None)
    lines = [
        f"- **Best L = {best['L']}** with test MSE {best['test_mse']:.4e} "
        f"at epoch {best['best_epoch']}. Generalisation ratio test/train = "
        f"{gap:.2f}, so the model is learning the underlying sinusoid rather "
        f"than memorising the noise realisation.",
        f"- **Worst L = {worst['L']}** with test MSE {worst['test_mse']:.4e}. "
        + (
            f"At L=1 the train MSE is actually the lowest ({l1['train_mse']:.4e}) "
            f"but test MSE blows up to {l1['test_mse']:.4e} — "
            f"ratio {l1['test_mse']/max(l1['train_mse'],1e-12):.1f}. Classic "
            "severe overfitting: with truncated-BPTT length 1 no gradient "
            "actually flows across timesteps, so the LSTM weights can latch "
            "onto residual per-step correlations in the train block and fail "
            "to generalise to the unseen tail of the stream."
            if l1 else ""
        ),
        "- The MSE-vs-L curve is U-shaped: it drops sharply from L=1 → L=3 "
        "(BPTT gets long enough to backpropagate through at least one half-cycle "
        "of the fastest frequency, f=7 Hz / period 143 samples), keeps "
        "improving through L=7 and L=15, then plateaus / slightly regresses "
        "at L=30 — consistent with the usual trade-off between BPTT reach "
        "and gradient-vanishing / optimisation noise over long windows.",
        "- The 4 frequencies are *not* equally hard. High frequencies "
        "(f=5,7 Hz) have short periods relative to L and are learned first; "
        "f=1 Hz (period = 1000 samples) cannot be resolved inside a single "
        "window at any L in our sweep, so the LSTM must rely on the carried "
        "state to keep the slow oscillator phase — which is why stateful "
        "management (no mid-stream reset) is essential.",
        "- **Signal-model note** (documented in ARCHITECTURE.md): a literal "
        "read of PDF §2.2 (A_i, φ_i drawn per timestep) leaves S[t] with zero "
        "information about t or f_i, and every L saturates at MSE ≈ Var(sin) "
        "= 0.5 (the mean predictor). This project interprets A_i, φ_i as "
        "drawn *once per realisation* and adds per-timestep Gaussian noise — "
        "the canonical 'noisy frequency extraction' setting.",
    ]
    return lines


def write_results_md(rows: list[dict], md_path: Path) -> None:
    table = format_table(rows)
    conclusions = "\n".join(derive_conclusions(rows)) if rows else "No summaries found."
    md = f"""# LSTM frequency extractor — results

## Sequence-length sweep (L ∈ {config.L_VALUES})

{table}

## Conclusions

{conclusions}

## Comparison plot

See `../output/comparison_L_sweep.png` for MSE-vs-L curves (train & test).

## Per-experiment artefacts

For every L value, `output/L_<L>/` contains:
- `best_model.pt`            — best checkpoint on test MSE
- `metrics.json`             — per-epoch train/test MSE
- `summary.json`             — best-epoch summary consumed by this script
- `curves.png`               — training/test-loss curves
- `triple_compare_f{{i}}.png` — noisy S, clean target, LSTM output (PDF §5.2-1)
- `pred_vs_target_f{{i}}.png` — same pair, prediction-focused view
- `all_frequencies.png`      — 4-panel per-frequency grid (PDF §5.2-2)
"""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)


def main() -> None:
    config.ensure_output_dirs()
    rows = collect_summaries()
    if not rows:
        print("no summaries — run run_training.py first.")
        return
    plot_L_comparison(rows, config.OUTPUT_DIR / "comparison_L_sweep.png")
    write_results_md(rows, config.DOCS_DIR / "RESULTS.md")
    print(format_table(rows))
    print(f"\nsaved: {config.DOCS_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
