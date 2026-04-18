# Tasks — E51_LSTM

Implementation checklist (phase → deliverable).

## Configuration
- [x] `config.py` — paths, seeds, FS, DURATION, FREQS, L_VALUES, training hyperparams.
- [x] `ensure_output_dirs()` creates `data/`, `output/`, `output/L_<L>/`, `docs/`.

## Data module (`src/data.py`)
- [x] `generate_signals(seed)` — per-frequency noisy sinusoids + mixed S + clean targets.
- [x] `build_dataset(seed)` — tile into 4 selector blocks `(4, N, 5)` + `(4, N)` targets.
- [x] `load_or_build()` caches to `data/dataset_seed1.npz`.
- [x] `temporal_split(ds)` — 80/20 per selector block.

## Model (`src/model.py`)
- [x] `FrequencyExtractorLSTM(input=5, hidden=64, layers=1)` returns `(y, state)`.
- [x] `init_state(batch_size, device)` — zeros tuple `(h, c)`.

## Training (`src/train.py`)
- [x] `_run_pass()` streams L-windows, carries state, detaches between windows.
- [x] `run_training(L, out_dir, device)` — Adam + MSE + early stopping + checkpoint.
- [x] Saves `metrics.json`, `summary.json`, `best_model.pt`, `curves.png`.

## Utilities (`src/utils.py`)
- [x] `plot_curves`, `plot_triple_compare`, `plot_per_frequency_grid`, `plot_L_comparison`, `save_metrics`.

## Entry scripts
- [x] `run_training.py` — argparse `--only L`, full sweep by default.
- [x] `save_sample_outputs.py` — reload checkpoints, re-run stateful forward, save plots.
- [x] `analyze_results.py` — table + `comparison_L_sweep.png` + `docs/RESULTS.md`.

## Documentation
- [x] `README.md`, `PRD.md`, `ARCHITECTURE.md`, `TASKS.md` (this file).
- [x] `requirements.txt`.

## Verification
- [ ] Create venv + install deps.
- [ ] Run full `python run_training.py`.
- [ ] Run `python save_sample_outputs.py` and `python analyze_results.py`.
- [ ] Inspect `docs/RESULTS.md` and per-L plots; confirm LSTM output tracks the clean target.
