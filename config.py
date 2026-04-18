"""Project-wide configuration for the LSTM frequency extractor.

Constants-and-helpers style (matches the E43_AE_Denoise convention).
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"

# Signal parameters (PDF §2.1)
FS = 1000                     # sampling rate (Hz)
DURATION = 10                 # seconds
N_SAMPLES = FS * DURATION     # 10_000 samples per selector
FREQS = [1, 3, 5, 7]          # Hz
N_SELECTORS = len(FREQS)
AMP_RANGE = (0.8, 1.2)        # Uniform A_i (drawn once per realization)
PHASE_RANGE = (0.0, 6.283185307179586)  # Uniform φ_i ∈ [0, 2π) (once per realization)
NOISE_STD = 0.1               # additive Gaussian noise σ per timestep

# Dataset composition
SEED = 1
TRAIN_FRACTION = 0.8          # user override: 80/20 temporal split per selector
TRAIN_STEPS = int(N_SAMPLES * TRAIN_FRACTION)   # 8_000
TEST_STEPS = N_SAMPLES - TRAIN_STEPS            # 2_000

# Model
INPUT_SIZE = 1 + N_SELECTORS  # S[t] + one-hot(4) = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 1
OUTPUT_SIZE = 1

# Training
L_VALUES = [1, 3, 7, 15, 30]
EPOCHS = 60
LR = 1e-3
PATIENCE = 8                  # early-stopping on test MSE

# Plot settings
PRED_PLOT_WINDOW = 500        # timesteps drawn in per-frequency plot


def ensure_output_dirs() -> None:
    """Create output/data/docs folders (idempotent)."""
    for path in (DATA_DIR, OUTPUT_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for L in L_VALUES:
        (OUTPUT_DIR / f"L_{L}").mkdir(parents=True, exist_ok=True)
