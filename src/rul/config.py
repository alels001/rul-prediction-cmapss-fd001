"""Fixed constants of the FD001 pipeline.

Every value here is a decision documented in the thesis. Changing one changes
the model, so they are defined once and imported everywhere else.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"

TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

# ── Raw file layout (26 space-separated columns, no header) ─────────────────
RAW_COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# ── Retained sensors, in the fixed order the features are built ─────────────
# 10 constant / near-constant variables removed; s14 removed as redundant
# with s9 (Pearson r = 0.963). Order matters: it defines the feature layout.
SENSORS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s15",
    "s17",
    "s20",
    "s21",
]

# ── Feature representation ──────────────────────────────────────────────────
STATISTICS = ["mean", "std", "slope", "last", "delta"]
FEATURE_NAMES = [f"{s}_{stat}" for s in SENSORS for stat in STATISTICS]  # 65

# ── Target and output constraint ────────────────────────────────────────────
R_MAX = 125  # piecewise-linear cap; also the upper output bound
MAX_HISTORY = 362  # longest training trajectory, upper bound on input length
