"""Loading the raw C-MAPSS FD001 files and building the training target."""

from pathlib import Path

import numpy as np
import pandas as pd

from rul import config


def read_cmapss(path: Path) -> pd.DataFrame:
    """Read one space-separated C-MAPSS file (no header, 26 columns)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for how to obtain FD001.")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=config.RAW_COLUMNS)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_train(path: Path = config.TRAIN_FILE) -> pd.DataFrame:
    """Training engines, each recorded until failure."""
    return read_cmapss(path)


def load_test(path: Path = config.TEST_FILE) -> pd.DataFrame:
    """Test engines, each truncated before failure."""
    return read_cmapss(path)


def load_test_rul(path: Path = config.RUL_FILE) -> pd.Series:
    """True remaining life of each test engine at its last recorded cycle.

    The file holds one value per line in engine order 1..100.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for how to obtain FD001.")
    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    return pd.Series(values, index=pd.RangeIndex(1, len(values) + 1, name="unit"), name="rul")


def add_targets(df: pd.DataFrame, r_max: int = config.R_MAX) -> pd.DataFrame:
    """Add the linear and the piecewise-linear remaining-life targets.

    linear:    final recorded cycle of the engine minus the current cycle
    piecewise: the linear target capped at ``r_max``
    """
    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul_linear"] = (last_cycle - out["cycle"]).astype(np.int64)
    out["rul_piecewise"] = out["rul_linear"].clip(upper=r_max)
    return out


def engine_histories(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Raw sensor history of each engine as an (n_cycles, 13) array, in cycle order."""
    return {
        int(unit): g.sort_values("cycle")[config.SENSORS].to_numpy(dtype=np.float64)
        for unit, g in df.groupby("unit")
    }
