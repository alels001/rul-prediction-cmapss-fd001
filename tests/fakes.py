"""Synthetic C-MAPSS data for tests.

The tests never use the NASA files: they are not in the repository and CI has
no access to them. Instead, a small synthetic fleet is generated with the same
column layout, so every code path can be exercised in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rul import config


def make_fleet(n_engines: int = 6, seed: int = 0, min_life: int = 30, max_life: int = 60):
    """Synthetic C-MAPSS-shaped data: every engine drifts towards failure with noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_engines + 1):
        life = int(rng.integers(min_life, max_life + 1))
        base = rng.normal(0, 1, size=21)
        drift = rng.normal(0, 0.05, size=21)
        for cycle in range(1, life + 1):
            sensors = 100 + base + drift * cycle + rng.normal(0, 0.1, size=21)
            rows.append([unit, cycle, 0.0, 0.0, 100.0, *sensors])
    return pd.DataFrame(rows, columns=config.RAW_COLUMNS)


def write_cmapss(df: pd.DataFrame, path) -> None:
    """Write a dataframe in the raw space-separated, header-less C-MAPSS format."""
    df.to_csv(path, sep=" ", header=False, index=False)
