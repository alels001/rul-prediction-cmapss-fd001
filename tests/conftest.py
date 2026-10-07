"""Shared test fixtures (synthetic data only; see tests/fakes.py)."""

import numpy as np
import pandas as pd
import pytest

from rul import config
from tests.fakes import make_fleet


@pytest.fixture
def fleet() -> pd.DataFrame:
    return make_fleet()


@pytest.fixture
def raw_history() -> np.ndarray:
    """One engine's raw readings for the 13 retained sensors, shape (40, 13)."""
    df = make_fleet(n_engines=1, seed=1, min_life=40, max_life=40)
    return df[config.SENSORS].to_numpy()
