"""Checks on the real NASA FD001 files.

These run only where the data is present (your machine). Without the files
they are skipped automatically, so CI stays green without access to the data.
They are also marked `slow` because the last one trains the full model.

    make test       -> fast tests only (skips this file)
    make test-all   -> everything, including these
"""

import numpy as np
import pytest

from rul import config
from rul.data import add_targets, load_test, load_test_rul, load_train

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not all(p.exists() for p in (config.TRAIN_FILE, config.TEST_FILE, config.RUL_FILE)),
        reason="NASA FD001 files not in data/ (see data/README.md)",
    ),
]

# Upper bound for the test MAE of the full model. Observed values: 8.36 (Linux
# x86, xgboost 3.4.1), 8.71 (macOS arm64), thesis five-seed mean 8.59 +/- 0.08.
# The margin absorbs platform differences but catches a broken pipeline.
MAX_ACCEPTABLE_MAE = 9.5


def test_dataset_matches_the_thesis():
    train, test, rul = load_train(), load_test(), load_test_rul()
    assert train.shape == (20_631, 26)
    assert test.shape == (13_096, 26)
    assert len(rul) == 100

    lifetimes = train.groupby("unit")["cycle"].max()
    assert (lifetimes.min(), lifetimes.max()) == (128, 362)
    assert round(lifetimes.mean(), 1) == 206.3


def test_cap_alters_the_documented_share_of_rows():
    df = add_targets(load_train())
    altered = (df["rul_linear"] > config.R_MAX).sum()
    assert altered == 8_031  # 38.9 % of rows, Table 5.3 of the thesis


def test_full_pipeline_meets_the_quality_bar():
    from rul.evaluate import evaluate
    from rul.train import load_config, train

    model = train(load_config(config.CONFIGS_DIR / "train.yaml"))
    metrics = evaluate(model, config.TEST_FILE, config.RUL_FILE)

    assert metrics["n_engines"] == 100
    assert np.isfinite(metrics["mae"])
    assert metrics["mae"] < MAX_ACCEPTABLE_MAE
