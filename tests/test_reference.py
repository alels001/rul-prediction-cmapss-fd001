"""Reference tests: the production features must equal the thesis implementation.

`_thesis_summarize` below is copied verbatim from notebook
10_Feature_Engineering_Wilcoxon.ipynb (the code that produced the reported
results). If anyone later rewrites `summarize_history` -- to speed it up, for
example -- these tests prove the model still receives the same inputs.
"""

import numpy as np
import pytest

from rul.features import summarize_history


def _thesis_summarize(w):  # verbatim from the notebook, do not edit
    n = w.shape[0]
    x = np.arange(n)
    feats = []
    for j in range(w.shape[1]):
        c = w[:, j]
        slope = np.polyfit(x, c, 1)[0] if n >= 2 else 0.0
        feats.extend([c.mean(), c.std(), slope, c[-1], c[-1] - c[0]])
    return np.array(feats, dtype=np.float32)


@pytest.mark.parametrize("n_cycles", [1, 2, 3, 31, 128, 362])
def test_matches_thesis_implementation(n_cycles):
    rng = np.random.default_rng(n_cycles)
    # Scaled readings lie roughly in [0, 1]; the notebook passes float32.
    history = rng.random((n_cycles, 13)).astype(np.float32)
    np.testing.assert_array_equal(summarize_history(history), _thesis_summarize(history))


def test_matches_thesis_on_a_degrading_trajectory():
    t = np.arange(200, dtype=np.float32)
    trend = 0.2 + 0.6 * (t / 199) ** 3  # slow start, fast end, like FD001
    history = np.tile(trend[:, None], (1, 13)).astype(np.float32)
    history += np.random.default_rng(0).normal(0, 0.02, history.shape).astype(np.float32)
    np.testing.assert_array_equal(summarize_history(history), _thesis_summarize(history))
