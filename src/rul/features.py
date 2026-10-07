"""Feature construction shared by training and serving.

There is exactly one implementation of each step. Training, evaluation and the
API all call these functions, so the model always receives inputs built the
same way it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rul import config

N_SENSORS = len(config.SENSORS)


# ── Scaling ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SensorScaler:
    """Min-max scaling of the 13 sensors, fitted on the training engines only.

    Reproduces scikit-learn's ``MinMaxScaler`` arithmetic (x * scale + min_)
    but stores only two plain lists, so it can be saved as JSON and loaded
    without pickle or a matching scikit-learn version.
    """

    data_min: tuple[float, ...]
    data_max: tuple[float, ...]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> SensorScaler:
        values = df[config.SENSORS].to_numpy(dtype=np.float64)
        return cls(
            data_min=tuple(float(v) for v in values.min(axis=0)),
            data_max=tuple(float(v) for v in values.max(axis=0)),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Scale an (n, 13) array of raw readings; returns float64."""
        data_min = np.asarray(self.data_min, dtype=np.float64)
        data_range = np.asarray(self.data_max, dtype=np.float64) - data_min
        scale = 1.0 / data_range
        min_ = -data_min * scale
        return np.asarray(values, dtype=np.float64) * scale + min_

    def to_dict(self) -> dict:
        return {"sensors": config.SENSORS, "data_min": self.data_min, "data_max": self.data_max}

    @classmethod
    def from_dict(cls, d: dict) -> SensorScaler:
        if list(d["sensors"]) != config.SENSORS:
            raise ValueError("Saved scaler was fitted on a different sensor list or order.")
        return cls(data_min=tuple(d["data_min"]), data_max=tuple(d["data_max"]))


# ── Summary statistics ───────────────────────────────────────────────────────


def summarize_history(history: np.ndarray) -> np.ndarray:
    """Five statistics per sensor over the whole observed history.

    Parameters
    ----------
    history : array of shape (n_cycles, 13)
        Scaled readings in cycle order, oldest first.

    Returns
    -------
    array of shape (65,), float32, ordered sensor by sensor:
        mean, standard deviation (population, divisor n), least-squares slope
        per cycle, last value, last minus first.
        For a one-cycle history the slope is 0; std and delta are 0 by
        construction.
    """
    h = np.asarray(history, dtype=np.float32)
    if h.ndim != 2 or h.shape[1] != N_SENSORS:
        raise ValueError(f"history must have shape (n_cycles, {N_SENSORS}), got {h.shape}")
    n = h.shape[0]
    if n < 1:
        raise ValueError("history must contain at least one cycle")

    x = np.arange(n)
    feats: list[float] = []
    for j in range(N_SENSORS):
        c = h[:, j]
        slope = np.polyfit(x, c, 1)[0] if n >= 2 else 0.0
        feats.extend([c.mean(), c.std(), slope, c[-1], c[-1] - c[0]])
    return np.array(feats, dtype=np.float32)


def features_from_raw(history_raw: np.ndarray, scaler: SensorScaler) -> np.ndarray:
    """Raw readings of one engine -> the 65 model inputs. Used by serving."""
    scaled = scaler.transform(history_raw).astype(np.float32)
    return summarize_history(scaled)


# ── Training set ─────────────────────────────────────────────────────────────


def build_training_set(
    train_df: pd.DataFrame, scaler: SensorScaler, target_col: str = "rul_piecewise"
) -> tuple[np.ndarray, np.ndarray]:
    """One sample per training cycle.

    For every engine and every cycle t, the features summarise cycles 1..t
    only (causal: nothing after t is used) and the label is the target at t.
    """
    X, y = [], []
    for _, g in train_df.groupby("unit"):
        g = g.sort_values("cycle")
        scaled = scaler.transform(g[config.SENSORS].to_numpy()).astype(np.float32)
        labels = g[target_col].to_numpy(dtype=np.float32)
        for end in range(len(g)):
            X.append(summarize_history(scaled[: end + 1]))
            y.append(labels[end])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
