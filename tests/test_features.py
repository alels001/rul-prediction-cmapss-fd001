"""The 65-feature summary and the sensor scaler."""

import json

import numpy as np
import pytest

from rul import config
from rul.data import add_targets
from rul.features import (
    SensorScaler,
    build_training_set,
    features_from_raw,
    summarize_history,
)

STD, SLOPE, LAST, DELTA = 1, 2, 3, 4  # positions of each statistic within a sensor's block


def block(features: np.ndarray, sensor_idx: int) -> np.ndarray:
    """The five statistics of one sensor."""
    return features[sensor_idx * 5 : sensor_idx * 5 + 5]


# ── summarize_history ────────────────────────────────────────────────────────


def test_output_shape_and_dtype():
    f = summarize_history(np.random.default_rng(0).random((20, 13)))
    assert f.shape == (65,)
    assert f.dtype == np.float32


def test_single_cycle_history():
    h = np.arange(13, dtype=float).reshape(1, 13)
    f = summarize_history(h)
    for j in range(13):
        mean, std, slope, last, delta = block(f, j)
        assert mean == last == j
        assert std == slope == delta == 0


def test_known_linear_trend():
    n, intercept, slope = 10, 0.2, 0.05
    column = intercept + slope * np.arange(n)
    h = np.tile(column[:, None], (1, 13))
    f = summarize_history(h)
    assert block(f, 0)[SLOPE] == pytest.approx(slope, rel=1e-5)
    assert block(f, 0)[DELTA] == pytest.approx(slope * (n - 1), rel=1e-5)
    assert block(f, 0)[LAST] == pytest.approx(column[-1], rel=1e-6)
    # population standard deviation (divisor n), as in the thesis implementation
    assert block(f, 0)[STD] == pytest.approx(np.std(column, ddof=0), rel=1e-5)


def test_features_are_sensor_major():
    h = np.zeros((5, 13))
    h[:, 3] = np.arange(5)  # only the fourth sensor changes
    f = summarize_history(h)
    changed = np.flatnonzero(f)
    assert set(changed) <= set(range(15, 20))  # block of sensor index 3


@pytest.mark.parametrize("bad", [np.zeros((5, 12)), np.zeros((0, 13)), np.zeros(13)])
def test_rejects_wrong_shapes(bad):
    with pytest.raises(ValueError):
        summarize_history(bad)


# ── SensorScaler ─────────────────────────────────────────────────────────────


def test_scaler_maps_training_range_to_unit_interval(fleet):
    scaler = SensorScaler.fit(fleet)
    scaled = scaler.transform(fleet[config.SENSORS].to_numpy())
    np.testing.assert_allclose(scaled.min(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.max(axis=0), 1, atol=1e-12)


def test_scaler_matches_sklearn(fleet):
    from sklearn.preprocessing import MinMaxScaler

    values = fleet[config.SENSORS].to_numpy()
    ours = SensorScaler.fit(fleet).transform(values)
    ref = MinMaxScaler().fit(values).transform(values)
    np.testing.assert_array_equal(ours, ref)


def test_scaler_json_roundtrip_is_exact(fleet):
    scaler = SensorScaler.fit(fleet)
    restored = SensorScaler.from_dict(json.loads(json.dumps(scaler.to_dict())))
    values = fleet[config.SENSORS].to_numpy()
    np.testing.assert_array_equal(scaler.transform(values), restored.transform(values))


def test_scaler_rejects_other_sensor_order(fleet):
    d = SensorScaler.fit(fleet).to_dict()
    d["sensors"] = list(reversed(d["sensors"]))
    with pytest.raises(ValueError):
        SensorScaler.from_dict(d)


# ── Training set ─────────────────────────────────────────────────────────────


def test_one_training_sample_per_cycle(fleet):
    df = add_targets(fleet)
    X, y = build_training_set(df, SensorScaler.fit(df))
    assert X.shape == (len(df), 65)
    assert y.shape == (len(df),)


def test_training_features_are_causal(fleet):
    """Changing a later cycle must not change the features of any earlier cycle."""
    df = add_targets(fleet)
    scaler = SensorScaler.fit(df)
    X_before, _ = build_training_set(df, scaler)

    engine1 = df.index[df["unit"] == 1]
    t = 10
    altered = df.copy()
    altered.loc[engine1[t:], config.SENSORS] += 50.0  # corrupt cycles after t
    X_after, _ = build_training_set(altered, scaler)

    np.testing.assert_array_equal(X_before[:t], X_after[:t])
    assert not np.array_equal(X_before[t], X_after[t])


def test_training_row_equals_serving_features(fleet):
    """The row for (engine, cycle t) equals what serving builds from cycles 1..t."""
    df = add_targets(fleet)
    scaler = SensorScaler.fit(df)
    X, _ = build_training_set(df, scaler)
    raw = df[df["unit"] == 1][config.SENSORS].to_numpy()
    for t in (1, 2, 15, len(raw)):
        np.testing.assert_array_equal(X[t - 1], features_from_raw(raw[:t], scaler))
