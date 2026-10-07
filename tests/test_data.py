"""Loading the raw files and building the training target."""

import numpy as np
import pytest

from rul import config
from rul.data import add_targets, engine_histories, load_test_rul, read_cmapss
from tests.fakes import make_fleet, write_cmapss


def test_read_cmapss_roundtrip(tmp_path, fleet):
    path = tmp_path / "train_FD001.txt"
    write_cmapss(fleet, path)
    df = read_cmapss(path)
    assert list(df.columns) == config.RAW_COLUMNS
    assert len(df) == len(fleet)


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        read_cmapss(tmp_path / "nope.txt")


def test_linear_target_counts_down_to_zero(fleet):
    df = add_targets(fleet, r_max=125)
    for _, g in df.groupby("unit"):
        assert g["rul_linear"].iloc[-1] == 0
        assert g["rul_linear"].iloc[0] == len(g) - 1
        assert (np.diff(g["rul_linear"].to_numpy()) == -1).all()


def test_piecewise_target_is_capped():
    df = add_targets(make_fleet(n_engines=2, min_life=200, max_life=200), r_max=125)
    assert df["rul_piecewise"].max() == 125
    early = df[df["rul_linear"] >= 125]
    assert (early["rul_piecewise"] == 125).all()
    late = df[df["rul_linear"] < 125]
    assert (late["rul_piecewise"] == late["rul_linear"]).all()


def test_engine_histories_shape_and_order(fleet):
    h = engine_histories(fleet.sample(frac=1, random_state=0))  # shuffled rows
    assert sorted(h) == list(range(1, 7))
    first = fleet[fleet["unit"] == 1]
    assert h[1].shape == (len(first), 13)
    np.testing.assert_array_equal(h[1], first[config.SENSORS].to_numpy())


def test_load_test_rul_indexes_engines_from_one(tmp_path):
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112\n98\n69\n")
    rul = load_test_rul(path)
    assert list(rul.index) == [1, 2, 3]
    assert rul.loc[2] == 98
