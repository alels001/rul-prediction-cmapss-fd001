"""Training, saving, loading and predicting with the full model bundle."""

import json

import numpy as np
import pytest

from rul import config
from rul.evaluate import compute_metrics, evaluate, phm_score
from rul.model import METADATA_FILE, RULModel
from rul.train import train
from tests.fakes import make_fleet, write_cmapss

SMALL_CFG = {
    "model_version": "test",
    "target": {"r_max": 125},
    "xgboost": {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.3,
        "objective": "reg:squarederror",
        "random_state": 0,
        "n_jobs": 1,
    },
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small model trained on synthetic data, saved to a temporary directory."""
    tmp = tmp_path_factory.mktemp("model")
    train_file = tmp / "train.txt"
    write_cmapss(make_fleet(n_engines=8, seed=3), train_file)
    model = train(SMALL_CFG, train_file=train_file)
    model.save(tmp / "artifacts")
    return model, tmp / "artifacts"


def test_bundle_contains_three_files(trained):
    _, directory = trained
    assert {p.name for p in directory.iterdir()} == {
        "xgb_model.json",
        "scaler.json",
        "metadata.json",
    }


def test_metadata_records_lineage(trained):
    model, _ = trained
    md = model.metadata
    assert md["feature_names"] == config.FEATURE_NAMES
    assert md["r_max"] == 125
    assert len(md["training_data"]["sha256"]) == 64
    assert md["xgboost_params"] == SMALL_CFG["xgboost"]


def test_save_load_roundtrip_gives_identical_predictions(trained, raw_history):
    model, directory = trained
    loaded = RULModel.load(directory)
    histories = [raw_history, raw_history[:5], raw_history[:1]]
    np.testing.assert_array_equal(
        model.predict_histories(histories), loaded.predict_histories(histories)
    )


def test_predictions_are_clipped_to_valid_range(trained):
    model, _ = trained
    rng = np.random.default_rng(0)
    extreme = [rng.normal(100, 50, size=(30, 13)) for _ in range(20)]
    pred = model.predict_histories(extreme)
    assert pred.min() >= 0
    assert pred.max() <= 125


def test_load_rejects_incomplete_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        RULModel.load(tmp_path)


def test_load_rejects_different_feature_layout(trained, tmp_path):
    model, _ = trained
    model.save(tmp_path)
    md = json.loads((tmp_path / METADATA_FILE).read_text())
    md["feature_names"] = md["feature_names"][::-1]
    (tmp_path / METADATA_FILE).write_text(json.dumps(md))
    with pytest.raises(ValueError, match="feature layout"):
        RULModel.load(tmp_path)


def test_evaluate_end_to_end(trained, tmp_path):
    model, _ = trained
    test = make_fleet(n_engines=4, seed=9)
    write_cmapss(test, tmp_path / "test.txt")
    (tmp_path / "rul.txt").write_text("10\n20\n30\n40\n")
    metrics = evaluate(model, tmp_path / "test.txt", tmp_path / "rul.txt")
    assert metrics["n_engines"] == 4
    assert metrics["mae"] >= 0


# ── Metrics ──────────────────────────────────────────────────────────────────


def test_phm_score_penalises_late_more_than_early():
    early = phm_score(np.array([100.0]), np.array([80.0]))  # 20 cycles early
    late = phm_score(np.array([100.0]), np.array([120.0]))  # 20 cycles late
    assert early == pytest.approx(np.exp(20 / 13) - 1)  # about 3.7
    assert late == pytest.approx(np.exp(20 / 10) - 1)  # about 6.4
    assert late > early


def test_compute_metrics():
    m = compute_metrics(np.array([10.0, 20.0]), np.array([13.0, 16.0]))
    assert m["mae"] == pytest.approx(3.5)
    assert m["rmse"] == pytest.approx(np.sqrt((9 + 16) / 2))


def test_command_line_train_then_evaluate(tmp_path, capsys):
    """The exact path `make train` and `make evaluate` take, on synthetic files."""
    import yaml

    from rul import evaluate as evaluate_cli
    from rul import train as train_cli

    write_cmapss(make_fleet(n_engines=6, seed=4), tmp_path / "train.txt")
    write_cmapss(make_fleet(n_engines=3, seed=5), tmp_path / "test.txt")
    (tmp_path / "rul.txt").write_text("5\n15\n25\n")
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(SMALL_CFG))
    out = tmp_path / "models"

    train_cli.main(
        [
            "--config",
            str(tmp_path / "train.yaml"),
            "--train-file",
            str(tmp_path / "train.txt"),
            "--output",
            str(out),
        ]
    )
    evaluate_cli.main(
        [
            "--model-dir",
            str(out),
            "--test-file",
            str(tmp_path / "test.txt"),
            "--rul-file",
            str(tmp_path / "rul.txt"),
        ]
    )

    md = json.loads((out / METADATA_FILE).read_text())
    assert md["test_metrics"]["n_engines"] == 3
    assert "MAE" in capsys.readouterr().out
