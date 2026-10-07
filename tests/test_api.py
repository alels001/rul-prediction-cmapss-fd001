"""HTTP tests for the FastAPI service.

FastAPI's TestClient sends real HTTP requests to the app in memory, without
starting a server. A small model trained on synthetic data is used, so these
tests need neither the NASA files nor a trained production model.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rul import config
from rul.api.main import app
from rul.model import RULModel
from rul.train import train
from tests.fakes import make_fleet, write_cmapss

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "engine_sample.json"

SMALL_CFG = {
    "model_version": "test-api",
    "target": {"r_max": 125},
    "xgboost": {"n_estimators": 10, "max_depth": 3, "random_state": 0, "n_jobs": 1},
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("api-model")
    write_cmapss(make_fleet(n_engines=8, seed=3), tmp / "train.txt")
    train(SMALL_CFG, train_file=tmp / "train.txt").save(tmp / "model")
    return tmp / "model"


@pytest.fixture
def client(model_dir, monkeypatch):
    """A client whose app loaded the synthetic model at start-up."""
    monkeypatch.setenv("RUL_MODEL_DIR", str(model_dir))
    with TestClient(app) as c:  # `with` runs the start-up (lifespan) code
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("RUL_MODEL_DIR", str(tmp_path))  # empty directory
    with TestClient(app) as c:
        yield c


def reading(value: float = 100.0, **overrides) -> dict:
    r = {s: value for s in config.SENSORS}
    r.update(overrides)
    return r


def engine(n: int = 20, engine_id: str = "e1") -> dict:
    rng = np.random.default_rng(n)
    return {
        "engine_id": engine_id,
        "cycles": [reading(100 + 0.05 * t + rng.normal(0, 0.1)) for t in range(n)],
    }


# ── Service endpoints ────────────────────────────────────────────────────────


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_missing_model(client_without_model):
    r = client_without_model.get("/health")
    assert r.status_code == 503
    assert r.json()["model_loaded"] is False


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_version"] == "test-api"
    assert body["r_max"] == 125
    assert body["sensors"] == config.SENSORS


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"


# ── Prediction ───────────────────────────────────────────────────────────────


def test_predict_returns_a_valid_prediction(client):
    r = client.post("/predict", json=engine(n=25))
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["rul"] <= 125
    assert body["n_cycles_observed"] == 25
    assert body["engine_id"] == "e1"
    assert body["model_version"] == "test-api"


def test_api_gives_the_same_answer_as_the_model(client, model_dir):
    """Training/serving parity end to end: HTTP + JSON + validation change nothing."""
    payload = engine(n=30)
    via_http = client.post("/predict", json=payload).json()["rul"]

    history = np.array([[c[s] for s in config.SENSORS] for c in payload["cycles"]])
    direct = RULModel.load(model_dir).predict_histories([history])[0]
    assert via_http == pytest.approx(round(float(direct), 2))


def test_single_cycle_history_is_accepted(client):
    r = client.post("/predict", json={"cycles": [reading()]})
    assert r.status_code == 200
    assert r.json()["n_cycles_observed"] == 1


def test_extra_columns_are_ignored(client):
    """A full raw C-MAPSS row (all 21 sensors and the settings) is accepted."""
    full = [reading(s1=518.67, s5=14.62, op1=0.0, op2=0.0, op3=100.0) for _ in range(5)]
    assert client.post("/predict", json={"cycles": full}).status_code == 200


def test_batch_preserves_order(client):
    engines = [engine(n=10, engine_id="a"), engine(n=40, engine_id="b")]
    r = client.post("/predict/batch", json={"engines": engines})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert [p["engine_id"] for p in preds] == ["a", "b"]
    assert [p["n_cycles_observed"] for p in preds] == [10, 40]


def test_example_request_file_is_valid(client):
    payload = json.loads(EXAMPLE.read_text())
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["n_cycles_observed"] == 31


def test_predict_refused_without_model(client_without_model):
    r = client_without_model.post("/predict", json=engine())
    assert r.status_code == 503


# ── Input validation (all rejected with 422 before reaching the model) ──────


def bad_requests():
    missing = reading()
    del missing["s11"]
    return {
        "missing sensor": {"cycles": [missing]},
        "text instead of number": {"cycles": [reading(s2="hot")]},
        "empty history": {"cycles": []},
        "too long history": {"cycles": [reading()] * (config.MAX_HISTORY + 1)},
        "no cycles field": {"engine_id": "x"},
        "cycles out of order": {"cycles": [reading(cycle=2), reading(cycle=1)]},
        "cycle numbers partly given": {"cycles": [reading(cycle=1), reading()]},
    }


@pytest.mark.parametrize("case", list(bad_requests()))
def test_invalid_input_is_rejected(client, case):
    r = client.post("/predict", json=bad_requests()[case])
    assert r.status_code == 422, case


def test_non_finite_value_is_rejected(client):
    body = json.dumps({"cycles": [reading()]}).replace("100.0", "NaN", 1)
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422
    [error] = r.json()["detail"]
    assert error["loc"] == ["body", "cycles", 0, "s2"]  # says exactly which value is wrong
    assert "input" not in error  # the request is not echoed back


def test_batch_size_is_limited(client):
    r = client.post("/predict/batch", json={"engines": [engine(n=2)] * 101})
    assert r.status_code == 422
