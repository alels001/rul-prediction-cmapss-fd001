"""FastAPI application.

Run locally:
    make serve                      # http://127.0.0.1:8000/docs

The model is loaded once at start-up from the directory in the environment
variable RUL_MODEL_DIR (default: models/). If it cannot be loaded the service
still starts, but /health reports 503 and predictions are refused, so a
container orchestrator can see that this instance is not ready.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from rul import __version__, config
from rul.api.schemas import (
    BatchRequest,
    BatchResponse,
    EngineHistory,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionResponse,
)
from rul.model import RULModel

logger = logging.getLogger("rul.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once when the server starts, not on every request."""
    model_dir = Path(os.environ.get("RUL_MODEL_DIR", config.MODELS_DIR))
    try:
        app.state.model = RULModel.load(model_dir)
        logger.info(
            "model loaded version=%s from=%s", app.state.model.metadata["model_version"], model_dir
        )
    except (FileNotFoundError, ValueError) as exc:
        app.state.model = None
        logger.error("model NOT loaded from %s: %s", model_dir, exc)
    yield


app = FastAPI(
    title="RUL Prediction API",
    description=(
        "Remaining Useful Life of turbofan engines (NASA C-MAPSS FD001). "
        "Send the raw sensor history of an engine; receive the predicted number of "
        "operating cycles left. XGBoost on 65 summary features of 13 sensors."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 with where and why the input is invalid, without echoing the input.

    FastAPI's default handler copies the offending value into the response; a
    NaN cannot be written as JSON, which turned a bad request into a 500 error.
    Omitting the value also keeps responses small for long histories.
    """
    errors = [
        {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


def get_model(request: Request) -> RULModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def predict_engines(model: RULModel, engines: list[EngineHistory]) -> list[Prediction]:
    histories = [np.array([c.values() for c in e.cycles], dtype=np.float64) for e in engines]
    started = time.perf_counter()
    ruls = model.predict_histories(histories)
    elapsed_ms = (time.perf_counter() - started) * 1000

    results = []
    for engine, history, rul in zip(engines, histories, ruls, strict=True):
        rul = round(float(rul), 2)
        results.append(
            Prediction(
                engine_id=engine.engine_id,
                rul=rul,
                capped=rul >= model.r_max,
                n_cycles_observed=len(history),
            )
        )
        logger.info(
            "prediction engine_id=%s n_cycles=%d rul=%.2f", engine.engine_id, len(history), rul
        )
    logger.info("predicted %d engine(s) in %.1f ms", len(engines), elapsed_ms)
    return results


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> JSONResponse:
    """200 when the model is loaded and the service can answer, 503 otherwise."""
    loaded = request.app.state.model is not None
    body = HealthResponse(status="ok" if loaded else "unavailable", model_loaded=loaded)
    return JSONResponse(status_code=200 if loaded else 503, content=body.model_dump())


@app.get("/model-info", response_model=ModelInfo, tags=["service"])
def model_info(request: Request) -> ModelInfo:
    """Version, training data fingerprint and test metrics of the loaded model."""
    md = get_model(request).metadata
    return ModelInfo(
        model_version=md["model_version"],
        trained_at=md.get("trained_at"),
        r_max=md["r_max"],
        sensors=md["sensors"],
        training_data=md.get("training_data"),
        test_metrics=md.get("test_metrics"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(engine: EngineHistory, request: Request) -> PredictionResponse:
    """Predict the remaining useful life of one engine at its last given cycle."""
    model = get_model(request)
    [prediction] = predict_engines(model, [engine])
    return PredictionResponse(
        **prediction.model_dump(), model_version=model.metadata["model_version"]
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    """Predict several engines in one request (up to 100). Order is preserved."""
    model = get_model(request)
    return BatchResponse(
        model_version=model.metadata["model_version"],
        predictions=predict_engines(model, batch.engines),
    )
