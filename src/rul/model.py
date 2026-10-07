"""The deployable model: scaler + XGBoost booster + metadata, saved together.

A model directory always contains three files:

    xgb_model.json   the fitted trees (XGBoost's portable JSON format)
    scaler.json      min/max of each sensor from the training engines
    metadata.json    version, configuration, feature layout, data hash, metrics
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb

from rul import config
from rul.features import SensorScaler, features_from_raw

MODEL_FILE = "xgb_model.json"
SCALER_FILE = "scaler.json"
METADATA_FILE = "metadata.json"


@dataclass
class RULModel:
    regressor: xgb.XGBRegressor
    scaler: SensorScaler
    metadata: dict = field(default_factory=dict)
    r_max: int = config.R_MAX

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        """Predict from an already-built (n, 65) feature matrix, clipped to [0, r_max]."""
        raw = self.regressor.predict(np.asarray(X, dtype=np.float32))
        return np.clip(raw, 0, self.r_max)

    def predict_histories(self, histories: Sequence[np.ndarray]) -> np.ndarray:
        """Predict remaining life from raw sensor histories, one per engine.

        Each history is an (n_cycles, 13) array of raw readings, oldest first,
        with the sensors in ``config.SENSORS`` order. The prediction refers to
        the last cycle of each history.
        """
        X = np.stack([features_from_raw(h, self.scaler) for h in histories])
        return self.predict_features(X)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.regressor.save_model(directory / MODEL_FILE)
        (directory / SCALER_FILE).write_text(json.dumps(self.scaler.to_dict(), indent=2))
        (directory / METADATA_FILE).write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def load(cls, directory: Path = config.MODELS_DIR) -> RULModel:
        directory = Path(directory)
        missing = [
            f for f in (MODEL_FILE, SCALER_FILE, METADATA_FILE) if not (directory / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Model directory {directory} is missing {missing}. Run `make train` first."
            )
        metadata = json.loads((directory / METADATA_FILE).read_text())
        if metadata.get("feature_names") != config.FEATURE_NAMES:
            raise ValueError("Saved model was trained on a different feature layout.")
        regressor = xgb.XGBRegressor()
        regressor.load_model(directory / MODEL_FILE)
        scaler = SensorScaler.from_dict(json.loads((directory / SCALER_FILE).read_text()))
        return cls(
            regressor=regressor,
            scaler=scaler,
            metadata=metadata,
            r_max=int(metadata.get("r_max", config.R_MAX)),
        )
