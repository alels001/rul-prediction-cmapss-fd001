"""Train the production model and write it to models/.

Usage:
    python -m rul.train                      # defaults
    python -m rul.train --config configs/train.yaml --output models
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

from rul import __version__, config
from rul.data import add_targets, load_train
from rul.features import SensorScaler, build_training_set
from rul.model import RULModel


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def file_sha256(path: Path) -> str:
    """Fingerprint of the training file, so a model can be traced to its data."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(cfg: dict, train_file: Path = config.TRAIN_FILE) -> RULModel:
    r_max = int(cfg["target"]["r_max"])

    print(f"Loading training data from {train_file}")
    train_df = add_targets(load_train(train_file), r_max=r_max)
    n_engines = train_df["unit"].nunique()

    scaler = SensorScaler.fit(train_df)

    print("Building features (one sample per training cycle) ...")
    t0 = time.perf_counter()
    X, y = build_training_set(train_df, scaler, target_col="rul_piecewise")
    print(f"  X {X.shape}, y {y.shape} in {time.perf_counter() - t0:.1f}s")

    print("Fitting XGBoost ...")
    t0 = time.perf_counter()
    regressor = xgb.XGBRegressor(**cfg["xgboost"]).fit(X, y)
    fit_seconds = time.perf_counter() - t0
    print(f"  fitted in {fit_seconds:.1f}s")

    metadata = {
        "model_version": cfg["model_version"],
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "training_data": {
            "file": Path(train_file).name,
            "sha256": file_sha256(train_file),
            "n_engines": int(n_engines),
            "n_samples": int(X.shape[0]),
        },
        "target": "piecewise_linear",
        "r_max": r_max,
        "sensors": config.SENSORS,
        "feature_names": config.FEATURE_NAMES,
        "xgboost_params": cfg["xgboost"],
        "fit_seconds": round(fit_seconds, 2),
        "environment": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
        },
    }
    return RULModel(regressor=regressor, scaler=scaler, metadata=metadata, r_max=r_max)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the FD001 RUL model.")
    parser.add_argument("--config", type=Path, default=config.CONFIGS_DIR / "train.yaml")
    parser.add_argument("--train-file", type=Path, default=config.TRAIN_FILE)
    parser.add_argument("--output", type=Path, default=config.MODELS_DIR)
    args = parser.parse_args(argv)

    model = train(load_config(args.config), train_file=args.train_file)
    model.save(args.output)
    print(f"Saved model {model.metadata['model_version']} to {args.output}/")


if __name__ == "__main__":
    main()
