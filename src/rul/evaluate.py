"""Score a trained model on the 100 official test engines.

The test engines are passed through the same code path the API uses
(raw readings -> scaling -> summary features -> prediction), so the reported
error is the error of the deployed pipeline, not of a separate notebook.

Usage:
    python -m rul.evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rul import config
from rul.data import engine_histories, load_test, load_test_rul
from rul.model import METADATA_FILE, RULModel


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric PHM 2008 score; late predictions (d > 0) are penalised harder."""
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "phm_score": phm_score(y_true, y_pred),
        "n_engines": int(len(err)),
    }


def evaluate(model: RULModel, test_file: Path, rul_file: Path) -> dict[str, float]:
    histories = engine_histories(load_test(test_file))
    rul_true = load_test_rul(rul_file)
    units = sorted(histories)
    if units != list(rul_true.index):
        raise ValueError("Test engines and ground-truth file do not list the same units.")
    y_pred = model.predict_histories([histories[u] for u in units])
    return compute_metrics(rul_true.loc[units].to_numpy(), y_pred)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the FD001 RUL model on the test set.")
    parser.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--test-file", type=Path, default=config.TEST_FILE)
    parser.add_argument("--rul-file", type=Path, default=config.RUL_FILE)
    parser.add_argument(
        "--max-mae",
        type=float,
        default=None,
        help="Quality gate: exit with an error if the test MAE exceeds this value.",
    )
    args = parser.parse_args(argv)

    model = RULModel.load(args.model_dir)
    metrics = evaluate(model, args.test_file, args.rul_file)

    print(f"Model {model.metadata.get('model_version')} on {metrics['n_engines']} test engines")
    print(f"  MAE   {metrics['mae']:.2f} cycles")
    print(f"  RMSE  {metrics['rmse']:.2f} cycles")
    print(f"  PHM   {metrics['phm_score']:.2f}")

    # Record the result next to the model it belongs to.
    model.metadata["test_metrics"] = {k: round(v, 4) for k, v in metrics.items()}
    (Path(args.model_dir) / METADATA_FILE).write_text(json.dumps(model.metadata, indent=2))
    print(f"Metrics written to {args.model_dir}/{METADATA_FILE}")

    if args.max_mae is not None:
        if metrics["mae"] > args.max_mae:
            raise SystemExit(
                f"QUALITY GATE FAILED: MAE {metrics['mae']:.2f} > allowed {args.max_mae:.2f}"
            )
        print(f"Quality gate passed: MAE {metrics['mae']:.2f} <= {args.max_mae:.2f}")


if __name__ == "__main__":
    main()
