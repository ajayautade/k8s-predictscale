#!/usr/bin/env python3
# ============================================
# K8s PredictScale - Model Training Script
# ============================================
# Standalone script for training the LSTM and
# Prophet models on historical or synthetic data.
# ============================================

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor.lstm_model import LSTMModel
from src.predictor.prophet_model import ProphetModel
from src.predictor.model_manager import ModelManager
from src.preprocessor.pipeline import PreprocessingPipeline
from src.utils.logger import setup_logging, get_logger


def load_data(path: str) -> pd.DataFrame:
    """Load metrics from a CSV file."""
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df


def train_models(
    data_path: str,
    model_dir: str = "./models",
    lookback: int = 60,
    forecast: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
):
    """Train both LSTM and Prophet models.

    Args:
        data_path: Path to the training data CSV.
        model_dir: Directory to save model artifacts.
        lookback: LSTM lookback window size.
        forecast: Number of future steps to predict.
        epochs: Training epochs for LSTM.
        batch_size: Mini-batch size for LSTM.
    """
    setup_logging("INFO")
    logger = get_logger("train")

    logger.info("loading_data", path=data_path)
    df = load_data(data_path)
    logger.info("data_loaded", rows=len(df), columns=list(df.columns))

    # ---- Preprocessing ----
    pipeline = PreprocessingPipeline(
        lookback_steps=lookback,
        forecast_steps=forecast,
        target_column="cpu_usage",
    )

    X, y = pipeline.fit_transform(df)
    logger.info("preprocessing_complete", X_shape=X.shape, y_shape=y.shape)

    # Train/validation split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    manager = ModelManager(base_path=model_dir)

    # ---- LSTM Training ----
    logger.info("training_lstm", epochs=epochs)
    lstm = LSTMModel(
        lookback_steps=lookback,
        forecast_steps=forecast,
        n_features=X.shape[2],
    )
    lstm.build()
    history = lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Validation metrics
    val_pred = lstm.predict(X_val)
    val_mae = float(np.mean(np.abs(y_val - val_pred)))
    logger.info("lstm_validation", mae=round(val_mae, 6))

    # ---- Prophet Training ----
    logger.info("training_prophet")
    prophet = ProphetModel(forecast_steps=forecast)
    prophet_result = prophet.train(df, target_column="cpu_usage")

    # ---- Save models ----
    metrics = {
        "lstm_val_mae": val_mae,
        "lstm_final_loss": history["loss"][-1],
        "lstm_epochs": len(history["loss"]),
        "prophet_in_sample_mae": prophet_result.get("in_sample_mae", 0),
        "training_rows": len(df),
    }

    version = manager.create_version(metrics=metrics)
    lstm.save(version.path)
    prophet.save(version.path)
    manager.promote_version(version.version)

    logger.info("training_complete", version=version.version, metrics=metrics)

    # Save training report
    report = {
        "version": version.version,
        "metrics": metrics,
        "lstm_history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    report_path = os.path.join(version.path, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Training Complete!")
    print(f"   Version: {version.version}")
    print(f"   LSTM Val MAE: {val_mae:.6f}")
    print(f"   Prophet MAE: {prophet_result.get('in_sample_mae', 'N/A')}")
    print(f"   Models saved to: {version.path}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Train PredictScale models")
    parser.add_argument(
        "--data", type=str, default="data/synthetic_metrics.csv",
        help="Path to training data CSV",
    )
    parser.add_argument("--model-dir", type=str, default="./models", help="Model output directory")
    parser.add_argument("--lookback", type=int, default=60, help="LSTM lookback steps")
    parser.add_argument("--forecast", type=int, default=10, help="Forecast horizon steps")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train_models(
        data_path=args.data,
        model_dir=args.model_dir,
        lookback=args.lookback,
        forecast=args.forecast,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
