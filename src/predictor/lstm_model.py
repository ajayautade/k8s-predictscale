# ============================================
# K8s PredictScale - LSTM Model
# ============================================
# Defines, trains, and runs inference on a
# stacked LSTM network for multi-step time-
# series forecasting.
# ============================================

import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# TensorFlow is imported lazily so the module can be imported even if
# TF is not installed (e.g. during unit-testing with mocks).
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:  # pragma: no cover
    tf = None  # type: ignore[assignment]
    keras = None  # type: ignore[assignment]
    layers = None  # type: ignore[assignment]


class LSTMModel:
    """Stacked LSTM for multi-step time-series prediction.

    Architecture::

        Input (lookback × features)
          → LSTM(128) + BatchNorm + Dropout(0.2)
          → LSTM(64)  + BatchNorm + Dropout(0.2)
          → Dense(32, ReLU)
          → Dense(forecast_steps)

    The model predicts the next *forecast_steps* values for a single
    target metric (e.g. CPU usage).
    """

    def __init__(
        self,
        lookback_steps: int = 60,
        forecast_steps: int = 10,
        n_features: int = 8,
        lstm_units: List[int] | None = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        self._lookback = lookback_steps
        self._forecast = forecast_steps
        self._n_features = n_features
        self._lstm_units = lstm_units or [128, 64]
        self._dropout = dropout_rate
        self._lr = learning_rate

        self._model: Optional[Any] = None
        self._history: Optional[Any] = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct the Keras model graph."""
        if keras is None:
            raise ImportError("TensorFlow is required but not installed.")

        inputs = keras.Input(shape=(self._lookback, self._n_features))
        x = inputs

        for i, units in enumerate(self._lstm_units):
            return_seq = i < len(self._lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_seq, name=f"lstm_{i+1}")(x)
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
            x = layers.Dropout(self._dropout, name=f"drop_{i+1}")(x)

        x = layers.Dense(32, activation="relu", name="dense_hidden")(x)
        outputs = layers.Dense(self._forecast, name="forecast_output")(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name="predictscale_lstm")
        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self._lr),
            loss="mse",
            metrics=["mae"],
        )
        logger.info(
            "lstm_model_built",
            params=self._model.count_params(),
            lstm_units=self._lstm_units,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            X_train: Training features ``(samples, lookback, features)``.
            y_train: Training targets ``(samples, forecast_steps)``.
            X_val: Optional validation features.
            y_val: Optional validation targets.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            Training history dict (``loss``, ``mae``, etc.).
        """
        if self._model is None:
            self.build()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self._history = self._model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        self._is_trained = True
        final_loss = float(self._history.history["loss"][-1])
        final_mae = float(self._history.history["mae"][-1])
        logger.info(
            "lstm_training_complete",
            epochs_run=len(self._history.history["loss"]),
            final_loss=round(final_loss, 6),
            final_mae=round(final_mae, 6),
        )
        return self._history.history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Input array ``(samples, lookback, features)``.

        Returns:
            Predictions of shape ``(samples, forecast_steps)``.
        """
        if self._model is None or not self._is_trained:
            raise RuntimeError("Model is not trained yet.")
        return self._model.predict(X, verbose=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(path, exist_ok=True)
        self._model.save(os.path.join(path, "lstm_model.keras"))
        logger.info("lstm_model_saved", path=path)

    def load(self, path: str) -> None:
        """Load model weights from disk."""
        model_path = os.path.join(path, "lstm_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file at {model_path}")
        self._model = keras.models.load_model(model_path)
        self._is_trained = True
        logger.info("lstm_model_loaded", path=model_path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_history(self) -> Optional[Dict[str, List[float]]]:
        return self._history.history if self._history else None

    def get_model_summary(self) -> Dict[str, Any]:
        """Return a summary dict for the REST API."""
        return {
            "type": "LSTM",
            "is_trained": self._is_trained,
            "lookback_steps": self._lookback,
            "forecast_steps": self._forecast,
            "n_features": self._n_features,
            "lstm_units": self._lstm_units,
            "dropout_rate": self._dropout,
            "total_params": self._model.count_params() if self._model else 0,
        }
