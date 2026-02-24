# ============================================
# K8s PredictScale - Predictor Service
# ============================================
# High-level orchestrator that ties the LSTM,
# Prophet, Ensemble, and Model Manager together
# into a single prediction lifecycle.
# ============================================

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.predictor.ensemble import EnsembleCombiner, PredictionResult
from src.predictor.lstm_model import LSTMModel
from src.predictor.model_manager import ModelManager
from src.predictor.prophet_model import ProphetModel
from src.preprocessor.pipeline import PreprocessingPipeline
from src.utils.config import PredictionConfig, ScalingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictorService:
    """Orchestrates the full prediction lifecycle.

    Responsibilities:
        - Initial model training (fit phase).
        - Real-time prediction using the fitted pipeline.
        - Periodic re-training when model drift is detected.
        - Computing recommended replica count from predictions.
    """

    def __init__(
        self,
        prediction_config: PredictionConfig,
        scaling_config: ScalingConfig,
    ):
        self._pred_cfg = prediction_config
        self._scale_cfg = scaling_config

        self._pipeline = PreprocessingPipeline(
            lookback_steps=prediction_config.lookback_steps,
            forecast_steps=prediction_config.horizon,
            target_column="cpu_usage",
        )
        self._lstm = LSTMModel(
            lookback_steps=prediction_config.lookback_steps,
            forecast_steps=prediction_config.horizon,
            lstm_units=prediction_config.lstm_units,
            dropout_rate=prediction_config.dropout_rate,
        )
        self._prophet = ProphetModel(
            forecast_steps=prediction_config.horizon,
        )
        self._ensemble = EnsembleCombiner()
        self._model_manager = ModelManager(base_path=prediction_config.model_path)

        self._last_prediction: Optional[PredictionResult] = None
        self._prediction_count = 0
        self._training_data_hours = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Train both models on historical data.

        Args:
            historical_data: Raw metrics DataFrame (timestamp-indexed).

        Returns:
            Training summary dict.
        """
        logger.info("training_started", rows=len(historical_data))

        # Track how many hours of data we have
        if not historical_data.empty:
            span = (historical_data.index.max() - historical_data.index.min())
            self._training_data_hours = span.total_seconds() / 3600

        # ---- LSTM ----
        lstm_history = {}
        if self._training_data_hours >= self._pred_cfg.min_training_hours:
            X, y = self._pipeline.fit_transform(historical_data)
            if len(X) > 0:
                split = int(len(X) * 0.8)
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]

                # Update feature count on the LSTM before building
                self._lstm._n_features = X.shape[2]
                self._lstm.build()
                lstm_history = self._lstm.train(
                    X_train, y_train, X_val, y_val,
                    epochs=self._pred_cfg.epochs,
                    batch_size=self._pred_cfg.batch_size,
                )
        else:
            logger.info(
                "lstm_skipped_insufficient_data",
                hours=round(self._training_data_hours, 1),
                required=self._pred_cfg.min_training_hours,
            )
            # Still fit the pipeline for Prophet
            self._pipeline.fit_transform(historical_data)

        # ---- Prophet ----
        prophet_result = {}
        try:
            prophet_result = self._prophet.train(historical_data, target_column="cpu_usage")
        except Exception as exc:
            logger.warning("prophet_training_failed", error=str(exc))

        # ---- Save models ----
        metrics = {
            "lstm_trained": self._lstm.is_trained,
            "prophet_trained": self._prophet.is_trained,
            "training_data_hours": round(self._training_data_hours, 1),
        }
        if lstm_history:
            metrics["lstm_final_mae"] = lstm_history.get("mae", [None])[-1]

        version = self._model_manager.create_version(metrics=metrics)
        if self._lstm.is_trained:
            self._lstm.save(version.path)
        if self._prophet.is_trained:
            self._prophet.save(version.path)

        self._model_manager.promote_version(version.version)
        self._model_manager.cleanup_old_versions(keep=5)

        logger.info("training_complete", version=version.version, metrics=metrics)
        return {
            "version": version.version,
            "metrics": metrics,
            "lstm_history": lstm_history,
            "prophet_result": prophet_result,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, recent_data: pd.DataFrame) -> PredictionResult:
        """Generate a prediction from recent metrics.

        Args:
            recent_data: Most-recent DataFrame from the collector buffer.

        Returns:
            :class:`PredictionResult` with ensemble output.
        """
        lstm_pred = None
        prophet_pred = None
        prophet_lower = None
        prophet_upper = None

        # ---- LSTM prediction ----
        if self._lstm.is_trained and self._pipeline.is_fitted:
            try:
                X = self._pipeline.transform(recent_data)
                raw = self._lstm.predict(X)
                # Inverse-transform the target column back to original scale
                lstm_pred = self._pipeline.normalizer.inverse_transform_column(
                    raw.flatten(), self._pipeline.target_column
                )
            except Exception as exc:
                logger.warning("lstm_prediction_failed", error=str(exc))

        # ---- Prophet prediction ----
        if self._prophet.is_trained:
            try:
                last_ts = recent_data.index.max() if not recent_data.empty else None
                intervals = self._prophet.predict_with_intervals(last_timestamp=last_ts)
                prophet_pred = intervals["yhat"]
                prophet_lower = intervals["yhat_lower"]
                prophet_upper = intervals["yhat_upper"]
            except Exception as exc:
                logger.warning("prophet_prediction_failed", error=str(exc))

        # ---- Ensemble ----
        result = self._ensemble.combine(
            lstm_prediction=lstm_pred,
            prophet_prediction=prophet_pred,
            prophet_lower=prophet_lower,
            prophet_upper=prophet_upper,
        )

        # ---- Compute recommended replicas ----
        peak_predicted = float(np.max(result.predicted_values))
        target_utilization = self._scale_cfg.target_cpu_utilization
        recommended = math.ceil(peak_predicted / target_utilization) if target_utilization > 0 else 1
        recommended = max(self._scale_cfg.min_replicas, min(self._scale_cfg.max_replicas, recommended))
        result.recommended_replicas = recommended

        # ---- Urgency ----
        result.scaling_urgency = self._compute_urgency(result)

        self._last_prediction = result
        self._prediction_count += 1

        logger.info(
            "prediction_generated",
            peak=round(peak_predicted, 4),
            recommended_replicas=recommended,
            confidence=round(result.confidence, 4),
            urgency=result.scaling_urgency,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_urgency(self, result: PredictionResult) -> str:
        """Classify scaling urgency based on prediction trend."""
        values = result.predicted_values
        if len(values) < 2:
            return "low"

        trend = values[-1] - values[0]
        pct_change = abs(trend) / (abs(values[0]) + 1e-8)

        if pct_change > 0.5:
            return "high"
        elif pct_change > 0.2:
            return "medium"
        return "low"

    def update_weights_with_actuals(self, actuals: np.ndarray) -> None:
        """Feed ground-truth back to the ensemble for weight updates."""
        self._ensemble.update_weights(
            actual=actuals,
            lstm_pred=None,  # Would need cached per-model preds
            prophet_pred=None,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def last_prediction(self) -> Optional[PredictionResult]:
        return self._last_prediction

    def get_status(self) -> Dict[str, Any]:
        """Return operational status for the REST API."""
        active = self._model_manager.get_active_version()
        return {
            "lstm_trained": self._lstm.is_trained,
            "prophet_trained": self._prophet.is_trained,
            "pipeline_fitted": self._pipeline.is_fitted,
            "active_model_version": active.version if active else None,
            "prediction_count": self._prediction_count,
            "training_data_hours": round(self._training_data_hours, 1),
            "ensemble": self._ensemble.get_combiner_stats(),
            "last_prediction": self._last_prediction.to_dict() if self._last_prediction else None,
        }
