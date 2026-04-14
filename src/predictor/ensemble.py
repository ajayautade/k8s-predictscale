# ============================================
# K8s PredictScale - Ensemble Combiner
# ============================================
# Blends predictions from the LSTM and Prophet
# models using dynamically adjusted weights
# based on recent prediction accuracy.
# ============================================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Structured output from the ensemble predictor.

    Attributes:
        predicted_values: Array of forecasted values (forecast_steps,).
        confidence: Scalar confidence score in [0, 1].
        confidence_band_lower: Lower bound of the uncertainty interval.
        confidence_band_upper: Upper bound of the uncertainty interval.
        lstm_weight: Current LSTM weight in the ensemble.
        prophet_weight: Current Prophet weight in the ensemble.
        recommended_replicas: Suggested pod count based on prediction.
        scaling_urgency: ``"low"``, ``"medium"``, or ``"high"``.
    """

    predicted_values: np.ndarray
    confidence: float
    confidence_band_lower: np.ndarray
    confidence_band_upper: np.ndarray
    lstm_weight: float
    prophet_weight: float
    recommended_replicas: int = 0
    scaling_urgency: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "predicted_values": self.predicted_values.tolist(),
            "confidence": round(self.confidence, 4),
            "confidence_band_lower": self.confidence_band_lower.tolist(),
            "confidence_band_upper": self.confidence_band_upper.tolist(),
            "lstm_weight": round(self.lstm_weight, 4),
            "prophet_weight": round(self.prophet_weight, 4),
            "recommended_replicas": self.recommended_replicas,
            "scaling_urgency": self.scaling_urgency,
        }


class EnsembleCombiner:
    """Combines LSTM and Prophet forecasts with adaptive weighting.

    Weight update rule::

        α_lstm ∝ 1 / (lstm_mae + ε)
        α_prophet ∝ 1 / (prophet_mae + ε)
        weights are normalized so α_lstm + α_prophet = 1

    When only one model is available (cold-start) the available model
    receives 100 % weight.
    """

    def __init__(
        self,
        initial_lstm_weight: float = 0.6,
        initial_prophet_weight: float = 0.4,
        confidence_window: int = 30,
    ):
        self._lstm_weight = initial_lstm_weight
        self._prophet_weight = initial_prophet_weight
        self._confidence_window = confidence_window

        # Rolling error buffers
        self._lstm_errors: List[float] = []
        self._prophet_errors: List[float] = []

    # ------------------------------------------------------------------
    # Combining
    # ------------------------------------------------------------------

    def combine(
        self,
        lstm_prediction: Optional[np.ndarray] = None,
        prophet_prediction: Optional[np.ndarray] = None,
        prophet_lower: Optional[np.ndarray] = None,
        prophet_upper: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        """Produce an ensemble prediction.

        Args:
            lstm_prediction: LSTM forecast array (forecast_steps,).
            prophet_prediction: Prophet forecast array.
            prophet_lower: Prophet lower confidence band.
            prophet_upper: Prophet upper confidence band.

        Returns:
            :class:`PredictionResult` with blended predictions and
            confidence metrics.
        """
        if lstm_prediction is not None and prophet_prediction is not None:
            blended = (
                self._lstm_weight * lstm_prediction + self._prophet_weight * prophet_prediction
            )
        elif lstm_prediction is not None:
            blended = lstm_prediction
        elif prophet_prediction is not None:
            blended = prophet_prediction
        else:
            raise ValueError("At least one model prediction is required.")

        # Confidence band — use Prophet's intervals if available, otherwise
        # construct a synthetic band from the blended prediction spread.
        if prophet_lower is not None and prophet_upper is not None:
            lower = prophet_lower
            upper = prophet_upper
        else:
            margin = np.std(blended) * 1.96 if len(blended) > 1 else 0.1
            lower = blended - margin
            upper = blended + margin

        confidence = self._calculate_confidence(blended, lower, upper)

        result = PredictionResult(
            predicted_values=blended.astype(np.float32),
            confidence=confidence,
            confidence_band_lower=lower.astype(np.float32),
            confidence_band_upper=upper.astype(np.float32),
            lstm_weight=self._lstm_weight,
            prophet_weight=self._prophet_weight,
        )

        logger.debug(
            "ensemble_prediction_generated",
            confidence=round(confidence, 4),
            lstm_w=round(self._lstm_weight, 4),
            prophet_w=round(self._prophet_weight, 4),
        )
        return result

    # ------------------------------------------------------------------
    # Weight adaptation
    # ------------------------------------------------------------------

    def update_weights(
        self,
        actual: np.ndarray,
        lstm_pred: Optional[np.ndarray] = None,
        prophet_pred: Optional[np.ndarray] = None,
    ) -> None:
        """Recompute ensemble weights based on recent errors.

        Call this after each prediction cycle with the ground-truth
        values to keep the weights calibrated.

        Args:
            actual: Observed ground-truth values.
            lstm_pred: The LSTM's prediction for comparison.
            prophet_pred: Prophet's prediction for comparison.
        """
        eps = 1e-8

        if lstm_pred is not None:
            mae = float(np.mean(np.abs(actual - lstm_pred)))
            self._lstm_errors.append(mae)
            if len(self._lstm_errors) > self._confidence_window:
                self._lstm_errors = self._lstm_errors[-self._confidence_window :]

        if prophet_pred is not None:
            mae = float(np.mean(np.abs(actual - prophet_pred)))
            self._prophet_errors.append(mae)
            if len(self._prophet_errors) > self._confidence_window:
                self._prophet_errors = self._prophet_errors[-self._confidence_window :]

        if self._lstm_errors and self._prophet_errors:
            lstm_mae = np.mean(self._lstm_errors)
            prophet_mae = np.mean(self._prophet_errors)

            inv_lstm = 1.0 / (lstm_mae + eps)
            inv_prophet = 1.0 / (prophet_mae + eps)
            total = inv_lstm + inv_prophet

            self._lstm_weight = float(inv_lstm / total)
            self._prophet_weight = float(inv_prophet / total)

            logger.info(
                "ensemble_weights_updated",
                lstm_mae=round(float(lstm_mae), 6),
                prophet_mae=round(float(prophet_mae), 6),
                lstm_weight=round(self._lstm_weight, 4),
                prophet_weight=round(self._prophet_weight, 4),
            )

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        prediction: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """Calculate a scalar confidence score in [0, 1].

        Narrower confidence bands → higher confidence.
        """
        band_width = np.mean(upper - lower)
        pred_mean = np.mean(np.abs(prediction)) + 1e-8
        relative_width = band_width / pred_mean

        # Map relative width to a 0-1 confidence score
        # Width of 0 → confidence 1.0; width ≥ 1 → confidence ~0.0
        confidence = float(np.exp(-relative_width))
        return np.clip(confidence, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def lstm_weight(self) -> float:
        return self._lstm_weight

    @property
    def prophet_weight(self) -> float:
        return self._prophet_weight

    def get_combiner_stats(self) -> Dict[str, Any]:
        return {
            "lstm_weight": round(self._lstm_weight, 4),
            "prophet_weight": round(self._prophet_weight, 4),
            "lstm_error_history_size": len(self._lstm_errors),
            "prophet_error_history_size": len(self._prophet_errors),
            "avg_lstm_mae": (
                round(float(np.mean(self._lstm_errors)), 6) if self._lstm_errors else None
            ),
            "avg_prophet_mae": (
                round(float(np.mean(self._prophet_errors)), 6) if self._prophet_errors else None
            ),
        }
