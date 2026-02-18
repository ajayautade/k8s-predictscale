# ============================================
# K8s PredictScale - Prophet Model
# ============================================
# Wrapper around Facebook Prophet for seasonal
# time-series forecasting.  Used as a baseline
# and ensemble partner to the LSTM model.
# ============================================

import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover
    Prophet = None  # type: ignore[assignment, misc]


class ProphetModel:
    """Prophet-based time-series forecaster.

    Prophet excels at capturing daily/weekly seasonality and is more
    robust than LSTM when historical data is limited (cold-start).
    """

    def __init__(
        self,
        forecast_steps: int = 10,
        frequency: str = "1min",
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
    ):
        """Initialize the Prophet wrapper.

        Args:
            forecast_steps: Number of future periods to forecast.
            frequency: Pandas frequency string (``"1min"``, ``"5min"``).
            yearly_seasonality: Enable yearly seasonality component.
            weekly_seasonality: Enable weekly seasonality component.
            daily_seasonality: Enable daily seasonality component.
        """
        self._forecast_steps = forecast_steps
        self._frequency = frequency
        self._yearly = yearly_seasonality
        self._weekly = weekly_seasonality
        self._daily = daily_seasonality

        self._model: Optional[Any] = None
        self._is_trained = False
        self._last_mae: Optional[float] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        target_column: str = "cpu_usage",
    ) -> Dict[str, Any]:
        """Fit Prophet on historical data.

        Args:
            df: DataFrame indexed by datetime with a *target_column*.
            target_column: The metric to forecast.

        Returns:
            Training diagnostics dict.
        """
        if Prophet is None:
            raise ImportError("Prophet is required but not installed.")

        # Prophet expects columns named ``ds`` and ``y``
        prophet_df = pd.DataFrame({
            "ds": df.index,
            "y": df[target_column].values,
        })

        self._model = Prophet(
            yearly_seasonality=self._yearly,
            weekly_seasonality=self._weekly,
            daily_seasonality=self._daily,
            changepoint_prior_scale=0.05,
            seasonality_mode="multiplicative",
        )

        # Suppress Prophet's verbose logging
        import logging as _logging
        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

        self._model.fit(prophet_df)
        self._is_trained = True

        # In-sample MAE for diagnostics
        in_sample = self._model.predict(prophet_df)
        residuals = np.abs(in_sample["yhat"].values - prophet_df["y"].values)
        self._last_mae = float(np.mean(residuals))

        logger.info(
            "prophet_training_complete",
            data_points=len(prophet_df),
            in_sample_mae=round(self._last_mae, 6),
        )

        return {
            "data_points": len(prophet_df),
            "in_sample_mae": self._last_mae,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, last_timestamp: Optional[pd.Timestamp] = None) -> np.ndarray:
        """Forecast the next *forecast_steps* periods.

        Args:
            last_timestamp: The timestamp of the latest known data
                point.  If ``None``, uses ``pd.Timestamp.utcnow()``.

        Returns:
            1-D array of shape ``(forecast_steps,)``.
        """
        if self._model is None or not self._is_trained:
            raise RuntimeError("Prophet model is not trained yet.")

        future = self._model.make_future_dataframe(
            periods=self._forecast_steps,
            freq=self._frequency,
            include_history=False,
        )

        if last_timestamp is not None:
            # Shift the future dates relative to the last known timestamp
            future["ds"] = pd.date_range(
                start=last_timestamp + pd.Timedelta(self._frequency),
                periods=self._forecast_steps,
                freq=self._frequency,
            )

        forecast = self._model.predict(future)
        predictions = forecast["yhat"].values.astype(np.float32)
        return predictions

    def predict_with_intervals(
        self, last_timestamp: Optional[pd.Timestamp] = None
    ) -> Dict[str, np.ndarray]:
        """Forecast with uncertainty intervals.

        Returns:
            Dict with ``yhat``, ``yhat_lower``, ``yhat_upper`` arrays.
        """
        if self._model is None or not self._is_trained:
            raise RuntimeError("Prophet model is not trained yet.")

        future = self._model.make_future_dataframe(
            periods=self._forecast_steps,
            freq=self._frequency,
            include_history=False,
        )

        if last_timestamp is not None:
            future["ds"] = pd.date_range(
                start=last_timestamp + pd.Timedelta(self._frequency),
                periods=self._forecast_steps,
                freq=self._frequency,
            )

        forecast = self._model.predict(future)
        return {
            "yhat": forecast["yhat"].values.astype(np.float32),
            "yhat_lower": forecast["yhat_lower"].values.astype(np.float32),
            "yhat_upper": forecast["yhat_upper"].values.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the fitted Prophet model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "prophet_model.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("prophet_model_saved", path=filepath)

    def load(self, path: str) -> None:
        """Load a pickled Prophet model."""
        filepath = os.path.join(path, "prophet_model.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file at {filepath}")
        with open(filepath, "rb") as f:
            self._model = pickle.load(f)  # noqa: S301
        self._is_trained = True
        logger.info("prophet_model_loaded", path=filepath)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "type": "Prophet",
            "is_trained": self._is_trained,
            "forecast_steps": self._forecast_steps,
            "frequency": self._frequency,
            "last_mae": self._last_mae,
            "seasonality": {
                "daily": self._daily,
                "weekly": self._weekly,
                "yearly": self._yearly,
            },
        }
