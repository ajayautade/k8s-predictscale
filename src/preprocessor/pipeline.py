# ============================================
# K8s PredictScale - Preprocessing Pipeline
# ============================================
# Orchestrates the three preprocessing stages
# (clean → engineer → normalize) and prepares
# windowed sequences for LSTM input.
# ============================================

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocessor.cleaner import DataCleaner
from src.preprocessor.feature_engineer import FeatureEngineer
from src.preprocessor.normalizer import Normalizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """End-to-end preprocessing: raw metrics → LSTM-ready sequences.

    The pipeline runs three stages in order:

    1. **Clean** — handle NaN, outliers, data validation.
    2. **Feature Engineer** — rolling stats, lags, time encodings.
    3. **Normalize** — scale to [0, 1] via MinMaxScaler.

    After normalization the data is sliced into overlapping windows
    of shape ``(lookback_steps, n_features)`` that the LSTM consumes.
    """

    def __init__(
        self,
        lookback_steps: int = 60,
        forecast_steps: int = 10,
        target_column: str = "cpu_usage",
        normalization_method: str = "minmax",
    ):
        """Initialize the pipeline.

        Args:
            lookback_steps: Number of historical time steps per sample.
            forecast_steps: Number of future steps to predict.
            target_column: The metric column used as the prediction
                target.
            normalization_method: ``"minmax"`` or ``"standard"``.
        """
        self._lookback = lookback_steps
        self._forecast = forecast_steps
        self._target_col = target_column

        self._cleaner = DataCleaner()
        self._engineer = FeatureEngineer()
        self._normalizer = Normalizer(method=normalization_method)

        self._is_fitted = False

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the pipeline on *df* and return training arrays.

        Args:
            df: Raw metrics DataFrame.

        Returns:
            ``(X, y)`` where ``X`` has shape
            ``(samples, lookback, features)`` and ``y`` has shape
            ``(samples, forecast_steps)``.
        """
        processed = self._run_stages(df, fit=True)
        X, y = self._create_sequences(processed)
        logger.info(
            "pipeline_fit_transform_complete",
            X_shape=X.shape,
            y_shape=y.shape,
        )
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the already-fitted pipeline.

        Args:
            df: Raw metrics DataFrame.

        Returns:
            Array of shape ``(samples, lookback, features)`` or a
            single sample ``(1, lookback, features)`` for real-time
            inference.

        Raises:
            RuntimeError: If :meth:`fit_transform` was not called first.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted — call fit_transform() first.")

        processed = self._run_stages(df, fit=False)

        if len(processed) < self._lookback:
            logger.warning(
                "insufficient_data_for_sequence",
                rows=len(processed),
                required=self._lookback,
            )
            # Pad with zeros if not enough data
            padding = pd.DataFrame(
                np.zeros((self._lookback - len(processed), processed.shape[1])),
                columns=processed.columns,
            )
            processed = pd.concat([padding, processed], ignore_index=True)

        # Return the latest window as a single sample
        values = processed.values[-self._lookback:]
        return values.reshape(1, self._lookback, -1)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_stages(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Execute clean → engineer → normalize.

        Args:
            df: Raw DataFrame.
            fit: If ``True``, fit the normalizer on this data.

        Returns:
            Fully processed DataFrame.
        """
        cleaned = self._cleaner.clean(df)
        engineered = self._engineer.engineer(cleaned)

        if fit:
            normalized = self._normalizer.fit_transform(engineered)
            self._is_fitted = True
        else:
            normalized = self._normalizer.transform(engineered)

        return normalized

    def _create_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Slice a DataFrame into overlapping (X, y) windows.

        Args:
            df: Fully preprocessed DataFrame.

        Returns:
            ``(X, y)`` numpy arrays.
        """
        values = df.values
        target_idx = list(df.columns).index(self._target_col) if self._target_col in df.columns else 0

        X_list, y_list = [], []
        total = len(values) - self._lookback - self._forecast + 1

        for i in range(total):
            X_list.append(values[i: i + self._lookback])
            y_list.append(
                values[
                    i + self._lookback: i + self._lookback + self._forecast,
                    target_idx,
                ]
            )

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        return X, y

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def normalizer(self) -> Normalizer:
        """Access the fitted normalizer (needed for inverse transforms)."""
        return self._normalizer

    @property
    def target_column(self) -> str:
        return self._target_col

    @property
    def lookback_steps(self) -> int:
        return self._lookback

    @property
    def forecast_steps(self) -> int:
        return self._forecast

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Return diagnostic metadata about the pipeline."""
        return {
            "lookback_steps": self._lookback,
            "forecast_steps": self._forecast,
            "target_column": self._target_col,
            "is_fitted": self._is_fitted,
            "normalizer_fitted": self._normalizer.is_fitted,
            "feature_count": len(self._normalizer.feature_names) if self._is_fitted else 0,
        }
