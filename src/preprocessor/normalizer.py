# ============================================
# K8s PredictScale - Normalizer
# ============================================
# Third stage of the preprocessing pipeline.
# Scales feature values to a consistent range
# suitable for neural-network consumption.
# ============================================

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Normalizer:
    """Fits and applies feature-wise normalization.

    Two strategies are supported:
        - ``minmax`` — scales values to [0, 1].
        - ``standard`` — zero-mean, unit-variance (Z-score).

    The fitted scaler state can be exported and restored so that
    inference uses the same parameters learned during training.
    """

    def __init__(self, method: str = "minmax"):
        """Initialize the normalizer.

        Args:
            method: ``"minmax"`` or ``"standard"``.

        Raises:
            ValueError: If *method* is not recognized.
        """
        if method not in ("minmax", "standard"):
            raise ValueError(f"Unknown normalization method: {method}")

        self._method = method
        self._scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
        self._is_fitted = False
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        """Learn normalization parameters from *df*.

        Args:
            df: Feature DataFrame (numeric columns only).

        Returns:
            ``self`` for chaining.
        """
        numeric = df.select_dtypes(include=[np.number])
        self._feature_names = list(numeric.columns)
        self._scaler.fit(numeric.values)
        self._is_fitted = True
        logger.info(
            "normalizer_fitted",
            method=self._method,
            features=len(self._feature_names),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned normalization to *df*.

        Args:
            df: Feature DataFrame.

        Returns:
            Normalized DataFrame (same index and columns).

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer has not been fitted yet — call fit() first.")

        numeric = df[self._feature_names]
        scaled = self._scaler.transform(numeric.values)
        result = df.copy()
        result[self._feature_names] = scaled
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit and transform in a single call."""
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse the normalization.

        Useful for converting predictions back to interpretable units.

        Args:
            df: Normalized DataFrame.

        Returns:
            De-normalized DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer has not been fitted yet.")

        numeric = df[self._feature_names]
        original = self._scaler.inverse_transform(numeric.values)
        result = df.copy()
        result[self._feature_names] = original
        return result

    def inverse_transform_column(
        self, values: np.ndarray, column_name: str
    ) -> np.ndarray:
        """Inverse-transform a single column's values.

        Constructs a dummy array with all- zero columns except the one
        of interest, transforms, and extracts the result.

        Args:
            values: 1-D array of normalized values.
            column_name: Name of the original column.

        Returns:
            1-D array of de-normalized values.
        """
        if column_name not in self._feature_names:
            raise ValueError(f"Unknown column: {column_name}")

        idx = self._feature_names.index(column_name)
        dummy = np.zeros((len(values), len(self._feature_names)))
        dummy[:, idx] = values.flatten()
        inversed = self._scaler.inverse_transform(dummy)
        return inversed[:, idx]

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Export scaler parameters for persistence."""
        params: Dict[str, Any] = {
            "method": self._method,
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
        }
        if self._is_fitted:
            if self._method == "minmax":
                params["data_min"] = self._scaler.data_min_.tolist()
                params["data_max"] = self._scaler.data_max_.tolist()
                params["scale"] = self._scaler.scale_.tolist()
                params["min_"] = self._scaler.min_.tolist()
            else:
                params["mean"] = self._scaler.mean_.tolist()
                params["var"] = self._scaler.var_.tolist()
                params["scale"] = self._scaler.scale_.tolist()
        return params

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)
