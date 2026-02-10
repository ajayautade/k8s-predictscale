# ============================================
# K8s PredictScale - Feature Engineer
# ============================================
# Second stage of the preprocessing pipeline.
# Extracts temporal, statistical, and domain-
# specific features from cleaned metric data.
# ============================================

from typing import List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Derives ML-ready features from cleaned time-series metrics.

    Features produced:
        - Rolling aggregates (mean, std) at multiple windows
        - Rate-of-change (first derivative)
        - Lag features
        - Exponential moving averages (EMA)
        - Cyclical time encodings (hour, day-of-week)
    """

    # Source columns we operate on (those created by the collector)
    PRIMARY_METRICS = [
        "cpu_usage",
        "memory_usage",
        "request_rate",
        "response_latency_p99",
        "error_rate",
    ]

    ROLLING_WINDOWS = [5, 15, 60]  # minutes

    def __init__(self, primary_metrics: List[str] | None = None):
        """Initialize the feature engineer.

        Args:
            primary_metrics: Override the default list of source metrics.
        """
        self._primary = primary_metrics or self.PRIMARY_METRICS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature-engineering pipeline.

        Args:
            df: Cleaned DataFrame indexed by timestamp.

        Returns:
            Enriched DataFrame with new feature columns appended.
        """
        if df.empty:
            logger.warning("feature_engineer_received_empty_dataframe")
            return df

        df = df.copy()
        present = [m for m in self._primary if m in df.columns]

        df = self._add_rolling_features(df, present)
        df = self._add_rate_of_change(df, present)
        df = self._add_lag_features(df, present)
        df = self._add_ema(df, present)
        df = self._add_time_features(df)

        # Drop rows that became NaN due to rolling/lag operations
        initial_len = len(df)
        df.dropna(inplace=True)
        dropped = initial_len - len(df)

        logger.info(
            "feature_engineering_complete",
            total_features=len(df.columns),
            rows=len(df),
            rows_dropped=dropped,
        )
        return df

    # ------------------------------------------------------------------
    # Feature generators
    # ------------------------------------------------------------------

    def _add_rolling_features(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """Add rolling mean and std for each window size."""
        for col in columns:
            for window in self.ROLLING_WINDOWS:
                df[f"{col}_rolling_mean_{window}m"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                df[f"{col}_rolling_std_{window}m"] = (
                    df[col].rolling(window=window, min_periods=1).std().fillna(0)
                )
        return df

    def _add_rate_of_change(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """Add first-order difference (rate of change)."""
        for col in columns:
            df[f"{col}_rate_of_change"] = df[col].diff().fillna(0)
        return df

    def _add_lag_features(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """Add lagged values at 15-min and 60-min offsets."""
        for col in columns:
            df[f"lag_{col}_15m"] = df[col].shift(15)
            df[f"lag_{col}_60m"] = df[col].shift(60)
        return df

    def _add_ema(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Add exponential moving average (span = 30 periods)."""
        for col in columns:
            df[f"{col}_ema_30"] = df[col].ewm(span=30, adjust=False).mean()
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode timestamp as cyclical features.

        Uses sin/cos encoding so that 23:00 and 00:00 are close in
        feature space.
        """
        idx = df.index

        # Hour-of-day (0-23)
        hour = idx.hour + idx.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # Day-of-week (0=Mon … 6=Sun)
        dow = idx.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        # Boolean weekday vs weekend
        df["is_weekend"] = (dow >= 5).astype(int)

        return df

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Return an ordered list of all feature column names.

        Useful for documenting the feature vector fed to the LSTM.
        """
        return list(df.columns)
