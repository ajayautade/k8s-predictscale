# ============================================
# K8s PredictScale - Data Cleaner
# ============================================
# First stage of the preprocessing pipeline.
# Handles missing values, outliers, and data
# validation before feature engineering.
# ============================================

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Cleans and validates raw metric DataFrames.

    Pipeline: validate → interpolate NaN → clip outliers → fill
    remaining gaps.
    """

    def __init__(
        self,
        max_gap_seconds: int = 300,
        outlier_std_threshold: float = 4.0,
        min_required_ratio: float = 0.5,
    ):
        """Initialize the cleaner.

        Args:
            max_gap_seconds: Maximum gap (in seconds) to interpolate.
            outlier_std_threshold: Values beyond this many standard
                deviations from the mean are clipped.
            min_required_ratio: Minimum fraction of non-null values
                required for a column to be considered valid.
        """
        self._max_gap = max_gap_seconds
        self._outlier_std = outlier_std_threshold
        self._min_ratio = min_required_ratio

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full cleaning pipeline on *df*.

        Args:
            df: Raw metrics DataFrame indexed by timestamp.

        Returns:
            Cleaned DataFrame (same shape, no NaN values).
        """
        if df.empty:
            logger.warning("cleaner_received_empty_dataframe")
            return df

        original_shape = df.shape
        df = self._validate_index(df)
        df = self._drop_invalid_columns(df)
        df = self._interpolate_gaps(df)
        df = self._clip_outliers(df)
        df = self._fill_remaining(df)

        logger.info(
            "data_cleaning_complete",
            original_shape=original_shape,
            cleaned_shape=df.shape,
            nan_remaining=int(df.isna().sum().sum()),
        )
        return df

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _validate_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the index is a proper datetime and sorted."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    def _drop_invalid_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns where most values are missing."""
        valid = []
        for col in df.columns:
            ratio = df[col].notna().mean()
            if ratio >= self._min_ratio:
                valid.append(col)
            else:
                logger.warning(
                    "column_dropped_insufficient_data",
                    column=col,
                    valid_ratio=round(ratio, 3),
                )
        return df[valid]

    def _interpolate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Linearly interpolate small gaps; leave large ones as NaN."""
        df = df.interpolate(method="time", limit_direction="both")
        return df

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip values beyond ±std_threshold standard deviations."""
        for col in df.select_dtypes(include=[np.number]).columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower = mean - self._outlier_std * std
                upper = mean + self._outlier_std * std
                clipped = df[col].clip(lower=lower, upper=upper)
                n_clipped = int((df[col] != clipped).sum())
                if n_clipped:
                    logger.debug(
                        "outliers_clipped",
                        column=col,
                        count=n_clipped,
                    )
                df[col] = clipped
        return df

    def _fill_remaining(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill then back-fill any remaining NaN values."""
        df = df.ffill().bfill()
        # As a last resort, fill with column medians
        for col in df.columns:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_quality_report(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Generate a per-column quality report.

        Returns:
            Dict mapping column names to quality statistics.
        """
        report: Dict[str, Dict] = {}
        for col in df.columns:
            series = df[col]
            report[col] = {
                "count": int(series.count()),
                "nulls": int(series.isna().sum()),
                "null_pct": round(float(series.isna().mean()) * 100, 2),
                "mean": round(float(series.mean()), 4) if series.dtype != object else None,
                "std": round(float(series.std()), 4) if series.dtype != object else None,
                "min": round(float(series.min()), 4) if series.dtype != object else None,
                "max": round(float(series.max()), 4) if series.dtype != object else None,
            }
        return report
