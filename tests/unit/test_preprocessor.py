# ============================================
# K8s PredictScale - Preprocessor Unit Tests
# ============================================

import numpy as np
import pandas as pd
import pytest

from src.preprocessor.cleaner import DataCleaner
from src.preprocessor.feature_engineer import FeatureEngineer
from src.preprocessor.normalizer import Normalizer
from src.preprocessor.pipeline import PreprocessingPipeline


class TestDataCleaner:
    """Tests for DataCleaner."""

    def test_clean_removes_nans(self, small_metrics_df):
        # Inject some NaN values
        df = small_metrics_df.copy()
        df.iloc[3, 0] = np.nan
        df.iloc[7, 1] = np.nan

        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)

        assert cleaned.isna().sum().sum() == 0

    def test_clean_preserves_shape(self, small_metrics_df):
        cleaner = DataCleaner()
        cleaned = cleaner.clean(small_metrics_df)
        assert cleaned.shape[1] == small_metrics_df.shape[1]

    def test_clean_empty_dataframe(self):
        cleaner = DataCleaner()
        result = cleaner.clean(pd.DataFrame())
        assert result.empty

    def test_quality_report(self, small_metrics_df):
        cleaner = DataCleaner()
        report = cleaner.get_quality_report(small_metrics_df)
        assert "cpu_usage" in report
        assert "nulls" in report["cpu_usage"]

    def test_outlier_clipping(self, small_metrics_df):
        df = small_metrics_df.copy()
        df.iloc[0, 0] = 1e6  # Extreme outlier

        cleaner = DataCleaner(outlier_std_threshold=3.0)
        cleaned = cleaner.clean(df)

        assert cleaned.iloc[0, 0] < 1e6


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_engineer_adds_columns(self, sample_metrics_df):
        eng = FeatureEngineer()
        result = eng.engineer(sample_metrics_df)
        # Should have many more columns than the original
        assert result.shape[1] > sample_metrics_df.shape[1]

    def test_time_features_present(self, sample_metrics_df):
        eng = FeatureEngineer()
        result = eng.engineer(sample_metrics_df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "dow_sin" in result.columns
        assert "is_weekend" in result.columns

    def test_rolling_features_present(self, sample_metrics_df):
        eng = FeatureEngineer()
        result = eng.engineer(sample_metrics_df)
        assert "cpu_usage_rolling_mean_5m" in result.columns
        assert "cpu_usage_rolling_std_15m" in result.columns

    def test_lag_features_present(self, sample_metrics_df):
        eng = FeatureEngineer()
        result = eng.engineer(sample_metrics_df)
        assert "lag_cpu_usage_15m" in result.columns
        assert "lag_cpu_usage_60m" in result.columns

    def test_empty_dataframe(self):
        eng = FeatureEngineer()
        result = eng.engineer(pd.DataFrame())
        assert result.empty


class TestNormalizer:
    """Tests for Normalizer."""

    def test_minmax_range(self, small_metrics_df):
        norm = Normalizer(method="minmax")
        result = norm.fit_transform(small_metrics_df)
        numeric = result.select_dtypes(include=[np.number])
        assert numeric.min().min() >= -0.01  # Floating point tolerance
        assert numeric.max().max() <= 1.01

    def test_standard_mean_zero(self, small_metrics_df):
        norm = Normalizer(method="standard")
        result = norm.fit_transform(small_metrics_df)
        means = result.mean()
        # Means should be close to 0 (within tolerance)
        assert all(abs(m) < 0.1 for m in means)

    def test_inverse_transform(self, small_metrics_df):
        norm = Normalizer(method="minmax")
        transformed = norm.fit_transform(small_metrics_df)
        restored = norm.inverse_transform(transformed)
        pd.testing.assert_frame_equal(
            small_metrics_df, restored, atol=1e-4
        )

    def test_inverse_transform_column(self, small_metrics_df):
        norm = Normalizer(method="minmax")
        norm.fit_transform(small_metrics_df)
        original = small_metrics_df["cpu_usage"].values

        # Normalize then inverse just one column
        transformed = norm.transform(small_metrics_df)
        inversed = norm.inverse_transform_column(
            transformed["cpu_usage"].values, "cpu_usage"
        )
        np.testing.assert_allclose(original, inversed, atol=1e-4)

    def test_fit_required_before_transform(self, small_metrics_df):
        norm = Normalizer()
        with pytest.raises(RuntimeError):
            norm.transform(small_metrics_df)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            Normalizer(method="invalid")

    def test_get_params(self, small_metrics_df):
        norm = Normalizer(method="minmax")
        norm.fit_transform(small_metrics_df)
        params = norm.get_params()
        assert params["method"] == "minmax"
        assert params["is_fitted"] is True
        assert "data_min" in params


class TestPreprocessingPipeline:
    """Tests for the integrated pipeline."""

    def test_fit_transform_produces_sequences(self, sample_metrics_df):
        pipeline = PreprocessingPipeline(
            lookback_steps=30,
            forecast_steps=5,
            target_column="cpu_usage",
        )
        X, y = pipeline.fit_transform(sample_metrics_df)

        assert X.ndim == 3
        assert y.ndim == 2
        assert X.shape[1] == 30  # lookback
        assert y.shape[1] == 5   # forecast

    def test_transform_single_sample(self, sample_metrics_df):
        pipeline = PreprocessingPipeline(
            lookback_steps=30,
            forecast_steps=5,
        )
        pipeline.fit_transform(sample_metrics_df)

        # Transform a recent window
        recent = sample_metrics_df.tail(50)
        X = pipeline.transform(recent)
        assert X.shape == (1, 30, X.shape[2])

    def test_pipeline_info(self, sample_metrics_df):
        pipeline = PreprocessingPipeline(lookback_steps=30, forecast_steps=5)
        pipeline.fit_transform(sample_metrics_df)
        info = pipeline.get_pipeline_info()
        assert info["is_fitted"] is True
        assert info["lookback_steps"] == 30
