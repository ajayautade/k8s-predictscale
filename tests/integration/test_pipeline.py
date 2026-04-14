# ============================================
# K8s PredictScale - Integration Tests
# ============================================
# Tests the preprocessor → predictor pipeline
# end-to-end using synthetic data.
# ============================================

import numpy as np
import pandas as pd
import pytest

from src.predictor.ensemble import EnsembleCombiner
from src.preprocessor.pipeline import PreprocessingPipeline


class TestPreprocessorPredictorIntegration:
    """Integration test for the data pipeline flow."""

    @pytest.fixture
    def large_df(self):
        """Generate 500 rows of synthetic metric data."""
        n = 500
        ts = pd.date_range("2026-03-01", periods=n, freq="1min")
        np.random.seed(42)

        return pd.DataFrame(
            {
                "cpu_usage": 0.3
                + 0.2 * np.sin(np.linspace(0, 8 * np.pi, n))
                + np.random.normal(0, 0.02, n),
                "memory_usage": 400e6 + np.random.normal(0, 30e6, n),
                "request_rate": 120
                + 40 * np.sin(np.linspace(0, 8 * np.pi, n))
                + np.random.normal(0, 5, n),
                "response_latency_p99": 0.05 + np.random.exponential(0.01, n),
                "error_rate": np.random.poisson(0.3, n).astype(float),
            },
            index=ts,
        )

    def test_pipeline_produces_valid_sequences(self, large_df):
        pipeline = PreprocessingPipeline(
            lookback_steps=30,
            forecast_steps=5,
            target_column="cpu_usage",
        )

        X, y = pipeline.fit_transform(large_df)

        # Shape checks
        assert X.ndim == 3
        assert y.ndim == 2
        assert X.shape[1] == 30
        assert y.shape[1] == 5
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0

        # Value checks — no NaN or Inf
        assert np.isfinite(X).all()
        assert np.isfinite(y).all()

    def test_transform_for_inference(self, large_df):
        pipeline = PreprocessingPipeline(
            lookback_steps=30,
            forecast_steps=5,
        )
        pipeline.fit_transform(large_df)

        # Simulate real-time: last 60 rows
        recent = large_df.tail(60)
        X_infer = pipeline.transform(recent)

        assert X_infer.shape == (1, 30, X_infer.shape[2])
        assert np.isfinite(X_infer).all()

    def test_ensemble_with_synthetic_predictions(self):
        combiner = EnsembleCombiner()

        lstm_pred = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
        prophet_pred = np.array([0.38, 0.42, 0.48, 0.53, 0.58])
        prophet_lower = prophet_pred - 0.05
        prophet_upper = prophet_pred + 0.05

        result = combiner.combine(
            lstm_prediction=lstm_pred,
            prophet_prediction=prophet_pred,
            prophet_lower=prophet_lower,
            prophet_upper=prophet_upper,
        )

        assert len(result.predicted_values) == 5
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.confidence_band_lower) == 5
        assert len(result.confidence_band_upper) == 5

    def test_weight_adaptation_over_cycles(self):
        combiner = EnsembleCombiner(initial_lstm_weight=0.5, initial_prophet_weight=0.5)

        # Simulate 10 cycles where LSTM is consistently better
        for _ in range(10):
            actual = np.random.uniform(0.3, 0.6, 5)
            lstm_pred = actual + np.random.normal(0, 0.01, 5)
            prophet_pred = actual + np.random.normal(0, 0.1, 5)

            combiner.update_weights(
                actual=actual,
                lstm_pred=lstm_pred,
                prophet_pred=prophet_pred,
            )

        # LSTM should now have higher weight
        assert combiner.lstm_weight > 0.6
