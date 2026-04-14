# ============================================
# K8s PredictScale - Predictor Unit Tests
# ============================================

import numpy as np
import pytest

from src.predictor.ensemble import EnsembleCombiner, PredictionResult


class TestEnsembleCombiner:
    """Tests for the ensemble combiner."""

    def test_combine_both_models(self):
        combiner = EnsembleCombiner(initial_lstm_weight=0.6, initial_prophet_weight=0.4)
        lstm_pred = np.array([1.0, 2.0, 3.0])
        prophet_pred = np.array([1.5, 2.5, 3.5])

        result = combiner.combine(
            lstm_prediction=lstm_pred,
            prophet_prediction=prophet_pred,
        )

        assert isinstance(result, PredictionResult)
        expected = 0.6 * lstm_pred + 0.4 * prophet_pred
        np.testing.assert_allclose(result.predicted_values, expected, atol=1e-5)

    def test_combine_lstm_only(self):
        combiner = EnsembleCombiner()
        pred = np.array([1.0, 2.0, 3.0])
        result = combiner.combine(lstm_prediction=pred)
        np.testing.assert_array_equal(result.predicted_values, pred)

    def test_combine_prophet_only(self):
        combiner = EnsembleCombiner()
        pred = np.array([1.0, 2.0, 3.0])
        result = combiner.combine(prophet_prediction=pred)
        np.testing.assert_array_equal(result.predicted_values, pred)

    def test_combine_no_models_raises(self):
        combiner = EnsembleCombiner()
        with pytest.raises(ValueError):
            combiner.combine()

    def test_confidence_score_range(self):
        combiner = EnsembleCombiner()
        result = combiner.combine(
            lstm_prediction=np.array([1.0, 1.1, 1.2]),
            prophet_prediction=np.array([1.0, 1.1, 1.2]),
            prophet_lower=np.array([0.8, 0.9, 1.0]),
            prophet_upper=np.array([1.2, 1.3, 1.4]),
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_update_weights(self):
        combiner = EnsembleCombiner(initial_lstm_weight=0.5, initial_prophet_weight=0.5)
        actual = np.array([1.0, 2.0, 3.0])

        # LSTM prediction is much better
        combiner.update_weights(
            actual=actual,
            lstm_pred=np.array([1.01, 2.01, 3.01]),
            prophet_pred=np.array([1.5, 2.5, 3.5]),
        )

        assert combiner.lstm_weight > combiner.prophet_weight

    def test_to_dict(self):
        combiner = EnsembleCombiner()
        result = combiner.combine(lstm_prediction=np.array([1.0, 2.0]))
        d = result.to_dict()
        assert "predicted_values" in d
        assert "confidence" in d
        assert isinstance(d["predicted_values"], list)

    def test_combiner_stats(self):
        combiner = EnsembleCombiner()
        stats = combiner.get_combiner_stats()
        assert "lstm_weight" in stats
        assert "prophet_weight" in stats


class TestModelManager:
    """Tests for model versioning."""

    def test_create_and_list_versions(self, tmp_path):
        from src.predictor.model_manager import ModelManager

        mgr = ModelManager(base_path=str(tmp_path / "models"))
        mgr.create_version(metrics={"mae": 0.05})
        mgr.create_version(metrics={"mae": 0.04})

        versions = mgr.list_versions()
        assert len(versions) == 2

    def test_promote_version(self, tmp_path):
        from src.predictor.model_manager import ModelManager

        mgr = ModelManager(base_path=str(tmp_path / "models"))
        v1 = mgr.create_version()
        mgr.promote_version(v1.version)

        active = mgr.get_active_version()
        assert active is not None
        assert active.version == v1.version

    def test_should_retrain_no_active(self, tmp_path):
        from src.predictor.model_manager import ModelManager

        mgr = ModelManager(base_path=str(tmp_path / "models"))
        assert mgr.should_retrain(current_mae=0.1) is True

    def test_should_retrain_high_mae(self, tmp_path):
        from src.predictor.model_manager import ModelManager

        mgr = ModelManager(base_path=str(tmp_path / "models"))
        v = mgr.create_version()
        mgr.promote_version(v.version)

        assert mgr.should_retrain(current_mae=0.5, mae_threshold=0.2) is True
        assert mgr.should_retrain(current_mae=0.1, mae_threshold=0.2) is False

    def test_cleanup_old_versions(self, tmp_path):
        from src.predictor.model_manager import ModelManager

        mgr = ModelManager(base_path=str(tmp_path / "models"))
        for i in range(10):
            mgr.create_version()
        mgr.promote_version(mgr.list_versions()[-1]["version"])

        removed = mgr.cleanup_old_versions(keep=3)
        assert removed > 0
        assert len(mgr.list_versions()) <= 3
