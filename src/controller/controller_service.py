# ============================================
# K8s PredictScale - Controller Service
# ============================================
# Top-level controller loop that ties metrics
# collection, prediction, decision-making, and
# scaling execution together.
# ============================================

from typing import Any, Dict, Optional

import numpy as np

from src.collector.collector_service import CollectorService
from src.controller.decision_engine import DecisionEngine, ScalingDecision
from src.controller.k8s_scaler import K8sScaler
from src.controller.safety_guard import SafetyGuard
from src.predictor.ensemble import PredictionResult
from src.predictor.predictor_service import PredictorService
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ControllerService:
    """Main control loop: collect → predict → decide → scale.

    This service is invoked periodically (e.g. every 60 s) by the
    scheduler.  Each ``tick()`` call executes one full cycle::

        1. Collect latest metrics from Prometheus.
        2. Run the prediction pipeline.
        3. Make a scaling decision.
        4. Apply safety checks.
        5. Execute scaling (or dry-run).
        6. Emit Prometheus metrics for Grafana.
    """

    def __init__(self, config: AppConfig):
        """Initialize all sub-services from the root config."""
        self._config = config

        self._collector = CollectorService(
            prometheus_config=config.prometheus,
            scaling_config=config.scaling,
        )

        self._predictor = PredictorService(
            prediction_config=config.prediction,
            scaling_config=config.scaling,
        )

        self._decision_engine = DecisionEngine(
            target_cpu_utilization=config.scaling.target_cpu_utilization,
            min_replicas=config.scaling.min_replicas,
            max_replicas=config.scaling.max_replicas,
            confidence_threshold=config.scaling.confidence_threshold,
        )

        self._scaler = K8sScaler(
            namespace=config.scaling.target_namespace,
            deployment=config.scaling.target_deployment,
            dry_run=config.scaling.dry_run,
        )

        self._safety = SafetyGuard(
            cooldown_seconds=config.scaling.cooldown_period,
            max_scale_up_step=config.scaling.scale_up_rate,
            max_scale_down_step=config.scaling.scale_down_rate,
        )

        self._tick_count = 0
        self._last_decision: Optional[ScalingDecision] = None
        self._last_prediction: Optional[PredictionResult] = None

    # ------------------------------------------------------------------
    # Main loop tick
    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """Execute one control-loop cycle.

        Returns:
            Summary dict with collection, prediction, and decision info.
        """
        self._tick_count += 1
        logger.info("control_loop_tick", tick=self._tick_count)

        result: Dict[str, Any] = {"tick": self._tick_count}

        # ---- 1. Collect ----
        try:
            snapshot = self._collector.collect_once()
            result["metrics_collected"] = len(snapshot)
        except Exception as exc:
            logger.error("collection_failed", error=str(exc))
            result["error"] = f"Collection failed: {exc}"
            return result

        # ---- 2. Predict ----
        try:
            recent = self._collector.get_latest_metrics(
                n=self._config.prediction.lookback_steps
            )
            if recent.empty:
                result["skipped"] = "No data in buffer"
                return result

            prediction = self._predictor.predict(recent)
            self._last_prediction = prediction
            result["prediction"] = prediction.to_dict()
        except Exception as exc:
            logger.error("prediction_failed", error=str(exc))
            result["error"] = f"Prediction failed: {exc}"
            return result

        # ---- 3. Decide ----
        current_replicas = self._scaler.get_current_replicas()
        peak = float(np.max(prediction.predicted_values))

        decision = self._decision_engine.decide(
            predicted_peak=peak,
            confidence=prediction.confidence,
            current_replicas=current_replicas,
        )

        # ---- 4. Safety check ----
        decision = self._safety.check(decision)
        result["decision"] = decision.to_dict()

        # ---- 5. Execute ----
        if decision.approved and decision.target_replicas != current_replicas:
            success = self._scaler.scale(decision.target_replicas)
            if success:
                self._safety.record_event(decision)
            result["scaled"] = success
        else:
            result["scaled"] = False

        self._last_decision = decision
        return result

    # ------------------------------------------------------------------
    # Training trigger
    # ------------------------------------------------------------------

    def initial_training(self, hours: int = 24) -> Dict[str, Any]:
        """Backfill historical data and train the models.

        Call this once at startup or when retraining is needed.

        Args:
            hours: Hours of historical data to load.

        Returns:
            Training result summary.
        """
        logger.info("initial_training_started", hours=hours)

        # Backfill metrics
        data = self._collector.collect_range(hours=hours)
        if data.empty:
            return {"error": "No historical data available from Prometheus"}

        # Train models
        result = self._predictor.train(data)
        logger.info("initial_training_complete")
        return result

    # ------------------------------------------------------------------
    # Status & accessors
    # ------------------------------------------------------------------

    @property
    def collector(self) -> CollectorService:
        return self._collector

    @property
    def predictor(self) -> PredictorService:
        return self._predictor

    @property
    def scaler(self) -> K8sScaler:
        return self._scaler

    def get_full_status(self) -> Dict[str, Any]:
        """Return comprehensive system status."""
        return {
            "tick_count": self._tick_count,
            "collector": self._collector.get_buffer_status(),
            "predictor": self._predictor.get_status(),
            "scaler": {
                "dry_run": self._scaler.dry_run,
                "deployment": self._scaler.get_deployment_info(),
            },
            "safety": self._safety.get_guard_status(),
            "last_decision": self._last_decision.to_dict() if self._last_decision else None,
            "prometheus_healthy": self._collector.is_prometheus_healthy(),
        }

    def get_scaling_history(self, limit: int = 20):
        """Return recent scaling event history."""
        return self._safety.get_event_history(limit=limit)
