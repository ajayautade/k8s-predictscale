# ============================================
# K8s PredictScale - Decision Engine
# ============================================
# Core scaling logic: converts a PredictionResult
# into a concrete scaling decision (how many
# replicas, scale-up or scale-down).
# ============================================

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScaleDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class ScalingDecision:
    """Represents a single scaling decision.

    Attributes:
        timestamp: When this decision was made.
        current_replicas: Current pod count in the deployment.
        target_replicas: Desired pod count after scaling.
        direction: up / down / none.
        reason: Human-readable explanation of the decision.
        confidence: Prediction confidence that triggered this.
        predicted_peak: The peak value the model predicted.
        approved: Whether the decision passed safety checks.
    """

    timestamp: str
    current_replicas: int
    target_replicas: int
    direction: ScaleDirection
    reason: str
    confidence: float
    predicted_peak: float
    approved: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_replicas": self.current_replicas,
            "target_replicas": self.target_replicas,
            "direction": self.direction.value,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "predicted_peak": round(self.predicted_peak, 4),
            "approved": self.approved,
        }


class DecisionEngine:
    """Stateless engine that produces scaling decisions.

    The engine applies several policies:
        1. **Target utilization** — how many pods needed for predicted load.
        2. **Confidence threshold** — ignore low-confidence predictions.
        3. **Dead-band** — don't scale for tiny differences.
        4. **Min / max bounds** — enforce replica guardrails.
    """

    def __init__(
        self,
        target_cpu_utilization: float = 0.7,
        min_replicas: int = 2,
        max_replicas: int = 50,
        confidence_threshold: float = 0.7,
        scale_dead_band: int = 1,
    ):
        """Initialize the decision engine.

        Args:
            target_cpu_utilization: Per-pod target CPU utilization.
            min_replicas: Absolute lower bound on replicas.
            max_replicas: Absolute upper bound on replicas.
            confidence_threshold: Minimum prediction confidence to act.
            scale_dead_band: If the difference between current and target
                replicas is within this band, don't scale (avoids thrashing).
        """
        self._target_util = target_cpu_utilization
        self._min = min_replicas
        self._max = max_replicas
        self._confidence_thr = confidence_threshold
        self._dead_band = scale_dead_band

    def decide(
        self,
        predicted_peak: float,
        confidence: float,
        current_replicas: int,
    ) -> ScalingDecision:
        """Produce a scaling decision.

        Args:
            predicted_peak: The highest predicted load value.
            confidence: Ensemble confidence score [0, 1].
            current_replicas: Current replica count from K8s.

        Returns:
            A :class:`ScalingDecision` indicating what to do.
        """
        now = datetime.utcnow().isoformat()

        # -- Check confidence -------------------------------------------
        if confidence < self._confidence_thr:
            return ScalingDecision(
                timestamp=now,
                current_replicas=current_replicas,
                target_replicas=current_replicas,
                direction=ScaleDirection.NONE,
                reason=f"Prediction confidence ({confidence:.2f}) below threshold ({self._confidence_thr})",
                confidence=confidence,
                predicted_peak=predicted_peak,
                approved=False,
            )

        # -- Compute target replicas ------------------------------------
        raw_target = (
            math.ceil(predicted_peak / self._target_util)
            if self._target_util > 0
            else current_replicas
        )
        clamped = max(self._min, min(self._max, raw_target))

        # -- Dead-band check --------------------------------------------
        diff = clamped - current_replicas
        if abs(diff) <= self._dead_band:
            return ScalingDecision(
                timestamp=now,
                current_replicas=current_replicas,
                target_replicas=current_replicas,
                direction=ScaleDirection.NONE,
                reason=f"Target ({clamped}) within dead-band of current ({current_replicas})",
                confidence=confidence,
                predicted_peak=predicted_peak,
            )

        # -- Build decision ---------------------------------------------
        direction = ScaleDirection.UP if clamped > current_replicas else ScaleDirection.DOWN
        reason = (
            f"Predicted peak {predicted_peak:.2f} requires {clamped} replicas "
            f"(target util {self._target_util:.0%}), currently {current_replicas}"
        )

        decision = ScalingDecision(
            timestamp=now,
            current_replicas=current_replicas,
            target_replicas=clamped,
            direction=direction,
            reason=reason,
            confidence=confidence,
            predicted_peak=predicted_peak,
        )

        logger.info(
            "scaling_decision_made",
            direction=direction.value,
            current=current_replicas,
            target=clamped,
            confidence=round(confidence, 4),
        )
        return decision
