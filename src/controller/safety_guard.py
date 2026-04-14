# ============================================
# K8s PredictScale - Safety Guard
# ============================================
# Applies rate-limiting, cooldown periods, and
# sanity checks on top of scaling decisions to
# prevent runaway scaling.
# ============================================

import time
from typing import Any, Dict, List

from src.controller.decision_engine import ScaleDirection, ScalingDecision
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SafetyGuard:
    """Post-decision safety layer for scaling operations.

    Enforced rules:
        1. **Cooldown** — minimum seconds between consecutive scaling actions.
        2. **Rate-limiting** — max pods added/removed per scaling event.
        3. **Event history** — keeps a log of recent actions for alerting.
    """

    def __init__(
        self,
        cooldown_seconds: int = 120,
        max_scale_up_step: int = 5,
        max_scale_down_step: int = 2,
        max_history: int = 100,
    ):
        """Initialize the safety guard.

        Args:
            cooldown_seconds: Minimum gap between actual scaling events.
            max_scale_up_step: Maximum pods to add in one event.
            max_scale_down_step: Maximum pods to remove in one event.
            max_history: Number of events to keep in the log.
        """
        self._cooldown = cooldown_seconds
        self._max_up = max_scale_up_step
        self._max_down = max_scale_down_step
        self._max_history = max_history

        self._last_scale_time: float = 0.0
        self._event_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check(self, decision: ScalingDecision) -> ScalingDecision:
        """Apply safety checks and return a (possibly modified) decision.

        The returned decision's ``target_replicas`` may be clamped, and
        ``approved`` set to ``False`` if cooldown hasn't elapsed.

        Args:
            decision: Raw decision from the :class:`DecisionEngine`.

        Returns:
            Adjusted :class:`ScalingDecision`.
        """
        if decision.direction == ScaleDirection.NONE:
            return decision

        # -- Cooldown --
        elapsed = time.time() - self._last_scale_time
        if elapsed < self._cooldown:
            remaining = round(self._cooldown - elapsed, 1)
            logger.info(
                "scaling_blocked_cooldown",
                remaining_seconds=remaining,
            )
            decision.approved = False
            decision.reason += f" | BLOCKED: cooldown ({remaining}s remaining)"
            return decision

        # -- Rate limiting --
        diff = decision.target_replicas - decision.current_replicas

        if diff > 0 and diff > self._max_up:
            clamped = decision.current_replicas + self._max_up
            logger.info(
                "scale_up_rate_limited",
                requested=decision.target_replicas,
                clamped=clamped,
            )
            decision.target_replicas = clamped
            decision.reason += f" | Rate-limited: capped at +{self._max_up}"

        elif diff < 0 and abs(diff) > self._max_down:
            clamped = decision.current_replicas - self._max_down
            logger.info(
                "scale_down_rate_limited",
                requested=decision.target_replicas,
                clamped=clamped,
            )
            decision.target_replicas = clamped
            decision.reason += f" | Rate-limited: capped at -{self._max_down}"

        decision.approved = True
        return decision

    def record_event(self, decision: ScalingDecision) -> None:
        """Log a scaling event after it's been executed.

        Updates the cooldown timer and appends to the event history.
        """
        self._last_scale_time = time.time()
        self._event_history.append(decision.to_dict())

        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

        logger.info(
            "scaling_event_recorded",
            direction=decision.direction.value,
            target=decision.target_replicas,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_event_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most-recent scaling events."""
        return self._event_history[-limit:]

    def seconds_until_ready(self) -> float:
        """Return seconds remaining in the cooldown period (0 if ready)."""
        elapsed = time.time() - self._last_scale_time
        remaining = self._cooldown - elapsed
        return max(0.0, remaining)

    def get_guard_status(self) -> Dict[str, Any]:
        return {
            "cooldown_seconds": self._cooldown,
            "max_scale_up_step": self._max_up,
            "max_scale_down_step": self._max_down,
            "seconds_until_ready": round(self.seconds_until_ready(), 1),
            "total_events": len(self._event_history),
        }
