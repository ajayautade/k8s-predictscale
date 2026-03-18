# ============================================
# K8s PredictScale - Controller Unit Tests
# ============================================

import time

import numpy as np
import pytest

from src.controller.decision_engine import (
    DecisionEngine,
    ScaleDirection,
    ScalingDecision,
)
from src.controller.safety_guard import SafetyGuard


class TestDecisionEngine:
    """Tests for the scaling decision engine."""

    def test_scale_up_decision(self):
        engine = DecisionEngine(
            target_cpu_utilization=0.7,
            min_replicas=2,
            max_replicas=50,
            confidence_threshold=0.5,
        )
        decision = engine.decide(
            predicted_peak=3.5,  # needs ceil(3.5/0.7) = 5 replicas
            confidence=0.9,
            current_replicas=2,
        )
        assert decision.direction == ScaleDirection.UP
        assert decision.target_replicas == 5
        assert decision.approved is True

    def test_scale_down_decision(self):
        engine = DecisionEngine(
            target_cpu_utilization=0.7,
            min_replicas=2,
            max_replicas=50,
            confidence_threshold=0.5,
            scale_dead_band=0,
        )
        decision = engine.decide(
            predicted_peak=1.4,  # needs ceil(1.4/0.7) = 2 replicas
            confidence=0.9,
            current_replicas=5,
        )
        assert decision.direction == ScaleDirection.DOWN
        assert decision.target_replicas == 2

    def test_low_confidence_blocks_scaling(self):
        engine = DecisionEngine(confidence_threshold=0.8)
        decision = engine.decide(
            predicted_peak=5.0,
            confidence=0.3,
            current_replicas=2,
        )
        assert decision.direction == ScaleDirection.NONE
        assert decision.approved is False

    def test_dead_band_prevents_small_changes(self):
        engine = DecisionEngine(
            target_cpu_utilization=0.7,
            scale_dead_band=2,
            confidence_threshold=0.5,
        )
        decision = engine.decide(
            predicted_peak=2.1,  # needs 3, current is 2 → diff=1 ≤ dead_band=2
            confidence=0.9,
            current_replicas=2,
        )
        assert decision.direction == ScaleDirection.NONE

    def test_min_replicas_enforced(self):
        engine = DecisionEngine(
            target_cpu_utilization=0.7,
            min_replicas=3,
            confidence_threshold=0.5,
            scale_dead_band=0,
        )
        decision = engine.decide(
            predicted_peak=0.1,  # needs 1 replica, but min is 3
            confidence=0.9,
            current_replicas=5,
        )
        assert decision.target_replicas >= 3

    def test_max_replicas_enforced(self):
        engine = DecisionEngine(
            target_cpu_utilization=0.7,
            max_replicas=10,
            confidence_threshold=0.5,
            scale_dead_band=0,
        )
        decision = engine.decide(
            predicted_peak=100.0,
            confidence=0.9,
            current_replicas=5,
        )
        assert decision.target_replicas <= 10


class TestSafetyGuard:
    """Tests for the safety guard."""

    def test_rate_limit_scale_up(self):
        guard = SafetyGuard(
            cooldown_seconds=0,
            max_scale_up_step=3,
        )
        decision = ScalingDecision(
            timestamp="",
            current_replicas=2,
            target_replicas=10,  # wants +8, but max is +3
            direction=ScaleDirection.UP,
            reason="test",
            confidence=0.9,
            predicted_peak=5.0,
        )
        result = guard.check(decision)
        assert result.target_replicas <= 2 + 3

    def test_rate_limit_scale_down(self):
        guard = SafetyGuard(
            cooldown_seconds=0,
            max_scale_down_step=2,
        )
        decision = ScalingDecision(
            timestamp="",
            current_replicas=10,
            target_replicas=3,  # wants -7, but max is -2
            direction=ScaleDirection.DOWN,
            reason="test",
            confidence=0.9,
            predicted_peak=1.0,
        )
        result = guard.check(decision)
        assert result.target_replicas >= 10 - 2

    def test_cooldown_blocks_scaling(self):
        guard = SafetyGuard(cooldown_seconds=300)
        # Simulate a recent event
        guard._last_scale_time = time.time()

        decision = ScalingDecision(
            timestamp="",
            current_replicas=2,
            target_replicas=5,
            direction=ScaleDirection.UP,
            reason="test",
            confidence=0.9,
            predicted_peak=3.0,
        )
        result = guard.check(decision)
        assert result.approved is False

    def test_record_event(self):
        guard = SafetyGuard(cooldown_seconds=0)
        decision = ScalingDecision(
            timestamp="2026-04-14T00:00:00",
            current_replicas=2,
            target_replicas=5,
            direction=ScaleDirection.UP,
            reason="test",
            confidence=0.9,
            predicted_peak=3.0,
        )
        guard.record_event(decision)
        history = guard.get_event_history()
        assert len(history) == 1

    def test_no_action_passes_through(self):
        guard = SafetyGuard(cooldown_seconds=300)
        decision = ScalingDecision(
            timestamp="",
            current_replicas=3,
            target_replicas=3,
            direction=ScaleDirection.NONE,
            reason="no change",
            confidence=0.9,
            predicted_peak=2.0,
        )
        result = guard.check(decision)
        assert result.direction == ScaleDirection.NONE
