# ============================================
# K8s PredictScale - Collector Unit Tests
# ============================================

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector.metric_registry import (
    DEFAULT_METRICS,
    MetricCategory,
    MetricDefinition,
    MetricRegistry,
)


class TestMetricRegistry:
    """Tests for MetricRegistry."""

    def test_default_registry_has_metrics(self):
        registry = MetricRegistry()
        assert len(registry.all_metrics) > 0

    def test_register_and_get(self):
        registry = MetricRegistry(metrics=[])
        metric = MetricDefinition(
            name="test_metric",
            promql="up",
            category=MetricCategory.HEALTH,
            description="Test",
        )
        registry.register(metric)
        assert registry.get("test_metric") == metric

    def test_get_raises_for_unknown(self):
        registry = MetricRegistry(metrics=[])
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_critical_metrics_filter(self):
        registry = MetricRegistry()
        critical = registry.critical_metrics
        assert all(m.critical for m in critical)
        assert len(critical) < len(registry.all_metrics)

    def test_resolve_query_replaces_placeholders(self):
        registry = MetricRegistry()
        query = registry.resolve_query("cpu_usage", "prod", "web-server")
        assert "prod" in query
        assert "web-server" in query
        assert "{namespace}" not in query
        assert "{deployment}" not in query

    def test_resolve_all(self):
        registry = MetricRegistry()
        resolved = registry.resolve_all("default", "sample-app")
        assert isinstance(resolved, dict)
        assert "cpu_usage" in resolved
        for name, promql in resolved.items():
            assert "{namespace}" not in promql


class TestMetricsBuffer:
    """Tests for the in-memory MetricsBuffer."""

    def test_append_and_read(self, small_metrics_df):
        from src.collector.collector_service import MetricsBuffer

        buf = MetricsBuffer(max_hours=24)
        buf.append(small_metrics_df)
        assert buf.size == len(small_metrics_df)

    def test_get_latest(self, small_metrics_df):
        from src.collector.collector_service import MetricsBuffer

        buf = MetricsBuffer()
        buf.append(small_metrics_df)
        latest = buf.get_latest(5)
        assert len(latest) == 5

    def test_clear(self, small_metrics_df):
        from src.collector.collector_service import MetricsBuffer

        buf = MetricsBuffer()
        buf.append(small_metrics_df)
        buf.clear()
        assert buf.size == 0

    def test_deduplication(self, small_metrics_df):
        from src.collector.collector_service import MetricsBuffer

        buf = MetricsBuffer()
        buf.append(small_metrics_df)
        buf.append(small_metrics_df)  # duplicate timestamps
        assert buf.size == len(small_metrics_df)
