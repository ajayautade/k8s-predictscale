# ============================================
# K8s PredictScale - Collector Service
# ============================================
# Main orchestrator that periodically scrapes
# Prometheus metrics, converts them to DataFrames,
# and stores them in an in-memory time-series
# buffer for downstream consumption.
# ============================================

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from src.collector.metric_registry import MetricRegistry
from src.collector.prometheus_client import PrometheusClient
from src.utils.config import PrometheusConfig, ScalingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsBuffer:
    """Thread-safe rolling buffer for collected time-series data.

    Keeps up to ``max_hours`` of metric history in a single
    :class:`pd.DataFrame` indexed by timestamp, with one column
    per metric.
    """

    def __init__(self, max_hours: int = 168):
        """Initialize the buffer.

        Args:
            max_hours: Maximum number of hours of data to retain.
                       Defaults to 7 days (168 h).
        """
        self._max_hours = max_hours
        self._data: pd.DataFrame = pd.DataFrame()

    @property
    def data(self) -> pd.DataFrame:
        """Return the current buffer contents."""
        return self._data.copy()

    @property
    def size(self) -> int:
        """Number of rows in the buffer."""
        return len(self._data)

    def append(self, new_data: pd.DataFrame) -> None:
        """Append new rows and trim old ones beyond the retention window.

        Args:
            new_data: DataFrame indexed by ``datetime`` with metric columns.
        """
        if new_data.empty:
            return

        if self._data.empty:
            self._data = new_data
        else:
            self._data = pd.concat([self._data, new_data])
            # Remove duplicate timestamps, keeping the latest values
            self._data = self._data[~self._data.index.duplicated(keep="last")]
            self._data.sort_index(inplace=True)

        # Trim to retention window
        cutoff = datetime.utcnow() - timedelta(hours=self._max_hours)
        self._data = self._data[self._data.index >= cutoff]

    def get_latest(self, n: int = 60) -> pd.DataFrame:
        """Return the last *n* rows of buffered data.

        Args:
            n: Number of most-recent rows to return.

        Returns:
            Tail slice of the buffer DataFrame.
        """
        return self._data.tail(n).copy()

    def clear(self) -> None:
        """Drop all buffered data."""
        self._data = pd.DataFrame()


class CollectorService:
    """Orchestrates periodic metric collection from Prometheus.

    Usage::

        service = CollectorService(prom_cfg, scaling_cfg)
        service.collect_once()
        df = service.get_latest_metrics(n=60)
    """

    def __init__(
        self,
        prometheus_config: PrometheusConfig,
        scaling_config: ScalingConfig,
        registry: MetricRegistry | None = None,
        buffer_hours: int = 168,
    ):
        """Initialize the collector service.

        Args:
            prometheus_config: Prometheus connection settings.
            scaling_config: Scaling target settings (namespace / deployment).
            registry: Metric registry (defaults to :data:`DEFAULT_METRICS`).
            buffer_hours: How many hours of data to keep in-memory.
        """
        self._prom_client = PrometheusClient(url=prometheus_config.url)
        self._registry = registry or MetricRegistry()
        self._buffer = MetricsBuffer(max_hours=buffer_hours)

        self._namespace = scaling_config.target_namespace
        self._deployment = scaling_config.target_deployment

        logger.info(
            "collector_service_initialized",
            namespace=self._namespace,
            deployment=self._deployment,
            metrics_count=len(self._registry.all_metrics),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def collect_once(self) -> Dict[str, Optional[float]]:
        """Scrape all registered metrics once (instant query) and buffer.

        Returns:
            ``{metric_name: latest_value}`` for each metric.
        """
        resolved = self._registry.resolve_all(self._namespace, self._deployment)
        snapshot: Dict[str, Optional[float]] = {}

        for name, promql in resolved.items():
            value = self._prom_client.fetch_latest_value(promql)
            snapshot[name] = value

        # Build a single-row DataFrame and append to buffer
        row = pd.DataFrame(
            [snapshot],
            index=[datetime.utcnow()],
        )
        row.index.name = "timestamp"
        self._buffer.append(row)

        logger.info(
            "metrics_collected",
            metrics=len(snapshot),
            buffer_size=self._buffer.size,
        )
        return snapshot

    def collect_range(self, hours: int = 1, step: str = "60s") -> pd.DataFrame:
        """Backfill the buffer with range-query data.

        Useful at startup to pre-load historical data for the LSTM.

        Args:
            hours: How many hours of history to pull.
            step: Query resolution step.

        Returns:
            Combined DataFrame of all metrics over the requested range.
        """
        end = datetime.utcnow()
        start = end - timedelta(hours=hours)

        frames: list[pd.DataFrame] = []
        resolved = self._registry.resolve_all(self._namespace, self._deployment)

        for name, promql in resolved.items():
            df = self._prom_client.query_range_as_dataframe(
                promql, start, end, step=step, metric_name=name
            )
            if not df.empty:
                frames.append(df)

        if frames:
            combined = pd.concat(frames, axis=1)
            self._buffer.append(combined)
            logger.info(
                "range_collection_complete",
                hours=hours,
                rows=len(combined),
                columns=list(combined.columns),
            )
            return combined

        logger.warning("range_collection_empty", hours=hours)
        return pd.DataFrame()

    def get_latest_metrics(self, n: int = 60) -> pd.DataFrame:
        """Return the last *n* buffered data points.

        Args:
            n: Number of rows.

        Returns:
            DataFrame slice suitable for the preprocessor pipeline.
        """
        return self._buffer.get_latest(n)

    def get_all_metrics(self) -> pd.DataFrame:
        """Return the full buffer contents."""
        return self._buffer.data

    def get_buffer_status(self) -> Dict[str, Any]:
        """Return diagnostic info about the collection buffer."""
        data = self._buffer.data
        return {
            "rows": self._buffer.size,
            "columns": list(data.columns) if not data.empty else [],
            "oldest": data.index.min().isoformat() if not data.empty else None,
            "newest": data.index.max().isoformat() if not data.empty else None,
        }

    def is_prometheus_healthy(self) -> bool:
        """Check if the Prometheus backend is reachable."""
        return self._prom_client.is_healthy()
