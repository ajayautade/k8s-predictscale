# ============================================
# K8s PredictScale - Prometheus Client
# ============================================
# Executes PromQL queries against a Prometheus
# server and returns structured metric data.
# ============================================

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from prometheus_api_client import PrometheusConnect

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PrometheusClient:
    """Client for querying Prometheus metrics via PromQL.

    Wraps the prometheus-api-client library to provide a clean
    interface for fetching instant and range-based metric data.
    """

    def __init__(self, url: str = "http://localhost:9090", disable_ssl: bool = True):
        """Initialize the Prometheus client.

        Args:
            url: Prometheus server URL.
            disable_ssl: Whether to skip SSL verification.
        """
        self._url = url
        self._prom = PrometheusConnect(url=url, disable_ssl=disable_ssl)
        logger.info("prometheus_client_initialized", url=url)

    # ------------------------------------------------------------------
    # Core query helpers
    # ------------------------------------------------------------------

    def query_instant(self, promql: str) -> List[Dict[str, Any]]:
        """Execute an instant PromQL query.

        Args:
            promql: PromQL query string.

        Returns:
            List of result dicts with ``metric`` and ``value`` keys.
        """
        try:
            results = self._prom.custom_query(query=promql)
            logger.debug("instant_query_executed", query=promql, result_count=len(results))
            return results
        except Exception as exc:
            logger.error("instant_query_failed", query=promql, error=str(exc))
            raise

    def query_range(
        self,
        promql: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "60s",
    ) -> List[Dict[str, Any]]:
        """Execute a range PromQL query.

        Args:
            promql: PromQL query string.
            start_time: Range start (UTC).
            end_time: Range end (UTC).
            step: Query resolution step (e.g. ``"15s"``, ``"60s"``).

        Returns:
            List of result dicts with ``metric`` and ``values`` keys.
        """
        try:
            results = self._prom.custom_query_range(
                query=promql,
                start_time=start_time,
                end_time=end_time,
                step=step,
            )
            logger.debug(
                "range_query_executed",
                query=promql,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                step=step,
                result_count=len(results),
            )
            return results
        except Exception as exc:
            logger.error("range_query_failed", query=promql, error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    def query_range_as_dataframe(
        self,
        promql: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "60s",
        metric_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Execute a range query and return the result as a DataFrame.

        The DataFrame is indexed by timestamp with one column per
        metric series.  If *metric_name* is supplied it is used as the
        column name when there is exactly one series.

        Args:
            promql: PromQL query string.
            start_time: Range start (UTC).
            end_time: Range end (UTC).
            step: Query resolution step.
            metric_name: Optional friendly column name.

        Returns:
            ``pd.DataFrame`` indexed by ``datetime`` with float values.
        """
        raw = self.query_range(promql, start_time, end_time, step)

        if not raw:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for idx, series in enumerate(raw):
            timestamps = [datetime.fromtimestamp(float(v[0])) for v in series["values"]]
            values = [float(v[1]) for v in series["values"]]

            col = metric_name if metric_name and len(raw) == 1 else f"series_{idx}"
            df = pd.DataFrame({"timestamp": timestamps, col: values})
            df.set_index("timestamp", inplace=True)
            frames.append(df)

        result = pd.concat(frames, axis=1)
        return result

    def fetch_latest_value(self, promql: str) -> Optional[float]:
        """Fetch the latest scalar value for a PromQL query.

        Returns:
            The float value, or ``None`` if no data.
        """
        results = self.query_instant(promql)
        if results:
            try:
                return float(results[0]["value"][1])
            except (KeyError, IndexError, ValueError):
                return None
        return None

    def is_healthy(self) -> bool:
        """Check if Prometheus is reachable.

        Returns:
            ``True`` when the ``up`` metric returns results.
        """
        try:
            result = self.query_instant("up")
            return len(result) > 0
        except Exception:
            return False
