# ============================================
# K8s PredictScale - Metric Registry
# ============================================
# Central registry of all metrics we collect
# from Prometheus.  Each metric is defined as a
# dataclass carrying its PromQL query, friendly
# name, and collection metadata.
# ============================================

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class MetricCategory(str, Enum):
    """Logical grouping for collected metrics."""

    RESOURCE = "resource"
    TRAFFIC = "traffic"
    PERFORMANCE = "performance"
    HEALTH = "health"
    STATE = "state"


@dataclass(frozen=True)
class MetricDefinition:
    """Schema for a single Prometheus metric to collect.

    Attributes:
        name: Internal identifier (used as DataFrame column name).
        promql: PromQL query to execute.
        category: Logical grouping.
        description: Human-readable description.
        unit: Unit of measurement for display / documentation.
        critical: If ``True`` the metric is essential for prediction.
    """

    name: str
    promql: str
    category: MetricCategory
    description: str
    unit: str = ""
    critical: bool = False


# ---------------------------------------------------------------
# Default metric catalogue
# ---------------------------------------------------------------

DEFAULT_METRICS: List[MetricDefinition] = [
    # ---- Resource Metrics ----
    MetricDefinition(
        name="cpu_usage",
        promql='sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", pod=~"{deployment}.*"}}[5m])) by (pod)',
        category=MetricCategory.RESOURCE,
        description="CPU core-seconds consumed per pod (5m rate)",
        unit="cores",
        critical=True,
    ),
    MetricDefinition(
        name="memory_usage",
        promql='sum(container_memory_working_set_bytes{{namespace="{namespace}", pod=~"{deployment}.*"}}) by (pod)',
        category=MetricCategory.RESOURCE,
        description="Working-set memory per pod",
        unit="bytes",
        critical=True,
    ),
    # ---- Traffic Metrics ----
    MetricDefinition(
        name="request_rate",
        promql='sum(rate(http_requests_total{{namespace="{namespace}", pod=~"{deployment}.*"}}[5m]))',
        category=MetricCategory.TRAFFIC,
        description="Incoming HTTP requests per second (5m rate)",
        unit="req/s",
        critical=True,
    ),
    MetricDefinition(
        name="request_rate_by_status",
        promql='sum(rate(http_requests_total{{namespace="{namespace}", pod=~"{deployment}.*"}}[5m])) by (status_code)',
        category=MetricCategory.TRAFFIC,
        description="HTTP requests per second grouped by status code",
        unit="req/s",
    ),
    # ---- Performance Metrics ----
    MetricDefinition(
        name="response_latency_p99",
        promql='histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{{namespace="{namespace}", pod=~"{deployment}.*"}}[5m])) by (le))',
        category=MetricCategory.PERFORMANCE,
        description="99th-percentile response latency",
        unit="seconds",
        critical=True,
    ),
    MetricDefinition(
        name="response_latency_p50",
        promql='histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{{namespace="{namespace}", pod=~"{deployment}.*"}}[5m])) by (le))',
        category=MetricCategory.PERFORMANCE,
        description="50th-percentile (median) response latency",
        unit="seconds",
    ),
    # ---- Health Metrics ----
    MetricDefinition(
        name="error_rate",
        promql='sum(rate(http_requests_total{{namespace="{namespace}", pod=~"{deployment}.*", status=~"5.."}}[5m]))',
        category=MetricCategory.HEALTH,
        description="5xx error rate",
        unit="errors/s",
        critical=True,
    ),
    MetricDefinition(
        name="network_receive",
        promql='sum(rate(container_network_receive_bytes_total{{namespace="{namespace}", pod=~"{deployment}.*"}}[5m]))',
        category=MetricCategory.HEALTH,
        description="Inbound network throughput",
        unit="bytes/s",
    ),
    # ---- State Metrics ----
    MetricDefinition(
        name="ready_replicas",
        promql='kube_deployment_status_replicas_ready{{namespace="{namespace}", deployment="{deployment}"}}',
        category=MetricCategory.STATE,
        description="Number of pods in Ready state",
        unit="pods",
        critical=True,
    ),
    MetricDefinition(
        name="desired_replicas",
        promql='kube_deployment_spec_replicas{{namespace="{namespace}", deployment="{deployment}"}}',
        category=MetricCategory.STATE,
        description="Desired replica count from the deployment spec",
        unit="pods",
    ),
]


class MetricRegistry:
    """Holds and resolves the set of metrics to collect.

    Metric PromQL templates contain ``{namespace}`` and ``{deployment}``
    placeholders that are filled in at query time via :meth:`resolve`.
    """

    def __init__(self, metrics: List[MetricDefinition] | None = None):
        self._metrics: Dict[str, MetricDefinition] = {}
        for m in metrics or DEFAULT_METRICS:
            self.register(m)

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(self, metric: MetricDefinition) -> None:
        """Add or replace a metric definition."""
        self._metrics[metric.name] = metric

    def get(self, name: str) -> MetricDefinition:
        """Retrieve a metric by name.

        Raises:
            KeyError: If the metric is not registered.
        """
        return self._metrics[name]

    @property
    def all_metrics(self) -> List[MetricDefinition]:
        """Return all registered metrics."""
        return list(self._metrics.values())

    @property
    def critical_metrics(self) -> List[MetricDefinition]:
        """Return only the critical metrics needed for prediction."""
        return [m for m in self._metrics.values() if m.critical]

    # ------------------------------------------------------------------
    # PromQL resolution
    # ------------------------------------------------------------------

    def resolve_query(self, name: str, namespace: str, deployment: str) -> str:
        """Return the PromQL query for *name* with placeholders filled.

        Args:
            name: Metric identifier.
            namespace: Target Kubernetes namespace.
            deployment: Target deployment name.

        Returns:
            Fully resolved PromQL string.
        """
        metric = self.get(name)
        return metric.promql.format(namespace=namespace, deployment=deployment)

    def resolve_all(self, namespace: str, deployment: str) -> Dict[str, str]:
        """Resolve every registered metric's PromQL query.

        Returns:
            ``{metric_name: resolved_promql}`` mapping.
        """
        return {name: self.resolve_query(name, namespace, deployment) for name in self._metrics}
