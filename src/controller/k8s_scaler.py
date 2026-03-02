# ============================================
# K8s PredictScale - K8s Scaler
# ============================================
# Interacts with the Kubernetes API to read
# deployment state and apply replica changes.
# ============================================

from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from kubernetes import client, config as k8s_config
except ImportError:  # pragma: no cover
    client = None  # type: ignore[assignment]
    k8s_config = None  # type: ignore[assignment]


class K8sScaler:
    """Thin wrapper around the Kubernetes Python client.

    Provides methods to fetch current deployment state and patch the
    replica count.  Operations are no-ops when ``dry_run=True``.
    """

    def __init__(
        self,
        namespace: str = "default",
        deployment: str = "sample-app",
        dry_run: bool = True,
        in_cluster: bool = False,
    ):
        """Initialize the Kubernetes scaler.

        Args:
            namespace: Target namespace.
            deployment: Target deployment name.
            dry_run: When ``True``, log scaling actions without
                actually patching the deployment.
            in_cluster: Load in-cluster config (ServiceAccount) if
                ``True``; otherwise use ``~/.kube/config``.
        """
        self._namespace = namespace
        self._deployment = deployment
        self._dry_run = dry_run

        self._apps_v1: Optional[Any] = None

        if client is not None:
            try:
                if in_cluster:
                    k8s_config.load_incluster_config()
                else:
                    k8s_config.load_kube_config()
                self._apps_v1 = client.AppsV1Api()
                logger.info(
                    "k8s_scaler_initialized",
                    namespace=namespace,
                    deployment=deployment,
                    dry_run=dry_run,
                )
            except Exception as exc:
                logger.warning("k8s_config_load_failed", error=str(exc))
        else:
            logger.warning("kubernetes_client_not_installed")

    # ------------------------------------------------------------------
    # Read state
    # ------------------------------------------------------------------

    def get_current_replicas(self) -> int:
        """Fetch the current ready-replica count from K8s.

        Falls back to ``0`` if the cluster is unreachable.
        """
        if self._apps_v1 is None:
            logger.warning("k8s_not_configured_returning_default")
            return 0

        try:
            deploy = self._apps_v1.read_namespaced_deployment(
                name=self._deployment, namespace=self._namespace
            )
            ready = deploy.status.ready_replicas or 0
            return int(ready)
        except Exception as exc:
            logger.error(
                "failed_to_read_replicas",
                deployment=self._deployment,
                error=str(exc),
            )
            return 0

    def get_deployment_info(self) -> Dict[str, Any]:
        """Return detailed deployment status."""
        if self._apps_v1 is None:
            return {"error": "kubernetes client not configured"}

        try:
            deploy = self._apps_v1.read_namespaced_deployment(
                name=self._deployment, namespace=self._namespace
            )
            return {
                "name": self._deployment,
                "namespace": self._namespace,
                "desired_replicas": deploy.spec.replicas,
                "ready_replicas": deploy.status.ready_replicas or 0,
                "available_replicas": deploy.status.available_replicas or 0,
                "updated_replicas": deploy.status.updated_replicas or 0,
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------

    def scale(self, target_replicas: int) -> bool:
        """Patch the deployment's replica count.

        Args:
            target_replicas: Desired number of replicas.

        Returns:
            ``True`` if the patch was applied (or would be in dry-run
            mode); ``False`` on error.
        """
        if self._dry_run:
            logger.info(
                "dry_run_scale",
                deployment=self._deployment,
                target_replicas=target_replicas,
            )
            return True

        if self._apps_v1 is None:
            logger.error("cannot_scale_k8s_not_configured")
            return False

        try:
            body = {"spec": {"replicas": target_replicas}}
            self._apps_v1.patch_namespaced_deployment_scale(
                name=self._deployment,
                namespace=self._namespace,
                body=body,
            )
            logger.info(
                "deployment_scaled",
                deployment=self._deployment,
                target_replicas=target_replicas,
            )
            return True
        except Exception as exc:
            logger.error(
                "scaling_failed",
                deployment=self._deployment,
                target_replicas=target_replicas,
                error=str(exc),
            )
            return False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        self._dry_run = value
        logger.info("dry_run_mode_changed", dry_run=value)
