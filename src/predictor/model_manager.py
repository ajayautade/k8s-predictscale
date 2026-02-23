# ============================================
# K8s PredictScale - Model Manager
# ============================================
# Handles model versioning, lifecycle, and
# automatic retraining triggers.
# ============================================

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelVersion:
    """Represents a single saved model version."""

    def __init__(
        self,
        version: str,
        path: str,
        created_at: str,
        metrics: Dict[str, float] | None = None,
        is_active: bool = False,
    ):
        self.version = version
        self.path = path
        self.created_at = created_at
        self.metrics = metrics or {}
        self.is_active = is_active

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "path": self.path,
            "created_at": self.created_at,
            "metrics": self.metrics,
            "is_active": self.is_active,
        }


class ModelManager:
    """Manages versioned model storage on the local filesystem.

    Directory layout::

        models/
        ├── registry.json          ← version metadata
        ├── v1_20260413_120000/
        │   ├── lstm_model.keras
        │   └── prophet_model.pkl
        ├── v2_20260414_080000/
        │   └── ...
        └── active -> v2_...       ← symlink to active version
    """

    REGISTRY_FILE = "registry.json"

    def __init__(self, base_path: str = "./models"):
        self._base_path = os.path.abspath(base_path)
        os.makedirs(self._base_path, exist_ok=True)
        self._registry: List[ModelVersion] = []
        self._load_registry()

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def create_version(self, metrics: Dict[str, float] | None = None) -> ModelVersion:
        """Create a new version directory and register it.

        Args:
            metrics: Training / evaluation metrics (MAE, loss, …).

        Returns:
            The newly created :class:`ModelVersion`.
        """
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        version_id = f"v{len(self._registry) + 1}_{ts}"
        version_path = os.path.join(self._base_path, version_id)
        os.makedirs(version_path, exist_ok=True)

        version = ModelVersion(
            version=version_id,
            path=version_path,
            created_at=datetime.utcnow().isoformat(),
            metrics=metrics or {},
            is_active=False,
        )
        self._registry.append(version)
        self._save_registry()

        logger.info("model_version_created", version=version_id)
        return version

    def promote_version(self, version_id: str) -> None:
        """Set *version_id* as the active (production) model.

        Args:
            version_id: Identifier returned by :meth:`create_version`.
        """
        for v in self._registry:
            v.is_active = v.version == version_id

        # Update the ``active`` symlink
        active_link = os.path.join(self._base_path, "active")
        target = os.path.join(self._base_path, version_id)

        if os.path.islink(active_link):
            os.unlink(active_link)
        elif os.path.exists(active_link):
            shutil.rmtree(active_link)

        os.symlink(target, active_link)
        self._save_registry()
        logger.info("model_version_promoted", version=version_id)

    def get_active_version(self) -> Optional[ModelVersion]:
        """Return the currently active model version."""
        for v in self._registry:
            if v.is_active:
                return v
        return None

    def get_active_path(self) -> Optional[str]:
        """Convenience: return the filesystem path of the active model."""
        active = self.get_active_version()
        return active.path if active else None

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return metadata for all registered versions."""
        return [v.to_dict() for v in self._registry]

    # ------------------------------------------------------------------
    # Retraining triggers
    # ------------------------------------------------------------------

    def should_retrain(
        self,
        current_mae: float,
        mae_threshold: float = 0.2,
        max_age_hours: int = 48,
    ) -> bool:
        """Determine whether the active model needs retraining.

        Retraining is triggered when:
            1. Current MAE exceeds *mae_threshold*, OR
            2. The active model is older than *max_age_hours*, OR
            3. No active model exists.

        Args:
            current_mae: Latest observed prediction MAE.
            mae_threshold: Maximum acceptable MAE.
            max_age_hours: Maximum model age in hours.

        Returns:
            ``True`` if retraining is recommended.
        """
        active = self.get_active_version()

        if active is None:
            logger.info("retrain_recommended_no_active_model")
            return True

        # Check MAE
        if current_mae > mae_threshold:
            logger.info(
                "retrain_recommended_high_mae",
                current_mae=round(current_mae, 4),
                threshold=mae_threshold,
            )
            return True

        # Check age
        try:
            created = datetime.fromisoformat(active.created_at)
            age_hours = (datetime.utcnow() - created).total_seconds() / 3600
            if age_hours > max_age_hours:
                logger.info(
                    "retrain_recommended_stale_model",
                    age_hours=round(age_hours, 1),
                    max_hours=max_age_hours,
                )
                return True
        except (ValueError, TypeError):
            return True

        return False

    def cleanup_old_versions(self, keep: int = 5) -> int:
        """Remove the oldest versions, keeping the last *keep* versions.

        The active version is never removed.

        Returns:
            Number of versions removed.
        """
        if len(self._registry) <= keep:
            return 0

        # Sort oldest first
        non_active = [v for v in self._registry if not v.is_active]
        to_remove = non_active[: len(non_active) - (keep - 1)]

        removed = 0
        for v in to_remove:
            if os.path.exists(v.path):
                shutil.rmtree(v.path)
            self._registry.remove(v)
            removed += 1

        self._save_registry()
        logger.info("old_model_versions_cleaned", removed=removed)
        return removed

    # ------------------------------------------------------------------
    # Registry persistence
    # ------------------------------------------------------------------

    def _registry_path(self) -> str:
        return os.path.join(self._base_path, self.REGISTRY_FILE)

    def _save_registry(self) -> None:
        with open(self._registry_path(), "w") as f:
            json.dump([v.to_dict() for v in self._registry], f, indent=2)

    def _load_registry(self) -> None:
        path = self._registry_path()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self._registry = [
                ModelVersion(**entry) for entry in data
            ]
            logger.info("model_registry_loaded", versions=len(self._registry))
        else:
            self._registry = []
