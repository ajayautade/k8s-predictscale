# ============================================
# K8s PredictScale - Alert Helpers
# ============================================
# Sends notifications to Slack (or other
# channels) when important scaling events or
# model drift is detected.
# ============================================

from enum import Enum
from typing import Any, Dict, Optional

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertManager:
    """Sends scaling and model health alerts to external channels.

    Currently supports Slack webhooks.  Can be extended to PagerDuty,
    email, etc.
    """

    SEVERITY_EMOJI = {
        AlertSeverity.INFO: "ℹ️",
        AlertSeverity.WARNING: "⚠️",
        AlertSeverity.CRITICAL: "🚨",
    }

    def __init__(
        self,
        enabled: bool = False,
        slack_webhook_url: str = "",
    ):
        self._enabled = enabled
        self._slack_url = slack_webhook_url

    # ------------------------------------------------------------------
    # Generic send
    # ------------------------------------------------------------------

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Dispatch an alert to configured channels.

        Args:
            title: Alert title / summary.
            message: Alert body.
            severity: Severity level.
            metadata: Optional key-value pairs to include.

        Returns:
            ``True`` if the alert was sent (or skipped because disabled).
        """
        if not self._enabled:
            logger.debug("alert_skipped_disabled", title=title)
            return True

        emoji = self.SEVERITY_EMOJI.get(severity, "")
        fields = ""
        if metadata:
            fields = "\n".join(f"• *{k}*: {v}" for k, v in metadata.items())

        slack_payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {title}",
                    },
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": message},
                },
            ]
        }

        if fields:
            slack_payload["blocks"].append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": fields},
                }
            )

        return await self._send_slack(slack_payload)

    # ------------------------------------------------------------------
    # Pre-built alert types
    # ------------------------------------------------------------------

    async def alert_scaling_event(
        self,
        direction: str,
        current: int,
        target: int,
        confidence: float,
    ) -> bool:
        """Send a scaling-event notification."""
        return await self.send_alert(
            title=f"Scaling {direction.upper()}: {current} → {target} replicas",
            message=f"PredictScale has triggered a *{direction}* scaling event.",
            severity=AlertSeverity.INFO if direction == "up" else AlertSeverity.WARNING,
            metadata={
                "Current Replicas": current,
                "Target Replicas": target,
                "Confidence": f"{confidence:.1%}",
            },
        )

    async def alert_model_drift(self, current_mae: float, threshold: float) -> bool:
        """Send a model-drift warning."""
        return await self.send_alert(
            title="Model Drift Detected",
            message=(
                f"Prediction MAE ({current_mae:.4f}) exceeds threshold ({threshold:.4f}). "
                "Consider retraining the model."
            ),
            severity=AlertSeverity.WARNING,
            metadata={
                "Current MAE": round(current_mae, 4),
                "Threshold": round(threshold, 4),
            },
        )

    async def alert_low_confidence(self, confidence: float) -> bool:
        """Send a low-confidence warning."""
        return await self.send_alert(
            title="Low Prediction Confidence",
            message=f"Prediction confidence has dropped to {confidence:.1%}.",
            severity=AlertSeverity.WARNING,
            metadata={"Confidence": f"{confidence:.1%}"},
        )

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    async def _send_slack(self, payload: Dict[str, Any]) -> bool:
        """Post a message to a Slack webhook."""
        if not self._slack_url:
            logger.warning("slack_webhook_not_configured")
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._slack_url,
                    json=payload,
                    timeout=10.0,
                )
                if response.status_code == 200:
                    logger.info("slack_alert_sent")
                    return True
                else:
                    logger.error(
                        "slack_alert_failed",
                        status=response.status_code,
                        body=response.text,
                    )
                    return False
        except Exception as exc:
            logger.error("slack_alert_error", error=str(exc))
            return False
