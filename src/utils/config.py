# ============================================
# K8s PredictScale - Configuration
# ============================================

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class PrometheusConfig:
    """Prometheus connection settings."""
    url: str = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    scrape_interval: int = int(os.getenv("PROMETHEUS_SCRAPE_INTERVAL", "15"))


@dataclass
class PredictionConfig:
    """Prediction engine settings."""
    interval: int = int(os.getenv("PREDICTION_INTERVAL", "60"))
    horizon: int = int(os.getenv("PREDICTION_HORIZON", "10"))
    model_path: str = os.getenv("MODEL_PATH", "./models")
    min_training_hours: int = int(os.getenv("MIN_TRAINING_DATA_HOURS", "72"))
    lookback_steps: int = 60  # Number of historical steps for LSTM input
    lstm_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rate: float = 0.2
    batch_size: int = 32
    epochs: int = 50


@dataclass
class ScalingConfig:
    """Scaling controller settings."""
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    target_namespace: str = os.getenv("TARGET_NAMESPACE", "default")
    target_deployment: str = os.getenv("TARGET_DEPLOYMENT", "sample-app")
    min_replicas: int = int(os.getenv("MIN_REPLICAS", "2"))
    max_replicas: int = int(os.getenv("MAX_REPLICAS", "50"))
    scale_up_rate: int = int(os.getenv("SCALE_UP_RATE", "5"))
    scale_down_rate: int = int(os.getenv("SCALE_DOWN_RATE", "2"))
    cooldown_period: int = int(os.getenv("COOLDOWN_PERIOD", "120"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    target_cpu_utilization: float = 0.7  # 70% target CPU per pod


@dataclass
class APIConfig:
    """API server settings."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@dataclass
class AlertConfig:
    """Alert & notification settings."""
    enabled: bool = os.getenv("ALERT_ENABLED", "false").lower() == "true"
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")


@dataclass
class AppConfig:
    """Root application configuration."""
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    return AppConfig()
