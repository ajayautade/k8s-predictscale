# ============================================
# K8s PredictScale - API Schemas
# ============================================
# Pydantic models for REST API request /
# response validation.
# ============================================

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    prometheus: bool = Field(..., description="Prometheus reachability")
    models_loaded: bool = Field(..., description="Whether prediction models are loaded")
    version: str = Field(default="0.1.0")


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------


class PredictionResponse(BaseModel):
    predicted_values: List[float] = Field(..., description="Forecasted metric values")
    confidence: float = Field(..., description="Prediction confidence [0, 1]")
    confidence_band_lower: List[float]
    confidence_band_upper: List[float]
    lstm_weight: float
    prophet_weight: float
    recommended_replicas: int
    scaling_urgency: str


class PredictionHistoryEntry(BaseModel):
    timestamp: str
    predicted_peak: float
    actual_peak: Optional[float] = None
    confidence: float
    recommended_replicas: int


class PredictionHistoryResponse(BaseModel):
    entries: List[PredictionHistoryEntry]
    total: int


# ------------------------------------------------------------------
# Scaling
# ------------------------------------------------------------------


class ScalingEventResponse(BaseModel):
    timestamp: str
    current_replicas: int
    target_replicas: int
    direction: str
    reason: str
    confidence: float
    predicted_peak: float
    approved: bool


class ScalingEventsListResponse(BaseModel):
    events: List[ScalingEventResponse]
    total: int


class DryRunRequest(BaseModel):
    predicted_peak: float = Field(..., description="Simulated predicted peak load")
    confidence: float = Field(default=0.9, description="Simulated confidence level")
    current_replicas: int = Field(default=2, description="Simulated current replica count")


class DryRunResponse(BaseModel):
    decision: Dict[str, Any]


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------


class ModelStatusResponse(BaseModel):
    lstm_trained: bool
    prophet_trained: bool
    pipeline_fitted: bool
    active_model_version: Optional[str]
    prediction_count: int
    training_data_hours: float
    ensemble: Dict[str, Any]


class RetrainRequest(BaseModel):
    hours: int = Field(default=24, description="Hours of historical data for retraining")


class RetrainResponse(BaseModel):
    status: str
    version: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------


class ScalingConfigResponse(BaseModel):
    dry_run: bool
    target_namespace: str
    target_deployment: str
    min_replicas: int
    max_replicas: int
    scale_up_rate: int
    scale_down_rate: int
    cooldown_period: int
    confidence_threshold: float
    target_cpu_utilization: float


class ScalingConfigUpdate(BaseModel):
    dry_run: Optional[bool] = None
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    scale_up_rate: Optional[int] = None
    scale_down_rate: Optional[int] = None
    cooldown_period: Optional[int] = None
    confidence_threshold: Optional[float] = None


# ------------------------------------------------------------------
# System Status
# ------------------------------------------------------------------


class SystemStatusResponse(BaseModel):
    tick_count: int
    collector: Dict[str, Any]
    predictor: Dict[str, Any]
    scaler: Dict[str, Any]
    safety: Dict[str, Any]
    last_decision: Optional[Dict[str, Any]]
    prometheus_healthy: bool
