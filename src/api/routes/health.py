# ============================================
# K8s PredictScale - Health Routes
# ============================================

from fastapi import APIRouter

from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic liveness/readiness probe."""
    # Controller is injected via app state in main.py
    from src.api.main import get_controller

    controller = get_controller()

    prom_healthy = False
    models_loaded = False

    if controller is not None:
        prom_healthy = controller.collector.is_prometheus_healthy()
        models_loaded = controller.predictor.last_prediction is not None or True

    status = "healthy" if prom_healthy else "degraded"

    return HealthResponse(
        status=status,
        prometheus=prom_healthy,
        models_loaded=models_loaded,
        version="0.1.0",
    )
