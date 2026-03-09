# ============================================
# K8s PredictScale - Model Routes
# ============================================

from fastapi import APIRouter, HTTPException

from src.api.schemas import ModelStatusResponse, RetrainRequest, RetrainResponse

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Return current model training status and accuracy."""
    from src.api.main import get_controller

    controller = get_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    status = controller.predictor.get_status()
    return ModelStatusResponse(**status)


@router.post("/retrain", response_model=RetrainResponse)
async def trigger_retrain(request: RetrainRequest):
    """Trigger manual model retraining."""
    from src.api.main import get_controller

    controller = get_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    try:
        result = controller.initial_training(hours=request.hours)
        if "error" in result:
            return RetrainResponse(status="failed", error=result["error"])

        return RetrainResponse(
            status="success",
            version=result.get("version"),
            metrics=result.get("metrics"),
        )
    except Exception as exc:
        return RetrainResponse(status="failed", error=str(exc))
