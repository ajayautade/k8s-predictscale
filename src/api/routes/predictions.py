# ============================================
# K8s PredictScale - Prediction Routes
# ============================================

from fastapi import APIRouter, HTTPException

from src.api.schemas import PredictionHistoryResponse, PredictionResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=PredictionResponse)
async def get_current_prediction():
    """Return the most recent prediction and confidence."""
    from src.api.main import get_controller

    controller = get_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    prediction = controller.predictor.last_prediction
    if prediction is None:
        raise HTTPException(status_code=404, detail="No predictions available yet")

    return PredictionResponse(**prediction.to_dict())


@router.get("/history", response_model=PredictionHistoryResponse)
async def get_prediction_history():
    """Return historical prediction vs actual data."""
    from src.api.main import get_controller

    controller = get_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    # For now, return the last prediction and scaling events as proxy
    history = controller.get_scaling_history(limit=50)
    entries = [
        {
            "timestamp": e.get("timestamp", ""),
            "predicted_peak": e.get("predicted_peak", 0),
            "actual_peak": None,
            "confidence": e.get("confidence", 0),
            "recommended_replicas": e.get("target_replicas", 0),
        }
        for e in history
    ]

    return PredictionHistoryResponse(entries=entries, total=len(entries))
