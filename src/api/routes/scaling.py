# ============================================
# K8s PredictScale - Scaling Routes
# ============================================

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    DryRunRequest,
    DryRunResponse,
    ScalingConfigResponse,
    ScalingConfigUpdate,
    ScalingEventResponse,
    ScalingEventsListResponse,
)

router = APIRouter(prefix="/scaling", tags=["scaling"])


@router.get("/events", response_model=ScalingEventsListResponse)
async def get_scaling_events():
    """Return the history of scaling decisions."""
    from src.api.main import get_controller

    controller = get_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    events = controller.get_scaling_history(limit=50)
    return ScalingEventsListResponse(
        events=[ScalingEventResponse(**e) for e in events],
        total=len(events),
    )


@router.post("/dry-run", response_model=DryRunResponse)
async def simulate_scaling(request: DryRunRequest):
    """Simulate a scaling decision without applying it."""
    from src.api.main import get_controller

    controller = get_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    from src.controller.decision_engine import DecisionEngine

    engine = DecisionEngine(
        target_cpu_utilization=controller._config.scaling.target_cpu_utilization,
        min_replicas=controller._config.scaling.min_replicas,
        max_replicas=controller._config.scaling.max_replicas,
        confidence_threshold=controller._config.scaling.confidence_threshold,
    )

    decision = engine.decide(
        predicted_peak=request.predicted_peak,
        confidence=request.confidence,
        current_replicas=request.current_replicas,
    )

    return DryRunResponse(decision=decision.to_dict())


@router.get("/config", response_model=ScalingConfigResponse)
async def get_scaling_config():
    """Return current scaling configuration."""
    from src.api.main import get_controller

    controller = get_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    cfg = controller._config.scaling
    return ScalingConfigResponse(
        dry_run=cfg.dry_run,
        target_namespace=cfg.target_namespace,
        target_deployment=cfg.target_deployment,
        min_replicas=cfg.min_replicas,
        max_replicas=cfg.max_replicas,
        scale_up_rate=cfg.scale_up_rate,
        scale_down_rate=cfg.scale_down_rate,
        cooldown_period=cfg.cooldown_period,
        confidence_threshold=cfg.confidence_threshold,
        target_cpu_utilization=cfg.target_cpu_utilization,
    )


@router.put("/config", response_model=ScalingConfigResponse)
async def update_scaling_config(update: ScalingConfigUpdate):
    """Update scaling configuration at runtime."""
    from src.api.main import get_controller

    controller = get_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    cfg = controller._config.scaling

    if update.dry_run is not None:
        cfg.dry_run = update.dry_run
        controller.scaler.dry_run = update.dry_run
    if update.min_replicas is not None:
        cfg.min_replicas = update.min_replicas
    if update.max_replicas is not None:
        cfg.max_replicas = update.max_replicas
    if update.scale_up_rate is not None:
        cfg.scale_up_rate = update.scale_up_rate
    if update.scale_down_rate is not None:
        cfg.scale_down_rate = update.scale_down_rate
    if update.cooldown_period is not None:
        cfg.cooldown_period = update.cooldown_period
    if update.confidence_threshold is not None:
        cfg.confidence_threshold = update.confidence_threshold

    return ScalingConfigResponse(
        dry_run=cfg.dry_run,
        target_namespace=cfg.target_namespace,
        target_deployment=cfg.target_deployment,
        min_replicas=cfg.min_replicas,
        max_replicas=cfg.max_replicas,
        scale_up_rate=cfg.scale_up_rate,
        scale_down_rate=cfg.scale_down_rate,
        cooldown_period=cfg.cooldown_period,
        confidence_threshold=cfg.confidence_threshold,
        target_cpu_utilization=cfg.target_cpu_utilization,
    )
