# ============================================
# K8s PredictScale - FastAPI Application
# ============================================
# Main entry point for the REST API server.
# Bootstraps configuration, controller, and
# the periodic prediction scheduler.
# ============================================

from contextlib import asynccontextmanager
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)
from starlette.responses import Response

from src.api.routes import health, model, predictions, scaling
from src.controller.controller_service import ControllerService
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging

# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------

_controller: Optional[ControllerService] = None
_scheduler: Optional[BackgroundScheduler] = None


def get_controller() -> Optional[ControllerService]:
    """Accessor used by route handlers."""
    return _controller


# ------------------------------------------------------------------
# Prometheus metrics exported by *this* application
# ------------------------------------------------------------------

PREDICTION_CONFIDENCE = Gauge(
    "predictscale_prediction_confidence",
    "Latest ensemble prediction confidence",
)
PREDICTION_MAE = Gauge(
    "predictscale_prediction_mae",
    "Latest prediction mean absolute error",
)
SCALING_EVENTS = Counter(
    "predictscale_scaling_events_total",
    "Total scaling events executed",
    ["direction"],
)
PREDICTION_CYCLES = Counter(
    "predictscale_prediction_cycles_total",
    "Total prediction cycles completed",
)
MODEL_LAST_TRAINED = Gauge(
    "predictscale_model_last_trained_timestamp",
    "Epoch timestamp of last model training",
)
RECOMMENDED_REPLICAS = Gauge(
    "predictscale_recommended_replicas",
    "Latest recommended replica count",
)


def _tick_job():
    """Periodic control-loop invocation — runs in a background thread."""
    if _controller is None:
        return

    result = _controller.tick()

    # Update Prometheus gauges
    if "prediction" in result:
        pred = result["prediction"]
        PREDICTION_CONFIDENCE.set(pred.get("confidence", 0))
        RECOMMENDED_REPLICAS.set(pred.get("recommended_replicas", 0))
        PREDICTION_CYCLES.inc()

    if result.get("scaled"):
        direction = result.get("decision", {}).get("direction", "none")
        SCALING_EVENTS.labels(direction=direction).inc()


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup / shutdown of background services."""
    global _controller, _scheduler

    config = load_config()
    setup_logging(config.api.log_level)
    logger = get_logger("main")

    logger.info("starting_predictscale", version="0.1.0")

    _controller = ControllerService(config)

    # Start the periodic scheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _tick_job,
        "interval",
        seconds=config.prediction.interval,
        id="control_loop",
    )
    _scheduler.start()
    logger.info("scheduler_started", interval=config.prediction.interval)

    yield  # Application running

    # Shutdown
    if _scheduler:
        _scheduler.shutdown(wait=False)
    logger.info("predictscale_shutdown")


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------

app = FastAPI(
    title="K8s PredictScale",
    description="AI-Powered Predictive Auto-Scaler for Kubernetes",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Register routers --
app.include_router(health.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(scaling.router, prefix="/api/v1")
app.include_router(model.router, prefix="/api/v1")


# -- Prometheus metrics endpoint --
@app.get("/api/v1/metrics")
async def metrics():
    """Expose Prometheus-format metrics for scraping."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# -- Root redirect --
@app.get("/")
async def root():
    return {
        "service": "K8s PredictScale",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
