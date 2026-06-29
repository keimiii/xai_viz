"""FastAPI application for SSL attention visualization.

Run with:
    uvicorn app.backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add SSL attention source to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from app.backend.config import (
    ANNOTATIONS_PATH,
    API_PREFIX,
    ATTENTION_CACHE_PATH,
    CACHE_PATH,
    CORS_ORIGINS,
    DATASET_PATH,
    DEBUG,
    HEATMAPS_PATH,
    METRICS_DB_PATH,
)
from app.backend.routers import (
    attention_router,
    comparison_router,
    images_router,
    metrics_router,
)

logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage application startup and shutdown."""
    # --- Startup ---
    logger.info("Starting SSL Attention Visualization API v%s", APP_VERSION)
    logger.info("  Dataset path: %s", DATASET_PATH)
    logger.info("  Cache path: %s", CACHE_PATH)
    logger.info("  Debug mode: %s", DEBUG)

    # Validate critical data files
    checks = {
        "annotations": ANNOTATIONS_PATH.exists(),
        "attention_cache": ATTENTION_CACHE_PATH.exists(),
        "metrics_db": METRICS_DB_PATH.exists(),
        "heatmaps_dir": HEATMAPS_PATH.is_dir(),
    }
    for name, available in checks.items():
        level = logging.INFO if available else logging.WARNING
        status = "OK" if available else "MISSING"
        logger.log(level, "  %s: %s", name, status)

    # Eagerly load annotations so first request isn't slow
    if checks["annotations"]:
        from app.backend.services.image_service import image_service

        n = len(image_service.list_image_ids())
        logger.info("Loaded %d annotated images", n)

    missing = [k for k, v in checks.items() if not v]
    if missing:
        logger.warning("Starting in DEGRADED mode — missing: %s", missing)
    else:
        logger.info("All resources available — ready to serve")

    yield

    # --- Shutdown ---
    logger.info("Shutting down SSL Attention Visualization API")
    from ssl_attention.models.registry import clear_cache

    clear_cache()
    logger.info("Model cache cleared")


app = FastAPI(
    title="SSL Attention Visualization API",
    description="API for visualizing SSL model attention patterns on WikiChurches images",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(images_router, prefix=API_PREFIX)
app.include_router(attention_router, prefix=API_PREFIX)
app.include_router(metrics_router, prefix=API_PREFIX)
app.include_router(comparison_router, prefix=API_PREFIX)


@app.get("/")
async def root() -> dict:
    """Root endpoint with API info."""
    return {
        "name": "SSL Attention Visualization API",
        "version": APP_VERSION,
        "docs": "/docs",
        "endpoints": {
            "images": f"{API_PREFIX}/images",
            "attention": f"{API_PREFIX}/attention",
            "metrics": f"{API_PREFIX}/metrics",
            "comparison": f"{API_PREFIX}/compare",
        },
    }


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint with degraded-mode detection."""
    from app.backend.services.image_service import image_service
    from app.backend.services.metrics_service import metrics_service

    checks: dict[str, bool] = {}

    try:
        checks["annotations_loaded"] = len(image_service.list_image_ids()) > 0
    except Exception:
        checks["annotations_loaded"] = False

    try:
        checks["metrics_db_available"] = metrics_service.db_exists
    except Exception:
        checks["metrics_db_available"] = False

    checks["attention_cache_available"] = ATTENTION_CACHE_PATH.exists()

    status = "healthy" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks}


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    detail = str(exc) if DEBUG else "Internal server error"
    return JSONResponse(
        status_code=500,
        content={"detail": detail},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
