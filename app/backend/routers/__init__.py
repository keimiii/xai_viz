"""API route modules."""

from app.backend.routers.attention import router as attention_router
from app.backend.routers.comparison import router as comparison_router
from app.backend.routers.images import router as images_router
from app.backend.routers.metrics import router as metrics_router

__all__ = ["attention_router", "comparison_router", "images_router", "metrics_router"]
