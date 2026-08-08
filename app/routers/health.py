from fastapi import APIRouter

from app.config import settings
from app.core.llm_setup import models_ready
from app.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_version=settings.app_version,
        index_ready=models_ready(),
    )


@router.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": f"{settings.app_name} API is running. See /docs for endpoints."}
