"""Эндпоинты для проверки работоспособности"""

from fastapi import APIRouter
from api.rest.schemas import HealthResponse
from utils.logger import setup_logger

logger = setup_logger()
router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Health check")
async def health_check():
    """
    Returns:
        Ответ о статусе работоспособности
    """
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        message="Service is running"
    )

