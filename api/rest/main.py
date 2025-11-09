"""main FastAPI"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.rest.routes import health, models
from utils.logger import setup_logger

logger = setup_logger()

app = FastAPI(
    title="MLOps API",
    description="REST API for ML model training and inference",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models.router)


@app.on_event("startup")
async def startup_event():
    logger.info("MLOps API server starting")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("MLOps API server shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

