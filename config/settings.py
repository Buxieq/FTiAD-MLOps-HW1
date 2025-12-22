"""Centralized configuration for the application."""

import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""
    
    # Minio settings
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "mlops-models")
    S3_DATASET_BUCKET: str = os.getenv("S3_DATASET_BUCKET", "mlops-datasets")
    S3_USE_SSL: bool = os.getenv("S3_USE_SSL", "false").lower() == "true"
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    
    # Storage
    USE_S3_STORAGE: bool = os.getenv("USE_S3_STORAGE", "true").lower() == "true"
    LOCAL_STORAGE_DIR: str = os.getenv("LOCAL_STORAGE_DIR", "models_storage")
    
    # MLFlow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-hw2")
    
    # DVC
    DVC_REMOTE: str = os.getenv("DVC_REMOTE", "minio")
    DVC_DATA_DIR: str = os.getenv("DVC_DATA_DIR", "data")
    
    @classmethod
    def get_s3_config(cls) -> dict:
        """Get S3 configuration dictionary."""
        return {
            "endpoint_url": cls.S3_ENDPOINT,
            "aws_access_key_id": cls.S3_ACCESS_KEY,
            "aws_secret_access_key": cls.S3_SECRET_KEY,
            "region_name": cls.S3_REGION,
            "use_ssl": cls.S3_USE_SSL
        }

settings = Settings()

