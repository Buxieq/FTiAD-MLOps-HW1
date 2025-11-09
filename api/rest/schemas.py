"""Pydantic schemas for REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Request schema for model training."""
    
    model_class: str = Field(..., description="Name of the model class to train")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters for the model")
    X: List[List[float]] = Field(..., description="Training features (n_samples, n_features)")
    y: List[float] = Field(..., description="Training targets (n_samples,)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_class": "LinearRegression",
                "hyperparameters": {"fit_intercept": True},
                "X": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                "y": [3.0, 5.0, 7.0]
            }
        }


class PredictRequest(BaseModel):
    """Request schema for model prediction."""
    
    X: List[List[float]] = Field(..., description="Features for prediction (n_samples, n_features)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "X": [[1.0, 2.0], [2.0, 3.0]]
            }
        }


class TrainResponse(BaseModel):
    """Response schema for model training."""
    
    model_id: str = Field(..., description="ID of the trained model")
    model_class: str = Field(..., description="Name of the model class")
    hyperparameters: Dict[str, Any] = Field(..., description="Hyperparameters used for training")
    message: str = Field(..., description="Success message")


class PredictResponse(BaseModel):
    """Response schema for model prediction."""
    
    predictions: List[float] = Field(..., description="Model predictions")
    model_id: str = Field(..., description="ID of the model used")


class ModelInfo(BaseModel):
    """Schema for model information."""
    
    model_id: str = Field(..., description="Model ID")
    model_class_name: str = Field(..., description="Name of the model class")
    hyperparameters: Dict[str, Any] = Field(..., description="Hyperparameters used")
    created_at: str = Field(..., description="Creation timestamp")
    is_trained: bool = Field(..., description="Whether model is trained")


class ModelListResponse(BaseModel):
    """Response schema for listing models."""
    
    models: List[ModelInfo] = Field(..., description="List of trained models")


class ModelClassInfo(BaseModel):
    """Schema for model class information."""
    
    model_class: str = Field(..., description="Name of the model class")
    hyperparameter_schema: Dict[str, Any] = Field(..., description="Hyperparameter schema")


class ModelClassListResponse(BaseModel):
    """Response schema for listing available model classes."""
    
    available_models: List[ModelClassInfo] = Field(..., description="List of available model classes")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")

