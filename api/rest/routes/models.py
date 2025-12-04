"""Эндпоинты для управления моделями."""

import numpy as np
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from api.rest.schemas import (
    TrainRequest, TrainResponse, PredictRequest, PredictResponse,
    ModelListResponse, ModelInfo, ModelClassListResponse, ModelClassInfo,
    ErrorResponse
)
from models.registry import ModelRegistry
from storage.model_storage import ModelStorage
from utils.logger import setup_logger

logger = setup_logger()
router = APIRouter(prefix="/api/v1/models", tags=["models"])


model_storage = ModelStorage()


@router.get("/list", response_model=ModelClassListResponse, summary="List available model classes")
async def list_model_classes():
    """
    Returns:
        Список доступных классов моделей с их схемами гиперпараметров
    """
    logger.info("Listing available model classes")
    try:
        available_models = ModelRegistry.list_available_models()
        model_classes = []
        for name, schema in available_models.items():
            hyperparameter_schema = {}
            for param_name, param_info in schema["hyperparameter_schema"].items():
                param_type = param_info.get("type")
                if param_type is not None:
                    if isinstance(param_type, type):
                        param_type = param_type.__name__
                    else:
                        param_type = str(param_type)
                else:
                    param_type = "unknown"
                
                hyperparameter_schema[param_name] = {
                    "type": param_type,
                    "default": param_info.get("default"),
                    "description": param_info.get("description", "")
                }
            
            model_classes.append(ModelClassInfo(
                model_class=name,
                hyperparameter_schema=hyperparameter_schema
            ))
        logger.info(f"Found {len(model_classes)} available model classes")
        return ModelClassListResponse(available_models=model_classes)
    except Exception as e:
        logger.error(f"Error listing model classes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing model classes: {str(e)}"
        )


@router.post("/train", response_model=TrainResponse, status_code=status.HTTP_201_CREATED, summary="Train a model")
async def train_model(request: TrainRequest):
    """
    Args:
        request: Запрос на обучение с классом модели, гиперпараметрами и данными
        
    Returns:
        Ответ об обучении с идентификатором модели
    """
    logger.info(f"Training request received for model class: {request.model_class}")
    try:
        # Проверить класс модели
        try:
            model = ModelRegistry.create_model(request.model_class)
        except ValueError as e:
            logger.error(f"Invalid model class: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        try:
            X = np.array(request.X)
            y = np.array(request.y)
            
            # Проверить формы данных
            if len(X.shape) != 2:
                raise ValueError("X must be a 2D array")
            if len(y.shape) != 1:
                raise ValueError("y must be a 1D array")
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")
        except Exception as e:
            logger.error(f"Invalid data format: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data format: {str(e)}"
            )
        
        # Обучить модель
        try:
            logger.info(f"Training model with hyperparameters: {request.hyperparameters}")
            model.train(X, y, request.hyperparameters)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during model training: {str(e)}"
            )
        
        # Сохранить модель
        try:
            model_id = model_storage.save_model(
                model=model,
                model_class_name=request.model_class,
                hyperparameters=request.hyperparameters
            )
            logger.info(f"Model saved with ID: {model_id}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving model: {str(e)}"
            )
        
        return TrainResponse(
            model_id=model_id,
            model_class=request.model_class,
            hyperparameters=request.hyperparameters,
            message="Model trained successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("", response_model=ModelListResponse, summary="List all trained models")
async def list_trained_models():
    """
    Returns:
        Список всех обученных моделей с их метаданными
    """
    logger.info("Listing all trained models")
    try:
        models_data = model_storage.list_models()
        models = [
            ModelInfo(
                model_id=model_data["model_id"],
                model_class_name=model_data["model_class_name"],
                hyperparameters=model_data["hyperparameters"],
                created_at=model_data["created_at"],
                is_trained=model_data["is_trained"]
            )
            for model_data in models_data
        ]
        logger.info(f"Found {len(models)} trained models")
        return ModelListResponse(models=models)
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@router.post("/{model_id}/predict", response_model=PredictResponse, summary="Get predictions from a model")
async def predict(model_id: str, request: PredictRequest):
    """
   Args:
        model_id: Идентификатор обученной модели
        request: Запрос на прогнозирование с признаками
        
    Returns:
        Прогнозы от модели
    """
    logger.info(f"Prediction request received for model: {model_id}")
    try:
        try:
            model, metadata = model_storage.load_model(model_id)
            logger.info(f"Model {model_id} loaded successfully")
        except ValueError as e:
            logger.error(f"Model not found: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        
        try:
            X = np.array(request.X)
            if len(X.shape) != 2:
                raise ValueError("X must be a 2D array")
        except Exception as e:
            logger.error(f"Invalid data format: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data format: {str(e)}"
            )
        
        try:
            predictions = model.predict(X)
            logger.info(f"Predictions generated successfully for model {model_id}")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during prediction: {str(e)}"
            )
        
        return PredictResponse(
            predictions=predictions.tolist(),
            model_id=model_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.post("/{model_id}/retrain", response_model=TrainResponse, summary="Retrain a model")
async def retrain_model(model_id: str, request: TrainRequest):
    """
    Args:
        model_id: model id
        request: Запрос на обучение с новыми данными и гиперпараметрами
        
    Returns:
        Ответ об обучении
    """
    logger.info(f"Retrain request received for model: {model_id}")
    try:
        if not model_storage.model_exists(model_id):
            logger.error(f"Model {model_id} not found for retraining")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        try:
            model_storage.delete_model(model_id)
            logger.info(f"Old model {model_id} deleted")
        except Exception as e:
            logger.error(f"Error deleting old model: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting old model: {str(e)}"
            )
        
        try:
            X = np.array(request.X)
            y = np.array(request.y)
            
            if len(X.shape) != 2:
                raise ValueError("X must be a 2D array")
            if len(y.shape) != 1:
                raise ValueError("y must be a 1D array")
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")
        except Exception as e:
            logger.error(f"Invalid data format: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data format: {str(e)}"
            )
        
        try:
            model = ModelRegistry.create_model(request.model_class)
            model.train(X, y, request.hyperparameters)
            logger.info("Model retraining completed successfully")
        except Exception as e:
            logger.error(f"Error during model retraining: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during model retraining: {str(e)}"
            )
        
        try:
            new_model_id = model_storage.save_model(
                model=model,
                model_class_name=request.model_class,
                hyperparameters=request.hyperparameters,
                model_id=model_id
            )
            logger.info(f"Retrained model saved with ID: {new_model_id}")
        except Exception as e:
            logger.error(f"Error saving retrained model: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving retrained model: {str(e)}"
            )
        
        return TrainResponse(
            model_id=new_model_id,
            model_class=request.model_class,
            hyperparameters=request.hyperparameters,
            message="Model retrained successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during retraining: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a model")
async def delete_model(model_id: str):
    """
    Args:
        model_id: Идентификатор модели для удаления
    """
    logger.info(f"Delete request received for model: {model_id}")
    try:
        if not model_storage.model_exists(model_id):
            logger.error(f"Model {model_id} not found for deletion")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        model_storage.delete_model(model_id)
        logger.info(f"Model {model_id} deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model: {str(e)}"
        )

