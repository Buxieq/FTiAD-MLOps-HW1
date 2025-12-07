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
from storage.dataset_storage import DatasetStorage
from tracking.mlflow_tracker import MLFlowTracker
from utils.logger import setup_logger

logger = setup_logger()
router = APIRouter(prefix="/api/v1/models", tags=["models"])


model_storage = ModelStorage()
dataset_storage = DatasetStorage()
mlflow_tracker = MLFlowTracker()


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
            train = np.array(request.X)
            target = np.array(request.y)
        except Exception as e:
            logger.error(f"Failed to convert data to numpy arrays: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data format: {str(e)}"
            )
        
        # Сохранить датасет через DVC
        dvc_dataset_version = None
        try:
            dataset_path, dvc_dataset_version = dataset_storage.save_dataset(
                train=train,
                target=target,
                metadata={"model_class": request.model_class, "hyperparameters": request.hyperparameters}
            )
            logger.info(f"Dataset saved with DVC version: {dvc_dataset_version}")
        except Exception as e:
            logger.warning(f"Failed to save dataset with DVC: {str(e)}")
        
        # Начать MLFlow run
        mlflow_run_id = None
        try:
            run_name = f"{request.model_class}_{np.random.randint(1000, 9999)}"
            mlflow_run_id = mlflow_tracker.start_run(
                run_name=run_name,
                tags={"model_class": request.model_class, "api": "rest"}
            )
            logger.info(f"Started MLFlow run: {mlflow_run_id}")
        except Exception as e:
            logger.warning(f"Failed to start MLFlow run: {str(e)}")
        
        try:
            # Логировать параметры и информацию о датасете
            if mlflow_run_id:
                mlflow_tracker.log_params(request.hyperparameters)
                mlflow_tracker.log_params({"model_class": request.model_class})
                mlflow_tracker.log_dataset_info(
                    n_samples=train.shape[0],
                    n_features=train.shape[1],
                    dataset_path=dvc_dataset_version
                )
        except Exception as e:
            logger.warning(f"Failed to log parameters to MLFlow: {str(e)}")
        
        # Обучить модель
        try:
            logger.info(f"Training model with hyperparameters: {request.hyperparameters}")
            model.train(train, target, request.hyperparameters)
            logger.info("Model training completed successfully")
        except Exception as e:
            if mlflow_run_id:
                mlflow_tracker.end_run()
            logger.error(f"Error during model training: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during model training: {str(e)}"
            )
        
        # Вычислить метрики и залогировать в MLFlow
        try:
            target_pred = model.predict(train)
            if mlflow_run_id:
                metrics = mlflow_tracker.calculate_and_log_metrics(target, target_pred)
                mlflow_tracker.log_model(model.model, artifact_path="model")
                logger.info(f"Logged metrics to MLFlow: {metrics}")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLFlow: {str(e)}")
        
        # Сохранить модель
        try:
            model_id = model_storage.save_model(
                model=model,
                model_class_name=request.model_class,
                hyperparameters=request.hyperparameters,
                dvc_dataset_version=dvc_dataset_version,
                mlflow_run_id=mlflow_run_id
            )
            logger.info(f"Model saved with ID: {model_id}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            if mlflow_run_id:
                mlflow_tracker.end_run()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving model: {str(e)}"
            )
        finally:
            # Завершить MLFlow run
            if mlflow_run_id:
                mlflow_tracker.end_run()
        
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
            features = np.array(request.X)
        except Exception as e:
            logger.error(f"Failed to convert data to numpy array: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data format: {str(e)}"
            )
        
        try:
            predictions = model.predict(features)
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
            train = np.array(request.X)
            target = np.array(request.y)
        except Exception as e:
            logger.error(f"Failed to convert data to numpy arrays: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data format: {str(e)}"
            )
        
        # Сохранить датасет через DVC
        dvc_dataset_version = None
        try:
            dataset_path, dvc_dataset_version = dataset_storage.save_dataset(
                train=train,
                target=target,
                model_id=model_id,
                metadata={"model_class": request.model_class, "hyperparameters": request.hyperparameters, "retrain": True}
            )
            logger.info(f"Dataset saved with DVC version: {dvc_dataset_version}")
        except Exception as e:
            logger.warning(f"Failed to save dataset with DVC: {str(e)}")
        
        # Начать MLFlow run
        mlflow_run_id = None
        try:
            run_name = f"{request.model_class}_retrain_{np.random.randint(1000, 9999)}"
            mlflow_run_id = mlflow_tracker.start_run(
                run_name=run_name,
                tags={"model_class": request.model_class, "api": "rest", "retrain": "true"}
            )
            logger.info(f"Started MLFlow run: {mlflow_run_id}")
        except Exception as e:
            logger.warning(f"Failed to start MLFlow run: {str(e)}")
        
        try:
            if mlflow_run_id:
                mlflow_tracker.log_params(request.hyperparameters)
                mlflow_tracker.log_params({"model_class": request.model_class, "retrain": True})
                mlflow_tracker.log_dataset_info(
                    n_samples=train.shape[0],
                    n_features=train.shape[1],
                    dataset_path=dvc_dataset_version
                )
        except Exception as e:
            logger.warning(f"Failed to log parameters to MLFlow: {str(e)}")
        
        try:
            model = ModelRegistry.create_model(request.model_class)
            model.train(train, target, request.hyperparameters)
            logger.info("Model retraining completed successfully")
        except Exception as e:
            if mlflow_run_id:
                mlflow_tracker.end_run()
            logger.error(f"Error during model retraining: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during model retraining: {str(e)}"
            )
        
        # Вычислить метрики и залогировать в MLFlow
        try:
            target_pred = model.predict(train)
            if mlflow_run_id:
                metrics = mlflow_tracker.calculate_and_log_metrics(target, target_pred)
                mlflow_tracker.log_model(model.model, artifact_path="model")
                logger.info(f"Logged metrics to MLFlow: {metrics}")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLFlow: {str(e)}")
        
        try:
            new_model_id = model_storage.save_model(
                model=model,
                model_class_name=request.model_class,
                hyperparameters=request.hyperparameters,
                model_id=model_id,
                dvc_dataset_version=dvc_dataset_version,
                mlflow_run_id=mlflow_run_id
            )
            logger.info(f"Retrained model saved with ID: {new_model_id}")
        except Exception as e:
            logger.error(f"Error saving retrained model: {str(e)}")
            if mlflow_run_id:
                mlflow_tracker.end_run()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving retrained model: {str(e)}"
            )
        finally:
            if mlflow_run_id:
                mlflow_tracker.end_run()
        
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

