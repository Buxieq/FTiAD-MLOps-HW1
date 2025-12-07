import os
import pickle
import uuid
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import joblib
from models.base import BaseModel
from config.settings import settings
from storage.s3_storage import S3Storage
from utils.logger import setup_logger

logger = setup_logger()


class ModelStorage:
    
    def __init__(self, storage_dir: str = "models_storage", use_s3: Optional[bool] = None):
        """
        Args:
            storage_dir: директория для хранения моделей
            use_s3: по умолчанию из settings.USE_S3_STORAGE
        """
        self.storage_dir = storage_dir
        self.use_s3 = use_s3 if use_s3 is not None else settings.USE_S3_STORAGE
        self.metadata_file = os.path.join(storage_dir, "metadata.pkl")
        self._ensure_storage_dir()
        self.metadata = self._load_metadata()
        
        if self.use_s3:
            try:
                self.s3_storage = S3Storage(bucket_name=settings.S3_BUCKET)
                logger.info("S3 storage initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 storage, falling back to local: {str(e)}")
                self.use_s3 = False
                self.s3_storage = None
        else:
            self.s3_storage = None
    
    def _ensure_storage_dir(self) -> None:
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns:
            Словарь model_id -> метаданные
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def save_model(self, model: BaseModel, model_class_name: str, 
                   hyperparameters: Dict[str, Any], model_id: Optional[str] = None,
                   dvc_dataset_version: Optional[str] = None,
                   mlflow_run_id: Optional[str] = None) -> str:
        """
        Args:
            model: Экземпляр обученной модели
            model_class_name: Название класса модели
            hyperparameters: Гиперпараметры, использованные при обучении
            model_id: Необязательный идентификатор модели (генерируется, если не указан)
            dvc_dataset_version: версия датасета в DVC
            mlflow_run_id: ID запуска в MLFlow
            
        Returns:
            Model ID
        """
        if model_id is None:
            model_id = str(uuid.uuid4())
        
        model_filename = f"{model_id}.joblib"
        
        # еслм проблемы с s3
        if self.use_s3 and self.s3_storage:
            # Save to S3
            s3_key = f"models/{model_filename}"
            temp_path = os.path.join(self.storage_dir, model_filename)
            joblib.dump(model.model, temp_path)
            if self.s3_storage.upload_file(temp_path, s3_key):
                os.remove(temp_path)  # Clean up local temp file
                model_path = f"s3://{settings.S3_BUCKET}/{s3_key}"
            else:                
                model_path = temp_path
                logger.warning(f"Failed to upload to S3, saved locally: {model_path}")
        else:
            model_path = os.path.join(self.storage_dir, model_filename)
            joblib.dump(model.model, model_path)
        
        self.metadata[model_id] = {
            "model_id": model_id,
            "model_class_name": model_class_name,
            "hyperparameters": hyperparameters,
            "model_path": model_path,
            "created_at": datetime.now().isoformat(),
            "is_trained": model.is_trained,
            "dvc_dataset_version": dvc_dataset_version,
            "mlflow_run_id": mlflow_run_id
        }
        self._save_metadata()
        
        return model_id
    
    def load_model(self, model_id: str) -> tuple:
        """
        Args:
            model_id: Model ID
            
        Returns:
            Кортеж из (экземпляр модели, метаданные)

        """
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.metadata[model_id]
        model_class_name = metadata["model_class_name"]
        model_path = metadata["model_path"]
        
        from models.registry import ModelRegistry
        
        model = ModelRegistry.create_model(model_class_name)
        
        if model_path.startswith("s3://") and self.s3_storage:
            path_parts = model_path.replace("s3://", "").split("/", 1)
            if len(path_parts) == 2:
                s3_key = path_parts[1]
                file_obj = self.s3_storage.download_fileobj(s3_key)
                if file_obj:
                    model.model = joblib.load(file_obj)
                else:
                    raise ValueError(f"Failed to download model {model_id} from S3")
            else:
                raise ValueError(f"Invalid S3 path format: {model_path}")
        else:
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            model.model = joblib.load(model_path)
        
        model.is_trained = True
        
        return model, metadata
    
    def delete_model(self, model_id: str) -> None:
        """
        Args:
            model_id: Model ID
        """
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_path = self.metadata[model_id]["model_path"]
        
        if model_path.startswith("s3://") and self.s3_storage:
            path_parts = model_path.replace("s3://", "").split("/", 1)
            if len(path_parts) == 2:
                s3_key = path_parts[1]
                self.s3_storage.delete_file(s3_key)
        else:
            if os.path.exists(model_path):
                os.remove(model_path)
        
        del self.metadata[model_id]
        self._save_metadata()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Returns:
            Список словарей с метаданными моделей
        """
        return list(self.metadata.values())
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Args:
            model_id: Model ID
            
        Returns:
            Словарь с метаданными модели
            
        """
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        return self.metadata[model_id]
    
    def model_exists(self, model_id: str) -> bool:
        """
        Args:
            model_id: Model ID
            
        Returns:
            True, если модель существует, False в др
        """
        return model_id in self.metadata

