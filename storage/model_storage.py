"""Хранилище моделей"""

import os
import pickle
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import joblib
from models.base import BaseModel


class ModelStorage:
    """Хранилище обученных моделей с метаданными."""
    
    def __init__(self, storage_dir: str = "models_storage"):
        """
        Args:
            storage_dir: Директория для хранения моделей
        """
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.pkl")
        self._ensure_storage_dir()
        self.metadata = self._load_metadata()
    
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
                   hyperparameters: Dict[str, Any], model_id: Optional[str] = None) -> str:
        """
        Args:
            model: Экземпляр обученной модели
            model_class_name: Название класса модели
            hyperparameters: Гиперпараметры, использованные при обучении
            model_id: Необязательный идентификатор модели (генерируется, если не указан)
            
        Returns:
            Model ID
        """
        if model_id is None:
            model_id = str(uuid.uuid4())
        
        model_path = os.path.join(self.storage_dir, f"{model_id}.joblib")
        joblib.dump(model.model, model_path)
        
        self.metadata[model_id] = {
            "model_id": model_id,
            "model_class_name": model_class_name,
            "hyperparameters": hyperparameters,
            "model_path": model_path,
            "created_at": datetime.now().isoformat(),
            "is_trained": model.is_trained
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
        
        from models.registry import ModelRegistry
        
        model = ModelRegistry.create_model(model_class_name)
        
        model.model = joblib.load(metadata["model_path"])
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

