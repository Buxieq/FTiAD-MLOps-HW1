"""Реестр моделей"""

from typing import Dict, Type
from models.base import BaseModel
from models.linear_regression import LinearRegressionModel
from models.random_forest import RandomForestModel


class ModelRegistry:
    """Реестр доступных классов"""
    
    _models: Dict[str, Type[BaseModel]] = {
        "LinearRegression": LinearRegressionModel,
        "RandomForest": RandomForestModel
    }
    
    @classmethod
    def get_model_class(cls, model_name: str) -> Type[BaseModel]:
        """
        Args:
            model_name: Название класса модели
            
        Returns:
            Класс модели
            
        Raises:
            ValueError: Если класс модели не найден
        """
        if model_name not in cls._models:
            raise ValueError(f"Model class '{model_name}' not found. Available: {list(cls._models.keys())}")
        return cls._models[model_name]
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict]:
        """
        Returns:
            Словарь, сопоставляющий названия моделей с их схемами гиперпараметров
        """
        result = {}
        for model_name, model_class in cls._models.items():
            instance = model_class()
            result[model_name] = {
                "hyperparameter_schema": instance.get_hyperparameter_schema()
            }
        return result
    
    @classmethod
    def create_model(cls, model_name: str) -> BaseModel:
        """
        Args:
            model_name: Название класса модели
            
        Returns:
            Экземпляр модели
        """
        model_class = cls.get_model_class(model_name)
        return model_class()

