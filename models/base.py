"""Базовый класс для моделей"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Абстрактный базовый класс"""
    
    def __init__(self, model_name: str):
        """
        Args:
            model_name: Название класса модели
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def get_hyperparameter_schema(self) -> Dict[str, Any]:
        """
        Returns:
            Словарь со схемой гиперпараметров
        """
        pass
    
    @abstractmethod
    def _create_model(self, hyperparameters: Dict[str, Any]) -> Any:
        """
        Args:
            hyperparameters: Словарь гиперпараметров
            
        Returns:
            Экземпляр модели
        """
        pass
    
    def train(self, X: np.ndarray, y: np.ndarray, hyperparameters: Dict[str, Any]) -> None:
        """
        Args:
            X: Признаки для обучения (n_samples, n_features)
            y: таргет (n_samples,)
            hyperparameters: Словарь гиперпараметров для модели
        """
        self.model = self._create_model(hyperparameters)
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
Args:
            X: Признаки для прогнозирования (n_samples, n_features)
            
        Returns:
            Прогнозы (n_samples,)
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise ValueError(f"Model {self.model_name} is not trained yet")
        return self.model.predict(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns:
            Словарь с информацией о модели
        """
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained
        }

