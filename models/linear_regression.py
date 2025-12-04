"""Реализация ЛинРег"""

from typing import Any, Dict
from sklearn.linear_model import LinearRegression
from models.base import BaseModel


class LinearRegressionModel(BaseModel):
    
    def __init__(self):
        super().__init__("LinearRegression")
    
    def get_hyperparameter_schema(self) -> Dict[str, Any]:
        """
        Returns:
            Словарь со схемой гиперпараметров
        """
        return {
            "fit_intercept": {
                "type": bool,
                "default": True,
                "description": "Whether to calculate the intercept for this model"
            },
            "n_jobs": {
                "type": int,
                "default": None,
                "description": "Number of jobs to use for computation"
            }
        }
    
    def _create_model(self, hyperparameters: Dict[str, Any]) -> Any:
        """
        Args:
            hyperparameters: Словарь гиперпараметров
            
        Returns:
            Экземпляр модели LinearRegression
        """
        schema = self.get_hyperparameter_schema()
        fit_intercept = hyperparameters.get("fit_intercept", schema["fit_intercept"]["default"])
        n_jobs = hyperparameters.get("n_jobs", schema["n_jobs"]["default"])
        
        model_params = {
            "fit_intercept": fit_intercept,
            "n_jobs": n_jobs
        }
        model_params = {k: v for k, v in model_params.items() if v is not None}
        
        return LinearRegression(**model_params)

