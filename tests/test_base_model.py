"""Юнит тесты для BaseModel - обычные"""

import pytest
import numpy as np
from models.linear_regression import LinearRegressionModel
from models.base import BaseModel


class TestBaseModel:
    """Тесты для BaseModel без моков."""
    
    def test_model_initialization(self):
        model = LinearRegressionModel()
        
        assert model.model_name == "LinearRegression"
        assert model.model is None
        assert model.is_trained is False
    
    def test_model_training(self):
        model = LinearRegressionModel()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        
        model.train(X, y, {"fit_intercept": True})
        
        assert model.is_trained is True
        assert model.model is not None
    
    def test_model_predict(self):
        model = LinearRegressionModel()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        
        model.train(X_train, y_train, {"fit_intercept": True})
        
        X_test = np.array([[7, 8], [9, 10]])
        predictions = model.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == 2
        assert isinstance(predictions, np.ndarray)
    
    def test_get_model_info(self):
        model = LinearRegressionModel()
        
        info = model.get_model_info()
        
        assert info["model_name"] == "LinearRegression"
        assert info["is_trained"] is False
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        model.train(X, y, {"fit_intercept": True})
        
        info = model.get_model_info()
        assert info["is_trained"] is True
    
    def test_hyperparameter_schema(self):
        model = LinearRegressionModel()
        
        schema = model.get_hyperparameter_schema()
        
        assert "fit_intercept" in schema
        assert "n_jobs" in schema
        assert schema["fit_intercept"]["type"] == bool
        assert schema["fit_intercept"]["default"] is True

