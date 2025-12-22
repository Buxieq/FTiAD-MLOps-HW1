"""Юнит тесты для S3 с моками"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import joblib
import numpy as np
from storage.model_storage import ModelStorage
from models.linear_regression import LinearRegressionModel


@pytest.fixture
def temp_storage_dir():
    """Фикстура для создания временной директории"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_s3_storage():
    """Фикстура для мока S3"""
    mock_s3 = MagicMock()
    mock_s3.upload_file.return_value = True
    mock_s3.download_fileobj.return_value = None
    mock_s3.delete_file.return_value = True
    return mock_s3


class TestModelStorageWithS3Mocks:
   
    @patch('storage.model_storage.S3Storage')
    def test_save_model_with_s3(self, mock_s3_storage_class, temp_storage_dir):
        """Тест сохранения модели в S3"""
        mock_s3 = MagicMock()
        mock_s3.upload_file.return_value = True
        mock_s3_storage_class.return_value = mock_s3
        
        storage = ModelStorage(storage_dir=temp_storage_dir, use_s3=True)
        storage.s3_storage = mock_s3
        
        model = LinearRegressionModel()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        model.train(X, y, {"fit_intercept": True})
        
        model_id = storage.save_model(
            model=model,
            model_class_name="LinearRegression",
            hyperparameters={"fit_intercept": True}
        )
        
        assert model_id is not None
        assert storage.model_exists(model_id)
        
        assert mock_s3.upload_file.called
    
    @patch('storage.model_storage.S3Storage')
    @patch('storage.model_storage.ModelRegistry')
    def test_load_model_from_s3(self, mock_registry, mock_s3_storage_class, temp_storage_dir):
        """Тест загрузки модели из S3"""
        mock_s3 = MagicMock()
        
        model = LinearRegressionModel()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        model.train(X, y, {"fit_intercept": True})
        
        import io
        model_bytes = io.BytesIO()
        joblib.dump(model.model, model_bytes)
        model_bytes.seek(0)
        
        mock_s3.download_fileobj.return_value = model_bytes
        mock_s3_storage_class.return_value = mock_s3
        
        mock_registry.create_model.return_value = LinearRegressionModel()
        
        storage = ModelStorage(storage_dir=temp_storage_dir, use_s3=True)
        storage.s3_storage = mock_s3
        
        model_id = storage.save_model(
            model=model,
            model_class_name="LinearRegression",
            hyperparameters={"fit_intercept": True}
        )
        
        storage.metadata[model_id]["model_path"] = f"s3://test-bucket/models/{model_id}.joblib"
        
        loaded_model, metadata = storage.load_model(model_id)
        
        assert loaded_model is not None
        assert loaded_model.is_trained
        assert mock_s3.download_fileobj.called
        mock_registry.create_model.assert_called_once_with("LinearRegression")
        
        predictions = loaded_model.predict(X)
        expected_predictions = model.predict(X)
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

