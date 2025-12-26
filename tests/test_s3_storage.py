"""Юнит тесты для S3 с моками и фикстурами"""

import pytest
import io
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, BotoCoreError
from storage.s3_storage import S3Storage


@pytest.fixture
def mock_boto3_client():
    """Фикстура для мока boto3 S3 клиента"""
    with patch('storage.s3_storage.boto3.client') as mock_client:
        mock_s3_client = MagicMock()
        mock_client.return_value = mock_s3_client
        yield mock_s3_client


@pytest.fixture
def mock_settings():
    with patch('storage.s3_storage.settings') as mock_settings:
        mock_settings.S3_BUCKET = "test-bucket"
        mock_settings.get_s3_config.return_value = {
            "endpoint_url": "http://localhost:9000",
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
            "use_ssl": False
        }
        mock_settings.S3_ENDPOINT = "http://localhost:9000"
        yield mock_settings


class TestS3StorageWithMocks:
    """Тесты для S3Storage с моками"""
    
    def test_upload_file_success(self, mock_boto3_client, mock_settings):
        """Тест успешного загрузки файла в S3"""
        storage = S3Storage(bucket_name="test-bucket")
        mock_boto3_client.upload_file.return_value = None
        
        result = storage.upload_file("local_file.txt", "s3_key.txt")
        
        assert result is True
        mock_boto3_client.upload_file.assert_called_once_with(
            "local_file.txt", "test-bucket", "s3_key.txt"
        )
    
    def test_upload_file_failure(self, mock_boto3_client, mock_settings):
        """Тест обработки неудачной загрузки файла"""
        storage = S3Storage(bucket_name="test-bucket")
        
        mock_boto3_client.upload_file.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchBucket', 'Message': 'Bucket not found'}},
            'UploadFile'
        )
        
        result = storage.upload_file("local_file.txt", "s3_key.txt")
        
        assert result is False
    
    def test_download_fileobj_success(self, mock_boto3_client, mock_settings):
        """Тест успешного скачивания объекта файла из S3"""
        storage = S3Storage(bucket_name="test-bucket")
        
        test_data = b"test file content"
        mock_boto3_client.download_fileobj.return_value = None
        
        with patch('io.BytesIO', return_value=io.BytesIO(test_data)):
            result = storage.download_fileobj("s3_key.txt")
        
        assert result is not None
        assert isinstance(result, io.BytesIO)
        mock_boto3_client.download_fileobj.assert_called_once()
    
    def test_delete_file_success(self, mock_boto3_client, mock_settings):
        """Тест успешного удаления файла из S3"""
        storage = S3Storage(bucket_name="test-bucket")
        
        mock_boto3_client.delete_object.return_value = None
        
        result = storage.delete_file("s3_key.txt")
        
        assert result is True
        mock_boto3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="s3_key.txt"
        )
    
    def test_file_exists_true(self, mock_boto3_client, mock_settings):
        """Тест проверки существования файла"""
        storage = S3Storage(bucket_name="test-bucket")
        
        mock_boto3_client.head_object.return_value = {}
        
        result = storage.file_exists("s3_key.txt")
        
        assert result is True
        mock_boto3_client.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="s3_key.txt"
        )
    
    def test_file_exists_false(self, mock_boto3_client, mock_settings):
        """Тест проверки существования файла"""
        storage = S3Storage(bucket_name="test-bucket")
        
        mock_boto3_client.head_object.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}},
            'HeadObject'
        )
        
        result = storage.file_exists("s3_key.txt")
        
        assert result is False
    
    def test_list_files(self, mock_boto3_client, mock_settings):
        """Тест списка файлов в S3 бакете"""
        storage = S3Storage(bucket_name="test-bucket")
        
        mock_boto3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'file1.txt'},
                {'Key': 'file2.txt'},
                {'Key': 'models/model1.joblib'}
            ]
        }
        
        result = storage.list_files(prefix="models/")
        
        assert len(result) == 3
        assert 'file1.txt' in result
        assert 'models/model1.joblib' in result
        mock_boto3_client.list_objects_v2.assert_called_once_with(
            Bucket="test-bucket", Prefix="models/"
        )
    
    def test_get_presigned_url(self, mock_boto3_client, mock_settings):
        """Тест генерации presigned URL"""
        storage = S3Storage(bucket_name="test-bucket")
        
        mock_boto3_client.generate_presigned_url.return_value = "https://test-url.com/file"
        
        result = storage.get_presigned_url("s3_key.txt", expiration=3600)
        
        assert result == "https://test-url.com/file"
        mock_boto3_client.generate_presigned_url.assert_called_once_with(
            'get_object',
            Params={'Bucket': 'test-bucket', 'Key': 's3_key.txt'},
            ExpiresIn=3600
        )
