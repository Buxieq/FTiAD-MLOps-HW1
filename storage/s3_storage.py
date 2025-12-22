"""S3 storage for models and files using boto3."""

import os
import io
from typing import Optional, List
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger()


class S3Storage:
       
    def __init__(self, bucket_name: Optional[str] = None, **kwargs):
        self.bucket_name = bucket_name or settings.S3_BUCKET
        s3_config = settings.get_s3_config()
        s3_config.update(kwargs)
        
        self.client = boto3.client('s3', **s3_config)
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                try:
                    if settings.S3_ENDPOINT.startswith('http://localhost') or 'minio' in settings.S3_ENDPOINT.lower():
                        self.client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': settings.S3_REGION}
                        )
                    logger.info(f"Created bucket {self.bucket_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket {self.bucket_name}: {str(create_error)}")
                    raise
            else:
                logger.error(f"Error checking bucket {self.bucket_name}: {str(e)}")
                raise
    
    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """
        Args:
            local_path: Path to local file
            s3_key: S3 object key
        """
        try:
            self.client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to upload {local_path} to S3: {str(e)}")
            return False
    
    def upload_fileobj(self, file_obj: io.BytesIO, s3_key: str, content_type: Optional[str] = None) -> bool:
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.client.upload_fileobj(file_obj, self.bucket_name, s3_key, ExtraArgs=extra_args)
            logger.info(f"Uploaded file object to s3://{self.bucket_name}/{s3_key}")
            return True
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to upload file object to S3: {str(e)}")
            return False
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Args:
            s3_key: S3 object key (path in bucket)
            local_path: Path to save the file locally
        """
        try:
            os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else '.', exist_ok=True)
            self.client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to download {s3_key} from S3: {str(e)}")
            return False
    
    def download_fileobj(self, s3_key: str) -> Optional[io.BytesIO]:
        """
        Args:
            s3_key: S3 object key
            local_path: Path to save the file locally
        """
        try:
            file_obj = io.BytesIO()
            self.client.download_fileobj(self.bucket_name, s3_key, file_obj)
            file_obj.seek(0)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to memory")
            return file_obj
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to download {s3_key} from S3: {str(e)}")
            return None
    
    def delete_file(self, s3_key: str) -> bool:
        """
            s3_key: S3 object key
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to delete {s3_key} from S3: {str(e)}")
            return False
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.
            s3_key: S3 object key
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                return False
            logger.error(f"Error checking file existence in S3: {str(e)}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """
            prefix: Prefix to filter files
        """
        try:
            response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' not in response:
                return []
            return [obj['Key'] for obj in response['Contents']]
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to list files in S3: {str(e)}")
            return []
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
            s3_key: S3 object key
            expiration: URL expiration time in seconds
        """
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}")
            return None

