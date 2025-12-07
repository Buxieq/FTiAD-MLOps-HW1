
import os
import sys
import subprocess
from pathlib import Path
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger()


def init_dvc():
    repo_root = Path.cwd()
    
    dvc_dir = repo_root / ".dvc"
    if dvc_dir.exists() and (dvc_dir / "config").exists():
        logger.info("DVC is already initialized")
        response = input("DVC is already initialized. Reinitialize? (y/N): ")
        if response.lower() != 'y':
            logger.info("Skipping DVC initialization")
            return
    
    try:
        logger.info("Initializing DVC repository...")
        subprocess.run(["dvc", "init"], check=True, cwd=repo_root)
        logger.info("DVC repository initialized")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to initialize DVC: {str(e)}")
        sys.exit(1)
    
    remote_name = settings.DVC_REMOTE
    remote_url = f"s3://{settings.S3_DATASET_BUCKET}"
    
    try:
        subprocess.run(
            ["dvc", "remote", "remove", remote_name],
            cwd=repo_root,
            capture_output=True
        )
        
        logger.info(f"Adding DVC remote: {remote_name} -> {remote_url}")
        subprocess.run(
            ["dvc", "remote", "add", remote_name, remote_url],
            check=True,
            cwd=repo_root
        )
        
        subprocess.run(
            ["dvc", "remote", "modify", remote_name, "endpointurl", settings.S3_ENDPOINT],
            check=True,
            cwd=repo_root
        )
        subprocess.run(
            ["dvc", "remote", "modify", remote_name, "access_key_id", settings.S3_ACCESS_KEY],
            check=True,
            cwd=repo_root
        )
        subprocess.run(
            ["dvc", "remote", "modify", remote_name, "secret_access_key", settings.S3_SECRET_KEY],
            check=True,
            cwd=repo_root
        )
        
        logger.info("DVC remote configured successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to configure DVC remote: {str(e)}")
        sys.exit(1)
    
    dvcignore_path = repo_root / ".dvcignore"
    if not dvcignore_path.exists():
        dvcignore_content = """__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/
dist/
build/

myenv/

.vscode/

.ipynb_checkpoints/

.pytest_cache/
.coverage

*.log

Thumbs.db
"""
        with open(dvcignore_path, 'w') as f:
            f.write(dvcignore_content)
        logger.info("Created .dvcignore file")
    
    logger.info("DVC initialization completed successfully")
    logger.info(f"Remote: {remote_name} -> {remote_url}")
    logger.info(f"Endpoint: {settings.S3_ENDPOINT}")


if __name__ == "__main__":
    init_dvc()

