"""DVC"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dvc.repo import Repo
from dvc.exceptions import DvcException
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger()


class DatasetStorage:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or settings.DVC_DATA_DIR
        self._ensure_data_dir()
        
        try:
            self.repo = Repo(".")
            logger.info("DVC repository found")
        except DvcException:
            logger.warning("DVC repository not found. Run init_dvc script first.")
            self.repo = None
    
    def _ensure_data_dir(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_dataset(self, train: np.ndarray, target: np.ndarray, 
                    model_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        if self.repo is None:
            logger.warning("DVC repo not initialized, saving without versioning")
            return self._save_dataset_local(train, target, model_id, metadata), "no-version"
        
        dataset_id = model_id or str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_filename = f"dataset_{dataset_id}_{timestamp}"
        
        dataset_path = os.path.join(self.data_dir, f"{dataset_filename}.parquet")
        
        df = pd.DataFrame(train)
        df['target'] = target
        df.to_parquet(dataset_path, index=False)
        
        metadata_path = os.path.join(self.data_dir, f"{dataset_filename}_metadata.json")
        dataset_metadata = {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "timestamp": timestamp,
            "n_samples": len(train),
            "n_features": train.shape[1] if len(train.shape) > 1 else 1,
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        try:
            self.repo.add(dataset_path)
            self.repo.add(metadata_path)
            
            commit_message = f"Add dataset {dataset_id} for model {model_id or 'unknown'}"
            self.repo.commit(commit_message)
            
            try:
                dvc_version = self.repo.scm.resolve_rev("HEAD")
            except Exception:
                dvc_version = timestamp
            
            logger.info(f"Dataset saved and versioned with DVC: {dataset_path}, version: {dvc_version}")
            return dataset_path, dvc_version
            
        except DvcException as e:
            logger.error(f"Failed to version dataset with DVC: {str(e)}")
            return dataset_path, timestamp
    
    def _save_dataset_local(self, train: np.ndarray, target: np.ndarray,
                           model_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        dataset_id = model_id or str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_filename = f"dataset_{dataset_id}_{timestamp}"
        dataset_path = os.path.join(self.data_dir, f"{dataset_filename}.parquet")
        
        df = pd.DataFrame(train)
        df['target'] = target
        df.to_parquet(dataset_path, index=False)
        
        logger.info(f"Dataset saved locally: {dataset_path}")
        return dataset_path
    
    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset file not found: {dataset_path}")
        
        df = pd.read_parquet(dataset_path)
        target = df['target'].values
        train = df.drop('target', axis=1).values
        
        return train, target
    
    def get_dataset_metadata(self, dataset_path: str) -> Optional[Dict[str, Any]]:
        metadata_path = dataset_path.replace('.parquet', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def list_datasets(self) -> list:
        """
        List all datasets in the data directory
        """
        datasets = []
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.parquet'):
                    datasets.append(os.path.join(self.data_dir, filename))
        return datasets

