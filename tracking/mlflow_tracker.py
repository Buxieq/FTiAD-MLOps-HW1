"""MLFlow"""

from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger()


class MLFlowTracker:
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None):
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLFlow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLFlow experiment: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to set up MLFlow experiment: {str(e)}")
            experiment_id = None
        
        self.experiment_id = experiment_id
        self.current_run = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        try:
            mlflow.set_experiment(self.experiment_name)
            self.current_run = mlflow.start_run(run_name=run_name, tags=tags or {})
            run_id = self.current_run.info.run_id
            logger.info(f"Started MLFlow run: {run_id}")
            return run_id
        except Exception as e:
            logger.error(f"Failed to start MLFlow run: {str(e)}")
            return None
    
    def end_run(self) -> None:
        if self.current_run:
            try:
                mlflow.end_run()
                logger.info("Ended MLFlow run")
            except Exception as e:
                logger.error(f"Failed to end MLFlow run: {str(e)}")
            finally:
                self.current_run = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        if self.current_run:
            try:
                mlflow.log_params(params)
                logger.debug(f"Logged parameters: {params}")
            except Exception as e:
                logger.error(f"Failed to log parameters: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.current_run:
            try:
                mlflow.log_metrics(metrics, step=step)
                logger.debug(f"Logged metrics: {metrics}")
            except Exception as e:
                logger.error(f"Failed to log metrics: {str(e)}")
    
    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        if self.current_run:
            try:
                mlflow.sklearn.log_model(model, artifact_path)
                logger.info(f"Logged model to MLFlow: {artifact_path}")
            except Exception as e:
                logger.error(f"Failed to log model: {str(e)}")
    
    def calculate_and_log_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))
        }
        self.log_metrics(metrics)
        return metrics
    
    def log_dataset_info(self, n_samples: int, n_features: int, dataset_path: Optional[str] = None) -> None:
        """
        Args:
            n_samples: Number of samples
            n_features: Number of features
            dataset_path: Optional path to dataset
        """
        if self.current_run:
            try:
                mlflow.log_params({
                    "dataset_n_samples": n_samples,
                    "dataset_n_features": n_features
                })
                if dataset_path:
                    mlflow.log_param("dataset_path", dataset_path)
            except Exception as e:
                logger.error(f"Failed to log dataset info: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()

