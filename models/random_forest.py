"""Random Forest"""

from typing import Any, Dict
from sklearn.ensemble import RandomForestRegressor
from models.base import BaseModel


class RandomForestModel(BaseModel):
    
    def __init__(self):
        super().__init__("RandomForest")
    
    def get_hyperparameter_schema(self) -> Dict[str, Any]:
        """
        Returns:
            Словарь со схемой гиперпараметров
        """
        return {
            "n_estimators": {
                "type": int,
                "default": 100,
                "description": "Number of trees in the forest"
            },
            "max_depth": {
                "type": int,
                "default": None,
                "description": "Maximum depth of the tree"
            },
            "min_samples_split": {
                "type": int,
                "default": 2,
                "description": "Minimum number of samples required to split a node"
            },
            "min_samples_leaf": {
                "type": int,
                "default": 1,
                "description": "Minimum number of samples required at a leaf node"
            },
            "max_features": {
                "type": str,
                "default": "sqrt",
                "description": "Number of features to consider when looking for the best split"
            },
            "random_state": {
                "type": int,
                "default": None,
                "description": "Random state for reproducibility"
            }
        }
    
    def _create_model(self, hyperparameters: Dict[str, Any]) -> Any:
        """
        
        Args:
            hyperparameters: Словарь гиперпараметров
            
        Returns:
            Экземпляр модели RandomForestRegressor
        """
        schema = self.get_hyperparameter_schema()
        n_estimators = hyperparameters.get("n_estimators", schema["n_estimators"]["default"])
        max_depth = hyperparameters.get("max_depth", schema["max_depth"]["default"])
        min_samples_split = hyperparameters.get("min_samples_split", schema["min_samples_split"]["default"])
        min_samples_leaf = hyperparameters.get("min_samples_leaf", schema["min_samples_leaf"]["default"])
        max_features = hyperparameters.get("max_features", schema["max_features"]["default"])
        random_state = hyperparameters.get("random_state", schema["random_state"]["default"])
        
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

