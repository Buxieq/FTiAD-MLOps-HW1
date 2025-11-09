"""gRPC сервер"""

import grpc
from concurrent import futures
import numpy as np
from typing import Dict, Any
import json

from api.grpc import service_pb2, service_pb2_grpc
from models.registry import ModelRegistry
from storage.model_storage import ModelStorage
from utils.logger import setup_logger

logger = setup_logger()


class MLModelServicer(service_pb2_grpc.MLModelServiceServicer):
    
    def __init__(self):
        self.model_storage = ModelStorage()
    
    def _convert_hyperparameters(self, hyperparams_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Args:
            hyperparams_dict: Словарь строковых гиперпараметров
            
        Returns:
            Словарь с преобразованными типами
        """
        result = {}
        for key, value in hyperparams_dict.items():
            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                if value.lower() == "true":
                    result[key] = True
                elif value.lower() == "false":
                    result[key] = False
                elif value.lower() == "none" or value.lower() == "null":
                    result[key] = None
                else:
                    try:
                        if "." not in value:
                            result[key] = int(value)
                        else:
                            result[key] = float(value)
                    except ValueError:
                        result[key] = value
        return result
    
    def _convert_hyperparameters_to_strings(self, hyperparams_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Args:
            hyperparams_dict: Словарь гиперпараметров
            
        Returns:
            Словарь со строковыми значениями
        """
        return {k: json.dumps(v) if not isinstance(v, str) else v for k, v in hyperparams_dict.items()}
    
    def ListModelClasses(self, request, context):
        logger.info("gRPC: Listing available model classes")
        try:
            available_models = ModelRegistry.list_available_models()
            model_classes = []
            
            for model_name, schema_info in available_models.items():
                hyperparameter_schema = {}
                for param_name, param_info in schema_info["hyperparameter_schema"].items():
                    hyperparameter_schema[param_name] = service_pb2.HyperparameterInfo(
                        type=param_info["type"].__name__ if hasattr(param_info["type"], "__name__") else str(param_info["type"]),
                        default_value=json.dumps(param_info["default"]) if param_info["default"] is not None else "",
                        description=param_info.get("description", "")
                    )
                
                model_classes.append(service_pb2.ModelClassInfo(
                    model_class=model_name,
                    hyperparameter_schema=hyperparameter_schema
                ))
            
            logger.info(f"gRPC: Found {len(model_classes)} available model classes")
            return service_pb2.ListModelClassesResponse(available_models=model_classes)
        except Exception as e:
            logger.error(f"gRPC: Error listing model classes: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return service_pb2.ListModelClassesResponse()
    
    def TrainModel(self, request, context):
        logger.info(f"gRPC: Training request received for model class: {request.model_class}")
        try:
            try:
                model = ModelRegistry.create_model(request.model_class)
            except ValueError as e:
                logger.error(f"gRPC: Invalid model class: {str(e)}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(e))
                return service_pb2.TrainModelResponse()
            
            try:
                X = np.array([list(point.values) for point in request.X])
                y = np.array(request.y)
                
                if len(X.shape) != 2:
                    raise ValueError("X must be a 2D array")
                if len(y.shape) != 1:
                    raise ValueError("y must be a 1D array")
                if X.shape[0] != y.shape[0]:
                    raise ValueError("X and y must have the same number of samples")
            except Exception as e:
                logger.error(f"gRPC: Invalid data format: {str(e)}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Invalid data format: {str(e)}")
                return service_pb2.TrainModelResponse()
            
            hyperparameters = self._convert_hyperparameters(dict(request.hyperparameters))
            
            try:
                logger.info(f"gRPC: Training model with hyperparameters: {hyperparameters}")
                model.train(X, y, hyperparameters)
                logger.info("gRPC: Model training completed successfully")
            except Exception as e:
                logger.error(f"gRPC: Error during model training: {str(e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return service_pb2.TrainModelResponse()
            
            try:
                model_id = self.model_storage.save_model(
                    model=model,
                    model_class_name=request.model_class,
                    hyperparameters=hyperparameters
                )
                logger.info(f"gRPC: Model saved with ID: {model_id}")
            except Exception as e:
                logger.error(f"gRPC: Error saving model: {str(e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return service_pb2.TrainModelResponse()
            
            return service_pb2.TrainModelResponse(
                model_id=model_id,
                model_class=request.model_class,
                hyperparameters=self._convert_hyperparameters_to_strings(hyperparameters),
                message="Model trained successfully"
            )
        except Exception as e:
            logger.error(f"gRPC: Unexpected error during training: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return service_pb2.TrainModelResponse()
    
    def Predict(self, request, context):
        logger.info(f"gRPC: Prediction request received for model: {request.model_id}")
        try:
            try:
                model, metadata = self.model_storage.load_model(request.model_id)
                logger.info(f"gRPC: Model {request.model_id} loaded successfully")
            except ValueError as e:
                logger.error(f"gRPC: Model not found: {str(e)}")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(e))
                return service_pb2.PredictResponse()
            
            try:
                X = np.array([list(point.values) for point in request.X])
                if len(X.shape) != 2:
                    raise ValueError("X must be a 2D array")
            except Exception as e:
                logger.error(f"gRPC: Invalid data format: {str(e)}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Invalid data format: {str(e)}")
                return service_pb2.PredictResponse()
            
            try:
                predictions = model.predict(X)
                logger.info(f"gRPC: Predictions generated successfully for model {request.model_id}")
            except Exception as e:
                logger.error(f"gRPC: Error during prediction: {str(e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return service_pb2.PredictResponse()
            
            return service_pb2.PredictResponse(
                predictions=predictions.tolist(),
                model_id=request.model_id
            )
        except Exception as e:
            logger.error(f"gRPC: Unexpected error during prediction: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return service_pb2.PredictResponse()
    
    def RetrainModel(self, request, context):
        logger.info(f"gRPC: Retrain request received for model: {request.model_id}")
        try:
            if not self.model_storage.model_exists(request.model_id):
                logger.error(f"gRPC: Model {request.model_id} not found for retraining")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Model {request.model_id} not found")
                return service_pb2.TrainModelResponse()
            
            try:
                self.model_storage.delete_model(request.model_id)
                logger.info(f"gRPC: Old model {request.model_id} deleted")
            except Exception as e:
                logger.error(f"gRPC: Error deleting old model: {str(e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return service_pb2.TrainModelResponse()
            
            try:
                X = np.array([list(point.values) for point in request.X])
                y = np.array(request.y)
                
                if len(X.shape) != 2:
                    raise ValueError("X must be a 2D array")
                if len(y.shape) != 1:
                    raise ValueError("y must be a 1D array")
                if X.shape[0] != y.shape[0]:
                    raise ValueError("X and y must have the same number of samples")
            except Exception as e:
                logger.error(f"gRPC: Invalid data format: {str(e)}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Invalid data format: {str(e)}")
                return service_pb2.TrainModelResponse()
            
            hyperparameters = self._convert_hyperparameters(dict(request.hyperparameters))
            
            try:
                model = ModelRegistry.create_model(request.model_class)
                model.train(X, y, hyperparameters)
                logger.info("gRPC: Model retraining completed successfully")
            except Exception as e:
                logger.error(f"gRPC: Error during model retraining: {str(e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return service_pb2.TrainModelResponse()
            
            try:
                new_model_id = self.model_storage.save_model(
                    model=model,
                    model_class_name=request.model_class,
                    hyperparameters=hyperparameters,
                    model_id=request.model_id
                )
                logger.info(f"gRPC: Retrained model saved with ID: {new_model_id}")
            except Exception as e:
                logger.error(f"gRPC: Error saving retrained model: {str(e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return service_pb2.TrainModelResponse()
            
            return service_pb2.TrainModelResponse(
                model_id=new_model_id,
                model_class=request.model_class,
                hyperparameters=self._convert_hyperparameters_to_strings(hyperparameters),
                message="Model retrained successfully"
            )
        except Exception as e:
            logger.error(f"gRPC: Unexpected error during retraining: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return service_pb2.TrainModelResponse()
    
    def DeleteModel(self, request, context):
        logger.info(f"gRPC: Delete request received for model: {request.model_id}")
        try:
            if not self.model_storage.model_exists(request.model_id):
                logger.error(f"gRPC: Model {request.model_id} not found for deletion")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Model {request.model_id} not found")
                return service_pb2.DeleteModelResponse()
            
            self.model_storage.delete_model(request.model_id)
            logger.info(f"gRPC: Model {request.model_id} deleted successfully")
            return service_pb2.DeleteModelResponse(message="Model deleted successfully")
        except Exception as e:
            logger.error(f"gRPC: Error deleting model: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return service_pb2.DeleteModelResponse()
    
    def ListModels(self, request, context):
        logger.info("gRPC: Listing all trained models")
        try:
            models_data = self.model_storage.list_models()
            models = []
            
            for model_data in models_data:
                models.append(service_pb2.ModelInfo(
                    model_id=model_data["model_id"],
                    model_class_name=model_data["model_class_name"],
                    hyperparameters=self._convert_hyperparameters_to_strings(model_data["hyperparameters"]),
                    created_at=model_data["created_at"],
                    is_trained=model_data["is_trained"]
                ))
            
            logger.info(f"gRPC: Found {len(models)} trained models")
            return service_pb2.ListModelsResponse(models=models)
        except Exception as e:
            logger.error(f"gRPC: Error listing models: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return service_pb2.ListModelsResponse()
    
    def HealthCheck(self, request, context):
        logger.info("gRPC: Health check requested")
        return service_pb2.HealthCheckResponse(
            status="healthy",
            message="Service is running"
        )


def serve(port: int = 50051):
    """
    Args:
        port: Порт для запуска сервера
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_MLModelServiceServicer_to_server(MLModelServicer(), server)
    listen_addr = f'localhost:{port}'
    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"gRPC server started on port {port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC server shutting down")
        server.stop(0)


if __name__ == '__main__':
    serve()

