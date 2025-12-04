"""Скрипт gRPC клиента для тестирования сервиса"""

import grpc
import sys
import os
import json
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.grpc import service_pb2, service_pb2_grpc


class MLModelClient:
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        """
        Args:
            host: Хост сервера
            port: Порт сервера
        """
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = service_pb2_grpc.MLModelServiceStub(self.channel)
    
    def close(self):
        self.channel.close()
    
    def health_check(self) -> Dict[str, str]:
        """
        Проверить работоспособность сервиса.
        
        Returns:
            Ответ о проверке работоспособности
        """
        print("Checking service health...")
        try:
            response = self.stub.HealthCheck(service_pb2.HealthCheckRequest())
            result = {
                "status": response.status,
                "message": response.message
            }
            print(f"Проверка работоспособности: {result['status']} - {result['message']}")
            return result
        except grpc.RpcError as e:
            print(f"Проверка не удалась: {e.code()} - {e.details()}")
            return {"status": "error", "message": str(e)}
    
    def list_model_classes(self) -> List[Dict[str, Any]]:
        """
        Returns:
            Список доступных классов моделей
        """
        print("Listing available model classes...")
        try:
            response = self.stub.ListModelClasses(service_pb2.ListModelClassesRequest())
            result = []
            for model_info in response.available_models:
                hyperparams = {}
                for param_name, param_info in model_info.hyperparameter_schema.items():
                    hyperparams[param_name] = {
                        "type": param_info.type,
                        "default": param_info.default_value,
                        "description": param_info.description
                    }
                result.append({
                    "model_class": model_info.model_class,
                    "hyperparameter_schema": hyperparams
                })
            print(f"Найдено {len(result)} доступных классов моделей:")
            for model in result:
                print(f"  - {model['model_class']}")
            return result
        except grpc.RpcError as e:
            print(f"Не удалось получить список классов моделей: {e.code()} - {e.details()}")
            return []
    
    def train_model(self, model_class: str, hyperparameters: Dict[str, Any],
                   X: List[List[float]], y: List[float]) -> Dict[str, Any]:
        """
        Args:
            model_class: Название класса модели
            hyperparameters: Словарь гиперпараметров
            X: Признаки для обучения
            y: Целевые значения для обучения
            
        Returns:
            Ответ об обучении
        """
        print(f"Training {model_class} model...")
        try:
            hyperparams_str = {k: json.dumps(v) for k, v in hyperparameters.items()}
            
            data_points = [service_pb2.DataPoint(values=row) for row in X]
            
            request = service_pb2.TrainModelRequest(
                model_class=model_class,
                hyperparameters=hyperparams_str,
                X=data_points,
                y=y
            )
            
            response = self.stub.TrainModel(request)
            result = {
                "model_id": response.model_id,
                "model_class": response.model_class,
                "hyperparameters": {k: json.loads(v) for k, v in response.hyperparameters.items()},
                "message": response.message
            }
            print(f"Модель обучена. Model ID: {result['model_id']}")
            return result
        except grpc.RpcError as e:
            print(f"Обучение не удалось: {e.code()} - {e.details()}")
            return {}
    
    def predict(self, model_id: str, X: List[List[float]]) -> List[float]:
        """
        Args:
            model_id: Model ID
            X: Признаки для прогнозирования
            
        Returns:
            Прогнозы
        """
        print(f"Getting predictions from model {model_id}...")
        try:
            data_points = [service_pb2.DataPoint(values=row) for row in X]
            
            request = service_pb2.PredictRequest(
                model_id=model_id,
                X=data_points
            )
            
            response = self.stub.Predict(request)
            print(f"Прогнозы сгенерированы: {len(response.predictions)} прогнозов")
            return list(response.predictions)
        except grpc.RpcError as e:
            print(f"Прогнозирование не удалось: {e.code()} - {e.details()}")
            return []
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
Returns:
            Список обученных моделей
        """
        print("Listing all trained models...")
        try:
            response = self.stub.ListModels(service_pb2.ListModelsRequest())
            result = []
            for model_info in response.models:
                result.append({
                    "model_id": model_info.model_id,
                    "model_class_name": model_info.model_class_name,
                    "hyperparameters": {k: json.loads(v) for k, v in model_info.hyperparameters.items()},
                    "created_at": model_info.created_at,
                    "is_trained": model_info.is_trained
                })
            print(f"Найдено {len(result)} обученных моделей")
            return result
        except grpc.RpcError as e:
            print(f"Не удалось получить список моделей: {e.code()} - {e.details()}")
            return []
    
    def retrain_model(self, model_id: str, model_class: str, hyperparameters: Dict[str, Any],
                     X: List[List[float]], y: List[float]) -> Dict[str, Any]:
        """
        Args:
            model_id: Model ID
            model_class: Название класса модели
            hyperparameters: Словарь гиперпараметров
            X: Признаки для обучения
            y: Целевые значения для обучения
            
        Returns:
            Ответ об обучении
        """
        print(f"Retraining model {model_id}...")
        try:
            hyperparams_str = {k: json.dumps(v) for k, v in hyperparameters.items()}
            data_points = [service_pb2.DataPoint(values=row) for row in X]
            
            request = service_pb2.RetrainModelRequest(
                model_id=model_id,
                model_class=model_class,
                hyperparameters=hyperparams_str,
                X=data_points,
                y=y
            )
            
            response = self.stub.RetrainModel(request)
            result = {
                "model_id": response.model_id,
                "model_class": response.model_class,
                "hyperparameters": {k: json.loads(v) for k, v in response.hyperparameters.items()},
                "message": response.message
            }
            print(f"Модель  переобучена. Model ID: {result['model_id']}")
            return result
        except grpc.RpcError as e:
            print(f"Переобучение не удалось: {e.code()} - {e.details()}")
            return {}
    
    def delete_model(self, model_id: str) -> bool:
        """
        Args:
            model_id: Model ID

        """
        print(f"Deleting model {model_id}...")
        try:
            request = service_pb2.DeleteModelRequest(model_id=model_id)
            response = self.stub.DeleteModel(request)
            print(f"Модель  удалена: {response.message}")
            return True
        except grpc.RpcError as e:
            print(f"Удаление не удалось: {e.code()} - {e.details()}")
            return False


def main():
    print("=" * 60)
    print("gRPC ML Model Service Client")
    print("=" * 60)
    
    client = MLModelClient()
    
    try:
        print("\n1. Health Check")
        client.health_check()
        
        print("\n2. List Available Model Classes")
        print("-" * 60)
        available_models = client.list_model_classes()
        

        # тут просто на цифирках проверяем что все работает
        if available_models:
            print("\n3. Train Model")
            print("-" * 60)
            model_class = available_models[0]["model_class"]
            X_train = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
            y_train = [3.0, 5.0, 7.0, 9.0]
            hyperparameters = {}
            
            train_result = client.train_model(model_class, hyperparameters, X_train, y_train)
            
            if train_result:
                model_id = train_result["model_id"]
                
                print("\n4. List All Trained Models")
                print("-" * 60)
                client.list_models()
                
                print("\n5. Get Predictions")
                print("-" * 60)
                X_pred = [[5.0, 6.0], [6.0, 7.0]]
                predictions = client.predict(model_id, X_pred)
                if predictions:
                    print(f"Predictions: {predictions}")
                
                print("\n6. Retrain Model")
                print("-" * 60)
                X_retrain = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
                y_retrain = [3.0, 5.0, 7.0]
                client.retrain_model(model_id, model_class, hyperparameters, X_retrain, y_retrain)
                
                print("\n7. Delete Model")
                print("-" * 60)
                client.delete_model(model_id)
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        client.close()
        print("Client closed")


if __name__ == "__main__":
    main()

