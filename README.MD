# MLOps ДЗ 1

Проект реализует REST и gRPC API для обучения и использования моделей.  Доступно 2 класса моделей, хранение обученных моделей, дашборд и документация.

Проект выполнен:

- Иванов Илья
- Чалабян Левон
- Ханин Вадим

## Структура проекта

```
hw1/
├── api/
│   ├── rest/                    # REST API
│   │   ├── main.py              # FastAPI приложение
│   │   ├── routes/              # Эндпоинты
│   │   └── schemas.py           # Pydantic схемы
│   └── grpc/                    # gRPC API
│       ├── server.py            # gRPC сервер
│       ├── service.proto        # Protocol Buffers определение
│       └── service_pb2*.py      # Сгенерированные файлы
├── models/                      # ML модели
│   ├── base.py                  # Базовый класс модели
│   ├── linear_regression.py     # Linear Regression модель
│   ├── random_forest.py         # Random Forest модель
│   └── registry.py              # Реестр моделей
├── storage/                     # Хранилище моделей
│   └── model_storage.py       
├── utils/                     
│   └── logger.py                # Логирование
├── dashboard/                   # Дашборд
│   └── app.py                 
├── clients/                     # для тестирования
│   ├── grpc_client.py           # скрипт
│   └── grpc_client.ipynb        # Jupyter ноутбук
├── pyproject.toml               # Poetry
└── README.md                  
```

## Установка

### Требования

- Python 3.10+
- Poetry

### Зависимости

```
poetry install
```

### Генерация gRPC кода

```
python generate_grpc_code.py
```

## Запуск

### REST API

```
poetry run uvicorn api.rest.main:app --host 0.0.0.0 --port 8000
```

Swagger документация: `http://localhost:8000/docs`
ReDoc документация: `http://localhost:8000/redoc`

### gRPC сервер

```
poetry run python run_grpc_server.py --port 50051
```

gRPC сервер: `http://localhost:50051`

### Дашборд

```
poetry run streamlit run dashboard/app.py
```

Дашборд: `http://localhost:8501`

## Использование API

### REST API

#### 1. Проверка статуса

```
curl http://localhost:8000/health
```

#### 2. Список доступных классов моделей

```
curl http://localhost:8000/api/v1/models/list
```

#### 3. Обучение модели

```
curl -X POST http://localhost:8000/api/v1/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_class": "LinearRegression",
    "hyperparameters": {"fit_intercept": true},
    "X": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
    "y": [3.0, 5.0, 7.0]
  }'
```

#### 4. Список обученных моделей

```
curl http://localhost:8000/api/v1/models
```

#### 5. Получение предсказаний

```
curl -X POST http://localhost:8000/api/v1/models/{model_id}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[5.0, 6.0], [6.0, 7.0]]
  }'
```

#### 6. Переобучение модели

```
curl -X POST http://localhost:8000/api/v1/models/{model_id}/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "model_class": "LinearRegression",
    "hyperparameters": {"fit_intercept": true},
    "X": [[1.0, 2.0], [2.0, 3.0]],
    "y": [3.0, 5.0]
  }'
```

#### 7. Удаление модели

```
curl -X DELETE http://localhost:8000/api/v1/models/{model_id}
```

### gRPC API

#### Использование Python клиента

```
python clients/grpc_client.py
```

#### Jupyter ноутбук с примером работы

`clients/grpc_client.ipynb` в Jupyter и ячейки

## Доступные модели

### LinearRegression

Гиперпараметры:

- `fit_intercept` (bool, default: True)
- `n_jobs` (int, default: None)

### RandomForest

Гиперпараметры:

- `n_estimators` (int, default: 100)
- `max_depth` (int, default: None)
- `min_samples_split` (int, default: 2)
- `min_samples_leaf` (int, default: 1)
- `max_features` (str, default: "sqrt")
- `random_state` (int, default: None)

## Логирование

Важные операции логируются через Python logger:
Логи выводятся в консоль с timestamp и уровнями логирования.

- Запросы на обучение/предсказание
- Успешное/неуспешное обучение моделей
- Ошибки и исключения
- Загрузка/сохранение моделей

## Хранение моделей

Обученные модели сохраняются в `models_storage/`. Метаданные:

- ID модели
- Класс модели
- Гиперпараметры
- Дата создания
- Статус обучения

## Тестирование

Для тестирования gRPC API:

```
python clients/grpc_client.py
```

Или Jupyter ноутбук `clients/grpc_client.ipynb`. Тут реализован тест на данных с Kaggle

Для тестирования REST API - Swagger UI `http://localhost:8000/docs` или curl команды выше
