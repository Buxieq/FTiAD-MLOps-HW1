"""Дашборд Streamlit"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Конфигурация
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MLOps Model Dashboard",
    page_icon=None,
    layout="wide"
)

st.title("MLOps Model Dashboard")
st.markdown("---")

if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []


def get_available_models() -> Dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/models/list")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return {"available_models": []}
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return {"available_models": []}


def get_trained_models() -> List[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/models")
        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []


def train_model(model_class: str, hyperparameters: Dict[str, Any], X: List[List[float]], y: List[float]) -> Dict[str, Any]:
    """Обучить модель."""
    try:
        payload = {
            "model_class": model_class,
            "hyperparameters": hyperparameters,
            "X": X,
            "y": y
        }
        response = requests.post(f"{API_BASE_URL}/api/v1/models/train", json=payload)
        if response.status_code == 201:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return {}


def predict(model_id: str, X: List[List[float]]) -> List[float]:
    """Получить прогнозы от модели."""
    try:
        payload = {"X": X}
        response = requests.post(f"{API_BASE_URL}/api/v1/models/{model_id}/predict", json=payload)
        if response.status_code == 200:
            return response.json().get("predictions", [])
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error getting predictions: {str(e)}")
        return []


def delete_model(model_id: str) -> bool:
    """Удалить модель."""
    try:
        response = requests.delete(f"{API_BASE_URL}/api/v1/models/{model_id}")
        return response.status_code == 204
    except Exception as e:
        st.error(f"Error deleting model: {str(e)}")
        return False


def check_health() -> bool:
    """Проверить работоспособность API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except Exception as e:
        return False



with st.sidebar:
    st.header("Конфигурация")
    api_url = st.text_input("API Base URL", value=API_BASE_URL, key="api_url")
    API_BASE_URL = api_url
    
    st.markdown("---")
    st.header("Статус")
    if check_health():
        st.success("API работает")
    else:
        st.error("API не отвечает")
    
    st.markdown("---")
    if st.button("Обновить модели"):
        st.session_state.trained_models = get_trained_models()
        st.rerun()


# Основной контент
tab1, tab2, tab3, tab4 = st.tabs(["Доступные модели", "Обучить модель", "Прогнозы", "Обученные модели"])

# Вкладка 1: Доступные модели
with tab1:
    st.header("Доступные классы моделей")
    st.markdown("Список всех доступных классов моделей для обучения.")
    
    available_models = get_available_models()
    
    if available_models.get("available_models"):
        for model_info in available_models["available_models"]:
            with st.expander(f"{model_info['model_class']}"):
                st.subheader("Hyperparameter Schema")
                hyperparams = model_info.get("hyperparameter_schema", {})
                if hyperparams:
                    df = pd.DataFrame([
                        {
                            "Parameter": param,
                            "Type": info.get("type", "unknown"),
                            "Default": str(info.get("default", "None")),
                            "Description": info.get("description", "")
                        }
                        for param, info in hyperparams.items()
                    ])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No hyperparameters defined")
    else:
        st.info("No available models found. Make sure the API is running.")


# Вкладка 2: Обучить модель
with tab2:
    st.header("Обучить новую модель")
    
    # Получить доступные модели
    available_models = get_available_models()
    model_names = [m["model_class"] for m in available_models.get("available_models", [])]
    
    if not model_names:
        st.warning("Нет доступных классов моделей. Убедитесь, что API запущен.")
    else:
        model_class = st.selectbox("Выбрать класс модели", model_names)
        
        # Получить схему гиперпараметров для выбранной модели
        selected_model_info = next(
            (m for m in available_models["available_models"] if m["model_class"] == model_class),
            None
        )
        
        if selected_model_info:
            st.subheader("Hyperparameters")
            hyperparameters = {}
            hyperparams_schema = selected_model_info.get("hyperparameter_schema", {})
            
            for param_name, param_info in hyperparams_schema.items():
                param_type = param_info.get("type", "str")
                default_value = param_info.get("default")
                
                if param_type == "bool":
                    hyperparameters[param_name] = st.checkbox(
                        param_name,
                        value=default_value if default_value is not None else False,
                        help=param_info.get("description", "")
                    )
                elif param_type == "int":
                    hyperparameters[param_name] = st.number_input(
                        param_name,
                        value=int(default_value) if default_value is not None else 0,
                        step=1,
                        help=param_info.get("description", "")
                    )
                elif param_type == "float":
                    hyperparameters[param_name] = st.number_input(
                        param_name,
                        value=float(default_value) if default_value is not None else 0.0,
                        step=0.1,
                        help=param_info.get("description", "")
                    )
                else:
                    hyperparameters[param_name] = st.text_input(
                        param_name,
                        value=str(default_value) if default_value is not None else "",
                        help=param_info.get("description", "")
                    )
            
            st.subheader("Training Data")
            data_input_method = st.radio("Data Input Method", ["Manual Entry", "CSV Upload", "JSON Upload"])
            
            X = None
            y = None
            
            if data_input_method == "Manual Entry":
                st.markdown("Enter training data as JSON:")
                st.code('{"X": [[1.0, 2.0], [2.0, 3.0]], "y": [3.0, 5.0]}', language="json")
                data_json = st.text_area("Training Data (JSON)", height=200)
                if data_json:
                    try:
                        data = json.loads(data_json)
                        X = data.get("X", [])
                        y = data.get("y", [])
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")
            
            elif data_input_method == "CSV Upload":
                uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df)
                    
                    feature_cols = st.multiselect("Select feature columns", df.columns.tolist())
                    target_col = st.selectbox("Select target column", df.columns.tolist())
                    
                    if feature_cols and target_col:
                        X = df[feature_cols].values.tolist()
                        y = df[target_col].values.tolist()
            
            elif data_input_method == "JSON Upload":
                uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
                if uploaded_file:
                    try:
                        data = json.load(uploaded_file)
                        X = data.get("X", [])
                        y = data.get("y", [])
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")
            
            if X and y:
                st.success(f"Data loaded: {len(X)} samples, {len(X[0]) if X else 0} features")
                
                if st.button("Обучить модель", type="primary"):
                    with st.spinner("Training model..."):
                        result = train_model(model_class, hyperparameters, X, y)
                        if result:
                            st.success(f"Модель обучена")
                            st.json(result)
                            st.session_state.trained_models = get_trained_models()
                            st.rerun()


# Вкладка 3: Прогнозы
with tab3:
    st.header("Получить прогнозы")
    
    trained_models = get_trained_models()
    
    if not trained_models:
        st.info("No trained models available. Train a model first.")
    else:
        model_options = {f"{m['model_id']} ({m['model_class_name']})": m['model_id'] for m in trained_models}
        selected_model_label = st.selectbox("Select Trained Model", list(model_options.keys()))
        selected_model_id = model_options[selected_model_label]
        
        st.subheader("Input Data")
        prediction_input_method = st.radio("Input Method", ["Manual Entry", "CSV Upload", "JSON Upload"])
        
        X_pred = None
        
        if prediction_input_method == "Manual Entry":
            st.markdown("Enter prediction data as JSON:")
            st.code('{"X": [[1.0, 2.0], [2.0, 3.0]]}', language="json")
            data_json = st.text_area("Prediction Data (JSON)", height=200)
            if data_json:
                try:
                    data = json.loads(data_json)
                    X_pred = data.get("X", [])
                except json.JSONDecodeError:
                    st.error("Invalid JSON format")
        
        elif prediction_input_method == "CSV Upload":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="pred_csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
                feature_cols = st.multiselect("Select feature columns", df.columns.tolist(), key="pred_features")
                if feature_cols:
                    X_pred = df[feature_cols].values.tolist()
        
        elif prediction_input_method == "JSON Upload":
            uploaded_file = st.file_uploader("Upload JSON file", type=["json"], key="pred_json")
            if uploaded_file:
                try:
                    data = json.load(uploaded_file)
                    X_pred = data.get("X", [])
                except json.JSONDecodeError:
                    st.error("Invalid JSON format")
        
        if X_pred:
            st.success(f"Data loaded: {len(X_pred)} samples")
            
            if st.button("Получить прогнозы", type="primary"):
                with st.spinner("Getting predictions..."):
                    predictions = predict(selected_model_id, X_pred)
                    if predictions:
                        st.success("Прогнозы сгенерированы!")
                        st.subheader("Results")
                        result_df = pd.DataFrame({
                            "Sample": range(1, len(predictions) + 1),
                            "Prediction": predictions
                        })
                        st.dataframe(result_df, use_container_width=True)
                        
                        st.subheader("Visualization")
                        st.line_chart(result_df.set_index("Sample")["Prediction"])


# Вкладка 4: Обученные модели
with tab4:
    st.header("Обученные модели")
    
    trained_models = get_trained_models()
    
    if not trained_models:
        st.info("No trained models available.")
    else:
        st.success(f"Found {len(trained_models)} trained model(s)")
        
        for model in trained_models:
            with st.expander(f"{model['model_id']} - {model['model_class_name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Model ID:**", model['model_id'])
                    st.write("**Model Class:**", model['model_class_name'])
                    st.write("**Created At:**", model['created_at'])
                    st.write("**Is Trained:**", "Да" if model['is_trained'] else "Нет")
                
                with col2:
                    st.write("**Hyperparameters:**")
                    st.json(model['hyperparameters'])
                
                if st.button(f"Удалить модель", key=f"delete_{model['model_id']}"):
                    if delete_model(model['model_id']):
                        st.success("Model deleted successfully!")
                        st.session_state.trained_models = get_trained_models()
                        st.rerun()
                    else:
                        st.error("Failed to delete model")

