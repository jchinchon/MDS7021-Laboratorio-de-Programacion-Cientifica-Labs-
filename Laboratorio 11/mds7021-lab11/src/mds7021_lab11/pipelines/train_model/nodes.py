"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

import logging
from typing import Dict

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from xgboost import XGBRegressor


def split_data(data: pd.DataFrame, params: Dict):
    shuffled_data = data.sample(frac=1, random_state=params["random_state"])
    rows = shuffled_data.shape[0]

    train_ratio = params["train_ratio"]
    valid_ratio = params["valid_ratio"]

    train_idx = int(rows * train_ratio)
    valid_idx = train_idx + int(rows * valid_ratio)

    assert rows > valid_idx, "test split should not be empty"

    target = params["target"]
    X = shuffled_data.drop(columns=target)
    y = shuffled_data[[target]]

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X[valid_idx:], y[valid_idx:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_mae")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
):
    nombre_experimento = "Experimento_01"
    experiment_id = mlflow.create_experiment(nombre_experimento)
    mlflow.autolog()
    # Entrenar Linear Regression model
    run_name = "Regresion_lineal_parametros_por_defecto"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_valid_pred = lr_model.predict(X_valid)
        lr_mae = mean_absolute_error(y_valid, lr_valid_pred)
        mlflow.log_metric("valid_mae", lr_mae)

    # Entrenar Random Forest model
    run_name = "Random_Forest_parametros_por_defecto"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)
        rf_valid_pred = rf_model.predict(X_valid)
        rf_mae = mean_absolute_error(y_valid, rf_valid_pred)
        mlflow.log_metric("valid_mae", rf_mae)

    # Entrenar SVR model
    run_name = "SVR_parametros_por_defecto"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        svr_model = SVR()
        svr_model.fit(X_train, y_train)
        svr_valid_pred = svr_model.predict(X_valid)
        svr_mae = mean_absolute_error(y_valid, svr_valid_pred)
        mlflow.log_metric("valid_mae", svr_mae)

    # Entrenar XGBoost model
    run_name = "XGBoost_parametros_por_defecto"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train, y_train)
        xgb_valid_pred = xgb_model.predict(X_valid)
        xgb_mae = mean_absolute_error(y_valid, xgb_valid_pred)
        mlflow.log_metric("valid_mae", xgb_mae)

    # Entrenar LightGBM model
    run_name = "LightGBM_parametros_por_defecto"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        lgbm_model = LGBMRegressor()
        lgbm_model.fit(X_train, y_train)
        lgbm_valid_pred = lgbm_model.predict(X_valid)
        lgbm_mae = mean_absolute_error(y_valid, lgbm_valid_pred)
        mlflow.log_metric("valid_mae", lgbm_mae)

    return get_best_model(experiment_id)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a Mean Absolute Error of {mae} on test data.")
