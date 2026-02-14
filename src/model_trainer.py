"""
Model training module: Model factory, pipelines, hyperparameter tuning, MLflow logging.
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error
)


def get_available_models(task_type: str) -> dict:
    """Return available models for the task type."""
    if task_type == "classification":
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from xgboost import XGBClassifier
        return {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss"),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
        }
    else:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from xgboost import XGBRegressor
        return {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }


def build_pipeline(model, numeric_cols: list, cat_cols: list) -> Pipeline:
    """Build an sklearn Pipeline with preprocessing."""
    transformers = []
    if numeric_cols:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_transformer, numeric_cols))
    if cat_cols:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def get_hyperparameter_grids(task_type: str) -> dict:
    """Return hyperparameter search grids per model."""
    if task_type == "classification":
        return {
            "Random Forest": {"model__n_estimators": [50, 100, 200], "model__max_depth": [5, 10, None]},
            "XGBoost": {"model__n_estimators": [50, 100], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.01, 0.1]},
            "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
            "Gradient Boosting": {"model__n_estimators": [50, 100], "model__max_depth": [3, 5]},
            "SVM": {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]},
        }
    else:
        return {
            "Random Forest": {"model__n_estimators": [50, 100, 200], "model__max_depth": [5, 10, None]},
            "XGBoost": {"model__n_estimators": [50, 100], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.01, 0.1]},
            "Linear Regression": {},
            "Ridge": {"model__alpha": [0.1, 1, 10]},
            "Gradient Boosting": {"model__n_estimators": [50, 100], "model__max_depth": [3, 5]},
        }


def calculate_metrics(y_true, y_pred, y_prob, task_type: str) -> dict:
    """Calculate performance metrics."""
    if task_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                else:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            except Exception:
                metrics["auc_roc"] = 0.0
    else:
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
    return metrics


def train_with_mlflow(
    pipeline: Pipeline, X_train, y_train, X_test, y_test,
    model_name: str, task_type: str, experiment_id: str,
    params: dict = None, do_grid_search: bool = False,
    iteration: int = 1, sme_inputs: str = "", feature_list: list = None,
) -> dict:
    """Train model with MLflow tracking."""
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_name}_iter{iteration}") as run:
        # Log metadata
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("sme_inputs", sme_inputs[:250] if sme_inputs else "none")

        if feature_list:
            mlflow.log_param("feature_list", json.dumps(feature_list[:50]))

        # Hyperparameter tuning
        if do_grid_search and params:
            grid = GridSearchCV(pipeline, params, cv=3, scoring="f1_weighted" if task_type == "classification" else "r2", n_jobs=-1)
            grid.fit(X_train, y_train)
            best_pipeline = grid.best_estimator_
            mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
            mlflow.log_metric("cv_best_score", grid.best_score_)
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)

        # Predict & score
        y_pred = best_pipeline.predict(X_test)
        y_prob = None
        if task_type == "classification" and hasattr(best_pipeline, "predict_proba"):
            try:
                y_prob = best_pipeline.predict_proba(X_test)
            except Exception:
                pass

        metrics = calculate_metrics(y_test, y_pred, y_prob, task_type)
        mlflow.log_metrics(metrics)

        # Cross-val score
        try:
            scoring = "f1_weighted" if task_type == "classification" else "r2"
            cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring=scoring)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
        except Exception:
            pass

        # Log model
        mlflow.sklearn.log_model(best_pipeline, "model")

        # Log timestamp
        mlflow.log_param("train_timestamp", datetime.now().isoformat())

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "metrics": metrics,
            "pipeline": best_pipeline,
        }


def compare_runs(experiment_id: str) -> pd.DataFrame:
    """Compare all runs in an experiment."""
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    if runs.empty:
        return pd.DataFrame()
    return runs.sort_values("start_time", ascending=False)
