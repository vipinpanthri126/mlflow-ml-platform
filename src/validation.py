"""
Validation module: OOT validation, feedback loop, performance reporting.
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error, confusion_matrix,
    classification_report
)
import plotly.express as px
import plotly.graph_objects as go
import json


def validate_oot(model_uri: str, oot_df: pd.DataFrame, target: str, task_type: str) -> dict:
    """Load a trained model and validate on OOT dataset."""
    model = mlflow.sklearn.load_model(model_uri)
    X_oot = oot_df.drop(columns=[target])
    y_oot = oot_df[target]

    if y_oot.dtype == "object":
        y_oot = y_oot.astype("category").cat.codes

    y_pred = model.predict(X_oot)
    y_prob = None
    if task_type == "classification" and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_oot)
        except Exception:
            pass

    if task_type == "classification":
        metrics = {
            "oot_accuracy": accuracy_score(y_oot, y_pred),
            "oot_f1_score": f1_score(y_oot, y_pred, average="weighted", zero_division=0),
            "oot_precision": precision_score(y_oot, y_pred, average="weighted", zero_division=0),
            "oot_recall": recall_score(y_oot, y_pred, average="weighted", zero_division=0),
        }
        if y_prob is not None:
            try:
                if len(np.unique(y_oot)) == 2:
                    metrics["oot_auc_roc"] = roc_auc_score(y_oot, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                else:
                    metrics["oot_auc_roc"] = roc_auc_score(y_oot, y_prob, multi_class="ovr", average="weighted")
            except Exception:
                pass
    else:
        metrics = {
            "oot_rmse": np.sqrt(mean_squared_error(y_oot, y_pred)),
            "oot_mae": mean_absolute_error(y_oot, y_pred),
            "oot_r2": r2_score(y_oot, y_pred),
        }

    return {
        "metrics": metrics,
        "y_true": y_oot,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def generate_validation_report(metrics: dict, task_type: str) -> str:
    """Generate a text report of validation metrics."""
    lines = ["## OOT Validation Results\n"]
    for k, v in metrics.items():
        lines.append(f"- **{k.replace('oot_', '').replace('_', ' ').title()}**: {v:.4f}")
    return "\n".join(lines)


def generate_confusion_matrix_plot(y_true, y_pred):
    """Generate a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
        colorscale="Blues", text=cm, texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Confusion Matrix (OOT)", xaxis_title="Predicted",
        yaxis_title="Actual", template="plotly_dark",
        height=400, margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def record_feedback(run_id: str, accepted: bool, feedback_text: str, oot_metrics: dict):
    """Log validation feedback to MLflow."""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("oot_accepted", accepted)
        mlflow.log_param("oot_feedback", feedback_text[:250] if feedback_text else "none")
        mlflow.log_metrics({f"oot_{k}" if not k.startswith("oot") else k: v for k, v in oot_metrics.items()})


def generate_performance_comparison_plot(train_metrics: dict, oot_metrics: dict, task_type: str):
    """Bar chart comparing train vs OOT metrics."""
    # Normalize key names
    clean_oot = {}
    for k, v in oot_metrics.items():
        clean_key = k.replace("oot_", "")
        clean_oot[clean_key] = v

    common_keys = [k for k in train_metrics if k in clean_oot]
    if not common_keys:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Train/Test", x=common_keys, y=[train_metrics[k] for k in common_keys],
                         marker_color="#00d4ff"))
    fig.add_trace(go.Bar(name="OOT", x=common_keys, y=[clean_oot[k] for k in common_keys],
                         marker_color="#ff6b6b"))
    fig.update_layout(
        barmode="group", title="Train vs OOT Performance",
        template="plotly_dark", height=400,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig
