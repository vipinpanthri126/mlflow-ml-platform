"""
Shared utility functions for data loading, type detection, and reporting.
"""
import pandas as pd
import numpy as np
import mlflow
import io
import json
from datetime import datetime


def load_data(uploaded_file):
    """Load CSV or Excel file into a DataFrame."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {name}")


def detect_task_type(y: pd.Series) -> str:
    """Detect whether the target is classification or regression."""
    if y.dtype == "object" or y.nunique() <= 20:
        return "classification"
    return "regression"


def split_columns(df: pd.DataFrame, target: str):
    """Split DataFrame columns into numeric and categorical lists."""
    features = df.drop(columns=[target], errors="ignore")
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, cat_cols


def format_metrics(metrics: dict) -> dict:
    """Round all metric values for display."""
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}


def save_artifact_to_mlflow(content: str, filename: str, run_id: str):
    """Save text content as an MLflow artifact."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, filename)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(fpath)


def generate_markdown_report(sections: dict) -> str:
    """Generate a combined Markdown report from section dict {title: content}."""
    lines = [f"# Model Development Report\n", f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"]
    for title, content in sections.items():
        lines.append(f"\n## {title}\n")
        lines.append(content)
    return "\n".join(lines)


def get_mlflow_experiment(name: str = "MLflow_Agent_Experiment"):
    """Get or create an MLflow experiment."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        exp_id = mlflow.create_experiment(name)
    else:
        exp_id = experiment.experiment_id
    return exp_id


def dataframe_summary(df: pd.DataFrame) -> str:
    """Create a text summary of a DataFrame for LLM consumption."""
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    buf.write("Columns & Types:\n")
    for col in df.columns:
        buf.write(f"  - {col}: {df[col].dtype}, {df[col].nunique()} unique, {df[col].isnull().sum()} nulls\n")
    buf.write(f"\nNumeric Summary:\n{df.describe().to_string()}\n")
    return buf.getvalue()
