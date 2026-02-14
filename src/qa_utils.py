"""
Q&A / Search Agent module: Query MLflow experiments, chat with LLM.
"""
import pandas as pd
import mlflow
import json


def build_experiment_context(experiment_id: str) -> str:
    """Build a text summary of all runs in an experiment for LLM consumption."""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if runs.empty:
            return "No experiments found. Please train some models first."

        lines = [f"## Experiment Summary ({len(runs)} runs)\n"]

        for idx, row in runs.iterrows():
            run_id = row.get("run_id", "N/A")
            model_name = row.get("params.model_name", "Unknown")
            iteration = row.get("params.iteration", "?")
            status = row.get("status", "N/A")
            start_time = row.get("start_time", "N/A")

            lines.append(f"### Run: {model_name} (Iteration {iteration})")
            lines.append(f"- **Run ID**: {run_id}")
            lines.append(f"- **Status**: {status}")
            lines.append(f"- **Start Time**: {start_time}")

            # Add metrics
            metric_cols = [c for c in row.index if c.startswith("metrics.")]
            if metric_cols:
                lines.append("- **Metrics**:")
                for mc in metric_cols:
                    val = row[mc]
                    if pd.notna(val):
                        lines.append(f"  - {mc.replace('metrics.', '')}: {val:.4f}")

            # Add key params
            param_cols = [c for c in row.index if c.startswith("params.")]
            if param_cols:
                lines.append("- **Parameters**:")
                for pc in param_cols:
                    val = row[pc]
                    if pd.notna(val) and val != "none":
                        lines.append(f"  - {pc.replace('params.', '')}: {val}")

            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error building context: {str(e)}"


def search_runs(query: str, experiment_id: str) -> pd.DataFrame:
    """Search MLflow runs with a filter query."""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if runs.empty:
            return pd.DataFrame()

        # Simple keyword filtering across all columns
        query_lower = query.lower()
        mask = runs.apply(lambda row: any(query_lower in str(v).lower() for v in row.values), axis=1)
        return runs[mask]
    except Exception:
        return pd.DataFrame()


def get_best_run(experiment_id: str, metric: str = "metrics.accuracy") -> dict:
    """Get the best run based on a specific metric."""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if runs.empty or metric not in runs.columns:
            return {}

        best_idx = runs[metric].idxmax()
        best_run = runs.loc[best_idx]
        return best_run.to_dict()
    except Exception:
        return {}


def get_run_comparison(experiment_id: str) -> pd.DataFrame:
    """Create a comparison table of all runs."""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if runs.empty:
            return pd.DataFrame()

        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        key_cols = ["run_id", "params.model_name", "params.iteration", "status", "start_time"] + metric_cols
        available_cols = [c for c in key_cols if c in runs.columns]
        comparison = runs[available_cols].copy()

        # Clean column names
        comparison.columns = [c.replace("params.", "").replace("metrics.", "") for c in comparison.columns]
        return comparison.sort_values("start_time", ascending=False)
    except Exception:
        return pd.DataFrame()


def format_chat_history(history: list) -> str:
    """Format chat history for display."""
    formatted = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            formatted.append(f"**🧑 You:** {content}")
        else:
            formatted.append(f"**🤖 Agent:** {content}")
    return "\n\n".join(formatted)
