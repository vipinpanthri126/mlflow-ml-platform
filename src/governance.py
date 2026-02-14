"""
Governance module: Model card, audit trail, fairness, compliance, artifact export.
"""
import pandas as pd
import numpy as np
import mlflow
import json
import os
import io
import zipfile
from datetime import datetime


def generate_model_card(run_info: dict) -> str:
    """Generate a structured Model Card in markdown."""
    card = f"""# Model Card
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Model Overview
| Field | Value |
|-------|-------|
| **Model Name** | {run_info.get('model_name', 'N/A')} |
| **Run ID** | {run_info.get('run_id', 'N/A')} |
| **Task Type** | {run_info.get('task_type', 'N/A')} |
| **Training Date** | {run_info.get('train_timestamp', 'N/A')} |
| **Iteration** | {run_info.get('iteration', 'N/A')} |
| **Training Samples** | {run_info.get('n_train_samples', 'N/A')} |
| **Features Used** | {run_info.get('n_features', 'N/A')} |

## Performance Metrics
"""
    metrics = run_info.get("metrics", {})
    for k, v in metrics.items():
        card += f"- **{k.replace('_', ' ').title()}**: {v:.4f}\n"

    card += f"""
## Business Context
{run_info.get('problem_statement', 'Not provided')}

## SME Inputs
{run_info.get('sme_inputs', 'Not provided')}

## Model Limitations
- Model performance validated on available data only
- Feature drift may impact predictions over time
- Regular monitoring is recommended

## Ethical Considerations
- Review for potential bias in training data
- Ensure fairness across protected categories
- Consider impact on affected populations

## Change Log
| Date | Change | Author |
|------|--------|--------|
| {datetime.now().strftime('%Y-%m-%d')} | Initial model development | Auto-generated |
"""
    return card


def create_audit_trail(experiment_id: str) -> pd.DataFrame:
    """Create an audit trail of all runs in an experiment."""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if runs.empty:
            return pd.DataFrame()

        cols_to_keep = [c for c in runs.columns if c.startswith(("params.", "metrics.", "start_time", "run_id", "status"))]
        audit = runs[cols_to_keep].copy()
        audit = audit.sort_values("start_time", ascending=False)
        return audit
    except Exception:
        return pd.DataFrame()


def generate_governance_checklist() -> dict:
    """Return a model governance compliance checklist."""
    return {
        "Data Governance": [
            {"item": "Data source documentation", "status": "⬜", "notes": ""},
            {"item": "Data quality assessment completed", "status": "⬜", "notes": ""},
            {"item": "PII/sensitive data handling reviewed", "status": "⬜", "notes": ""},
            {"item": "Data lineage documented", "status": "⬜", "notes": ""},
            {"item": "Training/test data split rationale", "status": "⬜", "notes": ""},
        ],
        "Model Development": [
            {"item": "Business problem clearly defined", "status": "⬜", "notes": ""},
            {"item": "Feature engineering documented", "status": "⬜", "notes": ""},
            {"item": "Model selection rationale documented", "status": "⬜", "notes": ""},
            {"item": "Hyperparameter tuning approach documented", "status": "⬜", "notes": ""},
            {"item": "Cross-validation results reviewed", "status": "⬜", "notes": ""},
        ],
        "Validation & Testing": [
            {"item": "Out-of-time validation performed", "status": "⬜", "notes": ""},
            {"item": "Performance meets acceptance criteria", "status": "⬜", "notes": ""},
            {"item": "Sensitivity analysis completed", "status": "⬜", "notes": ""},
            {"item": "Edge case testing performed", "status": "⬜", "notes": ""},
            {"item": "Stability analysis (PSI) reviewed", "status": "⬜", "notes": ""},
        ],
        "Deployment & Monitoring": [
            {"item": "Monitoring plan established", "status": "⬜", "notes": ""},
            {"item": "Alert thresholds defined", "status": "⬜", "notes": ""},
            {"item": "Retraining triggers documented", "status": "⬜", "notes": ""},
            {"item": "Fallback/rollback plan in place", "status": "⬜", "notes": ""},
            {"item": "Model documentation complete", "status": "⬜", "notes": ""},
        ],
        "Ethics & Compliance": [
            {"item": "Fairness assessment completed", "status": "⬜", "notes": ""},
            {"item": "Bias testing performed", "status": "⬜", "notes": ""},
            {"item": "Regulatory requirements reviewed", "status": "⬜", "notes": ""},
            {"item": "Stakeholder sign-off obtained", "status": "⬜", "notes": ""},
            {"item": "Explainability requirements met", "status": "⬜", "notes": ""},
        ],
    }


def assess_fairness(pipeline, X: pd.DataFrame, y_true, protected_col: str) -> pd.DataFrame:
    """Basic fairness assessment across a protected attribute."""
    try:
        y_pred = pipeline.predict(X)
        groups = X[protected_col].unique()
        records = []

        for g in groups:
            mask = X[protected_col] == g
            if mask.sum() == 0:
                continue
            from sklearn.metrics import accuracy_score, f1_score
            records.append({
                "Group": g,
                "Count": int(mask.sum()),
                "Accuracy": round(accuracy_score(y_true[mask], y_pred[mask]), 4),
                "F1": round(f1_score(y_true[mask], y_pred[mask], average="weighted", zero_division=0), 4),
                "Positive_Rate": round(y_pred[mask].mean(), 4),
            })

        return pd.DataFrame(records)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def export_all_artifacts(experiment_id: str) -> bytes:
    """Export all experiment artifacts as a ZIP file."""
    buf = io.BytesIO()
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add run summary
            summary = runs.to_csv(index=False)
            zf.writestr("experiment_summary.csv", summary)

            # Add run artifacts
            for _, row in runs.iterrows():
                run_id = row["run_id"]
                try:
                    client = mlflow.tracking.MlflowClient()
                    artifacts = client.list_artifacts(run_id)
                    for art in artifacts:
                        local_path = client.download_artifacts(run_id, art.path)
                        if os.path.isfile(local_path):
                            zf.write(local_path, f"runs/{run_id}/{art.path}")
                        elif os.path.isdir(local_path):
                            for root, dirs, files in os.walk(local_path):
                                for f in files:
                                    full = os.path.join(root, f)
                                    arcname = f"runs/{run_id}/{art.path}/{os.path.relpath(full, local_path)}"
                                    zf.write(full, arcname)
                except Exception:
                    continue

            # Add governance checklist
            checklist = json.dumps(generate_governance_checklist(), indent=2)
            zf.writestr("governance_checklist.json", checklist)

    except Exception:
        pass

    buf.seek(0)
    return buf.getvalue()
