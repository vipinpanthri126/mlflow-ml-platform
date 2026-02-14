"""
Model monitoring module: PSI, drift detection, monitoring plan templates.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Calculate Population Stability Index."""
    try:
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            bins + 1,
        )
        expected_counts = np.histogram(expected.dropna(), breakpoints)[0]
        actual_counts = np.histogram(actual.dropna(), breakpoints)[0]

        expected_pct = expected_counts / expected_counts.sum()
        actual_pct = actual_counts / actual_counts.sum()

        # Avoid division by zero
        expected_pct = np.clip(expected_pct, 0.0001, None)
        actual_pct = np.clip(actual_pct, 0.0001, None)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return round(psi, 4)
    except Exception:
        return 0.0


def generate_psi_report(train_df: pd.DataFrame, oot_df: pd.DataFrame, target: str = None) -> pd.DataFrame:
    """Calculate PSI for all numeric features."""
    train_numeric = train_df.select_dtypes(include=[np.number])
    oot_numeric = oot_df.select_dtypes(include=[np.number])
    common_cols = [c for c in train_numeric.columns if c in oot_numeric.columns]

    if target and target in common_cols:
        common_cols.remove(target)

    records = []
    for col in common_cols:
        psi = calculate_psi(train_numeric[col], oot_numeric[col])
        status = (
            "✅ Stable" if psi < 0.1 else
            "⚠️ Moderate Shift" if psi < 0.25 else
            "🔴 Significant Shift"
        )
        records.append({"Feature": col, "PSI": psi, "Status": status})

    return pd.DataFrame(records).sort_values("PSI", ascending=False)


def detect_drift(train_stats: pd.DataFrame, new_stats: pd.DataFrame) -> pd.DataFrame:
    """Compare descriptive statistics to detect distribution drift."""
    common = train_stats.index.intersection(new_stats.index)
    drift_records = []

    for col in common:
        try:
            mean_shift = abs(train_stats.loc[col, "mean"] - new_stats.loc[col, "mean"]) / (train_stats.loc[col, "std"] + 1e-10)
            std_change = abs(train_stats.loc[col, "std"] - new_stats.loc[col, "std"]) / (train_stats.loc[col, "std"] + 1e-10)
            drift_records.append({
                "Feature": col,
                "Train_Mean": round(train_stats.loc[col, "mean"], 4),
                "New_Mean": round(new_stats.loc[col, "mean"], 4),
                "Mean_Shift_Sigma": round(mean_shift, 4),
                "Std_Change_%": round(std_change * 100, 2),
                "Alert": "🔴" if mean_shift > 2 or std_change > 0.5 else "⚠️" if mean_shift > 1 else "✅",
            })
        except Exception:
            continue

    return pd.DataFrame(drift_records).sort_values("Mean_Shift_Sigma", ascending=False)


def generate_monitoring_kpis() -> dict:
    """Return monitoring KPI templates."""
    return {
        "Performance KPIs": {
            "Model Accuracy / R²": {"frequency": "Monthly", "threshold": "-5% from baseline", "owner": "Data Science"},
            "F1 Score / RMSE": {"frequency": "Monthly", "threshold": "-10% from baseline", "owner": "Data Science"},
            "AUC-ROC": {"frequency": "Monthly", "threshold": "< 0.70", "owner": "Data Science"},
        },
        "Data Quality KPIs": {
            "Missing Value Rate": {"frequency": "Weekly", "threshold": "> 5% increase", "owner": "Data Engineering"},
            "Feature Distribution PSI": {"frequency": "Monthly", "threshold": "> 0.20", "owner": "Data Science"},
            "Data Volume": {"frequency": "Daily", "threshold": "< 50% of expected", "owner": "Data Engineering"},
        },
        "Operational KPIs": {
            "Prediction Latency": {"frequency": "Real-time", "threshold": "> 500ms", "owner": "ML Engineering"},
            "Error Rate": {"frequency": "Daily", "threshold": "> 1%", "owner": "ML Engineering"},
            "Model Uptime": {"frequency": "Daily", "threshold": "< 99.5%", "owner": "ML Engineering"},
        },
    }


def generate_drift_plot(psi_df: pd.DataFrame):
    """Generate a bar chart of PSI values."""
    if psi_df.empty:
        return None
    colors = ["#00d4ff" if row["PSI"] < 0.1 else "#ffd700" if row["PSI"] < 0.25 else "#ff6b6b" for _, row in psi_df.iterrows()]
    fig = go.Figure(go.Bar(
        x=psi_df["Feature"], y=psi_df["PSI"],
        marker_color=colors,
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="yellow", annotation_text="Moderate Threshold")
    fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="High Threshold")
    fig.update_layout(
        title="Feature Stability (PSI)", template="plotly_dark",
        yaxis_title="PSI Value", height=400,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig
