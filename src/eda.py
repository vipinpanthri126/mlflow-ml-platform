"""
EDA module: Descriptive statistics, outlier detection, distribution plots, correlation.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return comprehensive descriptive statistics."""
    numeric = df.select_dtypes(include=[np.number])
    stats = numeric.describe().T
    stats["missing"] = df.isnull().sum()
    stats["missing_%"] = (df.isnull().sum() / len(df) * 100).round(2)
    stats["skew"] = numeric.skew()
    stats["kurtosis"] = numeric.kurtosis()
    stats["unique"] = df.nunique()
    return stats


def get_categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return stats for categorical columns."""
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) == 0:
        return pd.DataFrame()
    records = []
    for col in cat_cols:
        records.append({
            "column": col,
            "unique": df[col].nunique(),
            "top_value": df[col].mode().iloc[0] if not df[col].mode().empty else "N/A",
            "top_freq": df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0,
            "missing": df[col].isnull().sum(),
            "missing_%": round(df[col].isnull().sum() / len(df) * 100, 2),
        })
    return pd.DataFrame(records)


def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> dict:
    """Detect outliers using IQR or Z-score method."""
    numeric = df.select_dtypes(include=[np.number])
    outlier_summary = {}

    if method == "iqr":
        for col in numeric.columns:
            Q1 = numeric[col].quantile(0.25)
            Q3 = numeric[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((numeric[col] < lower) | (numeric[col] > upper)).sum()
            outlier_summary[col] = {
                "n_outliers": int(n_outliers),
                "pct": round(n_outliers / len(df) * 100, 2),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
            }
    elif method == "zscore":
        for col in numeric.columns:
            z = np.abs((numeric[col] - numeric[col].mean()) / numeric[col].std())
            n_outliers = (z > 3).sum()
            outlier_summary[col] = {
                "n_outliers": int(n_outliers),
                "pct": round(n_outliers / len(df) * 100, 2),
            }

    return outlier_summary


def generate_distribution_plots(df: pd.DataFrame, max_cols: int = 12):
    """Generate histograms for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    figs = []
    for col in numeric_cols:
        fig = px.histogram(
            df, x=col, marginal="box", title=f"Distribution: {col}",
            template="plotly_dark", color_discrete_sequence=["#00d4ff"],
        )
        fig.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
        figs.append(fig)
    return figs


def generate_correlation_matrix(df: pd.DataFrame):
    """Generate a correlation heatmap."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Correlation Matrix", template="plotly_dark",
        height=500, margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def get_missing_value_report(df: pd.DataFrame):
    """Generate a bar chart of missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        return None
    fig = px.bar(
        x=missing.index, y=missing.values,
        labels={"x": "Column", "y": "Missing Count"},
        title="Missing Values by Column",
        template="plotly_dark", color_discrete_sequence=["#ff6b6b"],
    )
    fig.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
    return fig


def generate_target_analysis(df: pd.DataFrame, target: str):
    """Generate target variable distribution."""
    if df[target].dtype == "object" or df[target].nunique() <= 20:
        fig = px.pie(
            df, names=target, title=f"Target Distribution: {target}",
            template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        fig = px.histogram(
            df, x=target, marginal="box",
            title=f"Target Distribution: {target}",
            template="plotly_dark", color_discrete_sequence=["#00d4ff"],
        )
    fig.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
    return fig


def replace_special_with_nan(df: pd.DataFrame, columns: list, value: str):
    """Replace special values (e.g., '?', '-999') with np.nan."""
    for col in columns:
        if col in df.columns:
            # Handle numeric conversion if possible after replacement
            df[col] = df[col].replace(value, np.nan)
            # Try to convert to numeric if the column was object type solely due to the special char
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    return df


def impute_column(df: pd.DataFrame, column: str, method: str, value=None):
    """
    Impute missing values in a column.
    Methods: 'mean', 'median', 'mode', 'manual'
    """
    if column not in df.columns:
        return df

    if method == "mean":
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_val = df[column].mean()
            df[column] = df[column].fillna(fill_val)
    elif method == "median":
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_val = df[column].median()
            df[column] = df[column].fillna(fill_val)
    elif method == "mode":
        if not df[column].mode().empty:
            fill_val = df[column].mode()[0]
            df[column] = df[column].fillna(fill_val)
    elif method == "manual" and value is not None:
        # Try to convert value to column type
        if pd.api.types.is_numeric_dtype(df[column]):
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string or original type if conversion fails
        df[column] = df[column].fillna(value)
    
    return df
