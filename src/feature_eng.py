"""
Feature Engineering module: IV, SHAP, shadow model importance, feature selection.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def calculate_iv(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Calculate Information Value for each feature against target."""
    if df[target].nunique() != 2:
        return pd.DataFrame({
            "Feature": df.drop(columns=[target]).columns,
            "IV": ["N/A (non-binary target)"] * (len(df.columns) - 1),
            "Strength": ["N/A"] * (len(df.columns) - 1),
        })

    features = df.drop(columns=[target])
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    results = []

    for col in numeric_cols:
        try:
            data = df[[col, target]].dropna()
            bins = pd.qcut(data[col], q=10, duplicates="drop")
            grouped = data.groupby(bins)[target]
            total_events = data[target].sum()
            total_non_events = len(data) - total_events

            if total_events == 0 or total_non_events == 0:
                results.append({"Feature": col, "IV": 0, "Strength": "Useless"})
                continue

            iv = 0
            for _, group in grouped:
                events = group.sum()
                non_events = len(group) - events
                pct_events = max(events / total_events, 0.0001)
                pct_non_events = max(non_events / total_non_events, 0.0001)
                woe = np.log(pct_non_events / pct_events)
                iv += (pct_non_events - pct_events) * woe

            strength = (
                "Useless" if iv < 0.02 else
                "Weak" if iv < 0.1 else
                "Medium" if iv < 0.3 else
                "Strong" if iv < 0.5 else
                "Suspicious"
            )
            results.append({"Feature": col, "IV": round(iv, 4), "Strength": strength})
        except Exception:
            results.append({"Feature": col, "IV": 0, "Strength": "Error"})

    return pd.DataFrame(results).sort_values("IV", ascending=False, key=lambda x: pd.to_numeric(x, errors="coerce"))


def get_shadow_model_importance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Train a quick Random Forest to get feature importance."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer

    features = df.drop(columns=[target])
    y = df[target]
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return pd.DataFrame({"Feature": [], "Importance": []})

    X = features[numeric_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=numeric_cols)

    if y.dtype == "object" or y.nunique() <= 20:
        y_encoded = y.astype("category").cat.codes
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    else:
        y_encoded = y
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)

    model.fit(X_imputed, y_encoded)
    importance = pd.DataFrame({
        "Feature": numeric_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return importance


def calculate_shap_values(df: pd.DataFrame, target: str):
    """Calculate SHAP values using a shadow Random Forest."""
    import shap
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    import matplotlib.pyplot as plt

    features = df.drop(columns=[target])
    y = df[target]
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return None, None

    X = features[numeric_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=numeric_cols)

    if y.dtype == "object" or y.nunique() <= 20:
        y_encoded = y.astype("category").cat.codes
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    else:
        y_encoded = y
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)

    model.fit(X_imputed, y_encoded)

    sample = X_imputed.sample(n=min(100, len(X_imputed)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], sample, show=False)
    else:
        shap.summary_plot(shap_values, sample, show=False)
    fig = plt.gcf()
    plt.tight_layout()

    # Mean absolute SHAP for ranking
    try:
        if isinstance(shap_values, list):
            # For classification, taking the class 1 (positive) or 0 if binary
            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            vals = shap_values

        # Debugging: Print shapes
        print(f"DEBUG: shap_values type: {type(shap_values)}")
        if hasattr(vals, "shape"):
            print(f"DEBUG: vals shape: {vals.shape}")
        
        # Calculate mean absolute SHAP
        # vals shape might be (samples, features) or (samples, features, classes)
        
        # 1. Take absolute values
        abs_vals = np.abs(vals)
        
        # 2. Average over samples (axis 0)
        # Result shape: (features,) for regression, (features, classes) for classification
        feature_importance = abs_vals.mean(axis=0)
        
        if feature_importance.ndim > 1:
            feature_importance = feature_importance.sum(axis=1)
            
        mean_shap = feature_importance

    except Exception as e:
        print(f"ERROR calculating mean_shap: {e}")
        # Fallback to zeros
        mean_shap = np.zeros(len(numeric_cols))

    shap_importance = pd.DataFrame({
        "Feature": numeric_cols,
        "Mean_SHAP": mean_shap,
    }).sort_values("Mean_SHAP", ascending=False)

    return fig, shap_importance


def apply_feature_engineering(df: pd.DataFrame, sme_inputs: str, target: str) -> pd.DataFrame:
    """Apply basic automated feature engineering."""
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    # Auto-create interaction terms for top correlated pairs
    if len(numeric_cols) >= 2:
        corr = result[numeric_cols].corr().abs()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                pairs.append((numeric_cols[i], numeric_cols[j], corr.iloc[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        for a, b, c in pairs[:3]:  # Top 3 correlated pairs
            result[f"{a}_x_{b}"] = result[a] * result[b]

    # Log transforms for skewed features
    for col in numeric_cols:
        if result[col].min() > 0 and abs(result[col].skew()) > 1:
            result[f"{col}_log"] = np.log1p(result[col])

    return result


def select_features_by_importance(importance_df: pd.DataFrame, threshold: float = 0.01) -> list:
    """Select features above an importance threshold."""
    return importance_df[importance_df["Importance"] >= threshold]["Feature"].tolist()
