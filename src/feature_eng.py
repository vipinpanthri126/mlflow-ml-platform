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


def generate_shap_plots(pipeline, X: pd.DataFrame):
    """Generate SHAP summary plot for a trained pipeline."""
    import shap

    try:
        # Extract the model from the pipeline
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Transform data
        X_transformed = preprocessor.transform(X)

        # Get feature names after transformation
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                if hasattr(trans.named_steps.get("onehot", None), "get_feature_names_out"):
                    feature_names.extend(trans.named_steps["onehot"].get_feature_names_out(cols).tolist())
                else:
                    feature_names.extend(cols)

        if isinstance(X_transformed, np.ndarray):
            X_df = pd.DataFrame(X_transformed, columns=feature_names[:X_transformed.shape[1]])
        else:
            X_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
                                columns=feature_names[:X_transformed.shape[1]])

        sample = X_df.sample(n=min(100, len(X_df)), random_state=42)

        # Use TreeExplainer for tree-based, KernelExplainer otherwise
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, sample.iloc[:20])

        shap_values = explainer.shap_values(sample)

        # Summary plot
        fig_summary, ax = plt.subplots(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], sample, show=False)
        else:
            shap.summary_plot(shap_values, sample, show=False)
        fig_summary = plt.gcf()
        plt.tight_layout()

        # Bar plot
        fig_bar, ax2 = plt.subplots(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], sample, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        fig_bar = plt.gcf()
        plt.tight_layout()

        return fig_summary, fig_bar, feature_names
    except Exception as e:
        return None, None, str(e)


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
