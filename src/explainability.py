"""
Explainability module: SHAP plots, feature importance, PDP, model KPIs.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


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


def get_feature_importance_table(pipeline) -> pd.DataFrame:
    """Extract feature importance from a trained pipeline."""
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                if hasattr(trans.named_steps.get("onehot", None), "get_feature_names_out"):
                    feature_names.extend(trans.named_steps["onehot"].get_feature_names_out(cols).tolist())
                else:
                    feature_names.extend(cols)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            return pd.DataFrame({"Feature": ["N/A"], "Importance": [0]})

        n = min(len(feature_names), len(importances))
        df = pd.DataFrame({
            "Feature": feature_names[:n],
            "Importance": importances[:n],
        }).sort_values("Importance", ascending=False)

        return df
    except Exception as e:
        return pd.DataFrame({"Feature": [str(e)], "Importance": [0]})


def generate_importance_plot(importance_df: pd.DataFrame, top_n: int = 20):
    """Generate a Plotly bar chart of feature importance."""
    top = importance_df.head(top_n)
    fig = px.bar(
        top, x="Importance", y="Feature", orientation="h",
        title=f"Top {top_n} Feature Importances",
        template="plotly_dark", color="Importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=max(300, top_n * 25), yaxis=dict(autorange="reversed"),
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def generate_partial_dependence(pipeline, X: pd.DataFrame, features: list, target_name: str = "target"):
    """Generate Partial Dependence Plots."""
    from sklearn.inspection import PartialDependenceDisplay

    figs = []
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X)

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
            X_df = X_transformed

        for feat in features:
            if feat in feature_names:
                idx = feature_names.index(feat)
                fig, ax = plt.subplots(figsize=(8, 4))
                PartialDependenceDisplay.from_estimator(model, X_df, [idx], ax=ax, feature_names=feature_names)
                ax.set_title(f"Partial Dependence: {feat}")
                plt.tight_layout()
                figs.append(fig)
    except Exception:
        pass

    return figs


def create_explainability_summary(pipeline, X: pd.DataFrame, task_type: str) -> dict:
    """Create a comprehensive explainability summary."""
    importance_df = get_feature_importance_table(pipeline)
    summary = {
        "top_features": importance_df.head(10).to_dict("records"),
        "total_features_used": len(importance_df),
        "task_type": task_type,
    }
    return summary
