"""
MLflow Agentic ML Lifecycle Solution
Main Streamlit Application — 10 integrated pages from data upload through governance.
"""
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import json
import os
from datetime import datetime

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="MLflow Agentic ML Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp { font-family: 'Inter', sans-serif; }

    /* Hero gradient header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-header h1 {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .hero-header p { color: #a0a0b8; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h3 { color: #00d4ff; font-size: 1.8rem; margin: 0; }
    .metric-card p { color: #a0a0b8; font-size: 0.85rem; margin: 0.3rem 0 0 0; }

    /* Status badges */
    .badge-ok { background: #00c853; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; }
    .badge-warn { background: #ff9100; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; }
    .badge-err { background: #ff1744; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; }

    /* Section dividers */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #7b2ff7, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    /* ── Sidebar radio labels: large, bold, high-contrast ── */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: linear-gradient(135deg, rgba(30,30,60,0.9) 0%, rgba(20,20,50,0.95) 100%);
        border: 1px solid rgba(123,47,247,0.25);
        border-left: 4px solid #7b2ff7;
        border-radius: 10px;
        padding: 0.7rem 1rem !important;
        margin-bottom: 6px;
        transition: all 0.25s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: linear-gradient(135deg, rgba(50,40,90,0.95) 0%, rgba(30,30,70,1) 100%);
        border-left-color: #00d4ff;
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(0,212,255,0.15) 0%, rgba(123,47,247,0.18) 100%) !important;
        border: 1px solid rgba(0,212,255,0.5) !important;
        border-left: 4px solid #00d4ff !important;
        box-shadow: 0 0 14px rgba(0,212,255,0.15);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p,
    [data-testid="stSidebar"] [role="radiogroup"] label div {
        color: #e0e0f0 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }

    /* Next-step button styling */
    .next-step-container {
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 2px solid rgba(123,47,247,0.25);
        text-align: right;
    }

    /* Remove extra padding */
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ─────────────────────────────────────
defaults = {
    "train_data": None,
    "oot_data": None,
    "problem_statement": "",
    "sme_inputs": "",
    "target_variable": None,
    "eda_summary": "",
    "eda_stats_text": "",
    "fe_suggestions": "",
    "selected_features": [],
    "engineered_df": None,
    "trained_models": {},
    "best_model": None,
    "best_run_id": None,
    "experiment_id": None,
    "iteration": 1,
    "oot_accepted": False,
    "oot_metrics": {},
    "model_doc": "",
    "monitoring_plan": "",
    "chat_history": [],
    "api_key": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── MLflow Setup ────────────────────────────────────────────
mlflow.set_tracking_uri("mlruns")
if st.session_state.experiment_id is None:
    from src.utils import get_mlflow_experiment
    st.session_state.experiment_id = get_mlflow_experiment()

# ─── Sidebar Navigation ─────────────────────────────────────
pages = [
    "📤 Data Upload",
    "📊 Vibe Analysis (EDA)",
    "🔧 Feature Engineering",
    "🏗️ Model Building",
    "✅ OOT Validation",
    "📈 Monitoring Plan",
    "🔍 Explainability",
    "📝 Documentation",
    "🛡️ Governance",
    "🔎 Search Agent",
]

# Initialize the nav key ONCE so the radio widget has a default
if "_current_page" not in st.session_state:
    st.session_state["_current_page"] = pages[0]

# Apply pending navigation BEFORE the radio widget renders
if "_next_page" in st.session_state:
    st.session_state["_current_page"] = st.session_state.pop("_next_page")

def next_step_button(current_page: str):
    """Render a 'Next Step' button that advances to the next page."""
    idx = pages.index(current_page)
    if idx < len(pages) - 1:
        st.markdown('<div class="next-step-container">', unsafe_allow_html=True)
        if st.button(f"Next Step ➡️  {pages[idx + 1]}", type="primary", key=f"next_{idx}", use_container_width=True):
            st.session_state["_next_page"] = pages[idx + 1]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧠 ML Platform")
    st.markdown("---")

    st.session_state.api_key = st.text_input("🔑 Google Gemini API Key", type="password", value=st.session_state.api_key)

    st.markdown("### 📌 Workflow Steps")
    page = st.radio("Navigate", pages, key="_current_page", label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"**Iteration:** `{st.session_state.iteration}`")
    if st.session_state.train_data is not None:
        st.markdown(f"**Data:** {st.session_state.train_data.shape[0]} rows × {st.session_state.train_data.shape[1]} cols")
    if st.session_state.best_model:
        st.markdown(f"**Best Model:** `{st.session_state.best_model}`")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: DATA UPLOAD
# ═══════════════════════════════════════════════════════════════
if page == "📤 Data Upload":
    st.markdown("""<div class="hero-header"><h1>📤 Data Upload & Problem Definition</h1>
    <p>Upload your training data and define the business problem</p></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Training Data")
        train_file = st.file_uploader("Upload Training CSV/Excel", type=["csv", "xlsx", "xls"], key="train_upload")
        if train_file:
            from src.utils import load_data
            st.session_state.train_data = load_data(train_file)
            st.success(f"✅ Loaded {st.session_state.train_data.shape[0]} rows × {st.session_state.train_data.shape[1]} columns")

        st.subheader("📂 OOT / Validation Data (Optional)")
        oot_file = st.file_uploader("Upload OOT CSV/Excel", type=["csv", "xlsx", "xls"], key="oot_upload")
        if oot_file:
            from src.utils import load_data
            st.session_state.oot_data = load_data(oot_file)
            st.success(f"✅ OOT Data: {st.session_state.oot_data.shape[0]} rows")

    with col2:
        st.subheader("📋 Problem Statement")
        st.session_state.problem_statement = st.text_area(
            "Describe the business problem",
            value=st.session_state.problem_statement,
            height=120,
            placeholder="e.g., Predict customer churn based on usage and demographic data...",
        )
        st.subheader("💡 SME / Domain Inputs")
        st.session_state.sme_inputs = st.text_area(
            "Domain knowledge and constraints",
            value=st.session_state.sme_inputs,
            height=120,
            placeholder="e.g., Tenure and contract type are key drivers. Exclude income from model...",
        )

    if st.session_state.train_data is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Target variable selection
        st.subheader("🎯 Target Variable")
        cols = st.session_state.train_data.columns.tolist()
        default_idx = len(cols) - 1
        st.session_state.target_variable = st.selectbox("Select target column", cols, index=default_idx)

        # Data preview
        st.subheader("👀 Data Preview")
        st.dataframe(st.session_state.train_data.head(20), use_container_width=True)

        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        df = st.session_state.train_data
        c1.markdown(f'<div class="metric-card"><h3>{df.shape[0]:,}</h3><p>Rows</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Columns</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><h3>{df.isnull().sum().sum():,}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><h3>{df.select_dtypes(include=[np.number]).shape[1]}</h3><p>Numeric Features</p></div>', unsafe_allow_html=True)

    next_step_button(page)


# ═══════════════════════════════════════════════════════════════
# PAGE 2: VIBE ANALYSIS / EDA
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Vibe Analysis (EDA)":
    st.markdown("""<div class="hero-header"><h1>📊 Vibe Analysis — Exploratory Data Analysis</h1>
    <p>Automated profiling, outlier detection, and AI-powered insights</p></div>""", unsafe_allow_html=True)

    if st.session_state.train_data is None:
        st.warning("⚠️ Please upload training data first.")
    else:
        df = st.session_state.train_data
        target = st.session_state.target_variable

        from src.eda import (
            get_descriptive_stats, get_categorical_stats, detect_outliers,
            generate_distribution_plots, generate_correlation_matrix,
            get_missing_value_report, generate_target_analysis,
            replace_special_with_nan, impute_column,
        )
        from src.utils import dataframe_summary

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Statistics", "📉 Distributions", "🔗 Correlations", "⚡ Outliers", "🤖 AI Summary", "🛠️ Handling Missing Values"
        ])

        with tab1:
            st.subheader("Numeric Feature Statistics")
            stats = get_descriptive_stats(df)
            st.dataframe(stats, use_container_width=True)

            cat_stats = get_categorical_stats(df)
            if not cat_stats.empty:
                st.subheader("Categorical Feature Statistics")
                st.dataframe(cat_stats, use_container_width=True)

            if target:
                st.subheader(f"Target Distribution: `{target}`")
                st.plotly_chart(generate_target_analysis(df, target), use_container_width=True)

        with tab2:
            st.subheader("Feature Distributions")
            figs = generate_distribution_plots(df)
            cols = st.columns(2)
            for i, fig in enumerate(figs):
                cols[i % 2].plotly_chart(fig, use_container_width=True)

            missing_fig = get_missing_value_report(df)
            if missing_fig:
                st.subheader("Missing Values")
                st.plotly_chart(missing_fig, use_container_width=True)

        with tab3:
            st.subheader("Feature Correlations")
            st.plotly_chart(generate_correlation_matrix(df), use_container_width=True)

        with tab4:
            st.subheader("Outlier Detection")
            method = st.selectbox("Detection Method", ["iqr", "zscore"])
            outliers = detect_outliers(df, method)
            out_df = pd.DataFrame(outliers).T
            if not out_df.empty:
                st.dataframe(out_df, use_container_width=True)

                # Highlight columns with high outlier counts
                high_outlier = {k: v for k, v in outliers.items() if v["n_outliers"] > 0}
                if high_outlier:
                    st.info(f"⚡ {len(high_outlier)} features have outliers")

        with tab5:
            st.subheader("🤖 AI-Powered EDA Summary")
            stats_text = dataframe_summary(df)
            st.session_state.eda_stats_text = stats_text

            if st.button("🚀 Generate AI Summary", type="primary"):
                if not st.session_state.api_key:
                    st.warning("⚠️ Please enter your Gemini API key in the sidebar.")
                else:
                    with st.spinner("Analyzing data with AI..."):
                        from src.gemini_utils import summarize_eda
                        summary = summarize_eda(
                            st.session_state.api_key, stats_text,
                            st.session_state.problem_statement,
                            st.session_state.sme_inputs,
                        )
                        st.session_state.eda_summary = summary

            if st.session_state.eda_summary:
                st.markdown(st.session_state.eda_summary)

        with tab6:
            st.subheader("🛠️ Handling Missing & Special Values")
            
            # Section 1: Replace Special Values
            st.markdown("### 1️⃣ Replace Special Values with NaN")
            st.caption("Replace placeholders like `?`, `-999`, or `N/A` with standard missing values.")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                special_val = st.text_input("Value to replace", placeholder="e.g., ?")
            with c2:
                target_cols = st.multiselect("Select columns", df.columns.tolist())
                
            if st.button("Replace with NaN", disabled=not (special_val and target_cols)):
                st.session_state.train_data = replace_special_with_nan(df, target_cols, special_val)
                st.success(f"✅ Replaced '{special_val}' with NaN in {len(target_cols)} columns.")
                st.rerun()

            st.markdown("---")

            # Section 2: Impute Missing Values
            st.markdown("### 2️⃣ Impute Missing Values")
            cols_with_missing = df.columns[df.isnull().any()].tolist()
            
            if not cols_with_missing:
                st.success("🎉 No missing values found in the dataset!")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    imp_col = st.selectbox("Select Column", cols_with_missing)
                with c2:
                    imp_method = st.selectbox("Imputation Method", ["mean", "median", "mode", "manual"])
                with c3:
                    manual_val = None
                    if imp_method == "manual":
                        manual_val = st.text_input("Enter Value", placeholder="Value to fill")
                
                if st.button(f"Impute `{imp_col}`"):
                    if imp_method == "manual" and manual_val is None:
                        st.error("⚠️ Please enter a value for manual imputation.")
                    else:
                        st.session_state.train_data = impute_column(df, imp_col, imp_method, manual_val)
                        st.success(f"✅ Imputed `{imp_col}` using {imp_method}.")
                        st.rerun()

    next_step_button(page)


# ═══════════════════════════════════════════════════════════════
# PAGE 3: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
elif page == "🔧 Feature Engineering":
    st.markdown("""<div class="hero-header"><h1>🔧 Feature Engineering & Selection</h1>
    <p>IV analysis, SHAP importance, automated feature creation, and SME-guided selection</p></div>""", unsafe_allow_html=True)

    if st.session_state.train_data is None:
        st.warning("⚠️ Please upload training data first.")
    else:
        df = st.session_state.train_data
        target = st.session_state.target_variable

        from src.feature_eng import (
            calculate_iv, get_shadow_model_importance, calculate_shap_values,
            apply_feature_engineering, select_features_by_importance,
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 IV Analysis", "🌲 Feature Importance", "🔬 SHAP Analysis", "⚙️ Engineer & Select"
        ])

        with tab1:
            st.subheader("Information Value (IV)")
            iv_df = calculate_iv(df, target)
            st.dataframe(iv_df, use_container_width=True)

        with tab2:
            st.subheader("Shadow Model Feature Importance")
            if st.button("🏃 Run Shadow Model", key="shadow_btn"):
                with st.spinner("Training shadow Random Forest..."):
                    importance = get_shadow_model_importance(df, target)
                    st.session_state["shadow_importance"] = importance

            if "shadow_importance" in st.session_state:
                imp = st.session_state["shadow_importance"]
                st.dataframe(imp, use_container_width=True)
                import plotly.express as px
                fig = px.bar(imp.head(15), x="Importance", y="Feature", orientation="h",
                             template="plotly_dark", color="Importance", color_continuous_scale="Viridis")
                fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("SHAP Value Analysis")
            if st.button("🔬 Calculate SHAP Values", key="shap_btn"):
                with st.spinner("Computing SHAP values..."):
                    shap_fig, shap_imp = calculate_shap_values(df, target)
                    st.session_state["shap_fig"] = shap_fig
                    st.session_state["shap_imp"] = shap_imp

            if "shap_fig" in st.session_state and st.session_state["shap_fig"] is not None:
                st.pyplot(st.session_state["shap_fig"])
            if "shap_imp" in st.session_state and st.session_state["shap_imp"] is not None:
                st.dataframe(st.session_state["shap_imp"], use_container_width=True)

        with tab4:
            st.subheader("⚙️ Feature Engineering")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔧 Auto-Engineer Features", type="primary"):
                    with st.spinner("Engineering features..."):
                        engineered = apply_feature_engineering(df, st.session_state.sme_inputs, target)
                        st.session_state.engineered_df = engineered
                        new_cols = [c for c in engineered.columns if c not in df.columns]
                        st.success(f"✅ Created {len(new_cols)} new features: {', '.join(new_cols[:5])}")

            with col2:
                if st.session_state.api_key and st.button("🤖 Get AI Suggestions"):
                    with st.spinner("Asking AI for feature ideas..."):
                        from src.gemini_utils import suggest_features
                        suggestions = suggest_features(
                            st.session_state.api_key, st.session_state.eda_summary,
                            st.session_state.sme_inputs, df.columns.tolist(),
                        )
                        st.session_state.fe_suggestions = suggestions

            if st.session_state.fe_suggestions:
                with st.expander("🤖 AI Feature Suggestions", expanded=True):
                    st.markdown(st.session_state.fe_suggestions)

            # Feature selection
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("✅ Feature Selection")
            use_df = st.session_state.engineered_df if st.session_state.engineered_df is not None else df
            available_features = [c for c in use_df.columns if c != target]
            st.session_state.selected_features = st.multiselect(
                "Select features to keep for modeling",
                available_features, default=available_features,
            )
            st.info(f"📌 {len(st.session_state.selected_features)} features selected for modeling")

    next_step_button(page)


# ═══════════════════════════════════════════════════════════════
# PAGE 4: MODEL BUILDING
# ═══════════════════════════════════════════════════════════════
elif page == "🏗️ Model Building":
    st.markdown("""<div class="hero-header"><h1>🏗️ Model Factory & Training</h1>
    <p>Multi-model training, hyperparameter tuning, and MLflow experiment tracking</p></div>""", unsafe_allow_html=True)

    if st.session_state.train_data is None:
        st.warning("⚠️ Please upload training data first.")
    else:
        from src.utils import detect_task_type, split_columns
        from src.model_trainer import (
            get_available_models, build_pipeline, get_hyperparameter_grids,
            train_with_mlflow, compare_runs,
        )
        from sklearn.model_selection import train_test_split

        df = st.session_state.train_data
        target = st.session_state.target_variable
        task_type = detect_task_type(df[target])

        # Use engineered data if available
        use_df = st.session_state.engineered_df if st.session_state.engineered_df is not None else df
        if st.session_state.selected_features:
            features_to_use = st.session_state.selected_features
        else:
            features_to_use = [c for c in use_df.columns if c != target]

        st.info(f"📊 Task Type: **{task_type.title()}** | Features: **{len(features_to_use)}** | Target: **{target}**")

        # ── Choose Final Variables ────────────────────────────
        st.subheader("🎯 Choose Final Variables")
        features_to_use = st.multiselect(
            "Select the variables to use for model training",
            features_to_use,
            default=features_to_use,
            key="final_vars",
        )
        st.success(f"✅ **{len(features_to_use)}** variables selected for training")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            available_models = get_available_models(task_type)
            selected_models = st.multiselect("Select models to train", list(available_models.keys()), default=list(available_models.keys())[:3])
        with col2:
            test_size = st.slider("Test split ratio", 0.1, 0.4, 0.2, 0.05)
            do_grid_search = st.checkbox("Enable GridSearchCV (slower)", value=False)

        if st.button("🚀 Train All Selected Models", type="primary"):
            X = use_df[features_to_use]
            y = use_df[target]
            if y.dtype == "object":
                y = y.astype("category").cat.codes

            numeric_cols, cat_cols = split_columns(use_df[features_to_use + [target]], target)
            # Filter to only selected features
            numeric_cols = [c for c in numeric_cols if c in features_to_use]
            cat_cols = [c for c in cat_cols if c in features_to_use]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            grids = get_hyperparameter_grids(task_type)

            progress = st.progress(0)
            results = []

            for i, model_name in enumerate(selected_models):
                st.write(f"🔄 Training **{model_name}**...")
                model = available_models[model_name]
                pipeline = build_pipeline(model, numeric_cols, cat_cols)
                params = grids.get(model_name, {}) if do_grid_search else None

                result = train_with_mlflow(
                    pipeline, X_train, y_train, X_test, y_test,
                    model_name, task_type, st.session_state.experiment_id,
                    params=params, do_grid_search=do_grid_search and bool(params),
                    iteration=st.session_state.iteration,
                    sme_inputs=st.session_state.sme_inputs,
                    feature_list=features_to_use,
                )
                results.append(result)
                st.session_state.trained_models[model_name] = result
                progress.progress((i + 1) / len(selected_models))

            # Find best model
            primary_metric = "accuracy" if task_type == "classification" else "r2"
            best = max(results, key=lambda r: r["metrics"].get(primary_metric, 0))
            st.session_state.best_model = best["model_name"]
            st.session_state.best_run_id = best["run_id"]

            st.success(f"🏆 Best Model: **{best['model_name']}** ({primary_metric}: {best['metrics'][primary_metric]:.4f})")

        # Show results
        if st.session_state.trained_models:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("📊 Model Comparison")

            comparison_data = []
            for name, result in st.session_state.trained_models.items():
                row = {"Model": name, "Run ID": result["run_id"][:8]}
                row.update(result["metrics"])
                comparison_data.append(row)

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            # MLflow runs
            st.subheader("📋 MLflow Experiment Runs")
            all_runs = compare_runs(st.session_state.experiment_id)
            if not all_runs.empty:
                display_cols = [c for c in all_runs.columns if c.startswith(("params.model_name", "metrics.", "run_id", "start_time"))]
                if display_cols:
                    st.dataframe(all_runs[display_cols].head(20), use_container_width=True)

    next_step_button(page)


# ═══════════════════════════════════════════════════════════════
# PAGE 5: OOT VALIDATION
# ═══════════════════════════════════════════════════════════════
elif page == "✅ OOT Validation":
    st.markdown("""<div class="hero-header"><h1>✅ Out-of-Time Validation</h1>
    <p>Validate your best model on holdout data and provide feedback</p></div>""", unsafe_allow_html=True)

    if not st.session_state.best_run_id:
        st.warning("⚠️ Please train models first.")
    elif st.session_state.oot_data is None:
        st.warning("⚠️ Please upload OOT data on the Data Upload page.")
    else:
        from src.validation import (
            validate_oot, generate_validation_report,
            generate_confusion_matrix_plot, record_feedback,
            generate_performance_comparison_plot,
        )
        from src.utils import detect_task_type

        target = st.session_state.target_variable
        task_type = detect_task_type(st.session_state.train_data[target])
        model_uri = f"runs:/{st.session_state.best_run_id}/model"

        st.info(f"🏆 Validating: **{st.session_state.best_model}** (Run: `{st.session_state.best_run_id[:8]}`)")

        if st.button("🔄 Run OOT Validation", type="primary"):
            with st.spinner("Scoring OOT data..."):
                result = validate_oot(model_uri, st.session_state.oot_data, target, task_type)
                st.session_state.oot_metrics = result["metrics"]

                # Display metrics
                st.subheader("📊 OOT Metrics")
                cols = st.columns(len(result["metrics"]))
                for i, (k, v) in enumerate(result["metrics"].items()):
                    cols[i].metric(k.replace("oot_", "").replace("_", " ").title(), f"{v:.4f}")

                # Confusion matrix for classification
                if task_type == "classification":
                    cm_fig = generate_confusion_matrix_plot(result["y_true"], result["y_pred"])
                    st.plotly_chart(cm_fig, use_container_width=True)

                # Train vs OOT comparison
                if st.session_state.best_model in st.session_state.trained_models:
                    train_metrics = st.session_state.trained_models[st.session_state.best_model]["metrics"]
                    comp_fig = generate_performance_comparison_plot(train_metrics, result["metrics"], task_type)
                    if comp_fig:
                        st.plotly_chart(comp_fig, use_container_width=True)

                # Report
                report = generate_validation_report(result["metrics"], task_type)
                st.markdown(report)

        # Feedback loop
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📝 Model Acceptance")
        feedback_col1, feedback_col2 = st.columns(2)

        with feedback_col1:
            accepted = st.radio("Is the model performance acceptable?", ["Yes ✅", "No ❌"], index=0)
        with feedback_col2:
            feedback_text = st.text_area("Feedback / Comments", placeholder="Why is the performance not acceptable?")

        if st.button("📨 Submit Feedback"):
            is_accepted = accepted == "Yes ✅"
            st.session_state.oot_accepted = is_accepted

            if st.session_state.oot_metrics:
                record_feedback(st.session_state.best_run_id, is_accepted, feedback_text, st.session_state.oot_metrics)

            if is_accepted:
                st.success("✅ Model accepted! Proceed to Monitoring, Explainability, and Documentation.")
            else:
                st.session_state.iteration += 1
                st.warning(f"🔄 Model rejected. Iteration incremented to **{st.session_state.iteration}**. Go back to Feature Engineering or Model Building to iterate.")

    next_step_button(page)


# ═══════════════════════════════════════════════════════════════
# PAGE 6: MONITORING PLAN
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Monitoring Plan":
    st.markdown("""<div class="hero-header"><h1>📈 Ongoing Model Monitoring Plan</h1>
    <p>PSI analysis, drift detection, and monitoring KPI definitions</p></div>""", unsafe_allow_html=True)

    from src.monitoring import (
        generate_psi_report, detect_drift, generate_monitoring_kpis, generate_drift_plot,
    )

    tab1, tab2, tab3 = st.tabs(["📊 PSI Analysis", "📋 Monitoring KPIs", "🤖 AI Monitoring Plan"])

    with tab1:
        if st.session_state.train_data is not None and st.session_state.oot_data is not None:
            st.subheader("Population Stability Index (PSI)")
            target = st.session_state.target_variable
            psi_df = generate_psi_report(st.session_state.train_data, st.session_state.oot_data, target)

            if not psi_df.empty:
                st.dataframe(psi_df, use_container_width=True)
                drift_fig = generate_drift_plot(psi_df)
                if drift_fig:
                    st.plotly_chart(drift_fig, use_container_width=True)

                # Summary
                stable = len(psi_df[psi_df["PSI"] < 0.1])
                moderate = len(psi_df[(psi_df["PSI"] >= 0.1) & (psi_df["PSI"] < 0.25)])
                sig = len(psi_df[psi_df["PSI"] >= 0.25])

                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><h3>{stable}</h3><p>✅ Stable Features</p></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><h3>{moderate}</h3><p>⚠️ Moderate Shift</p></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><h3>{sig}</h3><p>🔴 Significant Shift</p></div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Need both training and OOT data for PSI analysis.")

    with tab2:
        st.subheader("Monitoring KPI Framework")
        kpis = generate_monitoring_kpis()
        for category, items in kpis.items():
            st.markdown(f"### {category}")
            kpi_rows = []
            for kpi, details in items.items():
                kpi_rows.append({
                    "KPI": kpi,
                    "Frequency": details["frequency"],
                    "Threshold": details["threshold"],
                    "Owner": details["owner"],
                })
            st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True)

    with tab3:
        st.subheader("🤖 AI-Generated Monitoring Plan")
        if st.button("📝 Generate Monitoring Plan", type="primary"):
            if not st.session_state.api_key:
                st.warning("⚠️ Please enter your Gemini API key.")
            else:
                with st.spinner("Generating monitoring plan..."):
                    from src.gemini_utils import generate_monitoring_recommendations
                    model_info = {
                        "best_model": st.session_state.best_model,
                        "iteration": st.session_state.iteration,
                        "problem_statement": st.session_state.problem_statement,
                        "metrics": st.session_state.oot_metrics,
                    }
                    plan = generate_monitoring_recommendations(st.session_state.api_key, model_info)
                    st.session_state.monitoring_plan = plan

        if st.session_state.monitoring_plan:
            st.markdown(st.session_state.monitoring_plan)

    next_step_button(page)


# ═══════════════════════════════════════════════════════════════
# PAGE 7: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Explainability":
    st.markdown("""<div class="hero-header"><h1>🔍 Model Explainability KPIs</h1>
    <p>SHAP analysis, feature importance, and partial dependence for the final model</p></div>""", unsafe_allow_html=True)

    if not st.session_state.best_run_id:
        st.warning("⚠️ Please train models first.")
    else:
        from src.explainability import (
            generate_shap_plots, get_feature_importance_table,
            generate_importance_plot, create_explainability_summary,
        )
        import mlflow.sklearn

        target = st.session_state.target_variable
        model_uri = f"runs:/{st.session_state.best_run_id}/model"

        tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🔬 SHAP Analysis", "📋 Summary"])

        with tab1:
            st.subheader("Feature Importance Rankings")
            if st.button("📊 Calculate Importance", key="calc_imp"):
                with st.spinner("Loading model and calculating importance..."):
                    pipeline = mlflow.sklearn.load_model(model_uri)
                    importance_df = get_feature_importance_table(pipeline)
                    st.session_state["explain_importance"] = importance_df

            if "explain_importance" in st.session_state:
                imp_df = st.session_state["explain_importance"]
                st.dataframe(imp_df, use_container_width=True)
                fig = generate_importance_plot(imp_df)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("SHAP Value Analysis")
            if st.button("🔬 Generate SHAP Plots", key="gen_shap"):
                with st.spinner("Computing SHAP values for final model..."):
                    pipeline = mlflow.sklearn.load_model(model_uri)
                    df = st.session_state.train_data
                    features_to_use = st.session_state.selected_features or [c for c in df.columns if c != target]
                    X = df[features_to_use]

                    fig_summary, fig_bar, feat_names = generate_shap_plots(pipeline, X)
                    if fig_summary:
                        st.pyplot(fig_summary)
                    if fig_bar:
                        st.pyplot(fig_bar)
                    if isinstance(feat_names, str):
                        st.warning(f"SHAP Error: {feat_names}")

        with tab3:
            st.subheader("Explainability Summary")
            if st.session_state.best_model and st.session_state.best_model in st.session_state.trained_models:
                result = st.session_state.trained_models[st.session_state.best_model]
                st.json({
                    "Model": st.session_state.best_model,
                    "Run ID": st.session_state.best_run_id[:8],
                    "Metrics": result["metrics"],
                    "N Features": len(st.session_state.selected_features) if st.session_state.selected_features else "N/A",
                })

    next_step_button(page)


# ═══════════════════════════════════════════════════════════════
# PAGE 8: DOCUMENTATION
# ═══════════════════════════════════════════════════════════════
elif page == "📝 Documentation":
    st.markdown("""<div class="hero-header"><h1>📝 Model Documentation</h1>
    <p>Auto-generated model documentation and model card</p></div>""", unsafe_allow_html=True)

    from src.governance import generate_model_card

    tab1, tab2 = st.tabs(["📄 Model Card", "🤖 AI Documentation"])

    with tab1:
        st.subheader("Model Card")
        if st.session_state.best_run_id and st.session_state.best_model:
            model_info = {
                "model_name": st.session_state.best_model,
                "run_id": st.session_state.best_run_id,
                "task_type": "classification",
                "train_timestamp": datetime.now().isoformat(),
                "iteration": st.session_state.iteration,
                "n_train_samples": len(st.session_state.train_data) if st.session_state.train_data is not None else "N/A",
                "n_features": len(st.session_state.selected_features) if st.session_state.selected_features else "N/A",
                "metrics": st.session_state.trained_models.get(st.session_state.best_model, {}).get("metrics", {}),
                "problem_statement": st.session_state.problem_statement,
                "sme_inputs": st.session_state.sme_inputs,
            }
            card = generate_model_card(model_info)
            st.markdown(card)

            # Download
            st.download_button("📥 Download Model Card", card, "model_card.md", "text/markdown")
        else:
            st.warning("⚠️ Train a model first to generate the model card.")

    with tab2:
        st.subheader("🤖 AI-Generated Documentation")
        if st.button("📝 Generate Full Documentation", type="primary"):
            if not st.session_state.api_key:
                st.warning("⚠️ Please enter your Gemini API key.")
            else:
                with st.spinner("Generating comprehensive documentation..."):
                    from src.gemini_utils import generate_documentation
                    model_info = {
                        "model_name": st.session_state.best_model,
                        "run_id": st.session_state.best_run_id,
                        "iteration": st.session_state.iteration,
                        "problem_statement": st.session_state.problem_statement,
                        "sme_inputs": st.session_state.sme_inputs,
                        "metrics": st.session_state.trained_models.get(st.session_state.best_model, {}).get("metrics", {}),
                        "oot_metrics": st.session_state.oot_metrics,
                        "selected_features": st.session_state.selected_features[:20],
                        "eda_summary": st.session_state.eda_summary[:500],
                    }
                    doc = generate_documentation(st.session_state.api_key, model_info)
                    st.session_state.model_doc = doc

        if st.session_state.model_doc:
            st.markdown(st.session_state.model_doc)
            st.download_button("📥 Download Documentation", st.session_state.model_doc, "model_documentation.md", "text/markdown")

    next_step_button(page)


# ═══════════════════════════════════════════════════════════════
# PAGE 9: GOVERNANCE
# ═══════════════════════════════════════════════════════════════
elif page == "🛡️ Governance":
    st.markdown("""<div class="hero-header"><h1>🛡️ Model Governance Controls</h1>
    <p>Audit trail, compliance checklist, fairness assessment, and artifact export</p></div>""", unsafe_allow_html=True)

    from src.governance import (
        create_audit_trail, generate_governance_checklist,
        export_all_artifacts,
    )

    tab1, tab2, tab3 = st.tabs(["📋 Audit Trail", "✅ Compliance Checklist", "📦 Export Artifacts"])

    with tab1:
        st.subheader("Experiment Audit Trail")
        if st.session_state.experiment_id:
            audit = create_audit_trail(st.session_state.experiment_id)
            if not audit.empty:
                st.dataframe(audit, use_container_width=True)

                # Iteration summary
                st.subheader("Iteration History")
                if "params.iteration" in audit.columns:
                    iterations = audit["params.iteration"].nunique()
                    st.info(f"📊 Total iterations: **{iterations}** | Total runs: **{len(audit)}**")
            else:
                st.info("No runs logged yet.")

    with tab2:
        st.subheader("Governance Compliance Checklist")
        checklist = generate_governance_checklist()

        for category, items in checklist.items():
            st.markdown(f"### {category}")
            for item in items:
                checked = st.checkbox(item["item"], key=f"gov_{category}_{item['item']}")
                if checked:
                    item["status"] = "✅"

    with tab3:
        st.subheader("Export All Artifacts")
        st.markdown("Download all experiment artifacts, model files, and governance documents as a ZIP.")

        if st.button("📦 Export All Artifacts", type="primary"):
            with st.spinner("Packaging artifacts..."):
                zip_data = export_all_artifacts(st.session_state.experiment_id)
                st.download_button(
                    "📥 Download ZIP", zip_data,
                    f"mlflow_artifacts_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                    "application/zip",
                )

    next_step_button(page)


# ═══════════════════════════════════════════════════════════════
# PAGE 10: SEARCH AGENT
# ═══════════════════════════════════════════════════════════════
elif page == "🔎 Search Agent":
    st.markdown("""<div class="hero-header"><h1>🔎 Model Development Search Agent</h1>
    <p>Ask any question about your model development process</p></div>""", unsafe_allow_html=True)

    from src.qa_utils import build_experiment_context, get_run_comparison, format_chat_history
    from src.gemini_utils import answer_query

    tab1, tab2 = st.tabs(["💬 Chat Agent", "📊 Run Comparison"])

    with tab1:
        st.subheader("💬 Ask Me Anything")
        st.markdown("Ask questions about your experiment runs, model performance, features, and more.")

        # Display chat history
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").write(msg["content"])

        # Chat input
        query = st.chat_input("Ask about your model development...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            if not st.session_state.api_key:
                answer = "⚠️ Please enter your Gemini API key in the sidebar."
            else:
                with st.spinner("Searching experiment history..."):
                    context = build_experiment_context(st.session_state.experiment_id)
                    # Add session state context
                    context += f"\n\n## Current Session Info\n"
                    context += f"- Best Model: {st.session_state.best_model}\n"
                    context += f"- Iteration: {st.session_state.iteration}\n"
                    context += f"- OOT Accepted: {st.session_state.oot_accepted}\n"
                    context += f"- Problem: {st.session_state.problem_statement}\n"
                    context += f"- Features Selected: {len(st.session_state.selected_features)}\n"

                    answer = answer_query(st.session_state.api_key, query, context)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

    with tab2:
        st.subheader("📊 Run Comparison Table")
        if st.session_state.experiment_id:
            comparison = get_run_comparison(st.session_state.experiment_id)
            if not comparison.empty:
                st.dataframe(comparison, use_container_width=True)
            else:
                st.info("No runs to compare. Train some models first!")

        # Quick query examples
        st.markdown("### 💡 Example Queries")
        examples = [
            "Which model had the best accuracy?",
            "Compare Random Forest and XGBoost performance",
            "What features were most important?",
            "How many iterations did we run?",
            "What was the OOT performance?",
            "Summarize the model development process",
        ]
        for ex in examples:
            st.code(ex, language=None)
