# 🧠 MLflow Agentic ML Lifecycle Platform

An end-to-end machine learning lifecycle platform powered by **Streamlit**, **MLflow**, and **Google Gemini AI**.

## ✨ Features

| Step | Description |
|------|-------------|
| 📤 Data Upload | Upload CSV/Excel, define problem statement & SME inputs |
| 📊 Vibe Analysis | Automated EDA with AI-powered insights |
| 🔧 Feature Engineering | IV, SHAP, shadow models, auto-feature creation |
| 🏗️ Model Building | 5 algorithms, GridSearchCV, MLflow tracking |
| ✅ OOT Validation | Out-of-time validation with feedback loop |
| 📈 Monitoring | PSI analysis, drift detection, monitoring KPIs |
| 🔍 Explainability | SHAP plots, feature importance, PDP |
| 📝 Documentation | Auto-generated model cards & AI documentation |
| 🛡️ Governance | Audit trail, compliance checklist, artifact export |
| 🔎 Search Agent | Chat-based Q&A over experiment history |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 Setup

Enter your **Google Gemini API key** in the sidebar to enable AI features (EDA summaries, feature suggestions, documentation generation, search agent).

## 🛠️ Tech Stack

- **UI:** Streamlit
- **Experiment Tracking:** MLflow
- **AI Agent:** Google Gemini via LangChain
- **ML:** scikit-learn, XGBoost, SHAP
- **Visualization:** Plotly, Matplotlib
