"""
Gemini LLM integration: EDA summary, FE suggestions, doc generation.
"""
import os


def get_llm(api_key: str):
    """Initialize ChatGoogleGenerativeAI."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3,
        )
        return llm
    except Exception as e:
        return None


def _invoke_llm(llm, prompt: str) -> str:
    """Safely invoke the LLM with retry logic for rate limits."""
    import time
    max_retries = 3
    base_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                # Rate limit hit
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)  # 5, 10, 20 seconds
                    print(f"LLM Rate Limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"⚠️ LLM Error: Rate limit exceeded after {max_retries} retries. Please try again later."
            else:
                # Proceed to other errors immediately
                return f"⚠️ LLM Error: {error_msg}"
    
    return "⚠️ LLM Error: Unknown error occurred."


def summarize_eda(api_key: str, stats_text: str, problem_statement: str, sme_inputs: str) -> str:
    """Generate an AI summary of the EDA findings."""
    llm = get_llm(api_key)
    if llm is None:
        return "⚠️ LLM not available. Please provide a valid API key."

    prompt = f"""You are a senior data scientist. Analyze the following dataset statistics and provide a comprehensive EDA summary.

    (Note: Statistics might be truncated for brevity)

**Business Problem:** {problem_statement}

**SME/Domain Inputs:** {sme_inputs}

**Dataset Statistics:**
{stats_text[:12000]} 

Please provide:
1. **Data Quality Assessment**: Missing values, data types, potential issues
2. **Key Statistical Insights**: Important distributions, anomalies, patterns
3. **Feature Observations**: Which features look promising and why
4. **Potential Concerns**: Data quality issues, class imbalance, outliers
5. **Recommendations**: Next steps for feature engineering

Format your response in clear markdown with headers."""

    return _invoke_llm(llm, prompt)


def suggest_features(api_key: str, eda_summary: str, sme_inputs: str, columns: list) -> str:
    """LLM suggests feature engineering ideas."""
    llm = get_llm(api_key)
    if llm is None:
        return "⚠️ LLM not available."

    prompt = f"""You are a senior data scientist. Based on the EDA summary and domain expertise, suggest feature engineering ideas.

**EDA Summary:**
{eda_summary[:10000]}

**SME/Domain Inputs:**
{sme_inputs}

**Available Columns:** {', '.join(columns)}

Please suggest:
1. **Interaction Features**: Combinations of existing features
2. **Transformations**: Log, binning, encoding strategies
3. **Domain Features**: Based on SME inputs
4. **Feature Selection Strategy**: Which features to keep/drop and why

Be specific with column names. Format in markdown."""

    return _invoke_llm(llm, prompt)


def generate_documentation(api_key: str, model_info: dict) -> str:
    """Generate comprehensive model documentation."""
    llm = get_llm(api_key)
    if llm is None:
        return "⚠️ LLM not available."

    prompt = f"""You are a model risk management expert. Generate comprehensive model documentation based on the following information.

**Model Information:**
{str(model_info)[:10000]}

Generate a complete Model Development Document covering:
1. **Executive Summary**
2. **Business Context & Objective**
3. **Data Description & Quality**
4. **Feature Engineering & Selection**
5. **Model Development Approach**
6. **Model Performance & Validation**
7. **Model Limitations & Assumptions**
8. **Implementation Recommendations**
9. **Ongoing Monitoring Plan**
10. **Appendix**: Technical details

Format as a professional markdown document suitable for model governance review."""

    return _invoke_llm(llm, prompt)


def answer_query(api_key: str, query: str, context: str) -> str:
    """Answer user queries about model development."""
    llm = get_llm(api_key)
    if llm is None:
        return "⚠️ LLM not available."

    prompt = f"""You are a helpful AI assistant with access to a machine learning experiment tracking system. Answer the user's question based on the provided context.

**Experiment Context:**
{context}

**User Question:** {query}

Provide a clear, concise answer. If the information is not in the context, say so. Include specific numbers, model names, and metrics when available."""

    return _invoke_llm(llm, prompt)


def generate_monitoring_recommendations(api_key: str, model_info: dict) -> str:
    """Generate ongoing model monitoring recommendations."""
    llm = get_llm(api_key)
    if llm is None:
        return "⚠️ LLM not available."

    prompt = f"""You are a model risk management expert. Create a comprehensive ongoing model monitoring plan.

**Model Information:**
{str(model_info)[:10000]}

Create a monitoring plan that covers:
1. **Performance Monitoring**: KPIs, thresholds, frequency
2. **Data Drift Detection**: PSI, CSI, feature drift monitoring
3. **Concept Drift Detection**: Methods and triggers
4. **Alert Framework**: Escalation procedures
5. **Retraining Strategy**: When and how to retrain
6. **Reporting Requirements**: Stakeholder communication plan
7. **Governance Checkpoints**: Periodic review schedule

Format as a professional markdown document."""

    return _invoke_llm(llm, prompt)
