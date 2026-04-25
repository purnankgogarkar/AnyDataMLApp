"""AI-powered insights and suggestions."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import AI_PROVIDERS
from .data_handler import classify_column, get_numeric_columns, get_categorical_columns

logger = logging.getLogger(__name__)


def detect_ai_provider(api_key: str) -> Optional[str]:
    """Detect AI provider from API key format."""
    if not api_key:
        return None
    
    for provider, config in AI_PROVIDERS.items():
        if api_key.startswith(config["prefix"]):
            return provider
    
    return None


def validate_api_key(api_key: str) -> Tuple[bool, str, Optional[str]]:
    """Validate API key and return provider info."""
    if not api_key:
        return False, "No API key provided", None
    
    provider = detect_ai_provider(api_key)
    if not provider:
        return False, "Invalid API key format. Supported: OpenAI (sk-), Groq (gsk_), Anthropic (sk-ant-)", None
    
    return True, f"{provider.upper()} detected", provider


def _call_ai_api(
    provider: str,
    model: str,
    prompt: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 200
) -> Optional[str]:
    """Unified AI API call handler for all providers."""
    try:
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        
        elif provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        
        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
    
    except Exception as e:
        logger.error(f"AI API call failed for {provider}: {str(e)}")
        return None
    
    return None



# ========== AI Feature Implementations ==========

def get_model_explanation(
    run_data: Dict[str, Any],
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str]
) -> str:
    """Generate AI-powered model explanation."""
    if not provider or not api_key or not model:
        return template_model_explanation(run_data)
    
    top_feats = ", ".join([f["feature"] for f in run_data.get("feature_importance", [])[:5]]) or "N/A"
    metrics_text = ", ".join([f"{k}={v}" for k, v in run_data.get("metrics", {}).items()])
    
    prompt = (
        f"You are a data-science assistant. A {run_data['problem_type']} model ({run_data['model_name']}) "
        f"was trained to predict '{run_data['target']}' for business outcome '{run_data.get('business_outcome', 'NA')}'. "
        f"Metrics: {metrics_text}. Top features: {top_feats}. "
        f"Write a short (3-4 sentence) plain-English explanation of what this means for the business "
        f"and which features matter most. Avoid jargon and technical terms."
    )
    
    result = _call_ai_api(provider, model, prompt, api_key, temperature=0.7, max_tokens=200)
    return result if result else template_model_explanation(run_data)


def get_feature_engineering_suggestions(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str]
) -> str:
    """Get AI suggestions for feature engineering."""
    if not provider or not api_key or not model:
        return template_feature_suggestions(df, target, features)
    
    numeric_cols = [c for c in features if classify_column(df[c]) == "numeric"]
    categorical_cols = [c for c in features if classify_column(df[c]) == "categorical"]
    
    data_desc = (
        f"Dataset: {len(df)} rows, target='{target}' ({classify_column(df[target])}) "
        f"Numeric features: {numeric_cols}. Categorical features: {categorical_cols}."
    )
    
    prompt = (
        f"You are a feature engineering expert. {data_desc}\n"
        f"Suggest 3 concrete feature engineering ideas to improve model performance. "
        f"Be specific (e.g., 'create interaction between X and Y', 'extract month from date Z'). "
        f"Format as bullet points."
    )
    
    result = _call_ai_api(provider, model, prompt, api_key, temperature=0.7, max_tokens=250)
    return result if result else template_feature_suggestions(df, target, features)


def get_smart_model_selection(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    problem_type: str,
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str]
) -> str:
    """Get AI recommendation for model selection."""
    if not provider or not api_key or not model:
        return template_model_selection(problem_type)
    
    data_desc = (
        f"Dataset: {len(df)} rows, {len(features)} features, "
        f"problem_type={problem_type}, target='{target}'"
    )
    
    prompt = (
        f"You are an ML expert. {data_desc}\n"
        f"Which model would perform best? (Choose from: linear_regression, random_forest, xgboost for {problem_type})\n"
        f"Explain why in 2 sentences. Consider: dataset size, feature count, interpretability needs."
    )
    
    result = _call_ai_api(provider, model, prompt, api_key, temperature=0.5, max_tokens=150)
    return result if result else template_model_selection(problem_type)


def get_business_recommendations(
    run_data: Dict[str, Any],
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str]
) -> str:
    """Get AI-generated business recommendations."""
    if not provider or not api_key or not model:
        return template_business_recommendations(run_data)
    
    metrics_text = ", ".join([f"{k}={v}" for k, v in run_data.get("metrics", {}).items()])
    top_feats = ", ".join([f["feature"] for f in run_data.get("feature_importance", [])[:3]])
    
    prompt = (
        f"You are a business analyst reviewing ML model results. "
        f"Model: {run_data['model_name']}, Target: {run_data['target']}, "
        f"Metrics: {metrics_text}, Top features: {top_feats}.\n"
        f"What are 3 actionable business recommendations based on this model? "
        f"Be specific and focus on business impact."
    )
    
    result = _call_ai_api(provider, model, prompt, api_key, temperature=0.7, max_tokens=200)
    return result if result else template_business_recommendations(run_data)


def get_data_quality_insights(
    quality_report: Dict[str, Any],
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str]
) -> str:
    """Get AI insights on data quality."""
    if not provider or not api_key or not model:
        return template_data_quality_insights(quality_report)
    
    issues_summary = "\n".join([
        f"- {issue['column']}: {', '.join(issue['issues'])}"
        for issue in quality_report.get("column_issues", [])[:5]
    ])
    
    prompt = (
        f"Data Quality Issues Detected:\n{issues_summary}\n"
        f"Provide brief recommendations to improve data quality before ML modeling. "
        f"Focus on most critical issues."
    )
    
    result = _call_ai_api(provider, model, prompt, api_key, temperature=0.7, max_tokens=200)
    return result if result else template_data_quality_insights(quality_report)


def get_hyperparameter_suggestions(
    df: pd.DataFrame,
    problem_type: str,
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str]
) -> str:
    """Get AI suggestions for hyperparameters."""
    if not provider or not api_key or not model:
        return template_hyperparameter_suggestions(df, problem_type)
    
    data_size = len(df)
    prompt = (
        f"For {problem_type} with {data_size} rows, suggest good hyperparameters for Random Forest and XGBoost. "
        f"Be specific (e.g., n_estimators=100, max_depth=7). Keep response concise."
    )
    
    result = _call_ai_api(provider, model, prompt, api_key, temperature=0.5, max_tokens=150)
    return result if result else template_hyperparameter_suggestions(df, problem_type)


# ========== Template Fallbacks (No AI) ==========

def template_model_explanation(run_data: Dict[str, Any]) -> str:
    """Fallback explanation without AI."""
    metrics = run_data.get("metrics", {})
    top = [f["feature"] for f in run_data.get("feature_importance", [])[:3]]
    top_str = ", ".join(top) if top else "the selected features"
    metric_str = ", ".join([f"{k.upper()}={v}" for k, v in metrics.items()])
    
    return (
        f"The model predicts '{run_data['target']}' using a {run_data['model_name']} approach "
        f"({run_data['problem_type']}). Evaluation metrics: {metric_str}. "
        f"The most influential features are {top_str}. "
        f"This supports the business goal of '{run_data.get('business_outcome', 'NA')}'."
    )


def template_feature_suggestions(df: pd.DataFrame, target: str, features: List[str]) -> str:
    """Fallback feature suggestions."""
    numeric = get_numeric_columns(df)
    categorical = get_categorical_columns(df)
    
    suggestions = [
        f"- Create interaction features between top numeric columns: {numeric[:2]}",
        f"- One-hot encode categorical features: {categorical[:2]}",
        f"- Extract temporal features (if dates exist in your data)",
    ]
    
    return "\n".join(suggestions)


def template_model_selection(problem_type: str) -> str:
    """Fallback model recommendation."""
    if problem_type == "regression":
        return "Random Forest or XGBoost typically perform well for regression. XGBoost is more accurate but slower."
    elif problem_type == "classification":
        return "Random Forest works well for balanced data. Consider XGBoost for better accuracy on imbalanced data."
    else:
        return "Random Forest is a good baseline for time-series."


def template_business_recommendations(run_data: Dict[str, Any]) -> str:
    """Fallback business recommendations."""
    top_feat = run_data.get("feature_importance", [{}])[0].get("feature", "top_feature")
    return (
        f"1. Focus on '{top_feat}' as it's the most influential predictor.\n"
        f"2. Monitor model performance regularly to ensure accuracy.\n"
        f"3. Use predictions to drive '{run_data.get('business_outcome', 'business outcome')}'."
    )


def template_data_quality_insights(quality_report: Dict[str, Any]) -> str:
    """Fallback data quality insights."""
    issues_count = len(quality_report.get("column_issues", []))
    if issues_count == 0:
        return "✅ Data quality looks good! No major issues detected."
    return f"⚠️ {issues_count} columns have quality issues. Review and clean before training."


def template_hyperparameter_suggestions(df: pd.DataFrame, problem_type: str) -> str:
    """Fallback hyperparameter suggestions."""
    data_size = len(df)
    
    if data_size < 1000:
        rf_depth = 5
        xgb_depth = 3
    elif data_size < 10000:
        rf_depth = 10
        xgb_depth = 6
    else:
        rf_depth = 15
        xgb_depth = 8
    
    return f"Random Forest: max_depth={rf_depth}, n_estimators=100\nXGBoost: max_depth={xgb_depth}, n_estimators=100"
