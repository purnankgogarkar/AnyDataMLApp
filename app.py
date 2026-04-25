"""Main Streamlit app - Predictive Analytics Decision Assistant."""
import logging
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import json
from io import BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime
import shutil
import os
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Predictive Analytics Decision Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import utilities
from utils.config import (
    REGRESSION_MODELS, CLASSIFICATION_MODELS, TIMESERIES_MODELS,
    MISSING_STRATEGIES, AI_PROVIDERS, DATASETS_DIR, MODELS_DIR,
    PREDICTIONS_DIR, DB_PATH
)
from utils.storage import init_db, list_datasets, list_runs
from utils.data_handler import (
    load_dataset_file, process_and_save_dataset, load_dataset,
    suggest_problem_type, classify_column
)
from utils.analysis import analyze_dataset, get_data_quality_report
from utils.ml_pipeline import train_model, TrainConfig, load_model, get_predictions_path, get_model_path
from utils.ai_handler import (
    detect_ai_provider, validate_api_key, get_model_explanation,
    get_feature_engineering_suggestions, get_smart_model_selection,
    get_business_recommendations, get_data_quality_insights,
    get_hyperparameter_suggestions
)


# ========== Initialize Session State ==========
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "current_step": 0,
        "dataset_id": None,
        "current_df": None,
        "dataset_meta": None,
        "target": "NA",
        "business_outcome": "NA",
        "problem_type": "NA",
        "features": [],
        "analyze_result": None,
        "feature_engineering": {
            "missing_strategy": "mean",
            "missing_strategies_per_column": {},
            "encode_categorical": True,
            "scale_features": False,
            "extract_datetime": True,
        },
        "model_name": "auto",
        "run_id": None,
        "run_data": None,
        "ai_enabled": False,
        "ai_provider": None,
        "ai_model": None,
        "api_key": None,
        "show_clean_dialog": False,
        "cleanup_datasets": False,
        "cleanup_models": False,
        "cleanup_predictions": False,
        "cleanup_database": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()
init_db()


# ========== Sidebar Config ==========
def render_sidebar():
    """Render sidebar with AI config and navigation."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # AI API Key Input
        st.markdown("### 🤖 AI Features (Optional)")
        st.caption("Get AI-powered insights to help you build better models")
        
        api_key_input = st.text_input(
            "API Key",
            type="password",
            help="""Paste your API key from:
            • OpenAI (sk-...) from https://platform.openai.com/api-keys
            • Groq (gsk_...) from https://console.groq.com
            • Anthropic (sk-ant-...) from https://console.anthropic.com
            
            Don't have one? Skip for now - app works without it!
            """,
            value=st.session_state.api_key or ""
        )
        
        if api_key_input:
            is_valid, msg, provider = validate_api_key(api_key_input)
            
            if is_valid:
                st.session_state.api_key = api_key_input
                st.session_state.ai_provider = provider
                st.session_state.ai_enabled = True
                
                # Show provider info
                st.success(f"✅ {msg}")
                
                # Model selection
                available_models = AI_PROVIDERS[provider]["models"]
                default_model = AI_PROVIDERS[provider]["default"]
                
                st.session_state.ai_model = st.selectbox(
                    "AI Model",
                    available_models,
                    index=available_models.index(default_model),
                    help="""Different models, different speeds:
                    • Smallest = Fastest (gpt-3.5, Gemma)
                    • Medium = Balanced (Claude Sonnet)
                    • Largest = Best quality but slow (GPT-4)
                    """
                )
            else:
                st.error(f"❌ {msg}")
                st.session_state.ai_enabled = False
        else:
            st.info("ℹ️ No API key configured\n\nApp still works! You just won't get AI suggestions. You can add one anytime.")
            st.session_state.ai_enabled = False
            st.session_state.api_key = None
        
        # Reset button
        st.markdown("---")
        col_reset, col_clean = st.columns(2)
        
        with col_reset:
            if st.button("🔄 Reset Workflow", help="Start over with a fresh dataset - clears current progress", key="reset_btn", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key not in ["theme", "chat_history"]:
                        del st.session_state[key]
                init_session_state()
                st.rerun()
        
        with col_clean:
            if st.button("🗑️ Clean Storage", help="Selectively delete storage items", key="clean_btn", use_container_width=True):
                st.session_state.show_clean_dialog = True
        
        # Clean storage dialog
        if st.session_state.get("show_clean_dialog", False):
            st.markdown("---")
            st.markdown("### 🗑️ Selective Storage Cleanup")
            st.caption("Choose what you'd like to delete:")
            
            # Checkboxes for selection
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.cleanup_datasets = st.checkbox(
                    "📊 Delete Datasets",
                    value=st.session_state.cleanup_datasets,
                    help="Remove all uploaded dataset files"
                )
                st.session_state.cleanup_models = st.checkbox(
                    "🤖 Delete Models",
                    value=st.session_state.cleanup_models,
                    help="Remove all trained model files"
                )
            
            with col2:
                st.session_state.cleanup_predictions = st.checkbox(
                    "📈 Delete Predictions",
                    value=st.session_state.cleanup_predictions,
                    help="Remove all prediction output files"
                )
                st.session_state.cleanup_database = st.checkbox(
                    "📚 Delete Run History",
                    value=st.session_state.cleanup_database,
                    help="Clear all previous training runs from database"
                )
            
            # Show what will be deleted
            st.markdown("---")
            selected_items = []
            if st.session_state.cleanup_datasets:
                selected_items.append("📊 Datasets")
            if st.session_state.cleanup_models:
                selected_items.append("🤖 Models")
            if st.session_state.cleanup_predictions:
                selected_items.append("📈 Predictions")
            if st.session_state.cleanup_database:
                selected_items.append("📚 Run History")
            
            if selected_items:
                st.info(f"✓ Will delete: {', '.join(selected_items)}")
            else:
                st.warning("⚠️ Please select at least one item to delete")
            
            # Confirmation buttons
            col_delete, col_cancel = st.columns(2)
            
            with col_delete:
                if st.button("🗑️ Confirm Delete", key="confirm_clean", use_container_width=True):
                    if not selected_items:
                        st.error("Please select at least one item to delete")
                    else:
                        try:
                            deleted = clean_storage(
                                delete_datasets=st.session_state.cleanup_datasets,
                                delete_models=st.session_state.cleanup_models,
                                delete_predictions=st.session_state.cleanup_predictions,
                                delete_database=st.session_state.cleanup_database
                            )
                            st.session_state.show_clean_dialog = False
                            st.session_state.cleanup_datasets = False
                            st.session_state.cleanup_models = False
                            st.session_state.cleanup_predictions = False
                            st.session_state.cleanup_database = False
                            st.success(f"✅ Successfully deleted: {', '.join(deleted)}")
                            st.info("Refreshing app...")
                            # Reset session state
                            for key in list(st.session_state.keys()):
                                if key not in ["theme", "chat_history"]:
                                    del st.session_state[key]
                            init_session_state()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting storage: {str(e)}")
                            st.session_state.show_clean_dialog = False
            
            with col_cancel:
                if st.button("❌ Cancel", key="cancel_clean", use_container_width=True):
                    st.session_state.show_clean_dialog = False
                    st.session_state.cleanup_datasets = False
                    st.session_state.cleanup_models = False
                    st.session_state.cleanup_predictions = False
                    st.session_state.cleanup_database = False
                    st.rerun()
        
        # History
        st.markdown("---")
        st.markdown("### 📚 Recent Runs")
        st.caption("Models you've trained before")
        recent_runs = list_runs()
        if recent_runs:
            for run in recent_runs[:5]:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"🎯 **{run['target']}**")
                        st.caption(f"📊 {run['model_name']} ({run['problem_type'][:4]})")
                    with col2:
                        if st.button("↩️", key=f"load_run_{run['run_id']}", help="Load this previous model", use_container_width=True):
                            st.session_state.run_id = run['run_id']
                            st.session_state.current_step = 7  # Jump to results
                            st.rerun()
                st.divider()
        else:
            st.caption("No previous models yet - train your first one!")


# ========== Storage Management ==========

def clean_storage(delete_datasets=False, delete_models=False, delete_predictions=False, delete_database=False):
    """Clean storage directories and database selectively.
    
    Args:
        delete_datasets: Whether to delete historical datasets
        delete_models: Whether to delete trained models
        delete_predictions: Whether to delete prediction files
        delete_database: Whether to delete the database (clears run history)
    """
    logger.info(f"Starting storage cleanup... Datasets:{delete_datasets}, Models:{delete_models}, Predictions:{delete_predictions}, Database:{delete_database}")
    
    deleted_items = []
    
    try:
        # Delete datasets
        if delete_datasets and os.path.exists(DATASETS_DIR):
            shutil.rmtree(DATASETS_DIR)
            os.makedirs(DATASETS_DIR, exist_ok=True)
            logger.info(f"Deleted {DATASETS_DIR}")
            deleted_items.append("📊 Datasets")
        
        # Delete models
        if delete_models and os.path.exists(MODELS_DIR):
            shutil.rmtree(MODELS_DIR)
            os.makedirs(MODELS_DIR, exist_ok=True)
            logger.info(f"Deleted {MODELS_DIR}")
            deleted_items.append("🤖 Models")
        
        # Delete predictions
        if delete_predictions and os.path.exists(PREDICTIONS_DIR):
            shutil.rmtree(PREDICTIONS_DIR)
            os.makedirs(PREDICTIONS_DIR, exist_ok=True)
            logger.info(f"Deleted {PREDICTIONS_DIR}")
            deleted_items.append("📈 Predictions")
        
        # Delete database
        if delete_database and os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            logger.info(f"Deleted {DB_PATH}")
            init_db()
            logger.info("Database reinitialized")
            deleted_items.append("📚 Run History")
    
    except Exception as e:
        logger.error(f"Error cleaning storage: {str(e)}")
        raise
    
    logger.info(f"Storage cleanup completed. Deleted: {deleted_items}")
    return deleted_items


# ========== Step Components ==========

def step_0_upload():
    """Step 0: Data Upload"""
    st.header("📤 Upload Your Dataset")
    st.write("Upload CSV or Excel file with your data.")
    st.info("""💡 **What data do you need?**
    - Must have at least 10 rows (50+ is better)
    - Include the column you want to predict
    - Include columns that might help predict it
    - Can be messy - we'll clean it!
    """)
    
    # Show current data preview if already loaded
    if st.session_state.dataset_id and st.session_state.current_df is not None:
        with st.expander("👁️ Current Data Preview", expanded=False):
            df = st.session_state.current_df
            st.markdown(f"**Showing first 10 rows of {len(df)} total rows**")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            st.markdown("**Column Summary:**")
            col_summary = pd.DataFrame([
                {
                    "Column": col,
                    "Type": df[col].dtype,
                    "Unique Values": df[col].nunique(),
                    "Missing %": f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%"
                }
                for col in df.columns
            ])
            st.dataframe(col_summary, use_container_width=True, hide_index=True)
    
    uploaded_file = st.file_uploader(
        "Choose file",
        type=["csv", "xlsx", "xls"],
        key="upload_file",
        help="Upload CSV or Excel file. Can contain any data: sales, customers, weather, etc."
    )
    
    if uploaded_file:
        try:
            with st.spinner("Processing..."):
                # Stage 1: Read file
                file_content = uploaded_file.read()
                
                # Stage 2: Parse data
                meta = process_and_save_dataset(file_content, uploaded_file.name)
                
                # Stage 3: Load to memory
                df = load_dataset(meta["id"])
                
                # Stage 4: Update session
                st.session_state.dataset_id = meta["id"]
                st.session_state.dataset_meta = meta
                st.session_state.current_df = df
                st.session_state.features = [c["name"] for c in meta["columns"]]
            
            # Show success
            st.success("✅ Data loaded successfully!", icon="✅")
            
            # Display stats with explanations
            st.markdown("### 📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", meta["rows"], help="Number of data records")
            with col2:
                st.metric("Columns", meta["cols"], help="Number of data fields")
            with col3:
                st.metric("Size", f"{len(file_content) / 1024:.1f} KB", help="File size")
            with col4:
                st.metric("Status", "Ready ✅", help="Ready for next step")
            
            st.markdown("---")
            
            st.markdown("### 📋 Column Details")
            st.caption("Here's what columns we found in your data:")
            
            col_df = pd.DataFrame(meta["columns"])
            # Format for display
            col_df = col_df[['name', 'dtype', 'unique', 'null_pct']]
            col_df.columns = ['Column Name', 'Type', 'Unique Values', 'Missing %']
            st.dataframe(col_df, use_container_width=True, hide_index=True)
            
            st.markdown("**Column Type Legend:**")
            st.caption("• **numeric**: Numbers (age, price, quantity) | **categorical**: Text/categories (color, city) | **datetime**: Dates/times (Jan 1, 2024)")
            
            st.markdown("---")
            
            st.markdown("### 👁️ Data Preview (First 10 Rows)")
            st.caption("Here's what your data looks like:")
            st.dataframe(pd.DataFrame(meta["preview"]), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.success("✅ Ready for next step - Select your prediction target!")
        
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
            st.write("**What went wrong?**")
            st.caption("• File might be corrupted")
            st.caption("• File format not supported (use CSV or Excel)")
            st.caption("• File is empty")
            st.write(traceback.format_exc())


def step_1_target():
    """Step 1: Select Target Column"""
    if not st.session_state.dataset_id:
        st.warning("Upload data first")
        return
    
    st.header("🎯 Select Prediction Target")
    st.write("What do you want to predict?")
    st.info("💡 **Target Column**: The column you want your model to learn to predict. For example, if you want to predict sales, 'Sales' is your target.")
    
    cols = st.session_state.dataset_meta["columns"]
    col_names = [c["name"] for c in cols]
    
    target = st.selectbox(
        "Target column",
        col_names,
        index=0 if st.session_state.target == "NA" else col_names.index(st.session_state.target) if st.session_state.target in col_names else 0,
        key="target_select",
        help="This is the column you want to predict. Choose carefully - it should contain the values you want the model to learn."
    )
    
    st.session_state.target = target
    
    # Show target info
    target_col = next((c for c in cols if c["name"] == target), None)
    if target_col:
        st.markdown("### 📊 Target Column Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Type", target_col['dtype'], help="The data type: numeric (numbers), categorical (text), or datetime (dates)")
        with col2:
            st.metric("Unique Values", target_col['unique'], help="How many different values. Few = Classification. Many = Regression.")
        with col3:
            st.metric("Missing %", f"{target_col['null_pct']}%", help="Percentage of empty/missing data. Under 30% is good.")
        
        # Provide guidance
        if target_col['null_pct'] > 30:
            st.warning(f"⚠️ This column has {target_col['null_pct']}% missing data. Consider choosing another target or using 'mean' imputation.")
    
    st.markdown("---")
    
    st.markdown("### 📌 Business Context (Optional)")
    st.write("Why are you making this prediction? This helps explain results.")
    
    business_outcome = st.text_input(
        "Business outcome",
        value=st.session_state.business_outcome,
        placeholder="e.g., Improve customer retention, Increase sales, Reduce costs",
        key="business_outcome_input",
        help="""Examples:
        • 'Predict which customers will leave to improve retention'
        • 'Forecast monthly sales to plan inventory'
        • 'Identify credit fraud to protect customers'
        """
    )
    st.session_state.business_outcome = business_outcome
    
    if business_outcome and business_outcome != "NA":
        st.success(f"✅ Your goal: {business_outcome}")


def step_2_problem():
    """Step 2: Select Problem Type"""
    if not st.session_state.target or st.session_state.target == "NA":
        st.warning("Select target first")
        return
    
    st.header("🔍 Problem Type")
    st.write("What type of prediction problem is this?")
    st.info("💡 **What is a Problem Type?** It tells the model what kind of predictions to make.")
    
    df = st.session_state.current_df
    suggested = suggest_problem_type(df, st.session_state.target)
    
    st.success(f"✅ **Recommended**: {suggested.upper()}")
    
    st.markdown("---")
    st.markdown("### Choose Your Problem Type:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Classification**")
        st.caption("Predicting categories/classes")
        st.write("Examples:")
        st.caption("• Will customer buy? (Yes/No)")
        st.caption("• Email is spam? (Spam/Not Spam)")
        st.caption("• Disease type? (A/B/C)")
    
    with col2:
        st.markdown("**📈 Regression**")
        st.caption("Predicting numbers/quantities")
        st.write("Examples:")
        st.caption("• What will sales be next month?")
        st.caption("• How much will product cost?")
        st.caption("• What's the house price?")
    
    with col3:
        st.markdown("**📅 Time Series**")
        st.caption("Predicting over time/dates")
        st.write("Examples:")
        st.caption("• Stock price tomorrow?")
        st.caption("• Weather next week?")
        st.caption("• Website traffic trend?")
    
    st.markdown("---")
    
    problem_type = st.radio(
        "Select Problem Type",
        ["classification", "regression", "time_series"],
        index=["classification", "regression", "time_series"].index(st.session_state.problem_type) if st.session_state.problem_type in ["classification", "regression", "time_series"] else 0,
        horizontal=True,
        key="problem_type_radio"
    )
    
    st.session_state.problem_type = problem_type


def step_3_features():
    """Step 3: Feature Selection"""
    if not st.session_state.target or st.session_state.target == "NA":
        st.warning("Select target first")
        return
    
    st.header("✨ Feature Selection")
    st.write("Which columns should the model use to make predictions?")
    st.info("""💡 **What are Features?** They're the columns (inputs) the model learns from to predict your target.
    
    **Good features:**
    • Related to what you're predicting
    • Have meaningful data (not all empty)
    • Different values (not all the same)
    
    **Example:** To predict house price, use: Location, Size, Age, Bedrooms (good features)
    """)
    
    all_cols = [c["name"] for c in st.session_state.dataset_meta["columns"]]
    available_features = [c for c in all_cols if c != st.session_state.target]
    
    st.markdown("---")
    st.markdown("### Available Columns")
    
    # Show what's excluded
    st.caption(f"🎯 Target (excluded): **{st.session_state.target}**")
    st.caption(f"📊 Available for selection: {len(available_features)} columns")
    
    # Filter default features to only include those available (exclude target)
    default_features = [f for f in st.session_state.features if f in available_features] if st.session_state.features else available_features
    
    selected_features = st.multiselect(
        "Select features for the model",
        available_features,
        default=default_features,
        key="features_multiselect",
        help="Hold Ctrl (or Cmd on Mac) to select/deselect multiple columns. Start with columns that seem related to your target."
    )
    
    st.session_state.features = selected_features
    
    # Show summary
    st.markdown("---")
    st.markdown("### 📋 Selection Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target", st.session_state.target, help="What you're predicting")
    with col2:
        st.metric("Features", len(selected_features), help="Columns for the model to learn from")
    with col3:
        st.metric("Ratio", f"1:{len(selected_features)}", help="1 prediction, N features")
    
    if len(selected_features) == 0:
        st.warning("⚠️ Please select at least 1 feature")
    elif len(selected_features) > 20:
        st.info("💡 You have 20+ features. Model might be slow - consider removing less important ones")
    else:
        st.success(f"✅ Ready with {len(selected_features)} features!")


def step_4_engineering():
    """Step 4: Feature Engineering"""
    st.header("🔧 Feature Engineering")
    st.write("Configure preprocessing and feature engineering options.")
    st.info("💡 **What is Feature Engineering?** It's the process of preparing your data for the model. These options help clean, transform, and organize your data automatically.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values")
        st.write("What to do if data is missing or empty?")
        
        # Show column-specific missing value strategies
        st.markdown("**Smart Missing Value Handling:**")
        df = st.session_state.current_df
        
        missing_summary = []
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 0:
                col_type = "Numeric" if df[col].dtype in ['int64', 'float64'] else "Text/Category"
                recommended = "mean" if col_type == "Numeric" else "mode"
                missing_summary.append({
                    "column": col,
                    "missing_pct": missing_pct,
                    "type": col_type,
                    "recommended": recommended
                })
        
        if missing_summary:
            st.caption("📋 Columns with missing values - Choose strategy for each:")
            
            # Initialize per-column strategy dict if not exists
            if "missing_strategies_per_column" not in st.session_state.feature_engineering:
                st.session_state.feature_engineering["missing_strategies_per_column"] = {}
            
            # Create individual strategy selection for each column
            for item in missing_summary:
                col_name = item['column']
                col_type = item['type']
                recommended = item['recommended']
                
                # Get current strategy or use recommended
                current_strategy = st.session_state.feature_engineering["missing_strategies_per_column"].get(
                    col_name, recommended
                )
                
                col_idx = MISSING_STRATEGIES.index(current_strategy) if current_strategy in MISSING_STRATEGIES else 0
                
                selected_strategy = st.selectbox(
                    f"**{col_name}** ({item['missing_pct']:.1f}% missing) - {col_type}",
                    MISSING_STRATEGIES,
                    index=col_idx,
                    key=f"missing_col_{col_name}",
                    help=f"Recommended: '{recommended}' for {col_type} data"
                )
                
                # Save the selected strategy
                st.session_state.feature_engineering["missing_strategies_per_column"][col_name] = selected_strategy
            
            st.caption("✅ Individual strategies saved for each column")
        else:
            st.caption("✅ No missing values detected!")
        
        st.markdown("---")
        st.session_state.feature_engineering["missing_strategy"] = st.selectbox(
            "Default fallback strategy",
            MISSING_STRATEGIES,
            index=MISSING_STRATEGIES.index(st.session_state.feature_engineering["missing_strategy"]),
            key="missing_strat",
            help="Used only for columns not explicitly configured above"
        )
        
    with col2:
        st.subheader("Data Transformations")
        st.write("How to prepare categorical and date data?")
        
        st.session_state.feature_engineering["encode_categorical"] = st.checkbox(
            "✓ One-hot encode categorical features",
            value=st.session_state.feature_engineering["encode_categorical"],
            key="encode_cat",
            help="""What does this do?
Converts text categories into numbers so the model understands them.

Example:
Before: Color = ['Red', 'Blue', 'Red']
After: Color_Red = [1, 0, 1], Color_Blue = [0, 1, 0]
            """
        )
        
        st.session_state.feature_engineering["scale_features"] = st.checkbox(
            "✓ Scale numeric features",
            value=st.session_state.feature_engineering["scale_features"],
            key="scale_feat",
            help="""What does this do?
Converts all numbers to same scale (0-1 or -1 to 1) so the model treats them equally.

Why? If Age (0-100) and Income (0-1,000,000) are unscaled, Income dominates.
            """
        )
        
        st.session_state.feature_engineering["extract_datetime"] = st.checkbox(
            "✓ Extract datetime features",
            value=st.session_state.feature_engineering["extract_datetime"],
            key="extract_dt",
            help="""What does this do?
Breaks down dates into Year, Month, Day so the model can find patterns.

Example:
Date: 2024-03-15 → Year: 2024, Month: 3, Day: 15
Useful for seasonal patterns (e.g., sales spike in December)
            """
        )
    
    st.markdown("---")
    
    # AI Suggestions
    if st.session_state.ai_enabled:
        with st.expander("💡 AI Feature Engineering Suggestions"):
            if st.button("Get suggestions", key="fe_suggestions_btn"):
                with st.spinner("Thinking..."):
                    suggestions = get_feature_engineering_suggestions(
                        st.session_state.current_df,
                        st.session_state.target,
                        st.session_state.features,
                        st.session_state.ai_provider,
                        st.session_state.ai_model,
                        st.session_state.api_key
                    )
                    st.markdown(suggestions)


def step_5_model():
    """Step 5: Model Selection"""
    st.header("🤖 Model Selection")
    st.info("💡 **What is a Model?** It's an algorithm that learns patterns from your data to make predictions. Different models work better for different situations.")
    
    if st.session_state.problem_type == "NA":
        st.warning("Select problem type first")
        return
    
    # Get available models
    if st.session_state.problem_type == "regression":
        available_models = REGRESSION_MODELS
    elif st.session_state.problem_type == "classification":
        available_models = CLASSIFICATION_MODELS
    else:
        available_models = TIMESERIES_MODELS
    
    st.markdown("---")
    st.markdown("### Compare Different Models:")
    
    # Create comparison table
    comparison = {
        "Model": ["Auto", "Linear", "Random Forest", "XGBoost"],
        "Speed": ["⚡ Medium", "⚡⚡⚡ Super Fast", "⚡⚡ Fast", "⚡ Slow"],
        "Accuracy": ["✅ Good", "✅ Fair", "✅✅ Excellent", "✅✅✅ Best"],
        "Interpretable": ["✅ Fair", "✅✅✅ Very Easy", "✅✅ Medium", "⚠️ Complex"],
        "Best For": ["Don't know?", "Simple patterns", "Complex patterns", "Best accuracy"]
    }
    
    comparison_df = pd.DataFrame(comparison)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    model_name = st.selectbox(
        "Select your model",
        available_models,
        index=available_models.index(st.session_state.model_name) if st.session_state.model_name in available_models else 0,
        help="Choose 'Auto' if unsure - it will pick the best model automatically!",
        key="model_select"
    )
    st.session_state.model_name = model_name
    
    # Detailed explanations
    model_details = {
        "auto": {
            "title": "🤖 Auto (Recommended for beginners)",
            "description": "The system automatically chooses the best model for you based on your data.",
            "when_to_use": "• You're not sure which model to pick\n• You want the system to decide\n• Quick prototyping"
        },
        "linear_regression": {
            "title": "📊 Linear Regression",
            "description": "Simple, fast model that finds straight-line relationships in data.",
            "when_to_use": "• Simple prediction tasks\n• When speed is important\n• When you need easy explanations\n• House price from square footage"
        },
        "logistic_regression": {
            "title": "✅ Logistic Regression",
            "description": "Fast model for yes/no predictions. Easy to understand results.",
            "when_to_use": "• Predicting categories (Yes/No, Pass/Fail)\n• When interpretability matters\n• Medical diagnosis (Disease/Healthy)"
        },
        "random_forest": {
            "title": "🌲 Random Forest",
            "description": "Strong model that combines many decision trees. Handles complex patterns well.",
            "when_to_use": "• Complex relationships in data\n• Mix of numeric and categorical data\n• Most general-purpose tasks\n• Balanced accuracy and speed"
        },
        "xgboost": {
            "title": "🚀 XGBoost (Advanced)",
            "description": "Powerful but complex model. Wins competitions! Slower but very accurate.",
            "when_to_use": "• When accuracy is most important\n• Large datasets\n• You have time to wait\n• Professional competitions"
        }
    }
    
    selected_detail = model_details.get(model_name.lower(), model_details.get("auto"))
    
    st.markdown(f"### {selected_detail['title']}")
    st.write(selected_detail['description'])
    st.markdown("**When to use this model:**")
    st.write(selected_detail['when_to_use'])
    
    # AI Recommendation
    if st.session_state.ai_enabled:
        with st.expander("💡 AI Model Recommendation"):
            if st.button("Get recommendation", key="model_rec_btn"):
                with st.spinner("Analyzing..."):
                    rec = get_smart_model_selection(
                        st.session_state.current_df,
                        st.session_state.target,
                        st.session_state.features,
                        st.session_state.problem_type,
                        st.session_state.ai_provider,
                        st.session_state.ai_model,
                        st.session_state.api_key
                    )
                    st.markdown(rec)


def step_6_train():
    """Step 6: Train & Evaluate"""
    st.header("🎯 Train & Evaluate")
    st.write("Click below to train your model.")
    
    if not st.session_state.features:
        st.warning("Select features first")
        return
    
    # Validation
    with st.expander("🔍 Pre-training checks", expanded=True):
        df = st.session_state.current_df
        
        checks = {
            "✅ Dataset loaded": st.session_state.dataset_id is not None,
            "✅ Target selected": st.session_state.target != "NA",
            "✅ Features selected": len(st.session_state.features) > 0,
            "✅ Problem type selected": st.session_state.problem_type != "NA",
        }
        
        for check, status in checks.items():
            st.write(check if status else check.replace("✅", "❌"))
        
        # Data quality - only check USED columns (selected features + target)
        quality_report = get_data_quality_report(df)
        st.caption(f"Data: {quality_report['total_rows']} rows, {quality_report['duplicates']['percentage']:.1f}% duplicates")
        
        # Filter issues to only show for columns being used in training
        used_columns = set(st.session_state.features + [st.session_state.target])
        relevant_issues = [issue for issue in quality_report['column_issues'] if issue['column'] in used_columns]
        
        if relevant_issues:
            st.warning(f"⚠️ {len(relevant_issues)} columns being used have issues")
            with st.expander("📋 Details of problematic columns"):
                for issue in relevant_issues:
                    st.markdown(f"**Column: {issue['column']}**")
                    for problem in issue['issues']:
                        st.caption(f"  • {problem}")
        else:
            if quality_report['column_issues']:
                st.info(f"ℹ️ Some unused columns have issues, but they won't affect training")
    
    # Train button
    if st.button("🚀 Train Model", key="train_btn", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                # Show strategy summary before training
                with st.expander("📋 Missing Value Strategy Summary"):
                    st.caption("**Strategies that will be used:**")
                    
                    df = st.session_state.current_df
                    for col in st.session_state.features:
                        if col in df.columns and df[col].isnull().any():
                            strategy = st.session_state.feature_engineering.get("missing_strategies_per_column", {}).get(
                                col, st.session_state.feature_engineering["missing_strategy"]
                            )
                            missing_pct = (df[col].isnull().sum() / len(df)) * 100
                            st.caption(f"  • **{col}** ({missing_pct:.1f}% missing) → {strategy.upper()}")
                
                config = TrainConfig(
                    target=st.session_state.target,
                    features=st.session_state.features,
                    problem_type=st.session_state.problem_type,
                    model_name=st.session_state.model_name,
                    business_outcome=st.session_state.business_outcome,
                    **st.session_state.feature_engineering
                )
                
                run_data = train_model(df, config, st.session_state.dataset_id)
                st.session_state.run_id = run_data["run_id"]
                st.session_state.run_data = run_data
                
                st.success("✅ Model trained successfully!")
                st.session_state.current_step = 7
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.write(traceback.format_exc())


def step_7_results():
    """Step 7: Results & Insights"""
    st.header("📈 Results & Insights")
    
    if not st.session_state.run_id:
        st.warning("Train a model first")
        return
    
    run = st.session_state.run_data
    
    # Key metrics
    st.markdown("#### 📊 Performance Metrics")
    cols = st.columns(len(run["metrics"]))
    for i, (metric_name, metric_val) in enumerate(run["metrics"].items()):
        with cols[i]:
            # Format metric name properly (R2 -> R²)
            display_name = "R²" if metric_name.lower() == "r2" else metric_name.upper()
            st.metric(display_name, f"{metric_val:.4f}")
    
    # Feature importance
    st.markdown("#### ⭐ Top Features")
    importance_df = pd.DataFrame(run["feature_importance"][:10])
    if not importance_df.empty:
        st.bar_chart(importance_df.set_index("feature")["importance"], use_container_width=True)
    
    # AI Explanations
    st.markdown("#### 🤖 AI-Powered Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Explanation", "Business", "Data Quality", "Hyperparameters"])
    
    with tab1:
        if st.button("Generate explanation", key="explain_btn"):
            with st.spinner("Generating..."):
                explanation = get_model_explanation(
                    run,
                    st.session_state.ai_provider if st.session_state.ai_enabled else None,
                    st.session_state.ai_model if st.session_state.ai_enabled else None,
                    st.session_state.api_key if st.session_state.ai_enabled else None
                )
                st.markdown(explanation)
    
    with tab2:
        if st.button("Business recommendations", key="biz_btn"):
            with st.spinner("Analyzing..."):
                recommendations = get_business_recommendations(
                    run,
                    st.session_state.ai_provider if st.session_state.ai_enabled else None,
                    st.session_state.ai_model if st.session_state.ai_enabled else None,
                    st.session_state.api_key if st.session_state.ai_enabled else None
                )
                st.markdown(recommendations)
    
    with tab3:
        quality_report = get_data_quality_report(st.session_state.current_df)
        if st.button("Data quality insights", key="quality_btn"):
            with st.spinner("Analyzing..."):
                insights = get_data_quality_insights(
                    quality_report,
                    st.session_state.ai_provider if st.session_state.ai_enabled else None,
                    st.session_state.ai_model if st.session_state.ai_enabled else None,
                    st.session_state.api_key if st.session_state.ai_enabled else None
                )
                st.markdown(insights)
    
    with tab4:
        if st.button("Hyperparameter suggestions", key="hyper_btn"):
            with st.spinner("Generating..."):
                suggestions = get_hyperparameter_suggestions(
                    st.session_state.current_df,
                    st.session_state.problem_type,
                    st.session_state.ai_provider if st.session_state.ai_enabled else None,
                    st.session_state.ai_model if st.session_state.ai_enabled else None,
                    st.session_state.api_key if st.session_state.ai_enabled else None
                )
                st.markdown(suggestions)




def generate_model_report_text(run_data, run_id):
    """Generate comprehensive text/markdown report for model results."""
    report = []
    
    report.append("# 📊 Model Performance Report")
    report.append("")
    report.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Model ID:** {run_id[:8]}")
    report.append("")
    
    # Dataset Information
    report.append("## 1️⃣ Dataset Information")
    report.append(f"- **Target Variable:** {run_data['target']}")
    report.append(f"- **Problem Type:** {run_data['problem_type']}")
    report.append(f"- **Number of Features:** {len(run_data['features'])}")
    report.append(f"- **Train/Test Split:** 80% / 20%")
    report.append("")
    
    # Model Information
    report.append("## 2️⃣ Model Information")
    report.append(f"- **Model Type:** {run_data['model_name'].upper()}")
    if run_data.get('business_outcome') and run_data['business_outcome'] != 'NA':
        report.append(f"- **Business Outcome:** {run_data['business_outcome']}")
    report.append("")
    
    # Performance Metrics
    report.append("## 3️⃣ Performance Metrics")
    metrics = run_data.get('metrics', {})
    if metrics:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                display_name = "R²" if metric_name.lower() == "r2" else metric_name.replace('_', ' ').title()
                report.append(f"- **{display_name}:** {metric_value:.4f}")
    report.append("")
    
    # Features Used
    report.append("## 4️⃣ Features Used")
    features = run_data.get('features', [])
    if features:
        for i, feature in enumerate(features, 1):
            report.append(f"{i}. {feature}")
    report.append("")
    
    # Feature Importance
    report.append("## 5️⃣ Top Features by Importance")
    feature_importance = run_data.get('feature_importance', [])
    if feature_importance:
        for idx, item in enumerate(feature_importance[:10], 1):
            feature = item.get('feature', 'Unknown')
            importance = item.get('importance', 0)
            # Create a visual bar
            bar_length = int(importance * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            report.append(f"{idx}. {feature:25s} {bar} {importance:.4f}")
    report.append("")
    
    report.append("---")
    report.append("*Report generated by AnyDataML - Predictive Analytics Decision Assistant*")
    
    return "\n".join(report)


def generate_model_report_pdf(run_data, run_id):
    """Generate comprehensive PDF report for model results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    
    # Title
    pdf.cell(0, 15, "Model Performance Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(5)
    
    # Dataset Info
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(0, 10, "1. Dataset Information", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Target Variable: {run_data['target']}", ln=True)
    pdf.cell(0, 8, f"Problem Type: {run_data['problem_type']}", ln=True)
    pdf.cell(0, 8, f"Number of Features: {len(run_data['features'])}", ln=True)
    pdf.cell(0, 8, f"Test Set Size: 20%", ln=True)
    pdf.ln(3)
    
    # Model Info
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(0, 10, "2. Model Information", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Model Type: {run_data['model_name']}", ln=True)
    pdf.ln(3)
    
    # Metrics
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(0, 10, "3. Performance Metrics", ln=True)
    pdf.set_font("Arial", size=10)
    
    metrics = run_data.get('metrics', {})
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            # Format metric name properly (R2 -> R²)
            display_name = "R²" if metric_name.lower() == "r2" else metric_name.replace('_', ' ').title()
            pdf.cell(0, 8, f"{display_name}: {metric_value:.4f}", ln=True)
    pdf.ln(3)
    
    # Features Used
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(0, 10, "4. Features Used", ln=True)
    pdf.set_font("Arial", size=9)
    
    features = run_data.get('features', [])
    feature_text = ", ".join(features[:10])
    if len(features) > 10:
        feature_text += f" and {len(features) - 10} more"
    
    pdf.multi_cell(0, 6, feature_text)
    pdf.ln(3)
    
    # Feature Importance
    feature_importance = run_data.get('feature_importance', [])
    if feature_importance:
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(0, 10, "5. Top Features by Importance", ln=True)
        pdf.set_font("Arial", size=10)
        
        # feature_importance is a list of dicts with 'feature' and 'importance' keys
        for idx, item in enumerate(feature_importance[:10], 1):
            feature = item.get('feature', 'Unknown')
            importance = item.get('importance', 0)
            pdf.cell(0, 8, f"{idx}. {feature}: {importance:.4f}", ln=True)
    
    pdf.ln(3)
    
    # Business Outcome
    if run_data.get('business_outcome') and run_data['business_outcome'] != 'NA':
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(0, 10, "6. Business Outcome", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, run_data['business_outcome'])
        pdf.ln(3)
    
    # Footer
    pdf.set_font("Arial", size=8, style='I')
    pdf.cell(0, 10, f"Model ID: {run_id[:8]}", ln=True)
    
    # Return PDF as bytes - properly formatted
    try:
        pdf_output = pdf.output()
        if isinstance(pdf_output, bytes):
            return pdf_output
        else:
            return pdf_output.encode('latin-1') if isinstance(pdf_output, str) else bytes(pdf_output)
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        raise


def step_8_export():
    """Step 8: Export & Download"""
    st.header("💾 Export Results")
    
    if not st.session_state.run_id:
        st.warning("Train a model first")
        return
    
    run = st.session_state.run_data
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            pred_path = get_predictions_path(st.session_state.run_id)
            with open(pred_path, "rb") as f:
                st.download_button(
                    label="📥 Download Predictions",
                    data=f.read(),
                    file_name=f"predictions_{st.session_state.run_id[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error: {e}")
    
    with col2:
        try:
            model_path = get_model_path(st.session_state.run_id)
            with open(model_path, "rb") as f:
                st.download_button(
                    label="🤖 Download Model",
                    data=f.read(),
                    file_name=f"model_{st.session_state.run_id[:8]}.joblib",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error: {e}")
    
    with col3:
        # Export as PDF Report
        try:
            pdf_bytes = generate_model_report_pdf(run, st.session_state.run_id)
            st.download_button(
                label="📊 Download PDF Report",
                data=pdf_bytes,
                file_name=f"report_{st.session_state.run_id[:8]}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            logger.error(f"PDF error: {e}")
    
    with col4:
        # Export as JSON
        run_json = json.dumps(run, indent=2, default=str)
        st.download_button(
            label="📄 Download JSON",
            data=run_json,
            file_name=f"report_{st.session_state.run_id[:8]}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Display full report preview
    st.markdown("---")
    st.markdown("#### 📄 Report Preview")
    report_preview = generate_model_report_text(run, st.session_state.run_id)
    st.markdown(report_preview)


# ========== Prediction with Existing Model Workflow ==========
def predict_with_existing_model():
    """Workflow for making predictions with existing models."""
    st.markdown("## 🔮 Predict with Existing Model")
    
    # Model selection section
    st.markdown("### Step 1️⃣: Select a Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_source = st.radio("Choose model source:", ["📁 System Models", "📤 Upload Model"], key="model_source")
    
    model_to_use = None
    model_features = None
    model_target = None
    model_info = {}
    
    if model_source == "📁 System Models":
        # List system models
        system_models = list(MODELS_DIR.glob("*.joblib"))
        
        if not system_models:
            st.warning("❌ No trained models found in the system yet.\n\nTrain a model first using the 'Train New Model' mode!")
            return
        
        # Get model metadata from runs database
        runs = list_runs()
        
        if not runs:
            st.warning("❌ No training runs found.")
            return
        
        # Create readable options
        model_options = {}
        for run in runs:
            label = f"📊 {run['model_name']} ({run['problem_type']}) - {run['created_at'][:10]}"
            model_options[label] = run['run_id']
        
        selected_model_label = st.selectbox("Select a trained model:", list(model_options.keys()))
        selected_run_id = model_options[selected_model_label]
        
        # Get full run details including features
        from utils.storage import get_run
        selected_run = get_run(selected_run_id)
        
        if not selected_run:
            st.error("❌ Error loading model details")
            return
        
        model_to_use = selected_run_id
        model_features = selected_run['features']
        model_target = selected_run['target']
        model_info = {
            'name': selected_run['model_name'],
            'problem_type': selected_run['problem_type'],
            'created_at': selected_run['created_at'],
            'metrics': selected_run['metrics'],
            'business_outcome': selected_run.get('business_outcome', 'N/A')
        }
        
        # Show model info
        with st.expander("📋 Model Details"):
            st.caption(f"**Algorithm:** {model_info['name']}")
            st.caption(f"**Problem Type:** {model_info['problem_type']}")
            st.caption(f"**Trained on:** {model_info['created_at'][:10]}")
            st.caption(f"**Features used:** {', '.join(model_features)}")
            st.caption(f"**Target:** {model_target}")
    
    else:
        # Upload external model
        uploaded_file = st.file_uploader("Upload .joblib model file:", type=['joblib'])
        
        if uploaded_file is not None:
            try:
                # Load the model
                model = joblib.load(uploaded_file)
                st.success("✅ Model loaded successfully!")
                
                # Try to extract feature names from model
                extracted_features = None
                if hasattr(model, 'feature_names_in_'):
                    extracted_features = list(model.feature_names_in_)
                elif hasattr(model, 'n_features_in_'):
                    n_features = model.n_features_in_
                    st.info(f"ℹ️ Model expects {n_features} features (names not available)")
                
                # Store in session
                st.session_state.uploaded_model = model
                st.session_state.uploaded_model_features = extracted_features
                model_to_use = "uploaded_model"
                
                if extracted_features:
                    st.caption(f"**Features expected:** {', '.join(extracted_features)}")
                
                st.warning("⚠️ **Note:** Since this is an external model, ensure your data columns match the training data. Extra columns will be automatically excluded.")
                
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                return
    
    # Data upload section
    st.markdown("### Step 2️⃣: Upload New Data")
    
    uploaded_data = st.file_uploader("Upload CSV or Excel file with new data:", type=['csv', 'xlsx', 'xls'], key="predict_data")
    
    if uploaded_data is None:
        st.info("👆 Upload a CSV or Excel file to proceed")
        return
    
    # Load and preview data
    try:
        if uploaded_data.name.endswith('.csv'):
            df_new = pd.read_csv(uploaded_data)
        elif uploaded_data.name.endswith(('.xlsx', '.xls')):
            # For Excel files, read the first sheet
            xls = pd.ExcelFile(uploaded_data)
            sheet_name = xls.sheet_names[0]  # Get first sheet
            
            # If multiple sheets, show selector
            if len(xls.sheet_names) > 1:
                st.info(f"ℹ️ Excel has {len(xls.sheet_names)} sheets. Using first sheet: **{sheet_name}**")
            
            df_new = pd.read_excel(uploaded_data, sheet_name=sheet_name)
        else:
            st.error("❌ Unsupported file format")
            return
        st.success(f"✅ Data loaded: {df_new.shape[0]} rows × {df_new.shape[1]} columns")
        
        with st.expander("📊 Data Preview"):
            st.dataframe(df_new.head(10), use_container_width=True)
            st.caption(f"Column summary: {', '.join(df_new.columns)}")
    
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return
    
    # Validation
    # Get features from model (system or uploaded)
    if model_to_use == "uploaded_model" and "uploaded_model_features" in st.session_state:
        model_features = st.session_state.uploaded_model_features
    
    if model_features:
        st.markdown("### Step 3️⃣: Validate Data")
        
        # Check if required features exist
        missing_features = [f for f in model_features if f not in df_new.columns]
        extra_features = [f for f in df_new.columns if f not in model_features]
        
        if missing_features:
            st.error(f"❌ **Missing features:** {', '.join(missing_features)}\n\nYour data must have these columns to use this model.")
            return
        
        if extra_features:
            st.info(f"ℹ️ **Extra columns will be excluded from prediction:** {', '.join(extra_features)}")
        
        # Prepare data with ONLY required features (most important: must match model training)
        st.caption("Selecting only the features the model was trained on...")
        df_new = df_new[model_features]
        st.success(f"✅ Data filtered to {len(df_new.columns)} model features")
        
        # Check for missing values
        missing_pct = df_new.isnull().sum()
        if missing_pct.any():
            st.warning("⚠️ **Missing values detected:**")
            for col, pct in missing_pct[missing_pct > 0].items():
                pct_val = (pct / len(df_new)) * 100
                st.caption(f"  • {col}: {pct_val:.1f}% missing")
        
        st.success("✅ Data validation passed!")
    else:
        st.info("ℹ️ Validation skipped (feature names not available from external model)")
    
    # Make predictions
    st.markdown("### Step 4️⃣: Generate Predictions")
    
    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
        with st.spinner("Making predictions..."):
            try:
                if model_to_use == "uploaded_model":
                    loaded_model = st.session_state.uploaded_model
                    # Get features for validation
                    if "uploaded_model_features" in st.session_state and st.session_state.uploaded_model_features:
                        final_features = st.session_state.uploaded_model_features
                        df_pred = df_new[final_features]  # Final safety: use only expected features
                    else:
                        df_pred = df_new
                else:
                    # Load model - returns dict with model + metadata
                    loaded_model = load_model(model_to_use)
                    df_pred = df_new
                
                # Extract actual model object (handle both dict and raw model)
                if isinstance(loaded_model, dict) and "model" in loaded_model:
                    model = loaded_model["model"]
                else:
                    model = loaded_model
                
                # Make predictions
                predictions = model.predict(df_pred)
                
                # Get prediction probabilities if available
                has_proba = hasattr(model, 'predict_proba')
                proba = None
                if has_proba:
                    try:
                        proba = model.predict_proba(df_pred)
                    except Exception:
                        pass
                
                # Create results dataframe (use original df_new for reference)
                results_df = df_new.copy()
                results_df['predicted_value'] = predictions
                
                # Add confidence scores if available
                if proba is not None and hasattr(model, 'classes_'):
                    try:
                        for i, class_label in enumerate(model.classes_):
                            results_df[f'confidence_{class_label}'] = proba[:, i]
                    except Exception:
                        pass
                
                st.session_state.prediction_results = results_df
                st.session_state.model_info_display = model_info
                
                st.success(f"✅ Predictions generated for {len(predictions)} rows!")
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {e}")
                logger.error(f"Prediction error: {e}")
                return
    
    # Display results
    if "prediction_results" in st.session_state:
        results_df = st.session_state.prediction_results
        
        st.markdown("### Step 5️⃣: Results")
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(results_df))
        with col2:
            unique_preds = results_df['predicted_value'].nunique()
            st.metric("Unique Classes", unique_preds)
        with col3:
            if 'confidence_' in ' '.join(results_df.columns):
                avg_conf = results_df[[c for c in results_df.columns if c.startswith('confidence_')]].max(axis=1).mean()
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
        
        # Show prediction distribution
        pred_counts = results_df['predicted_value'].value_counts()
        st.bar_chart(pred_counts)
        
        # Display results table
        st.markdown("#### Predictions Table")
        with st.expander("📋 View Full Results", expanded=True):
            st.dataframe(results_df, use_container_width=True)
        
        # Download options
        st.markdown("#### 💾 Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = results_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Export as Excel
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Predictions', index=False)
            excel_buffer.seek(0)
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# ========== Main Layout ==========
def main():
    """Main app layout."""
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>📊 Predictive Analytics Decision Assistant</h1>
        <p><em>From spreadsheet to prediction in 9 steps</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode selector
    mode = st.radio(
        "Choose what you want to do:",
        ["🚀 Train New Model", "🔮 Predict with Existing Model"],
        horizontal=True,
        help="Train: Build a new model from scratch | Predict: Use an existing model on new data"
    )
    
    st.markdown("---")
    
    if mode == "🔮 Predict with Existing Model":
        predict_with_existing_model()
        return
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    # Steps configuration
    step_components = [
        ("📤 Upload Data", step_0_upload),
        ("🎯 Select Target", step_1_target),
        ("🔍 Problem Type", step_2_problem),
        ("✨ Features", step_3_features),
        ("🔧 Engineering", step_4_engineering),
        ("🤖 Model", step_5_model),
        ("🚀 Train", step_6_train),
        ("📈 Results", step_7_results),
        ("💾 Export", step_8_export),
    ]
    
    # Get current step (ensure it's valid)
    current_step = min(st.session_state.current_step, len(step_components) - 1)
    
    # Display step navigation header
    st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 5px; margin-bottom: 1rem; flex-wrap: wrap;">
        {''.join([
            f"<span style='background-color: {'#00D9FF' if i == current_step else '#444'}; color: white; padding: 5px 10px; border-radius: 5px; font-size: 12px;'>Step {i}</span>"
            for i in range(len(step_components))
        ])}
    </div>
    """, unsafe_allow_html=True)
    
    # Display current step
    label, component_func = step_components[current_step]
    
    st.markdown(f"## {label} (Step {current_step}/{len(step_components)-1})")
    
    # Progress bar
    progress = current_step / (len(step_components) - 1)
    st.progress(progress)
    
    # Component
    try:
        component_func()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write(traceback.format_exc())
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if current_step > 0 and st.button("⬅️ Back", key=f"back_{current_step}"):
            st.session_state.current_step = current_step - 1
            st.rerun()
    
    with col2:
        st.caption(f"Progress: {current_step + 1}/{len(step_components)}")
    
    with col3:
        if current_step < len(step_components) - 1 and st.button("➡️ Next", key=f"next_{current_step}"):
            st.session_state.current_step = current_step + 1
            st.rerun()


if __name__ == "__main__":
    main()
