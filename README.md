# 📊 Predictive Analytics Decision Assistant

A Streamlit-based application that guides users through a 9-step ML workflow to build, train, and evaluate predictive models with optional AI-powered insights.

## 🎯 Overview

Transform raw data into actionable predictions through an intuitive step-by-step wizard. The app handles data loading, exploratory analysis, feature engineering, model training, and evaluation—all without requiring ML expertise.

### Key Features

- **9-Step Guided Workflow**: Walk-through from data upload to model export
- **AI-Powered Insights**: Integrate with OpenAI, Groq, or Anthropic for smart recommendations
- **Multiple Problem Types**: Classification, Regression, and Time Series prediction
- **Flexible Model Selection**: Auto-selection or manual choice (Linear, Random Forest, XGBoost)
- **Export Options**: Download predictions, models, PDF reports, and JSON exports
- **Run History**: Manage and reload previous model training runs
- **Data Quality Analysis**: Automatic detection of issues and recommendations

## 🏗️ Project Structure

```
AnyDataMLApp/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── storage/              # Data and model storage
│   ├── datasets/         # Uploaded dataset pickles
│   ├── models/           # Trained model joblib files
│   ├── predictions/      # Model predictions (CSV)
│   └── app.db           # SQLite database for metadata
├── utils/               # Core utility modules
│   ├── __init__.py      # Package exports
│   ├── config.py        # Configuration and constants
│   ├── storage.py       # Database operations
│   ├── data_handler.py  # Data loading and preprocessing
│   ├── analysis.py      # Data quality analysis
│   ├── ml_pipeline.py   # Model training orchestration
│   └── ai_handler.py    # AI API integrations (OpenAI, Groq, Anthropic)
└── .streamlit/
    ├── config.toml      # Streamlit configuration
    └── secrets.toml.example  # Example secrets template
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Activate:
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Configure AI API Keys**
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Add your API keys for AI features:
     ```toml
     [openai]
     api_key = "sk-..."
     
     [groq]
     api_key = "gsk_..."
     
     [anthropic]
     api_key = "sk-ant-..."
     ```
   - Never commit secrets.toml to version control

### Running the Application

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 (usually automatically)

## 📋 9-Step Workflow Explained

| Step | Component | Purpose |
|------|-----------|---------|
| 0 | **Upload Data** | Load CSV/Excel files, analyze structure |
| 1 | **Select Target** | Choose prediction target column |
| 2 | **Problem Type** | Classification, Regression, or Time Series |
| 3 | **Feature Selection** | Choose input features (auto-excludes target) |
| 4 | **Feature Engineering** | Configure preprocessing and transformations |
| 5 | **Model Selection** | Choose algorithm or use auto-selection |
| 6 | **Train & Evaluate** | Execute training pipeline with validation |
| 7 | **Results & Insights** | Analyze metrics, importance, and AI insights |
| 8 | **Export & Download** | Get predictions, models, and reports |

### Detailed Steps

#### Step 0: Upload Data
- Supports CSV and Excel formats
- Analyzes column types (numeric, categorical, datetime)
- Generates statistics and preview
- Stores metadata in SQLite database

#### Step 1: Target Selection
- Select column to predict
- Optional: Specify business outcome for context

#### Step 2: Problem Type
- **Classification**: Predicting categories/classes
- **Regression**: Predicting continuous values
- **Time Series**: Temporal prediction (auto-suggested for datetime targets)

#### Step 3: Feature Selection
- Multi-select features to use
- Target column automatically excluded
- Can select all, subset, or manually choose

#### Step 4: Feature Engineering
```
Options:
├── Missing Value Strategy: drop | mean | median | mode
├── One-Hot Encode Categoricals: Yes/No
├── Scale Numeric Features: Yes/No
└── Extract DateTime Features: Yes/No
```

#### Step 5: Model Selection
- **Auto**: Chooses based on problem type and data size
- **Manual**: Linear, Random Forest, or XGBoost
- Tailored recommendations based on data characteristics

#### Step 6: Train & Evaluate
- Pre-training validation checks
- Data quality report
- Model training with 80/20 train-test split
- Real-time progress updates

#### Step 7: Results & Insights
- **Metrics**: Accuracy, F1, MAE, RMSE, R² (depending on problem type)
- **Feature Importance**: Top contributing features
- **AI Insights** (if API key configured):
  - Model explanation
  - Business recommendations
  - Data quality insights
  - Hyperparameter suggestions

#### Step 8: Export & Download
- Predictions CSV (actual vs predicted)
- Serialized model (.joblib)
- PDF report with summary
- JSON export for integration

## 🤖 AI Features (Optional)

Enhance analysis with AI-powered recommendations. Requires API key from one of:

### OpenAI
- **Models**: GPT-4, GPT-4 Turbo, GPT-3.5-Turbo
- **Key Format**: `sk-...`
- **Setup**: Get key from https://platform.openai.com/api-keys

### Groq
- **Models**: Mixtral 8x7B, Llama 70B, Gemma 7B
- **Key Format**: `gsk_...`
- **Setup**: Register at https://console.groq.com

### Anthropic
- **Models**: Claude Opus, Sonnet, Haiku
- **Key Format**: `sk-ant-...`
- **Setup**: Get key from https://console.anthropic.com

### AI Capabilities

1. **Model Explanation** - Plain English summary of model behavior
2. **Feature Suggestions** - Ideas for feature engineering
3. **Model Recommendations** - Smart algorithm selection
4. **Business Insights** - Actionable recommendations
5. **Data Quality Report** - Issues and solutions
6. **Hyperparameter Tuning** - Optimal settings

## 📊 Model Support

### Classification
- Logistic Regression (interpretable, fast)
- Random Forest (robust, handles non-linearity)
- XGBoost (high performance, slower training)

### Regression
- Linear Regression (interpretable, simple)
- Random Forest (non-linear, feature interactions)
- XGBoost (state-of-the-art performance)

### Time Series
- Random Forest (baseline implementation)
- Supports datetime feature extraction

## 📦 Dependencies

### Core Framework
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
```

### Machine Learning
```
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
```

### AI Integration
```
openai>=1.0.0
groq>=0.4.0
anthropic>=0.7.0
```

### Data Handling
```
openpyxl>=3.10.0
python-dotenv>=1.0.0
```

### Export
```
fpdf2>=2.7.0
matplotlib>=3.7.0
```

See `requirements.txt` for exact versions.

## 🔧 Configuration

### `utils/config.py` - Key Settings

```python
# Model availability
REGRESSION_MODELS = ["auto", "linear_regression", "random_forest", "xgboost"]
CLASSIFICATION_MODELS = ["auto", "logistic_regression", "random_forest", "xgboost"]

# Preprocessing options
MISSING_STRATEGIES = ["drop", "mean", "median", "mode"]

# Validation thresholds
MIN_ROWS_FOR_TRAINING = 10          # Minimum samples needed
HIGH_CARDINALITY_THRESHOLD = 50     # Flag high-cardinality categoricals
HIGH_NULL_PCT_THRESHOLD = 30        # Flag columns with >30% missing
```

## 📋 Data Requirements

### Minimum
- **10 rows** after preprocessing
- **1+ feature columns** and 1 target column
- CSV or Excel format

### Recommended
- **100+ rows** for reliable models
- **Mix of features** (numeric + categorical)
- **<30% missing** per column for best results
- **Balanced classes** for classification (avoid 1:100 ratio)

## 🗄️ Database Schema

SQLite (`storage/app.db`) contains:

### `datasets` Table
| Column | Type | Purpose |
|--------|------|---------|
| id | TEXT PK | Unique dataset ID |
| filename | TEXT | Original file name |
| rows, cols | INT | Data dimensions |
| columns | JSON | Column metadata & stats |
| preview | JSON | First 10 rows |
| created_at | TIMESTAMP | Upload time |

### `runs` Table
| Column | Type | Purpose |
|--------|------|---------|
| id | TEXT PK | Model run ID |
| dataset_id | TEXT FK | Associated dataset |
| target | TEXT | Prediction target |
| features | JSON | Selected features list |
| problem_type | TEXT | classification/regression/time_series |
| model_name | TEXT | Model used |
| metrics | JSON | Performance metrics |
| feature_importance | JSON | Feature rankings |
| created_at | TIMESTAMP | Training time |

## 🔐 Security

### Best Practices
- ✅ Store API keys in `.streamlit/secrets.toml`
- ✅ Add `secrets.toml` to `.gitignore`
- ✅ Use environment variables for deployment
- ✅ Sanitize file uploads in production
- ⚠️ Models can execute code—only load trusted models

### Data Privacy
- Local storage (no cloud upload by default)
- Comply with GDPR/CCPA when using external APIs
- Consider data residency for sensitive data

## 🐛 Troubleshooting

### "Not enough rows after preprocessing"
- More rows dropped than expected due to missing values
- **Fix**: Use "mean" or "median" strategy instead of "drop"

### "Feature X not found"
- Feature was removed during preprocessing
- **Fix**: Check data quality report for high-missing or all-null columns

### API Key Invalid
- Format mismatch or expired key
- **Fix**: Verify key format: OpenAI `sk-`, Groq `gsk_`, Anthropic `sk-ant-`

### Model Training Slow
- XGBoost on large datasets is computationally intensive
- **Fix**: Use Random Forest or reduce data size

### Export Button Not Working
- Model or predictions file missing from storage
- **Fix**: Retrain model or check filesystem permissions

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

| Scenario | Recommendation |
|----------|-----------------|
| Small data (<1K rows) | Random Forest (fast) |
| Large data (>10K rows) | XGBoost (accuracy) |
| Many missing values | Use "mean" strategy |
| High cardinality | Consider dimensionality reduction |
| Imbalanced classes | Check F1 score, not just accuracy |

## 🤝 Extending & Contributing

### Add New Model Type
1. Update `config.py` model list
2. Modify `pick_model()` in `ml_pipeline.py`
3. Test with sample data

### Add Export Format
1. Implement in `step_8_export()` in `app.py`
2. Use `st.download_button()` for download

### Add AI Feature
1. Create function in `ai_handler.py`
2. Call unified `_call_ai_api()` handler
3. Wire into appropriate step

## 🆘 Common Errors & Fixes

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: streamlit` | Dependencies not installed | `pip install -r requirements.txt` |
| `Connection refused` | Port 8501 in use | `streamlit run app.py --server.port 8502` |
| `PermissionError: storage/` | No write access | Check directory permissions |
| `KeyError: 'target'` | Target not in DataFrame | Verify column name selected |

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [OpenAI API Docs](https://platform.openai.com/docs/)

## 📝 License

[Specify your license - MIT, Apache 2.0, etc.]

## 👥 Acknowledgments

Built with [Streamlit](https://streamlit.io/), [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/), and [XGBoost](https://xgboost.readthedocs.io/).

---

**Version**: 1.0.0  
**Last Updated**: April 2026  
**Status**: ✅ Production Ready  

source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **(Optional) Add AI API Key**
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your API key
```

Supported AI Providers:
- **OpenAI**: `sk-...` prefix → https://platform.openai.com/
- **Groq**: `gsk_...` prefix → https://console.groq.com/ (free tier)
- **Anthropic**: `sk-ant-...` prefix → https://console.anthropic.com/

## Usage

### Local Development

```bash
streamlit run streamlit_app.py
```

Visit: http://localhost:8501

### Workflow

1. **Upload Data** - CSV or Excel file
2. **Select Target** - What to predict
3. **Problem Type** - Classification / Regression / Time Series
4. **Feature Selection** - Which columns to use
5. **Feature Engineering** - Preprocessing options
6. **Model Selection** - Choose or auto-select
7. **Train** - Train the model
8. **Results** - View metrics, features, AI insights
9. **Export** - Download predictions, model, report

### AI Features

Add API key in sidebar (optional):
- Detects provider automatically
- Generates 6 types of insights
- Falls back to templates if unavailable

## Project Structure

```
streamlit_app/
├── streamlit_app.py           # Main app (9-step UI)
├── requirements.txt           # Dependencies
├── .streamlit/
│   ├── config.toml           # Streamlit config
│   └── secrets.toml.example  # Example secrets
├── utils/
│   ├── config.py             # Settings & constants
│   ├── storage.py            # SQLite operations
│   ├── data_handler.py       # Data loading/parsing
│   ├── analysis.py           # Data analysis
│   ├── ml_pipeline.py        # Model training/eval
│   └── ai_handler.py         # AI insights
└── storage/
    ├── app.db                # SQLite database
    ├── datasets/             # Uploaded data
    ├── models/               # Trained models
    └── predictions/          # Model outputs
```

## Deployment on Streamlit Cloud

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push
```

2. **Deploy on Streamlit Cloud**
- Visit: https://share.streamlit.io
- Connect GitHub repo
- Select `streamlit_app/streamlit_app.py` as entry point

3. **(Optional) Add Secrets**
- In Streamlit Cloud dashboard → Manage app → Secrets
- Paste contents of `.streamlit/secrets.toml`

## API Costs

| Provider | Cost | Free Tier | Notes |
|----------|------|-----------|-------|
| OpenAI | $0.0005/req (gpt-4o-mini) | No | Most capable |
| Groq | Free | 5K tokens/day | Fast inference |
| Anthropic | Pay as you go | No | High quality |

## Development

### Add New AI Feature

Edit `utils/ai_handler.py`:

```python
def get_new_insight(data, provider, model, api_key):
    # Implement for each provider (openai, groq, anthropic)
    # Add fallback template function
    pass
```

### Modify Model Options

Edit `utils/config.py`:

```python
REGRESSION_MODELS = ["auto", "linear_regression", "random_forest", "xgboost"]
```

Then update `utils/ml_pipeline.py` `pick_model()` function.

## Troubleshooting

### "Dataset not found"
- Re-upload data

### AI features not working
- Check API key format (must start with correct prefix)
- Verify API key has credits/quota
- Check internet connection

### Model training fails
- Ensure 10+ rows after preprocessing
- Check for missing values in target
- Try different missing value strategy

## License

MIT

## Support

Issues? Check:
1. Console output for error messages
2. `.streamlit/config.toml` for settings
3. API key format for AI features
