# Changelog

All notable changes to the Predictive Analytics Decision Assistant will be documented in this file.

## [1.1.0] - 2026-04-25

### Added
- 🔮 **Predict with Existing Model** - New workflow to make predictions using previously trained models
- 📤 **External Model Upload** - Support for uploading `.joblib` models from other sources
- 📊 **Excel Support** - Added `.xlsx` and `.xls` file support for data upload and predictions
- 🎯 **Per-Column Missing Value Strategies** - Individual strategy selection for each column with missing values
- 📥 **Multiple Export Formats** - CSV, JSON, and Excel download options for predictions

### Fixed
- Binary handling in PDF export
- Column validation to show only relevant issues
- Auto-advance in Step 0 data upload
- Feature name detection for model compatibility

### Enhanced
- Beginner-friendly UI with step-by-step guidance
- Data validation with automatic filtering of extra columns
- Model feature compatibility checking
- Confidence score display for classifiers

## [1.0.0] - 2026-04-15

### Initial Release
- 📊 9-step ML workflow (Upload → Export)
- Support for Classification, Regression, Time Series
- Multiple model algorithms (Random Forest, XGBoost, Logistic Regression, etc.)
- Feature engineering with missing value strategies
- AI-powered suggestions (OpenAI, Groq, Anthropic)
- PDF and JSON report generation
- Selective storage cleanup
