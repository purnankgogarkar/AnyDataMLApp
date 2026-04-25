# Contributing to Predictive Analytics Decision Assistant

Thank you for your interest in contributing! Here's how you can help improve this project.

## **Getting Started**

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/AnyDataMLApp.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make changes** and test thoroughly
5. **Commit** with clear messages
6. **Push** to your fork
7. **Create Pull Request** on GitHub

## **Development Setup**

```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## **Code Style**

- Follow PEP 8 Python guidelines
- Use clear variable names
- Add comments for complex logic
- Use type hints where applicable

## **Testing**

Before submitting PR:
- Test with different data types (CSV, Excel, numeric, categorical)
- Test with edge cases (empty files, missing values, large datasets)
- Verify no errors in logs

## **Commit Messages**

Use clear, descriptive messages:
```
✅ Good:
git commit -m "Add feature: ONNX model export support"
git commit -m "Fix: Feature validation for external models"
git commit -m "Improve: Faster data loading for large files"

❌ Bad:
git commit -m "update"
git commit -m "fix bugs"
```

## **PR Guidelines**

- Describe what you changed and why
- Link any related issues
- Include screenshots for UI changes
- Ensure all tests pass

## **Feature Ideas**

Areas for enhancement:
- [ ] ONNX model format support
- [ ] Cloud deployment (AWS, Azure, GCP)
- [ ] Model comparison dashboard
- [ ] Batch prediction API
- [ ] Advanced hyperparameter tuning
- [ ] Model versioning system
- [ ] Integration with popular platforms

## **Questions?**

Open an issue on GitHub or check existing discussions!
