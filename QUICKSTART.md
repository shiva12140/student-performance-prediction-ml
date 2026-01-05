# Quick Start Guide

Follow these steps to get the project running quickly:

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or if using Python 3 specifically:

```bash
python3 -m pip install -r requirements.txt
```

## 2. Generate Dataset

```bash
python3 src/create_dataset.py
```

This creates `data/student_data.csv` with 500 student records.

## 3. Train Models

```bash
python3 src/train_models.py
```

This will:
- Load and preprocess the data
- Train 3 models (Logistic Regression, Random Forest, Gradient Boosting)
- Compare their performance
- Save the best model

Expected output: Model comparison table and best model selection.

## 4. (Optional) Hyperparameter Tuning

```bash
python3 src/hyperparameter_tuning.py
```

This optimizes the best model's hyperparameters.

## 5. Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Troubleshooting

**If you get "ModuleNotFoundError":**
- Make sure you're in the project root directory
- Verify all dependencies are installed: `pip list`

**If the app says "Model not found":**
- Run step 3 (train_models.py) first to generate the model files

**If dataset is missing:**
- Run step 2 (create_dataset.py) first

## Expected File Structure After Setup

```
student-performance-prediction-ml/
├── data/
│   └── student_data.csv          # Generated dataset
├── models/
│   ├── best_model.pkl           # Trained model
│   ├── preprocessor.pkl         # Data preprocessor
│   ├── feature_names.pkl        # Feature names
│   ├── numerical_cols.pkl       # Numerical columns
│   └── categorical_cols.pkl     # Categorical columns
└── ... (other files)
```

Happy predicting! 🎓

