# Student Performance Prediction - Machine Learning Project

A complete end-to-end machine learning project that predicts student academic performance based on various academic, lifestyle, and demographic factors.

## 📋 Problem Statement

Predicting student academic performance is crucial for educational institutions to identify at-risk students early and provide targeted interventions. This project builds a classification model that predicts student performance categories (Excellent, Good, Average, Below Average, Poor) based on features such as study hours, attendance, previous grades, sleep patterns, and demographic information.

## 🎯 Project Goals

- Build a robust ML pipeline for student performance prediction
- Compare multiple classification algorithms
- Optimize model performance through hyperparameter tuning
- Deploy an interactive web application for predictions
- Follow best practices in ML engineering

## 📊 ML Workflow

```
┌─────────────────┐
│  Data Loading   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EDA & Analysis  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Eng.    │
│ - study_efficiency│
│ - academic_balance│
│ - performance_momentum│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ - Missing values│
│ - Encoding      │
│ - Scaling       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│ - Logistic Reg. │
│ - Random Forest │
│ - Gradient Boost│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hyperparameter  │
│    Tuning       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Selection │
│  & Evaluation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Saving    │
│ (joblib)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Streamlit App  │
│  Deployment     │
└─────────────────┘
```

## 🛠️ Tech Stack

- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Streamlit
- **Model Persistence**: joblib

## 📁 Project Structure

```
student-performance-prediction-ml/
│
├── data/                      # Dataset storage
│   └── student_data.csv       # Student performance dataset
│
├── notebooks/                   # Jupyter notebooks
│   └── 01_eda.ipynb          # Exploratory Data Analysis
│
├── src/                       # Source code
│   ├── data_loader.py        # Data loading utilities
│   ├── create_dataset.py     # Dataset generation script
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── feature_engineering.py # Feature creation
│   ├── train_models.py        # Model training script
│   └── hyperparameter_tuning.py # Hyperparameter optimization
│
├── models/                    # Saved models (generated)
│   ├── best_model.pkl        # Best trained model
│   ├── best_model_tuned.pkl  # Tuned model
│   ├── preprocessor.pkl      # Fitted preprocessor
│   ├── feature_names.pkl     # Feature names
│   ├── numerical_cols.pkl    # Numerical column names
│   └── categorical_cols.pkl  # Categorical column names
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- macOS (tested on macOS with 8GB RAM)
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd student-performance-prediction-ml
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Step 1: Generate Dataset

First, create the synthetic student performance dataset:

```bash
python src/create_dataset.py
```

This will generate `data/student_data.csv` with 500 student records.

#### Step 2: Exploratory Data Analysis (Optional)

Open the Jupyter notebook to explore the data:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

#### Step 3: Train Models

Train multiple models and compare their performance:

```bash
python src/train_models.py
```

This script will:
- Load and preprocess the data
- Create derived features
- Train Logistic Regression, Random Forest, and Gradient Boosting models
- Compare models using cross-validation
- Save the best model

#### Step 4: Hyperparameter Tuning (Optional)

Optimize the best model's hyperparameters:

```bash
python src/hyperparameter_tuning.py
```

This will:
- Load the previously trained model
- Perform GridSearchCV to find optimal hyperparameters
- Show before/after performance comparison
- Save the tuned model

#### Step 5: Run Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your default web browser (typically at `http://localhost:8501`).

## 📱 Using the Streamlit App

1. **Input Student Information:**
   - Use the sidebar to enter student details:
     - Academic metrics (age, study hours, attendance, previous grade, sleep, extracurricular hours)
     - Personal information (gender, parent education, internet access, study method, transportation)

2. **Make Prediction:**
   - Click the "🔮 Predict Performance" button
   - View the predicted performance category
   - See prediction probabilities for all categories
   - Review feature importance and insights

3. **Interpret Results:**
   - The app displays the predicted category (Excellent, Good, Average, Below Average, or Poor)
   - Shows probability distribution across all categories
   - Provides insights based on input values
   - Displays top contributing factors (for tree-based models)

## 📈 Model Performance

The project trains and compares three models:

1. **Logistic Regression**: Linear baseline model
2. **Random Forest**: Ensemble method with good interpretability
3. **Gradient Boosting**: Advanced ensemble with high performance

**Evaluation Metrics:**
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- 5-Fold Cross-Validation

The best model is selected based on F1-score and saved for deployment.

## 🔧 Feature Engineering

The project creates three derived features:

1. **study_efficiency**: Ratio of study hours to attendance
   - Identifies students who achieve results efficiently
   - Captures quality vs. quantity of studying

2. **academic_balance**: Balance between academic commitment and life factors
   - Combines study hours, attendance, sleep, and extracurriculars
   - Helps identify optimal study patterns and burnout risk

3. **performance_momentum**: Combination of previous grades and attendance
   - Leverages historical performance data
   - Identifies students with positive academic trajectory

## 📊 Dataset Information

The synthetic dataset includes:

**Numerical Features:**
- `age`: Student age (15-22)
- `study_hours`: Weekly study hours (0-50)
- `attendance`: Attendance percentage (0-100)
- `previous_grade`: Previous academic grade (0-100)
- `hours_sleep`: Average hours of sleep per night (4-12)
- `extracurricular_hours`: Weekly extracurricular hours (0-20)

**Categorical Features:**
- `gender`: Male/Female
- `parent_education`: High School, Some College, Bachelor, Master, PhD
- `has_internet`: Yes/No
- `study_method`: Self-study, Group, Tutor, Online
- `transport`: Bus, Car, Walk, Bike

**Target Variable:**
- `performance_category`: Excellent, Good, Average, Below Average, Poor

## 🎓 Key Features

- ✅ Complete ML pipeline from data loading to deployment
- ✅ Multiple model comparison with cross-validation
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Feature engineering with meaningful derived features
- ✅ Interactive web application with Streamlit
- ✅ Model persistence and loading
- ✅ Comprehensive EDA notebook
- ✅ Clean, modular code structure
- ✅ Well-documented code with docstrings

## 🔍 Sample Screenshots Description

### Streamlit App Interface:
- **Left Panel**: Input form with sliders and dropdowns for all student features
- **Main Panel**: 
  - Prediction button and results display
  - Bar chart showing prediction probabilities
  - Feature importance visualization
  - Insights and recommendations
- **Right Panel**: 
  - Input summary in JSON format
  - Information about the model
  - Performance category definitions

### Model Training Output:
- Model comparison table with metrics
- Cross-validation scores
- Best model selection
- Model saving confirmation

## 🐛 Troubleshooting

**Issue: ModuleNotFoundError**
- Solution: Ensure you're in the project root directory and all dependencies are installed

**Issue: Model not found error in Streamlit app**
- Solution: Run `python src/train_models.py` first to generate the model files

**Issue: Dataset not found**
- Solution: Run `python src/create_dataset.py` to generate the dataset

## 📝 Notes

- The dataset is synthetic and generated for demonstration purposes
- Model performance may vary slightly due to random seeds
- The project is optimized for systems with 8GB RAM
- No GPU is required - all models run on CPU

## 🔮 Future Enhancements

- Real-world dataset integration
- Additional feature engineering
- Model interpretability (SHAP values)
- Model versioning
- API deployment
- Database integration for predictions

## 📄 License

This project is for educational purposes.

## 👤 Author

Created as a complete ML engineering project demonstrating end-to-end machine learning workflows.

---

**Happy Learning! 🎓**

