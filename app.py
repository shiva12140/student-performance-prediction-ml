"""
Streamlit web application for student performance prediction.
Provides an interactive interface for making predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing import DataPreprocessor
from feature_engineering import create_derived_features


# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Student Performance Prediction System")
st.markdown("---")

# Load model and preprocessor
@st.cache_resource
def load_model_components():
    """Load model, preprocessor, and feature information."""
    models_dir = Path("models")
    
    try:
        # Try to load tuned model first, fallback to regular model
        model_path = models_dir / "best_model_tuned.pkl"
        if not model_path.exists():
            model_path = models_dir / "best_model.pkl"
        
        model = joblib.load(model_path)
        preprocessor = DataPreprocessor.load(models_dir / "preprocessor.pkl")
        numerical_cols = joblib.load(models_dir / "numerical_cols.pkl")
        categorical_cols = joblib.load(models_dir / "categorical_cols.pkl")
        
        return model, preprocessor, numerical_cols, categorical_cols
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run the training script first to generate the model.")
        return None, None, None, None


model, preprocessor, numerical_cols, categorical_cols = load_model_components()

if model is not None:
    # Sidebar for input
    st.sidebar.header("📝 Student Information")
    
    # Numerical inputs
    st.sidebar.subheader("Academic Metrics")
    age = st.sidebar.slider("Age", 15, 22, 18)
    study_hours = st.sidebar.slider("Study Hours per Week", 0, 50, 20)
    attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
    previous_grade = st.sidebar.slider("Previous Grade (%)", 0, 100, 70)
    hours_sleep = st.sidebar.slider("Hours of Sleep per Night", 4.0, 12.0, 7.0, 0.5)
    extracurricular_hours = st.sidebar.slider("Extracurricular Hours per Week", 0, 20, 5)
    
    # Categorical inputs
    st.sidebar.subheader("Personal Information")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    parent_education = st.sidebar.selectbox(
        "Parent Education Level",
        ["High School", "Some College", "Bachelor", "Master", "PhD"]
    )
    has_internet = st.sidebar.selectbox("Has Internet Access", ["Yes", "No"])
    study_method = st.sidebar.selectbox(
        "Study Method",
        ["Self-study", "Group", "Tutor", "Online"]
    )
    transport = st.sidebar.selectbox(
        "Transportation Method",
        ["Bus", "Car", "Walk", "Bike"]
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Prediction")
        
        # Create input DataFrame
        input_data = {
            'age': age,
            'study_hours': study_hours,
            'attendance': attendance,
            'previous_grade': previous_grade,
            'hours_sleep': hours_sleep,
            'extracurricular_hours': extracurricular_hours,
            'gender': gender,
            'parent_education': parent_education,
            'has_internet': has_internet,
            'study_method': study_method,
            'transport': transport
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Create derived features
        input_df = create_derived_features(input_df)
        
        # Preprocess
        input_processed = preprocessor.transform(
            input_df,
            numerical_cols,
            categorical_cols
        )
        
        # Make prediction
        if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
            prediction = model.predict(input_processed)[0]
            probabilities = model.predict_proba(input_processed)[0]
            classes = model.classes_
            
            # Display prediction
            st.success(f"**Predicted Performance Category: {prediction}**")
            
            # Display probabilities
            st.subheader("📈 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Category': classes,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            st.bar_chart(prob_df.set_index('Category'))
            
            # Display explanation
            st.subheader("💡 Explanation")
            
            # Calculate feature importance if available (for Random Forest)
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': input_processed.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("**Top Contributing Factors:**")
                for idx, row in feature_importance.head(5).iterrows():
                    st.write(f"- {row['Feature']}: {row['Importance']:.3f}")
            
            # Provide insights based on input
            insights = []
            if attendance >= 90:
                insights.append("✅ Excellent attendance - strong positive factor")
            elif attendance < 70:
                insights.append("⚠️ Low attendance - may negatively impact performance")
            
            if study_hours >= 30:
                insights.append("✅ High study hours - good commitment")
            elif study_hours < 15:
                insights.append("⚠️ Low study hours - consider increasing study time")
            
            if previous_grade >= 80:
                insights.append("✅ Strong previous academic performance")
            elif previous_grade < 60:
                insights.append("⚠️ Previous grades below average - may need extra support")
            
            if hours_sleep >= 7 and hours_sleep <= 9:
                insights.append("✅ Optimal sleep duration")
            elif hours_sleep < 6:
                insights.append("⚠️ Insufficient sleep - may affect performance")
            
            if insights:
                st.write("**Key Insights:**")
                for insight in insights:
                    st.write(insight)
    
    with col2:
        st.subheader("📋 Input Summary")
        st.json(input_data)
        
        st.subheader("ℹ️ About")
        st.info("""
        This model predicts student performance based on:
        - Academic metrics (study hours, attendance, previous grades)
        - Lifestyle factors (sleep, extracurricular activities)
        - Personal background (education, resources)
        """)
        
        st.subheader("🎯 Performance Categories")
        categories = {
            "Excellent": "80-100%",
            "Good": "70-79%",
            "Average": "60-69%",
            "Below Average": "50-59%",
            "Poor": "Below 50%"
        }
        for cat, score in categories.items():
            st.write(f"**{cat}**: {score}")

else:
    st.warning("⚠️ Model not found. Please run the training script first.")
    st.code("""
    # Run this command to train the model:
    python src/train_models.py
    """)

