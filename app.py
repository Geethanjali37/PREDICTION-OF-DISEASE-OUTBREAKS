import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

st.set_page_config(page_title="Prediction of Disease Outbreaks", layout="wide", page_icon="🏥")

# Load the pre-trained models with caching for better performance
@st.cache_resource
def load_models():
    try:
        return {
            "diabetes": pickle.load(open('Training_Models/diabetes_model.sav', 'rb')),
            "heart": pickle.load(open('Training_Models/heart_model.sav', 'rb')),
            "parkinsons": pickle.load(open('Training_Models/parkinsons_model.sav', 'rb'))
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

models = load_models()
if models is None:
    st.stop()

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Prediction of Disease Outbreaks",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson’s Prediction"],
        menu_icon="hospital-fill",
        icons=["activity", "heart", "person"],
        default_index=0
    )

# Function to validate and convert inputs
def validate_inputs(inputs):
    try:
        return [float(i) for i in inputs]
    except ValueError:
        return None

# Function to make predictions
def predict_disease(model, input_data, disease_name):
    processed_data = validate_inputs(input_data)
    if processed_data is None:
        st.error("Please enter valid numeric values.")
        return
    prediction = model.predict([np.array(processed_data)])[0]
    result = f"The person {'has' if prediction == 1 else 'does not have'} {disease_name}."
    st.success(result)

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    with col2:
        glucose = st.number_input("Glucose Level", min_value=0.0)
    with col3:
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
    with col1:
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
    with col2:
        insulin = st.number_input("Insulin", min_value=0.0)
    with col3:
        bmi = st.number_input("BMI", min_value=0.0)
    with col1:
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    with col2:
        age = st.number_input("Age", min_value=0, step=1)

    if st.button("Diabetes Test Result"):
        predict_disease(models["diabetes"], [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age], "diabetes")

# Heart Disease Prediction
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0, step=1)
    with col2:
        sex = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    with col3:
        cp = st.number_input("Chest Pain Type", min_value=0, step=1)
    with col1:
        trestbps = st.number_input("Resting Blood Pressure", min_value=0.0)
    with col2:
        chol = st.number_input("Cholesterol Level", min_value=0.0)
    with col3:
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    with col1:
        restecg = st.number_input("Resting ECG", min_value=0, step=1)
    with col2:
        thalach = st.number_input("Max Heart Rate", min_value=0.0)
    with col3:
        exang = st.radio("Exercise Induced Angina", [0, 1])
    with col1:
        oldpeak = st.number_input("ST Depression", min_value=0.0)
    with col2:
        slope = st.number_input("Slope", min_value=0, step=1)
    with col3:
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, step=1)
    with col1:
        thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, step=1)

    if st.button("Heart Disease Test Result"):
        predict_disease(models["heart"], [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal], "heart disease")

# Parkinson's Disease Prediction
if selected == "Parkinson’s Prediction":
    st.title("Parkinson’s Prediction")
    col1, col2, col3 = st.columns(3)

    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", 
        "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "Spread1", "Spread2", "D2", "PPE"
    ]

    input_values = []
    for i, feature in enumerate(features):
        with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
            input_values.append(st.number_input(feature, min_value=0.0))

    if st.button("Parkinson’s Test Result"):
        predict_disease(models["parkinsons"], input_values, "Parkinson’s disease")