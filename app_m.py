import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model and scaler
MODEL_PATH = 'xg_model.pkl'

with open(MODEL_PATH, 'rb') as model_file:
    rf_model = pickle.load(model_file)


# Streamlit UI
st.title("Alzheimer's Prediction App")
st.write("Enter patient details to predict the likelihood of Alzheimer's.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Hispanic", "Asian", "Other"])
education_level = st.number_input("Education Level (years)", min_value=0, max_value=30, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
smoking = st.selectbox("Smoking", ["Yes", "No"])
alcohol = st.number_input("Alcohol Consumption (units per week)", min_value=0, max_value=100, step=1)
activity = st.number_input("Physical Activity (hours per week)", min_value=0, max_value=40, step=1)
diet = st.number_input("Diet Quality (scale 1-10)", min_value=1, max_value=10, step=1)
sleep = st.number_input("Sleep Quality (hours per night)", min_value=0, max_value=24, step=1)
family_history = st.selectbox("Family History of Alzheimer's", ["Yes", "No"])
cardiovascular = st.selectbox("Cardiovascular Disease", ["Yes", "No"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
depression = st.selectbox("Depression", ["Yes", "No"])
head_injury = st.selectbox("Head Injury", ["Yes", "No"])
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, step=1)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, step=1)
cholesterol = st.number_input("Total Cholesterol", min_value=100, max_value=300, step=1)
mmse = st.number_input("MMSE Score", min_value=0, max_value=30, step=1)
functional_assessment = st.number_input("Functional Assessment Score", min_value=0, max_value=100, step=1)
memory_complaints = st.selectbox("Memory Complaints", ["Yes", "No"])
behavioral_problems = st.selectbox("Behavioral Problems", ["Yes", "No"])
adl = st.number_input("Activities of Daily Living (ADL) Score", min_value=0, max_value=100, step=1)
confusion = st.selectbox("Confusion", ["Yes", "No"])
personality_changes = st.selectbox("Personality Changes", ["Yes", "No"])
difficulty_tasks = st.selectbox("Difficulty Completing Tasks", ["Yes", "No"])
forgetfulness = st.selectbox("Forgetfulness", ["Yes", "No"])

# Encode categorical values
def encode_binary(value):
    return 1 if value == "Yes" else 0

gender = 1 if gender == "Male" else 0
smoking = encode_binary(smoking)
family_history = encode_binary(family_history)
cardiovascular = encode_binary(cardiovascular)
diabetes = encode_binary(diabetes)
depression = encode_binary(depression)
head_injury = encode_binary(head_injury)
hypertension = encode_binary(hypertension)
memory_complaints = encode_binary(memory_complaints)
behavioral_problems = encode_binary(behavioral_problems)
confusion = encode_binary(confusion)
personality_changes = encode_binary(personality_changes)
difficulty_tasks = encode_binary(difficulty_tasks)
forgetfulness = encode_binary(forgetfulness)

ethnicity_mapping = {"White": 0, "Black": 1, "Hispanic": 2, "Asian": 3, "Other": 4}
ethnicity = ethnicity_mapping[ethnicity]

# Prediction button
if st.button("Predict"):
    try:
        # Prepare input data
        input_features = np.array([[age, gender, ethnicity, education_level, bmi, smoking, alcohol, activity, diet, sleep, 
                            family_history, cardiovascular, diabetes, depression, head_injury, hypertension, 
                            systolic_bp, diastolic_bp, cholesterol, mmse, functional_assessment, 
                            memory_complaints, behavioral_problems, adl, confusion, personality_changes, 
                            difficulty_tasks, forgetfulness]])

        prediction = rf_model.predict(input_features)
        
        # Display result
        if prediction == 1:
            st.success("The model predicts a high risk of Alzheimer's.")
        else:
            st.success("The model predicts a low risk of Alzheimer's.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
