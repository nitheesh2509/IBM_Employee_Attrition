import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model_path = 'E:/Data Science projects/New folder/IBM_Employee_Attrition/models/random_forest_model.pkl'
scaler_path = 'E:/Data Science projects/New folder/IBM_Employee_Attrition/models/scaler.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app layout
st.title('Employee Attrition Prediction')

# Input fields for model features
st.sidebar.header('User Input Features')

# Actual feature names
feature_names = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
    'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree', 'Gender_Male',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes'
]

def user_input_features():
    features = []
    for feature in feature_names:  # Iterate through actual feature names
        feature_value = st.sidebar.number_input(feature, value=0.0)
        features.append(feature_value)
    
    return np.array(features)

# Get user input
features = user_input_features()

# Scale the features
features_scaled = scaler.transform(features.reshape(1, -1))

# Make prediction
if st.button('Predict'):
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)

    # Display the prediction
    st.write('Prediction: **', int(prediction[0]), '** (0: No Attrition, 1: Attrition)')
    st.write('Prediction Probability: **', prediction_proba[0][1], '**')

st.write('Enter the feature values in the sidebar and click on "Predict" to see the prediction.')
