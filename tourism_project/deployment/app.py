import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="nbhoite9988/Tourism-Package-Sales-Prediction", filename="best_tourism_package_sale_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Sales Prediction
st.title("Tourism Package Sales Prediction App")
st.write("""
This application predicts the likelihood of a tourism package being sold based on its parameters.
Please enter the data below to get a prediction.
""")

# User input
input_data = {
    'Age': st.number_input("Age", min_value=0, max_value=100, value=30),
    'TypeofContact': st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited']),
    'CityTier': st.selectbox("City Tier", [1, 2, 3]),
    'DurationOfPitch': st.number_input("Duration of Pitch (in minutes)", min_value=0, max_value=150, value=15),
    'Occupation': st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business']),
    'Gender': st.selectbox("Gender", ['Female', 'Male']),
    'NumberOfPersonVisiting': st.number_input("Number of People Visiting", min_value=0, max_value=10, value=2),
    'NumberOfFollowups': st.number_input("Number of Followups", min_value=0, max_value=10, value=3),
    'ProductPitched': st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King']),
    'PreferredPropertyStar': st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3),
    'MaritalStatus': st.selectbox("Marital Status", ['Single', 'Divorced', 'Married', 'Unmarried']),
    'NumberOfTrips': st.number_input("Number of Trips", min_value=0, max_value=30, value=5),
    'Passport': st.number_input("Passport (1 for Yes, 0 for No)", min_value=0, max_value=1, value=1),
    'PitchSatisfactionScore': st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=3),
    'OwnCar': st.number_input("Own Car (1 for Yes, 0 for No)", min_value=0, max_value=1, value=1),
    'NumberOfChildrenVisiting': st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0),
    'Designation': st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP']),
    'MonthlyIncome': st.number_input("Monthly Income", min_value=0, value=23000)
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction
if st.button("Predict Tourism Package Sale"):
    prediction = model.predict(input_df)[0]
    result = "Package Sold" if prediction == 1 else "Package Not Sold"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
