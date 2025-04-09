import streamlit as st
import joblib

# Load model
model = joblib.load("attrition_model.pkl")

st.title("👩‍💼 Employee Attrition Prediction App")

# Input fields
age = st.number_input("Age", min_value=18, max_value=65)
salary = st.number_input("Monthly Salary", min_value=1000, max_value=500000)
years = st.number_input("Years at Company", min_value=0, max_value=40)
wlb = st.slider("Work Life Balance (1=Poor, 5=Excellent)", 1, 5, 3)
involve = st.slider("Job Involvement (1=Low, 5=High)", 1, 5, 3)
overtime = st.selectbox("Overtime", ("Yes", "No"))
hours = st.number_input("Hours Worked Per Week", min_value=20, max_value=100, value=40)
distance = st.number_input("Distance from Home (km)", min_value=1, max_value=100)

# Encode overtime
overtime_val = 1 if overtime == "Yes" else 0

if st.button("Predict"):
    input_data = [[age, salary, years, wlb, involve, overtime_val, hours, distance]]
    result = model.predict(input_data)[0]

    if result == 1:
        st.error("⚠️ Employee likely to leave.")
    else:
        st.success("✅ Employee likely to stay.")