import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model', 'salary_model.pkl'))
le_company = joblib.load(os.path.join(BASE_DIR, 'model', 'le_company.pkl'))
le_location = joblib.load(os.path.join(BASE_DIR, 'model', 'le_location.pkl'))
le_employment = joblib.load(os.path.join(
    BASE_DIR, 'model', 'le_employment.pkl'))
le_job_title = joblib.load(os.path.join(BASE_DIR, 'model', 'le_job_title.pkl'))

st.title(" Software Professional Salary Prediction")

company_options = list(le_company.classes_)
job_title_options = list(le_job_title.classes_)
location_options = list(le_location.classes_)
employment_options = list(le_employment.classes_)

company_input = st.selectbox(" Company Name", company_options)
job_title_input = st.selectbox(" Job Title", job_title_options)
location_input = st.selectbox(" Location", location_options)
employment_input = st.selectbox(" Employment Status", employment_options)
rating_input = st.number_input(
    " Rating (0-5)", min_value=0.0, max_value=5.0, value=3.0, step=0.1)

if st.button(" Predict Salary"):
    try:
        company_encoded = le_company.transform([company_input])[0]
        job_title_encoded = le_job_title.transform([job_title_input])[0]
        location_encoded = le_location.transform([location_input])[0]
        employment_encoded = le_employment.transform([employment_input])[0]

        input_df = pd.DataFrame({
            'Rating': [rating_input],
            'company_encoded': [company_encoded],
            'job_title_encoded': [job_title_encoded],
            'location_encoded': [location_encoded],
            'employment_encoded': [employment_encoded]
        })

        pred_salary = model.predict(input_df)[0]

        st.success(f"🇮🇳 Predicted Salary: ₹{pred_salary:,.2f} per year")
        monthly_salary = pred_salary / 12
        st.info(f" Approx. Monthly Salary: ₹{monthly_salary:,.2f}")

    except Exception as e:
        st.error(f" Error: {e}")
