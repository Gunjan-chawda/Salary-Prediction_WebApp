import streamlit as st
import pandas as pd
import joblib

# 🎨 Custom CSS for layout and styling
custom_css = """
<style>
body {
    background-color: #FFF5EC;
}
h1 {
    color: #4A90E2;
    text-align: center;
    font-size: 36px;
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background-color: #FFF5EC;
}
.card {
    background-color: #F9F9F9;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    max-width: 600px;
    margin: auto;
}
.stButton > button {
    background-color: #A38CF8;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
}
.stButton > button:hover {
    background-color: #8b77d8;
}
</style>
"""

st.set_page_config(page_title="Salary Predictor", page_icon="💼")

# ✅ Inject the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# ✅ Main heading
st.markdown("<h1>💼 Software Salary Predictor</h1>", unsafe_allow_html=True)

# 🎯 Load model
model = joblib.load("model/salary_model.pkl")
df_sample = pd.read_csv("Salary_Dataset.csv")

company_options = sorted(df_sample['Company Name'].dropna().unique())
job_title_options = sorted(df_sample['Job Title'].dropna().unique())
location_options = sorted(df_sample['Location'].dropna().unique())
employment_options = sorted(df_sample['Employment Status'].dropna().unique())

# 📦 Card layout
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    company = st.selectbox("🏢 Company Name", company_options)
    job = st.selectbox("👔 Job Title", job_title_options)
    location = st.selectbox("📍 Location", location_options)
    employment = st.selectbox("💼 Employment Status", employment_options)
    rating = st.slider("⭐ Company Rating", 0.0, 5.0, 3.5, 0.1)

    if st.button("📊 Predict Salary"):
        input_data = pd.DataFrame([{
            'Company Name': company,
            'Job Title': job,
            'Location': location,
            'Employment Status': employment,
            'Rating': rating
        }])
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f} per year")
        st.info(f"🗓 Monthly Estimate: ₹{prediction / 12:,.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

