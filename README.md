# Salary-Prediction_WebApp

## Project Overview

This project is a Machine Learning-based salary prediction model designed for software professionals. It uses various job-related features such as company name, job title, location, employment status, and company rating to predict the expected annual salary. The model helps job seekers, recruiters, and HR professionals estimate a fair salary based on real-world data.

## Features

* Predicts software professional salary based on multiple factors.
* Uses a Random Forest Regression model for accurate predictions.
* Interactive web application built with Streamlit for easy use.
* Encodes categorical variables like company, job title, location, and employment status for model training.
* Provides both annual and approximate monthly salary estimates.

## Dataset

The model is trained on a dataset containing software professional salaries, along with related attributes such as:

* Rating (company rating)
* Company Name
* Job Title
* Location
* Employment Status (e.g., Full-time, Freelancer)

## How It Works

1. **Data Preprocessing:**

   * Replaces certain categories (e.g., "Contractor" with "Freelancer") for consistency.
   * Encodes categorical columns into numeric values using Label Encoding.
   * Splits data into training and testing sets.

2. **Model Training:**

   * Uses Random Forest Regressor to learn the relationship between features and salary.
   * Saves the trained model and label encoders for use in the web app.

3. **Web App Prediction:**

   * Allows users to select job-related inputs from dropdown menus.
   * Inputs are encoded and fed into the trained model.
   * Displays predicted salary and monthly salary estimate.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit

## Installation

To run the project locally:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Navigate into the project directory:

   ```bash
   cd your-repo-name
   ```
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Usage

* Open the Streamlit app in your browser.
* Select your Company Name, Job Title, Location, Employment Status, and Rate the company.
* Click **Predict Salary**.
* See the predicted annual and monthly salary displayed.

## Folder Structure

```
├── app.py                 # Streamlit web application
├── main.py                # Model training script
├── model/                 # Contains saved model and label encoders
├── Salary_Dataset.csv     # Dataset file
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
```

## Future Work

* Improve model accuracy with more data and advanced algorithms.
* Add support for more job titles and companies.
* Integrate salary trends over time and location-based cost-of-living adjustments.
* Deploy the app on cloud platforms for wider accessibility.

