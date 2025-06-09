# Salary-Prediction_WebApp

This project is a **Machine Learning pipeline** for predicting salaries based on features such as company name, job title, location, rating, and employment status. It is trained using a **Linear Regression model** and supports **end-to-end prediction** with preprocessing included via a `Pipeline`.

## 📁 Project Structure

```bash
.
├── app.py                 # Script to train and save the model
├── main.py                # (same as app.py, can be cleaned up later)
├── model/
│   └── salary_model.pkl   # Trained and saved model
├── Salary_Dataset.csv     # Dataset used for training
└── README.md              # Project documentation
````

## 🔍 Features

* ✅ Uses `pandas` for data manipulation
* ✅ Preprocessing with `OneHotEncoder` for categorical features
* ✅ Model building using `LinearRegression`
* ✅ Metrics: R² Score and Mean Absolute Error (MAE)
* ✅ Saved model using `joblib` for future use

## 📊 Input Features

The model takes the following features as input:

* `Rating` (numerical)
* `Company Name` (categorical)
* `Job Title` (categorical)
* `Location` (categorical)
* `Employment Status` (categorical)

## 🏃 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/salary-predictor.git
cd salary-predictor
```

### 2. Install Requirements

Make sure Python is installed (preferably 3.7+), then install dependencies:

```bash
pip install pandas scikit-learn joblib
```

### 3. Add Dataset

Make sure your `Salary_Dataset.csv` is placed in the root directory. It must include at least the following columns:

* `Salary`
* `Rating`
* `Company Name`
* `Job Title`
* `Location`
* `Employment Status`

### 4. Run the Training Script

```bash
python app.py
```

## 📈 Output Example

```bash
📁 Loading data...
✅ Data loaded. Preprocessing...
🏃 Training model... please wait.
📊 R² Score (Accuracy): 0.83
📊 MAE: ₹173,474.05
✅ Model saved successfully.
```

## 🧠 Future Improvements

* Add streamlit or flask-based frontend for predictions
* Support for more advanced models (e.g., Random Forest, XGBoost)
* Add data visualization for better insights
