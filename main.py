import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("📁 Loading data...")
df = pd.read_csv("Salary_Dataset.csv")
print("✅ Data loaded. Preprocessing...")

# Remove rows with missing salary
df.dropna(subset=['Salary'], inplace=True)

# Define features and target
X = df[['Rating', 'Company Name', 'Job Title', 'Location', 'Employment Status']]
y = df['Salary']

# Categorical feature names
categorical_features = ['Company Name', 'Job Title', 'Location', 'Employment Status']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Build pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model on full data
print("🏃 Training model... please wait.")
pipeline.fit(X, y)

# Predict on same data (just for metric display)
y_pred = pipeline.predict(X)

# Metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print(f"📊 R² Score (Accuracy): {r2:.4f}")
print(f"📊 MAE: ₹{mae:,.2f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/salary_model.pkl")
print("✅ Model saved successfully.")