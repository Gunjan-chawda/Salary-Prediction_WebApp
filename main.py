import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Salary_Dataset.csv')
df['Employment Status'] = df['Employment Status'].replace('Contractor', 'Freelancer')

columns_to_use = ['Rating', 'Company Name', 'Job Title',
                  'Location', 'Employment Status', 'Salary']
df = df[columns_to_use]

df.dropna(inplace=True)

le_company = LabelEncoder()
le_job_title = LabelEncoder()
le_location = LabelEncoder()
le_employment = LabelEncoder()

df['company_encoded'] = le_company.fit_transform(df['Company Name'])
df['job_title_encoded'] = le_job_title.fit_transform(df['Job Title'])
df['location_encoded'] = le_location.fit_transform(df['Location'])
df['employment_encoded'] = le_employment.fit_transform(df['Employment Status'])

X = df[['Rating', 'company_encoded', 'job_title_encoded',
        'location_encoded', 'employment_encoded']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model/salary_model.pkl')
joblib.dump(le_company, 'model/le_company.pkl')
joblib.dump(le_job_title, 'model/le_job_title.pkl')
joblib.dump(le_location, 'model/le_location.pkl')
joblib.dump(le_employment, 'model/le_employment.pkl')

print("Model and encoders saved successfully.")
