import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

# Step 1: Load data with correct delimiter
data = pd.read_csv('loan_data.csv', delimiter=';', header=None,
                   names=['loan_limit', 'approv_in_adv',
                          'loan_type', 'loan_purpose',
                          'business_or_commercial', 'loan_amount',
                          'upfront_charges', 'term',
                          'prortization', 'interest_only',
                          'property_value', 'occupancy_type',
                          'total_units', 'income', 'credit_type',
                          'credit_score', 'age', 'submission_of_application',
                          'LTV', 'Status', 'dt31', 'Gender'])

# Step 2: Convert numeric columns to numeric
numeric_columns = ['loan_amount', 'upfront_charges', 'term', 'property_value',
                   'total_units', 'income', 'credit_score', 'age', 'LTV', 'Status']

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Step 3: Fill missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Step 4: Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

data = data[0:1000]
data = data.drop(['loan_limit', 'approv_in_adv', 'loan_purpose',
                  'upfront_charges', 'term',
                  'prortization', 'interest_only',
                  'property_value', 'occupancy_type',
                  'total_units', 'income', 'credit_type',
                  'credit_score', 'submission_of_application',
                  'LTV', 'Status', 'dt31'], axis=1)

# Step 5: Split into features (X) and target (y)
X = data.drop('Gender', axis=1)
y = data['Gender']

# Step 6: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handle missing values in X_train (after split)
X_train.fillna(X_train.median(), inplace=True)

# Ensure y_train is aligned with X_train
y_train = y_train.loc[X_train.index]

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train logistic regression model

# Step 8: Build and train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

# Сортируем по абсолютному значению коэффициента
coefficients['abs_coef'] = np.abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values(by='abs_coef', ascending=False)

print(coefficients)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R²): {r2:.4f}')
