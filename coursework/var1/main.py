import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load data with correct delimiter
data = pd.read_csv('Loan_details_updated.csv', delimiter=',', header=None,
                   names=['Name', 'Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Defaulted',
                          'Marital_Status', 'Education_Level', 'Employment_Status'])

# Step 2: Drop the 'Name' column (non-relevant)
data = data.drop('Name', axis=1)
data = data.drop('Name', axis=1)

# Step 3: Convert numeric columns to numeric, allowing coercion for non-convertible values
numeric_columns = ['Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Defaulted']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Step 4: Separate numeric and non-numeric columns for filling missing values
numeric_data = data[numeric_columns]
categorical_data = data.drop(numeric_columns, axis=1)

# Fill missing values for numeric columns with the median
numeric_data.fillna(numeric_data.median(), inplace=True)

# Fill missing values for categorical columns with the mode (most frequent value)
categorical_data.fillna(categorical_data.mode().iloc[0], inplace=True)

# Step 5: Combine numeric and categorical columns back together
data = pd.concat([numeric_data, categorical_data], axis=1)

# Step 6: Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Step 7: Split into features (X) and target (y)
X = data.drop('Loan_Amount', axis=1)
y = data['Loan_Amount']

# Step 8: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handle missing values in X_train (after split)
X_train.fillna(X_train.median(), inplace=True)

# Ensure y_train is aligned with X_train
y_train = y_train.loc[X_train.index]

# Step 9: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Build and train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 11: Make predictions
y_pred = model.predict(X_test)

# Coefficients of the model
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

# Sort coefficients by their absolute values
coefficients['abs_coef'] = np.abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values(by='abs_coef', ascending=False)

print(coefficients)

# Step 12: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R²): {r2:.4f}')
