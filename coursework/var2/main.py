import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load data with correct delimiter
data = pd.read_csv('test.csv', delimiter=',', header=None,
                   names=['Customer ID', 'Name', 'Gender', 'Age', 'Income',
                          'Income Stability', 'Profession', 'Type of Employment',
                          'Location', 'Property Price', 'Loan Amount'])

# Step 2: Drop the 'Name' and 'Customer ID' columns (non-relevant)
data = data.drop('Customer ID', axis=1)
data = data.drop('Name', axis=1)
data = data.drop('Type of Employment', axis=1)


# Step 3: Convert numeric columns to numeric, allowing coercion for non-convertible values
numeric_columns = ['Age', 'Income', 'Property Price', 'Loan Amount']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Step 4: Fill missing values in numeric columns with the mean
for col in numeric_columns:
    data[col].fillna(data[col].mean(), inplace=True)

# Step 5: One-hot encoding for categorical columns
data = pd.get_dummies(data, drop_first=True)

# Step 6: Select a subset of data (optional)
# data = data[0:10000]

# Step 7: Define X (features) and y (target variable)
X = data.drop('Loan Amount', axis=1)
y = data['Loan Amount']

# Step 8: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

import matplotlib.pyplot as plt
import seaborn as sns

# Визуализация важности признаков (коэффициентов)
plt.figure(figsize=(10, 6))
sns.barplot(x='abs_coef', y='Feature', data=coefficients)
plt.title('Feature Importance (Absolute Coefficients)')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.show()

# Визуализация ошибок (разница между предсказанными и реальными значениями)
errors = y_test - y_pred

plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.show()

# Визуализация предсказанных значений против реальных
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.title('Actual vs Predicted Loan Amounts')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.show()

