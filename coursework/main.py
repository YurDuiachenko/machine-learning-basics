import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('loan_data_no_missing.csv', delimiter=';', header=None,
                   names=['loan_limit', 'approv_in_adv',
                          'loan_type', 'loan_purpose',
                          'business_or_commercial',
                          'upfront_charges', 'term',
                          'prortization', 'interest_only',
                          'property_value', 'occupancy_type',
                          'total_units', 'income', 'credit_type',
                          'credit_score', 'age', 'submission_of_application',
                          'LTV', 'Status', 'dt31', 'Gender', 'loan_amount'])



X = data.drop('loan_amount', axis=1)
y = data['loan_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

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