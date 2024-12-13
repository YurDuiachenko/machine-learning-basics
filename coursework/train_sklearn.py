import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/loan_data_no_missing.csv', delimiter=';', header=None,
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

prediction = model.predict(X_test)

mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=prediction, scatter_kws={'alpha':0.7}, line_kws={'color':'red'})
plt.title('Фактические и Предсказанные суммы кредитов')
plt.xlabel('Фактическая сумма кредита')
plt.ylabel('Предсказанная сумма кредита')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label='Фактическая сумма кредита', color='blue', alpha=0.7)
plt.scatter(range(len(prediction)), prediction, label='Предсказанная сумма кредита', color='red', alpha=0.7)
plt.title('Фактические и Предсказанные суммы кредитов')
plt.xlabel('Индекс примера')
plt.ylabel('Сумма кредита')
plt.legend()
plt.show()

