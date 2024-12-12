import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from split import split

# Чтение данных (можно использовать np.genfromtxt для чтения CSV файла)
data = np.genfromtxt('loan_data_no_missing.csv', delimiter=';')

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = split(X, y, test_size=0.3)

# Стандартизация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Функция для градиентного спуска
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
       m, n = X.shape
       theta = np.zeros(n)
       for epoch in range(epochs):
              predictions = X.dot(theta)
              error = predictions - y
              gradient = (2 / m) * X.T.dot(error)
              theta -= learning_rate * gradient
       return theta

# Добавление столбца единичных значений для свободного коэффициента (базового сдвига)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Применение градиентного спуска для нахождения коэффициентов
theta = gradient_descent(X_train, y_train, learning_rate=0.01, epochs=1000)

# Предсказания на тестовой выборке
y_pred = X_test.dot(theta)

# Вычисление коэффициентов и их важности
coefficients = theta[1:]  # Исключаем свободный коэффициент
abs_coefficients = np.abs(coefficients)
features = range(X.shape[1])

# Печать коэффициентов
for feature, coef in zip(features, coefficients):
       print(f'Feature {feature}: Coefficient = {coef}')

# Сортировка по абсолютному значению коэффициента
sorted_features = np.argsort(abs_coefficients)[::-1]

# Печать отсортированных коэффициентов
print("\nSorted by absolute coefficient value:")
for idx in sorted_features:
       print(f"Feature {idx}: Coefficient = {coefficients[idx]}")

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R²): {r2:.4f}')

# Визуализация важности признаков (коэффициентов)
plt.figure(figsize=(10, 6))
sns.barplot(x=abs_coefficients[sorted_features], y=sorted_features)
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
