import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from coursework.tools.metrics import R2, MSE
from coursework.tools.gradient_descent import gradient_descent
from coursework.tools.normalize import normalize
from coursework.tools.split import split

data = np.loadtxt('data/loan_data_no_missing.csv', delimiter=';')

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = split(X, y, test_size=0.3)

X_train_normalized = normalize(X_train)
X_test_normalized = normalize(X_test)

X_train_normalized = np.c_[np.ones(len(X_train_normalized)), X_train_normalized]
X_test_normalized = np.c_[np.ones(len(X_test_normalized)), X_test_normalized]

theta0 = np.zeros(len(X_train_normalized[1]))
theta = gradient_descent(X_train_normalized, y_train, theta=theta0, alpha=0.01, iters=1000)

prediction = X_test_normalized.dot(theta)

mse = MSE(y_test, prediction)
r2 = R2(y_test, prediction)

print(f'\nMean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R²): {r2:.4f}')


plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=prediction, scatter_kws={'alpha':0.7}, line_kws={'color':'red'})
plt.title('Actual vs Predicted Loan Amounts with Regression Line')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label='Actual Loan Amount', color='blue', alpha=0.7)
plt.scatter(range(len(prediction)), prediction, label='Predicted Loan Amount', color='red', alpha=0.7)
plt.title('Actual vs Predicted Loan Amounts')
plt.xlabel('Test Sample Index')
plt.ylabel('Loan Amount')
plt.legend()
plt.show()

joblib.dump(theta, 'weights.pkl')
