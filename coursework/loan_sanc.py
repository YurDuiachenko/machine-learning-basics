import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Step 1: Load data with correct delimiter
data = pd.read_csv('loan_sanction_train.csv', delimiter=',', header=None,
                   names=['Loan_ID', 'Gender', 'Married', 'Dependents',
                          'Education', 'Self_Employed', 'ApplicantIncome',
                          'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                          'Credit_History', 'Property_Area', 'Loan_Status'])

data = data.drop('Loan_ID', axis=1)
# Создание DataFrame
df = pd.DataFrame(data)

# Преобразование категориальных данных в числовые
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Step 3: Fill missing values with the median of each column
df.fillna(data.median(), inplace=True)

# Step 4: Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Определение признаков и целевой переменной
X = df.drop(columns=['Loan_ID', 'Gender'])  # Признаки
y = df['Gender']  # Целевая переменная

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Преобразование предсказанных значений в бинарные (0 или 1)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_binary)

print(f'Mean Squared Error: {mse}')
print(f'Accuracy: {accuracy}')
