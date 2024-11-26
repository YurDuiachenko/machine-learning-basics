import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load data with correct delimiter
data = pd.read_csv('loan_data.csv', delimiter=';', header=None,
                   names=['loan_limit', 'approv_in_adv', 'loan_type', 'loan_purpose', 'business_or_commercial', 'loan_amount', 'upfront_charges', 'term', 'prortization', 'interest_only', 'property_value', 'occupancy_type', 'total_units', 'income', 'credit_type', 'credit_score', 'age', 'submission_of_application', 'LTV', 'Status', 'dt31', 'Gender'])

# Step 2: Convert numeric columns to numeric
numeric_columns = ['loan_amount', 'upfront_charges', 'term', 'property_value',
                   'total_units', 'income', 'credit_score', 'age', 'LTV', 'Status']

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Step 3: Fill missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Step 4: Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

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
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
