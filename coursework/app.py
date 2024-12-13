import numpy as np
import joblib

theta = joblib.load('weights.pkl')


def predict(X):
    X = np.array(X)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    X_normalized = np.concatenate([[1], X_normalized])

    return X_normalized.dot(theta)

def get_input(prompt):
    while True:
        try:
            value = input(prompt)
            if value.replace('.', '', 1).isdigit() or value.isdigit():
                return float(value)
            else:
                return value
        except ValueError:
            print("Ошибка ввода. Пожалуйста, введите правильное значение.")


loan_limit = get_input("Введите одобренную сумму кредита (loan_limit): ")
approv_in_adv = get_input("Введите статус одобрения (approv_in_adv): ")
loan_type = get_input("Введите тип кредита (loan_type): ")
loan_purpose = get_input("Введите цель кредита (loan_purpose): ")
business_or_commercial = get_input("Введите тип бизнеса (business_or_commercial): ")
upfront_charges = get_input("Введите стартовые расходы (upfront_charges): ")
term = get_input("Введите срок кредита (term): ")
prortization = get_input("Введите способ амортизации (prortization): ")
interest_only = get_input("Только проценты (interest_only): ")
property_value = get_input("Введите стоимость недвижимости (property_value): ")
occupancy_type = get_input("Введите тип жилья (occupancy_type): ")
total_units = get_input("Введите количество единиц недвижимости (total_units): ")
income = get_input("Введите доход (income): ")
credit_type = get_input("Введите тип кредита (credit_type): ")
credit_score = get_input("Введите кредитный рейтинг (credit_score): ")
age = get_input("Введите возраст заемщика (age): ")
submission_of_application = get_input("Дата подачи заявки (submission_of_application): ")
LTV = get_input("Введите коэффициент заем/стоимость (LTV): ")
status = get_input("Введите статус заявки (Status): ")
dt31 = get_input("Введите значение dt31: ")
gender = get_input("Введите пол заемщика (Gender): ")


# 1
# 0
# 1
# 1
# 0
# 6260.0
# 360.0
# 0
# 0
# 468000.0
# 1
# 1
# 4920.0
# 1
# 647
# 3
# 0
# 93.26923077
# 0
# 42.0
# 0
#
# 436500
