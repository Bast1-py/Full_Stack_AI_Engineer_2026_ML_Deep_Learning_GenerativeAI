# hands-on exercise: 1. create new features from a date column(e.g., day of the week, mont year)
# 2. apply polynomial transformations to a dataset and compare model performance before and after transformation


import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv('bike_sharing_daily.csv')


# convert dteday to datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# create new features
df['day_of_week'] = df['dteday'].dt.day_name()
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year
print('\nNew Features Derived From Date Column')
print(df[['dteday', 'day_of_week', 'month', 'year']].head())


X = df[['temp']]
y = df['cnt']


# apply polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print('\nOriginal and Polynomial Features')
print(pd.DataFrame(X_poly, columns=['temp', 'temp^2']).head())


# train and evaluate model with original features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_poly_train, X_poly_test = train_test_split(X_poly, test_size=0.2, random_state=42)


model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
mse_original = mean_squared_error(y_test, y_pred_original)


# train and test evaluate model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_pred_poly = model_poly.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)


# compare result
print(f'\nMSE Original: {mse_original:.2f}')
print(f'MSE Polynomial: {mse_poly:.2f}')