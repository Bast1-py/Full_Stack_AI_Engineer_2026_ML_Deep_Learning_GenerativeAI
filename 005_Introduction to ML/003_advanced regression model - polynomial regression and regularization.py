# hands-on exercise 1: implement polynomial regression and visualize the fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data = fetch_california_housing(as_frame=True)
df = data.frame

X = df[['MedInc']]
y = df[['MedHouseVal']]
poly = PolynomialFeatures(degree = 2, include_bias=False)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
ridge_prediction = ridge_model.predict(X_test)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_prediction)
print('Ridge Predictions MSE: ', ridge_mse)
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print('Lasso Predictions MSE: ', lasso_mse)


# model = LinearRegression()
# model.fit(X_poly, y)
# y_pred = model.predict(X_poly)

plt.figure(figsize=(10,6))
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual Data', alpha=0.5)
plt.scatter(X_test[:, 0], ridge_prediction, color='green', label='Ridge Predictions', alpha=0.5)
plt.scatter(X_test[:, 0], lasso_predictions, color='black', label='Lasso Predictions', alpha=0.5)
plt.title('PRidge vs Lasso')
plt.xlabel('Median Income (Transformed)')
plt.ylabel('Median House Value in California')
plt.legend()
plt.show()

# mse = mean_squared_error(y, y_pred)
# print('Mean Squared Error: ', mse)