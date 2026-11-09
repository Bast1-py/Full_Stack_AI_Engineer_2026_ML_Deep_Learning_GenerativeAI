from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

california = fetch_california_housing()
X, y = california.data, california.target
features_names = california.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Feature Names: \n', features_names)
print('\n Sample Data: \n', pd.DataFrame(X, columns=features_names).head())

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred)

print(f'Linear Regression MSE (No Regularization): {mse_lr:.5f}')
print('\nCoefficient: \n', lr_model.coef_)


ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print('\n')
print(f'Ridge Regression MSE: {mse_ridge:.5f}')
print('Cofficient: \n', ridge_model.coef_)



lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print('\n')
print(f'Lasso Regression MSE: {mse_lasso:.5f}')
print('Cofficient: \n', lasso_model.coef_)