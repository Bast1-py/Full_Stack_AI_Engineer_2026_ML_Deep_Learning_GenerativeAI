import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score


data = load_iris()
X = data.data
y = (data.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# hands-on exercise 1: Classification model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Class 0', 'Class 0'])
disp.plot(cmap='RdPu')
plt.title('Confusion Matrix')
plt.show()

# classification matrix
print('\n Classification Report: ')
print(classification_report(y_test, y_pred))


# hands-on exercise: regression models
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error (MAE): , {mae:.2f}')
print(f'Mean Squared Error (MSE): , {mse:.2f}')
print(f'R-Squared Error (R2): {r2:.2f}')

model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print(f'\n\nMean Absolute Error (MAE) KNN: , {mae_knn:.2f}')
print(f'Mean Squared Error (MSE) KNN: , {mse_knn:.2f}')
print(f'R-Squared Error (R2) KNN: {r2_knn:.2f}')