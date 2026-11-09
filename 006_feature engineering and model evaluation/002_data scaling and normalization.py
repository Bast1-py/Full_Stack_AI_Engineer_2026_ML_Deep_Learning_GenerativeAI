# hands-on exercise: 1. apply min-max and standardization to a dataset using scikit-learn
# 2. observe the effects of scaling on model performance by training a KNN classifier before and after scaling


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print('Datasets Info: ')
print(X.describe())
print('\n Target Classes: ', data.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Accuracy Without Scaling: ', accuracy_score(y_test, y_pred)*100)


# apply min-max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state = 42)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)

# predict and evaluate
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print('Accuracy With Scaling (Min-max): ', accuracy_score(y_test, y_pred_scaled)*100)