from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


lr = LogisticRegression()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
knn.fit(X_train, y_train)


ensemble_model = VotingClassifier(
    estimators=[
        ('Logistic Regression', lr),
        ('Decision Tree', dt),
        ('K-Nearest Neighbors', knn)
    ], voting='hard'
)


ensemble_model.fit(X_train, y_train)

y_pred_ensemble = ensemble_model.predict(X_test)
acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f'Ensemble Model Accuracy: {acc_ensemble:.2f}')


y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_knn = knn.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f'Logistic Regression Model Accuracy: {acc_lr:.2f}')
print(f'Decision Tree Model Accuracy: {acc_dt:.2f}')
print(f'K-Nearest Neighbors Accuracy: {acc_knn:.2f}')