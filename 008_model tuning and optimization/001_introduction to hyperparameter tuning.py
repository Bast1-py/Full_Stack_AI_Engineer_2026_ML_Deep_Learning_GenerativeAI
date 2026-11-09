from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Feature Names: {data.feature_names}')
print(f'Target Names: {data.target_names}')


rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_predict_default = rf_default.predict(X_test)
accuracy_default = accuracy_score(y_test, y_predict_default)

print(f'Deafult Model Accuracy: {accuracy_default:.3f}')
print('\nClassification Report: \n', classification_report(y_test, y_predict_default))


rf_tuned = RandomForestClassifier(random_state=42, n_estimators=160, max_depth=4)
rf_tuned.fit(X_train, y_train)
y_pred_tuned = rf_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f'\nTuned Model Accuracy: {accuracy_tuned:.3f}')
print('\nClassification Report: \n', classification_report(y_test, y_pred_tuned))