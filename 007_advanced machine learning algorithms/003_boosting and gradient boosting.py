import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Features: {data.feature_names}')
print(f'\nClasses: {data.target_names}')


gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'\nGradient Boosting Accuracy: {acc:.3f}')
print('\Classification Report" \n ', classification_report(y_test, y_pred))



param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 60, 80, 100, 130, 150, 180, 200],
    'max_depth': [2, 3, 4, 6, 7]
}

grid_search = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)
print(f'\nBest Parameters Gradient Boosting: {grid_search.best_params_}')
print(f'Best Cross-validation Gradient Boosting: {grid_search.best_score_:.3f}')