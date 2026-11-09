from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Features Names: {data.feature_names}')
print(f'Class Names: {data.target_names}')


param_grid = {
    'n_estimators': [50, 70, 100, 130, 160, 200],
    'max_depth': [None, 3, 5, 7, 9, 10],
    'min_samples_split': [3, 4, 6, 7, 8]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5, scoring='accuracy', n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
acc_grid = accuracy_score(y_test, y_pred_grid)
print(f'Best Hyperparameter (Grid Search): {grid_search.best_params_}')
print(f'Grid Search Accuracy: {acc_grid:.3f}')




param_dist = {
    'n_estimators': np.arange(50,200,10),
    'max_depth': [None, 3, 5, 7, 9, 10],
    'min_samples_split': [3, 4, 6, 7, 8]
}



random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=35, random_state=42,
    cv=5, scoring='accuracy', n_jobs=-1
)

random_search.fit(X_train, y_train)
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
acc_random = accuracy_score(y_test, y_pred_random)
print(f'\n\nBest Hyperparameter (Random Search): {random_search.best_params_}')
print(f'\nRandom Search Accuracy: {acc_random:.3f}')