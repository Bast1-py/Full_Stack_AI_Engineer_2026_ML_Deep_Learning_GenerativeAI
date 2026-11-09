from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import numpy as np

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [50,70, 90, 100, 130, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3,5,6,7,8]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5, n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_

print(f'Best Parameter (GridSearchCV): {best_params_grid}')
print(f'Best Cross-validation Accuracy (GridSearchCV): {best_score_grid:.4f}')

best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
acc_grid = accuracy_score(y_test, y_pred_grid)
print(f'Test Accuracy (GridSearchCV): {acc_grid:.4f}')
print('\n Classification Report: \n', classification_report(y_test, y_pred_grid))




param_dist = {
    'C': np.logspace(-3, 3, 10),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist,
    scoring='accuracy',
    n_iter=32,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print(f'Best Parameters (RandomSearchCV): {best_params_random}')
print(f'Best Cross-validation Accuracy (RandomSearchCV): {best_score_random:.4f}')


best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
acc_random = accuracy_score(y_test, y_pred_random)
print(f'Test Accuracy (RandomSearchCV): {acc_random:.4f}')
print(f'\nClassification Report (RandomSearchCV):', classification_report(y_test, y_pred_random))