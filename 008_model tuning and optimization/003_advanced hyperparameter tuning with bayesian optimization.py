from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import optuna


data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f'Training Data Shape: {X_train.shape}')
print(f'Testing Data Shape: {X_test.shape}')


baseline_model = XGBClassifier(eval_metric='logloss', random_state=42)
baseline_model.fit(X_train, y_train)

baseline_preds = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_preds)
print(f'Baseline XGBoost Model Accuracy: {baseline_accuracy:.4f}')


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 230),
        'max_depth': trial.suggest_int('max_depth', 2, 36),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
    }

    model = XGBClassifier(eval_metric='logloss', random_state=42, **params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=60)

print('Best Hyperparameter: ', study.best_params)
print('Best Accuracy: ', study.best_value)
print('\n\n')



param_grid = {
    'n_estimator': [50, 80, 100, 140, 150, 200],
    'max_depth': [3,5,6,8,9,11],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
)

grid_search.fit(X_train, y_train)
print('Grid Search Best Parameters: ', grid_search.best_params_)
print('Grid Search Best Accuracy: ', grid_search.best_score_)
print('\n\n')



param_dist = {
    'n_estimator': [50, 80, 100, 140, 150, 200, 250],
    'max_depth': [3,5,6,8,9,11],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    n_iter=60,
    cv=5, scoring='accuracy',
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print('Random Search Best Parameters: ', random_search.best_params_)
print('Random Search Best Accuracy: ', random_search.best_score_)