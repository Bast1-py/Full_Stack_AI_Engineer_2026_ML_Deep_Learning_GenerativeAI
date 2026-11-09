from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
from xgboost import XGBClassifier

data = load_breast_cancer()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Features: {data.feature_names}')
print(f'\nTarget: {data.target_names}')


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'eta': 0.1
}

xgb_model = xgb.train(params, dtrain, num_boost_round=160)
y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f'\nXGBoost Accuracy: {acc:.4f}')
print('\nClassification Report: \n', classification_report(y_test, y_pred))


param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 70, 100, 130, 160, 200],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_clf = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)
print(f'\nBest Parameter: {grid_search.best_params_}')
print(f'\nBest Cross-validation Accuracy: , {grid_search.best_score_:.4f}')