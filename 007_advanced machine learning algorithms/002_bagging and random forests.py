from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Features: ', data.feature_names)
print('\nClasses: ', data.target_names)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('\nRandom Forest Accuracy: ', acc)
print('\n')
print('Classification Report: \n', classification_report(y_test, y_pred))


param_grid = {
    'n_estimators': [60, 80, 100, 130, 170, 200, 210],
    'max_depth': [None, 4, 5, 7, 9, 10, 12],
    'max_features': ['sqrt', 'log2', 'None']
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f'\n\nBest Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {grid_search.best_score_}')