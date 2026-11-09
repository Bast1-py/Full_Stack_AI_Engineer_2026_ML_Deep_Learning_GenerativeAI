import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('Telco-Customer-Churn.csv')
print('Datasets Info: \n')
print(df.info())
print('\n Class Distribution: \n')
print(df['Churn'].value_counts())
print('\n Sample Data: \n', df.head())


# handle missing value
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(
    {
        'TotalCharges': df['TotalCharges'].median()
    }, inplace=True
)

# encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = label_encoder.fit_transform(df[column])

# encode target variables
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# scale numerical features
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])


# features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# split datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
acc_pred = accuracy_score(y_test, y_pred)
print(f'\nInitial Model Accuracy: {acc_pred:.4f}')
print('\nClassififcation Report: \n', classification_report(y_test, y_pred))


# define parameter grid
param_dist = {
    'n_estimators': np.arange(40, 250, 10),
    'max_depth': [None, 4, 6, 10, 13, 16, 18],
    'min_samples_split': [2,4,5,7,8,10,13,20],
    'min_samples_leaf': [1,2,3,4,5,6,7]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    scoring='accuracy',
    n_iter=31,
    cv=5,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)
best_params_random = random_search.best_params_
print(f'Best Parameters (RandomSearchCV): {best_params_random}')

# train best model
best_model = random_search.best_estimator_

# predict and evaluate
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f'Tunes Model Accuracy Report (Tuned Model):\n', classification_report(y_test, y_pred_tuned))


# evaluate using cross-validation
cv_scores = cross_val_score(
    best_model, X, y, cv=5, scoring='accuracy', n_jobs=-1
)

print(f'Cross-validation Accuracy: {cv_scores.mean():.4f}')