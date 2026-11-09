import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
df = df[
    [
        'Pclass',
        'Sex',
        'Age',
        'Fare',
        'Embarked',
        'Survived'
    ]
]

df.fillna(
    {
        'Age': df['Age'].median()
    }, inplace=True
)

df.fillna(
    {
        'Embarked': df['Embarked'].mode()[0]
    }, inplace=True
)


X = df.drop(columns=['Survived'])
y = df['Survived']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Fare']),
        ('cat', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked'])
    ]
)

X_preprocessed = preprocessor.fit_transform(X)

lr = LogisticRegression()
lr_scores = cross_val_score(lr, X_preprocessed, y, cv=5, scoring='accuracy')
print(f'Logistic Regression Accuracy: {lr_scores.mean()*100}%')

rf = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf, X_preprocessed, y, cv=5, scoring='accuracy')
print(f'Random Forest Accuracy: {rf_scores.mean()*100}%')


param_grid = {
    'n_estimators': [20, 50, 60, 80, 100, 150, 200],
    'max_depth': [None, 4, 5, 7, 10, 12, 15],
    'min_samples_split': [2, 3, 5, 6, 8, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)


grid_search.fit(X_preprocessed, y)

# display best hyperparameter and score
print(f'Best Hyperparameter: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_*100}%')