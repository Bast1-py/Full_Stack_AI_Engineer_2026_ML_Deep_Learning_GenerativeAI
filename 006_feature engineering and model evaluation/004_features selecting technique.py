# hands-on exercise: 1. use correlation and mutual information to select important features from a dataset
# 2. apply a tree-based model (e.g., Random Forest) to identify the most important features

from sklearn.datasets import load_diabetes
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# display dataset information
# print(df.head())
# print(df.info())

correlation_matrix = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
plt.title('Correlation Matrix')
plt.show()


# select features with high correlation to the target
correlated_features = correlation_matrix['target'].sort_values(ascending=True)
print('Features Most Correlated With Target: ')
print(correlated_features)

X = df.drop(columns=['target'])
y = df['target']

mutual_info = mutual_info_regression(X, y)
mi_df = pd.DataFrame(
    {
        'Features': X.columns,
        'Mutual Information': mutual_info,
    }
)

mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)
print('Mutual Information Scores: ')
print(mi_df)

model = RandomForestRegressor(random_state=42)
model.fit(X, y)
feature_importance = model.feature_importances_
importance_df = pd.DataFrame(
    {
        'Feature': X.columns,
        'Importance': feature_importance,
    }
)

importance_df = importance_df.sort_values(by='Importance', ascending=False)
print('\nFeatures Importance From Random Forest: ')
print(importance_df)

plt.figure(figsize=(10,8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importance From Random Forest')
plt.show()