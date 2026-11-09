# task 1: perform EDA and preprocessing
# task 2: train and evaluate multiple models
# task 3: summarize findings in a report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import labeled_comprehension
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# data = fetch_california_housing(as_frame=True)
# df = data.frame
#
# print(df.head())
# print(df.describe())
#
# sns.pairplot(
#     df, vars = [
#         'MedInc',
#         'AveRooms',
#         'HouseAge',
#         'MedHouseVal'
#     ]
# )
# plt.show()
#
# print('Missing Value: \n', df.isnull().sum())

df_telco = pd.read_csv('Telco-Customer-Churn.csv')
print(df_telco.head())
print(df_telco.describe())

sns.countplot(
    x='churn',
    data=df_telco
)

plt.title('Churn Distribution')
plt.show()

le = LabelEncoder()
df_telco['churn'] = le.fit_transform(df_telco['churn'])
df_telco['gender'] = le.fit_transform(df_telco['gender'])
df_telco['contract_type'] = le.fit_transform(df_telco['contract_type'])
df_telco['payment_method'] = le.fit_transform(df_telco['payment_method'])



X = df_telco.drop(columns=['churn'])
y = df_telco['churn']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train Logistic Regression model
lr = LogisticRegression(max_iter=220)
lr.fit(X_train, y_train)


# train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# evaluate
lr_pred = lr.predict(X_test)
knn_pred = knn.predict(X_test)

print('\nLogistic Regression Classification Report: ')
print(classification_report(y_test, lr_pred))

print('\nKNN Classification Report: ')
print(classification_report(y_test, knn_pred))

print('\Confusion Matrix (Logistic Regression): \n', confusion_matrix(y_test, lr_pred))
print('\Confusion Matrix (KNN): \n', confusion_matrix(y_test, knn_pred))



df_telco.fillna(df_telco.mean(), inplace=True)
