# hands-on exercise: 1. apply one-hot encoding and label encoding to adataset with categorical variables
# 2. experiment with different encoding techniques and observe their impact on model performance

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# print('Datasets Info: ')
# print(df.info())
#
# print('\nDatasets Preview: ')
# print(df.describe())

df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print('\nOne-hot Encoding datasets: ')
print(df_one_hot.head())

le = LabelEncoder()
df['Pclass_encoded'] = le.fit_transform(df['Pclass'])
print('\nLabel Encoder Datasets: ')
print(df[['Pclass', 'Pclass_encoded']].head())


df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# display frequency
print('\n Frequency Encoded Features: ')
print(df[['Ticket', 'Ticket_frequency']].head())


X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'Age'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=225)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy With One-hot Encoding: ', accuracy_score(y_test, y_pred))