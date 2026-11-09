import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
# print(df.info())
# print('\n', df.describe())


## exercise 1: perform data cleaning, aggregation, and filtering
# handle missing value
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# remove duplicates
df = df.drop_duplicates()

# filter data: passengers in first class
first_class = df[df['Pclass'] == 1]
# print('First Class Passengers: \n', first_class.head(6))



## exercise 2: generate visualization to illustrate key insights
# survival_by_class = df.groupby('Pclass')['Survived'].mean()
# survival_by_class.plot(kind='bar', color='skyblue')
# plt.ylabel('Survival Rate')
# plt.title('Survival by Class')
# plt.show()

# sns.histplot(df['Age'], kde=True, bins=20, color='purple')
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# plt.scatter(df['Age'], df['Fare'], alpha=0.5, color='green')
# plt.title('Age vs Fare')
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.show()