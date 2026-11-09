# Hands-On Exercise 1: define features and target variables
# 2: split data into training and testing sets
# 3: visualize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
df = pd.read_csv(url)

# define features and target
features = df[['total_bill', 'size']]
target = df['tip']
# print('Features: \n', features.head())
# print('Target: \n', target.head())

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print('Training Datasets: ', X_train.shape)
print('Testing Datasets: ', X_test.shape)

sns.pairplot(df, x_vars=['total_bill', 'tip'], y_vars='tip', height=5, aspect=0.8, kind='scatter')
plt.title('Feature vs Target')
plt.show()

