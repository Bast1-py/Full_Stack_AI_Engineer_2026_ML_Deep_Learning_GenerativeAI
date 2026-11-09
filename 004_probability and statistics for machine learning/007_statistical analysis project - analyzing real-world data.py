import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
df = pd.read_csv(url)

# print(df.info())
# print(df.describe())

# del df['sex']
# del df['smoker']
# del df['day']
# del df['time']

# sns.histplot(df['total_bill'], kde=True)
# plt.title('Distribution of Total Bill')
# plt.show()

# sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
# plt.title('Correlation Heatmap')
# plt.show()


# male_tips = df[df['sex'] == 'Male']['tip']
# female_tips = df[df['sex'] == 'Female']['tip']
#
# # perform t_test
# t_stat, p_value = ttest_ind(male_tips, female_tips)
# print('T-Statistics', t_stat)
# print('P-Value: ', p_value)
#
# # interpret results
# alpha = 0.05
# if p_value < alpha:
#     print('Reject All Null Hypothesis: Significant Difference.')
# else:
#     print('Fail to Reject All Null Hypothesis: No Significant Difference.')


# # define variables
# X = df['total_bill'].values.reshape(-1,1)
# y = df['tip'].values
# model = LinearRegression()
# model.fit(X, y)
# print('Slope: ', model.coef_[0])
# print('Intercept: ', model.intercept_)
# print('R-Squared: ', model.score(X, y))
#
# sns.scatterplot(x=df['total_bill'], y=df['tip'], color='blue')
# plt.plot(df['total_bill'], model.predict(X), color='red', label='Regression Line')
# plt.title('Total Bill vs Tip')
# plt.legend()
# plt.show()
