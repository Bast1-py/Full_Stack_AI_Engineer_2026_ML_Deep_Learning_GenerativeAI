## exercise 1: calculate CI for sample data
# import numpy as np
# from scipy.stats import norm
#
# # generate random sample data
# data = np.random.normal(
#     loc=100,
#     scale=10,
#     size=10000
# )
#
# mean = np.mean(data)
# std = np.std(data, ddof=1)
# n = len(data)
#
# z_value = norm.ppf(0.975)
# margin_of_error = z_value * (std / np.sqrt(n))
# ci = (mean - margin_of_error, mean + margin_of_error)
#
#
# print('Sample Mean: ', mean)
# print('95% Confidence Interval: ', ci)


# exercise 2: conduct sampling and create a report
import pandas as pd
import numpy as np
from scipy.stats import norm

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(url)

sample = df['sepal_length'].sample(56, random_state=42)
mean = sample.mean()
std = sample.std()
n = len(sample)

# Confidence Intervals
z_value = norm.ppf(0.975)
margin_of_error = z_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)

print('Sample Mean: ', mean)
print('Sample Standard Deviation: ', ci)