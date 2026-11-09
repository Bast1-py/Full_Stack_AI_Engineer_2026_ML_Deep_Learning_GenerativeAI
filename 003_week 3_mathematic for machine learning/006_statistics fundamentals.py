# exercise 1: calculate mean, variance and standard deviation
import numpy as np


data = [9.033299, 11.66701, 23.11033, 19.66568, 22.9912, 25.99018, 26.424119, 12.76201, 13.8791112,
        16.891211, 13.10199, 17.011239, 29.881023, 34.881723, 26.091822, 27.98192, 35.019232, 32.9192,
        37.010223, 32.99102, 35.220321, 38.231023, 32.091823, 33.71023, 24.01923, 26.0283834, 30.992833,
        31.234922, 34.8728390, 35.02373213, 34.920393, 30.0238320, 28.0827326, 30.938230, 29.038278, 30.53421]

mean = np.mean(data)
variance = np.var(data)
std_dev = np.std(data)

# print('Mean Value: ', mean)
# print('Variance: ', variance)
# print('Standard Deviation: ', std_dev)



# exercise 2: perform a T-test
from scipy.stats import ttest_ind
group1 = [2.0123, 1.0795, 3.3451, 4.1034, 2.89120, 1.93898, 2.34511, 3.56211, 2.97364]
group2 = [2.672922, 1.57239, 3.0201, 3.9264, 2.29120, 2.01923, 2.31122, 3.16567, 3.24852]

t_stat, p_value = ttest_ind(group1, group2)
print('T-Statistic: ', t_stat)
print('P-Value: ', p_value)

alpha = 0.05
if p_value < alpha:
        print('Reject The Null Hypothesis: Significant Difference')
else:
        print('Failed to Reject The Null Hypothesis: No Significant Difference')