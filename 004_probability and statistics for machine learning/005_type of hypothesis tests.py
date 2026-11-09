# exercise 1: Conduct T-Test
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
from sklearn.feature_selection import f_oneway

# One-sample T-test
data = [12, 10, 11, 16, 15, 17, 18, 20, 19, 21, 22]
population_mean = 15
t_stat, p_value = ttest_1samp(data, population_mean)
# print('One-Sample T-Test: ', t_stat, p_value)

# two-sample T-test
group1 = [12, 10, 11, 16, 15, 17, 18, 20, 19, 21, 22]
group2 = [10, 13, 12, 14, 15, 16, 18, 20, 18, 20, 23]
t_stat, p_value = ttest_ind(group1, group2)
# print('Two-Sample T-Test: ', t_stat, p_value)

# Paired T-Test
pre_test = [12, 10, 11, 16, 15, 17, 18, 20, 19, 21, 22]
post_test = [13, 12, 10, 18, 15, 14, 15, 20, 23, 24, 27]
t_stat, p_value = ttest_rel(pre_test, post_test)
# print('Paired T-Test: ', t_stat, p_value)




# exercise 2: perform a Chi-Square Test
from scipy.stats import chi2_contingency

data = [[30, 35, 45, 60, 70, 80], [40, 50, 35, 45, 60, 65]]
chi2, p, dof, expected = chi2_contingency(data)
# print('Chi-Square Statistic:', chi2)
# print('P-Value: ', p)
# print('Expected Frequencies: \n', expected)



# exercise 3: conduct ANOVA
from scipy.stats import f_oneway

group1 = [12, 10, 11, 16, 15, 17, 18, 20, 19, 21, 22]
group2 = [10, 13, 12, 14, 15, 16, 18, 20, 18, 20, 23]
group3 = [11, 12, 13, 15, 17, 18, 20, 19, 17, 19, 21]

f_stat, p_value = f_oneway(group1, group2, group3)
print('F-Statistic:', f_stat)
print('P-Value: ', p_value)